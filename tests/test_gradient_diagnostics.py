"""Gradient diagnostics are checked independently of any forecasting data."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kan_phenotyping.training.gradient_diagnostics import GradientDiagnostics


class TwoComponentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_encoder = nn.Linear(2, 2, bias=True, dtype=torch.float64)
        self.dynamics = nn.Linear(2, 1, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.dynamics(self.static_encoder(x))


def model_and_input():
    torch.manual_seed(42)
    return TwoComponentModel(), torch.tensor([[1., 2.], [-2., 3.]], dtype=torch.float64)


def test_norms_match_independent_concatenated_gradients_and_components():
    model, inputs = model_and_input()
    checks = GradientDiagnostics(model)
    try:
        model(inputs).square().sum().backward()
        expected = torch.linalg.vector_norm(torch.cat([
            parameter.grad.flatten() for parameter in model.parameters()
        ])).item()
        expected_components = {
            name: torch.linalg.vector_norm(torch.cat([
                parameter.grad.flatten() for parameter in component.parameters()
            ])).item()
            for name, component in model.named_children()
        }
        record = checks.finish_backward()
        assert record["update"] == 1
        assert record["preclip_global_norm"] == pytest.approx(expected, rel=1e-14)
        assert record["component_norms"] == pytest.approx(expected_components, rel=1e-14)
        metadata = checks.metadata()
        assert metadata["completed_backward_updates"] == 1
        assert metadata["all_observed_gradients_finite"]
        assert metadata["all_parameters_received_gradients"]
        assert metadata["all_parameters_received_nonzero_gradients"]
        assert all(count == 1 for count in metadata["backward_calls_by_parameter"].values())
        assert metadata["pending_parameters"] == []
    finally:
        checks.close()


def test_finish_after_inplace_clipping_retains_preclip_values_without_mutation():
    model, inputs = model_and_input()
    checks = GradientDiagnostics(model, clip_norm=.01)
    try:
        (100 * model(inputs).square().sum()).backward()
        before = torch.cat([parameter.grad.flatten().clone() for parameter in model.parameters()])
        clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), .01, error_if_nonfinite=True)
        after = torch.cat([parameter.grad.flatten().clone() for parameter in model.parameters()])
        record = checks.finish_backward()
        assert record["preclip_global_norm"] == pytest.approx(torch.linalg.vector_norm(before).item())
        assert record["preclip_global_norm"] == pytest.approx(clipped_norm.item())
        assert torch.linalg.vector_norm(after).item() <= .01
        assert record["preclip_global_norm"] > 10
        assert checks.metadata()["preclip_norm_summary"]["above_clip_norm_count"] == 1
        assert all(torch.equal(parameter.grad.flatten(), value) for parameter, value in
                   zip(model.parameters(), [after[:4], after[4:6], after[6:]]))
    finally:
        checks.close()


def test_repeated_updates_and_optimizer_step_hook_use_fresh_captured_values():
    model, inputs = model_and_input()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    checks = GradientDiagnostics(model)

    def finish(*_):
        # Optimizer hooks must return None rather than the diagnostic record.
        checks.finish_backward()

    handle = optimizer.register_step_pre_hook(finish)
    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().sum().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        assert [record["update"] for record in checks.records] == [1, 2, 3]
        assert all(count == 3 for count in checks.calls.values())
        norms = [record["preclip_global_norm"] for record in checks.records]
        summary = checks.metadata()["preclip_norm_summary"]
        assert summary["min"] == min(norms)
        assert summary["max"] == max(norms)
        assert summary["mean"] == pytest.approx(sum(norms) / 3)
    finally:
        handle.remove()
        checks.close()


def test_shared_parameter_multiple_forward_uses_has_one_accumulated_hook():
    layer = nn.Linear(2, 1, dtype=torch.float64)
    checks = GradientDiagnostics(layer)
    try:
        inputs = torch.tensor([[2., 3.]], dtype=torch.float64)
        (layer(inputs) + layer(inputs * 2)).sum().backward()
        record = checks.finish_backward()
        assert record["preclip_global_norm"] == pytest.approx(math.sqrt(6**2 + 9**2 + 2**2))
        assert checks.calls == {"weight": 1, "bias": 1}
    finally:
        checks.close()


def test_zero_gradients_are_observed_without_being_claimed_nonzero():
    model = nn.Linear(2, 1, dtype=torch.float64)
    checks = GradientDiagnostics(model)
    try:
        (model(torch.zeros(1, 2, dtype=torch.float64)).sum() * 0).backward()
        assert checks.finish_backward()["preclip_global_norm"] == 0
        assert checks.metadata()["all_parameters_received_gradients"]
        assert not checks.metadata()["all_parameters_received_nonzero_gradients"]
    finally:
        checks.close()


def test_frozen_parameters_are_not_required_to_receive_gradients():
    model, inputs = model_and_input()
    model.static_encoder.bias.requires_grad_(False)
    checks = GradientDiagnostics(model)
    try:
        model(inputs).sum().backward()
        checks.finish_backward()
        assert "static_encoder.bias" not in checks.calls
    finally:
        checks.close()


def test_missing_parameter_gradient_rejects_incomplete_update_and_keeps_evidence():
    model, _ = model_and_input()
    checks = GradientDiagnostics(model)
    try:
        model.dynamics.weight.sum().backward()
        with pytest.raises(RuntimeError, match="did not produce all"):
            checks.finish_backward()
        assert checks.records == []
        assert checks.metadata()["pending_parameters"] == ["dynamics.weight"]
        assert not checks.metadata()["all_parameters_received_gradients"]
    finally:
        checks.close()


def test_duplicate_backward_without_finish_is_rejected():
    model, inputs = model_and_input()
    checks = GradientDiagnostics(model)
    try:
        model(inputs).sum().backward()
        with pytest.raises(RuntimeError, match="before finish_backward"):
            model(inputs).sum().backward()
        assert checks.records == []
    finally:
        checks.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradients_are_rejected_and_retained_as_failure_evidence(value):
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    checks = GradientDiagnostics(model)
    try:
        with pytest.raises(FloatingPointError, match="Nonfinite backward gradient"):
            (model.weight * value).sum().backward()
        metadata = checks.metadata()
        assert not metadata["all_observed_gradients_finite"]
        assert metadata["numerical_failures"] == [{
            "update": 1, "parameter": "weight", "kind": "nonfinite_gradient",
        }]
        assert checks.records == []
    finally:
        checks.close()


def test_hooks_leave_gradients_exactly_unchanged_and_close_removes_observation():
    reference, inputs = model_and_input()
    observed, _ = model_and_input()
    reference(inputs).square().sum().backward()
    checks = GradientDiagnostics(observed)
    observed(inputs).square().sum().backward()
    checks.finish_backward()
    for expected, actual in zip(reference.parameters(), observed.parameters()):
        assert torch.equal(expected.grad, actual.grad)
    checks.close()
    checks.close()
    observed.zero_grad(set_to_none=True)
    observed(inputs).sum().backward()
    assert all(count == 1 for count in checks.calls.values())
    with pytest.raises(RuntimeError, match="closed"):
        checks.finish_backward()


def test_empty_finish_cannot_be_counted_as_an_update():
    model, _ = model_and_input()
    checks = GradientDiagnostics(model)
    try:
        with pytest.raises(RuntimeError, match="did not produce all"):
            checks.finish_backward()
        assert checks.metadata()["preclip_norm_summary"] == {
            "n": 0, "min": None, "max": None, "mean": None, "above_clip_norm_count": 0,
        }
    finally:
        checks.close()


@pytest.mark.parametrize("clip_norm", [0., -1., float("nan"), float("inf"), True, "5"])
def test_invalid_clip_norm_is_rejected(clip_norm):
    with pytest.raises((TypeError, ValueError)):
        GradientDiagnostics(nn.Linear(1, 1), clip_norm=clip_norm)


def test_model_without_trainable_parameters_is_rejected():
    model = nn.Linear(1, 1)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        GradientDiagnostics(model)
