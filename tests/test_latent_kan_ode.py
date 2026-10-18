"""Architecture and information-flow checks using synthetic observations only."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kan_phenotyping.models.kan_layer import KANLinear
from kan_phenotyping.models.latent_kan_ode import LatentKANODE
from kan_phenotyping.models.sequential_kan import KANEncodedRecurrentBaseline


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def inputs(n=5, *, requires_grad=False):
    wearable = torch.randn(n, 2, 48, requires_grad=requires_grad)
    static = torch.randn(n, 63, requires_grad=requires_grad)
    return wearable, static


def test_fixed_architecture_has_shared_components_and_one_autonomous_kan():
    model = LatentKANODE()
    layers = [module for module in model.modules() if isinstance(module, KANLinear)]
    assert layers == [model.static_encoder, model.wearable_encoder, model.head, model.dynamics]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (63, 16), (48, 16), (16, 1), (16, 16)
    ]
    assert all(layer.grid_intervals == 8 and layer.spline_degree == 3 for layer in layers)
    assert all(layer.grid_range == (-3., 3.) for layer in layers)
    assert len([module for module in model.modules() if isinstance(module, nn.GRUCell)]) == 1
    assert model.gru.input_size == model.gru.hidden_size == 16
    assert sum(parameter.numel() for parameter in model.parameters()) == 26257
    assert model.step_size == .25
    assert not any(isinstance(module, (nn.Sigmoid, nn.Softmax)) for module in model.modules())


def test_initialization_preserves_recurrent_components_and_scales_only_dynamics():
    torch.manual_seed(42)
    recurrent = KANEncodedRecurrentBaseline()
    unscaled_dynamics = KANLinear(16, 16)
    torch.manual_seed(42)
    model = LatentKANODE()
    for name, value in recurrent.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(model.dynamics.base_weight,
                               unscaled_dynamics.base_weight * .01, rtol=0, atol=0)
    torch.testing.assert_close(model.dynamics.spline_weight,
                               unscaled_dynamics.spline_weight * .01, rtol=0, atol=0)
    assert torch.count_nonzero(model.dynamics.bias).item() == 0


def test_step_size_setter_delegates_validation_without_replacing_integrator():
    model = LatentKANODE()
    integrator = model.integrator
    original = {key: value.clone() for key, value in model.state_dict().items()}
    model.step_size = .125
    assert model.step_size == integrator.step_size == .125
    assert model.integrator is integrator
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    model.step_size = .25
    assert model.step_size == .25


@pytest.mark.parametrize("invalid", [0., -.125, float("nan"), float("inf"), True, ".125"])
def test_step_size_setter_rejects_invalid_values_without_mutating_step(invalid):
    model = LatentKANODE()
    with pytest.raises((TypeError, ValueError)):
        model.step_size = invalid
    assert model.step_size == .25


def test_flow_and_observation_order_are_exactly_the_requested_equations():
    model = LatentKANODE()
    wearable, static = inputs()
    packet_calls, context_calls, gru_calls = [], [], []
    handles = [
        model.wearable_encoder.register_forward_pre_hook(
            lambda module, args: packet_calls.append((id(module), args[0].clone()))),
        model.static_encoder.register_forward_pre_hook(
            lambda module, args: context_calls.append(args[0].clone())),
        model.gru.register_forward_pre_hook(
            lambda module, args: gru_calls.append((id(module), args[1].clone()))),
    ]
    try:
        states = model.trace_states(wearable, static)
    finally:
        for handle in handles:
            handle.remove()
    assert states.shape == (5, 6, 16)
    assert model.TRACE_STATE_NAMES == (
        "h0", "h1_before", "h1_after", "h2_before", "h2_after", "h3"
    )
    assert len(context_calls) == 1
    assert len(packet_calls) == len(gru_calls) == 2
    for position, before_index in ((0, 1), (1, 3)):
        assert packet_calls[position][0] == id(model.wearable_encoder)
        assert gru_calls[position][0] == id(model.gru)
        torch.testing.assert_close(packet_calls[position][1], wearable[:, position], rtol=0, atol=0)
        torch.testing.assert_close(gru_calls[position][1], states[:, before_index], rtol=0, atol=0)
    torch.testing.assert_close(context_calls[0], static, rtol=0, atol=0)
    h0 = model.static_encoder(static)
    h1_before = model.integrator.integrate(model.dynamics, h0, 0., 1.)
    h1_after = model.gru(model.wearable_encoder(wearable[:, 0]), h1_before)
    h2_before = model.integrator.integrate(model.dynamics, h1_after, 1., 2.)
    h2_after = model.gru(model.wearable_encoder(wearable[:, 1]), h2_before)
    h3 = model.integrator.integrate(model.dynamics, h2_after, 2., 3.)
    expected = torch.stack((h0, h1_before, h1_after, h2_before, h2_after, h3), dim=1)
    torch.testing.assert_close(states, expected, rtol=0, atol=0)
    # The stacked trace view and contiguous independently reconstructed h3 may
    # dispatch different GEMM kernels; their floating-point sums can differ.
    torch.testing.assert_close(model(wearable, static), model.head(h3).squeeze(-1),
                               rtol=1e-6, atol=1e-7)


def test_zero_dynamics_recovers_copied_non_ode_recurrent_control():
    recurrent = KANEncodedRecurrentBaseline()
    model = LatentKANODE()
    missing, unexpected = model.load_state_dict(recurrent.state_dict(), strict=False)
    assert set(missing) == {
        "dynamics.base_weight", "dynamics.spline_weight", "dynamics.bias", "dynamics.knots"
    }
    assert unexpected == []
    with torch.no_grad():
        for parameter in model.dynamics.parameters():
            parameter.zero_()
    wearable, static = inputs()
    ode_states, recurrent_states = model.trace_states(wearable, static), recurrent.trace_states(wearable, static)
    for ode_index, recurrent_index in ((0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)):
        torch.testing.assert_close(ode_states[:, ode_index], recurrent_states[:, recurrent_index], rtol=0, atol=0)
    torch.testing.assert_close(model(wearable, static), recurrent(wearable, static), rtol=0, atol=0)


def test_sm1_update_has_no_computational_access_to_sm2():
    model = LatentKANODE()
    wearable, static = inputs(4, requires_grad=True)
    states = model.trace_states(wearable, static)
    changed = wearable.detach().clone()
    changed[:, 1] = torch.linspace(-50., 50., 48)
    changed_states = model.trace_states(changed, static.detach())
    torch.testing.assert_close(states[:, :3], changed_states[:, :3], rtol=0, atol=0)
    packet_gradient, static_gradient = torch.autograd.grad(
        states[:, 2].square().sum(), (wearable, static)
    )
    assert torch.count_nonzero(packet_gradient[:, 1]).item() == 0
    assert torch.count_nonzero(packet_gradient[:, 0]).item() > 0
    assert torch.count_nonzero(static_gradient).item() > 0
    assert torch.isfinite(packet_gradient).all() and torch.isfinite(static_gradient).all()


def test_no_hidden_state_carries_between_calls_samples_or_batches():
    model = LatentKANODE()
    wearable, static = inputs(6)
    first = model(wearable, static)
    model(*inputs(3))
    repeated = model(wearable, static)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    reverse = torch.arange(5, -1, -1)
    torch.testing.assert_close(model(wearable[reverse], static[reverse]), first[reverse], rtol=1e-6, atol=1e-6)
    per_sample = torch.cat([model(wearable[i:i+1], static[i:i+1]) for i in range(6)])
    torch.testing.assert_close(per_sample, first, rtol=1e-6, atol=1e-6)
    partitioned = torch.cat([model(wearable[:2], static[:2]), model(wearable[2:], static[2:])])
    torch.testing.assert_close(partitioned, first, rtol=1e-6, atol=1e-6)


def test_every_component_and_parameter_receives_finite_nonzero_gradients():
    model = LatentKANODE()
    wearable, static = inputs(8)
    targets = torch.arange(8, dtype=torch.float32) + 4.
    predictions = model(wearable, static)
    assert predictions.shape == targets.shape == (8,)
    loss = nn.MSELoss()(predictions, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_checkpoint_restoration_reproduces_predictions_exactly(tmp_path):
    model = LatentKANODE()
    wearable, static = inputs()
    prediction = model(wearable, static).detach()
    checkpoint = tmp_path / "latent_kan_ode.pt"
    torch.save({"state_dict": model.state_dict(), "step_size": model.step_size}, checkpoint)
    torch.manual_seed(123)
    saved = torch.load(checkpoint, weights_only=True)
    restored = LatentKANODE(step_size=saved["step_size"])
    restored.load_state_dict(saved["state_dict"])
    torch.testing.assert_close(restored(wearable, static), prediction, rtol=0, atol=0)


def test_modular_autonomous_dynamics_can_be_supplied_without_reinitialization():
    class ZeroDynamics(nn.Module):
        def forward(self, hidden):
            return torch.zeros_like(hidden)

    dynamics = ZeroDynamics()
    model = LatentKANODE(dynamics=dynamics)
    assert model.dynamics is dynamics
    states = model.trace_states(*inputs())
    for before, after in ((0, 1), (2, 3), (4, 5)):
        torch.testing.assert_close(states[:, before], states[:, after], rtol=0, atol=0)


@pytest.mark.parametrize("value", [-8., 34., 4.125])
def test_head_is_unbounded_and_never_rounds_or_clips(value):
    model = LatentKANODE()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(value)
    output = model(torch.zeros(3, 2, 48), torch.zeros(3, 63))
    torch.testing.assert_close(output, torch.full((3,), value), rtol=0, atol=0)


@pytest.mark.parametrize("wearable,static", [
    (torch.zeros(3, 1, 48), torch.zeros(3, 63)),
    (torch.zeros(3, 3, 48), torch.zeros(3, 63)),
    (torch.zeros(3, 2, 47), torch.zeros(3, 63)),
    (torch.zeros(3, 2, 48), torch.zeros(3, 62)),
    (torch.zeros(3, 2, 48), torch.zeros(4, 63)),
    (torch.full((3, 2, 48), float("nan")), torch.zeros(3, 63)),
    (torch.zeros(3, 2, 48), torch.full((3, 63), float("inf"))),
])
def test_rejects_extra_packets_shape_mismatches_and_nonfinite_inputs(wearable, static):
    with pytest.raises(ValueError):
        LatentKANODE()(wearable, static)
