"""Numerical and masking invariants for the fixed-grid KAN implementation."""

import pytest
import torch

from kan_phenotyping.models import (
    KANConv1d,
    KANLinear,
    MLPBaseline,
    TabularDepressionKAN,
    TemporalCNNBaseline,
    TemporalDepressionKAN,
)


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _reference_basis(value, index, degree, knots):
    if degree == 0:
        return float(knots[index] <= value < knots[index + 1])
    left = (value - knots[index]) / (knots[index + degree] - knots[index])
    right = (knots[index + degree + 1] - value) / (knots[index + degree + 1] - knots[index + 1])
    return left * _reference_basis(value, index, degree - 1, knots) + right * _reference_basis(value, index + 1, degree - 1, knots)


def test_cubic_bases_match_scalar_cox_de_boor_reference_and_partition_unity():
    layer = KANLinear(1, 2).double()
    x = torch.linspace(-3, 3, 121, dtype=torch.float64).unsqueeze(-1)
    bases = layer.b_spline_basis(x)
    assert layer.n_basis == 11
    assert bases.shape == (121, 1, 11)
    assert torch.all(bases >= 0)
    torch.testing.assert_close(bases.sum(dim=-1), torch.ones_like(x), rtol=1e-12, atol=1e-12)
    reference = torch.tensor(
        [[[_reference_basis(v, i, 3, layer.knots.tolist()) for i in range(layer.n_basis)]] for v in x.flatten().tolist()],
        dtype=torch.float64,
    )
    torch.testing.assert_close(bases, reference, rtol=1e-12, atol=1e-12)
    assert (bases > 0).sum(dim=-1).max() <= 4


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_basis_count_and_partition_for_configurable_degree(degree):
    layer = KANLinear(2, 1, grid_intervals=4, spline_degree=degree)
    # For degree zero the rightmost knot is outside the half-open basis support.
    x = torch.tensor([[-2.9, 2.9], [-1.25, 1.25]])
    basis = layer.b_spline_basis(x)
    assert basis.shape == (2, 2, 4 + degree)
    torch.testing.assert_close(basis.sum(dim=-1), torch.ones_like(x))


def test_gradcheck_and_finite_parameter_gradients():
    torch.manual_seed(42)
    layer = KANLinear(2, 3).double()
    x = torch.tensor([[0.13, -0.27], [0.41, 1.12]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-5)
    layer(x).square().sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.isfinite(x.grad).all()


def test_feature_mask_excludes_nonzero_spline_at_zero_and_nan_padding():
    layer = KANLinear(2, 1, bias=False)
    with torch.no_grad():
        layer.base_weight.zero_()
        layer.spline_weight.fill_(1.0)
    x = torch.tensor([[0.0, float("nan")]])
    output = layer(x, torch.tensor([[True, False]]))
    torch.testing.assert_close(output, torch.ones(1, 1))


def test_full_edge_response_sums_to_layer_output_with_one_bias():
    layer = KANLinear(3, 2).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    expected = torch.stack([
        sum(layer.edge_response(i, j, x[:, i]) for i in range(3)) + layer.bias[j]
        for j in range(2)
    ], dim=-1)
    torch.testing.assert_close(layer(x), expected)


def test_explicit_mlp_mixing_ablation_and_single_scale_temporal_model():
    tabular = TabularDepressionKAN(3, mixing_type="mlp")
    assert isinstance(tabular.layers[0], torch.nn.Sequential)
    assert isinstance(tabular.layers[-1], KANLinear)
    assert tabular(torch.zeros(2, 3)).shape == (2,)
    temporal = TemporalDepressionKAN(3, kernel_sizes=(5,), branch_channels=(4,), mixing_width=5, mixing_type="mlp")
    assert temporal(torch.zeros(2, 6, 3)).shape == (2,)


def test_causal_convolution_cannot_observe_future():
    torch.manual_seed(7)
    convolution = KANConv1d(2, 3, 5)
    x = torch.randn(2, 2, 9)
    changed = x.clone()
    changed[:, :, 5:] += 10
    torch.testing.assert_close(convolution(x)[:, :, :5], convolution(changed)[:, :, :5])


def test_convolution_excludes_left_padding_spline_edges():
    convolution = KANConv1d(1, 1, 3)
    with torch.no_grad():
        convolution.edge_functions.base_weight.zero_()
        convolution.edge_functions.spline_weight.fill_(1.0)
        convolution.edge_functions.bias.zero_()
    torch.testing.assert_close(convolution(torch.zeros(1, 1, 4)), torch.tensor([[[1.0, 2.0, 3.0, 3.0]]]))


@pytest.mark.parametrize("model_class", [TemporalDepressionKAN, TemporalCNNBaseline])
def test_temporal_models_are_invariant_to_right_padding_and_masked_values(model_class):
    torch.manual_seed(6)
    model = model_class(3, 2, branch_channels=(3, 2, 2), mixing_width=5).double()
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    static = torch.randn(2, 2, dtype=torch.float64)
    expected = model(x, static)
    padding = torch.full((2, 3, 3), float("nan"), dtype=torch.float64)
    padded = torch.cat((x, padding), dim=1)
    mask = torch.tensor([[True] * 4 + [False] * 3] * 2)
    actual = model(padded, static, mask)
    assert actual.shape == (2,)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("model_class", [TemporalDepressionKAN, TemporalCNNBaseline])
def test_temporal_model_rejects_empty_sequences_and_missing_static(model_class):
    model = model_class(2, 1, branch_channels=(1, 1, 1), mixing_width=2)
    with pytest.raises(ValueError, match="valid timestep"):
        model(torch.zeros(1, 3, 2), torch.zeros(1, 1), torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="static"):
        model(torch.zeros(1, 3, 2))


@pytest.mark.parametrize("model_class", [TabularDepressionKAN, MLPBaseline])
def test_tabular_single_example_shape_and_no_implicit_sigmoid(model_class):
    model = model_class(3, hidden_width=4, task="binary")
    final = model.layers[-1]
    with torch.no_grad():
        if isinstance(final, KANLinear):
            final.base_weight.zero_()
            final.spline_weight.zero_()
        else:
            final.weight.zero_()
        final.bias.fill_(-2.0)
    output = model(torch.zeros(1, 3))
    assert output.shape == (1,)
    torch.testing.assert_close(output, torch.tensor([-2.0]))


def test_default_temporal_architecture_has_64_total_branch_channels():
    model = TemporalDepressionKAN(2)
    assert [branch.kernel_size for branch in model.branches] == [3, 5, 7]
    assert [branch.out_channels for branch in model.branches] == [22, 21, 21]
    assert model.mixing.in_features == model.mixing.out_features == 64
    assert model.head.out_features == 1
