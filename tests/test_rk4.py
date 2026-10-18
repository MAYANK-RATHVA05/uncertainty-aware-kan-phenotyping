"""Independent analytic and differentiation checks for the local RK4 solver."""

from __future__ import annotations

import math

import pytest
import torch

from kan_phenotyping.numerics.rk4 import RK4Integrator


def test_zero_interval_is_identity_without_any_dynamics_evaluation():
    initial = torch.tensor([[1., 2.]], requires_grad=True)
    solver = RK4Integrator()

    def forbidden(state):
        raise AssertionError("A zero interval must not call dynamics.")

    result = solver.integrate(forbidden, initial, 2., 2.)
    assert result is initial
    result.sum().backward()
    torch.testing.assert_close(initial.grad, torch.ones_like(initial))
    assert solver.stats() == {"integrate_calls": 1, "steps": 0, "rhs_evaluations": 0}


def test_zero_dynamics_preserves_state_and_direct_input_gradient():
    initial = torch.tensor([[1., -2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
    result = RK4Integrator(.25).integrate(torch.zeros_like, initial, 0., 3.)
    torch.testing.assert_close(result, initial, rtol=0, atol=0)
    gradient, = torch.autograd.grad(result.sum(), initial)
    torch.testing.assert_close(gradient, torch.ones_like(initial), rtol=0, atol=0)


@pytest.mark.parametrize("start,stop", [(0., 1.1), (3., 1.9), (-.3, .8)])
def test_constant_dynamics_uses_a_partial_final_step_and_exact_duration(start, stop):
    initial = torch.tensor([[2., -1.]], dtype=torch.float64)
    velocity = torch.tensor([[.4, -.7]], dtype=torch.float64)
    solver = RK4Integrator(.25)
    result = solver.integrate(lambda state: velocity.expand_as(state), initial, start, stop)
    torch.testing.assert_close(result, initial + (stop - start) * velocity, rtol=0, atol=1e-14)
    assert solver.stats() == {"integrate_calls": 1, "steps": 5, "rhs_evaluations": 20}


@pytest.mark.parametrize("start,stop", [(0., 1.), (2., 1.)])
def test_linear_system_matches_independent_matrix_exponential(start, stop):
    matrix = torch.tensor([[-.3, .2], [-.1, -.4]], dtype=torch.float64)
    initial = torch.tensor([[1., -2.], [3., .25]], dtype=torch.float64)
    expected = initial @ torch.matrix_exp(matrix * (stop - start)).T
    result = RK4Integrator(.125).integrate(lambda state: state @ matrix.T, initial, start, stop)
    torch.testing.assert_close(result, expected, rtol=2e-7, atol=2e-8)


def test_fourth_order_convergence_against_scalar_exponential():
    initial = torch.tensor([1.], dtype=torch.float64)
    exact = math.exp(1.)
    errors = []
    for step in (.25, .125, .0625):
        result = RK4Integrator(step).integrate(lambda state: state, initial, 0., 1.)
        errors.append(abs(result.item() - exact))
    assert errors[0] > errors[1] > errors[2]
    assert 14 < errors[0] / errors[1] < 17
    assert 14 < errors[1] / errors[2] < 17


def test_parameter_and_input_gradients_match_analytic_and_finite_difference():
    initial = torch.tensor([1.4], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(-.3, dtype=torch.float64, requires_grad=True)
    solver = RK4Integrator(.125)
    final = solver.integrate(lambda state: coefficient * state, initial, 0., 1.1)
    initial_gradient, parameter_gradient = torch.autograd.grad(final.sum(), (initial, coefficient))
    exponential = math.exp(-.3 * 1.1)
    assert initial_gradient.item() == pytest.approx(exponential, rel=1e-8)
    assert parameter_gradient.item() == pytest.approx(1.1 * 1.4 * exponential, rel=1e-7)

    def evaluate(value):
        return solver.integrate(lambda state: value * state,
                                torch.tensor([1.4], dtype=torch.float64), 0., 1.1).item()

    epsilon = 1e-6
    finite_difference = (evaluate(-.3 + epsilon) - evaluate(-.3 - epsilon)) / (2 * epsilon)
    assert parameter_gradient.item() == pytest.approx(finite_difference, rel=1e-8, abs=1e-9)


def test_three_unit_intervals_have_the_expected_compute_count_and_reset():
    solver = RK4Integrator(.25)
    state = torch.ones(3, 16)
    for start in range(3):
        state = solver.integrate(lambda hidden: -.1 * hidden, state, start, start + 1)
    assert solver.stats() == {"integrate_calls": 3, "steps": 12, "rhs_evaluations": 48}
    solver.reset_stats()
    assert solver.stats() == {"integrate_calls": 0, "steps": 0, "rhs_evaluations": 0}
    assert solver.implementation_version == "1.0"
    assert solver.algorithm == "classical_fixed_step_rk4"


@pytest.mark.parametrize("step", [0., -.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_step_sizes_are_rejected(step):
    with pytest.raises(ValueError, match="step_size"):
        RK4Integrator(step)


@pytest.mark.parametrize("step", [True, "0.25", torch.tensor(.25)])
def test_non_numeric_or_tensor_step_sizes_are_not_silently_coerced(step):
    with pytest.raises(TypeError, match="step_size"):
        RK4Integrator(step)


def test_step_size_assignment_preserves_validation():
    solver = RK4Integrator()
    with pytest.raises(ValueError, match="strictly positive"):
        solver.step_size = 0.
    assert solver.step_size == .25


@pytest.mark.parametrize("start,stop", [(float("nan"), 1.), (0., float("inf")),
                                        (-1e308, 1e308)])
def test_nonfinite_times_or_interval_are_rejected(start, stop):
    with pytest.raises(ValueError, match="finite"):
        RK4Integrator().integrate(torch.zeros_like, torch.ones(1), start, stop)


def test_time_tensors_are_rejected_rather_than_silently_detached():
    with pytest.raises(TypeError, match="ordinary real"):
        RK4Integrator().integrate(torch.zeros_like, torch.ones(1), torch.tensor(0.), 1.)


@pytest.mark.parametrize("state,error", [
    ([], TypeError), (torch.ones(1, dtype=torch.long), TypeError),
    (torch.empty(0), ValueError), (torch.tensor([float("nan")]), FloatingPointError),
])
def test_invalid_initial_states_are_rejected(state, error):
    with pytest.raises(error):
        RK4Integrator().integrate(torch.zeros_like, state, 0., 1.)


@pytest.mark.parametrize("dynamics,error,match", [
    (lambda state: [0.], TypeError, "PyTorch tensor"),
    (lambda state: torch.zeros(2), ValueError, "same shape"),
    (lambda state: torch.zeros_like(state, dtype=torch.float64), ValueError, "dtype and device"),
    (lambda state: torch.full_like(state, float("nan")), FloatingPointError, "non-finite"),
    (lambda state: torch.full_like(state, float("inf")), FloatingPointError, "non-finite"),
])
def test_bad_dynamics_outputs_fail_explicitly(dynamics, error, match):
    with pytest.raises(error, match=match):
        RK4Integrator().integrate(dynamics, torch.ones(1), 0., 1.)


def test_state_overflow_fails_instead_of_clipping_or_changing_the_step():
    initial = torch.tensor([1e38], dtype=torch.float32)
    with pytest.raises(FloatingPointError, match="non-finite"):
        RK4Integrator(1.).integrate(lambda state: state * 3, initial, 0., 1.)


def test_solver_does_not_change_input_or_keep_hidden_state_between_calls():
    solver = RK4Integrator(.25)
    initial = torch.tensor([[1., 2.]], dtype=torch.float64)
    original = initial.clone()
    first = solver.integrate(lambda state: -.2 * state, initial, 0., 1.)
    solver.integrate(lambda state: state, torch.full_like(initial, 9.), 0., 2.)
    second = solver.integrate(lambda state: -.2 * state, initial, 0., 1.)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
