"""Differentiable classical RK4 for autonomous latent-state dynamics.

Times and the step size are ordinary finite numbers in relative study-month
coordinates. They are not observed timestamps or learnable parameters. Only
the state and dynamics parameters participate in PyTorch's direct autograd
graph. This intentionally small implementation has no adaptive step selection,
adjoint approximation, interpolation, or persistent hidden state.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from numbers import Real

import torch
from torch import Tensor


RK4_IMPLEMENTATION_VERSION = "1.0"


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be an ordinary real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


class RK4Integrator:
    """Integrate ``dh/dt = dynamics(h)`` with fixed-size classical RK4.

    A final shorter step ends exactly at the requested interval boundary.
    Backward integration is supported; zero duration returns the same tensor
    without evaluating dynamics. Counters describe computation, never samples
    or elapsed clinical time, and are not PyTorch parameters or buffers.
    """

    algorithm = "classical_fixed_step_rk4"
    implementation_version = RK4_IMPLEMENTATION_VERSION

    def __init__(self, step_size: float = .25) -> None:
        self.step_size = step_size
        self.reset_stats()

    @property
    def step_size(self) -> float:
        return self._step_size

    @step_size.setter
    def step_size(self, value: float) -> None:
        result = _finite_real(value, "step_size")
        if result <= 0:
            raise ValueError("step_size must be strictly positive.")
        self._step_size = result

    def reset_stats(self) -> None:
        self._integrate_calls = 0
        self._steps = 0
        self._rhs_evaluations = 0

    def stats(self) -> dict[str, int]:
        return {
            "integrate_calls": self._integrate_calls,
            "steps": self._steps,
            "rhs_evaluations": self._rhs_evaluations,
        }

    @staticmethod
    def _validate_state(state: Tensor) -> None:
        if not isinstance(state, Tensor):
            raise TypeError("state must be a PyTorch tensor.")
        if not state.is_floating_point():
            raise TypeError("state must have a floating-point dtype.")
        if state.numel() == 0:
            raise ValueError("state must not be empty.")
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("state contains non-finite values.")

    def _evaluate(self, dynamics: Callable[[Tensor], Tensor], state: Tensor) -> Tensor:
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("RK4 stage contains non-finite values.")
        self._rhs_evaluations += 1
        derivative = dynamics(state)
        if not isinstance(derivative, Tensor):
            raise TypeError("dynamics must return a PyTorch tensor.")
        if derivative.shape != state.shape:
            raise ValueError("dynamics output must have the same shape as state.")
        if derivative.dtype != state.dtype or derivative.device != state.device:
            raise ValueError("dynamics output must match the state dtype and device.")
        if not bool(torch.isfinite(derivative).all()):
            raise FloatingPointError("dynamics output contains non-finite values.")
        return derivative

    def integrate(
        self,
        dynamics: Callable[[Tensor], Tensor],
        state: Tensor,
        t0: float,
        t1: float,
    ) -> Tensor:
        if not callable(dynamics):
            raise TypeError("dynamics must be callable.")
        self._validate_state(state)
        start = _finite_real(t0, "t0")
        stop = _finite_real(t1, "t1")
        duration = stop - start
        if not math.isfinite(duration):
            raise ValueError("integration interval must be finite.")
        self._integrate_calls += 1
        if duration == 0:
            return state

        distance = abs(duration)
        ratio = distance / self.step_size
        if not math.isfinite(ratio):
            raise ValueError("interval/step_size must be finite.")
        n_steps = max(1, math.ceil(ratio))
        direction = 1. if duration > 0 else -1.
        hidden = state
        for index in range(n_steps):
            remaining = distance - index * self.step_size
            step = direction * min(self.step_size, remaining)
            # Every update is out-of-place tensor arithmetic. In particular,
            # neither the state nor any RK4 stage is detached from autograd.
            k1 = self._evaluate(dynamics, hidden)
            k2 = self._evaluate(dynamics, hidden + (step / 2) * k1)
            k3 = self._evaluate(dynamics, hidden + (step / 2) * k2)
            k4 = self._evaluate(dynamics, hidden + step * k3)
            updated = hidden + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not bool(torch.isfinite(updated).all()):
                raise FloatingPointError("RK4 update contains non-finite values.")
            hidden = updated
            self._steps += 1
        return hidden

    __call__ = integrate
