"""Capture backward gradient norms before an optimizer can clip them.

The hooks only observe gradients: they never replace or mutate them.  A caller
must finish each backward pass with :meth:`GradientDiagnostics.finish_backward`
before starting another backward pass.  Finishing may occur after clipping,
because each hook has already copied its scalar norm into ordinary Python data.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class GradientDiagnostics:
    """Observe every trainable parameter once per backward update.

    Components are the top-level names from ``model.named_parameters()``.  Norms
    use detached float64 values outside the numerical solver and autograd graph;
    this does not affect the gradient that the optimizer subsequently receives.
    """

    def __init__(self, model: nn.Module, *, clip_norm: float = 5.0):
        if isinstance(clip_norm, bool) or not isinstance(clip_norm, (int, float)):
            raise TypeError("clip_norm must be a finite positive number")
        if not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("clip_norm must be a finite positive number")
        parameters = [(name, parameter) for name, parameter in model.named_parameters()
                      if parameter.requires_grad]
        if not parameters:
            raise ValueError("Gradient diagnostics require trainable parameters")
        self.clip_norm = float(clip_norm)
        self.calls = {name: 0 for name, _ in parameters}
        self.nonzero = {name: False for name, _ in parameters}
        self.records: list[dict[str, Any]] = []
        self.numerical_failures: list[dict[str, Any]] = []
        self._pending: dict[str, float] = {}
        self._closed = False
        self._handles = [parameter.register_hook(self._hook(name))
                         for name, parameter in parameters]

    def _hook(self, name: str):
        def observe(gradient: torch.Tensor) -> torch.Tensor:
            if name in self._pending:
                raise RuntimeError(
                    f"Parameter {name} received another gradient before finish_backward(); "
                    "finish every backward pass before starting another"
                )
            values = gradient.detach()
            if values.is_sparse:
                values = values.coalesce().values()
            if not bool(torch.isfinite(values).all()):
                self.numerical_failures.append({
                    "update": len(self.records) + 1, "parameter": name,
                    "kind": "nonfinite_gradient",
                })
                raise FloatingPointError(f"Nonfinite backward gradient: {name}")
            norm = float(torch.linalg.vector_norm(values.to(dtype=torch.float64)))
            if not math.isfinite(norm):
                self.numerical_failures.append({
                    "update": len(self.records) + 1, "parameter": name,
                    "kind": "nonfinite_gradient_norm",
                })
                raise FloatingPointError(f"Nonfinite backward gradient norm: {name}")
            self._pending[name] = norm
            self.calls[name] += 1
            self.nonzero[name] = self.nonzero[name] or norm > 0.0
            return gradient
        return observe

    def finish_backward(self) -> dict[str, Any]:
        """Record one completed backward, rejecting missing parameter gradients.

        Captured norms survive later in-place clipping.  Incomplete updates stay
        pending so that ``metadata()`` can record failure evidence; they cannot
        be silently accepted or reset.
        """
        if self._closed:
            raise RuntimeError("Gradient diagnostics are closed")
        missing = sorted(self.calls.keys() - self._pending.keys())
        if missing:
            raise RuntimeError(f"Backward did not produce all trainable gradients: {missing}")
        components: dict[str, float] = {}
        for name, norm in self._pending.items():
            component = name.split(".", 1)[0]
            components[component] = math.hypot(components.get(component, 0.0), norm)
        total_norm = math.hypot(*self._pending.values())
        if not math.isfinite(total_norm):
            self.numerical_failures.append({
                "update": len(self.records) + 1, "kind": "nonfinite_global_norm",
            })
            raise FloatingPointError("Nonfinite global backward gradient norm")
        record = {
            "update": len(self.records) + 1,
            "preclip_global_norm": total_norm,
            "component_norms": dict(sorted(components.items())),
        }
        self.records.append(record)
        self._pending.clear()
        return record

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics, including incomplete-run evidence."""
        norms = [record["preclip_global_norm"] for record in self.records]
        return {
            "completed_backward_updates": len(self.records),
            "clip_norm": self.clip_norm,
            "all_observed_gradients_finite": not self.numerical_failures,
            "backward_calls_by_parameter": self.calls.copy(),
            "nonzero_gradient_seen_by_parameter": self.nonzero.copy(),
            "all_parameters_received_gradients": all(self.calls.values()),
            "all_parameters_received_nonzero_gradients": all(self.nonzero.values()),
            "pending_parameters": sorted(self._pending),
            "preclip_norm_summary": {
                "n": len(norms),
                "min": min(norms) if norms else None,
                "max": max(norms) if norms else None,
                "mean": math.fsum(value / len(norms) for value in norms) if norms else None,
                "above_clip_norm_count": sum(value > self.clip_norm for value in norms),
            },
            "numerical_failures": [failure.copy() for failure in self.numerical_failures],
        }

    def close(self) -> None:
        """Remove hooks without clearing observations; safe to call repeatedly."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True
