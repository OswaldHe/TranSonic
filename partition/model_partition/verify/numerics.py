# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tolerances and comparison metrics for module verification.

Elementwise closeness alone is a poor gate in bf16 — a handful of outliers in a
large tensor is normal. Each comparison therefore reports a pass *fraction* and
cosine similarity alongside max errors, and the verdict requires all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: dtype -> (rtol, atol). Keyed by the dtype of the tensors being compared.
TOLERANCES: dict[str, tuple[float, float]] = {
    "float32": (1e-5, 1e-6),
    "float64": (1e-7, 1e-9),
    "float16": (1e-2, 1e-3),
    "bfloat16": (2e-2, 2e-2),
    "float8_e4m3fn": (1e-1, 1e-1),
    "float8_e5m2": (2e-1, 2e-1),
}

DEFAULT_TOLERANCE = (2e-2, 2e-2)


@dataclass
class Tolerance:
    """The bar a comparison must clear."""

    rtol: float
    atol: float
    min_pass_fraction: float = 0.999
    min_cosine: float = 0.9999

    @classmethod
    def for_dtype(cls, dtype: str, **overrides: Any) -> Tolerance:
        rtol, atol = TOLERANCES.get(dtype, DEFAULT_TOLERANCE)
        return cls(rtol=rtol, atol=atol, **overrides)

    @classmethod
    def accumulated(cls, dtype: str = "bfloat16") -> Tolerance:
        """Bar for a tensor many layers deep in a whole-model forward.

        Replaying one module against its own recorded input is a single step and
        should be near-exact. A module *boundary* observed during end-to-end
        emulation is the product of every layer before it, so bf16 rounding
        accumulates — and trace and emulation need not have run with identical
        device placement, which shifts kernel selection and accumulation order.

        Cosine is the discriminator here, not the elementwise pass fraction. Deep
        in a 64-layer stack the residual stream is large, so a *relative* per-element
        tolerance is uninformative: a measured-good boundary at layer 63 had only
        64% of elements within 2e-2 yet cosine 0.9997, while a genuine wiring
        error — comparing a layer group against the wrong submodule's output —
        scored cosine 0.707. The pass fraction is kept as a coarse floor and is
        still reported for diagnosis.
        """
        rtol, atol = TOLERANCES.get(dtype, DEFAULT_TOLERANCE)
        return cls(rtol=rtol * 2, atol=atol * 2,
                   min_pass_fraction=0.50, min_cosine=0.999)


@dataclass
class Comparison:
    """Result of comparing one tensor against its reference."""

    name: str
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    pass_fraction: float = 1.0
    cosine: float = 1.0
    n_elements: int = 0
    reason: str = ""
    worst_index: list[int] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return (f"{self.name}: ok (max_abs={self.max_abs_err:.3e}, "
                    f"cos={self.cosine:.6f}, pass={self.pass_fraction:.5f})")
        return (f"{self.name}: FAIL {self.reason} (max_abs={self.max_abs_err:.3e}, "
                f"max_rel={self.max_rel_err:.3e}, cos={self.cosine:.6f}, "
                f"pass={self.pass_fraction:.5f}, worst_index={self.worst_index})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "passed": self.passed,
            "max_abs_err": self.max_abs_err, "max_rel_err": self.max_rel_err,
            "pass_fraction": self.pass_fraction, "cosine": self.cosine,
            "n_elements": self.n_elements, "reason": self.reason,
            "worst_index": self.worst_index,
        }


def compare(actual: Any, reference: Any, name: str = "tensor",
            tolerance: Tolerance | None = None) -> Comparison:
    """Compare two tensors and decide whether they match."""
    import torch

    if actual is None or reference is None:
        return Comparison(name=name, passed=False,
                          reason="missing tensor" if actual is None else "missing reference")
    if tuple(actual.shape) != tuple(reference.shape):
        return Comparison(name=name, passed=False,
                          reason=f"shape {tuple(actual.shape)} != {tuple(reference.shape)}")

    dtype_name = str(reference.dtype).removeprefix("torch.")
    tol = tolerance or Tolerance.for_dtype(dtype_name)

    a = actual.detach().to(torch.float32).flatten()
    b = reference.detach().to(torch.float32).flatten()
    n = a.numel()
    if n == 0:
        return Comparison(name=name, passed=True, n_elements=0)

    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        bad = int((~finite).sum())
        return Comparison(name=name, passed=False, n_elements=n,
                          reason=f"{bad} non-finite value(s)")

    diff = (a - b).abs()
    max_abs = float(diff.max())
    if max_abs == 0.0:
        # Identical tensors. Worth special-casing: cosine is undefined for two zero
        # vectors, so an all-zero output — a fully masked region, say — would
        # otherwise be reported as a mismatch against itself.
        return Comparison(name=name, passed=True, n_elements=n)
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max())
    close = diff <= (tol.atol + tol.rtol * b.abs())
    pass_fraction = float(close.float().mean())
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12))
    flat_worst = int(diff.argmax())
    worst_index = list(_unravel(flat_worst, tuple(reference.shape)))

    reasons: list[str] = []
    if pass_fraction < tol.min_pass_fraction:
        reasons.append(f"pass fraction {pass_fraction:.5f} < {tol.min_pass_fraction}")
    if cosine < tol.min_cosine:
        reasons.append(f"cosine {cosine:.6f} < {tol.min_cosine}")

    return Comparison(
        name=name, passed=not reasons, max_abs_err=max_abs, max_rel_err=max_rel,
        pass_fraction=pass_fraction, cosine=cosine, n_elements=n,
        reason="; ".join(reasons), worst_index=worst_index,
    )


def compare_outputs(actual: Any, reference: Any, label: str = "output",
                    tolerance: Tolerance | None = None) -> list[Comparison]:
    """Compare every tensor in a possibly-nested output structure.

    One comparison per reference tensor, named by its path. Attention modules
    commonly return a tuple, and checking only the first element would let a
    corrupted auxiliary output through — so every caller that compares a module's
    output uses this rather than reducing to one tensor.
    """
    from model_partition.trace import is_tensor

    pairs = list(walk_pairs(actual, reference, label))
    if not pairs:
        from model_partition.runtime.module_runner import first_tensor

        return [compare(first_tensor(actual), first_tensor(reference), label, tolerance)]
    return [compare(a, b, name, tolerance) for name, a, b in pairs if is_tensor(b)]


def walk_pairs(actual: Any, reference: Any, path: str = "output"):
    """Yield ``(path, actual, reference)`` for each tensor in the reference tree."""
    from model_partition.trace import is_tensor

    if is_tensor(reference):
        yield path, actual, reference
        return
    if isinstance(reference, dict):
        for key, value in reference.items():
            child = actual.get(key) if isinstance(actual, dict) else None
            yield from walk_pairs(child, value, f"{path}.{key}")
        return
    if isinstance(reference, (list, tuple)):
        for i, value in enumerate(reference):
            child = actual[i] if isinstance(actual, (list, tuple)) and i < len(actual) else None
            yield from walk_pairs(child, value, f"{path}[{i}]")


def _unravel(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords: list[int] = []
    for size in reversed(shape):
        coords.append(flat % size)
        flat //= size
    return tuple(reversed(coords))


def top1_agreement(actual: Any, reference: Any) -> float:
    """Fraction of positions where the argmax over the last axis agrees."""
    import torch

    if actual is None or reference is None or actual.shape != reference.shape:
        return 0.0
    with torch.no_grad():
        return float((actual.argmax(-1) == reference.argmax(-1)).float().mean())
