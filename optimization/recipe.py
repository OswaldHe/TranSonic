# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checking a declared cut arithmetically, without knowing what the module computes.

The agent decides how to divide a module into per-rank work, which means no script can check that
the division is *right*. But the agent also has to declare how its ranks recombine — stage 4 needs
that to build the assembly at all — and a declaration of that form is checkable: take the N
per-rank goldens the agent dumped, apply the declared recipe, and see whether the result is the
module's recorded output.

That is the whole idea here, and it is worth being clear about what it does and does not prove.

It proves the *cut closes*: the parts, combined the declared way, are the whole. A cut that drops
an expert, double-counts a shared path, shards along the wrong axis, or dumps a golden from the
wrong rank fails, because none of those sum back to the recorded output. It is exactly the class
of error that would otherwise survive stage 2 and stage 3 — ten iterations spent optimizing a
kernel for the wrong subproblem — and only surface when stage 4 could not hit the golden.

It does not prove the cut is *good*. A recipe declaring one rank does everything and the other
three return zeros reproduces the golden perfectly. Stage 5's "no slower than 1.1x the submodule"
bound is what makes that cut fail, because three idle ranks cannot make the whole module fast.

Two operations, because two cover the parallelism the closed `PARTITION_DIMS` vocabulary allows:
summing partials (any split rejoined by `allreduce` — tensor parallelism, MoE expert parallelism,
whose all-to-all moves tokens but whose combine is still a sum) and concatenating shards (any split
rejoined by `allgather` — head, vocab, ngram). Anything else is a dimension the floorplan cannot
express either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Recipe operations. Closed, for the same reason `PARTITION_DIMS` is: an op the verifier cannot
#: evaluate is a claim it cannot check, and an unchecked claim here is worse than a rejected one.
OPERATIONS = frozenset({"sum", "concat"})

#: `np.dtype` names a recipe may name. bfloat16 has no numpy dtype, so it travels as uint16 and is
#: widened for the comparison — which is also the right thing numerically, since comparing bf16
#: sums in bf16 would charge the verifier for rounding the recipe never performed.
DTYPES: dict[str, Any] = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "bfloat16": "bfloat16",
    "int8": np.int8,
    "uint8": np.uint8,
    "int32": np.int32,
}


class RecipeError(ValueError):
    """The declared recipe cannot be evaluated, so nothing can be concluded from it."""


@dataclass
class RecipeOutcome:
    """Whether the declared reassembly reproduces the module's recorded output."""

    reproduces: bool
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"reproduces": self.reproduces, "detail": self.detail, "metrics": self.metrics}


def _load(path: Path, dtype: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    """Read a raw little-endian `.bin` as the declared dtype and shape.

    The same no-header convention `bootstrap init` writes and the module READMEs document: the
    dtype and shape in the declaration are the complete description of the file, so a wrong one
    shows up as a size mismatch rather than as silently reinterpreted bytes.
    """
    if not path.is_file():
        raise RecipeError(f"{path.name} named by the recipe does not exist")
    if dtype not in DTYPES:
        raise RecipeError(
            f"the recipe names dtype '{dtype}', which this verifier cannot read. "
            f"Known: {', '.join(sorted(DTYPES))}"
        )
    raw = np.fromfile(path, dtype=np.uint8)
    if dtype == "bfloat16":
        # bfloat16 is the top 16 bits of a float32, so widening is an exact shift — no rounding,
        # and no dependency on a numpy build that knows the dtype.
        array = (raw.view("<u2").astype("<u4") << 16).view("<f4")
    else:
        array = raw.view(np.dtype(DTYPES[dtype]).newbyteorder("<"))
    expected = int(np.prod(shape)) if len(shape) else 1
    if array.size != expected:
        raise RecipeError(
            f"{path.name} holds {array.size} element(s) of {dtype}, but the recipe declares "
            f"shape {tuple(shape)} ({expected} element(s))"
        )
    return array.reshape(tuple(shape)).astype(np.float64)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise RecipeError(f"{where} does not state '{key}'")
    return mapping[key]


def validate_shards(
    recipe: dict[str, Any], factor: int | None, module_golden: str | None,
) -> list[str]:
    """Findings if the declared shard list is not one golden per rank.

    The check that keeps this verifier honest. Without it a declaration can name the module's own
    recorded output as its single shard and reproduce the target exactly — the arithmetic passes
    while no rank output was involved at all, which is the one way a bad cut could get through the
    only early check standing against it.

    Three requirements, each closing a distinct way to satisfy the sum without doing the work:
    as many shards as there are ranks, all of them distinct, and none of them the module golden.
    """
    findings: list[str] = []
    shards = [str(s) for s in (recipe.get("shards") or [])]

    if factor is not None and len(shards) != factor:
        findings.append(
            f"reassembly.shards names {len(shards)} golden(s) but the cut is {factor}-way. "
            f"There has to be one per rank, or the sum is not over the whole module"
        )
    duplicated = sorted({s for s in shards if shards.count(s) > 1})
    if duplicated:
        findings.append(
            f"reassembly.shards repeats {', '.join(duplicated)}. Each rank's golden is a different "
            f"tensor; naming one twice counts its contribution twice and omits another rank's"
        )
    if module_golden:
        wanted = Path(module_golden).name
        for shard in shards:
            if Path(shard).name == wanted:
                findings.append(
                    f"reassembly.shards names '{shard}', which is the module's own recorded output. "
                    f"Reassembling the answer from the answer proves nothing about the cut"
                )
    return findings


def apply_recipe(repo: Path, recipe: dict[str, Any]) -> np.ndarray:
    """Combine the per-rank goldens the recipe names into one array.

    Pure arithmetic over files on disk: no device, no import of the candidate, nothing that could
    let the repo influence the answer.
    """
    op = str(_require(recipe, "op", "reassembly"))
    if op not in OPERATIONS:
        raise RecipeError(
            f"reassembly names op '{op}'. Known: {', '.join(sorted(OPERATIONS))} — a split this "
            f"verifier cannot evaluate is a claim it cannot check"
        )
    shards = _require(recipe, "shards", "reassembly")
    if not isinstance(shards, list) or not shards:
        raise RecipeError("reassembly.shards must be a non-empty list of per-rank golden files")
    dtype = str(_require(recipe, "dtype", "reassembly"))
    shape = _require(recipe, "shape", "reassembly")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise RecipeError("reassembly.shape must be a non-empty list of dimensions")

    arrays = [_load(repo / str(s), dtype, shape) for s in shards]

    if op == "sum":
        combined = arrays[0].copy()
        for array in arrays[1:]:
            combined += array
    else:
        dim = int(recipe.get("dim", 0))
        if not -len(shape) <= dim < len(shape):
            raise RecipeError(
                f"reassembly.dim is {dim}, outside the {len(shape)} dimension(s) of "
                f"{tuple(shape)}"
            )
        combined = np.concatenate(arrays, axis=dim)

    for extra in recipe.get("then_add") or []:
        # A term added *after* the ranks rejoin: for expert parallelism this is where a shared
        # expert legitimately lives, since the reference adds it after its all-reduce rather than
        # inside any rank's partial. Charged at the combined shape, so a mis-shaped one fails here.
        combined = combined + _load(repo / str(extra), dtype, combined.shape)
    return combined


def compare(
    got: np.ndarray, want: np.ndarray, bar: dict[str, float],
) -> tuple[bool, dict[str, float], str]:
    """The module's own bar, applied to two arrays: closeness, cosine and a hard ceiling.

    Reimplements the artifact's `compare_outputs` rather than importing it, for the same reason
    the bootstrapped validators do: the comparison has to be legible where it is used, and an
    imported one can be changed somewhere else.
    """
    if got.shape != want.shape:
        # A leading batch axis is allowed to differ, because the reference itself flattens: the
        # module's golden is `(1, 128, 5120)` while a rank's partial is the `(128, 5120)` that
        # `x.view(-1, dim)` produces and `y.view(shape)` restores. Same elements in the same order,
        # so comparing them is comparing the same tensor — the first real run failed here for
        # exactly that, with a cut that was otherwise correct.
        if got.size != want.size:
            return False, {}, (
                f"shape {got.shape} holds {got.size} element(s) against the recorded "
                f"{want.shape}'s {want.size}"
            )
        got = got.reshape(want.shape)

    a, b = got.ravel(), want.ravel()
    abs_err = np.abs(a - b)
    tolerance = bar["ATOL"] + bar["RTOL"] * np.abs(b)
    close = abs_err <= tolerance
    pass_fraction = float(close.mean()) if close.size else 0.0
    max_abs_err = float(abs_err.max()) if abs_err.size else 0.0

    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / norm) if norm > 0 else (1.0 if not np.any(a - b) else 0.0)

    metrics = {
        "pass_fraction": round(pass_fraction, 6),
        "cosine": round(cosine, 8),
        "max_abs_err": round(max_abs_err, 8),
    }
    reasons: list[str] = []
    if pass_fraction < bar["MIN_PASS_FRACTION"]:
        reasons.append(
            f"only {pass_fraction:.4%} of elements within RTOL/ATOL "
            f"(needs {bar['MIN_PASS_FRACTION']:.4%})"
        )
    if cosine < bar["MIN_COSINE"]:
        reasons.append(f"cosine {cosine:.6f} below {bar['MIN_COSINE']:.6f}")
    if max_abs_err > bar["MAX_ABS_ERR"]:
        reasons.append(
            f"worst element off by {max_abs_err:g}, over the ceiling of {bar['MAX_ABS_ERR']:g}"
        )
    return not reasons, metrics, "; ".join(reasons)


def _factor(declaration: dict[str, Any]) -> int | None:
    """The declared cut width, when it is stated as a positive integer."""
    try:
        factor = int(declaration.get("factor"))
    except (TypeError, ValueError):
        return None
    return factor if factor > 0 else None


def verify_recipe(
    repo: Path, declaration: dict[str, Any], manifest: dict[str, Any],
) -> RecipeOutcome:
    """Does the declared reassembly of the per-rank goldens reproduce the module's output?

    The golden and the bar come from the *manifest* — written when the repo was built, from the
    bootstrapped module — never from the declaration, so a declaration cannot nominate an easier
    target to be checked against.
    """
    recipe = _require(declaration, "reassembly", "the declaration")
    if not isinstance(recipe, dict):
        raise RecipeError("the declaration's 'reassembly' must be a mapping")

    module_golden = (manifest.get("module_output") or {})
    rel = str(module_golden.get("file") or "")
    if not rel:
        raise RecipeError(
            "the manifest records no module_output, so there is nothing to reassemble towards"
        )
    want = _load(repo / rel, str(module_golden.get("dtype", "bfloat16")),
                 list(module_golden.get("shape") or []))

    bar_source = manifest.get("module_tolerance") or manifest.get("tolerance") or {}
    missing = [n for n in ("RTOL", "ATOL", "MIN_COSINE", "MIN_PASS_FRACTION", "MAX_ABS_ERR")
               if n not in bar_source]
    if missing:
        raise RecipeError(
            f"the manifest records no {', '.join(missing)} for the module's own output, so the "
            f"reassembly has no bar to be judged at"
        )
    bar = {k: float(bar_source[k]) for k in bar_source if k in {
        "RTOL", "ATOL", "MIN_COSINE", "MIN_PASS_FRACTION", "MAX_ABS_ERR"}}

    problems = validate_shards(
        recipe, _factor(declaration), str(module_golden.get("file") or ""),
    )
    if problems:
        return RecipeOutcome(reproduces=False, detail="; ".join(problems))

    got = apply_recipe(repo, recipe)
    ok, metrics, detail = compare(got, want, bar)
    if ok:
        detail = (
            f"{len(recipe.get('shards') or [])} shard(s) combined by "
            f"'{recipe.get('op')}' reproduce {rel}"
        )
    return RecipeOutcome(reproduces=ok, detail=detail, metrics=metrics)
