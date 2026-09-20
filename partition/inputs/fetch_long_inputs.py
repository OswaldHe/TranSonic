#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the long-context input set.

Two sources:

* ``synthetic`` (default) — RULER-style needle-in-a-haystack and multi-hop
  prompts, generated from a fixed seed. Fully reproducible, no download, no
  licensing question, and the needle gives a cheap signal that long-range
  attention survived partitioning.
* ``longbench`` — real benchmark text from a pinned dataset revision, fetched at
  setup time. Nothing licensed is committed to this repo.

Token budgets are approximate here; the loop trims or extends to the exact
``target_tokens`` with the model's own tokenizer.

    python fetch_long_inputs.py                       # synthetic, 2k/8k/16k
    python fetch_long_inputs.py --budgets 4096 32768
    python fetch_long_inputs.py --source longbench --repo zai-org/LongBench-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

OUT_DEFAULT = Path(__file__).resolve().parent / "long.jsonl"
DEFAULT_BUDGETS = (2048, 8192, 16384)

#: ~0.75 tokens per word for English is the usual rule of thumb; we overshoot
#: slightly so trimming, not padding, does the final work.
WORDS_PER_TOKEN = 0.8

FILLER_SENTENCES = (
    "The maintenance log for sector {n} records routine calibration with no anomalies.",
    "Shipment {n} departed the northern depot ahead of schedule and arrived intact.",
    "Sensor array {n} reported nominal temperatures throughout the observation window.",
    "The archivist catalogued crate {n} under general supplies pending further review.",
    "Team {n} completed the quarterly inspection and filed the standard report.",
    "Batch {n} passed visual inspection with no deviations from the specification.",
    "Route {n} was surveyed in clear weather and marked as passable year round.",
    "Ledger entry {n} balances against the corresponding receipt with no discrepancy.",
)

NEEDLE = ("The access code for the {place} vault is {code}, and it must be entered "
          "before the second alarm cycle completes.")

PLACES = ("harbour", "observatory", "foundry", "greenhouse", "archive", "signal station")


def synthetic_sample(budget: int, index: int, rng: random.Random, kind: str) -> dict:
    """Build one long prompt with a verifiable fact buried inside it."""
    target_words = int(budget * WORDS_PER_TOKEN * 1.15)
    place = rng.choice(PLACES)
    code = f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNP')}{rng.randint(10, 99)}"

    lines: list[str] = [
        "The following is an internal operations record. Read it carefully; you "
        "will be asked about a specific detail at the end.",
        "",
    ]
    words = sum(len(line.split()) for line in lines)
    counter = 0
    needle_at = rng.uniform(0.35, 0.75) if kind == "needle" else rng.uniform(0.1, 0.4)
    needle_placed = False
    second_needle = None
    if kind == "multihop":
        second_needle = (f"The {place} vault was relocated last spring; its records are "
                         f"now filed under depot {rng.randint(10, 99)}.")

    while words < target_words:
        counter += 1
        if not needle_placed and words >= target_words * needle_at:
            sentence = NEEDLE.format(place=place, code=code)
            needle_placed = True
        elif second_needle and needle_placed and words >= target_words * 0.85:
            sentence, second_needle = second_needle, None
        else:
            sentence = rng.choice(FILLER_SENTENCES).format(n=counter)
        lines.append(sentence)
        words += len(sentence.split())

    if not needle_placed:
        lines.append(NEEDLE.format(place=place, code=code))
    lines += ["", f"Question: what is the access code for the {place} vault, and what "
                  "must happen before it is entered?", "Answer:"]

    return {
        "id": f"long-{kind}-{budget}-{index}",
        "role": "raw",
        "target_tokens": budget,
        "source": f"synthetic:{kind}",
        "expected_substring": code,
        "prompt": "\n".join(lines),
    }


def build_synthetic(budgets: tuple[int, ...], per_budget: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    for budget in budgets:
        for index in range(per_budget):
            kind = "needle" if index % 2 == 0 else "multihop"
            samples.append(synthetic_sample(budget, index, rng, kind))
    return samples


def build_longbench(repo: str, revision: str | None, budgets: tuple[int, ...],
                    per_budget: int) -> list[dict]:
    """Pull real long-context text from a pinned dataset revision."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The longbench source needs the datasets package:\n"
            "    uv pip install datasets"
        ) from exc

    dataset = load_dataset(repo, revision=revision, split="train", streaming=True)
    wanted = len(budgets) * per_budget
    rows: list[dict] = []
    for row in dataset:
        text = row.get("context") or row.get("input") or row.get("text") or ""
        if isinstance(text, str) and len(text.split()) > 1500:
            rows.append(row)
        if len(rows) >= wanted:
            break
    if not rows:
        raise SystemExit(f"No sufficiently long rows found in {repo}")

    samples: list[dict] = []
    for position, budget in enumerate(budgets):
        for index in range(per_budget):
            row = rows[(position * per_budget + index) % len(rows)]
            context = row.get("context") or row.get("input") or row.get("text") or ""
            question = row.get("question") or row.get("query") or ""
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:" if question else context
            samples.append({
                "id": f"long-longbench-{budget}-{index}",
                "role": "raw",
                "target_tokens": budget,
                "source": f"{repo}@{revision or 'main'}",
                "prompt": prompt,
            })
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "longbench"], default="synthetic")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--per-budget", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--repo", default="zai-org/LongBench-v2",
                        help="Dataset repo for --source longbench")
    parser.add_argument("--revision", default=None, help="Pinned dataset revision")
    args = parser.parse_args(argv)

    budgets = tuple(args.budgets)
    if args.source == "synthetic":
        samples = build_synthetic(budgets, args.per_budget, args.seed)
    else:
        samples = build_longbench(args.repo, args.revision, budgets, args.per_budget)

    header = (f"# Generated by fetch_long_inputs.py (source={args.source}, "
              f"seed={args.seed}, budgets={list(budgets)}). Do not edit by hand.")
    body = "\n".join(json.dumps(sample) for sample in samples)
    args.out.write_text(f"{header}\n{body}\n")

    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    print(f"wrote {len(samples)} sample(s) to {args.out}")
    print(f"budgets: {list(budgets)}  sha256[:16]={digest}")
    for sample in samples:
        print(f"  {sample['id']:<28} ~{len(sample['prompt'].split())} words "
              f"-> target {sample['target_tokens']} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
