#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight: report what the partition loop needs and what is present."""

from __future__ import annotations

import shutil
import sys


def main() -> int:
    ok = True

    print("dependencies")
    for name in ("torch", "safetensors", "transformers", "huggingface_hub", "yaml", "jinja2"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "?")
            print(f"  {name:<18} {version}")
        except ImportError:
            print(f"  {name:<18} MISSING")
            ok = False

    print("\nhardware")
    from model_partition.hardware import detect_gpus, detect_host, format_bytes, resolve_budget

    gpus = detect_gpus()
    if not gpus:
        print("  no GPU detected (plan with --gpu-memory-gib)")
    for gpu in gpus:
        arch = f"  sm_{gpu.capability[0]}{gpu.capability[1]}" if gpu.capability else ""
        fp8 = "  fp8-capable" if gpu.supports_fp8() else "  no fp8 (DeepSeek needs dequant)"
        print(f"  gpu {gpu.index}: {gpu.name}  {format_bytes(gpu.total_bytes)}{arch}{fp8}")
    if gpus:
        budget = resolve_budget(gpus)
        print(f"  per-module budget: {format_bytes(budget.usable_bytes)}"
              f" at {budget.headroom:.0%} headroom")

    from model_partition.layout import DEFAULT_ARTIFACT_ROOT

    host = detect_host(DEFAULT_ARTIFACT_ROOT)
    print(f"  cpus: {host.cpu_count}")
    print(f"  ram: {format_bytes(host.ram_available_bytes)} available"
          f" of {format_bytes(host.ram_total_bytes)}")
    print(f"  disk: {format_bytes(host.disk_free_bytes)} free at {DEFAULT_ARTIFACT_ROOT}")

    print("\nagent and judge")
    claude = shutil.which("claude")
    print(f"  claude CLI: {claude or 'MISSING (agent planner and judge unavailable)'}")
    import os
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK"):
        print("  routing: Bedrock")
        print(f"  judge model: {os.environ.get('ANTHROPIC_DEFAULT_SONNET_MODEL', 'sonnet')}")
    else:
        print("  routing: default (CLAUDE_CODE_USE_BEDROCK not set)")

    print("\nOK" if ok else "\nMissing dependencies: pip install -e \".[partition]\"")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
