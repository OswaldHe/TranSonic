#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smallest real end-to-end run: Qwen3.5-0.8B through every stage.
# Downloads ~1.6 GiB on the first run.
#
#   bash partition/scripts/run_end_to_end.sh              # real Claude judge
#   JUDGE=stub bash partition/scripts/run_end_to_end.sh   # offline
set -euo pipefail

SPEC="${SPEC:-qwen3.5-0.8b}"
JUDGE="${JUDGE:-claude}"
TOKENS="${TOKENS:-24}"

cd "$(dirname "$0")/../.."
source .venv/bin/activate

python partition/scripts/check_env.py
echo

if [ ! -f partition/inputs/long.jsonl ]; then
  echo "generating the long-context input set..."
  python partition/inputs/fetch_long_inputs.py
  echo
fi

autohelix partition inspect "$SPEC"
echo

autohelix partition run "$SPEC" --judge "$JUDGE" --max-new-tokens "$TOKENS" -n 2
