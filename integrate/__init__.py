# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix integrate` — drive a repo that does not satisfy its constraints to one that does.

The inverse of `optimization/`. There the baseline passes and the loop makes it faster; here the
baseline fails and the loop makes it pass. The metric is how many constraints hold, and a
constraint may be a script or an agent's judgement.

First stage: `baseline`, an unbounded loop that exits when every constraint is satisfied.
"""

__all__ = ["config", "constraints_doc"]

#: Short form of the two things that differ from `optimize`, for anything that needs to say so.
constraints_doc = (
    "the baseline is expected to fail, and a constraint may be an agent's judgement "
    "rather than a script's exit code"
)
