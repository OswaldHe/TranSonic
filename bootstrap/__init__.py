# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap Trainium NKI kernels for the modules of a partition artifact.

`autohelix bootstrap` takes one module of a published partition run, materializes it as a
self-contained git repo, and runs an AutoHelix-style loop whose gate is a fixed set of
NKI requirements rather than a metric. See `bootstrap/README.md`.

Nothing is imported here: the CLI is loaded lazily so `autohelix` stays fast for everyone
who is not bootstrapping a kernel, and the checker depends on the standard library alone.
"""

__all__ = ["cli", "driver", "materialize", "nki_checker", "preset", "templates"]
