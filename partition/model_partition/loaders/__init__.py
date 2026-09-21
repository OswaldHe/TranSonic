# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model loader selection."""

from __future__ import annotations

from model_partition.loaders.base import LoadedModel, LoaderError, ModelLoader, torch_dtype
from model_partition.loaders.repo_code_loader import RepoCodeLoader
from model_partition.loaders.transformers_loader import TransformersLoader

__all__ = [
    "LoadedModel", "LoaderError", "ModelLoader", "RepoCodeLoader",
    "TransformersLoader", "build_loader", "torch_dtype",
]


def build_loader(result, device_map=None, max_memory=None, tokenizer=None,
                 compat_paths=(), device_info=None) -> ModelLoader:
    """Construct the loader named by an :class:`~model_partition.ingest.IngestResult`.

    ``device_map``/``max_memory`` request layer placement across GPU and host, so
    a checkpoint larger than the GPU still runs most of its compute there; the vendor
    path instead streams a module's weights in for its forward and drops them after,
    which is what a checkpoint larger than GPU *and* host needs. ``tokenizer`` is for a
    vendor factory that builds part of the model from the vocabulary.
    """
    spec = result.spec
    if result.loader == "repo_code":
        return RepoCodeLoader(
            root=result.root,
            entry=result.entry or spec.entry or "",
            config=result.config,
            dtype=spec.dtype,
            code_paths=tuple(result.code_paths or spec.code_paths),
            trust_remote_code=spec.trust_remote_code,
            rename=spec.checkpoint.rename,
            tokenizer=tokenizer,
            compat_paths=tuple(compat_paths),
            device_info=device_info,
        )
    if result.loader == "transformers":
        return TransformersLoader(
            root=result.root,
            config=result.config,
            dtype=spec.dtype,
            trust_remote_code=spec.trust_remote_code,
            device_map=device_map,
            max_memory=max_memory,
        )
    raise LoaderError(f"Unknown loader {result.loader!r}")
