# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a model through transformers, for repos that ship no inference code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_partition.loaders.base import LoadedModel, LoaderError, torch_dtype


@dataclass
class TransformersLoader:
    """Instantiate via ``AutoModelForCausalLM``."""

    root: Path
    config: dict[str, Any]
    dtype: str = "bfloat16"
    trust_remote_code: bool = False

    def _auto_config(self):
        try:
            from transformers import AutoConfig
        except ImportError as exc:  # pragma: no cover
            raise LoaderError("transformers is required for the transformers loader") from exc
        try:
            return AutoConfig.from_pretrained(str(self.root), trust_remote_code=self.trust_remote_code)
        except Exception as exc:
            model_type = self.config.get("model_type", "?")
            raise LoaderError(
                f"transformers cannot read this config (model_type={model_type!r}): {exc}. "
                "If the repo ships its own inference code, use loader: repo_code."
            ) from exc

    def build_meta(self) -> LoadedModel:
        """Structure only, on the meta device — no weights, no memory."""
        import torch

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover
            raise LoaderError("transformers is required for the transformers loader") from exc
        auto_config = self._auto_config()
        try:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    auto_config, trust_remote_code=self.trust_remote_code,
                )
        except Exception as exc:
            raise LoaderError(f"Could not instantiate model structure: {exc}") from exc
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device="meta", meta=True, metadata={"loader": "transformers"})

    def build(self, state_dict: dict[str, Any] | None = None, device: str = "cpu") -> LoadedModel:
        """Load real weights. ``state_dict`` overrides the checkpoint on disk."""
        import torch

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover
            raise LoaderError("transformers is required for the transformers loader") from exc

        try:
            if state_dict is None:
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.root),
                    dtype=torch_dtype(self.dtype),
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(
                        self._auto_config(), trust_remote_code=self.trust_remote_code,
                    )
                model.to_empty(device=device)
                model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise LoaderError(f"Could not load weights from {self.root}: {exc}") from exc

        model = model.to(device=device, dtype=torch_dtype(self.dtype))
        model.eval()
        torch.set_grad_enabled(False)
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device=device, metadata={"loader": "transformers"})
