# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a model through transformers, for repos that ship no inference code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_partition.loaders.base import LoadedModel, LoaderError, torch_dtype

#: Auto classes tried in order. Multimodal repos expose
#: ``*ForConditionalGeneration``, which the causal-LM class refuses.
AUTO_CLASSES = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModel",
)


@dataclass
class TransformersLoader:
    """Instantiate via the first auto class that accepts this config."""

    root: Path
    config: dict[str, Any]
    dtype: str = "bfloat16"
    trust_remote_code: bool = False

    def _auto_classes(self):
        import transformers

        found = [(name, getattr(transformers, name)) for name in AUTO_CLASSES
                 if hasattr(transformers, name)]
        if not found:
            raise LoaderError("transformers exposes none of the expected auto classes")
        return found

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

        auto_config = self._auto_config()
        errors: list[str] = []
        for name, auto_class in self._auto_classes():
            try:
                with torch.device("meta"):
                    model = auto_class.from_config(
                        auto_config, trust_remote_code=self.trust_remote_code,
                    )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue
            return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                               device="meta", meta=True,
                               metadata={"loader": "transformers", "auto_class": name})
        raise LoaderError("Could not instantiate model structure:\n  " + "\n  ".join(errors))

    def build_config_only(self, device: str = "cpu") -> LoadedModel:
        """Instantiate from config with real storage and normal initialization.

        Unlike meta + ``to_empty``, this runs each module's own init, so
        non-persistent buffers (rotary ``inv_freq``, masks) hold correct values.
        """
        import torch

        auto_config = self._auto_config()
        errors: list[str] = []
        for name, auto_class in self._auto_classes():
            try:
                model = auto_class.from_config(
                    auto_config, trust_remote_code=self.trust_remote_code,
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue
            model = model.to(device=device, dtype=torch_dtype(self.dtype))
            model.eval()
            torch.set_grad_enabled(False)
            return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                               device=device,
                               metadata={"loader": "transformers", "auto_class": name})
        raise LoaderError("Could not instantiate from config:\n  " + "\n  ".join(errors))

    def build(self, state_dict: dict[str, Any] | None = None, device: str = "cpu") -> LoadedModel:
        """Load real weights. ``state_dict`` overrides the checkpoint on disk."""
        import torch

        errors: list[str] = []
        model = None
        used = ""
        for name, auto_class in self._auto_classes():
            try:
                if state_dict is None:
                    model = auto_class.from_pretrained(
                        str(self.root), dtype=torch_dtype(self.dtype),
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    with torch.device("meta"):
                        model = auto_class.from_config(
                            self._auto_config(), trust_remote_code=self.trust_remote_code,
                        )
                    model.to_empty(device=device)
                    model.load_state_dict(state_dict, strict=False)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue
            used = name
            break
        if model is None:
            raise LoaderError(f"Could not load weights from {self.root}:\n  " + "\n  ".join(errors))

        model = model.to(device=device, dtype=torch_dtype(self.dtype))
        model.eval()
        torch.set_grad_enabled(False)
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device=device, metadata={"loader": "transformers", "auto_class": used})
