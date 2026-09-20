# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a model from the inference code its own repo ships.

The preferred path: when a vendor publishes a reference implementation, that code
defines the model's numerics and may cover architectures transformers does not
know. Importing it executes repo code, so the spec must opt in with
``trust_remote_code: true``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_partition.loaders.base import LoadedModel, LoaderError, torch_dtype

#: Factory names tried in order, called as ``fn(config)`` or ``fn(config, state_dict)``.
BUILDER_NAMES = ("build_model", "build", "load_model", "create_model")

#: Class names tried when no factory function is found, called as ``Cls(config)``.
CLASS_NAMES = ("Transformer", "Model", "ModelForCausalLM", "CausalLM")


@dataclass
class RepoCodeLoader:
    """Instantiate a model via the repo's own entry module."""

    root: Path
    entry: str
    config: dict[str, Any]
    dtype: str = "bfloat16"
    code_paths: tuple[str, ...] = ()
    trust_remote_code: bool = False

    def _entry_path(self) -> Path:
        path = self.root / self.entry
        if not path.is_file():
            raise LoaderError(f"Entry module not found: {path}")
        return path

    def _import_entry(self):
        if not self.trust_remote_code:
            raise LoaderError(
                f"Loading {self.entry} executes code from the model repo. Set "
                "trust_remote_code: true in the spec to allow it."
            )
        path = self._entry_path()
        module_name = f"_model_partition_repo_{abs(hash(str(path)))}"
        if module_name in sys.modules:
            return sys.modules[module_name]

        search_paths = [str(self.root)] + [str(self.root / p) for p in self.code_paths]
        added = [p for p in search_paths if p not in sys.path]
        sys.path[:0] = added
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise LoaderError(f"Could not load {path} as a Python module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise LoaderError(f"Failed importing {path}: {exc}") from exc
        finally:
            for p in added:
                if p in sys.path:
                    sys.path.remove(p)

    def _find_factory(self, module):
        for name in BUILDER_NAMES:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate, "function"
        for name in CLASS_NAMES:
            candidate = getattr(module, name, None)
            if isinstance(candidate, type):
                return candidate, "class"
        raise LoaderError(
            f"{self.entry} exposes none of {BUILDER_NAMES + CLASS_NAMES}; "
            "point 'entry' at a module with a model factory"
        )

    def _instantiate(self, state_dict: dict[str, Any] | None):
        module = self._import_entry()
        factory, kind = self._find_factory(module)
        try:
            if kind == "function":
                try:
                    return factory(self.config, state_dict)
                except TypeError:
                    return self._load_into(factory(self.config), state_dict)
            return self._load_into(factory(self.config), state_dict)
        except Exception as exc:
            raise LoaderError(f"{self.entry} factory failed: {exc}") from exc

    @staticmethod
    def _load_into(model: Any, state_dict: dict[str, Any] | None) -> Any:
        """Apply a state dict non-strictly.

        A real checkpoint carries tensors for parts this run excludes (MTP, a
        vision tower), and a strict load would reject the whole thing. Weights
        that genuinely fail to arrive are caught later: verification NaN-poisons
        each module before applying its dump.
        """
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        return model

    def build_meta(self) -> LoadedModel:
        import torch

        with torch.device("meta"):
            model = self._instantiate(None)
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device="meta", meta=True, metadata={"loader": "repo_code"})

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Merge the repo's safetensors shards into one state dict."""
        shards = sorted(self.root.glob("*.safetensors"))
        if not shards:
            return None
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise LoaderError("safetensors is required to load vendor-code weights") from exc
        state: dict[str, Any] = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return state

    def build_config_only(self, device: str = "cpu") -> LoadedModel:
        """Instantiate from config, without reading the checkpoint."""
        return self._finalize(self._instantiate(None), device)

    def build(self, state_dict: dict[str, Any] | None = None, device: str = "cpu") -> LoadedModel:
        """Instantiate with real weights.

        Reads the repo's checkpoint when no ``state_dict`` is supplied — otherwise
        the model would silently run on freshly initialized weights.
        """
        if state_dict is None:
            state_dict = self.load_checkpoint()
        return self._finalize(self._instantiate(state_dict), device)

    def _finalize(self, model: Any, device: str) -> LoadedModel:
        import torch

        model = model.to(device=device, dtype=torch_dtype(self.dtype))
        model.eval()
        torch.set_grad_enabled(False)
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device=device, metadata={"loader": "repo_code"})
