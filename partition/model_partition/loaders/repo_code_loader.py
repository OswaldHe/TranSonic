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
from model_partition.loaders.streamed import rename as rename_key

#: Factory names tried in order, called as ``fn(config)`` or ``fn(config, state_dict)``.
BUILDER_NAMES = ("build_model", "build", "load_model", "create_model")

#: Class names tried when no factory function is found, called as ``Cls(config)``.
CLASS_NAMES = ("Transformer", "Model", "ModelForCausalLM", "CausalLM")

#: Dataclass names a vendor module uses for its own settings object. Reference
#: implementations usually take one of these rather than a plain dict, so the
#: config is converted before the factory sees it.
ARGS_CLASS_NAMES = ("ModelArgs", "Args", "ModelConfig", "Config", "TransformerConfig")

#: Spec dtypes meaning "whatever the model builds itself as". A quantized checkpoint's
#: own dtypes are the point: its fp8 and fp4 weights are a quarter the size of the
#: bfloat16 they would be cast to, and the vendor's kernels take them as they are.
NATIVE_DTYPES = ("checkpoint", "native", "auto", "")


@dataclass
class RepoCodeLoader:
    """Instantiate a model via the repo's own entry module."""

    root: Path
    entry: str
    config: dict[str, Any]
    dtype: str = "bfloat16"
    code_paths: tuple[str, ...] = ()
    trust_remote_code: bool = False
    #: Rename rules taking checkpoint keys to this model's parameter names.
    rename: tuple[tuple[str, str], ...] = ()
    #: Passed to the factory when it declares a parameter for one. A reference
    #: implementation may need it at construction — an n-gram memory layer builds its
    #: token map from the tokenizer's vocabulary — and then no config can stand in.
    tokenizer: Any = None

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

    def _settings(self, module) -> Any:
        """The object the factory wants: the vendor's args dataclass, or the dict.

        A reference implementation typically reads ``args.dim`` rather than
        ``config["dim"]``, so the config is fed through the module's own settings
        class — keeping only the fields it declares, since a repo's config file
        carries more than its model code uses.
        """
        from dataclasses import fields, is_dataclass

        for name in ARGS_CLASS_NAMES:
            candidate = getattr(module, name, None)
            if not (isinstance(candidate, type) and is_dataclass(candidate)):
                continue
            declared = {f.name: f for f in fields(candidate)}
            known = {}
            for key, value in self.config.items():
                field = declared.get(key)
                if field is None:
                    continue
                # JSON has no tuples, so a field declared as one arrives as a list.
                if isinstance(value, list) and "tuple" in str(field.type).lower():
                    value = tuple(value)
                known[key] = value
            return candidate(**known)
        return self.config

    def _instantiate(self, state_dict: dict[str, Any] | None):
        module = self._import_entry()
        factory, kind = self._find_factory(module)
        settings = self._settings(module)
        extra = self._factory_extras(factory)
        try:
            if kind == "function":
                try:
                    return factory(settings, state_dict, **extra)
                except TypeError:
                    return self._load_into(factory(settings, **extra), state_dict)
            return self._load_into(factory(settings, **extra), state_dict)
        except Exception as exc:
            raise LoaderError(f"{self.entry} factory failed: {exc}") from exc

    def _factory_extras(self, factory: Any) -> dict[str, Any]:
        """Arguments beyond the settings object that this factory declares."""
        import inspect

        if self.tokenizer is None:
            return {}
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            return {}
        if "tokenizer" not in parameters:
            return {}
        # The vendor's code reaches into the tokenizer's own API — a fast tokenizer's
        # `backend_tokenizer`, to enumerate the vocabulary — so it gets the real one
        # rather than the loop's wrapper around it.
        return {"tokenizer": getattr(self.tokenizer, "impl", self.tokenizer)}

    def _load_into(self, model: Any, state_dict: dict[str, Any] | None) -> Any:
        """Apply a state dict, keyed the way this model names its parameters.

        A real checkpoint carries tensors for parts this run excludes — an MTP head, a
        vision tower — so unexpected keys are fine and a strict load would reject the
        whole thing. A *missing* key is not fine: the parameter then keeps whatever it
        was initialized with, and a trace of that would be dumped, verified against
        itself and reported as correct. So the checkpoint's names are mapped onto the
        model's first, and anything still unmatched is an error rather than a shrug.
        """
        if state_dict is None:
            return model
        if self.rename:
            state_dict = {rename_key(k, self.rename): v for k, v in state_dict.items()}
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = [name for name in getattr(incompatible, "missing_keys", ())
                   if not _is_derived(model, name)]
        if missing:
            raise LoaderError(
                f"{len(missing)} parameter(s) are not in the checkpoint and would run on "
                f"their initial values: {', '.join(missing[:6])}. Map the checkpoint's "
                "names with checkpoint.rename in the spec, or narrow the scope."
            )
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

    def build_streamed(self, device: str = "cuda") -> LoadedModel:
        """Build on meta and read each module's weights only while it runs.

        For a checkpoint that cannot be resident. The dtypes are the model's own, which
        for a quantized checkpoint means its own: casting to bfloat16 is what would make
        it not fit.
        """
        import torch

        from model_partition.loaders.streamed import (
            build_index, install_streaming, sparse_allocation,
        )

        # Big tensors are placeholders to be streamed; the small ones the model computes
        # for itself are made for real, because no checkpoint can restore them.
        with sparse_allocation():
            model = self._instantiate(None)
        model.eval()
        torch.set_grad_enabled(False)
        shards = sorted(self.root.glob("*.safetensors"))
        if not shards:
            raise LoaderError(f"No safetensors shards under {self.root} to stream from")
        report = install_streaming(model, build_index(shards, self.rename), device=device)
        if report.unresolved:
            raise LoaderError(
                f"{len(report.unresolved)} parameter(s) have no tensor in the checkpoint "
                f"and no value of their own: {', '.join(report.unresolved[:6])}. Running "
                "on placeholders would verify nothing; check the shards are all present "
                "and that checkpoint.rename maps this checkpoint's names."
            )
        # Whatever was computed at construction landed on the host; the forward is on the
        # accelerator.
        for module in model.modules():
            for store in (module._parameters, module._buffers):
                for name, tensor in list(store.items()):
                    if tensor is not None and not tensor.is_meta and tensor.device.type != device.split(":")[0]:
                        store[name] = (torch.nn.Parameter(tensor.to(device), requires_grad=False)
                                       if store is module._parameters else tensor.to(device))
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device=device, placement="streamed",
                           metadata={"loader": "repo_code", "streaming": report.summary()})

    def _finalize(self, model: Any, device: str) -> LoadedModel:
        import torch

        if self.dtype in NATIVE_DTYPES:
            model = model.to(device=device)
        else:
            model = model.to(device=device, dtype=torch_dtype(self.dtype))
        model.eval()
        torch.set_grad_enabled(False)
        return LoadedModel(model=model, config=self.config, dtype=self.dtype,
                           device=device, metadata={"loader": "repo_code"})


def _is_derived(model: Any, name: str) -> bool:
    """Whether a parameter is one the model computes rather than loads.

    A non-persistent buffer is excluded from a state dict by design — rotary
    frequencies, a causal mask — so its absence from the checkpoint says nothing.
    """
    holder, _, leaf = name.rpartition(".")
    module = model
    for part in holder.split(".") if holder else []:
        module = getattr(module, part, None)
        if module is None:
            return False
    return leaf in (getattr(module, "_non_persistent_buffers_set", set()) or set())
