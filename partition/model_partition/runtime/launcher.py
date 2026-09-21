# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct a module from the source sitting next to it, plus config and weights.

This is what makes a module directory the deliverable rather than a description of
one. ``source.py`` is the implementation — the module's own classes, taken verbatim out
of the file that defines them — and this launches it: import that file, construct the
classes from the config recorded beside them, load the dumped weights in, hand back
something callable.

Nothing here reads the checkpoint or instantiates the whole model. The one thing not
vendored is the framework: ``source.py`` is loaded under its original package name so
its own imports resolve against the installed ``torch`` and ``transformers``. The
class bodies that run are the artifact's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCE_FILENAME = "source.py"

#: Leaf name the artifact's copy is registered under. Private, so it never takes the
#: place of the module it was copied from.
SOURCE_MODULE_PREFIX = "_model_partition_source_"

#: The config the module's subtree was built from, recorded by extraction.
CONFIG_FILENAME = "config.json"

#: Imported source per directory. Executing a modeling file is not cheap and every
#: module of a group asks for the same one.
_SOURCE_CACHE: dict[str, Any] = {}


def clear_source_cache() -> None:
    """Forget imported sources. Call when a ``source.py`` has been edited."""
    _SOURCE_CACHE.clear()


class LauncherError(RuntimeError):
    """Raised when a module cannot be built from its own source."""


@dataclass
class Launched:
    """A constructed module and how it was put together."""

    callable: Any
    class_name: str
    weights_loaded: int = 0
    missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = f"{self.class_name} from source.py, {self.weights_loaded} weight(s) loaded"
        if self.missing:
            text += f"; {len(self.missing)} parameter(s) left unset: {', '.join(self.missing[:4])}"
        return text


def load_source(directory: str | Path, source_module: str | None = None) -> Any:
    """Import the ``source.py`` beside a module directory.

    Loaded under the name its classes originally had, so the relative imports in a
    vendored modeling file still resolve. Falls back to a private name when no
    original is recorded.
    """
    import importlib.util
    import sys

    path = Path(directory) / SOURCE_FILENAME
    if not path.is_file():
        raise LauncherError(f"No {SOURCE_FILENAME} in {directory}")

    key = str(path.resolve())
    cached = _SOURCE_CACHE.get(key)
    if cached is not None:
        return cached

    # A private leaf name inside the original package: registering the artifact's copy
    # as `transformers...modeling_x` would shadow the installed module for everything
    # else in the process, while a bare private name would break the `from . import ...`
    # the file does. Naming it `<package>.<private>` gives it the right parent for
    # relative imports without taking the real module's place.
    leaf = f"{SOURCE_MODULE_PREFIX}{abs(hash(key))}"
    package = source_module.rsplit(".", 1)[0] if source_module and "." in source_module else ""
    name = f"{package}.{leaf}" if package else leaf
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LauncherError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # No `__pycache__` in the module directory: it is a deliverable someone reads and
    # copies, and a stale cache of a file they are editing is worse than no cache.
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise LauncherError(f"{path} failed to import: {exc}") from exc
    finally:
        sys.dont_write_bytecode = written
    _SOURCE_CACHE[key] = module
    return module


def module_config(directory: str | Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """The config recorded beside a module, or ``fallback`` when there is none.

    This is the config the module's own subtree was constructed with, which for a
    multimodal checkpoint is the text stack's config and not the model's — building a
    decoder layer from the top-level config gives library defaults for every width it
    does not name. A directory written before this was recorded falls back to the
    run-wide config.
    """
    import json

    path = Path(directory) / CONFIG_FILENAME
    if not path.is_file():
        return dict(fallback or {})
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise LauncherError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        return dict(fallback or {})
    return payload


def _resolve_class(module: Any, class_name: str) -> Any:
    """Find a class in ``source.py``, falling back to the framework.

    A group can mix the model's own classes with framework primitives — a norm and a
    ``Linear``. The model's classes are vendored; the primitives come from the
    installed framework, which is where ``source.py``'s own imports get them too.
    """
    found = getattr(module, class_name, None)
    if found is not None:
        return found
    try:
        import torch

        found = getattr(torch.nn, class_name, None)
    except ImportError:  # pragma: no cover
        found = None
    if found is None:
        raise LauncherError(
            f"{SOURCE_FILENAME} defines no {class_name} and neither does torch.nn"
        )
    return found


def build_group(
    directory: str | Path,
    config: dict[str, Any],
    weights: dict[str, Any],
    device: str = "cpu",
    source_module: str | None = None,
    class_names: list[str] | None = None,
    config_class: str | None = None,
    submodules: dict[str, list[str]] | None = None,
    layer_map: dict[str, list[int]] | None = None,
    submodule: str | None = None,
    source: Any = None,
) -> Any:
    """Build a whole partition module out of ``source.py`` and return it callable.

    One instance per submodule, each constructed from the source in this directory and
    given the weights recorded for it. A group spanning several submodules is returned
    as the chain of them, so the module runs as one computation without anything from
    the model being present.

    ``submodule`` asks for one piece alone: which expert of a parallel group, or which
    submodule when the implementations are installed into a model for emulation.

    ``config`` is what the caller has; the ``config.json`` in the directory wins when
    it is there, because that is the config this module's subtree was built from.
    """
    config = module_config(directory, config)
    paths = _module_paths(submodules or {}, weights)
    if not paths:
        raise LauncherError(
            "could not tell which module these weights belong to; keys look like "
            f"{next(iter(weights), '(none supplied)')}"
        )
    module_id, submodule_paths = paths
    names = list(class_names or [])
    if len(names) != len(submodule_paths):
        raise LauncherError(
            f"{module_id}: {len(submodule_paths)} submodule(s) but {len(names)} class "
            "name(s) recorded; re-run extraction"
        )

    wanted = [(path, name) for path, name in zip(submodule_paths, names)
              if submodule is None or path == submodule]
    if not wanted:
        raise LauncherError(f"{module_id} has no submodule {submodule!r}")

    layer_index = next(iter((layer_map or {}).get(module_id, [])), None)
    built = [
        build(directory, name, config, _weights_under(weights, path), device=device,
              source_module=source_module, layer_index=layer_index,
              config_class=config_class, source=source).callable
        for path, name in wanted
    ]
    return built[0] if len(built) == 1 else _chain(built)


def _module_paths(submodules: dict[str, list[str]],
                  weights: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Which module of the group these weights belong to, and its submodule paths.

    One implementation serves every module sharing a signature, so the group alone
    does not say which instance is running; the weight names do, because they are the
    original parameter names.
    """
    best: tuple[int, str, list[str]] | None = None
    for module_id, paths in submodules.items():
        matches = sum(1 for key in weights
                      if any(key == p or key.startswith(f"{p}.") for p in paths))
        if matches and (best is None or matches > best[0]):
            best = (matches, module_id, paths)
    return (best[1], best[2]) if best else None


def _weights_under(weights: dict[str, Any], path: str) -> dict[str, Any]:
    """The recorded weights belonging to one submodule, keyed relative to it."""
    prefix = f"{path}."
    subset = {key[len(prefix):]: value for key, value in weights.items()
              if key.startswith(prefix)}
    return subset or weights


def _chain(built: list[Any]) -> Any:
    """Run a group's submodules in order, each given the keywords it declares."""
    import torch

    def run(*args: Any, **kwargs: Any) -> Any:
        output = None
        flowing = args[0] if args else None
        rest = args[1:]
        with torch.no_grad():
            for index, target in enumerate(built):
                if index:
                    flowing = output[0] if isinstance(output, tuple) else output
                    rest = ()
                output = target(flowing, *rest, **_accepted(target, kwargs))
        return output

    return run


def _accepted(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The keywords this submodule declares, minus the flowing tensor it gets first."""
    import inspect

    forward = getattr(target, "forward", target)
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    names = list(parameters)
    flowing = names[0] if names else None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {k: v for k, v in kwargs.items() if k != flowing}
    return {k: v for k, v in kwargs.items() if k in set(names[1:])}


def build(
    directory: str | Path,
    class_name: str,
    config: dict[str, Any],
    weights: dict[str, Any],
    device: str = "cpu",
    source_module: str | None = None,
    layer_index: int | None = None,
    config_class: str | None = None,
    source: Any = None,
) -> Launched:
    """Construct ``class_name`` from ``source.py`` and load ``weights`` into it.

    ``source`` is the already-imported ``source.py`` when the caller imported it
    itself, which is how ``inference.py`` does it: the import is on the page there
    rather than hidden in here.
    """
    import torch

    module = source if source is not None else load_source(directory, source_module)
    cls = _resolve_class(module, class_name)

    settings = _config_object(module, config, config_class)
    instance = _construct(cls, settings, layer_index, weights, config)
    # The recorded weights decide the module's precision. A class constructs itself in
    # torch's default dtype, which is float32, and a bf16 checkpoint's feature maps
    # would then meet float32 parameters — the same matmul, refusing to run.
    dtype = _weights_dtype(weights)
    if dtype is not None:
        _cast_weights(instance, dtype)
    loaded, missing = load_weights(instance, weights)
    instance = instance.to(device)
    instance.eval()
    torch.set_grad_enabled(False)
    return Launched(callable=instance, class_name=class_name,
                    weights_loaded=loaded, missing=missing)


def _derived_buffers(instance: Any) -> set[str]:
    """Dotted names of the non-persistent buffers, which no checkpoint holds.

    A class marks a buffer non-persistent precisely because it is derived: it computes
    the value at construction and it is not part of the weights.
    """
    names: set[str] = set()
    for prefix, module in instance.named_modules():
        for leaf in getattr(module, "_non_persistent_buffers_set", set()) or set():
            names.add(f"{prefix}.{leaf}" if prefix else leaf)
    return names


def _cast_weights(instance: Any, dtype: Any) -> None:
    """Put the weights in the recorded dtype, leaving derived buffers as computed.

    A derived buffer's precision is part of what the module computes, not of what the
    checkpoint stores. Rotary ``inv_freq`` is the case that found this: computed in
    float32, used to build angles that reach thousands of radians, so rounding it to
    bfloat16 moves the last positions of a 2048-token sample by a tenth of a radian —
    a module that reproduces short samples and quietly drifts on long ones.
    """
    for module in instance.modules():
        derived = getattr(module, "_non_persistent_buffers_set", set()) or set()
        for parameter in module._parameters.values():
            if parameter is not None and parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.is_floating_point() and name not in derived:
                module._buffers[name] = buffer.to(dtype)


def _weights_dtype(weights: dict[str, Any]) -> Any:
    """The floating dtype most of the recorded weights are in, or None."""
    from collections import Counter

    counts: Counter = Counter()
    for tensor in weights.values():
        dtype = getattr(tensor, "dtype", None)
        if dtype is not None and getattr(dtype, "is_floating_point", False):
            counts[dtype] += 1
    return counts.most_common(1)[0][0] if counts else None


def load_weights(instance: Any, weights: dict[str, Any]) -> tuple[int, list[str]]:
    """Copy ``weights`` into ``instance``, matching by name suffix.

    The instance's names are relative to itself (``q_proj.weight``) while the recorded
    names are absolute (``model.layers.3.self_attn.q_proj.weight``), so the longest
    suffix that identifies exactly one recorded tensor wins.

    A derived buffer is installed at the dtype it was recorded in rather than cast into
    the one the class happened to construct: the trace records these as the forward saw
    them, and their precision is part of what the module computes. A rotary ``inv_freq``
    recorded in float32 and cast down to the weights' bfloat16 moves the last positions
    of a long sample by a tenth of a radian.
    """
    import torch

    loaded, missing = 0, []
    for name, tensor in instance.named_parameters():
        candidate = _recorded(weights, name)
        if candidate is None:
            missing.append(name)
            continue
        with torch.no_grad():
            tensor.copy_(candidate.reshape(tensor.shape).to(tensor.dtype))
        loaded += 1

    for prefix, module in instance.named_modules():
        derived = getattr(module, "_non_persistent_buffers_set", set()) or set()
        for leaf, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            name = f"{prefix}.{leaf}" if prefix else leaf
            candidate = _recorded(weights, name)
            if candidate is None:
                # A derived buffer the recording does not carry is not missing: the
                # class computed it at construction, which is where it comes from.
                if leaf not in derived:
                    missing.append(name)
                continue
            if leaf in derived and candidate.dtype != buffer.dtype:
                module._buffers[leaf] = candidate.reshape(buffer.shape).to(buffer.device)
            else:
                with torch.no_grad():
                    buffer.copy_(candidate.reshape(buffer.shape).to(buffer.dtype))
            loaded += 1
    return loaded, missing


def _recorded(weights: dict[str, Any], name: str) -> Any:
    """The recorded tensor for one instance-relative name, or None."""
    # Explicit None: `or` on a tensor asks for its truth value, which raises.
    found = weights.get(name)
    return found if found is not None else _by_suffix(weights, name)


def _by_suffix(weights: dict[str, Any], name: str) -> Any:
    matches = [value for key, value in weights.items()
               if key == name or key.endswith(f".{name}")]
    return matches[0] if len(matches) == 1 else None


def _config_object(module: Any, config: dict[str, Any], config_class: str | None) -> Any:
    """The settings object the class expects, built from the recorded config.

    A modeling file takes its own config class, not a dict. The class is usually
    importable from the same file; failing that the dict is passed through and the
    constructor decides whether it can use it.
    """
    for name in filter(None, (config_class, _guess_config_class(module))):
        candidate = getattr(module, name, None)
        if candidate is None:
            continue
        built = _instantiate_config(candidate, config)
        if built is not None:
            return _restore_private(built, config)
    return _AttributeConfig(config)


def _restore_private(settings: Any, config: dict[str, Any]) -> Any:
    """Put back recorded fields a config class does not take as a keyword.

    ``_attn_implementation`` is the one this exists for: a config class drops it on
    construction, and an attention module built without it runs a different kernel
    than the trace did — one that needs the explicit mask the recording does not carry.
    """
    for key, value in config.items():
        if not key.startswith("_"):
            continue
        try:
            setattr(settings, key, value)
        except Exception:  # a validating property may refuse; it keeps its own value
            continue
    return settings


def _instantiate_config(candidate: Any, config: dict[str, Any]) -> Any:
    """Build one config class from a recorded config dict, or return None.

    A dataclass takes its declared fields as keywords — passing the dict positionally
    would land the whole thing in its first field, which is how a MoE layer ended up
    built with zero experts. Everything else takes keywords or the dict.
    """
    from dataclasses import fields, is_dataclass

    if is_dataclass(candidate):
        declared = {f.name: f for f in fields(candidate)}
        known = {}
        for key, value in config.items():
            field = declared.get(key)
            if field is None:
                continue
            if isinstance(value, list) and "tuple" in str(field.type).lower():
                value = tuple(value)
            known[key] = value
        try:
            return candidate(**known)
        except Exception:
            return None

    for build in (lambda: candidate(**config), lambda: candidate(config), candidate):
        try:
            return build()
        except Exception:
            continue
    return None


def _guess_config_class(module: Any) -> str | None:
    return next((name for name in dir(module) if name.endswith("Config")), None)


class _AttributeConfig:
    """Last resort: a config the class can read attributes off.

    Better than failing outright — many modules only read a handful of fields, and a
    missing one raises where it is used rather than somewhere unrelated.
    """

    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)
        nested = values.get("text_config")
        if isinstance(nested, dict):
            self.__dict__.update(nested)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"the recorded config has no {name!r}; source.py needs it to build this module"
        )


#: Framework primitives take dimensions rather than a config, and the recorded
#: weights say exactly what those dimensions are. Keyed by class name so a vendored
#: copy of the same class is handled too.
SHAPE_CONSTRUCTORS: dict[str, Any] = {
    "Linear": lambda w: {"in_features": w["weight"].shape[1],
                         "out_features": w["weight"].shape[0],
                         "bias": "bias" in w},
    "Embedding": lambda w: {"num_embeddings": w["weight"].shape[0],
                            "embedding_dim": w["weight"].shape[1]},
    "LayerNorm": lambda w: {"normalized_shape": tuple(w["weight"].shape)},
}


def _construct(cls: Any, settings: Any, layer_index: int | None,
               weights: dict[str, Any], config: dict[str, Any]) -> Any:
    """Instantiate a module class the way that class actually wants to be built.

    The model's own classes take its config; a framework primitive takes dimensions,
    which the recorded weights determine; a norm takes its width and an epsilon. Each
    is tried in turn and the failures are reported together, because a module that
    cannot be constructed is a module nobody can optimize.
    """
    import inspect

    try:
        parameters = list(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        parameters = []

    attempts: list[tuple[tuple, dict]] = []
    # A class that names its dimensions can be built straight from the config, which
    # covers the plain building blocks a modeling file defines for itself.
    by_name = _config_kwargs(cls, config, layer_index)
    if by_name is not None:
        attempts.append(((), by_name))
    shaped = SHAPE_CONSTRUCTORS.get(cls.__name__)
    if shaped is not None and "weight" in weights:
        try:
            attempts.append(((), shaped(weights)))
        except Exception:
            pass
    # Only where the parameter is named for it. Passing the index into whatever the
    # second parameter happens to be built a MoE block with zero experts once.
    index_name = next((name for name in ("layer_idx", "layer_index", "layer")
                       if name in parameters), None)
    if layer_index is not None and index_name:
        attempts.append(((settings,), {index_name: layer_index}))
    # A class that wants the config plus a scalar the config does carry but does not
    # pass itself — `Qwen3_5MLP(config, config.intermediate_size)`. Filling the rest by
    # name gets it built; leaving it unbuilt would lose the whole feed-forward module.
    beyond = _named_kwargs(parameters, config, layer_index)
    if beyond:
        attempts.append(((settings,), beyond))
    attempts.append(((settings,), {}))
    # A norm is its width, and often an epsilon the config carries.
    weight = weights.get("weight")
    width = int(weight.shape[0]) if weight is not None and weight.dim() == 1 else None
    if width is not None:
        eps = next((config[key] for key in ("rms_norm_eps", "norm_eps", "layer_norm_eps")
                    if key in config), None)
        if eps is not None:
            attempts.append(((width, eps), {}))
        attempts.append(((width,), {}))

    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            return cls(*args, **kwargs)
        except Exception as exc:
            errors.append(f"{cls.__name__}({_render(args, kwargs)}): {exc}")
    raise LauncherError(
        f"could not construct {cls.__name__} from the recorded config and weights. "
        "Tried:\n  " + "\n  ".join(errors[:5])
    )


def _named_kwargs(parameters: list[str], config: dict[str, Any],
                  layer_index: int | None) -> dict[str, Any]:
    """Every constructor parameter after the first that the config can fill by name."""
    values: dict[str, Any] = {}
    for name in parameters[1:]:
        if name in config:
            values[name] = config[name]
        elif layer_index is not None and name in ("layer_idx", "layer_index", "layer"):
            values[name] = layer_index
    return values


def _config_kwargs(cls: Any, config: dict[str, Any],
                   layer_index: int | None) -> dict[str, Any] | None:
    """Constructor keywords taken from the config by name, when they cover it.

    Returns None unless every parameter without a default is satisfied, so this is
    tried as a real possibility rather than a guess that half-builds something.
    """
    import inspect

    try:
        parameters = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return None

    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name in config:
            kwargs[name] = config[name]
        elif name in ("layer_idx", "layer_index", "layer") and layer_index is not None:
            kwargs[name] = layer_index
        elif parameter.default is inspect.Parameter.empty:
            return None
    return kwargs or None


def _render(args: tuple, kwargs: dict) -> str:
    shown = [type(a).__name__ for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return ", ".join(shown)
