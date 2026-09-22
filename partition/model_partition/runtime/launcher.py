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

This module imports nothing else from the harness beyond its two small neighbours
(:mod:`~model_partition.runtime.compat` and :mod:`~model_partition.hardware`), which is
what lets :func:`~model_partition.extract.vendor_runtime` copy it into a run so a
module's ``inference.py`` builds from the artifact alone. Keep it that way: the names
below live here rather than in :mod:`~model_partition.extract` because that module
renders templates and a published artifact must not need a template engine to run.
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

#: The model's own package, copied into the run beside ``modules/``.
VENDOR_DIR = "vendor"

#: This module and its neighbours, copied into the run so a module directory builds
#: without the harness installed.
RUNTIME_DIR = "runtime"

#: Config key extraction records the model's activation dtype under.
COMPUTE_DTYPE_KEY = "_compute_dtype"

#: The recorded call and weight map a module's ``inference.py`` reads, written beside it.
CALLS_FILENAME = "calls.json"

#: Imported source per directory. Executing a modeling file is not cheap and every
#: module of a group asks for the same one.
_SOURCE_CACHE: dict[str, Any] = {}

#: Total weight bytes above which a module is constructed on the meta device and handed
#: its recorded tensors by reference instead of copying into freshly allocated ones.
#: Constructing for real allocates every parameter, and copying then holds the module
#: twice — which DeepSeek V4.1's 94.4 GiB n-gram tables do not allow on any machine that
#: holds them once. Below it construction stays real, because a class computes its
#: derived buffers there and on the meta device those would come out empty.
ASSIGN_THRESHOLD = 32 << 30


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
    #: Submodules whose weights the device would not hold, left on the host with their
    #: inputs and outputs moved across the boundary.
    on_host: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = f"{self.class_name} from source.py, {self.weights_loaded} weight(s) loaded"
        if self.on_host:
            text += f"; {', '.join(self.on_host)} on the host, too large for the device"
        if self.missing:
            text += f"; {len(self.missing)} parameter(s) left unset: {', '.join(self.missing[:4])}"
        return text


def load_source(directory: str | Path, source_module: str | None = None,
                device: Any = None) -> Any:
    """Import the ``source.py`` beside a module directory.

    Loaded under the name its classes originally had, so the relative imports in a
    vendored modeling file still resolve. Falls back to a private name when no
    original is recorded.

    Any compatibility patch of the run is applied to it, the same way the loader applies
    them to the model. ``source.py`` is a copy of the vendor's file and carries the
    vendor's kernels, so without this a module would ask this card for a kernel written
    for another one — and the reference it is checked against was produced with the patch
    in place, so running without it would be checking a different computation.
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
    _vendor_on_path(Path(directory))
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
    _apply_compat(module, Path(directory), device)
    _SOURCE_CACHE[key] = module
    return module


def _vendor_on_path(directory: Path) -> Path | None:
    """Put the run's copy of the model's package where ``source.py``'s imports find it.

    ``source.py`` is a slice of one file of that package and keeps its imports, so it
    needs its siblings — ``from kernel import fp8_gemm``. They travel with the run under
    ``vendor/``, which is what makes a module directory something you can take away and
    build from: nothing outside the run has to be present.
    """
    import sys

    for root in (directory.parent.parent, directory.parent, directory):
        candidate = root / VENDOR_DIR
        if candidate.is_dir():
            entry = str(candidate.resolve())
            if entry not in sys.path:
                sys.path.insert(0, entry)
            return candidate
    return None


def _apply_compat(module: Any, directory: Path, device: Any) -> None:
    """Apply the run's compatibility patches to a freshly imported ``source.py``.

    The patches live at the run root, beside ``modules/``, and are part of the artifacts —
    so a module directory copied elsewhere with ``compat/`` beside it gets them too.
    """
    from model_partition.runtime.compat import apply_patches, patch_paths

    for root in (directory.parent.parent, directory.parent, directory):
        paths = patch_paths(root)
        if paths:
            apply_patches(module, paths, _device_info(device))
            return


def _device_info(device: Any) -> Any:
    """What a patch needs to know about the card it is adapting to.

    A patch decides for itself whether this hardware needs it — the one that replaces a
    kernel asking for more shared memory than the card allows applies nowhere else — so
    it is handed the card rather than a device string.
    """
    if device is None or hasattr(device, "shared_memory_per_block"):
        return device
    if not str(device).startswith("cuda"):
        return None
    from model_partition.hardware import detect_gpus

    found = detect_gpus()
    return found[0] if found else None


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
    """Run a group's submodules in order, each given the arguments it declares.

    The group's extra arguments belong to whichever submodules take them: a
    normalization takes the flowing tensor alone while the attention after it also takes
    a position and a shared state. So each submodule gets the flowing value plus as much
    of the rest as its own signature has room for.
    """
    import torch

    def run(*args: Any, **kwargs: Any) -> Any:
        output = None
        flowing = args[0] if args else None
        rest = args[1:]
        with torch.no_grad():
            for index, target in enumerate(built):
                if index:
                    flowing = output[0] if isinstance(output, tuple) else output
                extras = _positional(target, rest)
                output = target(flowing, *extras,
                                **_accepted(target, kwargs, filled=1 + len(extras)))
        return output

    return run


def _parameters(target: Any) -> Any:
    """The signature of what this submodule is called as, or None."""
    import inspect

    try:
        return inspect.signature(getattr(target, "forward", target)).parameters
    except (TypeError, ValueError):
        return None


def _positional(target: Any, rest: tuple) -> tuple:
    """As many of the group's extra positional arguments as this submodule takes."""
    import inspect

    parameters = _parameters(target)
    if parameters is None:
        return rest
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values()):
        return rest
    return rest[:max(len(parameters) - 1, 0)]


def _accepted(target: Any, kwargs: dict[str, Any], filled: int = 1) -> dict[str, Any]:
    """The keywords this submodule declares and has not already been given.

    ``filled`` is how many of its parameters arrived positionally — the flowing tensor
    and any extras — so a name supplied both ways is not passed twice.
    """
    import inspect

    parameters = _parameters(target)
    if parameters is None:
        return dict(kwargs)
    names = list(parameters)
    taken = set(names[:filled])
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {k: v for k, v in kwargs.items() if k not in taken}
    return {k: v for k, v in kwargs.items() if k in set(names[filled:])}


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

    module = (source if source is not None
              else load_source(directory, source_module, device=device))
    cls = _resolve_class(module, class_name)

    settings = _config_object(module, config, config_class)
    by_reference = weights_bytes(weights) > ASSIGN_THRESHOLD
    # The recorded weights decide the module's precision, and the way to apply that is
    # torch's default dtype — the same thing a vendor's entry script sets before building
    # the model, and for the same two reasons. Casting the parameters afterwards instead
    # would overwrite a dtype the class asked for deliberately: DeepSeek builds its
    # confidence head and its hyper-connection scalars in float32 inside a bfloat16
    # model, and a module cast wholesale to bfloat16 meets its own float32 activations
    # and will not multiply. And it is not only construction that reads the default —
    # DeepSeek's fp8 GEMM allocates its output buffer with it, and a float32 buffer is
    # rejected by the kernel — so it is left set rather than restored, which is what
    # `generate.py` does for the whole process.
    dtype = _compute_dtype(config) or _weights_dtype(weights)
    if dtype is not None:
        torch.set_default_dtype(dtype)
    # And the default device, for the same reason and with the same reach: DeepSeek's
    # attention builds its window's top-k indices with a bare `torch.arange`, so they
    # land wherever the default points and its kernel requires them beside the KV it
    # gathers. `generate.py` sets this too.
    if device and not str(device).startswith("meta"):
        torch.set_default_device(device)
    if by_reference:
        with torch.device("meta"):
            instance = _construct(cls, settings, layer_index, weights, config)
    else:
        instance = _construct(cls, settings, layer_index, weights, config)
    loaded, missing = load_weights(instance, weights, derived_optional=not by_reference)
    on_host = place(instance, device)
    instance.eval()
    torch.set_grad_enabled(False)
    return Launched(callable=instance, class_name=class_name,
                    weights_loaded=loaded, missing=missing, on_host=on_host)


def _compute_dtype(config: dict[str, Any]) -> Any:
    """The dtype the model's activations flow in, as extraction recorded it."""
    import torch

    name = config.get(COMPUTE_DTYPE_KEY)
    if not name:
        return None
    short = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    name = str(name).removeprefix("torch.")
    dtype = getattr(torch, short.get(name, name), None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _weights_dtype(weights: dict[str, Any]) -> Any:
    """The dtype to construct the module in, from its weights: a fallback.

    Used for a module directory written before the compute dtype was recorded beside it.
    Counting tensors is a guess, and a poor one on a quantized model — DeepSeek's MoE
    votes float32 two to one, on two bias vectors.

    A quantized checkpoint's own dtype is not a candidate — ``torch.set_default_dtype``
    takes float16, bfloat16, float32 and float64 and nothing else, and rightly: fp8 is
    what a bfloat16 module multiplies with, not what it is built in. So a DeepSeek
    attention module is built in the bfloat16 its norms and biases are stored in, and its
    ``Linear`` weights keep the fp8 the class asks for.
    """
    from collections import Counter

    import torch

    accepted = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    counts: Counter = Counter()
    for tensor in weights.values():
        dtype = getattr(tensor, "dtype", None)
        if dtype in accepted:
            counts[dtype] += 1
    return counts.most_common(1)[0][0] if counts else None


def load_weights(instance: Any, weights: dict[str, Any],
                 derived_optional: bool = True) -> tuple[int, list[str]]:
    """Install ``weights`` into ``instance``, matching by name suffix.

    The instance's names are relative to itself (``q_proj.weight``) while the recorded
    names are absolute (``model.layers.3.self_attn.q_proj.weight``), so the longest
    suffix that identifies exactly one recorded tensor wins.

    A recorded tensor of the same dtype and shape becomes the parameter rather than
    being copied into it. Copying costs a second allocation of the whole module — which
    a 94.4 GiB n-gram table does not have room for — and some dtypes cannot be copied at
    all: fp4-packed expert weights have no ``copy_``. A tensor that does not match is
    reshaped and cast, which is the only case that needs one.

    A derived buffer is installed at the dtype it was recorded in rather than cast into
    the one the class happened to construct: the trace records these as the forward saw
    them, and their precision is part of what the module computes. A rotary ``inv_freq``
    recorded in float32 and cast down to the weights' bfloat16 moves the last positions
    of a long sample by a tenth of a radian.

    ``derived_optional`` is false for an instance built on the meta device, where the
    class computed nothing and a missing buffer cannot be recovered.
    """
    import torch

    loaded, missing = 0, []
    for prefix, module in instance.named_modules():
        derived = getattr(module, "_non_persistent_buffers_set", set()) or set()
        # Installing a tensor replaces the object the class put there, and a quantized
        # `Linear` hangs its block scale off the weight so its kernel can reach it.
        # Recorded before the replacements and put back after.
        aliases = aliases_of(module)
        for leaf, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            name = f"{prefix}.{leaf}" if prefix else leaf
            candidate = _recorded(weights, name)
            if candidate is None:
                missing.append(name)
                continue
            module._parameters[leaf] = torch.nn.Parameter(
                _fitted(candidate, parameter, _scale_for(weights, name)), requires_grad=False)
            loaded += 1
        for leaf, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            name = f"{prefix}.{leaf}" if prefix else leaf
            candidate = _recorded(weights, name)
            if candidate is None:
                # A derived buffer the recording does not carry is not missing: the
                # class computed it at construction, which is where it comes from.
                if leaf not in derived or not derived_optional:
                    missing.append(name)
                continue
            module._buffers[leaf] = (candidate.reshape(buffer.shape).to(buffer.device)
                                     if leaf in derived else _fitted(candidate, buffer))
            loaded += 1
        rebind(module, aliases)
    return loaded, missing


def aliases_of(module: Any) -> list[tuple[str, str, str]]:
    """Attributes on one tensor that point at another of the same module's tensors.

    A vendor's quantized ``Linear`` hangs its block scale off the weight so its kernel
    can reach it — ``self.weight.scale = self.scale`` — and replacing either tensor
    breaks the link unless it is put back.
    """
    found: list[tuple[str, str, str]] = []
    names = list(module._parameters) + list(module._buffers)
    for leaf in names:
        holder = tensor_of(module, leaf)
        if holder is None:
            continue
        for attribute, value in list(vars(holder).items()):
            for other in names:
                if other != leaf and tensor_of(module, other) is value:
                    found.append((leaf, attribute, other))
    return found


def rebind(module: Any, aliases: list[tuple[str, str, str]]) -> None:
    """Put back the links :func:`aliases_of` recorded."""
    for leaf, attribute, other in aliases:
        holder, target = tensor_of(module, leaf), tensor_of(module, other)
        if holder is not None and target is not None:
            try:
                setattr(holder, attribute, target)
            except Exception:  # pragma: no cover - a read-only attribute
                continue


def tensor_of(module: Any, leaf: str) -> Any:
    """The parameter or buffer a module holds under ``leaf``, or None."""
    if leaf in module._parameters:
        return module._parameters[leaf]
    return module._buffers.get(leaf)


def device_of(value: Any) -> Any:
    """The device of the first tensor in an argument tree."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, (tuple, list)):
        for item in value:
            found = device_of(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        return device_of(tuple(value.values()))
    return None


def moved(value: Any, device: Any) -> Any:
    """The same argument tree with every tensor on ``device``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(moved(item, device) for item in value)
    if isinstance(value, list):
        return [moved(item, device) for item in value]
    if isinstance(value, dict):
        return {key: moved(item, device) for key, item in value.items()}
    return value


def place(instance: Any, device: str) -> list[str]:
    """Move the module to ``device``, leaving behind what will not fit.

    Returns the submodules left on the host. An n-gram table is 94.4 GiB and no card
    here holds it, while the fp8 GEMM beside it in the same module runs nowhere else —
    so the table stays where it is and its lookup is wrapped to take its indices and
    hand its rows back across the boundary. Placing the module as a whole has no answer
    to that: one half does not fit and the other half does not run.
    """
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        instance.to(device)
        return []
    free, _total = torch.cuda.mem_get_info(torch.device(device))
    budget = int(free * 0.8)

    left: list[str] = []
    for name, module in instance.named_modules():
        own = [(leaf, t) for leaf, t in
               list(module._parameters.items()) + list(module._buffers.items())
               if t is not None]
        held = sum(t.numel() * t.element_size() for _leaf, t in own)
        if held > budget:
            left.append(name or "(root)")
            _host_boundary(module, device)
            continue
        aliases = aliases_of(module)
        for leaf, tensor in own:
            set_tensor(module, leaf, tensor.to(device))
        rebind(module, aliases)
    return left


def _host_boundary(module: Any, device: Any) -> None:
    """Take this module's inputs to where its weights are, and send its output back."""

    def to_host(_module: Any, args: tuple, kwargs: dict):
        return moved(args, "cpu"), moved(kwargs, "cpu")

    def to_device(_module: Any, _args: tuple, _kwargs: dict, output: Any):
        return moved(output, device)

    module.register_forward_pre_hook(to_host, with_kwargs=True)
    module.register_forward_hook(to_device, with_kwargs=True)


def set_tensor(module: Any, leaf: str, value: Any) -> None:
    """Install ``value`` as the module's ``leaf``, as a parameter or a buffer."""
    import torch

    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    else:
        module._buffers[leaf] = value


def dequantize(weight: Any, scale: Any, dtype: Any) -> Any:
    """Expand a block-scaled weight to ``dtype``, one scale per block.

    The arithmetic a vendor conversion script does for the weights its kernels want in
    bfloat16: each block of the weight is multiplied by its own scale.
    """
    if weight.dim() != 2 or scale.dim() != 2:
        return weight.to(dtype)
    out_block = weight.shape[0] // scale.shape[0]
    in_block = weight.shape[1] // scale.shape[1]
    if out_block < 1 or in_block < 1:
        return weight.to(dtype)
    expanded = (weight.unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block)).float()
                * scale[:, None, :, None].float())
    return expanded.flatten(2, 3).flatten(0, 1).to(dtype)


def _fitted(candidate: Any, existing: Any, scale: Any = None) -> Any:
    """``candidate`` as the parameter it replaces: by reference when it already is.

    Three ways a recorded tensor can differ from the parameter it fills, and they are not
    interchangeable:

    - same width, different dtype: the same bytes read differently. A checkpoint stores a
      pair of fp4 values as ``int8`` and the class declares ``float4_e2m1fn_x2``.
      Reinterpreting is both right and the only option — that dtype has no conversion and
      no ``copy_``.
    - narrower than the parameter, with a block scale beside it: the class wants this one
      dequantized, which is what the vendor's conversion script does for the weights its
      kernels take in bfloat16. DeepSeek's ``wo_a`` is the case, and casting it without
      its scale leaves an attention output four thousand times too large — wrong by a
      factor, which reads as a plausible-looking 0.94 cosine rather than as a crash.
    - anything else: cast.
    """
    import torch

    wide = (torch.bfloat16, torch.float16, torch.float32)
    if candidate.dtype != existing.dtype and candidate.element_size() == existing.element_size():
        candidate = candidate.view(existing.dtype)
    elif (scale is not None and existing.dtype in wide
          and candidate.element_size() < existing.element_size()):
        candidate = dequantize(candidate, scale, existing.dtype)
    if tuple(candidate.shape) != tuple(existing.shape):
        candidate = candidate.reshape(existing.shape)
    return candidate if candidate.dtype == existing.dtype else candidate.to(existing.dtype)


def _scale_for(weights: dict[str, Any], name: str) -> Any:
    """The block scale recorded beside a quantized weight, if there is one."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf != "weight":
        return None
    return _recorded(weights, f"{name[:-len(leaf)]}scale")


def weights_bytes(weights: dict[str, Any]) -> int:
    """Bytes the recorded weights occupy."""
    total = 0
    for tensor in weights.values():
        try:
            total += tensor.numel() * tensor.element_size()
        except AttributeError:  # pragma: no cover - a non-tensor in the dump
            continue
    return total


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


#: Parameter names that mean "which repetition of the stack is this".
INDEX_PARAMETERS = ("layer_idx", "layer_index", "layer_id", "layer")

#: Constructor parameter -> the names a config might hold its value under. A class names
#: its own argument for what it does with it and a config names it for what it configures,
#: so the two rarely agree on an epsilon or a width. Getting this wrong builds a module
#: that runs, which is worse than one that does not.
CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "eps": ("norm_eps", "rms_norm_eps", "layer_norm_eps", "layernorm_epsilon", "epsilon"),
    "dim": ("hidden_size", "d_model", "n_embd"),
    "hidden_size": ("dim", "d_model", "n_embd"),
    "vocab_size": ("padded_vocab_size",),
}

#: Classmethods that build a helper object from the model's config alone.
FROM_CONFIG = ("from_args", "from_config")


def _config_dataclass(cls: Any, parameters: list[str], config: dict[str, Any]) -> Any:
    """The model's own config object, from the recorded fields plus its own defaults.

    A vendor class annotated ``args: ModelArgs`` wants that dataclass and not a mapping:
    it carries defaults the published ``config.json`` never mentions — a maximum batch
    size, a sliding window — and the class reads them straight off it. Rebuilding it
    from its own definition is the only way to get those; a wrapper around the recorded
    fields raises on the first one the file left to its default.
    """
    import dataclasses
    import inspect
    import sys
    import typing

    try:
        hints = typing.get_type_hints(cls.__init__, vars(sys.modules.get(cls.__module__)))
    except Exception:
        return None
    for name in parameters:
        annotation = hints.get(name)
        if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation):
            fields = {f.name for f in dataclasses.fields(annotation)}
            try:
                return annotation(**{k: v for k, v in config.items() if k in fields})
            except Exception:
                return None
    return None


def _built_from_config(cls: Any, names: list[str], settings: Any) -> dict[str, Any]:
    """Arguments a class needs that are themselves built from the config.

    ``Engram(args, layer_id, layout)`` is the case: ``layout`` is an ``EngramLayout``,
    which the model builds once with ``EngramLayout.from_args(args)``. Nothing but that
    class knows how, and the annotation is what says which class it is.
    """
    import inspect
    import sys
    import typing

    try:
        hints = typing.get_type_hints(cls.__init__, vars(sys.modules.get(cls.__module__)))
    except Exception:
        return {}
    built: dict[str, Any] = {}
    for name in names:
        annotation = hints.get(name)
        factory = next((getattr(annotation, f) for f in FROM_CONFIG
                        if inspect.isclass(annotation) and hasattr(annotation, f)), None)
        if factory is None:
            return {}
        try:
            built[name] = factory(settings)
        except Exception:
            return {}
    return built


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
    index_name = next((name for name in INDEX_PARAMETERS if name in parameters), None)
    # The class's own config object where it has one, so its defaults are in play.
    own = _config_dataclass(cls, parameters, config)
    if layer_index is not None and index_name:
        attempts.append(((settings,), {index_name: layer_index}))
        for holder in ([own, settings] if own is not None else [settings]):
            if parameters[0] == index_name:
                # The index first and the config second: DeepSeek's whole modeling file
                # is written `Attention(layer_id, args)`, so the config cannot lead.
                attempts.append(((layer_index, holder), {}))
                rest = _built_from_config(cls, parameters[2:], holder)
                if rest:
                    attempts.append(((layer_index, holder), rest))
            elif len(parameters) > 1 and parameters[1] == index_name:
                # The config first, then the index, then whatever else it wants —
                # `Engram(args, layer_id, layout)`.
                attempts.append(((holder, layer_index), {}))
                rest = _built_from_config(cls, parameters[2:], holder)
                if rest:
                    attempts.append(((holder, layer_index), rest))
    if own is not None:
        attempts.append(((own,), {}))
    # A class that wants the config plus a scalar the config does carry but does not
    # pass itself — `Qwen3_5MLP(config, config.intermediate_size)`. Filling the rest by
    # name gets it built; leaving it unbuilt would lose the whole feed-forward module.
    beyond = _named_kwargs(parameters, config, layer_index)
    if beyond:
        attempts.append(((settings,), beyond))
    attempts.append(((settings,), {}))
    # A class taking one number takes a width, and the weights are what say which: a
    # head whose width is `dim + rank` is in no config field, but it is the in-features
    # of the projection the recording holds.
    required = [name for name, p in inspect.signature(cls).parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD)] if parameters else []
    if len(required) == 1:
        for tensor in weights.values():
            if getattr(tensor, "dim", lambda: 0)() == 2:
                attempts.append(((int(tensor.shape[1]),), {}))
                attempts.append(((int(tensor.shape[0]),), {}))
                break

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
        alias = next((key for key in (name, *CONFIG_ALIASES.get(name, ())) if key in config),
                     None)
        if alias is not None:
            values[name] = config[alias]
        elif layer_index is not None and name in INDEX_PARAMETERS:
            values[name] = layer_index
    return values


def _config_kwargs(cls: Any, config: dict[str, Any],
                   layer_index: int | None) -> dict[str, Any] | None:
    """Constructor keywords taken from the config by name, when they cover it.

    Returns None unless every parameter without a default is satisfied, so this is
    tried as a real possibility rather than a guess that half-builds something.

    A parameter the config names differently is taken through :data:`CONFIG_ALIASES`,
    and a parameter left at its class default is the thing to be most careful about: it
    looks like a successful build. DeepSeek's config says ``norm_eps: 1e-20`` and
    ``RMSNorm.__init__`` defaults ``eps`` to ``1e-6``, so every normalization came out
    quietly wrong — by 0.4% on its own output, and by enough after the attention it feeds
    to fail 87 checks while still reading as cosine 0.9999.
    """
    import inspect

    try:
        parameters = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return None

    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        alias = next((key for key in (name, *CONFIG_ALIASES.get(name, ())) if key in config),
                     None)
        if alias is not None:
            kwargs[name] = config[alias]
        elif name in INDEX_PARAMETERS and layer_index is not None:
            kwargs[name] = layer_index
        elif parameter.default is inspect.Parameter.empty:
            return None
    return kwargs or None


def _render(args: tuple, kwargs: dict) -> str:
    shown = [type(a).__name__ for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return ", ".join(shown)
