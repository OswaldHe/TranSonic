# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a whole model from the partition's own artifacts, and generate from it.

The emulation contract, in two halves. Fill: build the model structure and fill every
parameter from the partition artifacts, never from the original checkpoint, reporting
exactly which parameters were left unsatisfied rather than quietly falling back.
Install: replace every partitioned submodule with a wrapper that calls its group's
``inference.py``, so what generates is the implementations the loop verified and not
the vendor's own code. Assembling the vendor model from dumps and generating from that
would check the reference against itself and print tokens the shipped code never
produced.

The surrounding model still computes masks and rotary embeddings and passes them in,
which is what lets the chain run at a sequence length no recording covers. Module
boundaries are captured on their own forward, then released before generation: leaving
the hooks installed would hold one output per module for every step, and at long
context the logits alone are gigabytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import TraceBundle, load_named_weights
from model_partition.trace import _lookup


class EmulationError(RuntimeError):
    """Raised when a model cannot be assembled from its artifacts, or run from them."""


@dataclass
class FillReport:
    """Outcome of filling a model's parameters from dumps."""

    applied: int = 0
    missing: list[str] = field(default_factory=list)
    #: Parameters no module of the plan claims, which the plan should have covered.
    unclaimed: list[str] = field(default_factory=list)
    #: Parameters the spec deliberately leaves unpartitioned.
    out_of_scope: list[str] = field(default_factory=list)
    by_module: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Every parameter the plan is responsible for came from a dump.

        ``unclaimed`` counts too. A parameter no plan module lists keeps whatever the
        checkpoint put there, so emulation would run on live weights while reporting
        that it ran on dumps — the same hole as a missing dump, reached from the other
        side. Parts the spec puts out of scope are filtered out before they get here.
        """
        return not self.missing and not self.unclaimed

    def summary(self) -> str:
        text = f"{self.applied} parameter(s) filled from dumps"
        if self.unclaimed:
            text += f"; {len(self.unclaimed)} claimed by no module of the plan"
        if self.out_of_scope:
            text += f"; {len(self.out_of_scope)} out of scope"
        if self.missing:
            preview = ", ".join(self.missing[:5])
            text += f"; {len(self.missing)} missing ({preview}{'...' if len(self.missing) > 5 else ''})"
        return text


def fill_from_dumps(
    model: Any,
    bundle: TraceBundle,
    graph: PartitionGraph,
    device: str = "cpu",
    strict: bool = True,
    out_of_scope: tuple[str, ...] = (),
) -> FillReport:
    """Overwrite every partitioned module's parameters with its recorded values.

    ``out_of_scope`` names markers for subtrees the run deliberately does not partition
    — a vision tower, an MTP head — whose parameters are expected to be unclaimed.
    """
    import torch

    report = FillReport()
    claimed: set[str] = set()

    for module in graph.partitioned_modules:
        # Do not skip a module with no dumped weights: if it owns parameters,
        # every one of them is missing, and that must be reported rather than
        # leaving them silently unclaimed.
        dumped = load_named_weights(bundle, module.id, device=device)
        count = 0
        for submodule_name in module.submodules:
            submodule = _lookup(model, submodule_name)
            if submodule is None or not hasattr(submodule, "named_parameters"):
                continue
            items = list(submodule.named_parameters()) + list(submodule.named_buffers())
            for param_name, tensor in items:
                full = f"{submodule_name}.{param_name}" if param_name else submodule_name
                claimed.add(full)
                candidate = dumped.get(full)
                if candidate is None:
                    report.missing.append(full)
                    continue
                with torch.no_grad():
                    tensor.copy_(candidate.reshape(tensor.shape).to(tensor.dtype))
                count += 1
        report.by_module[module.id] = count
        report.applied += count

    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if name in claimed:
            continue
        if any(marker in f".{name}" for marker in out_of_scope):
            report.out_of_scope.append(name)
        else:
            report.unclaimed.append(name)

    if strict and not report.complete:
        detail = []
        if report.missing:
            detail.append(f"{len(report.missing)} parameter(s) have no dumped value "
                          f"({', '.join(report.missing[:6])})")
        if report.unclaimed:
            detail.append(f"{len(report.unclaimed)} parameter(s) belong to no module of "
                          f"the plan and would keep their checkpoint values "
                          f"({', '.join(report.unclaimed[:6])})")
        raise EmulationError("Cannot assemble the model from dumps: " + "; ".join(detail))
    return report


def place_across_devices(model: Any, max_memory: dict) -> tuple[Any, str]:
    """Spread an already-materialized model across GPU and host for the forward.

    Assembling from dumps needs writable parameters, which rules out loading with
    a device map. Dispatching afterwards gets the GPU back: the weights already
    exist, so placement only decides where each layer runs. Returns
    ``(model, input_device)``; on any failure the model is left untouched on its
    current device.
    """
    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except ImportError:
        return model, _first_param_device(model)
    try:
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        placed = dispatch_model(model, device_map=device_map)
    except Exception:
        return model, _first_param_device(model)
    return placed, _first_param_device(placed)


def _first_param_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def capture_boundaries(model: Any, graph: PartitionGraph) -> tuple[list, dict[str, Any]]:
    """Hook every partitioned module to record its output on the *first* forward.

    Generation re-runs the forward on a growing sequence, so only the first pass
    is comparable with a trace taken at the prompt's length.
    """
    sink: dict[str, Any] = {}
    handles = []
    owners: dict[str, list[str]] = {}
    for module in graph.partitioned_modules:
        if module.submodules:
            owners.setdefault(module.submodules[-1], []).append(module.id)

    def make_hook(module_ids: list[str]):
        def hook(_module, _args, output):
            for module_id in module_ids:
                sink.setdefault(module_id, output)
        return hook

    for submodule_name, module_ids in owners.items():
        submodule = _lookup(model, submodule_name)
        if submodule is not None and hasattr(submodule, "register_forward_hook"):
            handles.append(submodule.register_forward_hook(make_hook(module_ids)))
    return handles, sink


def greedy_step(logits: Any, temperature: float = 0.0, seed: int | None = None) -> int:
    """Pick the next token id: argmax, or seeded sampling when temperature > 0."""
    import torch

    last = logits[0, -1] if logits.dim() == 3 else logits[-1]
    if temperature <= 0:
        return int(last.argmax())
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    probs = torch.softmax(last.float().cpu() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


#: Names a causal LM uses for "apply the head to only the last N positions".
#: ``logits_to_keep`` in transformers 5, ``num_logits_to_keep`` before it.
LAST_LOGITS_ARGUMENTS = ("logits_to_keep", "num_logits_to_keep")


def _last_logits_argument(model: Any) -> str | None:
    """The keyword this model takes to skip computing logits it will not be asked for."""
    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    return next((name for name in LAST_LOGITS_ARGUMENTS if name in parameters), None)


def logits_extractor(returns: tuple[str, ...] = ()) -> Any:
    """How to find the logits in whatever this model's forward returns.

    A forward that returns a bare tuple says nothing about which element is which, so
    the spec names them positionally — DeepSeek V4.1's is
    ``(output_ids, logits, main_hidden)``. Treating the tuple itself as the logits fails
    on ``.dim()`` one frame later, which reads as a generation bug rather than as a
    model whose output was never unpacked.

    A model whose output carries ``.logits`` needs no help, and one that returns the
    tensor itself is already the answer.
    """
    index = list(returns).index("logits") if "logits" in tuple(returns or ()) else None

    def extract(output: Any) -> Any:
        if hasattr(output, "logits"):
            return output.logits
        if index is not None and isinstance(output, (tuple, list)):
            if index >= len(output):
                raise ValueError(
                    f"trace.returns names logits at position {index} of "
                    f"{len(tuple(returns))}, and this forward returned {len(output)} "
                    "value(s)"
                )
            return output[index]
        return output

    return extract


def generate(
    model: Any,
    input_ids: Any,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
    eos_token_id: int | None = None,
    logits_of: Any = None,
) -> tuple[list[int], Any]:
    """Greedy/sampled generation by re-running the forward each step.

    No KV cache: every step re-runs the full prefill on the extended sequence.
    Quadratic, but architecture-agnostic — it needs nothing from the model beyond
    ``model(input_ids) -> logits``, which is what keeps emulation independent of
    each architecture's cache implementation.

    Sampling reads only the final row, so the model is asked for only that row. A
    full ``[1, seq, vocab]`` tensor is 8 GB at 16k tokens with a large vocabulary and
    grows by a row every step, which the allocator cannot serve from the previous
    step's cached block — measured on a 1.6 GiB model, reserved memory climbed 10 →
    25 → 44 GiB and then started failing allocations. Keeping one row holds it flat
    at 3 GiB.
    """
    import torch

    from model_partition.trace import forward_no_cache

    extract = logits_of or logits_extractor()
    keep_last = _last_logits_argument(model)
    sequence = input_ids
    produced: list[int] = []
    last = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = (model(sequence, use_cache=False, **{keep_last: 1}) if keep_last
                      else forward_no_cache(model, sequence))
            logits = extract(output)
            last = logits[:, -1:].clone() if logits.dim() == 3 else logits[-1:].clone()
            del output, logits
            token = greedy_step(last, temperature=temperature, seed=seed)
            produced.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
            sequence = torch.cat(
                [sequence, torch.tensor([[token]], device=sequence.device, dtype=sequence.dtype)],
                dim=-1,
            )
    return produced, last


# -- installing the implementations ------------------------------------------
@dataclass
class InstallReport:
    """Which submodules now run the loop's code, and which do not."""

    installed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    #: Cross-module state holders rebound to the model's own object, and how many
    #: implementations each rebinding covered. Empty when the model has no such state.
    shared_state: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return bool(self.installed) and not self.skipped

    def summary(self) -> str:
        text = f"{len(self.installed)} submodule(s) running the extracted implementation"
        if self.shared_state:
            shared = ", ".join(f"{name} across {n + 1}" for name, n in
                               sorted(self.shared_state.items()))
            text += f"; cross-module state shared ({shared})"
        if self.skipped:
            first = ", ".join(list(self.skipped)[:3])
            text += f"; {len(self.skipped)} left as the model's own ({first})"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": list(self.installed),
            "skipped": dict(self.skipped),
            "shared_state": dict(self.shared_state),
            "complete": self.complete,
        }


def install_implementations(
    model: Any,
    graph: Any,
    bundle: Any,
    impl_dirs: dict[str, Any],
    device: str = "cpu",
    lazy: bool = False,
) -> InstallReport:
    """Replace every partitioned submodule with its implementation.

    ``lazy`` builds each implementation when it is first called, from the weights the
    model holds at that moment, and drops it afterwards. That is the only way to install
    into a model too large to be resident: its weights arrive one module at a time, so
    there is no instant at which all 96 implementations could be built, and the instant a
    submodule is called is exactly when its own weights are real.

    One wrapper per submodule rather than one per module: a wrapper installed at a
    submodule is called with exactly the arguments that submodule received, so the
    substitution needs to know nothing about how the surrounding architecture glues
    its pieces together. A group spanning several submodules is checked as a span by
    ``verify_modules`` and ``verify_chain``; here each of its submodules runs the
    group's code for its own part.
    """
    return _install_all(model, graph, bundle, impl_dirs, device, InstallReport(),
                        lazy=lazy)


def _install_all(model: Any, graph: Any, bundle: Any, impl_dirs: dict[str, Any],
                 device: str, report: InstallReport, lazy: bool = False) -> InstallReport:
    import torch

    from model_partition.runtime.module_impl import load_impl
    from model_partition.runtime.module_runner import load_named_weights
    from model_partition.trace import _lookup

    pending: list[tuple[str, Any]] = []
    #: One module's weights at a time, shared across the lazily built
    #: implementations so a module called twice in a forward reads once.
    lazy_weights: dict[str, Any] = {}
    #: Every implementation loaded here, so their cross-module state can be made one
    #: object before anything runs. See :func:`_share_state_holders`.
    loaded: list[Any] = []
    for module in graph.partitioned_modules:
        impl_dir = impl_dirs.get(module.id)
        if impl_dir is None:
            report.skipped[module.id] = "no extracted implementation"
            continue
        if module.functional:
            report.skipped[module.id] = "functional: nothing to replace"
            continue
        try:
            impl = load_impl(impl_dir)
            # The run-wide config, which `build_group` then narrows to the one recorded
            # beside the module. Passing the narrow one here instead loses it as a
            # fallback, and with it the class's own config dataclass — which is what a
            # constructor wanting more than a dict is built from.
            config = bundle.config
            weights = {} if lazy else load_named_weights(bundle, module.id, device=device)
            if not lazy and not weights:
                raise EmulationError("no weights available")
        except Exception as exc:
            report.skipped[module.id] = str(exc)
            continue
        loaded.append(impl)

        for submodule_name in module.submodules:
            original = _lookup(model, submodule_name)
            if original is None:
                report.skipped[submodule_name] = "not present in the model"
                continue
            if lazy:
                pending.append((submodule_name, _wrapper_class()(
                    None, original,
                    factory=_bundle_factory(impl, config, module.id, submodule_name,
                                            bundle, device, lazy_weights))))
                continue
            try:
                built = impl.build(config, weights, device, submodule=submodule_name)
            except Exception as exc:
                report.skipped[submodule_name] = f"build failed: {exc}"
                continue
            if not callable(built):
                report.skipped[submodule_name] = "build_module() returned a non-callable"
                continue
            pending.append((submodule_name, _wrapper_class()(built, original)))

    report.shared_state = _share_state_holders(model, loaded, _state_holders(bundle))
    for submodule_name, wrapper in pending:
        _install(model, submodule_name, wrapper)
        report.installed.append(submodule_name)
    if isinstance(model, torch.nn.Module):
        model.eval()
    return report


def _state_holders(bundle: Any) -> set[str]:
    """Names of the objects a model passes tensors between modules through.

    Taken from the recording, which names them because the trace had to capture them:
    a key like ``shared_attn.compress_kv`` says the holder is ``shared_attn``.
    """
    names: set[str] = set()
    for record in getattr(bundle, "records", None) or []:
        for key in (getattr(record, "state", None) or {}):
            holder, _, leaf = str(key).rpartition(".")
            if holder and leaf:
                names.add(holder.split(".")[0])
    return names


def _share_state_holders(model: Any, impls: list[Any], names: set[str]) -> dict[str, int]:
    """Make every implementation's cross-module state the *same* object as the model's.

    This is the one thing a per-module artifact cannot get right on its own. A model that
    passes tensors between layers through a module-level singleton — DeepSeek's compressed
    KV and top-k indices live on ``shared_attn`` — has that singleton *copied into every
    extracted group's* ``source.py``, because the slice keeps the module-level statements
    its classes reference. Each group is then imported as its own python module, so four
    attention groups mean four ``shared_attn`` objects.

    Layers 2, 8, 14 and 20 publish the compressed KV; the layers after each of them read
    it. Producer and consumer are in different groups whenever their structures differ, so
    the consumer read an object nobody had written and got ``None`` — surfacing as
    ``'NoneType' object is not subscriptable`` from inside otherwise-correct code. Module
    verification never sees this because it restores each call's recorded state instead;
    emulation is where the state is supposed to actually flow.

    The model's own object is canonical: everything the plan does not partition still runs
    the vendor's code against it. Returns how many bindings were made per holder.
    """
    import sys

    if not names or not impls:
        return {}
    vendor = sys.modules.get(getattr(type(model), "__module__", "") or "")
    sources = []
    for impl in impls:
        loader = getattr(getattr(impl, "module", None), "load_implementation", None)
        if loader is None:
            continue
        try:
            sources.append(loader())
        except Exception:
            continue

    bound: dict[str, int] = {}
    for name in sorted(names):
        canonical = getattr(vendor, name, None) if vendor is not None else None
        if canonical is None:
            canonical = next((getattr(s, name) for s in sources if hasattr(s, name)), None)
        if canonical is None:
            continue
        count = 0
        for source in sources:
            if hasattr(source, name) and getattr(source, name) is not canonical:
                setattr(source, name, canonical)
                count += 1
        if vendor is not None and hasattr(vendor, name):
            setattr(vendor, name, canonical)
        bound[name] = count
    return bound


def _module_config(impl_dir: Any) -> dict[str, Any]:
    """The config recorded beside an implementation, or ``{}`` when there is none."""
    from model_partition.runtime.artifact import load_config

    try:
        return load_config(impl_dir) or {}
    except Exception:
        return {}


def _bundle_factory(impl: Any, config: dict[str, Any], module_id: str,
                    submodule_name: str, bundle: Any, device: str, cache: dict):
    """Build this submodule's implementation from the *artifacts*, one module at a time.

    The streamed path holds a model of placeholders, so there are no live weights to
    borrow: the loader materializes a module's tensors in a forward-pre-hook on whatever
    unit it chose to place, and an implementation installed in a submodule's place is not
    that unit. Reading from the recording instead is what every other check already does —
    `verify_modules`, `verify_chain` and a module directory's own `inference.py` — so the
    weights here are the same weights those compared against, and nothing depends on where
    a hook happened to land.

    Onto the host, and one module's worth at a time. A 94.6 GiB n-gram table cannot be
    asked of the card, and holding two of them is more than the host has; `build_group`
    places what fits once it knows the size, which it can only know from real tensors.
    """
    def build() -> Any:
        weights = cache.get(module_id)
        if weights is None:
            # The previous module's first, or the two of them do not fit together.
            cache.clear()
            weights = load_named_weights(bundle, module_id, device="cpu")
            if not weights:
                raise EmulationError(
                    f"{module_id} has no recorded weights, so its implementation cannot be "
                    f"built from the artifacts")
            cache[module_id] = weights
        return impl.build(config, weights, device, submodule=submodule_name)

    return build


#: Defined on first use so importing this module does not require torch.
_WRAPPER: Any = None


def _wrapper_class() -> Any:
    """An ``nn.Module`` standing in for a submodule, calling the loop's code.

    It has to be an ``nn.Module``: the model walks its own tree to move parameters
    between devices, and forward hooks are what boundary capture attaches to.
    """
    global _WRAPPER
    if _WRAPPER is None:
        import torch

        class Implemented(torch.nn.Module):
            def __init__(self, built: Any, original: Any, factory: Any = None):
                super().__init__()
                self._built = built
                self._factory = factory
                #: Kept so the original parameters stay reachable from the model tree
                #: rather than being freed while the wrapper is in their place.
                self.original = original

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                if self._built is not None:
                    return self._built(*args, **kwargs)
                # Built for this call and dropped after it: the weights it is built from
                # are the model's, which are released again once the call returns.
                built = self._factory()
                try:
                    return built(*args, **kwargs)
                finally:
                    del built

            def __getattr__(self, name: str) -> Any:
                """Anything the wrapper does not have, the module it stands in for might.

                A model reads attributes off its own submodules, not just calls them:
                DeepSeek's `Transformer.forward` asks each layer for
                `layer.engram.layer_hash_index` to index the n-gram hashes. A stand-in that
                answered only `forward` would break the very code it is standing in for.
                """
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    # Reached through `__dict__` rather than `self.original`, which would
                    # come back through here and recurse.
                    original = self.__dict__.get("_modules", {}).get("original")
                    if original is not None and name != "original":
                        return getattr(original, name)
                    raise

            def extra_repr(self) -> str:
                return "extracted implementation"

        _WRAPPER = Implemented
    return _WRAPPER


def _install(model: Any, qualified_name: str, replacement: Any) -> None:
    """Put ``replacement`` at ``qualified_name`` in the module tree."""
    from model_partition.trace import _lookup

    parent_name, _, attribute = qualified_name.rpartition(".")
    parent = _lookup(model, parent_name) if parent_name else model
    if parent is None:
        raise EmulationError(f"no parent module for {qualified_name}")
    if attribute.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)
