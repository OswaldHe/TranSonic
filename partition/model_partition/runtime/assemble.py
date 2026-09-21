# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put the extracted implementations into the model, so emulation runs them.

The implementations are the deliverable. Assembling the vendor model from dumps and
generating from that would check the reference against itself and print tokens the
shipped code never produced — so every partitioned submodule is replaced by a
wrapper that calls its group's ``inference.py``. Generation then proceeds normally:
the surrounding model still computes masks and rotary embeddings and passes them in,
which is what lets the chain run at a sequence length no recording covers.

Two details make this cheap rather than a second copy of the model:

* Wrappers are built before any of them is installed, so each holds a direct
  reference to the original submodule rather than looking it up later and finding
  its own wrapper.
* The baseline's structure cache is seeded with the model being assembled, so an
  implementation that delegates to the baseline reuses these parameters instead of
  instantiating the architecture again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class AssembleError(RuntimeError):
    """Raised when the implementations cannot be installed."""


@dataclass
class InstallReport:
    """Which submodules now run the loop's code, and which do not."""

    installed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return bool(self.installed) and not self.skipped

    def summary(self) -> str:
        text = f"{len(self.installed)} submodule(s) running the extracted implementation"
        if self.skipped:
            first = ", ".join(list(self.skipped)[:3])
            text += f"; {len(self.skipped)} left as the model's own ({first})"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": list(self.installed),
            "skipped": dict(self.skipped),
            "complete": self.complete,
        }


def install_implementations(
    model: Any,
    graph: Any,
    bundle: Any,
    impl_dirs: dict[str, Any],
    run_dir: str | Path,
    device: str = "cpu",
) -> InstallReport:
    """Replace every partitioned submodule with its implementation.

    One wrapper per submodule rather than one per module: a wrapper installed at a
    submodule is called with exactly the arguments that submodule received, so the
    substitution needs to know nothing about how the surrounding architecture glues
    its pieces together. A group spanning several submodules is checked as a span by
    ``verify_modules`` and ``verify_chain``; here each of its submodules runs the
    group's code for its own part.
    """
    from model_partition.runtime import baseline

    report = InstallReport()
    # An implementation that delegates to the baseline should reuse these very
    # parameters. Without this the baseline would instantiate the architecture a
    # second time, which for a checkpoint that only just fits is fatal. Dropped again
    # before returning: from here on this model has wrappers in it, and a later stage
    # finding that in the cache would look up a submodule and get a wrapper.
    baseline.seed_structure(run_dir, device, model)
    try:
        return _install_all(model, graph, bundle, impl_dirs, device, report)
    finally:
        baseline.forget_structure(run_dir, device)


def _install_all(model: Any, graph: Any, bundle: Any, impl_dirs: dict[str, Any],
                 device: str, report: InstallReport) -> InstallReport:
    import torch

    from model_partition.runtime.module_impl import load_impl
    from model_partition.runtime.module_runner import load_named_weights
    from model_partition.trace import _lookup

    pending: list[tuple[str, Any]] = []
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
            weights = load_named_weights(bundle, module.id, device=device)
            if not weights:
                raise AssembleError("no weights available")
        except Exception as exc:
            report.skipped[module.id] = str(exc)
            continue

        for submodule_name in module.submodules:
            original = _lookup(model, submodule_name)
            if original is None:
                report.skipped[submodule_name] = "not present in the model"
                continue
            try:
                built = impl.build(bundle.config, weights, device, submodule=submodule_name)
            except Exception as exc:
                report.skipped[submodule_name] = f"build failed: {exc}"
                continue
            if not callable(built):
                report.skipped[submodule_name] = "build_module() returned a non-callable"
                continue
            pending.append((submodule_name, _wrapper_class()(built, original)))

    for submodule_name, wrapper in pending:
        _install(model, submodule_name, wrapper)
        report.installed.append(submodule_name)
    if isinstance(model, torch.nn.Module):
        model.eval()
    return report


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
            def __init__(self, built: Any, original: Any):
                super().__init__()
                self._built = built
                #: Kept so the original parameters stay reachable from the model tree
                #: rather than being freed while the wrapper is in their place.
                self.original = original

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self._built(*args, **kwargs)

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
        raise AssembleError(f"no parent module for {qualified_name}")
    if attribute.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)
