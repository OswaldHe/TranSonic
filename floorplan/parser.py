# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""System YAML in, simulation objects out — deterministically.

This is the half of the simulation platform that is *not* an agent's judgement. Every
number in the system YAML becomes a field on an object here by a fixed rule, with no
inference and no defaulting: a value the YAML does not give is ``None`` and stays ``None``,
and asking for it raises. That is what makes the hardware model reviewable — you can read
``trn2-16device.yaml`` and know exactly what the simulator believes.

The agent's contribution sits strictly on top: per-module cost models (``sim/modules/``)
and the prose constraints (``sim/constraints.py``), both of which consume these objects and
neither of which may edit them.

Two mechanisms keep the YAML honest:

``inherit:``    a section may borrow another system's, so ``trn2-1device`` cannot drift
                from ``trn2-16device`` on facts about the silicon they share.
``probed.yaml`` an overlay written by ``floorplan probe``, merged last. Probed values are
                the only ones that change between runs, and they are confined to a
                separate file so a diff of the datasheet is a diff of the datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Where the bundled system descriptions live.
SYSTEMS_DIR = Path(__file__).resolve().parent / "systems"

#: The overlay ``floorplan probe`` writes. Absent until a probe has run.
PROBED_NAME = "probed.yaml"

#: Sections that may carry ``inherit: <system>``.
INHERITABLE = ("device", "efficiency", "constraints_text")


class SystemError_(ValueError):
    """The system YAML is missing or does not say what the simulator needs."""


def as_float(value: Any) -> float | None:
    """Coerce a YAML scalar to a float, or None if it is absent.

    This exists because of a YAML 1.1 trap that is invisible until something downstream is
    mysteriously missing: ``6.67e14`` is **not** a float to PyYAML. The 1.1 float grammar
    requires a decimal point and a *signed* exponent, so ``6.67e14`` parses as the string
    ``"6.67e14"`` while ``6.67e+14`` parses as a number. Every compute peak in the system YAML
    was written the first way, and an ``isinstance(value, float)`` filter dropped all of them
    — leaving `device.compute` empty and every matmul un-costable, with no error until a cost
    model asked for a rate.

    Coercing here rather than demanding the signed form in the YAML, because the unsigned form
    is what anyone editing the file will naturally write.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _require_float(block: dict[str, Any], key: str, where: str) -> float:
    """``as_float`` with a clear error instead of a silent None."""
    value = as_float(block.get(key))
    if value is None:
        raise SystemError_(f"{where}.{key} is missing or not a number ({block.get(key)!r})")
    return value


# ---------------------------------------------------------------------------------------
# Loading and resolution
# ---------------------------------------------------------------------------------------
def system_path(name: str, systems_dir: Path | None = None) -> Path:
    """Resolve a target name or path to a system YAML."""
    directory = systems_dir or SYSTEMS_DIR
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    for suffix in (".yaml", ".yml"):
        bundled = directory / f"{name}{suffix}"
        if bundled.exists():
            return bundled
    available = sorted(
        p.stem for p in directory.glob("*.yaml") if p.name != PROBED_NAME
    )
    raise SystemError_(
        f"no system named '{name}'. Available: {', '.join(available) or 'none'}"
    )


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Overlay wins, recursing into mappings. Lists are replaced, not concatenated.

    Lists are replaced because the only list the overlay touches is ``tiers``, and a
    half-merged tier list — some entries probed, some not, order unspecified — would be
    far harder to reason about than a wholesale replacement.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return overlay


def load_system(
    name: str,
    systems_dir: Path | None = None,
    apply_probes: bool = True,
) -> dict[str, Any]:
    """Read a system YAML, resolve ``inherit:``, and merge the probe overlay.

    Returns the resolved mapping. ``Hardware.from_system`` turns it into objects; callers
    that only need to read a field (the checker, the schema's ``validate_against``) use
    this directly so there is one resolution path and not two.
    """
    directory = systems_dir or SYSTEMS_DIR
    path = system_path(name, directory)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SystemError_(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemError_(f"{path} did not parse as a mapping")

    for section in INHERITABLE:
        block = data.get(section)
        if isinstance(block, dict) and "inherit" in block:
            parent_name = str(block["inherit"])
            if parent_name == data.get("name"):
                raise SystemError_(f"{path}: section '{section}' inherits from itself")
            parent = load_system(parent_name, directory, apply_probes=False)
            inherited = parent.get(section)
            if inherited is None:
                raise SystemError_(
                    f"{path}: section '{section}' inherits from '{parent_name}', "
                    f"which has no '{section}'"
                )
            extra = {k: v for k, v in block.items() if k != "inherit"}
            data[section] = _deep_merge(inherited, extra) if extra else inherited
        elif isinstance(block, str):
            # `constraints_text: {inherit: x}` is a mapping; a bare string is prose.
            continue

    if apply_probes:
        probed = directory / PROBED_NAME
        if probed.exists():
            try:
                overlay = yaml.safe_load(probed.read_text()) or {}
            except yaml.YAMLError as exc:
                raise SystemError_(f"{probed} is not valid YAML: {exc}") from exc
            for_target = overlay.get(str(data.get("name")), {})
            shared = overlay.get("shared", {})
            data = _deep_merge(_deep_merge(data, shared), for_target)

    return data


# ---------------------------------------------------------------------------------------
# The objects the simulator runs against
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Engine:
    """One execution engine on a logical NeuronCore."""

    name: str
    role: str
    bandwidth_bytes_per_s: float | None = None
    stationary_max: int | None = None
    moving_max: int | None = None


@dataclass(frozen=True)
class Tier:
    """One level of the memory hierarchy."""

    name: str
    scope: str
    capacity_bytes: int | None
    bandwidth_bytes_per_s: float | None
    latency_us: float | None
    random_read_iops: float | None = None
    min_transfer_granularity_bytes: int = 1
    devices: int = 1
    inference_path: bool = True
    reachable_from: str = ""
    #: Tiers this one can only be reached *through*. NVMe names host_dram, which is what
    #: makes an NVMe read cost two hops rather than one.
    via: tuple[str, ...] = ()

    def require(self, field_name: str) -> float:
        value = getattr(self, field_name)
        if value is None:
            raise SystemError_(
                f"tier '{self.name}' has no {field_name}. It is marked probe_required — "
                f"run `autohelix floorplan probe` before simulating, or supply the value "
                f"in systems/probed.yaml"
            )
        return float(value)

    def transfer_seconds(self, nbytes: int, accesses: int = 1) -> float:
        """Time to move ``nbytes`` in ``accesses`` transfers, whichever bound dominates.

        Three bounds, and the max of them wins: bandwidth, IOPS, and per-access latency.
        A bandwidth-only model is the specific mistake this method exists to prevent — an
        Engram lookup is tens of 256-byte random reads, which is nothing in bytes and a
        great deal in IOPS and latency.
        """
        if nbytes <= 0 or accesses <= 0:
            return 0.0
        granular = max(nbytes, accesses * self.min_transfer_granularity_bytes)
        by_bandwidth = granular / self.require("bandwidth_bytes_per_s")
        by_latency = accesses * (self.latency_us or 0.0) * 1e-6
        by_iops = (accesses / self.random_read_iops) if self.random_read_iops else 0.0
        return max(by_bandwidth, by_latency, by_iops)


@dataclass(frozen=True)
class LogicalNC:
    """A logical NeuronCore: the placement grain, and its private HBM bank."""

    device: int
    index: int
    physical_cores: int
    sbuf_bytes: int
    psum_bytes: int
    hbm_bank_bytes: int
    hbm_bank_bandwidth_bytes_per_s: float
    engines: dict[str, Engine]

    @property
    def address(self) -> str:
        return f"d{self.device}.l{self.index}"


@dataclass(frozen=True)
class Device:
    """One Trainium2 chip and its position on the torus."""

    index: int
    hbm_bytes: int
    hbm_bandwidth_bytes_per_s: float
    dma_bandwidth_bytes_per_s: float
    sbuf_bytes: int
    cc_cores: int
    coord: tuple[int, int]
    logical_ncs: tuple[LogicalNC, ...]


@dataclass
class Hardware:
    """The resolved platform: what the simulator asks about the machine.

    Agent-written cost models receive one of these. They read it; they never mutate it,
    and the invariant suite checks that a simulation leaves it unchanged.
    """

    name: str
    lnc: int
    devices: tuple[Device, ...]
    tiers: dict[str, Tier]
    topology_kind: str
    topology_shape: tuple[int, int]
    topology_wrap: bool
    compute: dict[str, float]
    efficiency: dict[str, float | None]
    intra_device_bandwidth_bytes_per_s: float | None
    intra_device_latency_us: float | None
    inter_device_bandwidth_bytes_per_s: float | None
    inter_device_hop_latency_us: float | None
    constraints_text: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- addressing -------------------------------------------------------
    def logical_nc(self, address: str) -> LogicalNC:
        """Look up a unit by its ``d<i>.l<j>`` address, ignoring a physical suffix."""
        from floorplan.schema import Address

        parsed = Address.parse(address) if isinstance(address, str) else address
        if not 0 <= parsed.device < len(self.devices):
            raise SystemError_(f"{parsed}: no device {parsed.device} on '{self.name}'")
        device = self.devices[parsed.device]
        if not 0 <= parsed.logical_nc < len(device.logical_ncs):
            raise SystemError_(f"{parsed}: no logical NC {parsed.logical_nc} on d{parsed.device}")
        return device.logical_ncs[parsed.logical_nc]

    def all_units(self) -> list[LogicalNC]:
        return [nc for device in self.devices for nc in device.logical_ncs]

    def unit_count(self) -> int:
        return sum(len(device.logical_ncs) for device in self.devices)

    # -- topology ---------------------------------------------------------
    def hops(self, device_a: int, device_b: int) -> int:
        """Manhattan distance on the torus, with wrap-around when the YAML says torus.

        On a 4x4 torus the farthest device is 4 hops away, not 6: each axis wraps, so a
        distance of 3 along an axis is really 1 the other way.
        """
        if device_a == device_b:
            return 0
        if self.topology_kind == "single":
            raise SystemError_(
                f"'{self.name}' has one device; there is no hop between d{device_a} "
                f"and d{device_b}"
            )
        rows, cols = self.topology_shape
        ax, ay = self.devices[device_a].coord
        bx, by = self.devices[device_b].coord
        dx, dy = abs(ax - bx), abs(ay - by)
        if self.topology_wrap:
            dx, dy = min(dx, rows - dx), min(dy, cols - dy)
        return dx + dy

    def require_efficiency(self, key: str) -> float:
        value = self.efficiency.get(key)
        if value is None:
            raise SystemError_(
                f"efficiency.{key} is not set. It is marked probe_required — run "
                f"`autohelix floorplan probe`, or set it in systems/probed.yaml. The "
                f"simulator will not substitute peak, because that would make compute free"
            )
        return float(value)

    def require_link(self, kind: str) -> tuple[float, float]:
        """``(bandwidth_bytes_per_s, latency_us)`` for ``intra_device`` or ``inter_device``."""
        if kind == "intra_device":
            bandwidth, latency = (
                self.intra_device_bandwidth_bytes_per_s, self.intra_device_latency_us,
            )
        elif kind == "inter_device":
            bandwidth, latency = (
                self.inter_device_bandwidth_bytes_per_s, self.inter_device_hop_latency_us,
            )
        else:
            raise SystemError_(f"unknown link class '{kind}'")
        if bandwidth is None or latency is None:
            raise SystemError_(
                f"links.{kind} is incomplete (bandwidth={bandwidth}, latency={latency}). "
                f"Probe it or set it in systems/probed.yaml"
            )
        return float(bandwidth), float(latency)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_system(cls, system: dict[str, Any]) -> "Hardware":
        """Build the object graph. One fixed rule per field; nothing inferred."""
        for key in ("name", "hierarchy", "device", "tiers"):
            if key not in system:
                raise SystemError_(f"system YAML has no '{key}'")

        hierarchy = system["hierarchy"]
        device_count = int(hierarchy["device"]["count"])
        logical_count = int(hierarchy["logical_nc"]["count"])
        physical_count = int(hierarchy["physical_nc"]["count"])

        spec = system["device"]
        hbm = spec["hbm"]
        sbuf = spec["sbuf"]
        psum = spec["psum"]
        hbm_bytes = int(hbm["capacity_bytes"])
        bank_bytes = int(hbm.get("bank_capacity_bytes") or hbm_bytes // logical_count)
        bank_bandwidth = _require_float(hbm, "bandwidth_bytes_per_s", "device.hbm") / logical_count
        sbuf_total = int(sbuf["capacity_bytes"])
        sbuf_per_nc = int(sbuf.get("per_logical_nc_bytes") or sbuf_total // logical_count)
        psum_bytes = int(psum["bank_bytes"]) * int(psum["banks"])

        engines: dict[str, Engine] = {}
        for name, entry in (spec.get("engines") or {}).items():
            engines[name] = Engine(
                name=name,
                role=str(entry.get("role", "")),
                bandwidth_bytes_per_s=as_float(entry.get("bandwidth_bytes_per_s")),
                stationary_max=(
                    None if entry.get("stationary_max") is None
                    else int(entry["stationary_max"])
                ),
                moving_max=(
                    None if entry.get("moving_max") is None else int(entry["moving_max"])
                ),
            )

        topology = system.get("topology") or {}
        kind = str(topology.get("kind", "single"))
        shape_raw = topology.get("shape") or [1, 1]
        shape = (int(shape_raw[0]), int(shape_raw[1]))
        wrap = bool(topology.get("wrap", False))
        if kind != "single" and shape[0] * shape[1] != device_count:
            raise SystemError_(
                f"topology shape {shape} holds {shape[0] * shape[1]} devices but "
                f"hierarchy.device.count is {device_count}"
            )

        devices: list[Device] = []
        for index in range(device_count):
            coord = (index // shape[1], index % shape[1]) if kind != "single" else (0, 0)
            ncs = tuple(
                LogicalNC(
                    device=index,
                    index=j,
                    physical_cores=physical_count,
                    sbuf_bytes=sbuf_per_nc,
                    psum_bytes=psum_bytes,
                    hbm_bank_bytes=bank_bytes,
                    hbm_bank_bandwidth_bytes_per_s=bank_bandwidth,
                    engines=engines,
                )
                for j in range(logical_count)
            )
            devices.append(Device(
                index=index,
                hbm_bytes=hbm_bytes,
                hbm_bandwidth_bytes_per_s=_require_float(
                    hbm, "bandwidth_bytes_per_s", "device.hbm"),
                dma_bandwidth_bytes_per_s=(as_float(
                    (spec.get("engines") or {}).get("dma", {}).get("bandwidth_bytes_per_s")
                ) or 0.0),
                sbuf_bytes=sbuf_total,
                cc_cores=int((spec.get("collective") or {}).get("cc_cores", 0)),
                coord=coord,
                logical_ncs=ncs,
            ))

        tiers: dict[str, Tier] = {}
        for entry in system["tiers"]:
            name = str(entry["name"])
            # An explicit field, not inferred from the `reachable_from` prose. The first
            # version of this matched the substring "through host_dram", which the YAML
            # writes as "*through* host_dram" — so NVMe was silently charged one leg instead
            # of two, making it look twice as cheap as it is. Exactly the mis-costing that
            # would skew the Engram residency decision, and invisible from the outside.
            via = tuple(str(hop) for hop in (entry.get("via") or ()))
            reachable = str(entry.get("reachable_from", ""))
            tiers[name] = Tier(
                name=name,
                scope=str(entry.get("scope", "instance")),
                capacity_bytes=(
                    None if entry.get("capacity_bytes") is None
                    else int(entry["capacity_bytes"])
                ),
                bandwidth_bytes_per_s=(
                    as_float(entry.get("bandwidth_bytes_per_s"))
                ),
                latency_us=as_float(entry.get("latency_us")),
                random_read_iops=as_float(entry.get("random_read_iops")),
                min_transfer_granularity_bytes=int(
                    entry.get("min_transfer_granularity_bytes", 1)
                ),
                devices=int(entry.get("devices", 1)),
                inference_path=bool(entry.get("inference_path", True)),
                reachable_from=reachable,
                via=via,
            )

        compute: dict[str, float] = {}
        for key, value in (spec.get("compute") or {}).items():
            coerced = as_float(value)
            if coerced is not None:
                compute[key] = coerced

        raw_efficiency = system.get("efficiency") or {}
        # Only the keys that are coefficients: the block also carries `source`,
        # `confidence`, `probed_at` and explanatory notes, none of which are numbers.
        efficiency: dict[str, float | None] = {}
        for key, value in raw_efficiency.items():
            if key in {"source", "confidence", "inherit", "probed_at"} or key.endswith("note"):
                continue
            efficiency[key] = as_float(value)

        links = system.get("links") or {}
        intra = links.get("intra_device") or {}
        inter = links.get("inter_device") or {}

        constraints = system.get("constraints_text")
        if isinstance(constraints, dict):
            constraints = str(constraints.get("text", ""))

        return cls(
            name=str(system["name"]),
            lnc=int(system.get("lnc", 2)),
            devices=tuple(devices),
            tiers=tiers,
            topology_kind=kind,
            topology_shape=shape,
            topology_wrap=wrap,
            compute=compute,
            efficiency=efficiency,
            intra_device_bandwidth_bytes_per_s=as_float(intra.get("bandwidth_bytes_per_s")),
            intra_device_latency_us=as_float(intra.get("latency_us")),
            inter_device_bandwidth_bytes_per_s=as_float(inter.get("bandwidth_bytes_per_s")),
            inter_device_hop_latency_us=as_float(inter.get("per_hop_latency_us")),
            constraints_text=str(constraints or ""),
            raw=system,
        )

    @classmethod
    def load(cls, name: str, systems_dir: Path | None = None) -> "Hardware":
        return cls.from_system(load_system(name, systems_dir))

    # -- reporting --------------------------------------------------------
    def unresolved(self) -> list[str]:
        """Every field still awaiting a probe, as dotted paths.

        The simulator refuses to run while this is non-empty, and ``floorplan probe``
        reports it as its to-do list. One place computes it so the two cannot disagree
        about what "ready" means.
        """
        missing: list[str] = []
        for key, value in sorted(self.efficiency.items()):
            if value is None:
                missing.append(f"efficiency.{key}")
        if self.intra_device_bandwidth_bytes_per_s is None:
            missing.append("links.intra_device.bandwidth_bytes_per_s")
        if self.intra_device_latency_us is None:
            missing.append("links.intra_device.latency_us")
        if len(self.devices) > 1:
            if self.inter_device_bandwidth_bytes_per_s is None:
                missing.append("links.inter_device.bandwidth_bytes_per_s")
            if self.inter_device_hop_latency_us is None:
                missing.append("links.inter_device.per_hop_latency_us")
        for name in sorted(self.tiers):
            tier = self.tiers[name]
            if not tier.inference_path:
                continue
            if tier.bandwidth_bytes_per_s is None:
                missing.append(f"tiers.{name}.bandwidth_bytes_per_s")
            if tier.latency_us is None:
                missing.append(f"tiers.{name}.latency_us")
            if name == "nvme" and tier.random_read_iops is None:
                missing.append(f"tiers.{name}.random_read_iops")
        return missing

    def describe(self) -> str:
        """What a floorplan author needs to know about this machine, in a few lines."""
        units = self.unit_count()
        bank = self.devices[0].logical_ncs[0].hbm_bank_bytes / 2 ** 30
        total = sum(d.hbm_bytes for d in self.devices) / 2 ** 30
        topology = (
            "single device" if self.topology_kind == "single"
            else f"{self.topology_shape[0]}x{self.topology_shape[1]} "
                 f"{'torus' if self.topology_wrap else 'mesh'}"
        )
        return (
            f"{self.name}: {len(self.devices)} device(s), {units} logical NeuronCore(s) "
            f"at LNC={self.lnc}, {bank:.0f} GiB per bank, {total:.0f} GiB total, {topology}"
        )


def format_bytes(nbytes: float) -> str:
    """Human-readable byte count (base-1024), matching partition/hardware.py's."""
    step = 1024.0
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < step or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= step
    return f"{value:.1f} TiB"


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division, for shard sizes that do not divide evenly."""
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return -(-numerator // denominator)


