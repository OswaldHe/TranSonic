# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic seed plan.

Groups the repeating stack into runnable modules and splits a layer that is too
large on its own. Groups stay signature-homogeneous so one extracted
implementation serves every layer in a group — which also keeps each kernel
variant of a hybrid stack in its own module.

This is a starting point, not the final answer: the agent planner refines it for
kernel-development convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_partition.planner.graph import ModuleNode, PartitionGraph, TensorRef
from model_partition.sizing import CostModel, ModelInventory, config_get

#: Global tensor name fragments -> module kind.
GLOBAL_KINDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("embed_tokens", "embed_in", "wte", "tok_embeddings"), "embed", "embed"),
    (("lm_head", "output.weight", "embed_out"), "lm_head", "lm_head"),
    (("norm", "ln_f", "final_layernorm"), "norm", "final_norm"),
)


@dataclass
class PlanOptions:
    """Knobs the seed planner honours."""

    seq_len: int = 16384
    max_layers_per_module: int | None = None
    one_layer_per_module: bool = False
    experts_per_group: int | None = None
    #: Never group layers with different structural signatures together.
    homogeneous_groups: bool = True


def classify_global(name: str) -> tuple[str, str] | None:
    """Map a global tensor name to ``(kind, module_id)``."""
    for fragments, kind, module_id in GLOBAL_KINDS:
        if any(fragment in name for fragment in fragments):
            return kind, module_id
    return None


#: Trailing components of a tensor name that belong to the parameter, not the module.
PARAM_SUFFIXES = (
    "weight", "bias", "scale", "weight_scale", "weight_scale_inv",
    "scale_inv", "A_log", "dt_bias", "e_score_correction_bias",
)


def module_path_of(tensor_name: str) -> str:
    """The owning module's path for a tensor, e.g. ``model.embed_tokens``.

    Hooks attach to modules, so a module's ``submodules`` must name modules even
    when the plan was derived from tensor names.
    """
    head, _, tail = tensor_name.rpartition(".")
    return head if head and tail in PARAM_SUFFIXES else tensor_name


def _group_layers(inventory: ModelInventory, per_module: int, options: PlanOptions) -> list[list[int]]:
    """Split layer indices into consecutive, signature-homogeneous groups."""
    indices = [layer.index for layer in inventory.layers]
    if not indices:
        return []
    size = 1 if options.one_layer_per_module else max(per_module, 1)
    signature = {layer.index: layer.signature for layer in inventory.layers}

    groups: list[list[int]] = []
    current: list[int] = []
    for index in indices:
        breaks_run = current and (
            index != current[-1] + 1
            or (options.homogeneous_groups and signature[index] != signature[current[0]])
            or len(current) >= size
        )
        if breaks_run:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


def plan(
    inventory: ModelInventory,
    budget_bytes: int,
    options: PlanOptions | None = None,
    cost: CostModel | None = None,
    model_name: str = "",
    revision: str | None = None,
) -> PartitionGraph:
    """Build a seed partition graph that fits ``budget_bytes`` per module."""
    options = options or PlanOptions()
    dtype_bytes = 2
    cost = cost or CostModel.from_config(inventory.config, dtype_bytes=dtype_bytes,
                                         seq_len=options.seq_len)
    graph = PartitionGraph(
        model=model_name or str(config_get(inventory.config, "model_type", "model")),
        revision=revision,
        budget_bytes=budget_bytes,
        num_layers=len(inventory.layers),
        metadata={"planner": "auto", "seq_len": options.seq_len},
    )

    hidden = inventory.hidden_size
    def declare(name: str, *, kind: str = "activation", shape: list | None = None,
                dtype: str = "bfloat16") -> str:
        graph.tensors[name] = TensorRef(
            name=name, dtype=dtype,
            shape=shape if shape is not None else ["batch", "seq", hidden],
            kind=kind,
        )
        return name

    declare("tokens", kind="activation", shape=["batch", "seq"], dtype="int64")
    graph.entry_tensors = ["tokens"]

    # -- globals ---------------------------------------------------------------
    global_bytes: dict[str, int] = {}
    global_names: dict[str, list[str]] = {}
    for name in inventory.global_tensor_names:
        classified = classify_global(name)
        key = classified[1] if classified else "other_globals"
        global_bytes[key] = global_bytes.get(key, 0) + _tensor_bytes(inventory, name)
        path = module_path_of(name)
        if path not in global_names.setdefault(key, []):
            global_names[key].append(path)

    layer_groups = _group_layers(
        inventory,
        options.max_layers_per_module or cost.max_layers_per_module(
            max((layer.param_bytes for layer in inventory.layers), default=0),
            budget_bytes, options.seq_len,
        ),
        options,
    )

    first_hidden = declare("h.0")
    graph.modules.append(ModuleNode(
        id="embed", kind="embed", inputs=["tokens"], outputs=[first_hidden],
        submodules=global_names.get("embed", []),
        param_bytes=global_bytes.get("embed", 0),
        activation_bytes=cost.activation_bytes(options.seq_len),
        code_signature="embed",
    ))

    # -- the repeating stack ---------------------------------------------------
    cursor = first_hidden
    for group in layer_groups:
        cursor = _add_layer_group(graph, inventory, cost, options, group, cursor, budget_bytes, declare)

    # -- tail ------------------------------------------------------------------
    if "final_norm" in global_bytes:
        normed = declare("h.norm")
        graph.modules.append(ModuleNode(
            id="final_norm", kind="norm", inputs=[cursor], outputs=[normed],
            submodules=global_names["final_norm"],
            param_bytes=global_bytes["final_norm"],
            activation_bytes=cost.activation_bytes(options.seq_len),
            code_signature="final_norm",
        ))
        cursor = normed

    logits = declare("logits", kind="logits", shape=["batch", "seq", inventory.vocab_size])
    graph.modules.append(ModuleNode(
        id="lm_head", kind="lm_head", inputs=[cursor], outputs=[logits],
        submodules=global_names.get("lm_head", global_names.get("embed", [])),
        param_bytes=global_bytes.get("lm_head", global_bytes.get("embed", 0)),
        activation_bytes=_logits_bytes(inventory, options.seq_len, dtype_bytes),
        code_signature="lm_head",
        notes="tied to embedding" if "lm_head" not in global_bytes else "",
    ))
    graph.output_tensors = [logits]

    for key, names in global_names.items():
        if key == "other_globals":
            graph.modules.append(ModuleNode(
                id="other_globals", kind="other", inputs=[], outputs=[],
                submodules=names, param_bytes=global_bytes[key], partitioned=False,
                notes="unclassified global tensors",
            ))

    for subtree, nbytes in sorted(inventory.subtree_bytes.items()):
        graph.modules.append(ModuleNode(
            id=subtree, kind=subtree if subtree in ("vision", "mtp", "engram") else "other",
            param_bytes=nbytes, partitioned=False,
            notes="discovered but out of scope for this run",
        ))

    return graph


def _tensor_bytes(inventory: ModelInventory, name: str) -> int:
    return sum(e.nbytes for e in inventory.index.entries if e.name == name)


#: Name fragments -> role, checked in order. ``experts`` before ``mlp`` so an
#: expert's own ``mlp`` naming does not swallow it.
ROLE_MARKERS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((".experts.",), "experts"),
    (("mlp.gate.weight", "router", "gate.e_score", "correction_bias"), "router"),
    (("self_attn", "attention", ".attn.", "linear_attn"), "attention"),
    (("layernorm", "_norm", ".norm."), "norm"),
    (("mlp", "feed_forward", "ffn"), "mlp"),
)


def _layer_role_bytes(inventory: ModelInventory, index: int) -> dict[str, int]:
    """Parameter bytes of one layer bucketed by role."""
    buckets = dict.fromkeys(("attention", "mlp", "router", "experts", "norm", "other"), 0)
    for entry in inventory.index.entries:
        if entry.layer_index != index or entry.layer_prefix != inventory.stack_prefix:
            continue
        role = next((r for fragments, r in ROLE_MARKERS
                     if any(f in entry.name for f in fragments)), "other")
        buckets[role] += entry.nbytes
    return buckets


def _logits_bytes(inventory: ModelInventory, seq_len: int, dtype_bytes: int) -> int:
    return seq_len * max(inventory.vocab_size, 1) * dtype_bytes


def _add_layer_group(
    graph: PartitionGraph,
    inventory: ModelInventory,
    cost: CostModel,
    options: PlanOptions,
    group: list[int],
    input_tensor: str,
    budget_bytes: int,
    declare,
) -> str:
    """Append one group of layers, splitting further if it cannot fit."""
    profiles = {layer.index: layer for layer in inventory.layers}
    group_bytes = sum(profiles[i].param_bytes for i in group)
    resident = cost.resident_bytes(group_bytes, len(group), options.seq_len)

    if resident > budget_bytes and len(group) > 1:
        cursor = input_tensor
        for index in group:
            cursor = _add_layer_group(graph, inventory, cost, options, [index],
                                      cursor, budget_bytes, declare)
        return cursor

    if resident > budget_bytes:
        return _split_single_layer(graph, inventory, cost, options, group[0],
                                   input_tensor, budget_bytes, declare)

    output = declare(f"h.{group[-1] + 1}")
    first, last = group[0], group[-1]
    graph.modules.append(ModuleNode(
        id=f"layers.{first}" if first == last else f"layers.{first}-{last}",
        kind="decoder_layers",
        inputs=[input_tensor], outputs=[output],
        layer_indices=list(group),
        submodules=[f"{inventory.stack_prefix or ''}{i}" for i in group],
        param_bytes=group_bytes,
        activation_bytes=cost.activation_bytes(options.seq_len),
        kv_bytes=cost.kv_bytes(len(group), options.seq_len),
        code_signature=_signature_key(profiles[first].signature),
    ))
    return output


def _split_single_layer(
    graph: PartitionGraph,
    inventory: ModelInventory,
    cost: CostModel,
    options: PlanOptions,
    index: int,
    input_tensor: str,
    budget_bytes: int,
    declare,
) -> str:
    """Split one oversized layer into attention, router, expert groups, combine."""
    profile = next(layer for layer in inventory.layers if layer.index == index)
    prefix = f"{inventory.stack_prefix or ''}{index}"
    signature = _signature_key(profile.signature)
    roles = _layer_role_bytes(inventory, index)

    attn_out = declare(f"h.{index}.attn")
    graph.modules.append(ModuleNode(
        id=f"layers.{index}.attention", kind="attention",
        inputs=[input_tensor], outputs=[attn_out], layer_indices=[index],
        submodules=[f"{prefix}.self_attn"],
        param_bytes=roles["attention"] + roles["norm"],
        activation_bytes=cost.activation_bytes(options.seq_len),
        kv_bytes=cost.kv_bytes(1, options.seq_len),
        code_signature=f"{signature}:attention",
    ))

    if not profile.has_experts:
        output = declare(f"h.{index + 1}")
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.mlp", kind="mlp",
            inputs=[attn_out], outputs=[output], layer_indices=[index],
            submodules=[f"{prefix}.mlp"],
            param_bytes=roles["mlp"] + roles["other"],
            activation_bytes=cost.activation_bytes(options.seq_len),
            code_signature=f"{signature}:mlp",
        ))
        return output

    route = declare(f"h.{index}.route", kind="routing")
    graph.modules.append(ModuleNode(
        id=f"layers.{index}.router", kind="moe_router",
        inputs=[attn_out], outputs=[route], layer_indices=[index],
        submodules=[f"{prefix}.mlp.gate"],
        param_bytes=roles["router"],
        activation_bytes=cost.activation_bytes(options.seq_len),
        code_signature=f"{signature}:router",
    ))

    per_expert = profile.expert_bytes // max(profile.n_experts, 1)
    group_size = options.experts_per_group or _experts_per_group(
        per_expert, cost, options, budget_bytes,
    )
    partials: list[str] = []
    for start in range(0, profile.n_experts, group_size):
        end = min(start + group_size, profile.n_experts)
        partial = declare(f"h.{index}.moe.{start}")
        partials.append(partial)
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.experts.{start}-{end - 1}", kind="moe_experts",
            inputs=[attn_out, route], outputs=[partial], layer_indices=[index],
            submodules=[f"{prefix}.mlp.experts.{e}" for e in range(start, end)],
            param_bytes=per_expert * (end - start),
            activation_bytes=cost.activation_bytes(options.seq_len),
            expert_range=[start, end - 1],
            code_signature=f"{signature}:experts",
        ))

    output = declare(f"h.{index + 1}")
    graph.modules.append(ModuleNode(
        id=f"layers.{index}.combine", kind="mlp",
        inputs=[attn_out, *partials], outputs=[output], layer_indices=[index],
        activation_bytes=cost.activation_bytes(options.seq_len),
        code_signature=f"{signature}:combine",
        notes="sums expert-group partials into the residual stream",
    ))
    return output


def _experts_per_group(per_expert: int, cost: CostModel, options: PlanOptions, budget: int) -> int:
    """Largest expert count per module that still fits."""
    if per_expert <= 0:
        return 1
    available = budget - cost.activation_bytes(options.seq_len)
    return max(int(available // per_expert), 1)


def _signature_key(signature: str) -> str:
    """Short stable key for a structural signature."""
    import hashlib

    return "sig-" + hashlib.sha256(signature.encode()).hexdigest()[:12]
