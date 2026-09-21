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

#: Module path used for a head whose weights are tied to the embedding, so the
#: checkpoint holds no tensor naming it. Reconciliation remaps it if the model
#: calls it something else.
DEFAULT_LM_HEAD = "lm_head"

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
    #: Give attention and the FFN/MoE block of every layer their own modules,
    #: whatever their size. One kernel per module is the point, not capacity.
    split_attention_ffn: bool = False


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

    # A tied head owns no checkpoint tensor, but it is still a distinct module in
    # the model tree and consumes the final hidden state. Naming the conventional
    # path keeps the head's own output as its reference; falling back to the
    # embedding would hook the wrong module and verify the wrong tensor.
    tied_head = "lm_head" not in global_bytes
    logits = declare("logits", kind="logits", shape=["batch", "seq", inventory.vocab_size])
    graph.modules.append(ModuleNode(
        id="lm_head", kind="lm_head", inputs=[cursor], outputs=[logits],
        submodules=global_names.get("lm_head") or [DEFAULT_LM_HEAD],
        param_bytes=global_bytes.get("lm_head", global_bytes.get("embed", 0)),
        activation_bytes=_logits_bytes(inventory, options.seq_len, dtype_bytes),
        code_signature="lm_head",
        notes="weights tied to the embedding" if tied_head else "",
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


#: Component-name fragments that put a layer's direct child on the FFN side of the
#: residual stream even though it also looks like a norm or an attention name
#: (``post_attention_layernorm``). Checked before everything else.
FFN_SIDE_NORMS = ("post_attention", "pre_feedforward", "post_feedforward", "post_mlp", "ffn_norm")
ATTENTION_COMPONENTS = ("attn", "attention")
FFN_COMPONENTS = ("mlp", "ffn", "feed_forward", "moe", "expert")


def component_role(component: str) -> str:
    """Bucket a layer's direct child module by which side of the layer it serves."""
    lowered = component.lower()
    if any(fragment in lowered for fragment in FFN_SIDE_NORMS):
        return "ffn"
    if any(fragment in lowered for fragment in ATTENTION_COMPONENTS):
        return "attention"
    if any(fragment in lowered for fragment in FFN_COMPONENTS):
        return "ffn"
    if "norm" in lowered:
        return "attention"  # the pre-attention norm
    return "other"


@dataclass
class LayerParts:
    """One layer's direct children, bucketed by role.

    Derived from tensor names rather than hard-coded paths, because the same
    architecture family names these differently: ``self_attn``/``mlp`` in a
    transformers checkpoint, ``linear_attn`` on a hybrid Qwen layer, ``attn``/
    ``ffn`` in DeepSeek's.
    """

    attention: list[str] = field(default_factory=list)
    ffn: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    #: Tensors owned by the layer module itself, not by any child. They cannot be
    #: assigned to a submodule, so they are reported rather than partitioned.
    own_params: list[str] = field(default_factory=list)
    nbytes: dict[str, int] = field(default_factory=dict)
    #: Bytes per child module path.
    path_bytes: dict[str, int] = field(default_factory=dict)

    def bytes_for(self, role: str) -> int:
        return self.nbytes.get(role, 0)

    def bytes_of(self, paths: list[str]) -> int:
        return sum(self.path_bytes.get(path, 0) for path in paths)

    def ffn_compute(self) -> str | None:
        """The FFN side's actual computation, as opposed to its normalization."""
        return next((p for p in self.ffn if not _is_norm(p)), None)


def layer_parts(inventory: ModelInventory, index: int) -> LayerParts:
    """Split one layer's tensors across its direct child modules."""
    prefix = f"{inventory.stack_prefix or ''}{index}."
    parts = LayerParts()
    buckets = {"attention": parts.attention, "ffn": parts.ffn, "other": parts.other}
    seen: set[str] = set()
    for entry in inventory.index.entries:
        if entry.layer_index != index or entry.layer_prefix != inventory.stack_prefix:
            continue
        if not entry.name.startswith(prefix):
            continue
        component, _, remainder = entry.name[len(prefix):].partition(".")
        if not remainder:
            parts.own_params.append(entry.name)
            parts.nbytes["own"] = parts.bytes_for("own") + entry.nbytes
            continue
        role = component_role(component)
        parts.nbytes[role] = parts.bytes_for(role) + entry.nbytes
        path = prefix + component
        parts.path_bytes[path] = parts.path_bytes.get(path, 0) + entry.nbytes
        if path not in seen:
            seen.add(path)
            buckets[role].append(path)
    # A module's submodules are run in listed order, and tensor order in a
    # checkpoint says nothing about execution order. Norms first is right for the
    # pre-norm family every current LLM belongs to; if a model normalizes after
    # instead, verification says so rather than quietly computing the wrong thing.
    for paths in buckets.values():
        paths.sort(key=_is_norm, reverse=True)
    return parts


def _is_norm(path: str) -> bool:
    return "norm" in path.rsplit(".", 1)[-1].lower()


def moe_parts(inventory: ModelInventory, ffn_path: str) -> tuple[list[str], dict[int, str], list[str], dict[str, int]]:
    """Router, per-expert and shared submodule paths beneath one FFN module.

    Returns ``(router_paths, {expert_index: path}, shared_paths, bytes_by_part)``.
    """
    prefix = f"{ffn_path}."
    router: list[str] = []
    shared: list[str] = []
    experts: dict[int, str] = {}
    nbytes: dict[str, int] = {}

    def account(key: str, value: int) -> None:
        nbytes[key] = nbytes.get(key, 0) + value

    for entry in inventory.index.entries:
        if not entry.name.startswith(prefix):
            continue
        component, _, remainder = entry.name[len(prefix):].partition(".")
        path = prefix + component
        head = remainder.partition(".")[0]
        if "expert" in component and head.isdigit():
            experts[int(head)] = f"{path}.{head}"
            account("experts", entry.nbytes)
        elif "expert" in component:
            if path not in shared:
                shared.append(path)
            account("shared", entry.nbytes)
        else:
            if path not in router:
                router.append(path)
            account("router", entry.nbytes)
    return router, experts, shared, nbytes


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

    if options.split_attention_ffn or (resident > budget_bytes and len(group) > 1):
        cursor = input_tensor
        for index in group:
            cursor = (_split_layer(graph, inventory, cost, options, index, cursor,
                                   budget_bytes, declare)
                      if options.split_attention_ffn
                      else _add_layer_group(graph, inventory, cost, options, [index],
                                            cursor, budget_bytes, declare))
        return cursor

    if resident > budget_bytes:
        return _split_layer(graph, inventory, cost, options, group[0],
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


def _split_layer(
    graph: PartitionGraph,
    inventory: ModelInventory,
    cost: CostModel,
    options: PlanOptions,
    index: int,
    input_tensor: str,
    budget_bytes: int,
    declare,
) -> str:
    """Split one layer into attention, anything in between, and the FFN/MoE side.

    The FFN side stays one module while it fits the budget; when it does not it
    becomes router, expert groups, and the combine step.
    """
    profile = next(layer for layer in inventory.layers if layer.index == index)
    signature = _signature_key(profile.signature)
    parts = layer_parts(inventory, index)
    activation = cost.activation_bytes(options.seq_len)

    cursor = input_tensor
    if parts.attention:
        cursor = declare(f"h.{index}.attn")
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.attention", kind="attention",
            inputs=[input_tensor], outputs=[cursor], layer_indices=[index],
            submodules=list(parts.attention),
            param_bytes=parts.bytes_for("attention"),
            activation_bytes=activation,
            kv_bytes=cost.kv_bytes(1, options.seq_len),
            code_signature=f"{signature}:attention",
        ))

    for path in parts.other:
        component = path.rsplit(".", 1)[-1]
        output = declare(f"h.{index}.{component}")
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.{component}", kind="other",
            inputs=[cursor], outputs=[output], layer_indices=[index],
            submodules=[path], param_bytes=parts.path_bytes.get(path, 0),
            activation_bytes=activation,
            code_signature=f"{signature}:{component}",
        ))
        cursor = output

    if not parts.ffn:
        return cursor

    # Experts are what make an FFN too large to hold, and routing is what makes it
    # divisible. A dense FFN over budget stays one module and the graph says so:
    # splitting it would mean cutting a single matmul chain apart, which is the
    # operator's call (raise the budget) or the agent's, not the seed planner's.
    ffn_bytes = parts.bytes_for("ffn")
    fits = cost.resident_bytes(ffn_bytes, 1, options.seq_len) <= budget_bytes
    if fits or not profile.has_experts:
        output = declare(f"h.{index + 1}")
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.ffn", kind="mlp",
            inputs=[cursor], outputs=[output], layer_indices=[index],
            submodules=list(parts.ffn), param_bytes=ffn_bytes,
            activation_bytes=activation,
            code_signature=f"{signature}:ffn",
            notes=("MoE block: routes and runs experts" if profile.has_experts
                   else "" if fits else "dense FFN over the per-module budget"),
        ))
        return output

    return _split_ffn(graph, inventory, cost, options, index, parts, cursor,
                      budget_bytes, declare, signature)


def _split_ffn(
    graph: PartitionGraph,
    inventory: ModelInventory,
    cost: CostModel,
    options: PlanOptions,
    index: int,
    parts: LayerParts,
    input_tensor: str,
    budget_bytes: int,
    declare,
    signature: str,
) -> str:
    """Break an over-budget FFN into router, expert groups, and a combine step."""
    activation = cost.activation_bytes(options.seq_len)
    ffn_path = parts.ffn_compute()
    if ffn_path is None:
        raise ValueError(f"layer {index} has no FFN computation to split")
    router, experts, shared, nbytes = moe_parts(inventory, ffn_path)
    # The FFN side's normalization runs before the gate, so it leads the router
    # module rather than becoming a module of its own.
    norms = [path for path in parts.ffn if _is_norm(path)]

    route = declare(f"h.{index}.route", kind="routing")
    graph.modules.append(ModuleNode(
        id=f"layers.{index}.router", kind="moe_router",
        inputs=[input_tensor], outputs=[route], layer_indices=[index],
        submodules=norms + router,
        param_bytes=parts.bytes_of(norms) + nbytes.get("router", 0),
        activation_bytes=activation,
        code_signature=f"{signature}:router",
    ))

    # A shared expert is a dense feed-forward every token passes through: separate
    # work from the routed experts, and separate work from the router.
    partials: list[str] = []
    if shared:
        shared_out = declare(f"h.{index}.shared")
        partials.append(shared_out)
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.shared_experts", kind="mlp",
            inputs=[input_tensor], outputs=[shared_out], layer_indices=[index],
            submodules=shared, param_bytes=nbytes.get("shared", 0),
            activation_bytes=activation,
            code_signature=f"{signature}:shared_experts",
        ))

    ordered = [experts[i] for i in sorted(experts)]
    per_expert = nbytes.get("experts", 0) // max(len(ordered), 1)
    group_size = options.experts_per_group or _experts_per_group(
        per_expert, cost, options, budget_bytes,
    )
    for start in range(0, len(ordered), group_size):
        end = min(start + group_size, len(ordered))
        partial = declare(f"h.{index}.moe.{start}")
        partials.append(partial)
        graph.modules.append(ModuleNode(
            id=f"layers.{index}.experts.{start}-{end - 1}", kind="moe_experts",
            inputs=[input_tensor, route], outputs=[partial], layer_indices=[index],
            submodules=ordered[start:end],
            param_bytes=per_expert * (end - start),
            activation_bytes=activation,
            expert_range=[start, end - 1],
            code_signature=f"{signature}:experts",
            # Experts in a group are alternatives, not a pipeline: each sees the
            # routed activation for its own tokens.
            composition="parallel",
        ))

    output = declare(f"h.{index + 1}")
    graph.modules.append(ModuleNode(
        id=f"layers.{index}.combine", kind="mlp",
        inputs=[input_tensor, *partials], outputs=[output], layer_indices=[index],
        activation_bytes=activation,
        code_signature=f"{signature}:combine",
        notes=("functional: sums expert-group partials into the residual stream. "
               "No submodule of its own, so no traced reference"),
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
