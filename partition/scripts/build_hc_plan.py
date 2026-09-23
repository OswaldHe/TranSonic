"""Rewrite a DeepSeek V4.1 partition plan so the mHC mix has modules of its own.

The plan this produces is the one the reviewer asked for. Before it, `hc_mixes`,
`hc_pre` and `hc_post` were methods on `Block` with their coefficients held as
parameters on the block itself, so a hook-based trace had nothing to attach to: the
arithmetic sitting on every sublayer boundary belonged to no module, the 258 `hc_*`
parameters sat in `metadata.layer_owned_params` owned by nothing, and `verify_chain`
could carry 6 of 255 boundaries because the plan's residual stream was the wrong rank.

`compat/hyper_connections.py` makes them four real submodules per layer. This declares
the modules for them and rewires the stream through them:

    hs.L --[hc_attn_in]--> x.L.attn --[attention]--> h.L.attn
         --[hc_attn_out]-> hs.L.mid --[hc_ffn_in]--> x.L.ffn
         --[ffn]---------> h.L+1    --[hc_ffn_out]-> hs.L+1

Every edge there is one module's first output feeding the next module's first argument,
which is exactly what `verify_chain` can carry. `pre_mix`, `post` and `comb` are *not*
declared as edges on purpose: they are second and later arguments, and the chain takes
those from the recording for the same reason it takes masks and rotary embeddings from it
— they are inputs to the computation rather than products of the stream.

    python scripts/build_hc_plan.py <run-dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_partition import yamlio  # noqa: E402

#: One implementation per structural signature, and the four kinds here are four
#: structures however many layers they serve.
SIGNATURES = {
    "in": "sig-hyperconnect:sublayer_in",
    "out": "sig-hyperconnect:sublayer_out",
    "expand": "sig-hyperconnect:expand",
    "collapse": "sig-hyperconnect:collapse",
}

#: `hc_*_in` owns the layer's coefficient projection, base and scale: fp32,
#: [(2 + hc) * hc, hc * dim] plus two small vectors. `hc_*_out` owns nothing.
IN_PARAM_BYTES = 24 * 4 * 5120 * 4 + 24 * 4 + 3 * 4


def declare(tensors: list[dict], name: str, rank4: bool = False) -> None:
    """Add a tensor to the plan's declarations if it is not already there."""
    if any(t.get("name") == name for t in tensors):
        return
    shape = ["batch", "seq", 4, 5120] if rank4 else ["batch", "seq", 5120]
    tensors.append({"name": name, "dtype": "bfloat16", "shape": shape,
                    "kind": "activation"})


def node(module_id: str, submodule: str, inputs: list[str], outputs: list[str],
         signature: str, layer: int | None, param_bytes: int = 0,
         activation_bytes: int = 503316480) -> dict:
    entry = {
        "id": module_id,
        "kind": "other",
        "inputs": list(inputs),
        "outputs": list(outputs),
        "param_bytes": param_bytes,
        "activation_bytes": activation_bytes,
        "partitioned": True,
        "submodules": [submodule],
        "code_signature": signature,
    }
    if layer is not None:
        entry["layer_indices"] = [layer]
    return entry


def rewrite(plan: dict) -> dict:
    """Insert the mHC modules and rewire the residual stream to rank 4."""
    modules = {m["id"]: m for m in plan["modules"]}
    order = [m["id"] for m in plan["modules"]]
    tensors = plan.setdefault("tensors", [])
    added: list[dict] = []

    # Layer prefixes that have both halves of a Block: `layers.7`, `mtp.1`.
    prefixes = [mid.rsplit(".", 1)[0] for mid in order if mid.endswith(".attention")
                and f"{mid.rsplit('.', 1)[0]}.ffn" in modules]

    engram_of = {mid.rsplit(".", 1)[0]: mid for mid in order if mid.endswith(".engram")}

    for prefix in prefixes:
        attention, ffn = modules[f"{prefix}.attention"], modules[f"{prefix}.ffn"]
        layer = (attention.get("layer_indices") or [None])[0]
        backbone = prefix.startswith("layers.")
        index = int(prefix.split(".")[1])

        # Where this layer's rank-4 stream comes from, and where it goes. Only the
        # backbone is a stream: the draft stack is driven from its own entry points, so
        # its blocks get the four modules but keep the edges the plan already had.
        if backbone:
            stream_in = f"hs.{index}"
            stream_out = f"hs.{index + 1}"
            declare(tensors, stream_in, rank4=True)
            declare(tensors, stream_out, rank4=True)
            # The n-gram memory writes into the stream before the block reads it.
            if prefix in engram_of:
                engram = modules[engram_of[prefix]]
                engram["inputs"] = [stream_in]
                engram["outputs"] = [f"{stream_in}.engram"]
                declare(tensors, f"{stream_in}.engram", rank4=True)
                stream_in = f"{stream_in}.engram"
        else:
            # The draft stack keeps the edges it had, so the exit gets a name of its own
            # rather than a second producer for the ffn's output. Nothing consumes it:
            # `mtp.*` is driven from `forward_embed` and `forward_head`, so those edges
            # were never the stream in the first place, and saying so beats implying they
            # were.
            stream_in = attention["inputs"][0]
            stream_out = f"hs.{prefix}.out"
            declare(tensors, stream_out, rank4=True)

        attn_out = attention["outputs"][0]
        ffn_out = ffn["outputs"][0]
        mid, x_attn, x_ffn = f"hs.{prefix}.mid", f"x.{prefix}.attn", f"x.{prefix}.ffn"
        declare(tensors, mid, rank4=True)
        declare(tensors, x_attn)
        declare(tensors, x_ffn)

        added += [
            node(f"{prefix}.hc_attn_in", f"{prefix}.hc_attn_in", [stream_in], [x_attn],
                 SIGNATURES["in"], layer, IN_PARAM_BYTES),
            node(f"{prefix}.hc_attn_out", f"{prefix}.hc_attn_out", [attn_out], [mid],
                 SIGNATURES["out"], layer, activation_bytes=4 * 503316480),
            node(f"{prefix}.hc_ffn_in", f"{prefix}.hc_ffn_in", [mid], [x_ffn],
                 SIGNATURES["in"], layer, IN_PARAM_BYTES),
            node(f"{prefix}.hc_ffn_out", f"{prefix}.hc_ffn_out", [ffn_out], [stream_out],
                 SIGNATURES["out"], layer, activation_bytes=4 * 503316480),
        ]
        # The sublayers now read what the mix hands them, not the stream itself.
        attention["inputs"] = [x_attn]
        ffn["inputs"] = [x_ffn]

    # The two ends of the backbone stream. `embed` still produces `h.0` and `final_norm`
    # still consumes a rank-3 tensor, so only the join moves.
    first = next((m for m in plan["modules"] if m["id"] == "embed"), None)
    last_norm = next((m for m in plan["modules"] if m["id"] == "final_norm"), None)
    n_layers = int(plan.get("num_layers") or 0)
    if first is not None:
        declare(tensors, "hs.0", rank4=True)
        added.append(node("hc_expand", "hc_expand", [first["outputs"][0]], ["hs.0"],
                          SIGNATURES["expand"], None,
                          activation_bytes=4 * 503316480))
    if last_norm is not None and n_layers:
        collapsed = f"h.{n_layers}.collapsed"
        declare(tensors, collapsed)
        added.append(node("hc_collapse", "hc_collapse", [f"hs.{n_layers}"], [collapsed],
                          SIGNATURES["collapse"], None))
        last_norm["inputs"] = [collapsed]

    plan["modules"].extend(added)
    # The coefficients belong to the `hc_*_in` modules now, so they are no longer
    # parameters the plan merely knows about and nothing owns.
    metadata = plan.setdefault("metadata", {})
    owned = [p for p in (metadata.get("layer_owned_params") or []) if "hc_" not in p]
    metadata["layer_owned_params"] = owned
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = args.run_dir / "plan" / "partition_graph.yaml"
    plan = yamlio.load_path(path)
    before = len(plan["modules"])
    plan = rewrite(plan)
    hc = [m for m in plan["modules"] if ".hc_" in m["id"] or m["id"].startswith("hc_")]
    print(f"modules {before} -> {len(plan['modules'])} ({len(hc)} mHC)")
    print(f"tensors: {len(plan['tensors'])}")
    print(f"layer_owned_params: {len(plan['metadata']['layer_owned_params'])}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    path.write_text(yamlio.dumps(plan, sort_keys=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
