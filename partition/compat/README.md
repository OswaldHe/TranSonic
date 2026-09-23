# Bundled compatibility patches

A *compatibility patch* is a small python file a run keeps under `<run-dir>/compat/`,
applied to the model's imported entry module before anything is constructed. The loop's
own agent writes them when a vendor kernel will not run on the card in front of it — see
`prompts/port_kernels.md` and the protocol in `model_partition/runtime/compat.py`.

The patches here are different: they are **written by hand, kept with the repo, and copied
into a run's `compat/` by the operator**, because they are not adaptations to one machine.
They are structural, and they are part of how a given model is partitioned at all, so they
have to be reproducible rather than re-derived per run.

```bash
cp partition/compat/deepseek-v4.1-flash/*.py <run-dir>/compat/
```

## `deepseek-v4.1-flash/hyper_connections.py`

Makes the mHC hyper-connection mix four real submodules per layer. Without it
`hc_mixes`, `hc_pre` and `hc_post` are *methods* on `Block` holding their coefficients as
parameters on the block itself, so a hook-based trace has nothing to attach to: the
arithmetic on every sublayer boundary belongs to no module, the 258 `hc_*` parameters are
owned by nothing, and `verify_chain` carries almost none of the stream.

It pairs with two other things, and none of the three works alone:

- the `hc_*` → `hc_*_in.*` rules in `config/models/deepseek-v4.1-flash.yaml`, which put
  the checkpoint's flat coefficient names onto the submodules that now own them;
- `scripts/build_hc_plan.py`, which declares the modules and rewires the stream to rank 4.

The arithmetic is copied verbatim from the vendor's methods and is asserted identical on
random inputs, because a patch that changed a number would silently move the reference
every later stage is measured against.
