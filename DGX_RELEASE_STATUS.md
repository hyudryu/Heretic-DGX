# Heretic DGX validation status

## Release 0.1 (two nodes) — validated

Release 0.1 is validated for a deliberately narrow deployment: exactly two
NVIDIA DGX Spark systems running one NCCL rank per node.

### Completed validation

- A two-rank small-model fixture exercised coordinator launch, collective
  startup, prompt ingestion, residual calculation, two optimization trials,
  winner restoration, coordinated export, clean shutdown, artifact reload, and
  token generation.
- The full `poolside/Laguna-S-2.1-FP8` checkpoint loaded on both ranks, completed
  scoring and optimization, restored the selected trial, and produced a
  standalone checkpoint.
- Export verification confirmed that only intended target intervals changed,
  reconstructed targets matched the merge oracle, source FP8 tensors remained
  unchanged, and the artifact checksum manifest passed.
- The exported Laguna checkpoint passed a clean distributed runtime load and
  generated a completion.
- The selected Laguna trial measured KL divergence `0.0156` against the
  untouched model's first-token distributions on five
  `mlabonne/harmless_alpaca` prompts.

## TP4 and DeepSeek V4.1 Flash — blocked upstream

The four-node (TP4) topology is implemented and covered by unit tests, but the
target model **cannot currently be loaded at all**. `transformers` implements no
`deepseek_v41` architecture — not at the pinned commit, not in the latest
release (5.17.0), and not on `main` — and Heretic loads models only through
`transformers`.

A load attempt on `gx10-node-1` against the materialised checkpoint reached
`Model(settings)` and failed there, for every configured dtype:

```
* Failed: The checkpoint you are trying to load has model type `deepseek_v41`
  but Transformers does not recognize this architecture.
Exception: Failed to load model with all configured dtypes.
```

Full evidence and analysis, including four further blockers:
[`docs/BLOCKERS.md`](docs/BLOCKERS.md).

### What is implemented

- The cluster config, launch planner, rank environment, preflight, collective
  probe, command channel, and application runner are generalized from exactly
  two ranks to N ranks. Two-node configurations behave as before.
- `heretic.engram_disk` reads DeepSeek V4.1 Flash's Engram n-gram tables from
  the safetensors shards on demand. The tables are opened read-only
  (`O_RDONLY` + `POSIX_FADV_RANDOM`), are never modified, and never occupy
  device memory. Each rank reads only its own row range.
- `model_loading` detects DeepSeek V4.1 Flash and computes the per-rank Engram
  row ranges from the checkpoint's `engram_layer_ids` /
  `engram_num_embeddings` / `engram_head_dim`.

**Caveat: neither of the last two is connected to anything.** Both modules are
referenced only by `tests/`; `model.py` never calls them, so `engram_disk = true`
has no effect and the tables are loaded as ordinary parameters. That is blocker 3
in [`docs/BLOCKERS.md`](docs/BLOCKERS.md).

### What is tested

- The full suite (76 tests) passes on Linux, including:
  - rank-offset correctness for the disk-backed Engram reader, per rank;
  - a negative control proving that test can actually detect the original
    "ranks 1-3 read rank 0's rows" bug;
  - de-duplication, unowned-row zeroing, and two-layer batched reads;
  - N-node config loading, launch planning, rank-environment validation, and
    the Engram geometry of the released checkpoint.
- These are offline tests against synthetic shards in the real layout. They do
  not exercise a GPU, NCCL, or the real checkpoint.

### Required before calling it supported

Following release 0.1's own discipline, each new topology and model needs its
own proof. For TP4 + DeepSeek V4.1 Flash:

1. Load on all four ranks with `engram_disk = true`. — **Attempted; fails for
   reasons external to this repository** (see above). The "Engram table
   DISK-backed" log line previously named here cannot appear at all: it is
   produced by `EngramOffloadPlan.describe()`, which nothing calls, because the
   offload is not wired into the load path.
2. Run a short optimization pass on a small prompt set.
3. Export a standalone checkpoint, reload it, and generate. — also blocked: the
   exporter is hard-coded to Laguna S 2.1 FP8 (`expected_layer_count=48`; this
   model has 40 layers).
4. Measure KL divergence against the untouched model as a sanity check.

Until then, treat TP4 as implemented and unit-tested, not validated. The runbook
for that validation is in [`docs/TP4.md`](docs/TP4.md).

## Supported boundary

The evidence above does not establish general support for more than two nodes,
multiple ranks per node, other hardware, every model family, or every
quantization format. Each new combination requires its own load, optimization,
export, reload, and generation proof.

The validated Laguna model artifact and its detailed limitations are documented
at <https://huggingface.co/cbert33/Laguna-S-2.1-Heretic-FP8>.
