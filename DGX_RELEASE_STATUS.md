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

## TP4 and DeepSeek V4.1 Flash — implemented, not yet hardware-validated

The four-node (TP4) topology and the disk-backed Engram path are implemented
and covered by unit tests, but they have **not** been run end to end on
physical four-node hardware.

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

1. Load on all four ranks with `engram_disk = true`; confirm the
   "Engram table DISK-backed" log line on every rank and that memory stays
   within the 128 GB-per-node budget.
2. Run a short optimization pass on a small prompt set.
3. Export a standalone checkpoint, reload it, and generate.
4. Measure KL divergence against the untouched model as a sanity check.

Until then, treat TP4 as implemented and unit-tested, not validated. The
runbook for that validation is in [`docs/TP4.md`](docs/TP4.md).

## Supported boundary

The evidence above does not establish general support for more than two nodes,
multiple ranks per node, other hardware, every model family, or every
quantization format. Each new combination requires its own load, optimization,
export, reload, and generation proof.

The validated Laguna model artifact and its detailed limitations are documented
at <https://huggingface.co/cbert33/Laguna-S-2.1-Heretic-FP8>.
