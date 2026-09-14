# Heretic DGX

Heretic DGX is a multi-node NVIDIA DGX Spark implementation of
[`p-e-w/heretic`](https://github.com/p-e-w/heretic). It runs Heretic's
directional-ablation optimization across a cluster of DGX Spark systems and
exports a standalone checkpoint.

It supports two configurations:

- **Two nodes** — the original release-0.1 topology, validated end to end.
- **Four nodes (TP4)** — for models that need more memory, specifically
  DeepSeek V4.1 Flash, whose Engram n-gram tables stay on the SSD.
  See [`docs/TP4.md`](docs/TP4.md).

## Relationship to the original Heretic project

This repository is an independent downstream fork of Philipp Emanuel
Weidmann's original
[`p-e-w/heretic`](https://github.com/p-e-w/heretic), based on upstream commit
[`bedb94e`](https://github.com/p-e-w/heretic/commit/bedb94ef117a271532ac2058447fbc165d5051bd),
itself downstream of
[`cbertucci33/Heretic-DGX`](https://github.com/cbertucci33/Heretic-DGX).
Heretic's abliteration method, scorer model, optimization approach, and core
configuration remain upstream work.

Heretic DGX adds the distributed execution layer needed to run that workflow
across a DGX Spark cluster:

- coordinator-driven launch and rank supervision;
- source, checkpoint, topology, and collective preflight checks;
- mirrored prompt, residual, scoring, and optimization operations;
- coordinated cancellation, failure reporting, and teardown;
- standalone export that verifies target changes while preserving quantized
  and non-target artifacts; and
- read-only, disk-backed Engram (n-gram) tables for DeepSeek V4.1 Flash.

For the original single-system project, documentation, and community, use the
[upstream Heretic repository](https://github.com/p-e-w/heretic). Issues specific
to the DGX implementation belong in this repository.

## DeepSeek V4.1 Flash and Engram on disk

DeepSeek V4.1 Flash carries two Engram n-gram hash tables (layers 1 and 14).
Measured from the released checkpoint they are **189.1 GiB** of fp8 data —
**47.3 GiB per rank at TP4** — on top of a 71.5 GiB-per-rank MoE backbone. A
DGX Spark has 128 GB of unified memory shared with the host, so holding both
would leave no headroom for activations or residual tensors.

With `engram_disk = true` the tables stay on the SSD and are read on demand, so
only the transformer weights occupy memory. The tables are opened read-only
(`O_RDONLY` + `POSIX_FADV_RANDOM`) and are never modified.

This is practical because each rank reads only its own row range (~47 GiB at
TP4, not 189 GiB), the whole forward's hash ids are gathered in one batch, and
rows are de-duplicated before reading. It is a port of the mechanism used by
the working vLLM DGX Spark deployment, without the vLLM dependency.

Set it up in the cluster file:

```toml
engram_disk = true
engram_disk_path = "/models/DeepSeek-V4.1-Flash"
```

See [`docs/TP4.md`](docs/TP4.md) for the full runbook.

## Release 0.1 scope

- One coordinator command launches one GPU-backed rank on each node.
- Every rank loads the model through Transformers tensor parallelism.
- Preflight checks verify node reachability, source identity, checkpoint
  identity, topology, and collective communication before optimization.
- Prompt ingestion, residual calculation, scoring, optimization, winner
  restoration, and model materialization are coordinated across all ranks.
- Failure, cancellation, timeout, and teardown behavior is bounded so a failed
  peer does not leave the other ranks running indefinitely.
- The standalone exporter preserves non-target files and quantized tensors and
  verifies intended tensor changes before reporting success.

This release is narrow by design: **Linux, NCCL, and one rank per node.** The
two-node topology is validated end to end; the TP4 topology and the disk-backed
Engram path are implemented and unit-tested but have not yet been run on
physical four-node hardware. See
[`docs/TP4.md`](docs/TP4.md#10-validation-status-and-limits).

## Validated model

Release 0.1 was proven end to end with
[`poolside/Laguna-S-2.1-FP8`](https://huggingface.co/poolside/Laguna-S-2.1-FP8).
The resulting standalone checkpoint is available as
[`cbert33/Laguna-S-2.1-Heretic-FP8`](https://huggingface.co/cbert33/Laguna-S-2.1-Heretic-FP8).

The selected trial changed only the intended BF16 attention-output projection
targets in layers 29-47 while preserving the source FP8 tensors and all
non-target artifacts. It measured a KL divergence of `0.0156` from the
untouched model's first-token probability distributions across five prompts
from `mlabonne/harmless_alpaca`.

## Requirements

- Two or four DGX Spark systems running Linux (one rank per node)
- CUDA, NCCL, and working node-to-node GPU collective communication
- Key-based SSH from the coordinator to every worker node
- The same clean Heretic DGX revision and model checkpoint on every node
- Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/)
- For DeepSeek V4.1 Flash: the extracted Hugging Face checkpoint on local
  storage on every node, for the Engram disk path

Use a dedicated high-speed fabric for rank traffic and keep a separate
management path for SSH and recovery.

## Install

Run on every node at the same absolute path:

```sh
git clone https://github.com/hyudryu/Heretic-DGX.git
cd Heretic-DGX
uv sync --frozen
```

## Configure the cluster

Copy an example outside the repository and replace every placeholder:

```sh
cp cluster.example.toml ../heretic-cluster.toml      # two nodes
cp cluster.tp4.example.toml ../heretic-cluster.toml  # four nodes (TP4)
```

The entries in `[[nodes]]` are ordered by rank: the first is the coordinator
(rank 0), and the rest are workers. `host` is the SSH destination;
`rank_address` is the address used for distributed traffic.
`nccl_socket_ifname` must name the fabric interface present on every node.

Do not commit live hostnames, addresses, credentials, or private cluster
configuration.

## Run

From the coordinator:

```sh
uv run heretic \
  --cluster ../heretic-cluster.toml \
  --config ./config.default.toml \
  --model /path/to/model
```

The coordinator rejects mismatched source trees or checkpoint payloads before
launching the optimization. Output behavior and target selection are controlled
by the Heretic configuration file.

## Verification

For a source checkout:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Model-family and quantization support must be proven independently with a full
load, optimization, standalone export, clean reload, and generation test.

## Safety and liability

Heretic changes model refusal behavior. It does not guarantee correctness,
capability retention, safety, legality, or suitability for a particular use.
KL divergence and automated checks are limited indicators, not substitutes for
broad evaluation. Review the source model's license and usage restrictions
before creating or distributing a derivative.

**User responsibility:** this software is provided without warranty. The creators, uploaders, and maintainers are not responsible or liable for what others generate, publish, deploy, or otherwise do with any abliterated models made by this. Users must operate it responsibly, apply appropriate safeguards, comply with applicable law, and respect third-party rights. This software is for research purposes only and is not intended for production use.

## Attribution and license

Heretic DGX retains the original project's AGPL-3.0-or-later license and
copyright notices. The distributed implementation and release-specific changes
are maintained in this repository. Heretic DGX is not presented as an official
upstream release.
