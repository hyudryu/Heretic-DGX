# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from .checkpoint_identity import (
    CheckpointFileIdentity,
    CheckpointPayloadIdentity,
    build_checkpoint_payload_identity,
)
from .rank_environment import read_rank_environment
from .source_identity import SourceIdentity, build_source_identity


@dataclass(frozen=True, slots=True)
class RankPreflightIdentity:
    """Source/runtime and checkpoint evidence reported by one rank."""

    rank: int
    source: SourceIdentity
    checkpoint: CheckpointPayloadIdentity

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("preflight rank must be a nonnegative integer")
        if type(self.source) is not SourceIdentity:
            raise TypeError("preflight source must be exactly SourceIdentity")
        if type(self.checkpoint) is not CheckpointPayloadIdentity:
            raise TypeError(
                "preflight checkpoint must be exactly CheckpointPayloadIdentity"
            )

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def require_matching_rank_preflights(
    identities: tuple[RankPreflightIdentity, ...],
) -> RankPreflightIdentity:
    """Require exact identity agreement across every rank before process launch."""

    if len(identities) < 2:
        raise RuntimeError("rank preflights require at least two identities")
    if any(type(identity) is not RankPreflightIdentity for identity in identities):
        raise TypeError("rank preflights must be exactly RankPreflightIdentity")
    if tuple(identity.rank for identity in identities) != tuple(range(len(identities))):
        raise RuntimeError(
            "rank preflights must be ordered by ascending contiguous rank"
        )
    first = identities[0]
    for identity in identities[1:]:
        if identity.source != first.source:
            raise RuntimeError(
                f"rank {identity.rank} source/runtime identity does not match rank 0"
            )
        if identity.checkpoint != first.checkpoint:
            raise RuntimeError(
                f"rank {identity.rank} checkpoint-payload identity does not match rank 0"
            )
    return first


def parse_rank_preflight_identity(payload: str) -> RankPreflightIdentity:
    """Parse the canonical single-record output emitted by a rank preflight."""

    try:
        raw = json.loads(payload)
        source = SourceIdentity(**raw["source"])
        checkpoint = CheckpointPayloadIdentity(
            files=tuple(
                CheckpointFileIdentity(**file) for file in raw["checkpoint"]["files"]
            ),
            digest=raw["checkpoint"]["digest"],
        )
        identity = RankPreflightIdentity(
            rank=raw["rank"], source=source, checkpoint=checkpoint
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid rank preflight output") from error
    if identity.canonical_json() != payload.strip():
        raise ValueError("rank preflight output must be canonical JSON")
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("checkpoint_directory")
    args = parser.parse_args(argv)
    environment = read_rank_environment(os.environ)
    identity = RankPreflightIdentity(
        rank=environment.rank,
        source=build_source_identity(
            args.workdir,
            python_executable=str(Path(sys.executable).resolve()),
            python_version=sys.version.split()[0],
            package_version=version("heretic-llm"),
        ),
        checkpoint=build_checkpoint_payload_identity(args.checkpoint_directory),
    )
    print(identity.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
