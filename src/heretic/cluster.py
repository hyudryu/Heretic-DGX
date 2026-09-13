# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True, slots=True)
class ClusterNode:
    """One physical node in the supported multi-node topology."""

    host: str
    rank_address: str


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Validated static configuration for the DGX Spark runtime.

    The topology is one rank per node, ordered by rank: index 0 is the
    coordinator. Two nodes is the original validated configuration; four nodes
    is the TP4 configuration used for DeepSeek V4.1 Flash.
    """

    nodes: tuple[ClusterNode, ...]
    python: str
    workdir: str
    backend: str = "nccl"
    master_port: int = 29500
    timeout_seconds: int = 900
    nccl_socket_ifname: str | None = None
    engram_disk: bool = False
    engram_disk_path: str | None = None
    engram_disk_threads: int = 32
    engram_disk_chunk: int = 16

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("DGX cluster must define at least two nodes")
        if len({node.host for node in self.nodes}) != len(self.nodes):
            raise ValueError("DGX cluster must use distinct SSH hosts")
        if len({node.rank_address for node in self.nodes}) != len(self.nodes):
            raise ValueError("DGX cluster must use distinct rank addresses")

    @property
    def master_address(self) -> str:
        return self.nodes[0].rank_address

    @property
    def world_size(self) -> int:
        return len(self.nodes)


def _required_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DGX cluster field {key!r} must be a non-empty string")
    return value


def load_cluster_config(path: str | Path) -> ClusterConfig:
    """Load a multi-node cluster definition and reject ambiguity."""

    with Path(path).open("rb") as file:
        data = tomllib.load(file)

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
        raise ValueError("DGX cluster must define at least two nodes")

    nodes: list[ClusterNode] = []
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise ValueError(f"DGX cluster node {index} must be a TOML table")
        nodes.append(
            ClusterNode(
                host=_required_string(raw_node, "host"),
                rank_address=_required_string(raw_node, "rank_address"),
            )
        )

    if len({node.host for node in nodes}) != len(nodes):
        raise ValueError("DGX cluster must use distinct SSH hosts")
    if len({node.rank_address for node in nodes}) != len(nodes):
        raise ValueError("DGX cluster must use distinct rank addresses")

    backend = data.get("backend", "nccl")
    if backend != "nccl":
        raise ValueError("DGX runtime backend must be 'nccl'")

    master_port = data.get("master_port", 29500)
    if not isinstance(master_port, int) or isinstance(master_port, bool):
        raise ValueError("DGX cluster field 'master_port' must be an integer")
    if not 1 <= master_port <= 65535:
        raise ValueError("DGX cluster field 'master_port' must be between 1 and 65535")

    timeout_seconds = data.get("timeout_seconds", 900)
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise ValueError("DGX cluster field 'timeout_seconds' must be an integer")
    if timeout_seconds <= 0:
        raise ValueError("DGX cluster field 'timeout_seconds' must be positive")

    nccl_socket_ifname = data.get("nccl_socket_ifname")
    if nccl_socket_ifname is not None and (
        not isinstance(nccl_socket_ifname, str) or not nccl_socket_ifname.strip()
    ):
        raise ValueError(
            "DGX cluster field 'nccl_socket_ifname' must be a non-empty string"
        )

    engram_disk = data.get("engram_disk", False)
    if not isinstance(engram_disk, bool):
        raise ValueError("DGX cluster field 'engram_disk' must be a boolean")

    engram_disk_path = data.get("engram_disk_path")
    if engram_disk_path is not None and (
        not isinstance(engram_disk_path, str) or not engram_disk_path.strip()
    ):
        raise ValueError(
            "DGX cluster field 'engram_disk_path' must be a non-empty string"
        )
    if engram_disk and engram_disk_path is None:
        raise ValueError(
            "DGX cluster field 'engram_disk_path' is required when engram_disk is true"
        )

    engram_disk_threads = data.get("engram_disk_threads", 32)
    if not isinstance(engram_disk_threads, int) or isinstance(
        engram_disk_threads, bool
    ):
        raise ValueError("DGX cluster field 'engram_disk_threads' must be an integer")
    if engram_disk_threads <= 0:
        raise ValueError(
            "DGX cluster field 'engram_disk_threads' must be a positive integer"
        )

    engram_disk_chunk = data.get("engram_disk_chunk", 16)
    if not isinstance(engram_disk_chunk, int) or isinstance(engram_disk_chunk, bool):
        raise ValueError("DGX cluster field 'engram_disk_chunk' must be an integer")
    if engram_disk_chunk <= 0:
        raise ValueError(
            "DGX cluster field 'engram_disk_chunk' must be a positive integer"
        )

    return ClusterConfig(
        nodes=tuple(nodes),
        python=_required_string(data, "python"),
        workdir=_required_string(data, "workdir"),
        backend=backend,
        master_port=master_port,
        timeout_seconds=timeout_seconds,
        nccl_socket_ifname=nccl_socket_ifname,
        engram_disk=engram_disk,
        engram_disk_path=engram_disk_path,
        engram_disk_threads=engram_disk_threads,
        engram_disk_chunk=engram_disk_chunk,
    )
