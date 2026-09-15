#!/usr/bin/env python3
"""Tiny SparkDeck MCP client.

The SparkDeck tools are not exposed to the agent as native tools, so we drive
the stateless HTTP JSON-RPC endpoint directly.

Usage:
    python sparkdeck.py list
    python sparkdeck.py tools
    python sparkdeck.py call <tool_name> ['{"json": "args"}'] [--select dotted.path] [--max N]

--select walks a dotted path (ints index lists) and prints only that subtree,
which is how we keep 40-node cluster snapshots out of the agent's context.
"""

import argparse
import json
import os
import sys
import urllib.request

ENDPOINT = "http://100.97.4.16:7878/mcp"


def rpc(method, params=None, req_id=1):
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    # wait_for_cluster_ready legitimately blocks for many minutes, so the
    # client timeout has to be raisable without editing this file.
    timeout = float(os.environ.get("SPARKDECK_TIMEOUT", "180"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    return json.loads(body)


def walk(obj, path):
    cur = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {part!r}")
    return cur


def unwrap(result):
    """MCP tool results come back as content blocks; prefer structuredContent."""
    if not isinstance(result, dict):
        return result
    if "structuredContent" in result:
        return result["structuredContent"]
    blocks = result.get("content")
    if isinstance(blocks, list) and blocks:
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
        joined = "\n".join(t for t in texts if t)
        try:
            return json.loads(joined)
        except (ValueError, TypeError):
            return joined
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tools", "call", "list", "deploy", "recipes"])
    ap.add_argument("tool", nargs="?")
    ap.add_argument("args", nargs="?", default="{}")
    ap.add_argument("--select", default=None)
    ap.add_argument("--max", type=int, default=None, help="truncate output chars")
    ns = ap.parse_args()

    if ns.mode == "tools":
        res = rpc("tools/list")
        tools = res.get("result", {}).get("tools", [])
        if ns.tool:
            for t in tools:
                if t["name"] == ns.tool:
                    print(json.dumps(t, indent=2, default=str))
                    return 0
            print(f"no such tool: {ns.tool}", file=sys.stderr)
            return 1
        print(json.dumps([t["name"] for t in tools], indent=2))
        return

    if ns.mode == "list":
        ns.mode = "call"
        ns.tool = "list_cluster_deployments"
        ns.args = "{}"

    if ns.mode in ("deploy", "recipes"):
        tool = (
            "list_cluster_deployments"
            if ns.mode == "deploy"
            else "list_cluster_recipes"
        )
        res = rpc("tools/call", {"name": tool, "arguments": {}})
        rows = unwrap(res.get("result", res)).get("result", [])
        for r in rows:
            if ns.mode == "deploy":
                nodes = ",".join(
                    f"{m.get('node_name')}#{m.get('rank')}:{m.get('status')}"
                    for m in r.get("members", [])
                )
                ls = r.get("launch_settings") or {}
                print(
                    f"{r.get('id')}  {r.get('name')!r}\n"
                    f"    engine={r.get('engine')} status={r.get('status')}"
                    f"/{r.get('desired_state')} image={ls.get('image')}\n"
                    f"    recipe_id={r.get('recipe_id')} managed_by={r.get('managed_by')}"
                    f" dirty={r.get('settings_dirty')} api_port={r.get('api_port')}\n"
                    f"    nodes: {nodes}"
                )
            else:
                args = r.get("extra_args") or []
                print(
                    f"{r.get('id')}  {r.get('name')!r}  image={r.get('image')}\n"
                    f"    model={r.get('model')}  gpu_mem_util={r.get('gpu_memory_utilization')}\n"
                    f"    extra_args({len(args)}): {' '.join(str(a) for a in args)[:400]}"
                )
        return 0

    try:
        arguments = json.loads(ns.args)
    except ValueError as exc:
        print(f"bad JSON args: {exc}", file=sys.stderr)
        return 2

    res = rpc("tools/call", {"name": ns.tool, "arguments": arguments})
    if "error" in res:
        print("RPC ERROR: " + json.dumps(res["error"], indent=2))
        return 1

    out = unwrap(res.get("result", res))
    if ns.select:
        try:
            out = walk(out, ns.select)
        except (KeyError, IndexError, ValueError) as exc:
            print(f"select failed: {exc}", file=sys.stderr)
            print(json.dumps(out)[:2000], file=sys.stderr)
            return 3

    text = json.dumps(out, indent=2, default=str)
    if ns.max and len(text) > ns.max:
        text = text[: ns.max] + f"\n... [truncated, {len(text)} chars total]"
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
