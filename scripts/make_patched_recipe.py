#!/usr/bin/env python3
"""Create the patched-profile recipe by copying the LIVE deployment's config.

Hand-writing the recipe would drift from what is actually running: the live
profile gained its LoRA flags and VLLM_ALLOW_RUNTIME_LORA_UPDATING through
deploy-time overrides, not through the base recipe. So read the running
deployment's configuration and change exactly one field -- the image.

Read-only with respect to the running deployment; this only creates a new recipe.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sparkdeck  # noqa: E402

LIVE = "8fd087c422c5"
NEW_IMAGE = "vllm-dsv41:pinned-ehs3-hccollapse"
NAME = "Heretic abliteration - V4.1 TP4 (hc-collapse patch)"

# Heretic-only. The patched collapse is wrong for the production DSpark drafter.
NOTE = (
    "[HERETIC ONLY] aux hidden state collapsed with the carried pre-mix instead "
    "of a plain mean. Do not use with --speculative-config dspark: the draft "
    "model was trained against mean-collapsed states. See patches/README.md."
)


def main() -> int:
    rpc = sparkdeck.rpc(
        "tools/call",
        {
            "name": "get_cluster_deployment_configuration",
            "arguments": {"deployment_id": LIVE},
        },
    )
    if "error" in rpc:
        print("RPC ERROR:", json.dumps(rpc["error"], indent=2))
        return 1
    cfg = sparkdeck.unwrap(rpc["result"])

    old_image = cfg.get("image")
    extra = list(cfg.get("extra_args") or [])
    env = dict(cfg.get("environment") or {})

    print(f"live deployment : {LIVE}  ({cfg.get('alias')})")
    print(f"image           : {old_image}  ->  {NEW_IMAGE}")
    print(f"extra_args      : {len(extra)}")
    print(f"environment     : {len(env)} keys")
    print(f"gpu_memory_util : {cfg.get('gpu_memory_utilization')}")

    # Guard: extract_hidden_states must be the speculative method.
    spec = ""
    if "--speculative-config" in extra:
        spec = extra[extra.index("--speculative-config") + 1]
    if "extract_hidden_states" not in spec:
        print("ABORT: live profile is not using extract_hidden_states")
        print("       spec =", spec[:200])
        return 1
    if "dspark" in spec:
        print("ABORT: live profile uses dspark; the patch is not safe for it")
        return 1
    # Guard: the LoRA surface must be present.
    for flag in ("--enable-lora", "--lora-target-modules"):
        if flag not in extra:
            print(f"ABORT: live profile is missing {flag}")
            return 1

    recipe = {
        "name": NAME,
        "model": cfg.get("model", {}).get("repository"),
        "engine": cfg.get("runtime") or "vllm",
        "image": NEW_IMAGE,
        "extra_args": extra,
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization"),
        "gpu_memory_gb": cfg.get("gpu_memory_gb"),
        "environment": env,
        "sg_tp_size": None,
        "sg_context_length": None,
        "sg_max_running_requests": None,
        "sg_mem_fraction": None,
        "sg_image": None,
        "deployment_mode": cfg.get("deployment_mode") or "sharded",
        "node_ids": cfg.get("node_ids"),
    }
    print()
    print("recipe to create:")
    print(json.dumps({k: (v if k not in ("extra_args", "environment") else f"<{len(v)} entries>")
                      for k, v in recipe.items()}, indent=2))
    print()
    print("note:", NOTE)

    res = sparkdeck.rpc(
        "tools/call",
        {"name": "create_cluster_recipe", "arguments": {"recipe": recipe}},
    )
    if "error" in res:
        print("RPC ERROR:", json.dumps(res["error"], indent=2))
        return 1
    out = sparkdeck.unwrap(res["result"])
    print("created recipe id:", out.get("id"))
    print("image            :", out.get("image"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
