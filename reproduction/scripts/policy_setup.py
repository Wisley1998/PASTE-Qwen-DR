#!/usr/bin/env python3
"""Freeze and render the policy protocol; execute only the original entrypoints.

No GPU, model or tool execution is performed by this utility. Generated scripts
start fresh servers, verify the lock before each cell, and retain every result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DR = ROOT / "PASTE-Qwen-DR"
PYTHON = ROOT / ".conda/envs/paste/bin/python"
CONFIG = DR / "reproduction/configs/policy_setup_v11.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def profile_environment(path):
    # Source a profile in an environment without ambient scheduler overrides.
    env = {key: os.environ[key] for key in ("HOME", "PATH", "LD_LIBRARY_PATH", "LANG")
           if key in os.environ}
    raw = subprocess.check_output(["bash", "--noprofile", "--norc", "-c",
        'set -a; source "$1"; env -0', "profile", str(path)], env=env)
    values = dict(item.decode().split("=", 1) for item in raw.split(b"\0") if b"=" in item)
    return {k: v for k, v in values.items()
            if (k.startswith("VLLM_") or k in {"MODEL_ID", "MODEL_REVISION", "HF_HOME",
                "PASTE_ENV_PREFIX", "CUDA_VISIBLE_DEVICES"}) and k != "VLLM_API_KEY"}


def runtime_versions():
    code = ('import json,sys,importlib.metadata as m; '
            'print(json.dumps({"python":sys.version, **{k:m.version(k) for k in '
            '["vllm","torch","transformers","aiohttp"]}}))')
    return json.loads(subprocess.check_output([str(PYTHON), "-c", code]))


def input_files(config):
    paths = {Path(__file__).resolve(), CONFIG, DR / "docs/RESUBMISSION_EXPERIMENT_MANUAL.md",
             DR / "docs/POLICY_EXPERIMENT_SETUP_V11.md"}
    # Freeze executable Python source, launch scripts and every config dependency,
    # including unchanged modules imported transitively by either original runner.
    for repo, package in [(DR, "paste_repro"), (ROOT / "gemini-cli-PASTE", "paste_gemini")]:
        for folder in [repo / "reproduction" / package, repo / "reproduction/scripts",
                       repo / "reproduction/configs", repo / "scripts/pythonhooks"]:
            if folder.exists():
                paths.update(p for p in folder.rglob("*") if p.is_file()
                             and "__pycache__" not in p.parts
                             and p.suffix in {".py", ".sh", ".json", ".example"})
    for project in ("dr", "coding"):
        c = config[project]
        repo = ROOT / c["repository"]
        paths.update(repo / c[key] for key in ("profile", "runner", "tune_plan", "test_plan"))
        if project == "dr":
            paths.add(repo / c["predictor"])
        else:
            for split in ("tune", "test"):
                plan = json.loads((repo / c[f"{split}_plan"]).read_text())
                paths.update(Path(t["trace_path"]) for t in plan["templates"])
                directory = ROOT / c[f"{split}_predictions"]
                files = list(directory.glob("*/typed_tool_results.jsonl"))
                expected = len(c[f"{split}_sources"]) * c[f"{split}_replicas_per_source"]
                if len(files) != expected:
                    raise ValueError(f"{directory}: expected {expected} task audits, found {len(files)}")
                paths.update(files)
                paths.update(directory.glob("*/repo_read_prefetch.jsonl"))
    return paths


def freeze(args):
    config = json.loads(args.config.read_text())
    paths = input_files(config) | {args.config.resolve()}
    selection = None
    if config["status"] == "selected_on_tune":
        if args.selection is None:
            raise ValueError("selected status requires --selection evidence, not just a changed label")
        selection = json.loads(args.selection.read_text())
        if selection.get("candidate_config_sha256") != canonical(config):
            raise ValueError("selection does not bind this exact configuration")
        results = selection.get("result_files", {})
        if not results:
            raise ValueError("selection must bind all candidate tune results")
        for name, sha in results.items():
            path = Path(name)
            if digest(path) != sha:
                raise ValueError(f"selection result drift: {path}")
            metadata = path.parent / "cell_manifest.json"
            if json.loads(metadata.read_text()).get("split") != "tune":
                raise ValueError("selection may use tune results only")
            paths.update([path, metadata])
        paths.add(args.selection.resolve())
    elif config["status"] != "corrected_reference_not_yet_tuned":
        raise ValueError("unknown setup status")
    frozen = {"schema": "paste.policy_setup_lock.v11", "config": config,
              "status": config["status"], "selection_evidence": selection,
              "files": {str(p.resolve()): digest(p) for p in sorted(paths)},
              "runtime_versions": runtime_versions(),
              "profiles": {project: profile_environment(ROOT / config[project]["repository"] /
                  config[project]["profile"]) for project in ("dr", "coding")}}
    frozen["lock_sha256"] = canonical(frozen)
    write_new(args.output, frozen)
    print(json.dumps({"lock": str(args.output), "sha256": frozen["lock_sha256"],
                      "files": len(frozen["files"]), "status": frozen["status"]}))


def verify(path):
    lock = json.loads(Path(path).read_text())
    unsigned = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if canonical(unsigned) != lock["lock_sha256"]:
        raise ValueError("lock checksum mismatch")
    bad = [p for p, sha in lock["files"].items() if not Path(p).is_file() or digest(p) != sha]
    if bad:
        raise ValueError("frozen input drift: " + ", ".join(bad))
    if runtime_versions() != lock["runtime_versions"]:
        raise ValueError("serving Python/package version drift")
    return lock


def cell_command(lock, project, split, cell, directory):
    cfg = lock["config"]
    common, c = cfg["common"], cfg[project]
    native = cell == "native_fcfs"
    spec = cell in {"length_spec", "full", "gated_fcfs_spec"}
    ranker = "fcfs" if cell == "gated_fcfs_spec" or native else "tool-aware" if cell == "full" else "length-aware"
    repo = ROOT / c["repository"]
    env = dict(lock["profiles"][project])
    env.update(CUDA_VISIBLE_DEVICES=c["gpus"], VLLM_PORT=str(c["port"]),
        VLLM_SCHED_POLICY="fcfs" if native else "online_joint_pacer_v2",
        VLLM_SCHED_JOINT_V2_RANKER=ranker,
        # The client owns exact session lifetime. A disabled duplicate server
        # foreground gate avoids counting departed causal sessions forever.
        VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS=str(c.get("server_working_set", c["working_set"])),
        VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING=str(c["decode_target"]),
        VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING=str(c["decode_max"]),
        VLLM_SCHED_JOINT_V2_FINAL_LANE="0", VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE="0",
        VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY="0",
        VLLM_SCHED_JOINT_V2_REALIZED_GAIN_WEIGHT=str(common["realized_gain_weight"]),
        VLLM_STATE_DIR=str(directory / "state"), VLLM_LOG_DIR=str(directory / "logs"),
        VLLM_SAFE_WORKING_DIR=str(directory / "empty"),
        VLLM_SCHED_TURN_AUDIT=str(directory / "ready_turns.jsonl"))
    cmd = [str(PYTHON), str(repo / c["runner"]), "run-cell", "--plan", str(repo / c[f"{split}_plan"]),
        "--system", "full" if spec else "baseline", "--model", env["MODEL_ID"],
        "--output", str(directory / "result.json"), "--base-url", f"http://127.0.0.1:{c['port']}/v1",
        "--max-active-tasks", str(80 if native else c["working_set"]),
        "--preengine-policy", "fifo", "--preengine-coalesce-s", str(common["coalesce_s"]),
        "--arrival-scale", str(common["arrival_scale"]), "--tool-capacity", str(c["tool_capacity"]),
        "--speculation-capacity", str(c["speculation_capacity"] if spec else 0),
        "--request-timeout-s", str(common["request_timeout_s"])]
    if project == "dr":
        cmd += ["--tool-backend", "timed", "--predictor", str(repo / c["predictor"]),
            "--speculation-top-k", str(c["top_k"]), "--speculation-ttl-s", "inf",
            "--speculation-pending-capacity", str(common["pending_capacity"]),
            "--speculation-cache-capacity", str(common["completed_cache_capacity"]),
            "--session-result-cache", "--preempt-speculation", "--replace-queued-predictions"]
    else:
        cmd += ["--tool-replay-mode", "timed", "--timed-prediction-dir", str(ROOT / c[f"{split}_predictions"])]
        for source in c[f"{split}_sources"]:
            cmd += ["--source-session", source]
    return env, cmd


def render(args):
    lock = verify(args.lock)
    cfg = lock["config"]
    cells = cfg["cells"]
    if args.formal and lock["status"] != "selected_on_tune":
        raise ValueError("reference config is not a tune-selected optimum; --formal requires a selected lock")
    output_root = args.run_root.resolve()
    if output_root.exists():
        raise ValueError("run root already exists; use a new run root")
    q = shlex.quote
    launcher = DR / "reproduction/scripts"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
        f"test ! -e {q(str(output_root))}",
        f"mkdir -p {q(str(output_root))}",
        f"exec 9>{q(str(ROOT / 'resubmission-runs/frozen_20260920' / (args.project + '_commands.lock')))}",
        "flock -n 9",
        # Clear inherited scheduler aliases before applying the captured profile.
        'for name in ${!VLLM_@} ${!PASTE_@} ${!MODEL_@} ${!GEMINI_UNIFIED_@}; do unset "$name"; done',
        "unset PYTHONPATH",
        f"cd {q(str(ROOT / cfg[args.project]['repository']))}",
        f"cp {q(str(args.lock.resolve()))} {q(str(output_root / 'setup_lock.json'))}"]
    for block in range(1, cfg["blocks"] + 1):
        # Five arms: alternating direction plus rotation avoids fixed arm position.
        order = cells[block - 1:] + cells[:block - 1]
        if block % 2 == 0:
            order = list(reversed(order))
        for cell in order:
            directory = output_root / f"block{block}" / cell
            env, cmd = cell_command(lock, args.project, args.split, cell, directory)
            verify_cmd = [str(PYTHON), str(Path(__file__).resolve()), "verify", "--lock", str(args.lock.resolve())]
            lines += [shlex.join(verify_cmd), f"mkdir -p {q(str(directory / 'empty'))}",
                      f"chmod 555 {q(str(directory / 'empty'))}"]
            lines += [f"export {key}={q(value)}" for key, value in sorted(env.items())]
            metadata = {"setup_lock_sha256": lock["lock_sha256"], "project": args.project,
                        "split": args.split, "cell": cell, "block": block,
                        "server_environment": env, "client_argv": cmd,
                        "scope": "formal_selected" if args.formal else "reference_or_tuning"}
            lines += [f"cat > {q(str(directory / 'cell_manifest.json'))} <<'PASTE_CELL_JSON'",
                      json.dumps(metadata, indent=2), "PASTE_CELL_JSON",
                      "kill -0 2528",
                      f"nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv > {q(str(directory / 'gpu_before.csv'))}",
                      f"trap {q('bash ' + str(launcher / 'stop_vllm.sh'))} EXIT",
                      f"bash {q(str(launcher / 'start_vllm.sh'))}",
                      shlex.join(cmd) + f" > {q(str(directory / 'client.log'))} 2>&1",
                      f"bash {q(str(launcher / 'stop_vllm.sh'))}", "trap - EXIT",
                      f"nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv > {q(str(directory / 'gpu_after.csv'))}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write("\n".join(lines) + "\n")
    print(json.dumps({"script": str(args.output), "run_root": str(output_root),
                      "cells": len(cells) * cfg["blocks"], "executed": False}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    f = subs.add_parser("freeze")
    f.add_argument("--config", type=Path, default=CONFIG)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--selection", type=Path,
        help="Required for selected_on_tune: exact config digest and all tune result hashes")
    v = subs.add_parser("verify")
    v.add_argument("--lock", type=Path, required=True)
    r = subs.add_parser("render")
    r.add_argument("--lock", type=Path, required=True)
    r.add_argument("--project", choices=("dr", "coding"), required=True)
    r.add_argument("--split", choices=("tune", "test"), required=True)
    r.add_argument("--run-root", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--formal", action="store_true")
    args = p.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "verify":
        lock = verify(args.lock)
        print(json.dumps({"verified": True, "sha256": lock["lock_sha256"]}))
    else:
        render(args)


if __name__ == "__main__":
    main()
