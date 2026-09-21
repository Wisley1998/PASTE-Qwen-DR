#!/usr/bin/env python3
"""Materialize load replicas of tune roots; no environments, model or tools."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "resubmission-runs/setup_v11"


def load(path):
    return json.loads(path.read_text())


def write_plan(path, plan):
    plan.pop("plan_sha256", None)
    raw = json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    plan["plan_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with path.open("x") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dr_dir = ROOT / "PASTE-Qwen-DR/reproduction/artifacts/resubmission"
    tune, test = load(dr_dir / "dr_tune_original_entry.json"), load(dr_dir / "dr_test_original_entry.json")
    assert not {t["session_id"] for t in tune["traces"]} & {t["session_id"] for t in test["traces"]}
    expanded = copy.deepcopy(tune)
    expanded["traces"] = []
    for i in range(80):
        task = copy.deepcopy(tune["traces"][i % 20])
        task.update(task_id=f"dr-tune-{i:03d}", replica_index=i // 20,
            release_offset_s=test["traces"][i]["release_offset_s"],
            arrival_source_id=test["traces"][i]["arrival_source_id"],
            arrival_source_row=test["traces"][i]["arrival_source_row"])
        expanded["traces"].append(task)
    expanded["sources"]["tuning_replication"] = {
        "source_plan_sha256": tune["plan_sha256"], "roots": 20, "replicas": 4,
        "arrival_only_plan_sha256": test["plan_sha256"],
        "statistical_unit": "source session, never replica"}
    for key in ["all_tool_service_s", "executable_visit_urls", "fixed_completion_tokens",
                "offline_cache_hits", "offline_saved_visit_s", "prompt_tokens", "requests",
                "sessions", "tools", "visit_service_s"]:
        expanded["summary"][key] *= 4
    expanded["summary"]["arrival_span_s"] = test["summary"]["arrival_span_s"]
    write_plan(OUT / "dr_tune_80.json", expanded)

    base = load(ROOT / "gemini-cli-PASTE/reproduction/analysis/real_trace_hybrid_v3/plan.json")
    roots = {"sphinx-doc__sphinx-8474", "pylint-dev__pylint-7114"}
    templates = {int(t["template_index"]): t for t in base["templates"]}
    tuning = [t for t in base["sessions"] if templates[t["template_index"]]["session_id"] in roots]
    arrivals = [t for t in base["sessions"] if templates[t["template_index"]]["session_id"] not in roots]
    assert len(tuning) == 16 and len(arrivals) == 64
    coding = copy.deepcopy(base)
    coding["sessions"] = []
    audit = OUT / "coding_tune_predictions_64"
    audit.mkdir(exist_ok=False)
    for i in range(64):
        original = tuning[i % 16]
        task = copy.deepcopy(original)
        task.update(task_id=f"swe-tune-{i:03d}", replica_index=i // 2,
                    release_offset_s=arrivals[i]["release_offset_s"],
                    source_id=arrivals[i]["source_id"], source_row=arrivals[i]["source_row"])
        coding["sessions"].append(task)
        destination = audit / task["task_id"]
        destination.mkdir()
        source = ROOT / "resubmission-runs/tune-original/coding/cap8/tasks" / original["task_id"] / "trace"
        for name in ["typed_tool_results.jsonl", "repo_read_prefetch.jsonl"]:
            shutil.copyfile(source / name, destination / name)
    coding["sources"]["tuning_replication"] = {"source_plan_sha256": base["plan_sha256"],
        "roots": sorted(roots), "replicas_per_root": 32,
        "candidate_source": "tune-original/coding/cap8/tasks/*/trace; executor records only; no outcomes used for selection",
        "statistical_unit": "source task, never replica"}
    # Remove unused test templates: tune command cannot accidentally select them.
    coding["templates"] = [t for t in coding["templates"] if t["session_id"] in roots]
    coding["summary"] = {"sessions": 64, "source_tasks": 2,
        "arrival_span_s": max(t["release_offset_s"] for t in coding["sessions"]),
        "live_requests": sum(len(templates[t["template_index"]]["calls"]) for t in coding["sessions"])}
    write_plan(OUT / "coding_tune_64.json", coding)
    print("Prepared DR 20x4 and Coding 2x32 independent runtime sessions; source identities preserved.")


if __name__ == "__main__":
    main()
