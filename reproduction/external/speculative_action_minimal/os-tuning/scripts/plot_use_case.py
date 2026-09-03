#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRICE_MAP_USD_PER_MILLION = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

SETUP_LABELS = {
    "untuned": "Untuned",
    "actor_only": "Actor-only",
    "speculator_only": "Speculator-only",
    "dual": "Actor + Speculator",
}

SETUP_COLORS = {
    "actor_only": "#5DADE2",
    "speculator_only": "#85C1E9",
    "dual": "#E74C3C",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sysbench CPU use-case outputs from a manifest.json.")
    parser.add_argument("--manifest", required=True, help="Path to the manifest.json file.")
    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def step_series(history_data: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    t0 = history_data.get("started_at") or history_data["initial_apply_timestamp"]
    run_duration = float(history_data["run_duration_s"])
    initial = float(history_data["initial_parameters"]["min_granularity_ns"]) / 1_000_000.0

    applied_requests = [
        request
        for request in history_data.get("requests", [])
        if request.get("applied") and request.get("apply_timestamp") is not None
    ]
    applied_requests.sort(key=lambda item: item["apply_timestamp"])

    times = [0.0]
    values = [initial]
    current = initial
    for request in applied_requests:
        apply_time = max(0.0, float(request["apply_timestamp"]) - float(t0))
        proposed = request.get("proposed_parameters") or {}
        if "min_granularity_ns" not in proposed:
            continue
        current = float(proposed["min_granularity_ns"]) / 1_000_000.0
        times.append(apply_time)
        values.append(current)

    times.append(run_duration)
    values.append(current)
    return times, values


def applied_request_points(history_data: Dict[str, Any], request_type: str) -> Tuple[List[float], List[float]]:
    t0 = float(history_data.get("started_at") or history_data["initial_apply_timestamp"])
    points = []
    for request in history_data.get("requests", []):
        if request.get("request_type") != request_type:
            continue
        if not request.get("applied") or request.get("apply_timestamp") is None:
            continue
        proposed = request.get("proposed_parameters") or {}
        value = proposed.get("min_granularity_ns")
        if value is None:
            continue
        points.append(
            (
                float(request["apply_timestamp"]) - t0,
                float(value) / 1_000_000.0,
            )
        )
    points.sort(key=lambda item: item[0])
    return [item[0] for item in points], [item[1] for item in points]


def actor_line_points(history_data: Dict[str, Any], request_type: str) -> Tuple[List[float], List[float]]:
    initial = float(history_data["initial_parameters"]["min_granularity_ns"]) / 1_000_000.0
    times, values = applied_request_points(history_data, request_type)
    return [0.0, *times], [initial, *values]


def mean_p95(history_data: Dict[str, Any]) -> float:
    values = [
        float(window["benchmark_metrics"]["latency_p95"])
        for window in history_data.get("history", [])
    ]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def request_series(history_data: Dict[str, Any]) -> Dict[str, List[Tuple[float, int, float]]]:
    t0 = float(history_data.get("started_at") or history_data["initial_apply_timestamp"])
    series: Dict[str, List[Tuple[float, int, float]]] = {}
    for request in history_data.get("requests", []):
        response_time = request.get("response_timestamp")
        tokens = request.get("token_metrics") or {}
        if response_time is None:
            continue
        token_total = int(tokens.get("total_tokens", 0))
        model = request.get("model")
        cost = token_cost_usd(model, tokens) or 0.0
        series.setdefault(request["request_type"], []).append(
            (float(response_time) - t0, token_total, cost)
        )
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return series


def token_cost_usd(model_name: Optional[str], token_metrics: Optional[Dict[str, int]]) -> Optional[float]:
    if not model_name or not token_metrics:
        return None
    rates = PRICE_MAP_USD_PER_MILLION.get(model_name)
    if not rates:
        return None
    input_tokens = max(0, int(token_metrics.get("input_tokens", 0)))
    output_tokens = max(0, int(token_metrics.get("output_tokens", 0)))
    return (
        input_tokens / 1_000_000.0 * rates["input"]
        + output_tokens / 1_000_000.0 * rates["output"]
    )


def cumulative_points(events: Iterable[Tuple[float, int, float]]) -> Tuple[List[float], List[int], List[float]]:
    xs: List[float] = [0.0]
    token_totals: List[int] = [0]
    cost_totals: List[float] = [0.0]
    running_tokens = 0
    running_cost = 0.0
    for timestamp, tokens, cost in events:
        running_tokens += tokens
        running_cost += cost
        xs.append(timestamp)
        token_totals.append(running_tokens)
        cost_totals.append(running_cost)
    return xs, token_totals, cost_totals


def plot_figure5(manifest: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    convergence = {name: load_json(path) for name, path in manifest.get("convergence", {}).items()}
    reaction = {name: load_json(path) for name, path in manifest.get("reaction", {}).items()}
    paper_reaction = {name: load_json(path) for name, path in manifest.get("paper_reaction", {}).items()}
    if not any(convergence.values()) and not any(reaction.values()) and not any(paper_reaction.values()):
        return None

    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1.8, 1.0]})

    os_default_ms = None
    actor_only = convergence.get("actor_only")
    spec_only = convergence.get("speculator_only")
    dual = convergence.get("dual")
    for data in (actor_only, spec_only, dual):
        if data and os_default_ms is None:
            os_default_ms = float(data["os_defaults"]["min_granularity_ns"]) / 1_000_000.0

    if actor_only:
        xs, ys = actor_line_points(actor_only, "actor")
        left_ax.plot(
            xs,
            ys,
            color=SETUP_COLORS["actor_only"],
            linewidth=2.2,
            label=SETUP_LABELS["actor_only"],
        )
    if spec_only:
        xs, ys = actor_line_points(spec_only, "speculator")
        left_ax.scatter(
            xs,
            ys,
            color=SETUP_COLORS["speculator_only"],
            marker="x",
            s=55,
            linewidths=1.5,
            label=SETUP_LABELS["speculator_only"],
        )
    if dual:
        xs, ys = actor_line_points(dual, "actor")
        left_ax.plot(
            xs,
            ys,
            color=SETUP_COLORS["dual"],
            linewidth=2.4,
            label=SETUP_LABELS["dual"],
        )
        spec_xs, spec_ys = applied_request_points(dual, "speculator")
        left_ax.scatter(
            spec_xs,
            spec_ys,
            color=SETUP_COLORS["dual"],
            marker="x",
            s=55,
            linewidths=1.5,
        )

    if os_default_ms is not None:
        left_ax.axhline(os_default_ms, color="black", linestyle="--", linewidth=1.0, label="OS Default")
    left_ax.set_yscale("log")
    left_ax.set_xlabel("Time (s)")
    left_ax.set_ylabel("min_granularity (ms)")
    left_ax.set_ylim(0.09, 40)
    left_ax.legend(loc="upper right")
    left_ax.grid(True, which="both", alpha=0.3)

    right_ax.axis("off")
    rows = []
    reaction_table = paper_reaction if any(paper_reaction.values()) else reaction
    reaction_modes = ("untuned", "actor_only", "dual") if any(paper_reaction.values()) else ("actor_only", "speculator_only", "dual")
    for mode in reaction_modes:
        data = reaction_table.get(mode)
        if not data:
            continue
        rows.append([SETUP_LABELS[mode], f"{mean_p95(data):.2f}"])
    if rows:
        table = right_ax.table(
            cellText=rows,
            colLabels=["Configuration", "Latency p95 (ms)"],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.scale(1.1, 1.6)
    output_path = output_dir / "figure5_use_case.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure7(manifest: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    paper_reaction = manifest.get("paper_reaction", {})
    actor = load_json(paper_reaction.get("actor_only") or manifest.get("reaction", {}).get("actor_only"))
    dual = load_json(paper_reaction.get("dual") or manifest.get("reaction", {}).get("dual"))
    if not actor or not dual:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    actor_xs, actor_ys = actor_line_points(actor, "actor")
    ax.plot(
        actor_xs,
        actor_ys,
        label=SETUP_LABELS["actor_only"],
        color=SETUP_COLORS["actor_only"],
        linewidth=2.2,
    )
    dual_actor_xs, dual_actor_ys = actor_line_points(dual, "actor")
    ax.plot(
        dual_actor_xs,
        dual_actor_ys,
        label=SETUP_LABELS["dual"],
        color=SETUP_COLORS["dual"],
        linewidth=2.4,
    )
    dual_spec_xs, dual_spec_ys = applied_request_points(dual, "speculator")
    ax.scatter(
        dual_spec_xs,
        dual_spec_ys,
        color=SETUP_COLORS["dual"],
        marker="x",
        s=55,
        linewidths=1.5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("min_granularity (ms)")
    ax.set_ylim(0.09, 40)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    output_path = output_dir / "figure7_reaction_trace.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_token_cost(manifest: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    datasets: Dict[str, Dict[str, Any]] = {}
    for section in ("convergence", "reaction"):
        for mode, path in manifest.get(section, {}).items():
            data = load_json(path)
            if data:
                datasets[f"{section}:{mode}"] = data
    if not datasets:
        return None

    fig, (cost_ax, token_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    plotted = False
    for key, data in datasets.items():
        role_series = request_series(data)
        for request_type, events in role_series.items():
            if not events:
                continue
            xs, token_totals, cost_totals = cumulative_points(events)
            label = f"{key}:{request_type}"
            cost_ax.step(xs, cost_totals, where="post", label=label)
            token_ax.step(xs, token_totals, where="post", label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    cost_ax.set_title("Cumulative Cost")
    cost_ax.set_xlabel("Time (s)")
    cost_ax.set_ylabel("USD")
    cost_ax.grid(True, alpha=0.3)
    cost_ax.legend(loc="best", fontsize=8)

    token_ax.set_title("Cumulative Tokens")
    token_ax.set_xlabel("Time (s)")
    token_ax.set_ylabel("Tokens")
    token_ax.grid(True, alpha=0.3)
    token_ax.legend(loc="best", fontsize=8)

    output_path = output_dir / "figure9_tokens_cost.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(str(manifest_path))
    if manifest is None:
        raise SystemExit(f"Manifest not found: {manifest_path}")

    output_dir = manifest_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "figure5": plot_figure5(manifest, output_dir),
        "figure7": plot_figure7(manifest, output_dir),
        "tokens_cost": plot_token_cost(manifest, output_dir),
    }

    for name, path in outputs.items():
        if path is not None:
            print(f"{name}: {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
