#!/usr/bin/env python3
"""Render request-level Pattern-v2 accuracy and latency as SVG and PNG.

The renderer deliberately has no matplotlib dependency.  It accepts either a
CSV file or a JSON payload containing a list under ``requests``,
``per_request``, ``per_trace``, or ``rows``.  Missing request numbers are kept
as explicit N/A columns so a 1..100 request axis cannot silently collapse.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


WIDTH = 2400
HEIGHT = 1450
PLOT_LEFT = 230.0
PLOT_RIGHT = 2090.0
PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT

WHITE = "#FFFFFF"
INK = "#17202A"
MUTED = "#5F6B76"
GRID = "#D9DEE3"
GRID_LIGHT = "#EEF1F4"
NA_FILL = "#E5E7EB"
TOP_COLORS = {
    "Top-1": "#0072B2",
    "Top-3": "#009E73",
    "Top-5": "#E69F00",
    "Runtime overlap": "#CC79A7",
}
BASELINE_COLOR = "#6B7280"
PATTERN_COLOR = "#0072B2"
POSITIVE_COLOR = "#009E73"
NEGATIVE_COLOR = "#D55E00"


@dataclass(frozen=True)
class RequestMetric:
    request_number: int
    trace_id: str
    top_target_count: int | None
    executable_target_count: int | None
    top1_hits: float | None
    top3_hits: float | None
    top5_hits: float | None
    runtime_hits: float | None
    top1_recall: float | None
    top3_recall: float | None
    top5_recall: float | None
    runtime_hit_rate: float | None
    baseline_latency_ms: float | None
    pattern_latency_ms: float | None
    speedup_factor: float | None
    speedup_min: float | None
    speedup_max: float | None


def _flatten(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a few common result wrappers without losing top-level keys."""

    result = dict(mapping)
    for outer, value in mapping.items():
        if not isinstance(value, Mapping):
            continue
        for inner, nested in value.items():
            result.setdefault(str(inner), nested)
            result.setdefault(f"{outer}_{inner}", nested)
    return result


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, "", "null", "None"):
            return mapping[name]
    return None


def _number(value: Any) -> float | None:
    if value in (None, "", "null", "None", "NA", "N/A"):
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _rate(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"rate is outside [0,1]: {value!r}")
    return number


def _request_number(row: Mapping[str, Any]) -> int:
    direct = _integer(
        _first(
            row,
            (
                "request_number",
                "task_number",
                "trace_number",
                "request_index",
                "task_index",
            ),
        )
    )
    if direct is not None:
        # Zero-based indexes are only accepted through explicitly named index
        # fields.  The result artifact itself should prefer request_number.
        if direct == 0 and "request_number" not in row:
            return 1
        return direct
    trace_id = str(
        _first(row, ("trace_id", "session_id", "trace", "request_id")) or ""
    )
    match = re.search(r"(?:^|[_-])task[_-]?(\d+)(?:[_-]|$)", trace_id)
    if not match:
        raise ValueError(f"cannot determine request number from row: {trace_id!r}")
    return int(match.group(1))


def _metric_pair(
    row: Mapping[str, Any],
    *,
    rate_names: Sequence[str],
    hit_names: Sequence[str],
    denominator: int | None,
) -> tuple[float | None, float | None]:
    hits = _number(_first(row, hit_names))
    rate = _rate(_first(row, rate_names))
    if denominator == 0:
        return None, None
    if rate is None and hits is not None and denominator:
        rate = hits / denominator
    if hits is None and rate is not None and denominator is not None:
        hits = rate * denominator
    return hits, rate


def parse_request_metric(raw: Mapping[str, Any]) -> RequestMetric:
    row = _flatten(raw)
    request_number = _request_number(row)
    trace_id = str(
        _first(row, ("trace_id", "session_id", "trace", "request_id")) or ""
    )
    top_targets = _integer(
        _first(
            row,
            (
                "top_target_count",
                "authoritative_target_labels",
                "target_labels",
                "authoritative_targets",
                "target_count",
            ),
        )
    )
    executable_targets = _integer(
        _first(
            row,
            (
                # Runtime rows are pooled across paired repetitions, so their
                # realized-hit denominator can exceed the single-replay
                # authoritative target count used by Top-k recall.
                "runtime_target_observations",
                "executable_target_count",
                "executable_authoritative_targets",
                "executable_targets",
                "runtime_target_count",
            ),
        )
    )
    if executable_targets is None:
        executable_targets = top_targets

    top1_hits, top1 = _metric_pair(
        row,
        rate_names=(
            "top1_recall",
            "top_1_recall",
            "top1_target_recall",
            "top1_exact_recall",
            "top1_exact_target_recall",
            "exact_top1_recall",
        ),
        hit_names=(
            "top1_hits",
            "top_1_hits",
            "top1_target_hits",
            "exact_top1_hits",
        ),
        denominator=top_targets,
    )
    top3_hits, top3 = _metric_pair(
        row,
        rate_names=(
            "top3_recall",
            "top_3_recall",
            "top3_target_recall",
            "top3_exact_recall",
            "top3_exact_target_recall",
            "exact_top3_recall",
        ),
        hit_names=(
            "top3_hits",
            "top_3_hits",
            "top3_target_hits",
            "exact_top3_hits",
        ),
        denominator=top_targets,
    )
    top5_hits, top5 = _metric_pair(
        row,
        rate_names=(
            "top5_recall",
            "top_5_recall",
            "top5_target_recall",
            "top5_exact_recall",
            "top5_exact_target_recall",
            "exact_top5_recall",
        ),
        hit_names=(
            "top5_hits",
            "top_5_hits",
            "top5_target_hits",
            "exact_top5_hits",
        ),
        denominator=top_targets,
    )
    runtime_hits, runtime_rate = _metric_pair(
        row,
        rate_names=(
            "runtime_hit_rate",
            "runtime_overlap_hit_rate",
            "runtime_overlap_rate",
            "runtime_overall_hit_rate",
            "overall_hit_rate",
            "overlap_coverage",
        ),
        hit_names=(
            "runtime_overlap_hits",
            "runtime_hits",
            "overlap_hits",
        ),
        denominator=executable_targets,
    )

    baseline = _number(
        _first(
            row,
            (
                "baseline_request_critical_path_proxy_ms_mean",
                "baseline_latency_ms",
                "demand_only_latency_ms",
                "baseline_ms",
                "baseline_total_exposed_wait_ms",
                "baseline_authority_wait_ms_mean_per_request",
                "baseline_request_critical_path_proxy_ms",
            ),
        )
    )
    pattern = _number(
        _first(
            row,
            (
                "pattern_request_critical_path_proxy_ms_mean",
                "pattern_conservative_latency_ms",
                "conservative_pattern_latency_ms",
                "pattern_latency_ms",
                "treatment_latency_ms",
                "speculative_latency_ms",
                "pattern_total_exposed_wait_ms",
                "pattern_conservative_latency_ms_mean_per_request",
                "pattern_conservative_request_critical_path_proxy_ms",
            ),
        )
    )
    if pattern is None:
        raw_pattern = _number(
            _first(row, ("pattern_raw_latency_ms", "treatment_raw_latency_ms"))
        )
        overhead = _number(
            _first(
                row,
                (
                    "predictor_overhead_ms",
                    "runtime_overhead_ms",
                    "conservative_runtime_overhead_ms",
                ),
            )
        )
        if raw_pattern is not None:
            pattern = raw_pattern + (overhead or 0.0)

    factor = _number(
        _first(
            row,
            (
                "request_critical_path_speedup_ratio",
                "conservative_authority_wait_speedup_ratio",
                "conservative_speedup_factor",
                "conservative_speedup_ratio",
                "speedup_factor",
                "speedup_ratio",
                "baseline_over_pattern",
            ),
        )
    )
    if baseline is not None and pattern is not None and pattern > 0.0:
        factor = baseline / pattern
    elif factor is None:
        benefit_pct = _number(
            _first(row, ("speedup_pct", "net_latency_benefit_pct"))
        )
        if benefit_pct is not None and benefit_pct < 100.0:
            factor = 1.0 / (1.0 - benefit_pct / 100.0)

    speedup_min = _number(
        _first(
            row,
            (
                "speedup_factor_min",
                "repeat_speedup_factor_min",
                "conservative_speedup_factor_min",
                "conservative_speedup_ratio_repeat_min",
                "conservative_speedup_ratio_min",
            ),
        )
    )
    speedup_max = _number(
        _first(
            row,
            (
                "speedup_factor_max",
                "repeat_speedup_factor_max",
                "conservative_speedup_factor_max",
                "conservative_speedup_ratio_repeat_max",
                "conservative_speedup_ratio_max",
            ),
        )
    )

    return RequestMetric(
        request_number=request_number,
        trace_id=trace_id,
        top_target_count=top_targets,
        executable_target_count=executable_targets,
        top1_hits=top1_hits,
        top3_hits=top3_hits,
        top5_hits=top5_hits,
        runtime_hits=runtime_hits,
        top1_recall=top1,
        top3_recall=top3,
        top5_recall=top5,
        runtime_hit_rate=runtime_rate,
        baseline_latency_ms=baseline,
        pattern_latency_ms=pattern,
        speedup_factor=factor,
        speedup_min=speedup_min,
        speedup_max=speedup_max,
    )


def _extract_json_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in (
            "requests",
            "per_request",
            "per_trace",
            "request_metrics",
            "rows",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
            if isinstance(value, Mapping):
                rows = list(value.values())
                break
        if rows is None:
            raise ValueError("JSON has no request-level row collection")
    else:
        raise ValueError("JSON input must be an object or array")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("every request-level row must be an object")
    return list(rows)


def load_request_metrics(path: Path) -> list[RequestMetric]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        with path.open(encoding="utf-8") as handle:
            raw_rows = _extract_json_rows(json.load(handle))
    rows = [parse_request_metric(row) for row in raw_rows]
    seen: set[int] = set()
    for row in rows:
        if row.request_number in seen:
            raise ValueError(f"duplicate request number: {row.request_number}")
        seen.add(row.request_number)
        rates = (row.top1_recall, row.top3_recall, row.top5_recall)
        present = [value for value in rates if value is not None]
        if len(present) == 3 and not (
            row.top1_recall <= row.top3_recall <= row.top5_recall
        ):
            raise ValueError(
                f"Top-k recall is not monotone for request {row.request_number}"
            )
    return rows


def complete_request_axis(
    rows: Sequence[RequestMetric], request_count: int
) -> list[RequestMetric]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    by_number = {row.request_number: row for row in rows}
    outside = sorted(number for number in by_number if not 1 <= number <= request_count)
    if outside:
        raise ValueError(f"request numbers outside 1..{request_count}: {outside}")
    missing = RequestMetric(
        request_number=0,
        trace_id="",
        top_target_count=None,
        executable_target_count=None,
        top1_hits=None,
        top3_hits=None,
        top5_hits=None,
        runtime_hits=None,
        top1_recall=None,
        top3_recall=None,
        top5_recall=None,
        runtime_hit_rate=None,
        baseline_latency_ms=None,
        pattern_latency_ms=None,
        speedup_factor=None,
        speedup_min=None,
        speedup_max=None,
    )
    return [
        by_number.get(number, RequestMetric(**{**missing.__dict__, "request_number": number}))
        for number in range(1, request_count + 1)
    ]


def _hex_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mix_with_white(color: str, value: float) -> str:
    value = max(0.0, min(1.0, value))
    rgb = _hex_rgb(color)
    # Keep zero distinguishable from N/A while leaving room for saturation.
    strength = 0.10 + 0.90 * value
    mixed = tuple(round(255 - (255 - channel) * strength) for channel in rgb)
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _aggregate_rate(
    rows: Sequence[RequestMetric],
    *,
    hits_name: str,
    rate_name: str,
    denominator_name: str,
) -> tuple[float | None, float | None, float | None, str]:
    usable = [
        row
        for row in rows
        if getattr(row, rate_name) is not None
        and (getattr(row, denominator_name) or 0) > 0
    ]
    if not usable:
        return None, None, None, "N/A"
    denominator = float(sum(getattr(row, denominator_name) or 0 for row in usable))
    available_hits = [getattr(row, hits_name) for row in usable]
    if all(value is not None for value in available_hits):
        hits = float(sum(value or 0.0 for value in available_hits))
    else:
        hits = float(
            sum(
                (getattr(row, rate_name) or 0.0)
                * (getattr(row, denominator_name) or 0)
                for row in usable
            )
        )
    rate = hits / denominator if denominator else None
    if rate is None:
        return hits, denominator, rate, "N/A"
    rounded_hits = round(hits)
    hit_text = str(rounded_hits) if math.isclose(hits, rounded_hits) else f"{hits:.1f}"
    return hits, denominator, rate, f"{hit_text}/{int(denominator)}  {100 * rate:.1f}%"


def aggregate_summary(rows: Sequence[RequestMetric]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    definitions = (
        ("top1", "top1_hits", "top1_recall", "top_target_count"),
        ("top3", "top3_hits", "top3_recall", "top_target_count"),
        ("top5", "top5_hits", "top5_recall", "top_target_count"),
        (
            "runtime",
            "runtime_hits",
            "runtime_hit_rate",
            "executable_target_count",
        ),
    )
    for key, hits, rate, denominator in definitions:
        raw_hits, raw_denominator, raw_rate, text = _aggregate_rate(
            rows,
            hits_name=hits,
            rate_name=rate,
            denominator_name=denominator,
        )
        result[key] = {
            "hits": raw_hits,
            "denominator": raw_denominator,
            "rate": raw_rate,
            "text": text,
        }
    runtime_multipliers = {
        row.executable_target_count / row.top_target_count
        for row in rows
        if row.runtime_hit_rate is not None
        and row.executable_target_count
        and row.top_target_count
    }
    if len(runtime_multipliers) == 1:
        multiplier = next(iter(runtime_multipliers))
        if (
            multiplier.is_integer()
            and multiplier > 1
            and result["runtime"]["rate"] is not None
        ):
            repetitions = int(multiplier)
            hits = result["runtime"]["hits"]
            rounded_hits = round(hits) if hits is not None else 0
            hit_text = (
                str(rounded_hits)
                if hits is None or math.isclose(hits, rounded_hits)
                else f"{hits:.1f}"
            )
            top_denominator = sum(
                row.top_target_count or 0
                for row in rows
                if row.runtime_hit_rate is not None
            )
            result["runtime"]["text"] = (
                f"{hit_text}/({top_denominator}x{repetitions})  "
                f"{100 * result['runtime']['rate']:.1f}%"
            )
    latency_rows = [
        row
        for row in rows
        if row.baseline_latency_ms is not None
        and row.pattern_latency_ms is not None
        and row.pattern_latency_ms >= 0.0
    ]
    baseline_total = sum(row.baseline_latency_ms or 0.0 for row in latency_rows)
    pattern_total = sum(row.pattern_latency_ms or 0.0 for row in latency_rows)
    factor = baseline_total / pattern_total if pattern_total > 0 else None
    factors = [row.speedup_factor for row in latency_rows if row.speedup_factor is not None]
    result["latency"] = {
        "requests": len(latency_rows),
        "baseline_total_ms": baseline_total,
        "pattern_total_ms": pattern_total,
        "weighted_speedup_factor": factor,
        "median_speedup_factor": statistics.median(factors) if factors else None,
    }
    result["na_requests"] = sum(
        row.top1_recall is None and row.runtime_hit_rate is None for row in rows
    )
    return result


class SvgCanvas:
    def __init__(self, width: int, height: int, title: str, description: str):
        self.width = width
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                'aria-labelledby="chart-title chart-desc">'
            ),
            f'<title id="chart-title">{escape(title)}</title>',
            f'<desc id="chart-desc">{escape(description)}</desc>',
            "<defs>",
            (
                '<pattern id="na-hatch" width="8" height="8" '
                'patternUnits="userSpaceOnUse"><rect width="8" height="8" '
                f'fill="{NA_FILL}"/><path d="M-2 2 L2 -2 M0 8 L8 0 M6 10 L10 6" '
                'stroke="#AAB2BA" stroke-width="1"/></pattern>'
            ),
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 0.0,
        tooltip: str | None = None,
        hatch: bool = False,
    ) -> None:
        body = (
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
            f'height="{height:.2f}" fill="{"url(#na-hatch)" if hatch else fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )
        if tooltip:
            self.parts.extend(("<g>", f"<title>{escape(tooltip)}</title>", body, "</g>"))
        else:
            self.parts.append(body)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        fill: str,
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{fill}" stroke-width="{width:.2f}"{dashed}/>'
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        fill: str,
        width: float,
    ) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{encoded}" fill="none" stroke="{fill}" '
            f'stroke-width="{width:.2f}" stroke-linejoin="round"/>'
        )

    def circle(
        self, x: float, y: float, radius: float, *, fill: str, stroke: str = "none"
    ) -> None:
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int,
        fill: str = INK,
        anchor: str = "start",
        weight: str = "normal",
    ) -> None:
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{escape(value)}</text>'
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join((*self.parts, "</svg>", "")), encoding="utf-8")


class PillowCanvas:
    def __init__(self, width: int, height: int):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("Pillow is required for PNG output") from exc
        self.ImageFont = ImageFont
        self.image = Image.new("RGB", (width, height), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts: dict[tuple[int, str], Any] = {}

    def _font(self, size: int, weight: str = "normal") -> Any:
        key = (size, weight)
        if key not in self.fonts:
            self.fonts[key] = self.ImageFont.load_default(size=size)
        return self.fonts[key]

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 0.0,
        tooltip: str | None = None,
        hatch: bool = False,
    ) -> None:
        box = (round(x), round(y), round(x + width), round(y + height))
        self.draw.rectangle(
            box,
            fill=NA_FILL if hatch else fill,
            outline=None if stroke == "none" else stroke,
            width=max(1, round(stroke_width)) if stroke != "none" else 1,
        )
        if hatch:
            local_width = round(width)
            local_height = round(height)
            for offset in range(-local_height, local_width + 1, 8):
                start = max(0, -offset)
                stop = min(local_height, local_width - offset)
                if start > stop:
                    continue
                self.draw.line(
                    (
                        round(x + offset + start),
                        round(y + local_height - start),
                        round(x + offset + stop),
                        round(y + local_height - stop),
                    ),
                    fill="#AAB2BA",
                    width=1,
                )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        fill: str,
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        if dash:
            dash_length, gap = (float(part) for part in dash.split()[:2])
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance == 0:
                return
            cursor = 0.0
            while cursor < distance:
                stop = min(cursor + dash_length, distance)
                self.draw.line(
                    (
                        x1 + (x2 - x1) * cursor / distance,
                        y1 + (y2 - y1) * cursor / distance,
                        x1 + (x2 - x1) * stop / distance,
                        y1 + (y2 - y1) * stop / distance,
                    ),
                    fill=fill,
                    width=max(1, round(width)),
                )
                cursor = stop + gap
        else:
            self.draw.line(
                (round(x1), round(y1), round(x2), round(y2)),
                fill=fill,
                width=max(1, round(width)),
            )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        fill: str,
        width: float,
    ) -> None:
        self.draw.line(points, fill=fill, width=max(1, round(width)), joint="curve")

    def circle(
        self, x: float, y: float, radius: float, *, fill: str, stroke: str = "none"
    ) -> None:
        box = (x - radius, y - radius, x + radius, y + radius)
        self.draw.ellipse(box, fill=fill, outline=None if stroke == "none" else stroke)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int,
        fill: str = INK,
        anchor: str = "start",
        weight: str = "normal",
    ) -> None:
        font = self._font(size, weight)
        bbox = self.draw.textbbox((0, 0), value, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        self.draw.text((round(x), round(y - height / 2 - bbox[1])), value, font=font, fill=fill)

    def save(self, path: Path) -> None:
        self.image.save(path, format="PNG", optimize=True)


def _line_segments(
    rows: Sequence[RequestMetric], field: str, x_for: Any, y_for: Any
) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    for row in rows:
        value = getattr(row, field)
        if value is None:
            if segment:
                yield segment
                segment = []
            continue
        segment.append((x_for(row.request_number), y_for(value)))
    if segment:
        yield segment


def _nice_ceiling(value: float) -> float:
    if value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0**exponent
    scaled = value / base
    if scaled <= 1.0:
        step = 1.0
    elif scaled <= 2.0:
        step = 2.0
    elif scaled <= 2.5:
        step = 2.5
    elif scaled <= 5.0:
        step = 5.0
    else:
        # Preserve visual resolution for values just above 5×base.  Rounding
        # those all the way to 10×base wastes half the panel (for example,
        # 52.8 ms should use a 60 ms ceiling, not 100 ms).
        step = math.ceil(scaled)
    return step * base


def draw_chart(
    canvas: Any,
    rows: Sequence[RequestMetric],
    *,
    title: str,
    subtitle: str,
) -> dict[str, Any]:
    summary = aggregate_summary(rows)
    count = len(rows)
    cell_width = PLOT_WIDTH / count

    def x_for(number: int) -> float:
        return PLOT_LEFT + (number - 0.5) * cell_width

    canvas.text(WIDTH / 2, 46, title, size=34, anchor="middle", weight="bold")
    canvas.text(WIDTH / 2, 82, subtitle, size=19, fill=MUTED, anchor="middle")

    # Panel A: exact Top-k and actual runtime overlap rates.
    canvas.text(PLOT_LEFT, 120, "A  Per-request hit rate", size=25, weight="bold")
    canvas.text(
        PLOT_RIGHT + 24,
        120,
        "Micro aggregate",
        size=17,
        fill=MUTED,
        weight="bold",
    )
    heat_top = 150.0
    row_height = 48.0
    heat_rows = (
        ("Top-1", "top1_recall", "top1"),
        ("Top-3", "top3_recall", "top3"),
        ("Top-5", "top5_recall", "top5"),
        ("Runtime overlap", "runtime_hit_rate", "runtime"),
    )
    for row_index, (label, field, summary_key) in enumerate(heat_rows):
        y = heat_top + row_index * row_height
        canvas.text(PLOT_LEFT - 18, y + row_height / 2, label, size=19, anchor="end", weight="bold")
        for metric in rows:
            value = getattr(metric, field)
            tooltip = f"Request {metric.request_number}: {label} {_format_rate(value)}"
            canvas.rect(
                PLOT_LEFT + (metric.request_number - 1) * cell_width,
                y,
                cell_width,
                row_height - 4,
                fill=NA_FILL if value is None else _mix_with_white(TOP_COLORS[label], value),
                stroke=WHITE,
                stroke_width=0.6,
                tooltip=tooltip,
                hatch=value is None,
            )
        canvas.text(
            PLOT_RIGHT + 24,
            y + row_height / 2,
            summary[summary_key]["text"],
            size=18,
            weight="bold",
        )
    legend_y = heat_top + 4 * row_height + 20
    canvas.text(PLOT_LEFT, legend_y, "0%", size=16, fill=MUTED)
    gradient_width = 220.0
    for index in range(44):
        value = index / 43
        canvas.rect(
            PLOT_LEFT + 36 + index * gradient_width / 44,
            legend_y - 8,
            gradient_width / 44 + 0.5,
            16,
            fill=_mix_with_white(TOP_COLORS["Top-1"], value),
        )
    canvas.text(PLOT_LEFT + 270, legend_y, "100%", size=16, fill=MUTED)
    canvas.rect(PLOT_LEFT + 350, legend_y - 10, 24, 20, fill=NA_FILL, hatch=True)
    canvas.text(PLOT_LEFT + 384, legend_y, "N/A (no eligible target)", size=16, fill=MUTED)

    # Shared decile guides.
    for number in range(10, count + 1, 10):
        x = PLOT_LEFT + number * cell_width
        canvas.line(x, heat_top, x, 1260, fill=GRID_LIGHT, width=1.0)

    # Panel B: paired latency curves.
    panel_b_heading = 445.0
    panel_b_top = 485.0
    panel_b_bottom = 800.0
    canvas.text(
        PLOT_LEFT,
        panel_b_heading,
        "B  Paired drained tool-path wall time",
        size=25,
        weight="bold",
    )
    latency_values = [
        value
        for metric in rows
        for value in (metric.baseline_latency_ms, metric.pattern_latency_ms)
        if value is not None and value >= 0.0
    ]
    latency_max = _nice_ceiling(max(latency_values, default=1.0) * 1.05)

    def latency_y(value: float) -> float:
        return panel_b_bottom - value / latency_max * (panel_b_bottom - panel_b_top)

    for tick in range(6):
        value = latency_max * tick / 5
        y = latency_y(value)
        canvas.line(PLOT_LEFT, y, PLOT_RIGHT, y, fill=GRID, width=1.0)
        canvas.text(PLOT_LEFT - 18, y, _format_number(value), size=16, fill=MUTED, anchor="end")
    canvas.text(
        22,
        (panel_b_top + panel_b_bottom) / 2,
        "Drained wall (ms / trace)",
        size=17,
        fill=MUTED,
    )
    for field, color in (
        ("baseline_latency_ms", BASELINE_COLOR),
        ("pattern_latency_ms", PATTERN_COLOR),
    ):
        for segment in _line_segments(rows, field, x_for, latency_y):
            if len(segment) >= 2:
                canvas.polyline(segment, fill=color, width=2.2)
            for x, y in segment:
                canvas.circle(x, y, 2.8, fill=color)
    legend_x = PLOT_RIGHT - 355
    canvas.line(legend_x, panel_b_heading, legend_x + 42, panel_b_heading, fill=BASELINE_COLOR, width=4)
    canvas.text(legend_x + 52, panel_b_heading, "Demand-only", size=17, fill=MUTED)
    canvas.line(legend_x + 185, panel_b_heading, legend_x + 227, panel_b_heading, fill=PATTERN_COLOR, width=4)
    canvas.text(legend_x + 237, panel_b_heading, "Pattern-v2", size=17, fill=MUTED)
    latency = summary["latency"]
    canvas.text(
        PLOT_RIGHT + 24,
        panel_b_top + 30,
        f"Demand-only total: {_format_number(latency['baseline_total_ms'])} ms",
        size=17,
        fill=MUTED,
    )
    canvas.text(
        PLOT_RIGHT + 24,
        panel_b_top + 60,
        f"Pattern total: {_format_number(latency['pattern_total_ms'])} ms",
        size=17,
        fill=MUTED,
    )

    # Panel C: conservative factor, with 1x as the no-speculation reference.
    panel_c_heading = 870.0
    panel_c_top = 910.0
    panel_c_bottom = 1260.0
    canvas.text(PLOT_LEFT, panel_c_heading, "C  Conservative speedup vs demand-only", size=25, weight="bold")
    factors = [metric.speedup_factor for metric in rows if metric.speedup_factor is not None and metric.speedup_factor >= 0]
    if factors:
        distance = max(max(factors) - 1.0, 1.0 - min(factors), 0.05) * 1.12
    else:
        distance = 0.2
    factor_low = max(0.0, 1.0 - distance)
    factor_high = 1.0 + distance

    def factor_y(value: float) -> float:
        return panel_c_bottom - (value - factor_low) / (factor_high - factor_low) * (panel_c_bottom - panel_c_top)

    for tick in range(6):
        value = factor_low + (factor_high - factor_low) * tick / 5
        y = factor_y(value)
        canvas.line(PLOT_LEFT, y, PLOT_RIGHT, y, fill=GRID, width=1.0)
        canvas.text(PLOT_LEFT - 18, y, f"{value:.2f}x", size=16, fill=MUTED, anchor="end")
    baseline_y = factor_y(1.0)
    canvas.line(PLOT_LEFT, baseline_y, PLOT_RIGHT, baseline_y, fill=INK, width=2.4, dash="9 5")
    canvas.text(PLOT_RIGHT + 24, baseline_y, "1.00x baseline", size=17, weight="bold")
    bar_width = max(2.0, cell_width * 0.70)
    for metric in rows:
        value = metric.speedup_factor
        if value is None:
            continue
        x = x_for(metric.request_number)
        y = factor_y(value)
        color = POSITIVE_COLOR if value >= 1.0 else NEGATIVE_COLOR
        canvas.rect(
            x - bar_width / 2,
            min(y, baseline_y),
            bar_width,
            max(1.0, abs(y - baseline_y)),
            fill=color,
            tooltip=f"Request {metric.request_number}: {value:.3f}x",
        )
        if metric.speedup_min is not None and metric.speedup_max is not None:
            low_y = factor_y(max(factor_low, metric.speedup_min))
            high_y = factor_y(min(factor_high, metric.speedup_max))
            canvas.line(x, low_y, x, high_y, fill=INK, width=1.1)
            canvas.line(x - 3, low_y, x + 3, low_y, fill=INK, width=1.1)
            canvas.line(x - 3, high_y, x + 3, high_y, fill=INK, width=1.1)
    weighted = latency["weighted_speedup_factor"]
    median = latency["median_speedup_factor"]
    canvas.text(
        PLOT_RIGHT + 24,
        panel_c_top + 30,
        "Weighted total: " + ("N/A" if weighted is None else f"{weighted:.3f}x"),
        size=18,
        weight="bold",
    )
    canvas.text(
        PLOT_RIGHT + 24,
        panel_c_top + 62,
        "Request median: " + ("N/A" if median is None else f"{median:.3f}x"),
        size=17,
        fill=MUTED,
    )

    # Shared x axis.
    tick_numbers = sorted(set((1, *range(10, count + 1, 10), count)))
    for number in tick_numbers:
        if not 1 <= number <= count:
            continue
        x = x_for(number)
        canvas.line(x, panel_c_bottom, x, panel_c_bottom + 8, fill=INK, width=1.2)
        canvas.text(x, panel_c_bottom + 27, str(number), size=16, fill=MUTED, anchor="middle")
    canvas.text((PLOT_LEFT + PLOT_RIGHT) / 2, panel_c_bottom + 62, "Request number (one source trace per request)", size=19, anchor="middle", weight="bold")

    canvas.text(
        PLOT_LEFT,
        1366,
        "Top-k: whole-session OOF exact-URL recall. Runtime: overlap-producing reuse only. Gray: N/A, not a miss.",
        size=17,
        fill=MUTED,
    )
    canvas.text(
        PLOT_LEFT,
        1400,
        "Synthetic proxy (replay start through broker cleanup), not production E2E; Pattern-v2 includes charged predictor/selection overhead.",
        size=17,
        fill=MUTED,
    )
    return summary


def render(
    rows: Sequence[RequestMetric],
    *,
    svg_path: Path,
    png_path: Path,
    title: str,
    subtitle: str,
) -> dict[str, Any]:
    description = (
        f"Three-panel comparison for requests 1 through {len(rows)}. "
        "The first panel is a four-row hit-rate heatmap, the second compares "
        "paired demand-only and Pattern-v2 drained wall time, and the third shows "
        "conservative speedup relative to a 1.0x demand-only baseline. "
        "Gray cells represent unavailable denominators rather than misses."
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg = SvgCanvas(WIDTH, HEIGHT, title, description)
    summary = draw_chart(svg, rows, title=title, subtitle=subtitle)
    svg.save(svg_path)
    png = PillowCanvas(WIDTH, HEIGHT)
    draw_chart(png, rows, title=title, subtitle=subtitle)
    png.save(png_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument(
        "--title",
        default="Pattern-v2 prediction and latency by request",
    )
    parser.add_argument(
        "--subtitle",
        default="Whole-session grouped OOF; paired demand-only comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = complete_request_axis(
        load_request_metrics(args.input), args.request_count
    )
    prefix = args.output_prefix
    summary = render(
        rows,
        svg_path=prefix.with_suffix(".svg"),
        png_path=prefix.with_suffix(".png"),
        title=args.title,
        subtitle=args.subtitle,
    )
    print(
        json.dumps(
            {
                "svg": str(prefix.with_suffix(".svg")),
                "png": str(prefix.with_suffix(".png")),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
