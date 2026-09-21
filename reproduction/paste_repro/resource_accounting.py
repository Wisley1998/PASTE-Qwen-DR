"""Physical tool cost, separate from speculative benefit and task latency.

Unknown counters remain null. Payload retention measures serialized private
results, not Python heap/RSS; worker occupancy is never called CPU time.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Iterable, Mapping


current_usage: ContextVar[dict[str, Any] | None] = ContextVar("paste_usage", default=None)


def summarize_resources(records: Iterable[Mapping[str, Any]], *, now: float) -> dict[str, Any]:
    rows = [r for r in records if r.get("admitted")]
    simulated = [r for r in rows if r.get("backend") == "simulated_search"]
    started = [r for r in rows if r.get("started_at") is not None
               and r.get("backend") != "simulated_search"]
    # A candidate promoted before dispatch is ordinary physical work.
    speculative = [r for r in started if r.get("dispatch_lane") == "speculative"]

    def measured_sum(key: str) -> float | None:
        values = [r.get(key) for r in started]
        return sum(values) if all(v is not None for v in values) else None

    events = []
    byte_seconds = 0.0
    for row in rows:
        size = row.get("retained_result_bytes") or 0
        begin = row.get("finished_at")
        end = row.get("cleanup_at") or now
        if size and begin is not None:
            byte_seconds += size * max(0.0, end - begin)
            events.extend([(begin, size), (end, -size)])
    current = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        current += delta
        peak = max(peak, current)
    return {
        "physical_calls": len(started),
        "simulated_search_calls": len(simulated),
        "simulated_search_worker_s": sum(r.get("service_s") or 0 for r in simulated),
        "worker_occupancy_s": measured_sum("service_s"),
        "cpu_core_s": measured_sum("cpu_core_s"),
        "http_requests": measured_sum("http_requests"),
        "http_attempts": measured_sum("http_attempts"),
        "http_response_body_bytes": measured_sum("bytes_read"),
        "http_request_bytes": measured_sum("bytes_written"),
        "private_result_peak_bytes": peak,
        "private_result_byte_s": byte_seconds,
        "useful_speculative_calls": sum(bool(r.get("committed")) for r in speculative),
        "unused_speculative_calls": sum(not r.get("committed") for r in speculative),
        "unused_speculative_worker_s": sum(r.get("service_s") or 0.0 for r in speculative if not r.get("committed")),
        "cleanup_complete": all(r.get("cleanup_at") is not None for r in rows),
        "measurement_scope": "physical executor; serialized private result payload; sent HTTP requests include redirects; body bytes delivered after decompression exclude headers and unread/cancelled transport buffering",
    }


def per_task_resources(records: Iterable[Mapping[str, Any]], *, now: float) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row["session_id"]), []).append(row)
    return {task: summarize_resources(rows, now=now) for task, rows in groups.items()}
