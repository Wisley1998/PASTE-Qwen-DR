#!/usr/bin/env python3
"""Profile lightweight read-only Web speculation without process isolation.

Search is an untimed in-memory URL lookup. Web fetches are real aiohttp GETs
against a loopback HTTP server, including body read/decode/HTML stripping.
Speculation uses one persistent thread-local asyncio loop and an independent
connection pool. All speculative URLs are deliberately wrong, so AB/BA deltas
measure pure interference with a fixed AUTH arrival trace.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from threading import Event, Lock, Thread
import time
from typing import Any

import aiohttp
from aiohttp import web


SCRIPT = Path(__file__).resolve()
REPRO_ROOT = SCRIPT.parents[1]
DEFAULT_OUTPUT = REPRO_ROOT / "results" / "lightweight_web_spec_profile"
SCHEMA = "paste_repro.lightweight_web_spec_profile.v1"
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
T95 = (
    math.nan, 6.313752, 2.919986, 2.353363, 2.131847, 2.015048,
    1.943180, 1.894579, 1.859548, 1.833113, 1.812461, 1.795885,
    1.782288, 1.770933, 1.761310, 1.753050, 1.745884, 1.739607,
    1.734064, 1.729133, 1.725, 1.721, 1.717, 1.714, 1.711,
    1.708, 1.706, 1.703, 1.701, 1.699, 1.697,
)


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def parse_widths(raw: str) -> tuple[int, ...]:
    try:
        widths = tuple(sorted({int(part.strip()) for part in raw.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("widths must be comma-separated integers") from exc
    if not widths or widths[0] < 0:
        raise argparse.ArgumentTypeError("widths must be non-negative")
    return widths if 0 in widths else (0, *widths)


def inference(values: Sequence[float]) -> dict[str, float | int]:
    samples = list(map(float, values))
    mean = statistics.fmean(samples) if samples else 0.0
    if len(samples) < 2:
        return {"n": len(samples), "mean": mean, "se": math.inf, "upper_95": math.inf}
    se = statistics.stdev(samples) / math.sqrt(len(samples))
    df = len(samples) - 1
    critical = T95[df] if df < len(T95) else 1.644854
    return {"n": len(samples), "mean": mean, "se": se, "upper_95": mean + critical * se}


def make_payload(size: int) -> bytes:
    head, tail = b"<html><body>", b"</body></html>"
    unit = b"<p>Deterministic read-only local web fetch payload.</p>"
    body = bytearray(head)
    while len(body) + len(tail) < size:
        body.extend(unit[: size - len(body) - len(tail)])
    return bytes(body) + tail


class LocalServer:
    def __init__(self, service_ms: float, payload_bytes: int) -> None:
        self.delay = service_ms / 1000
        self.payload = make_payload(payload_bytes)
        self.ready, self.thread = Event(), Thread(target=self._main, daemon=True)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner: web.AppRunner | None = None
        self.port: int | None = None
        self.error: BaseException | None = None

    async def handle(self, request: web.Request) -> web.Response:
        await asyncio.sleep(self.delay)
        return web.Response(
            body=self.payload,
            content_type="text/html",
            headers={"Cache-Control": "no-store", "X-Key": request.match_info["key"]},
        )

    async def start_async(self) -> None:
        app = web.Application()
        app.router.add_get("/page/{key}", self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0, backlog=4096)
        await site.start()
        assert site._server is not None
        self.port = int(site._server.sockets[0].getsockname()[1])

    def _main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.start_async())
            self.ready.set()
            self.loop.run_forever()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            if self.runner:
                self.loop.run_until_complete(self.runner.cleanup())
            self.loop.close()

    def __enter__(self) -> "LocalServer":
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            raise RuntimeError("local HTTP server failed") from self.error
        return self

    @property
    def base_url(self) -> str:
        assert self.port is not None
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *_: object) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


async def fetch_raw(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()


def parse_page(payload: bytes) -> None:
    text = payload.decode("utf-8", "replace")
    if not SPACE_RE.sub(" ", TAG_RE.sub(" ", text)).strip():
        raise RuntimeError("empty parsed page")


async def fetch(session: aiohttp.ClientSession, url: str) -> None:
    parse_page(await fetch_raw(session, url))


class SpecLane:
    """One long-lived thread/loop. Wrong results never notify the AUTH loop."""

    def __init__(self, width: int) -> None:
        self.limit = max(1, width)
        self.ready, self.thread = Event(), Thread(target=self._main, daemon=True)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: aiohttp.ClientSession | None = None
        self.tasks: set[asyncio.Task[bytes]] = set()
        self.handles: set[asyncio.TimerHandle] = set()
        self.lock = Lock()
        self.counts = {"started": 0, "completed": 0, "failed": 0, "active": 0, "max_active": 0}
        self.error: BaseException | None = None

    def _main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            async def setup() -> None:
                connector = aiohttp.TCPConnector(
                    limit=self.limit,
                    limit_per_host=self.limit,
                    ttl_dns_cache=0,
                )
                self.session = aiohttp.ClientSession(connector=connector)

            self.loop.run_until_complete(setup())
            self.ready.set()
            self.loop.run_forever()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            if self.session:
                self.loop.run_until_complete(self.session.close())
            self.loop.close()

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            raise RuntimeError("spec lane failed") from self.error

    def arm(self, urls: Sequence[str], release_at: float) -> None:
        if not urls:
            return
        assert self.loop is not None

        def schedule() -> None:
            assert self.loop is not None

            def launch() -> None:
                assert self.session is not None
                for url in urls:
                    # Misses retain raw bytes only. Exact URL confirmation can
                    # parse on claim; wrong speculation never pays HTML parsing.
                    task = self.loop.create_task(fetch_raw(self.session, url))
                    self.tasks.add(task)
                    with self.lock:
                        self.counts["started"] += 1
                        self.counts["active"] += 1
                        self.counts["max_active"] = max(self.counts["max_active"], self.counts["active"])

                    def done(item: asyncio.Task[bytes]) -> None:
                        self.tasks.discard(item)
                        with self.lock:
                            self.counts["active"] -= 1
                            key = "failed" if item.cancelled() or item.exception() else "completed"
                            self.counts[key] += 1

                    task.add_done_callback(done)
                self.handles.discard(handle)

            handle = self.loop.call_at(release_at, launch)
            self.handles.add(handle)

        self.loop.call_soon_threadsafe(schedule)

    async def drain(self) -> None:
        assert self.loop is not None

        async def local() -> None:
            while self.handles or self.tasks:
                await asyncio.sleep(0.001)

        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(local(), self.loop))

    async def close(self) -> None:
        await self.drain()
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.loop.stop)
        await asyncio.to_thread(self.thread.join, 10)

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {key: value for key, value in self.counts.items() if key != "active"}


async def auth_call(session: aiohttp.ClientSession, url: str, batch: int, release: float) -> tuple[int, float, float, float]:
    await asyncio.sleep(max(0, release - time.monotonic()))
    first = time.monotonic()
    await fetch(session, url)
    return batch, release, first, time.monotonic()


async def run_cell(args: argparse.Namespace, base_url: str, width: int, rep: int, label: str) -> dict[str, Any]:
    lane = SpecLane(width)
    lane.start()
    connector = aiohttp.TCPConnector(
        limit=args.authority_concurrency,
        limit_per_host=args.authority_concurrency,
        ttl_dns_cache=0,
    )
    completions: list[tuple[int, float, float, float]] = []
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            period = (args.lead_ms + args.service_ms + args.gap_ms) / 1000
            origin = time.monotonic() + 0.05
            tasks = []
            for batch in range(args.batches):
                spec_release = origin + batch * period
                auth_release = spec_release + args.lead_ms / 1000
                # Simulated search result reuse: unique fetch URLs, no timed search I/O.
                lane.arm(
                    [f"{base_url}/page/spec-{rep}-{label}-{batch}-{i}" for i in range(width)],
                    spec_release,
                )
                tasks.extend(
                    asyncio.create_task(
                        auth_call(session, f"{base_url}/page/auth-{rep}-{label}-{batch}-{i}", batch, auth_release)
                    )
                    for i in range(args.authority_concurrency)
                )
            completions = await asyncio.gather(*tasks)
        await lane.drain()  # outside AUTH measurements
    finally:
        await lane.close()

    latencies = [(terminal - release) * 1000 for _, release, _, terminal in completions]
    lags = [(first - release) * 1000 for _, release, first, _ in completions]
    walls = []
    for batch in range(args.batches):
        rows = [row for row in completions if row[0] == batch]
        walls.append((max(row[3] for row in rows) - rows[0][1]) * 1000)
    return {
        "width": width,
        "authority_targets": len(completions),
        "authority_mean_latency_ms": statistics.fmean(latencies),
        "authority_p95_latency_ms": percentile(latencies, 0.95),
        "authority_p99_latency_ms": percentile(latencies, 0.99),
        "authority_mean_first_run_lag_ms": statistics.fmean(lags),
        "authority_mean_batch_wall_ms": statistics.fmean(walls),
        "speculation": lane.stats(),
        "simulated_search_cache_hits": width * args.batches,
    }


def evaluate(pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    mean = inference([p["treatment"]["authority_mean_latency_ms"] - p["baseline"]["authority_mean_latency_ms"] for p in pairs])
    p95 = inference([p["treatment"]["authority_p95_latency_ms"] - p["baseline"]["authority_p95_latency_ms"] for p in pairs])
    wall = inference([
        100 * (p["treatment"]["authority_mean_batch_wall_ms"] - p["baseline"]["authority_mean_batch_wall_ms"])
        / p["baseline"]["authority_mean_batch_wall_ms"]
        for p in pairs
    ])
    enough = len(pairs) >= args.minimum_repetitions
    passed = bool(
        enough and mean["upper_95"] <= args.mean_margin_ms
        and p95["upper_95"] <= args.p95_margin_ms
        and wall["upper_95"] <= args.wall_margin_percent
    )
    return {
        "decision": "pass" if passed else ("insufficient_repetitions" if not enough else "fail"),
        "authority_mean_latency_diff_ms": mean,
        "authority_p95_latency_diff_ms": p95,
        "authority_batch_wall_diff_percent": wall,
        "spec_started": sum(p["treatment"]["speculation"]["started"] for p in pairs),
        "spec_failures": sum(p["treatment"]["speculation"]["failed"] for p in pairs),
        "pairs": pairs,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    width_results = {}
    with LocalServer(args.service_ms, args.payload_bytes) as server:
        for width in args.widths:
            pairs = []
            for rep in range(args.repetitions):
                order = ("baseline", "treatment") if rep % 2 == 0 else ("treatment", "baseline")
                cells = {}
                for label in order:
                    cells[label] = await run_cell(args, server.base_url, 0 if label == "baseline" else width, rep, f"w{width}-{label}")
                pairs.append({"repetition": rep, "order": list(order), **cells})
            width_results[str(width)] = evaluate(pairs, args)

    safe, contiguous = 0, True
    for width in args.widths:
        if width == 0:
            continue
        if contiguous and width_results[str(width)]["decision"] == "pass":
            safe = width
        else:
            contiguous = False
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(SCRIPT), "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()},
        "environment": {"python": sys.version, "aiohttp": aiohttp.__version__, "cpu_count": os.cpu_count()},
        "configuration": {
            key: getattr(args, key) for key in (
                "widths", "repetitions", "minimum_repetitions", "authority_concurrency",
                "batches", "service_ms", "lead_ms", "gap_ms", "payload_bytes",
                "mean_margin_ms", "p95_margin_ms", "wall_margin_percent",
            )
        },
        "workload": {
            "search": "simulated untimed in-memory URL result",
            "fetch": "real loopback aiohttp GET + body parsing",
            "speculation": "persistent thread asyncio raw-fetch lane, independent connector, all wrong; parse only on exact claim",
            "authority": "direct parent-loop aiohttp, fixed arrivals, independent connector",
        },
        "selection": {
            "rule": "largest contiguous positive K passing mean/p95/wall one-sided 95% UCB margins",
            "max_safe_speculative_parallelism": safe,
        },
        "width_results": width_results,
    }


def report(result: dict[str, Any]) -> str:
    c = result["configuration"]
    lines = [
        "# Lightweight Web Speculation Capacity Profile", "",
        f"AUTH concurrency `{c['authority_concurrency']}`, local service `{c['service_ms']} ms`, payload `{c['payload_bytes']} B`, paired R=`{c['repetitions']}`.", "",
        f"Selected maximum safe speculative parallelism: **K={result['selection']['max_safe_speculative_parallelism']}**.", "",
        "| K | Decision | Mean diff ms (UCB) | p95 diff ms (UCB) | Batch wall diff % (UCB) | Failures |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for width, row in result["width_results"].items():
        mean, p95, wall = row["authority_mean_latency_diff_ms"], row["authority_p95_latency_diff_ms"], row["authority_batch_wall_diff_percent"]
        lines.append(
            f"| {width} | {row['decision']} | {mean['mean']:+.4f} ({mean['upper_95']:+.4f}) | "
            f"{p95['mean']:+.4f} ({p95['upper_95']:+.4f}) | {wall['mean']:+.4f} ({wall['upper_95']:+.4f}) | {row['spec_failures']} |"
        )
    lines += ["", "This is a host-specific all-wrong loopback profile; it does not certify remote backend quotas.", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--widths", type=parse_widths, default=parse_widths("0,1,2,4,8,16,32,64"))
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--minimum-repetitions", type=int, default=8)
    parser.add_argument("--authority-concurrency", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--service-ms", type=float, default=20.0)
    parser.add_argument("--lead-ms", type=float, default=5.0)
    parser.add_argument("--gap-ms", type=float, default=5.0)
    parser.add_argument("--payload-bytes", type=int, default=65536)
    parser.add_argument("--mean-margin-ms", type=float, default=0.10)
    parser.add_argument("--p95-margin-ms", type=float, default=0.50)
    parser.add_argument("--wall-margin-percent", type=float, default=0.10)
    args = parser.parse_args()
    for name in ("repetitions", "minimum_repetitions", "authority_concurrency", "batches", "payload_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_repetitions > args.repetitions:
        parser.error("--minimum-repetitions cannot exceed --repetitions")
    if args.service_ms <= 0 or min(args.lead_ms, args.gap_ms, args.mean_margin_ms, args.p95_margin_ms, args.wall_margin_percent) < 0:
        parser.error("service must be positive and timing/margin values non-negative")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "REPORT.md").write_text(report(result))
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
