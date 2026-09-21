#!/usr/bin/env python3
"""Paired authoritative audit using the production broker and HTTP executor.

The HTTP fixture is immutable except in the explicit dynamic-content case.
This is an executor audit, not an autonomous-agent quality/latency benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiohttp import web
from paste_repro.invocation import Invocation
from paste_repro.live_broker import LiveToolBroker
from paste_repro.live_executor import WikipediaLiveExecutor


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def audit():
    state = {"body": "fixed authoritative content", "fail": False, "freshness": True}
    started, release = asyncio.Event(), asyncio.Event()
    release.set()

    async def page(request):
        started.set()
        await release.wait()
        if state["fail"]:
            return web.Response(status=503, text="injected failure")
        headers = {"Cache-Control": "public, max-age=60"} if state["freshness"] else {}
        if state.get("vary"):
            headers["Vary"] = state["vary"]
        return web.Response(text=state["body"], headers=headers)

    app = web.Application()
    app.router.add_get('/{name}', page)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/page"
    cases = []
    try:
        for name in ["completed", "inflight", "wrong_args", "cross_session", "expired", "failure", "missing_freshness", "dynamic_fallback", "fixed_encoding_vary", "cookie_vary"]:
            pair = []
            for enabled in (False, True):
                state.update(body="fixed authoritative content", fail=False, freshness=name not in {"missing_freshness", "dynamic_fallback"})
                state["vary"] = {"fixed_encoding_vary": "Accept-Encoding", "cookie_vary": "Cookie"}.get(name, "")
                started.clear()
                release.set()
                now = [time.monotonic()]
                async with WikipediaLiveExecutor() as executor:
                    broker = LiveToolBroker(executor, require_reuse_validity=True, max_workers=2, max_speculative_workers=int(enabled), ttl_s=30, clock=lambda: now[0] + time.monotonic() - initial)
                    initial = time.monotonic()
                    arrival = time.monotonic()
                    invocation = Invocation("visit", {"url": url, "goal": "read"})
                    try:
                        if enabled:
                            candidate = Invocation("visit", {"url": url, "goal": "wrong"}) if name == "wrong_args" else invocation
                            state["fail"] = name == "failure"
                            if name == "inflight":
                                release.clear()
                            await broker.speculate(candidate, session_id="other" if name == "cross_session" else "task")
                            await asyncio.wait_for(started.wait(), 5)
                            if name != "inflight":
                                while broker.snapshot()["counts"]["running_speculative"]:
                                    await asyncio.sleep(0.001)
                            assert not broker.authoritative_state, "speculation entered history"
                            if name == "expired":
                                now[0] += 31
                                await broker.sweep()
                        state["fail"] = False
                        if name == "dynamic_fallback":
                            state["body"] = "changed authoritative content"
                        call = asyncio.create_task(broker.authoritative(invocation, session_id="task"))
                        await asyncio.sleep(0)
                        release.set()
                        result = await call
                        completed = time.monotonic()
                        history = [{"tool": r.invocation.tool_name, "args": r.invocation.arguments, "result": r.result} for r in broker.authoritative_state]
                        assert len(history) == 1
                    finally:
                        release.set()
                        await broker.close()
                    pair.append({"enabled": enabled, "source": result.source, "history": history, "state_sha256": digest(state["body"]), "e2e_s": completed - arrival, "resources": broker.resource_summary(), "jobs": broker.tool_records()})
            expected_reuse = name in {"completed", "inflight", "fixed_encoding_vary"}
            assert (pair[1]["source"] in {"reused", "promoted_inflight"}) == expected_reuse, (name, pair[1]["source"])
            difference = int(pair[0]["history"] != pair[1]["history"] or pair[0]["state_sha256"] != pair[1]["state_sha256"])
            assert difference == 0, name
            cases.append({"case": name, "differences": difference, "off": pair[0], "on": pair[1]})
    finally:
        await server.cleanup()
    return {"schema": "paste.correctness_audit.v1", "scope": "fixed authoritative calls; real local HTTP; production broker/executor; unknown HTTP freshness falls back; reuse assumes the origin freshness contract; not arbitrary dynamic-web equality or agent quality", "cases": cases, "differences": sum(c["differences"] for c in cases)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(audit())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({"cases": len(report["cases"]), "differences": report["differences"], "output": str(args.output)}))
