"""Executors that connect :class:`LiveToolBroker` to real tools."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from html import unescape
import json
import math
import re
import time
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, urlparse

from .invocation import Invocation


class SyncToolMapExecutor:
    """Run existing blocking ``tool.call(arguments)`` implementations safely.

    The repository's production ``Search`` and ``Visit`` tools expose this
    interface.  The adapter keeps their blocking HTTP/API work off the asyncio
    event loop; the surrounding ``LiveToolBroker`` remains the authoritative
    source of queue and capacity limits.
    """

    def __init__(
        self,
        tool_map: Mapping[str, Any],
        *,
        thread_workers: int,
        thread_name_prefix: str = "paste-live-tool",
    ) -> None:
        if thread_workers <= 0:
            raise ValueError("thread_workers must be positive")
        self._tool_map = dict(tool_map)
        self._pool = ThreadPoolExecutor(
            max_workers=thread_workers, thread_name_prefix=thread_name_prefix
        )
        self._closed = False

    async def __call__(self, invocation: Invocation) -> Any:
        if self._closed:
            raise RuntimeError("executor is closed")
        try:
            tool = self._tool_map[invocation.tool_name]
        except KeyError as exc:
            raise ValueError(f"unknown tool: {invocation.tool_name}") from exc
        call = getattr(tool, "call", None)
        if not callable(call):
            raise TypeError(
                f"tool {invocation.tool_name!r} does not provide call(arguments)"
            )
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._pool, partial(call, invocation.arguments)
        )
        try:
            # Shielding matters for physical capacity accounting: cancelling
            # an asyncio wrapper does not stop an already-running thread.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # The blocking call may already have started.  Do not let the
            # broker release its worker slot until that real call has drained.
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Do not block the event loop while already-running HTTP calls drain.
        await asyncio.to_thread(self._pool.shutdown, True, cancel_futures=True)

    async def __aenter__(self) -> "SyncToolMapExecutor":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class WikipediaLiveExecutor:
    """No-key live ``search``/``visit`` executor for controlled experiments.

    Search uses the public MediaWiki REST API by default.  Visit fetches the URL
    directly or routes it through ``r.jina.ai``.  Both operations are genuine
    network calls and therefore consume the shared broker worker that invokes
    them.  An aiohttp-like session may be injected for deterministic tests.
    """

    _TAG_RE = re.compile(r"<[^>]+>")
    _SPACE_RE = re.compile(r"\s+")

    # Public, JSON-serializable policy identifiers let experiment runners
    # freeze the exact retry contract without duplicating it.  Only idempotent
    # GETs are issued by this executor.
    HTTP_RETRY_POLICY_VERSION = "idempotent-get-v1"
    # aiohttp 3.12 retries a stale persistent connection once inside a single
    # ``ClientSession.get`` call.  That hidden attempt is outside our physical
    # attempt ledger, so formal live runs disable it fail-closed and let the
    # explicit loop below own every retry.
    HTTP_LIBRARY_RETRY_CONTROL_VERSION = "aiohttp-private-retry-connection-v1"
    HTTP_ATTEMPT_START_GATE_VERSION = "shared-per-tool-monotonic-v1"
    RETRYABLE_HTTP_STATUSES = (429, 500, 502, 503, 504)
    RETRYABLE_HTTP_EXCEPTION_TYPES = (
        "asyncio.TimeoutError",
        "ConnectionError",
        "aiohttp.ClientConnectionError",
        "aiohttp.ClientPayloadError",
    )

    def __init__(
        self,
        *,
        visit_mode: str = "direct",
        timeout_s: float = 20.0,
        max_results: int = 5,
        max_visit_urls: int = 3,
        max_response_bytes: int = 512 * 1024,
        visit_max_chars: int = 15_000,
        search_mode: str = "rest",
        wikipedia_api_template: str = "https://{language}.wikipedia.org/w/api.php",
        wikipedia_rest_search_template: str = (
            "https://{language}.wikipedia.org/w/rest.php/v1/search/page"
        ),
        bing_search_url: str = "https://www.bing.com/search",
        jina_base_url: str = "https://r.jina.ai/",
        max_http_attempts: int = 1,
        retry_backoff_s: float = 1.0,
        http_attempt_min_start_intervals_s: Mapping[str, float] | None = None,
        session: Any | None = None,
    ) -> None:
        if visit_mode not in {"direct", "jina"}:
            raise ValueError("visit_mode must be 'direct' or 'jina'")
        if search_mode not in {"rest", "action", "bing"}:
            raise ValueError("search_mode must be 'rest', 'action', or 'bing'")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_results <= 0 or max_visit_urls <= 0:
            raise ValueError("search and visit limits must be positive")
        if max_response_bytes <= 0 or visit_max_chars <= 0:
            raise ValueError("response limits must be positive")
        if (
            not isinstance(max_http_attempts, int)
            or isinstance(max_http_attempts, bool)
            or max_http_attempts <= 0
        ):
            raise ValueError("max_http_attempts must be a positive integer")
        if (
            isinstance(retry_backoff_s, bool)
            or not isinstance(retry_backoff_s, (int, float))
            or not math.isfinite(float(retry_backoff_s))
            or float(retry_backoff_s) < 0
        ):
            raise ValueError("retry_backoff_s must be a finite non-negative number")
        attempt_intervals: dict[str, float] = {}
        for raw_tool_name, raw_interval_s in dict(
            http_attempt_min_start_intervals_s or {}
        ).items():
            if raw_tool_name not in {"search", "visit"}:
                raise ValueError(
                    "HTTP-attempt start intervals only support search and visit"
                )
            if (
                isinstance(raw_interval_s, bool)
                or not isinstance(raw_interval_s, (int, float))
                or not math.isfinite(float(raw_interval_s))
                or float(raw_interval_s) < 0
            ):
                raise ValueError(
                    "HTTP-attempt start intervals must be finite non-negative numbers"
                )
            interval_s = float(raw_interval_s)
            if interval_s > 0:
                attempt_intervals[raw_tool_name] = interval_s
        self._visit_mode = visit_mode
        self._request_timeout_s = float(timeout_s)
        self._max_search_results = int(max_results)
        self._max_visit_urls = int(max_visit_urls)
        self._max_response_bytes = int(max_response_bytes)
        self._max_output_chars = int(visit_max_chars)
        self._search_mode = search_mode
        self._wikipedia_api_template = wikipedia_api_template
        self._wikipedia_rest_search_template = wikipedia_rest_search_template
        self._bing_search_url = bing_search_url
        self._jina_base_url = jina_base_url.rstrip("/") + "/"
        # The compatibility default is one attempt.  Formal experiments that
        # opt into controlled retry pass two explicitly, so older development
        # artifacts retain their original transport semantics.
        self._max_http_attempts = max_http_attempts
        self._retry_backoff_s = float(retry_backoff_s)
        # This gate is deliberately distinct from LiveToolBroker's job-start
        # gate.  A single broker job can issue multiple concurrent GETs and an
        # explicit retry issues another physical GET; all of those attempts
        # share this executor-local, per-tool monotonic start schedule.
        self._http_attempt_min_start_intervals_s = attempt_intervals
        self._http_attempt_gate_locks = {
            tool_name: asyncio.Lock() for tool_name in attempt_intervals
        }
        self._http_attempt_next_eligible_at: dict[str, float] = {}
        self._session = session
        self._owns_session = session is None
        self._http_library_retry_control_checked = False
        self._http_library_retry_disabled_effective = False
        self._http_library_name: str | None = None
        self._http_library_version: str | None = None
        self._closed = False

    @property
    def max_http_attempts(self) -> int:
        return self._max_http_attempts

    @property
    def retry_backoff_s(self) -> float:
        return self._retry_backoff_s

    @property
    def http_attempt_min_start_intervals_s(self) -> dict[str, float]:
        return dict(self._http_attempt_min_start_intervals_s)

    @property
    def http_library_retry_disabled_effective(self) -> bool:
        return self._http_library_retry_disabled_effective

    @property
    def http_library_name(self) -> str | None:
        return self._http_library_name

    @property
    def http_library_version(self) -> str | None:
        return self._http_library_version

    async def _wait_for_http_attempt_start(
        self, tool_name: str
    ) -> tuple[float, float]:
        """Reserve one real GET start and return monotonic start/wait evidence."""

        interval_s = self._http_attempt_min_start_intervals_s.get(tool_name, 0.0)
        if interval_s <= 0:
            return time.monotonic(), 0.0

        wait_started = time.monotonic()
        lock = self._http_attempt_gate_locks[tool_name]
        async with lock:
            now = time.monotonic()
            delay_s = max(
                0.0,
                self._http_attempt_next_eligible_at.get(tool_name, now) - now,
            )
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            started_monotonic_s = time.monotonic()
            self._http_attempt_next_eligible_at[tool_name] = (
                started_monotonic_s + interval_s
            )
        return started_monotonic_s, max(
            0.0, started_monotonic_s - wait_started
        )

    def _disable_aiohttp_internal_retry(self, session: Any) -> None:
        """Make one outer attempt equal exactly one physical HTTP GET.

        aiohttp exposes no public constructor flag for this behavior in the
        frozen 3.12 runtime.  We therefore validate the private shape before
        changing it and fail closed on real ``ClientSession`` objects if that
        shape ever changes.  Injected non-aiohttp fake sessions used by unit
        tests have no hidden library retry and are left alone.
        """

        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("WikipediaLiveExecutor requires aiohttp") from exc
        if not isinstance(session, aiohttp.ClientSession):
            self._http_library_retry_control_checked = True
            return
        if not hasattr(session, "_retry_connection"):
            raise RuntimeError(
                "unsupported aiohttp ClientSession: internal retry control is missing"
            )
        current = getattr(session, "_retry_connection")
        if not isinstance(current, bool):
            raise RuntimeError(
                "unsupported aiohttp ClientSession: internal retry control is not bool"
            )
        setattr(session, "_retry_connection", False)
        if getattr(session, "_retry_connection") is not False:
            raise RuntimeError("failed to disable aiohttp internal connection retry")
        self._http_library_retry_control_checked = True
        self._http_library_retry_disabled_effective = True
        self._http_library_name = "aiohttp"
        self._http_library_version = str(aiohttp.__version__)

    @staticmethod
    def _exception_type_name(exc: BaseException) -> str:
        cls = type(exc)
        return f"{cls.__module__}.{cls.__qualname__}"

    @staticmethod
    def _failure_status(
        exc: BaseException, response_status: int | None
    ) -> int | None:
        if response_status is not None:
            return response_status
        raw_status = getattr(exc, "status", None)
        if isinstance(raw_status, int) and not isinstance(raw_status, bool):
            return raw_status
        return None

    @classmethod
    def _retryable_http_failure(
        cls,
        exc: BaseException,
        *,
        response_status: int | None,
    ) -> bool:
        """Classify a failed GET without retrying semantic/parser errors."""

        # A response may have returned HTTP 200 before its body stream times
        # out or disconnects.  Those explicit transport failures take
        # precedence over the intermediate response status.  In contrast,
        # response-status exceptions remain governed solely by the whitelist
        # below.
        if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
            return True
        try:
            import aiohttp
        except ImportError:  # pragma: no cover - only relevant without aiohttp
            transient_aiohttp_types: tuple[type[BaseException], ...] = ()
        else:
            transient_aiohttp_types = (
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,
            )
        if isinstance(exc, transient_aiohttp_types):
            return True

        status = cls._failure_status(exc, response_status)
        if status is not None:
            return status in cls.RETRYABLE_HTTP_STATUSES
        return False

    @staticmethod
    def _attach_attempt_log(
        exc: BaseException, attempt_log: list[dict[str, Any]]
    ) -> None:
        """Attach diagnostics while preserving the final exception object."""

        try:
            setattr(
                exc,
                "paste_http_attempt_log",
                tuple(dict(entry) for entry in attempt_log),
            )
        except Exception:
            # Some third-party exception implementations may reject arbitrary
            # attributes.  Never replace the underlying transport exception.
            pass

    async def _ensure_session(self) -> Any:
        if self._closed:
            raise RuntimeError("executor is closed")
        created = False
        if self._session is None:
            try:
                import aiohttp
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("WikipediaLiveExecutor requires aiohttp") from exc
            timeout = aiohttp.ClientTimeout(total=self._request_timeout_s)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "PASTE-live-tool-experiment/1.0",
                    "Accept": "application/json,text/html,text/plain;q=0.9,*/*;q=0.8",
                },
            )
            created = True
        if not self._http_library_retry_control_checked:
            try:
                self._disable_aiohttp_internal_retry(self._session)
            except Exception:
                if created and self._session is not None:
                    await self._session.close()
                    self._session = None
                raise
        return self._session

    async def __call__(self, invocation: Invocation) -> dict[str, Any]:
        if invocation.tool_name == "search":
            return await self._search(invocation.arguments)
        if invocation.tool_name == "visit":
            return await self._visit(invocation.arguments)
        raise ValueError(f"unsupported public web tool: {invocation.tool_name}")

    def transport_plan(self, invocation: Invocation) -> dict[str, Any]:
        """Return fail-closed transport identity before dispatch.

        The broker records this plan for started attempts so a cancelled or
        failed HTTP job still identifies the external backend it was sent to.
        Response status and byte counts remain actual-only fields.
        """

        arguments = invocation.arguments
        if invocation.tool_name == "search":
            raw_queries = arguments.get("query")
            if isinstance(raw_queries, str):
                queries = [raw_queries]
            elif isinstance(raw_queries, list) and all(
                isinstance(item, str) for item in raw_queries
            ):
                queries = raw_queries
            else:
                return {}
            if not queries or any(not query.strip() for query in queries):
                return {}
            if self._search_mode == "bing":
                hosts = [str(urlparse(self._bing_search_url).hostname or "")]
                backend = "bing_html_search"
            elif self._search_mode == "rest":
                hosts = sorted(
                    {
                        str(
                            urlparse(
                                self._wikipedia_rest_search_template.format(
                                    language=(
                                        "zh"
                                        if self._contains_chinese(query)
                                        else "en"
                                    )
                                )
                            ).hostname
                            or ""
                        )
                        for query in queries
                    }
                )
                backend = "wikipedia_rest_search"
            else:
                hosts = sorted(
                    {
                        str(
                            urlparse(
                                self._wikipedia_api_template.format(
                                    language=(
                                        "zh"
                                        if self._contains_chinese(query)
                                        else "en"
                                    )
                                )
                            ).hostname
                            or ""
                        )
                        for query in queries
                    }
                )
                backend = "wikipedia_mediawiki_action"
            hosts = [host for host in hosts if host]
            if not hosts:
                return {}
            return {
                "backend": backend,
                "request_host": ",".join(hosts),
                "http_attempts": len(queries),
            }

        if invocation.tool_name == "visit":
            raw_urls = arguments.get("url")
            if isinstance(raw_urls, str):
                urls = [raw_urls]
            elif isinstance(raw_urls, list) and all(
                isinstance(item, str) for item in raw_urls
            ):
                urls = raw_urls
            else:
                return {}
            selected = urls[: self._max_visit_urls]
            if not selected:
                return {}
            if any(
                urlparse(url).scheme not in {"http", "https"}
                or not urlparse(url).netloc
                for url in selected
            ):
                return {}
            if self._visit_mode == "jina":
                hosts = [str(urlparse(self._jina_base_url).hostname or "")]
                backend = "r.jina.ai"
            else:
                hosts = sorted(
                    {str(urlparse(url).hostname or "") for url in selected}
                )
                backend = "direct_http"
            hosts = [host for host in hosts if host]
            if not hosts:
                return {}
            return {
                "backend": backend,
                "request_host": ",".join(hosts),
                "http_attempts": len(selected),
            }
        return {}

    @staticmethod
    def _contains_chinese(value: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in value)

    async def _search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        raw_queries = arguments.get("query")
        if isinstance(raw_queries, str):
            queries = [raw_queries]
        elif isinstance(raw_queries, list) and all(
            isinstance(item, str) for item in raw_queries
        ):
            queries = raw_queries
        else:
            raise ValueError("search arguments require a string or string-list 'query'")
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("search query must not be empty")
        batches = await asyncio.gather(
            *(
                self._search_one(query, query_index)
                for query_index, query in enumerate(queries)
            )
        )
        results = [entry for batch, _, _, _, _, _ in batches for entry in batch]
        statuses = [status for _, status, _, _, _, _ in batches]
        hosts = sorted({host for _, _, _, host, _, _ in batches})
        http_attempts = sum(attempts for _, _, _, _, attempts, _ in batches)
        attempt_log = [
            entry
            for _, _, _, _, _, entries in batches
            for entry in entries
        ]
        return {
            "tool": "search",
            "query": queries,
            "results": results,
            "_paste_transport": {
                "response_status": max(statuses) if statuses else None,
                "bytes_read": sum(size for _, _, size, _, _, _ in batches),
                "backend": (
                    "wikipedia_rest_search"
                    if self._search_mode == "rest"
                    else (
                        "bing_html_search"
                        if self._search_mode == "bing"
                        else "wikipedia_mediawiki_action"
                    )
                ),
                "request_host": ",".join(hosts),
                "http_attempts": http_attempts,
                "http_retries": http_attempts - len(batches),
                "http_attempt_log": attempt_log,
            },
        }

    async def _search_one(
        self, query: str, query_index: int
    ) -> tuple[
        list[dict[str, Any]],
        int,
        int,
        str,
        int,
        list[dict[str, Any]],
    ]:
        session = await self._ensure_session()
        language = "zh" if self._contains_chinese(query) else "en"
        if self._search_mode == "bing":
            endpoint = self._bing_search_url
            params = {
                "q": f"{query} site:{language}.wikipedia.org/wiki",
                "count": str(max(10, self._max_search_results * 3)),
            }
        elif self._search_mode == "rest":
            endpoint = self._wikipedia_rest_search_template.format(language=language)
            params = {"q": query, "limit": str(self._max_search_results)}
        else:
            endpoint = self._wikipedia_api_template.format(language=language)
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(self._max_search_results),
                "utf8": "1",
                "format": "json",
                "origin": "*",
            }
        attempt_log: list[dict[str, Any]] = []
        for attempt in range(1, self._max_http_attempts + 1):
            status: int | None = None
            started_monotonic_s, start_gate_wait_s = (
                await self._wait_for_http_attempt_start("search")
            )
            try:
                async with session.get(endpoint, params=params) as response:
                    status = int(response.status)
                    response.raise_for_status()
                    body = await response.read()
                    charset = response.charset or "utf-8"
                attempt_log.append(
                    {
                        "request_index": query_index,
                        "attempt": attempt,
                        "status": status,
                        "error_type": None,
                        "retried": False,
                        "started_monotonic_s": started_monotonic_s,
                        "start_gate_wait_s": start_gate_wait_s,
                        "retry_backoff_s": 0.0,
                    }
                )
                break
            except Exception as exc:
                failure_status = self._failure_status(exc, status)
                retry = (
                    attempt < self._max_http_attempts
                    and self._retryable_http_failure(
                        exc, response_status=failure_status
                    )
                )
                attempt_log.append(
                    {
                        "request_index": query_index,
                        "attempt": attempt,
                        "status": failure_status,
                        "error_type": self._exception_type_name(exc),
                        "retried": retry,
                        "started_monotonic_s": started_monotonic_s,
                        "start_gate_wait_s": start_gate_wait_s,
                        "retry_backoff_s": 0.0,
                    }
                )
                if not retry:
                    self._attach_attempt_log(exc, attempt_log)
                    raise
                # This delay intentionally remains inside the executor call:
                # the same broker worker stays occupied and service_s includes
                # failed transport time plus the fixed backoff.
                backoff_started = time.monotonic()
                await asyncio.sleep(self._retry_backoff_s)
                attempt_log[-1]["retry_backoff_s"] = max(
                    0.0, time.monotonic() - backoff_started
                )
        else:  # pragma: no cover - every iteration either breaks or raises
            raise AssertionError("HTTP retry loop ended without a result")

        decoded_body = body.decode(charset, errors="replace")
        assert status is not None
        if self._search_mode == "bing":
            results = self._parse_bing_results(
                decoded_body,
                query=query,
                query_index=query_index,
                language=language,
            )
            return (
                results,
                status,
                len(body),
                str(urlparse(endpoint).hostname or ""),
                len(attempt_log),
                attempt_log,
            )

        payload = json.loads(decoded_body)
        entries = (
            payload.get("pages", [])
            if self._search_mode == "rest"
            else payload.get("query", {}).get("search", [])
        )
        results = []
        for index, entry in enumerate(entries[: self._max_search_results], 1):
            title = str(entry.get("title") or "Untitled")
            snippet = self._plain_text(
                str(entry.get("excerpt") or entry.get("snippet") or "")
            )
            page_key = str(entry.get("key") or title.replace(" ", "_"))
            url = f"https://{language}.wikipedia.org/wiki/{quote(page_key)}"
            results.append(
                {
                    "query": query,
                    "query_index": query_index,
                    "rank": index,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
        return (
            results,
            status,
            len(body),
            str(urlparse(endpoint).hostname or ""),
            len(attempt_log),
            attempt_log,
        )

    @staticmethod
    def _decode_bing_redirect(value: str) -> str:
        href = unescape(value)
        parsed = urlparse(href)
        encoded = parse_qs(parsed.query).get("u", [""])[0]
        if encoded.startswith("a1"):
            raw = encoded[2:]
            try:
                padding = "=" * (-len(raw) % 4)
                decoded = base64.urlsafe_b64decode(raw + padding).decode("utf-8")
                if decoded.startswith(("http://", "https://")):
                    return decoded
            except (ValueError, UnicodeDecodeError):
                pass
        return href

    def _parse_bing_results(
        self,
        body: str,
        *,
        query: str,
        query_index: int,
        language: str,
    ) -> list[dict[str, Any]]:
        matches = re.findall(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.I | re.S,
        )
        decoded: list[tuple[str, str]] = []
        for href, raw_title in matches:
            url = self._decode_bing_redirect(href)
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            title = self._plain_text(raw_title) or "Untitled"
            decoded.append((title, url))

        preferred = [
            item
            for item in decoded
            if (urlparse(item[1]).hostname or "").endswith("wikipedia.org")
        ]
        selected = preferred or decoded
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for title, url in selected:
            if url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "query": query,
                    "query_index": query_index,
                    "rank": len(results) + 1,
                    "title": title,
                    "url": url,
                    "snippet": "",
                }
            )
            if len(results) >= self._max_search_results:
                break
        return results

    async def _visit(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        raw_urls = arguments.get("url")
        if isinstance(raw_urls, str):
            urls = [raw_urls]
        elif isinstance(raw_urls, list) and all(
            isinstance(item, str) for item in raw_urls
        ):
            urls = raw_urls
        else:
            raise ValueError("visit arguments require a string or string-list 'url'")
        if not urls or any(not url.strip() for url in urls):
            raise ValueError("visit URL must not be empty")
        selected = urls[: self._max_visit_urls]
        fetched = await asyncio.gather(
            *(
                self._visit_one(url, url_index)
                for url_index, url in enumerate(selected)
            )
        )
        pages = [page for page, _, _, _, _, _ in fetched]
        statuses = [status for _, status, _, _, _, _ in fetched]
        hosts = sorted({host for _, _, _, host, _, _ in fetched})
        http_attempts = sum(attempts for _, _, _, _, attempts, _ in fetched)
        attempt_log = [
            entry
            for _, _, _, _, _, entries in fetched
            for entry in entries
        ]
        return {
            "tool": "visit",
            "goal": str(arguments.get("goal") or ""),
            "pages": pages,
            "_paste_transport": {
                "response_status": max(statuses) if statuses else None,
                "bytes_read": sum(size for _, _, size, _, _, _ in fetched),
                "backend": (
                    "r.jina.ai"
                    if self._visit_mode == "jina"
                    else (
                        "wikipedia_rest_page"
                        if all(
                            urlparse(url).hostname in {"en.wikipedia.org", "zh.wikipedia.org"}
                            and urlparse(url).path.startswith("/wiki/")
                            for url in selected
                        )
                        else "direct_http"
                    )
                ),
                "request_host": ",".join(hosts),
                "http_attempts": http_attempts,
                "http_retries": http_attempts - len(fetched),
                "http_attempt_log": attempt_log,
            },
        }

    async def _visit_one(
        self, url: str, url_index: int
    ) -> tuple[
        dict[str, Any],
        int,
        int,
        str,
        int,
        list[dict[str, Any]],
    ]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"visit only accepts absolute HTTP(S) URLs: {url!r}")
        session = await self._ensure_session()
        if self._visit_mode == "jina":
            target = self._jina_base_url + url
        elif (
            parsed.hostname in {"en.wikipedia.org", "zh.wikipedia.org"}
            and parsed.path.startswith("/wiki/")
        ):
            page_key = parsed.path[len("/wiki/") :]
            target = (
                f"{parsed.scheme}://{parsed.netloc}/w/rest.php/v1/page/"
                f"{page_key}/html"
            )
        else:
            target = url
        attempt_log: list[dict[str, Any]] = []
        for attempt in range(1, self._max_http_attempts + 1):
            status: int | None = None
            started_monotonic_s, start_gate_wait_s = (
                await self._wait_for_http_attempt_start("visit")
            )
            try:
                async with session.get(target, allow_redirects=True) as response:
                    status = int(response.status)
                    response.raise_for_status()
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.content.iter_chunked(16 * 1024):
                        if not chunk:
                            continue
                        remaining = self._max_response_bytes - received
                        if remaining <= 0:
                            break
                        clipped = chunk[:remaining]
                        chunks.append(clipped)
                        received += len(clipped)
                        if received >= self._max_response_bytes:
                            break
                    charset = response.charset or "utf-8"
                    content_type = str(
                        response.headers.get("Content-Type", "")
                    ).lower()
                attempt_log.append(
                    {
                        "request_index": url_index,
                        "attempt": attempt,
                        "status": status,
                        "error_type": None,
                        "retried": False,
                        "started_monotonic_s": started_monotonic_s,
                        "start_gate_wait_s": start_gate_wait_s,
                        "retry_backoff_s": 0.0,
                    }
                )
                break
            except Exception as exc:
                failure_status = self._failure_status(exc, status)
                retry = (
                    attempt < self._max_http_attempts
                    and self._retryable_http_failure(
                        exc, response_status=failure_status
                    )
                )
                attempt_log.append(
                    {
                        "request_index": url_index,
                        "attempt": attempt,
                        "status": failure_status,
                        "error_type": self._exception_type_name(exc),
                        "retried": retry,
                        "started_monotonic_s": started_monotonic_s,
                        "start_gate_wait_s": start_gate_wait_s,
                        "retry_backoff_s": 0.0,
                    }
                )
                if not retry:
                    self._attach_attempt_log(exc, attempt_log)
                    raise
                backoff_started = time.monotonic()
                await asyncio.sleep(self._retry_backoff_s)
                attempt_log[-1]["retry_backoff_s"] = max(
                    0.0, time.monotonic() - backoff_started
                )
        else:  # pragma: no cover - every iteration either breaks or raises
            raise AssertionError("HTTP retry loop ended without a result")

        assert status is not None
        raw = b"".join(chunks).decode(charset, errors="replace")
        text = self._plain_text(raw) if "html" in content_type else raw.strip()
        if len(text) > self._max_output_chars:
            text = text[: self._max_output_chars] + "\n\n[Content truncated...]"
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
        title = self._plain_text(title_match.group(1)) if title_match else ""
        request_host = str(urlparse(target).hostname or "")
        return (
            {"url": url, "title": title, "content": text},
            status,
            received,
            request_host,
            len(attempt_log),
            attempt_log,
        )

    @classmethod
    def _plain_text(cls, value: str) -> str:
        without_tags = cls._TAG_RE.sub(" ", value)
        return cls._SPACE_RE.sub(" ", unescape(without_tags)).strip()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_session and self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> "WikipediaLiveExecutor":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


# Compatibility name for the first experimental draft.
PublicWebToolExecutor = WikipediaLiveExecutor
