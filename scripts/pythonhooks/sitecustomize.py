import os
import sys


def _fail_strict_python_startup(message):
    """Abort instead of letting ``site`` silently ignore this safety failure."""

    encoded = (f"[sitecustomize] strict safe-path failure: {message}\n").encode(
        "utf-8", errors="replace"
    )
    try:
        os.write(2, encoded)
    finally:
        os._exit(86)


def _enforce_strict_safe_path():
    """Make the Python 3.10 ``-m`` working directory non-importable.

    The pinned vLLM environment uses Python 3.10, which predates ``-P`` and
    does not implement ``PYTHONSAFEPATH``.  The strict launcher therefore
    enters a fresh, empty, non-writable directory and supplies its absolute
    path in ``VLLM_SAFE_WORKING_DIR``.  This startup hook verifies that
    contract and filters that directory (and the empty-string alias for it)
    from all filesystem import searches before ``-m vllm...`` is imported.
    EngineCore children repeat the same check when they import sitecustomize.
    """

    configured = os.environ.get("VLLM_SAFE_WORKING_DIR")
    if not configured:
        return
    if not os.path.isabs(configured):
        _fail_strict_python_startup("VLLM_SAFE_WORKING_DIR is not absolute")
    expected = os.path.realpath(configured)
    actual = os.path.realpath(os.getcwd())
    if actual != expected:
        _fail_strict_python_startup(
            f"working directory {actual!r} differs from frozen {expected!r}"
        )
    try:
        if any(True for _ in os.scandir(actual)):
            _fail_strict_python_startup("frozen working directory is not empty")
        if os.stat(actual).st_mode & 0o222:
            _fail_strict_python_startup("frozen working directory is writable")
    except OSError as exc:
        _fail_strict_python_startup(f"cannot verify frozen working directory: {exc}")

    # CPython 3.10 inserts the ``-m``/``-c`` path[0] *after* sitecustomize has
    # run.  Removing its current value is therefore insufficient.  Replace
    # only the standard filesystem finder with a transparent wrapper that
    # filters the frozen working directory on every subsequent import.
    from importlib.machinery import PathFinder

    def _filtered_search_path(search_path):
        filtered = []
        for entry in search_path:
            try:
                raw_entry = os.fspath(entry)
            except TypeError:
                # Preserve the standard finder's treatment of non-path
                # entries.  They cannot name the frozen working directory.
                filtered.append(entry)
                continue
            if isinstance(raw_entry, bytes):
                candidate = (
                    os.fsencode(actual)
                    if raw_entry == b""
                    else os.path.realpath(raw_entry)
                )
                is_working_directory = candidate == os.fsencode(actual)
            else:
                candidate = (
                    actual if raw_entry == "" else os.path.realpath(raw_entry)
                )
                is_working_directory = candidate == actual
            if not is_working_directory:
                filtered.append(entry)
        return filtered

    class _StrictPathFinder:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            search_path = sys.path if path is None else path
            return PathFinder.find_spec(
                fullname, _filtered_search_path(search_path), target
            )

        @classmethod
        def find_distributions(cls, context=None):
            """Delegate metadata discovery after applying the same CWD guard.

            ``importlib.metadata`` discovers distributions through a separate
            method on meta-path finders.  Replacing ``PathFinder`` without
            preserving that method makes every installed ``*.dist-info``
            package disappear; delegating the original context unchanged
            would instead let a late-inserted CWD forge package metadata.
            """

            from importlib.metadata import DistributionFinder

            original_context = context or DistributionFinder.Context()
            context_values = dict(vars(original_context))
            context_values["path"] = _filtered_search_path(
                original_context.path
            )
            filtered_context = DistributionFinder.Context(**context_values)
            return PathFinder.find_distributions(filtered_context)

    replaced = False
    meta_path = []
    for finder in sys.meta_path:
        if finder is PathFinder:
            meta_path.append(_StrictPathFinder)
            replaced = True
        else:
            meta_path.append(finder)
    if not replaced:
        _fail_strict_python_startup("standard PathFinder is absent")
    sys.meta_path[:] = meta_path
    os.environ["PASTE_STRICT_SAFE_PATH_ENFORCED"] = "1"
    os.environ["PASTE_STRICT_CWD_IMPORT_FILTER_ENFORCED"] = "1"


_enforce_strict_safe_path()

import json
import time
from functools import wraps
from pathlib import Path


def _append_jsonl(path_str, payload):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _install_vllm_swap_patch():
    if os.getenv("VLLM_TRACE_SWAP_PATCH") != "1":
        return

    swap_log_path = os.getenv("VLLM_TRACE_SWAP_LOG")
    if not swap_log_path:
        return

    try:
        from vllm.core.scheduler import Scheduler
    except Exception as exc:  # pragma: no cover
        print(f"[sitecustomize] failed to import vllm scheduler: {exc}", file=sys.stderr)
        return

    if getattr(Scheduler, "_vllm_trace_swap_patch_installed", False):
        return

    def emit_event(op, seq_group, mapping_len, duration_s, ok, error=None):
        payload = {
            "ts": time.time(),
            "pid": os.getpid(),
            "op": op,
            "mapping_len": int(mapping_len),
            "duration_s": float(duration_s),
            "ok": bool(ok),
        }
        request_id = getattr(seq_group, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id
        if error:
            payload["error"] = error
        _append_jsonl(swap_log_path, payload)

    def patch_method(op_name, method_name):
        original = getattr(Scheduler, method_name)

        @wraps(original)
        def wrapped(self, seq_group, blocks_to_swap):
            before_len = len(blocks_to_swap)
            start_s = time.time()
            ok = False
            error = None
            try:
                result = original(self, seq_group, blocks_to_swap)
                ok = True
                return result
            except Exception as exc:  # pragma: no cover
                error = repr(exc)
                raise
            finally:
                added = max(0, len(blocks_to_swap) - before_len)
                emit_event(
                    op=op_name,
                    seq_group=seq_group,
                    mapping_len=added,
                    duration_s=time.time() - start_s,
                    ok=ok,
                    error=error,
                )

        setattr(Scheduler, method_name, wrapped)

    patch_method("swap_in", "_swap_in")
    patch_method("swap_out", "_swap_out")
    Scheduler._vllm_trace_swap_patch_installed = True


try:
    _install_vllm_swap_patch()
except Exception as exc:  # pragma: no cover
    print(f"[sitecustomize] unexpected error: {exc}", file=sys.stderr)

try:
    import sched_policy_patch

    sched_policy_patch.install()
except Exception as exc:  # pragma: no cover
    print(f"[sitecustomize] scheduler policy patch error: {exc}", file=sys.stderr)
