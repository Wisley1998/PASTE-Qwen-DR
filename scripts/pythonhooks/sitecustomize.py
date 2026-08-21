import json
import os
import sys
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
