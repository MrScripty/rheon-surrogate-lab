"""Bounded serial Linux launcher for the lab's already admitted native binary.

The caller owns scientific admission. This owner controls one native session and
its descendants; it does not import models, register Runs or authenticate apps.
Supported caller: main Python thread, no other child work, no concurrent call.
Memory is observed process high-water, not OS or adversarial containment.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time

REQUEST_LIMIT = 1024 * 1024
STDOUT_LIMIT = 512 * 1024
STDERR_LIMIT = 64 * 1024
RSS_LIMIT = 4 * 1024**3
MODE_LIMITS = {"--identity": 10.0, "--evaluate": 60.0}
_CALL_LOCK = threading.Lock()
# On an unkillable-process failure retain stable handles and exclusive ownership.
# The caller must stop; a later call cannot silently inherit unresolved work.
_UNRESOLVED_OWNER = None


class NativeProcessError(RuntimeError):
    def __init__(self, code, detail, measurement):
        self.code = code
        self.measurement = measurement
        super().__init__(code + ": " + detail)


def strict_json(raw):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate JSON field: " + name)
            result[name] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON number: " + value)

    def floating(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("overflowing JSON number")
        return parsed

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    if type(result) is not dict:
        raise ValueError("native JSON result must be an object")
    return result


def _parent_peak():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _proc(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        peak = 0
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                peak = max(peak, int(line.split()[1]) * 1024)
        return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": fields[19], "rss_bytes": peak}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _children(pid):
    result = set()
    try:
        tids = list(Path(f"/proc/{pid}/task").iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return result
    children_files_supported = False
    for tid in tids:
        try:
            result.update(int(value) for value in (tid / "children").read_text().split())
            children_files_supported = True
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not children_files_supported:
        # CONFIG_CHECKPOINT_RESTORE is not enabled on every qualified Linux
        # kernel. PPID in procfs stat supplies the same process relationship;
        # read process metadata only, never unrelated files or process memory.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[1]) == pid:
                    result.add(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
    return result


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    if value is None:
        result = ctypes.c_int()
        status = libc.prctl(37, ctypes.byref(result), 0, 0, 0)  # PR_GET_CHILD_SUBREAPER
        if status != 0:
            raise OSError(ctypes.get_errno(), "get subreaper")
        return result.value
    if libc.prctl(36, int(value), 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "set subreaper")


def _environment():
    # No bearer/session/config environment is forwarded to the native process.
    env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "TZ", "TMPDIR") if name in os.environ}
    env.update(RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONNOUSERSITE="1", RUST_BACKTRACE="0")
    return env


def native_call(binary, mode, request, deadline, timeout):
    """Return (strict JSON object, lifecycle measurement), or NativeProcessError.

    deadline is an absolute time.monotonic value. timeout includes cleanup and is
    at most 10s for identity/60s for evaluate. A bounded portion is reserved for
    terminate/kill/drain. No retries. ECHILD is global only under the enforced
    exclusive caller/no-existing-child boundary; unrelated work is never reaped.
    """
    global _UNRESOLVED_OWNER
    started = time.monotonic()
    m = {"schema": "rheon-global-pressure-native-process-v1", "mode": mode,
         "status": "failed", "failure_code": None, "wall_s": 0.0,
         "parent_peak_rss_bytes": _parent_peak(), "child_peak_rss_bytes": 0,
         "returncode": None, "leader_pid": None, "leader_start_ticks": None,
         "process_group": None, "spawn_closed": True, "drained": False,
         "echild": False, "group_empty": False, "streams_closed": False,
         "owned_processes": [], "reaped_processes": [], "survivors": [],
         "signals": [], "escaped_descendant_observed": False,
         "subreaper_previous": None, "subreaper_restored": False, "ownership_released": False,
         "cleanup_errors": [],
         "environment_threads": 1, "rss_policy": "observed process high-water; not OS or aggregate containment"}
    streams = {name: bytearray() for name in ("stdout", "stderr")}
    stream_eof = {name: False for name in streams}
    stream_truncated = {name: False for name in streams}
    encoded = b""
    proc = None
    selector = None
    old_subreaper = None
    locked = False
    owned = {}
    pidfds = {}
    error = None
    hard_end = None
    cleanup_started = None
    offset = 0
    signal_records = set()

    def fail(code, detail):
        nonlocal error, cleanup_started
        if error is None:
            error = (code, detail)
            cleanup_started = time.monotonic()

    def close_stream(file):
        if file is None or file.closed:
            return
        if selector is not None:
            try:
                selector.unregister(file)
            except KeyError:
                pass
        file.close()

    def observe():
        pending = list(_children(os.getpid()))
        seen = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            info = _proc(pid)
            if info is None:
                continue
            previous = owned.get(pid)
            if previous is not None and previous["start_ticks"] != info["start_ticks"]:
                fail("ownership_changed", "PID identity changed during owned work")
                continue
            if previous is None:
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                confirmed = _proc(pid)
                if confirmed is None or confirmed["start_ticks"] != info["start_ticks"]:
                    os.close(fd)
                    continue
                pidfds[pid] = fd
                owned[pid] = {name: info[name] for name in ("pid", "ppid", "pgid", "sid", "start_ticks")}
            m["child_peak_rss_bytes"] = max(m["child_peak_rss_bytes"], info["rss_bytes"])
            if pid != proc.pid and (info["pgid"] != proc.pid or info["sid"] != proc.pid):
                m["escaped_descendant_observed"] = True
                fail("descendant_escape", "owned descendant left native session/group")
            pending.extend(_children(pid))
        m["parent_peak_rss_bytes"] = max(m["parent_peak_rss_bytes"], _parent_peak())
        if max(m["parent_peak_rss_bytes"], m["child_peak_rss_bytes"]) > RSS_LIMIT:
            fail("rss_limit", "observed process high-water exceeds4GiB policy")

    def reap():
        while True:
            try:
                pid, status, usage = os.wait4(-1, os.WNOHANG)
            except ChildProcessError:
                m["echild"] = True
                return
            if pid == 0:
                m["echild"] = False
                return
            returncode = os.waitstatus_to_exitcode(status)
            peak = int(usage.ru_maxrss) * 1024
            m["child_peak_rss_bytes"] = max(m["child_peak_rss_bytes"], peak)
            if pid not in owned:
                # A short-lived adopted child can disappear before procfs scan.
                # Exclusive subreaper ownership still binds this reaped child.
                owned[pid] = {"pid": pid, "ppid": os.getpid(), "pgid": None, "sid": None, "start_ticks": None}
            m["reaped_processes"].append({"pid": pid, "returncode": returncode, "peak_rss_bytes": peak})
            if pid == proc.pid:
                proc.returncode = returncode
                m["returncode"] = returncode
            fd = pidfds.pop(pid, None)
            if fd is not None:
                os.close(fd)
            if peak > RSS_LIMIT:
                fail("rss_limit", "reaped child high-water exceeds4GiB policy")

    def send(sig):
        # Stable handles still work when procfs observation is the failing
        # operation. Signal them before consulting process-group metadata.
        for pid, fd in list(pidfds.items()):
            try:
                signal.pidfd_send_signal(fd, sig)
                if (pid, sig) not in signal_records:
                    signal_records.add((pid, sig))
                    m["signals"].append({"target": "owned_pidfd", "pid": pid, "signal": sig})
            except ProcessLookupError:
                continue
        # Group signalling covers descendants born between procfs observations.
        # Only signal a still-owned native session; PID reuse cannot authorize it.
        group_owned = any((info := _proc(pid)) is not None and info["start_ticks"] == own["start_ticks"]
                          and info["pgid"] == proc.pid and info["sid"] == proc.pid for pid, own in owned.items())
        if group_owned:
            try:
                os.killpg(proc.pid, sig)
                if ("native_group", sig) not in signal_records:
                    signal_records.add(("native_group", sig))
                    m["signals"].append({"target": "native_group", "signal": sig})
            except ProcessLookupError:
                pass
        # Escaping descendants are addressed through stable pidfds, never a
        # guessed numeric PID. Adoption scans continue until actual ECHILD.

    try:
        if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise NativeProcessError("unsupported_platform", "Linux procfs/subreaper/pidfd required", m)
        if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
            raise NativeProcessError("ownership_busy", "single main Python thread required", m)
        if not _CALL_LOCK.acquire(blocking=False):
            raise NativeProcessError("ownership_busy", "another native call owns child admission", m)
        locked = True
        if _children(os.getpid()):
            raise NativeProcessError("ownership_busy", "existing caller children cannot be reaped", m)
        if type(mode) is not str or mode not in MODE_LIMITS or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= MODE_LIMITS[mode]:
            raise NativeProcessError("invalid_timeout", "only bounded identity/evaluate modes", m)
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise NativeProcessError("invalid_deadline", "absolute finite monotonic deadline required", m)
        hard_end = min(float(deadline), started + float(timeout))
        reserve = min(1.0, float(timeout) / 4)
        work_end = hard_end - reserve
        if time.monotonic() >= work_end:
            raise NativeProcessError("deadline_budget", "insufficient remaining work/drain budget", m)
        if m["parent_peak_rss_bytes"] > RSS_LIMIT:
            raise NativeProcessError("rss_limit", "caller high-water already exceeds 4GiB", m)
        path = Path(binary).absolute()
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or path.resolve() != path or not os.access(path, os.X_OK):
            raise NativeProcessError("unsafe_binary", "canonical regular executable required", m)
        if mode == "--identity" and request is not None or mode == "--evaluate" and type(request) is not dict:
            raise NativeProcessError("invalid_request", "identity has no request; evaluate has JSON object", m)
        encoded = b"" if request is None else json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > REQUEST_LIMIT:
            raise NativeProcessError("request_limit", "native request exceeds1MiB", m)
        if request is not None:
            strict_json(encoded)  # Reject key coercion that would duplicate JSON fields.
        old_subreaper = _subreaper()
        _subreaper(1)
        m["subreaper_previous"] = old_subreaper
        selector = selectors.DefaultSelector()
        proc = subprocess.Popen([str(path), mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=_environment(), start_new_session=True, close_fds=True, bufsize=0)
        m["leader_pid"] = proc.pid
        m["process_group"] = proc.pid
        # Bind the direct child before descendant discovery, including on
        # kernels whose /proc task children files are unavailable.
        pidfds[proc.pid] = os.pidfd_open(proc.pid)
        leader = _proc(proc.pid)
        if leader is None:
            raise OSError("direct child's procfs identity unavailable before reap")
        owned[proc.pid] = {name: leader[name] for name in ("pid", "ppid", "pgid", "sid", "start_ticks")}
        observe()
        m["leader_start_ticks"] = owned.get(proc.pid, {}).get("start_ticks")
        for name in ("stdin", "stdout", "stderr"):
            file = getattr(proc, name)
            os.set_blocking(file.fileno(), False)
            selector.register(file, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name)
        while True:
            observe()
            reap()
            now = time.monotonic()
            if error is None and now >= work_end:
                fail("timeout", "child work deadline reached; drain budget reserved")
            if error is None and proc.returncode is not None:
                if proc.returncode != 0:
                    fail("native_exit", "native leader returned nonzero")
                elif _children(os.getpid()):
                    fail("descendant_leak", "leader exited with live/adopted descendants")
            if error is not None:
                close_stream(proc.stdin)
                send(signal.SIGTERM if now - cleanup_started < min(.1, reserve / 3) else signal.SIGKILL)
            if m["echild"] and not selector.get_map():
                break
            if now >= hard_end:
                send(signal.SIGKILL)
                fail("cleanup_deadline", "hard attempt deadline reached before drain proof")
                break
            pause = min(.01, max(0., hard_end - now))
            for key, _mask in selector.select(pause):
                name, file = key.data, key.fileobj
                if name == "stdin":
                    if offset == len(encoded):
                        close_stream(file)
                        continue
                    try:
                        offset += os.write(file.fileno(), encoded[offset:offset + 65536])
                    except BrokenPipeError:
                        close_stream(file)
                        if offset != len(encoded):
                            fail("stdin_closed", "child closed input before full request")
                    except BlockingIOError:
                        continue
                else:
                    limit = STDOUT_LIMIT if name == "stdout" else STDERR_LIMIT
                    try:
                        chunk = os.read(file.fileno(), min(65536, limit + 1 - len(streams[name])))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        stream_eof[name] = True
                        close_stream(file)
                    else:
                        streams[name].extend(chunk)
                        if len(streams[name]) > limit:
                            stream_truncated[name] = True
                            fail(name + "_limit", "native stream exceeds admitted bound")
                            close_stream(file)
        observe()
        reap()
        if not m["echild"]:
            # Preserve unresolved ownership rather than false drain/ECHILD. The
            # caller must stop; subreaper/lock remain held for this process.
            fail("cleanup_unavailable", "owned children remain after hard deadline")
        if error is None:
            if offset != len(encoded) or not all(stream_eof.values()):
                fail("stream_incomplete", "request/EOF proof incomplete")
            else:
                try:
                    result = strict_json(bytes(streams["stdout"]))
                except (ValueError, UnicodeDecodeError) as exc:
                    fail("invalid_json", str(exc))
    except BaseException as exc:
        if isinstance(exc, NativeProcessError):
            fail(exc.code, str(exc))
        else:
            fail("cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "process_failure", str(exc))
        # Exceptions in observation/selector must not orphan the admitted group.
        if proc is not None:
            while hard_end is not None and time.monotonic() < hard_end:
                # Keep signalling and reaping even if procfs observation itself
                # caused the failure; neither operation depends on its success.
                for operation in (send, observe, reap):
                    try:
                        operation(signal.SIGKILL) if operation is send else operation()
                    except (OSError, ValueError) as cleanup_error:
                        detail = type(cleanup_error).__name__ + ": " + str(cleanup_error)
                        if detail not in m["cleanup_errors"] and len(m["cleanup_errors"]) < 8:
                            m["cleanup_errors"].append(detail)
                if m["echild"]:
                    break
                time.sleep(.005)
    finally:
        if proc is not None:
            for name in ("stdin", "stdout", "stderr"):
                close_stream(getattr(proc, name))
        if selector is not None:
            selector.close()
        m["streams_closed"] = True
        m["owned_processes"] = list(owned.values())
        reaped = {value["pid"] for value in m["reaped_processes"]}
        m["survivors"] = [value for pid, value in owned.items() if pid not in reaped
                          and (live := _proc(pid)) is not None and live["start_ticks"] == value["start_ticks"]]
        if proc is not None:
            m["returncode"] = proc.returncode
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                m["group_empty"] = True
            if not m["group_empty"]:
                fail("group_not_empty", "native process group remains")
        if old_subreaper is not None and (proc is None or m["echild"]):
            _subreaper(old_subreaper)
            m["subreaper_restored"] = True
        if locked and (proc is None or m["echild"]):
            _CALL_LOCK.release()
            m["ownership_released"] = True
        if proc is not None and not m["echild"]:
            _UNRESOLVED_OWNER = {"process": proc, "pidfds": pidfds, "owned_processes": owned}
        else:
            for fd in pidfds.values():
                os.close(fd)
        m["parent_peak_rss_bytes"] = max(m["parent_peak_rss_bytes"], _parent_peak())
        if max(m["parent_peak_rss_bytes"], m["child_peak_rss_bytes"]) > RSS_LIMIT:
            fail("rss_limit", "observed process high-water exceeds 4GiB policy")
        m["drained"] = proc is not None and m["echild"] and not m["survivors"] and m["streams_closed"]
        m["request"] = {"bytes": len(encoded), "written_bytes": offset, "sha256": hashlib.sha256(encoded).hexdigest()}
        for name, values in streams.items():
            m[name] = {"bytes": len(values), "sha256": hashlib.sha256(values).hexdigest(), "eof": stream_eof[name],
                       "truncated": stream_truncated[name], "sha256_scope": "observed_prefix" if stream_truncated[name] or not stream_eof[name] else "complete_stream"}
        m["stderr_preview"] = bytes(streams["stderr"][:4096]).decode("utf8", errors="replace")
        m["stdout_preview"] = bytes(streams["stdout"][:4096]).decode("utf8", errors="replace") if error else None
        m["wall_s"] = time.monotonic() - started
    if error is not None:
        m["failure_code"] = error[0]
        raise NativeProcessError(error[0], error[1], m)
    if not m["drained"] or not m["group_empty"] or m["wall_s"] > timeout or time.monotonic() > deadline:
        m["failure_code"] = "completion_proof"
        raise NativeProcessError("completion_proof", "deadline/drain/group proof failed", m)
    m["status"] = "complete"
    return result, m
