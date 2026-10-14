"""Synthetic executable tests only; no model, native evaluator or credentials."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("gn_native_process_tests", Path(__file__).with_name("process.py"))
process = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(process)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gn-process-fixture-")
        self.root = Path(self.directory.name)
        self.previous_subreaper = process._subreaper()
        self.fds_before = len(list(Path("/proc/self/fd").iterdir()))

    def tearDown(self):
        self.assertFalse(process._children(os.getpid()), "fixture left a child")
        self.assertEqual(process._subreaper(), self.previous_subreaper)
        self.assertFalse(process._CALL_LOCK.locked())
        self.assertIsNone(process._UNRESOLVED_OWNER)
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), self.fds_before)
        self.directory.cleanup()

    def executable(self, source):
        path = self.root / "fixture"
        path.write_text("#!" + sys.executable + "\n" + source + "\n")
        path.chmod(0o700)
        return path

    def call(self, source, request=None, timeout=2.0, mode=None):
        return process.native_call(self.executable(source), mode or ("--identity" if request is None else "--evaluate"),
                                   request, time.monotonic() + timeout + 1, timeout)

    def fails(self, source, code, request=None, timeout=2.0):
        with self.assertRaises(process.NativeProcessError) as raised:
            self.call(source, request, timeout)
        self.assertEqual(raised.exception.code, code, raised.exception.measurement)
        measurement = raised.exception.measurement
        self.assertEqual(measurement["status"], "failed")
        self.assertEqual(measurement["failure_code"], code)
        self.assertLessEqual(measurement["wall_s"], timeout)
        self.assertTrue(measurement["spawn_closed"])
        if measurement["leader_pid"] is not None:
            self.assertTrue(measurement["drained"], measurement)
            self.assertTrue(measurement["echild"], measurement)
            self.assertTrue(measurement["group_empty"], measurement)
            self.assertTrue(measurement["subreaper_restored"], measurement)
            self.assertTrue(measurement["ownership_released"], measurement)
            self.assertFalse(measurement["survivors"])
            self.assertTrue(measurement["reaped_processes"])
        return measurement

    def test_normal_identity_exact_measurement_and_reusable_owner(self):
        raw = b'{"ok":true}\n'
        for _ in range(2):
            result, measurement = self.call("print('{\"ok\":true}')")
            self.assertEqual(result, {"ok": True})
            self.assertEqual(measurement["status"], "complete")
            self.assertEqual(measurement["returncode"], 0)
            self.assertTrue(measurement["drained"] and measurement["echild"] and measurement["group_empty"])
            self.assertTrue(measurement["streams_closed"] and measurement["spawn_closed"])
            self.assertTrue(measurement["subreaper_restored"] and measurement["ownership_released"])
            self.assertIsNotNone(measurement["leader_start_ticks"])
            self.assertEqual(measurement["stdout"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(measurement["stdout"]["bytes"], len(raw))
            self.assertEqual(measurement["stdout"]["sha256_scope"], "complete_stream")
            self.assertEqual(measurement["request"]["written_bytes"], 0)
            self.assertGreater(measurement["child_peak_rss_bytes"], 0)
            self.assertGreater(measurement["parent_peak_rss_bytes"], 0)

    def test_interleaved_bounded_io_and_request_hash(self):
        request = {"large": "x" * 180000}
        source = """import json,os,sys
os.write(2,b'e'*32000)
os.write(1,b'{"large":"'+b'y'*200000+b'",')
request=json.load(sys.stdin)
os.write(1,b'"read":'+str(len(request['large'])).encode()+b'}')"""
        result, m = self.call(source, request)
        self.assertEqual(result, {"large": "y" * 200000, "read": 180000})
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(m["request"], {"bytes": len(encoded), "written_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()})
        self.assertEqual(m["stderr"]["bytes"], 32000)
        self.assertTrue(m["stderr"]["eof"])

    def test_exact_stream_and_request_caps_are_admitted(self):
        # Compact encoding contributes eight bytes around the string.
        request = {"x": "x" * (process.REQUEST_LIMIT - 8)}
        source = """import json,os,sys
request=json.load(sys.stdin)
os.write(2,b'e'*65536)
os.write(1,b'{"x":"'+b'y'*(524288-8)+b'"}')"""
        result, m = self.call(source, request)
        self.assertEqual(len(result["x"]), process.STDOUT_LIMIT - 8)
        self.assertEqual(m["request"]["bytes"], process.REQUEST_LIMIT)
        self.assertEqual(m["request"]["written_bytes"], process.REQUEST_LIMIT)
        self.assertEqual(m["stdout"]["bytes"], process.STDOUT_LIMIT)
        self.assertEqual(m["stderr"]["bytes"], process.STDERR_LIMIT)
        self.assertFalse(m["stdout"]["truncated"] or m["stderr"]["truncated"])

    def test_one_thread_minimal_environment(self):
        source = """import os,json
names=['RAYON_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']
print(json.dumps({'threads':[os.environ.get(n) for n in names], 'secret_present':'GN_FIXTURE_SECRET' in os.environ,'ld_present':'LD_PRELOAD' in os.environ}))"""
        with mock.patch.dict(os.environ, {"GN_FIXTURE_SECRET": "synthetic-token"}):
            result, _m = self.call(source)
        self.assertEqual(result, {"threads": ["1"] * 5, "secret_present": False, "ld_present": False})

    def test_nonzero_exit_retains_partial_stream(self):
        m = self.fails("import os\nos.write(1,b'partial')\nos.write(2,b'failure')\nraise SystemExit(7)", "native_exit")
        self.assertEqual(m["returncode"], 7)
        self.assertEqual(m["stdout"]["bytes"], 7)
        self.assertEqual(m["stderr_preview"], "failure")

    def test_timeout_term_kill_and_drain(self):
        m = self.fails("import time,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint('partial',flush=True)\ntime.sleep(5)", "timeout", timeout=.4)
        self.assertEqual(m["returncode"], -signal.SIGKILL)
        self.assertEqual(m["stdout"]["bytes"], 8)
        self.assertIn(signal.SIGKILL, [record["signal"] for record in m["signals"]])

    def test_blocked_stdin_is_nonblocking_and_deadline_bounded(self):
        m = self.fails("import time\ntime.sleep(5)", "timeout", request={"large": "x" * 250000}, timeout=.4)
        self.assertLess(m["request"]["written_bytes"], m["request"]["bytes"])
        self.assertGreater(m["request"]["written_bytes"], 0)

    def test_remaining_workflow_deadline_bounds_child_attempt(self):
        started = time.monotonic()
        with self.assertRaises(process.NativeProcessError) as raised:
            process.native_call(self.executable("import time\ntime.sleep(5)"), "--identity", None, started + .3, .8)
        self.assertEqual(raised.exception.code, "timeout")
        m = raised.exception.measurement
        self.assertTrue(m["drained"] and m["echild"] and m["group_empty"])
        self.assertLess(time.monotonic() - started, .3)

    def test_stdout_limit_bounds_captured_prefix(self):
        m = self.fails("import os,time\nos.write(1,b'x'*600000)\ntime.sleep(5)", "stdout_limit")
        self.assertEqual(m["stdout"]["bytes"], process.STDOUT_LIMIT + 1)
        self.assertTrue(m["stdout"]["truncated"])
        self.assertEqual(m["stdout"]["sha256_scope"], "observed_prefix")
        self.assertEqual(m["stdout"]["sha256"], hashlib.sha256(b'x' * (process.STDOUT_LIMIT + 1)).hexdigest())

    def test_stderr_limit_bounds_captured_prefix(self):
        m = self.fails("import os,time\nos.write(2,b'x'*100000)\ntime.sleep(5)", "stderr_limit")
        self.assertEqual(m["stderr"]["bytes"], process.STDERR_LIMIT + 1)
        self.assertTrue(m["stderr"]["truncated"])
        self.assertEqual(m["stderr"]["sha256_scope"], "observed_prefix")

    def test_live_descendant_after_zero_leader_is_killed_and_adopted(self):
        m = self.fails("""import os,time
if os.fork()==0:
    time.sleep(5)
else:
    print('{"ok":true}',flush=True)
    os._exit(0)""", "descendant_leak")
        self.assertEqual(m["returncode"], 0)
        self.assertEqual(len(m["reaped_processes"]), 2)

    def test_escaped_session_double_fork_is_owned_and_reaped(self):
        m = self.fails("""import os,time
if os.fork()==0:
    os.setsid()
    if os.fork()==0:
        time.sleep(5)
    os._exit(0)
time.sleep(.05)
print('{"ok":true}',flush=True)
time.sleep(5)""", "descendant_escape")
        self.assertTrue(m["escaped_descendant_observed"])
        self.assertGreaterEqual(len(m["reaped_processes"]), 2)
        self.assertTrue(any(record["target"] == "owned_pidfd" for record in m["signals"]))

    def test_existing_unrelated_child_is_never_signalled_or_reaped(self):
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(5)"])
        try:
            m = self.fails("print('{}')", "ownership_busy")
            self.assertIsNone(m["leader_pid"])
            self.assertFalse(m["echild"])
            self.assertEqual(m["reaped_processes"], [])
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=2)

    def test_external_watchdog_exception_owns_cleanup(self):
        previous_handler = signal.getsignal(signal.SIGALRM)
        def alarm(_signal, _frame):
            raise ValueError("workflow_watchdog_deadline")
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, .1)
        try:
            m = self.fails("import time\ntime.sleep(5)", "process_failure")
            self.assertIn(signal.SIGKILL, [record["signal"] for record in m["signals"]])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def test_keyboard_interrupt_owns_cleanup(self):
        real_select = process.selectors.DefaultSelector.select
        calls = []
        def interrupt(selector, timeout=None):
            if not calls:
                calls.append(True)
                raise KeyboardInterrupt()
            return real_select(selector, timeout)
        with mock.patch.object(process.selectors.DefaultSelector, "select", interrupt):
            self.fails("import time\ntime.sleep(5)", "cancelled")

    def test_procfs_observation_failure_does_not_skip_kill_or_reap(self):
        children = process._children
        def broken(pid):
            if process._CALL_LOCK.locked() and process._subreaper() == 1:
                raise OSError("synthetic procfs observation failure")
            return children(pid)
        with mock.patch.object(process, "_children", broken):
            m = self.fails("import time\ntime.sleep(5)", "process_failure")
        self.assertTrue(m["cleanup_errors"])
        self.assertTrue(any(record["target"] == "owned_pidfd" for record in m["signals"]))

    def test_child_observed_rss_violation_terminates_and_reaps(self):
        with mock.patch.object(process, "RSS_LIMIT", 32 * 1024**2), mock.patch.object(process, "_parent_peak", return_value=1):
            m = self.fails("import time\na=bytearray(80*1024**2)\ntime.sleep(5)", "rss_limit")
        self.assertGreater(m["child_peak_rss_bytes"], 32 * 1024**2)

    def test_parent_rss_violation_precedes_spawn(self):
        with mock.patch.object(process, "_parent_peak", return_value=process.RSS_LIMIT + 1):
            m = self.fails("print('{}')", "rss_limit")
        self.assertIsNone(m["leader_pid"])

    def test_closed_stdin_preserves_partial_request(self):
        m = self.fails("import os,time\nos.close(0)\ntime.sleep(.05)\nprint('{}')", "stdin_closed", request={"x": "x" * 150000})
        self.assertLess(m["request"]["written_bytes"], m["request"]["bytes"])

    def test_strict_native_json(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '[]', '\\xff'):
            with self.subTest(raw=raw):
                literal = repr(raw.encode()) if raw != '\\xff' else "b'\\xff'"
                self.fails("import os\nos.write(1," + literal + ")", "invalid_json")

    def test_invalid_admission_never_spawns(self):
        path = self.executable("print('{}')")
        for mode, request, deadline, timeout, code in (
            ("--test", None, time.monotonic()+3, 1, "invalid_timeout"),
            ("--identity", None, time.monotonic()+3, True, "invalid_timeout"),
            ("--identity", None, time.monotonic()+20, 11, "invalid_timeout"),
            ("--evaluate", {}, time.monotonic()+70, 61, "invalid_timeout"),
            ("--identity", None, float('nan'), 1, "invalid_deadline"),
            ("--identity", None, time.monotonic()-.1, 1, "deadline_budget"),
            ("--identity", {}, time.monotonic()+3, 1, "invalid_request"),
            ("--evaluate", [], time.monotonic()+3, 1, "invalid_request"),
            ("--evaluate", {"x":"x"*process.REQUEST_LIMIT}, time.monotonic()+3, 1, "request_limit"),
        ):
            with self.subTest(code=code, mode=mode):
                with self.assertRaises(process.NativeProcessError) as raised:
                    process.native_call(path, mode, request, deadline, timeout)
                self.assertEqual(raised.exception.code, code)
                self.assertIsNone(raised.exception.measurement["leader_pid"])

    def test_symlink_binary_rejected(self):
        path = self.executable("print('{}')")
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(process.NativeProcessError) as raised:
            process.native_call(link, "--identity", None, time.monotonic()+3, 1)
        self.assertEqual(raised.exception.code, "unsafe_binary")
        self.assertIsNone(raised.exception.measurement["leader_pid"])

    def test_concurrent_call_lock_rejects_without_spawn(self):
        self.assertTrue(process._CALL_LOCK.acquire(blocking=False))
        try:
            self.fails("print('{}')", "ownership_busy")
        finally:
            process._CALL_LOCK.release()


if __name__ == "__main__":
    unittest.main()
