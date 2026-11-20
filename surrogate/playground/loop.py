"""The simulation thread: steps the backend in real time or as fast as it can, and measures every step.

Only this thread touches the backend. Everything else (the server's request threads) hands it work with `call`,
which runs between two steps and returns the result, so a frame is always read from a state no step is writing.
"""
import concurrent.futures
import math
import statistics
import threading
import time
from collections import deque

from .backends.base import DT, Unsupported

MODES = ('realtime', 'fastest')


class Pacer:
    """When the next real-time step is due: step k of a run that started at `start` is due at start + k * dt.

    When the steps fall further behind than `max_lag` seconds the backlog is dropped (the run restarts from now)
    instead of being caught up in a burst; `late_at` records when that last happened.
    """

    def __init__(self, dt=DT, max_lag=0.25):
        self.dt, self.max_lag = dt, max_lag
        self.start, self.count, self.late_at = None, 0, None

    def restart(self, now):
        self.start, self.count = now, 0

    def delay(self, now):
        """Seconds to wait before the next step may start (0 when it is due or overdue)."""
        if self.start is None:
            self.restart(now)
        return max(0.0, self.start + self.count * self.dt - now)

    def stepped(self, now):
        """Record a finished step. Returns True when the run had fallen behind and was restarted."""
        self.count += 1
        if now - (self.start + self.count * self.dt) > self.max_lag:
            self.late_at = now
            self.restart(now)
            return True
        return False


class Stats:
    """Step times over the last `window` seconds of wall time."""

    def __init__(self, window=1.0, dt=DT):
        self.window, self.dt = window, dt
        self.samples = deque()          # (finished at, ms)
        self.since = None

    def clear(self, now):
        self.samples.clear()
        self.since = now

    def add(self, now, ms):
        if self.since is None:
            self.since = now - ms / 1000
        self.samples.append((now, ms))
        self._trim(now)

    def _trim(self, now):
        while self.samples and self.samples[0][0] < now - self.window:
            self.samples.popleft()

    def summary(self, now):
        self._trim(now)
        times = sorted(ms for _, ms in self.samples)
        if not times:
            return dict(median_ms=None, p95_ms=None, steps_per_s=0.0, realtime_factor=0.0, headroom=None, samples=0)
        span = min(self.window, max(now - (self.since if self.since is not None else now), 1e-9))
        rate = len(times) / span
        median = statistics.median(times)
        p95 = times[max(0, math.ceil(0.95 * len(times)) - 1)]   # nearest rank
        return dict(median_ms=median, p95_ms=p95, steps_per_s=rate, realtime_factor=rate * self.dt,
                    headroom=(self.dt * 1000 / median) if median > 0 else None, samples=len(times))


class Simulation:
    """Owns the backend and the thread that steps it."""

    def __init__(self, backend, spec, mode='realtime', clock=time.perf_counter):
        if mode not in MODES:
            raise ValueError(f'The mode is one of {", ".join(MODES)}.')
        self.backend, self.spec, self.mode, self.clock = backend, spec, mode, clock
        self.paused = False
        self.status, self.message = 'loading', ''
        self.pacer, self.stats = Pacer(), Stats()
        self.single_steps = 0
        self.version = 0                 # changes whenever the state changes
        self._queue = deque()
        self._wake = threading.Condition()
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name='playground-simulation', daemon=True)

    # ---- used from other threads ----------------------------------------------------------------------------
    def start(self, reset=True):
        self._thread.start()
        if not reset:
            return None
        try:
            return self.call(lambda: self._reset(self.spec), timeout=600)
        except Exception as error:
            self.status, self.message = 'error', str(error)
            raise

    def stop(self, timeout=30):
        with self._wake:
            self._stopping = True
            self._wake.notify_all()
        self._thread.join(timeout)

    def call(self, function, timeout=30.0):
        """Run `function()` on the simulation thread between steps and return its result (or raise its error)."""
        future = concurrent.futures.Future()
        with self._wake:
            if self._stopping:
                raise RuntimeError('The simulation has stopped.')
            self._queue.append((function, future))
            self._wake.notify_all()
        return future.result(timeout)

    # ---- commands (they run on the simulation thread through `call`) ----------------------------------------
    def _reset(self, spec, backend=None):
        """Reset the current backend to `spec`, or switch to `backend` reset to `spec` (the old one is closed only
        once the new one runs; when it cannot, the old one carries on)."""
        previous = self.status, self.message
        self.status, self.message = 'loading', 'Preparing the grid…'
        target = backend if backend is not None else self.backend
        try:
            target.reset(spec)
            target.set_forces(target.forces[:target.max_blobs])
        except Exception as error:
            if backend is not None and backend is not self.backend:
                backend.close()
                self.status, self.message = previous
            elif isinstance(error, Unsupported):     # refused before anything changed
                self.status, self.message = previous
            else:
                self.status, self.message = 'error', str(error)
            raise
        if target is not self.backend:
            target.forces = self.backend.forces[:target.max_blobs] if self.backend else []
            old, self.backend = self.backend, target
            if old is not None:
                old.close()
        self.spec = spec
        self.status, self.message = ('paused' if self.paused else 'running'), ''
        self.version += 1
        self._restart_clock()
        return self.backend.info()

    def _restart_clock(self):
        now = self.clock()
        self.pacer.restart(now)
        self.stats.clear(now)

    def control(self, action, mode=None, steps=1):
        """pause, resume, step (n single steps while paused), mode (realtime or fastest)."""
        def run():
            if action == 'pause':
                self.paused = True
            elif action == 'resume':
                self.paused = False
                self._restart_clock()
            elif action == 'step':
                self.paused = True
                self.single_steps += max(1, min(int(steps), 600))
            elif action == 'mode':
                if mode not in MODES:
                    raise ValueError(f'The mode is one of {", ".join(MODES)}.')
                self.mode = mode
                self._restart_clock()
            else:
                raise ValueError('The action is pause, resume, step or mode.')
            if self.status in ('running', 'paused'):
                self.status = 'paused' if self.paused else 'running'
            return self.snapshot()
        return self.call(run)

    def snapshot(self):
        """Status, mode and stats; cheap, any thread."""
        now = self.clock()
        stats = self.stats.summary(now)
        late = self.pacer.late_at is not None and now - self.pacer.late_at < 1.0
        running = self.status == 'running' and not self.paused
        stats['behind'] = bool(running and self.mode == 'realtime' and stats['samples'] > 0 and
                               (late or stats['realtime_factor'] < 0.95 and now - (self.stats.since or now) > 1.0))
        backend = self.backend
        return dict(status=self.status, message=self.message, mode=self.mode, paused=self.paused,
                    version=self.version, steps=backend.steps if backend else 0,
                    time_s=(backend.steps if backend else 0) * DT, stats=stats)

    # ---- the thread -----------------------------------------------------------------------------------------
    def _drain(self):
        while True:
            with self._wake:
                if not self._queue:
                    return
                function, future = self._queue.popleft()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function())
                except BaseException as error:   # handed to the caller
                    future.set_exception(error)

    def _wait(self, seconds):
        with self._wake:
            if not self._queue and not self._stopping:
                self._wake.wait(seconds)

    def _run(self):
        while not self._stopping:
            self._drain()
            if self._stopping:
                break
            ready = self.status in ('running', 'paused') and self.backend is not None and self.backend.spec is not None
            if not ready or (self.paused and self.single_steps == 0):
                self._wait(0.1)
                continue
            if self.mode == 'realtime' and not self.paused:
                delay = self.pacer.delay(self.clock())
                if delay > 0:
                    self._wait(delay)
                    continue
            started = self.clock()
            try:
                self.backend.step()
            except Exception as error:
                self.status, self.message, self.paused = 'error', f'The step failed: {error}', True
                continue
            finished = self.clock()
            self.stats.add(finished, (finished - started) * 1000)
            self.version += 1
            if self.paused:
                self.single_steps -= 1
            elif self.mode == 'realtime':
                self.pacer.stepped(finished)
        backend = self.backend
        if backend is not None:
            backend.close()
