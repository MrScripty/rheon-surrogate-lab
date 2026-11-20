"""Headless acceptance run: a backend in real time for a few seconds with a moving force, measured.

    python -m surrogate.playground.acceptance --backend composition --grid 32 --seconds 10

The force pushes up (+y) from y = 0.3 of the box and sweeps along x (0.5 ± 0.2 of the box, a 5 s period) through
the middle of z. The dye starts as a layer in the lower third. The run steps in real time through the playground's
own simulation thread, as the page would drive it, and prints one JSON document: every step's time (median, p95,
mean, max), steps per second, the real-time factor, and where the dye went: after about 1 s the column of the box
where the dye sits highest (against where the force is), and after the run its mean height and how much of it is
above the middle inside and outside the band the force sweeps.
"""
import argparse
import datetime
import json
import math
import statistics
import subprocess
import sys
import threading
import time

import torch

from .__main__ import REPOSITORY, grid
from .backends import SceneSpec, backend_class
from .loop import Simulation


def blob(spec, t, magnitude):
    size = min(spec.extents)
    return dict(centre=[0.5 + 0.2 * math.sin(math.tau * t / 5), 0.3, 0.5], radius=0.08 * size,
                direction=[0, 1, 0], magnitude=magnitude * size)


def dye_report(backend):
    tracer = backend.fields()['tracer'].double().cpu()          # [Z, Y, X]
    nz, ny, nx = tracer.shape
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx
    total = float(tracer.sum())
    upper = tracer[:, ny // 2:, :]
    band = (x - 0.5).abs() <= 0.3
    width_in, width_out = int(band.sum()), int((~band).sum())
    return dict(mean_height=float((tracer * y.reshape(1, ny, 1)).sum()) / total if total else None,
                dye_above_middle_in_band=float(upper[..., band].sum()) / max(1, width_in) / (nz * ny / 2),
                dye_above_middle_outside_band=float(upper[..., ~band].sum()) / max(1, width_out) / (nz * ny / 2))


def highest_column(backend):
    """The x (fraction of the box) of the column whose dye sits highest, averaged over z."""
    tracer = backend.fields()['tracer'].double().cpu()
    nz, ny, nx = tracer.shape
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny
    height = (tracer * y.reshape(1, ny, 1)).sum(dim=(0, 1)) / tracer.sum(dim=(0, 1)).clamp_min(1e-12)
    column = int(height.argmax())
    return dict(x=(column + 0.5) / nx, mean_height=float(height[column]), median_column_height=float(height.median()))


def commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(backend_name, counts, cells_per_metre, seconds, magnitude):
    backend = backend_class(backend_name)()
    spec = SceneSpec(tuple(counts), cells_per_metre, 'layer')
    sim = Simulation(backend, spec, mode='realtime')
    loaded = time.perf_counter()
    sim.start()
    load_s = time.perf_counter() - loaded
    times, original = [], backend.step

    def timed():
        started = time.perf_counter()
        original()
        times.append((time.perf_counter() - started) * 1000)
    backend.step = timed
    before = sim.call(lambda: dye_report(backend))
    stop = threading.Event()
    early = {}

    def mover():
        began = time.perf_counter()
        while not stop.wait(0.05):
            t = time.perf_counter() - began
            sim.call(lambda: backend.set_forces([blob(spec, t, magnitude)]))
            if t >= 1.0 and not early:
                early.update(highest_column=sim.call(lambda: highest_column(backend)), at_s=round(t, 2),
                             force_x=blob(spec, t, magnitude)['centre'][0])
    sim.call(lambda: backend.set_forces([blob(spec, 0.0, magnitude)]))
    sim.call(lambda: times.clear())
    started = time.perf_counter()
    thread = threading.Thread(target=mover, daemon=True)
    thread.start()
    time.sleep(seconds)
    stop.set()
    thread.join()
    after = sim.call(lambda: dye_report(backend))
    steps = sim.call(lambda: len(times))
    wall = time.perf_counter() - started
    last_second = sim.snapshot()['stats']
    sim.stop()
    ordered = sorted(times[:steps])
    return dict(
        date=datetime.date.today().isoformat(), backend=backend_name, label=backend.label, kind=backend.kind,
        grid=list(counts), cells_per_metre=cells_per_metre, box_m=list(spec.extents), seconds=round(wall, 3),
        device=torch.cuda.get_device_name() if backend.device == 'cuda' and torch.cuda.is_available() else 'CPU',
        torch=torch.__version__, commit=commit(), load_and_prepare_s=round(load_s, 2), steps=steps,
        median_ms=statistics.median(ordered), p95_ms=ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        mean_ms=statistics.fmean(ordered), max_ms=ordered[-1], steps_per_s=steps / wall,
        realtime_factor=steps / wall / 60, headroom=(1000 / 60) / statistics.median(ordered),
        behind_in_last_second=last_second['behind'],
        force=dict(blob(spec, 0.0, magnitude), note='centre x sweeps 0.5 ± 0.2 with a 5 s period'),
        dye_before=before, dye_after=after, dye_after_1s=early)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', default='composition', choices=('composition', 'generator', 'reference', 'fake'))
    p.add_argument('--grid', type=grid, default=(32, 32, 32))
    p.add_argument('--cells-per-metre', type=float, default=16.0)
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--magnitude', type=float, default=2.0, help='force in m/s² per metre of box (default 2)')
    args = p.parse_args(argv)
    if not 1 <= args.seconds <= 60:
        sys.exit('Run 1 to 60 seconds.')
    print(json.dumps(run(args.backend, args.grid, args.cells_per_metre, args.seconds, args.magnitude), indent=1))


if __name__ == '__main__':
    main()
