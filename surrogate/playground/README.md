# Playground

A sealed box of water stepped live at 60 steps per simulated second, shown as a slice in a web page, pushed with the
mouse. The time of every whole step is on screen. Roadmap phase 3, first cut; [spec.md](spec.md) is what was asked.

## Running it

```sh
python -m surrogate.playground --backend composition --grid 32 --port 8095
```

then open <http://127.0.0.1:8095/>. The server binds 127.0.0.1 only and uses only the Python standard library and
PyTorch.

The GPU backends need PyTorch with CUDA. When the interpreter that starts the playground has none (the lab's `.venv`
is CPU-only), it restarts itself under `--python`, else `$PLAYGROUND_PYTHON`, else Pumas Library's PyTorch 2.14
runtime at its usual place on this machine (`…/Pumas-Library/torch-versions/v2.14.0/venv/bin/python`). Pumas's
active version is not consulted or changed.

| Option | Default | |
| --- | --- | --- |
| `--backend` | `composition` | `composition`, `generator`, `reference`, `fake` |
| `--grid` | `32` | `N` for a cube or `NXxNYxNZ`; each backend lists the grids it runs and refuses others |
| `--cells-per-metre` | `16` | 8 or 16; the box is grid ÷ cells per metre (32 at 16 is a 2 m cube) |
| `--tracer` | `layer` | the dye at the start: `layer` (lower third), `stripes`, `ball`, `none` |
| `--mode` | `realtime` | or `fastest` |
| `--python` | | the CUDA interpreter for the GPU backends |

Without a GPU, `--backend fake` runs the page against a stand-in (below).

## Controls

- **Press and drag in the box** to push: the force sits at the press point, in the slice shown, and pushes along the
  drag for as long as the button is held; the drag's length sets its strength (a drag the size of the box gives the
  *Strength* value, in m/s²). **Push** pushes along the drag, **Swirl** turns the fluid about the axis out of the
  screen (anticlockwise when dragged to the right), **Shake** shakes it back and forth twice a second. **Radius** is
  the force's width (it reaches four radii).
- **Keep forces after release** leaves each force running; kept forces are listed, each with *Remove*, and
  *Remove all forces* clears them.
- **Run**: pause and resume, one step at a time, reset (the box at rest, same forces), real time or as fast as the
  backend goes, the dye pattern and *Reset dye*.
- **View**: the field (dye, speed, velocity x/y/z, pressure, vorticity magnitude, divergence), the cut (z, y or x)
  and its position, velocity arrows, and a small second view of another cut (its position is the dashed line).
  Signed fields use a blue–red scale symmetric about zero, the others a viridis scale; the legend gives the range of
  the frame shown (the dye is always 0 to 1).
- **Backend and box**: backend, grid and cells per metre, applied with *Apply* (restarts the box).
- The line at the top: backend, grid, median and p95 milliseconds per step over the last second, steps per second,
  the real-time factor (simulated seconds per second) and how many times faster than real time the median step
  could run. The badge says *Behind real time* when the steps cannot keep up.

## Backends

| Backend | What it is | Grids | Forces |
| --- | --- | --- | --- |
| `composition` (default) | The GPU solver composition ("F64 cell composition"): Rheon's step as fixed GPU operations replayed as one CUDA graph; reproduces the solver to rounding, listed as an exact solver | 9×11×11, 15×6×8, 16³, 16×24×40, 32×24×48, 32³, 64³, 104×112×120, 128³ (its kernel builds) | the first through the graph's own forcing kernel, after transport (as Rheon does); further ones added to the velocity before the step |
| `generator` | Learned full-state generator *b1-fixed-viscosity*: the whole next state from the current one and the force | 9×11×11, 15×6×8, 16³, 32³ (trained) | all through its force input, after its transport |
| `reference` | Rheon's own solver through the lab's reference probe (`surrogate/solver.py`), conjugate gradients on one CPU thread | 16³ to 64³ | added to the state before the step |
| `fake` | Not physics: the forces add velocity, which decays, and carries the dye. For tests and machines without CUDA | up to 64³ | before the step |

Every backend takes a force as the same blob (`blobs.py`): the forcing shape of the lab's scenes,
`magnitude · exp(-r²/2)(1-r²/16)² · (direction + swirl · turn about direction + shake · travelling wave)`, evaluated
exactly as the composition's CUDA kernel `known_control_dynamic` does. The tests check that evaluation against the
kernel's arithmetic and against `surrogate/forcing.py`.

Code: `backends/` (one module per backend, `base.py` the interface), `loop.py` (the simulation thread, pacing and
stats), `frames.py` and `views.py` (slices, quantised for the page), `server.py`, `static/` (`view.js` draws and
knows nothing of HTTP; `transport.js` is the HTTP client; `app.js` the page), `acceptance.py` (the measurement below).

## Measured

2026-11-19, NVIDIA GeForce RTX 5090 Laptop GPU, PyTorch 2.14.0+cu132 (Pumas), commit `de8a5a6c4` plus this
playground. `python -m surrogate.playground.acceptance --backend … --grid N --seconds 10`: real-time pacing through
the playground's own simulation thread, 16 cells per metre, the dye a layer in the lower third, one force pushing up
(+y) at 2 m/s² per metre of box from y = 0.3, radius 8 % of the box, sweeping x = 0.5 ± 0.2 every 5 s. A step is the
whole step: forces, the solver or model, and the copies around it, synchronised.

| Backend | Grid (box) | Median ms/step | p95 ms | Real-time factor | Median step vs 1/60 s |
| --- | --- | --- | --- | --- | --- |
| composition | 16³ (1 m) | 0.53 | 0.60 | 1.00 | 32× faster |
| composition | 32³ (2 m) | 0.72 | 0.80 | 1.00 | 23× |
| composition | 64³ (4 m) | 2.57 | 2.74 | 1.00 | 6.5× |
| composition | 128³ (8 m) | 22.5 | 25.7 | **0.73** (cannot keep up) | 0.7× |
| generator b1-fixed-viscosity | 16³ (1 m) | 5.66 | 6.57 | 1.00 | 2.9× |
| reference (CPU, 1 thread) | 16³ (1 m), 5 s | 14.7 | 18.2 | 1.00 | 1.1× |
| reference (CPU, 1 thread) | 32³ (2 m), 5 s | 173 | 180 | **0.10** | 0.1× |

- Paced steps are slower than back-to-back ones at small grids, since the GPU idles between them: back to back the
  composition took 0.14 ms at 16³ (120 steps) and 2.49 ms at 64³ (*Fastest* in the page, 2 s, 381 steps per second).
- Preparing a grid takes about 0.5 s for the composition (CUDA graph capture). Loading the generator takes about
  30 s (its construction evaluates small CPU convolutions; it runs on 4 CPU threads, on all 24 it took minutes).
- **The dye goes where it is pushed.** Its mean height rose from 0.16–0.17 of the box to 0.46 (16³), 0.43 (32³),
  0.38 (64³), 0.37 (128³) and 0.46 for the generator at 16³. In a 3 s run at 32³ the dye above the middle of the box
  was 0.13 (as a fraction of that half's volume) inside the band the force sweeps and 0.006 outside it; after 1 s
  the column where the dye stood highest was at x = 0.55 of the box, where the force had swept (0.5 to 0.69). After
  10 s the plume has reached the lid and spread along it, so the band no longer stands out.
- The generator's run used the same force as the composition's, about 15 times stronger and half the width of what
  it was trained on (0.08–0.16 m/s², radius 15–24 % of the box): its time per step does not depend on the force, but
  its answer is outside what it learned. Its mean dye height after 10 s (0.46) was the composition's.

## Limits

- **Sealed box only**: no free surface, no open or periodic boundaries, no obstacles. **Water only** (1000 kg/m³,
  1 mPa·s) and **no gravity**: the dye is a passive tracer, not a second fluid.
- **The composition takes one force per step through its own kernel**; any further forces are added to the velocity
  before the step (before transport rather than after it). With several forces held the difference is one step of
  transport of the extra ones.
- **The generator is out of its distribution** for anything but one gentle blob like its training's (one
  sinusoidally moving blob of 0.08–0.16 m/s², radius 15–24 % of the box, a little swirl and wave): several forces,
  the page's default drag strengths, swirl and shake are all new to it, and it is trained only on 9×11×11, 15×6×8,
  16³ and 32³. Set *Strength* to about 1 m/s² and *Radius* to about a fifth of the box to stay near it.
- Composition grids are the ones its kernels were built for; 128³ does not keep up with real time on this GPU.
- The reference solver is one CPU thread: real time at 16³ only.
- The GPU backends need the earlier studies' kernel builds, coefficients and weights under `runs/`, which are not in
  Git (see `surrogate/README.md`).
- The page shows slices only (no volume view yet), and polls the server about 30 times a second.

## Tests

```sh
.venv/bin/python -m unittest surrogate.tests.test_playground       # interface, forces, pacing, server (CPU)
node surrogate/tests/browser_playground.cjs [--shots <dir>]        # the page in a headless browser, fake backend
```

The browser check needs Node 22 or later and a Chromium headless shell (Playwright's cache, or `$BROWSER`).
