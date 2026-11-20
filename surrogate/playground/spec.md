# Playground: specification

Roadmap phase 3, first cut. Owner, 2026-11-19: "a really simple example app where the model can be run so we can
actually see the simulation in real time ... a configurable domain where you can position a force and be able to
visualize the effects of that force on the fluid."

## What it is

A web page served from this machine (`python -m surrogate.playground`, 127.0.0.1 only, Python standard library plus
the PyTorch the backend needs). A sealed box of water steps at 60 steps per simulated second. The page shows a slice of
the box coloured by a field and lets you push the fluid with the mouse. The real time of every whole step is on screen.

## Parts

1. **Backend interface** (`backends/`). One object per running simulation:
   - `reset(spec)`: grid counts (x, y, z), cells per metre, dt = 1/60 s, initial tracer pattern;
   - `set_forces(blobs)`: the forces, kept until changed;
   - `step()`: one whole step; the caller times it;
   - `slice(axis, index, field)`: float32 2D at cell centres; fields `speed`, `vx`, `vy`, `vz`, `pressure`,
     `tracer`, `vorticity` (magnitude), `divergence`;
   - `info()`, `capabilities()`: name, exact solver or learned model, device, grids, cells per metre, maximum blobs,
     where the force enters the step.

   A **blob** is `{centre (fractions of the box, x y z), radius (m), direction (unit or zero), magnitude (m/s²),
   swirl, shake}`. Its acceleration is the formula of the solver composition's forcing kernel
   (`known_control_dynamic`): `magnitude · exp(-r²/2)(1-r²/16)² · (direction + swirl · rotation about direction +
   shake · travelling wave)`, `r` in radii, cut off at 4 radii, zero on the walls.

   Adapters: `composition` (the GPU solver composition, default), `generator` (learned full-state generator
   b1-fixed-viscosity, GPU), `reference` (Rheon's own solver through the reference probe, one CPU thread), `fake`
   (pure PyTorch on the CPU, a tracer advected by the blob-shaped velocity; for tests and machines without CUDA).
   Each refuses grids it cannot run, with the reason.
2. **Simulation thread** (`loop.py`). The only thread that touches the backend. Real time (one step per 1/60 s,
   sleeping when ahead, flagging when behind) or as fast as possible; pause, single step, reset. Stats over the last
   second: median and p95 ms per step, steps per second, real-time factor (simulated seconds per wall second) and
   headroom (1/60 s over the median step).
3. **Server** (`server.py`). Endpoints for capabilities, backend, reset, forces, control and frames. A frame is JSON:
   the slices as uint8 (base64) with their range, optional in-plane arrows as int8, and the stats. The page polls.
4. **Page** (`static/`). `view.js` draws decoded frames and knows nothing of HTTP; `transport.js` is the HTTP client;
   `app.js` wires them. A large slice view and a small view of another axis; field, axis and position; arrows;
   press-drag places a force at the press point pushing along the drag (push, swirl, shake), held while the button is
   down, kept on release when "keep" is on; kept forces listed and removable; real time or fastest; pause, step,
   reset, reset tracer; stats line. Light and dark; works at 390 px wide.
5. **Tests** on the CPU (lab `.venv`): the interface on the fake backend, blob to control packet against the kernel
   formula evaluated in Python, the endpoints, the pacing; a headless browser check on the fake backend.
6. **GPU acceptance**: about 10 s each of the composition at 16³, 32³, 64³ (and 128³), and the generator at 16³, with
   a moving blob; numbers in the README.

## Limits accepted for this cut

Sealed box only; water only; no gravity. The composition takes one blob per step through its own forcing kernel;
further blobs are added to the state as a velocity increment before the step. The generator was trained on one
sinusoidally moving blob, so any other forcing is out of its distribution.

## As built (2026-11-19)

- All six parts as above; the page polls `/api/frame` about 30 times a second (no server-sent events). Frames are
  uint8 planes with their range, the first plane with int8 arrows; the second view is the middle x cut (the middle
  z cut when the main view cuts x or y), with dashed lines where each view cuts the other.
- Forces from the page: push along the drag; swirl turns about the axis out of the screen with `swirl = 4` and a
  quarter of the magnitude along that axis (the kernel's swirl always comes with some push along its axis); shake is
  the kernel's travelling wave with no push (`direction = 0`, `shake = 1`, 2 Hz).
- The composition keeps its state on the GPU and replays its CUDA graph directly (`_replay_borrowed`) with the
  playground's own control-packet row, so a step is one replay plus the extra forces and a clone. Loading it needed
  five kernel sources and a code-generation receipt that had left the tree with the earlier-study records; they are
  restored unchanged (their digests match the build receipts), in a separate commit.
- CPU work (the fake and reference backends, the generator's construction) runs on at most 4 threads: on all 24 the
  small tensor operations were 50 times slower and the generator took minutes to load.
- The reference probe needs whole cells per metre; its scene's own forcing is set to zero and every step restores
  the state at step 0.
- Measured numbers and limits: [README.md](README.md).
