# What the earlier work found (October 2026)

The lab's first month of models, in one page. Each model's card (method, purpose, outcome, lesson) is in [`projects/surrogate-lab/notes/fits/`](../projects/surrogate-lab/notes/fits/), its benchmark results are in [`projects/surrogate-lab/results/`](../projects/surrogate-lab/results/), and Tuldok draws them on the Map and the Benchmarks tab. The full records of these studies (plans, reports, logs) were removed from the tree on 2026-11-19 and remain in Git history.

All of this work used **`rheon-reference-v1`**: Rheon `bc2f4727` plus the lab's stepping and forcing additions, now Rheon branch `lab/reference-solver-v1` (`bbe01bd3`), pinned by [`probes/reference/`](../probes/reference/). New data and models use the current Rheon only.

## What was tried

| Stage | Models | Question |
| --- | --- | --- |
| Pressure-only fixed box | the first experiments | Can a network replace the pressure solve in one 32³ sealed box? Yes. A modal model replaced the iterative solve and ran a 120-frame rollout 5–9× faster than Rheon's Jacobi and SGS solvers on the CPU, with all frames within the solver's gates. An exact spectral solve also exists for such boxes, so this was not an advantage over the best non-learned method. |
| Pressure inputs, residuals, stability (C03–C25) | 23 | Which inputs and training make a pressure model generalise across scenes and grid shapes? |
| Grids and recurrence (C26–C37), directional feedback (C38–C53) | 28 | Can one model serve several grids and several correction passes? |
| Pressure variants (C54–C105) and successors | about 70 | Architecture and training-policy variants, multigrid, and heads: the frontier of pressure error against GPU call time |
| Full-state generators | 35 | Can one network take the whole state to the next state (pressure, velocity, tracer) on the GPU? |
| Components | 26 | Separately trained parts (heads, readouts, transport, viscosity) reused by later models |

## What was learned

- **The frontier.** The best pressure models reach a mean-free relative pressure error of about 2·10⁻⁶ at 7–7.5 ms per GPU pressure call (128 requests). The fastest reach about 0.3 ms with errors of a few percent. Tuldok's Benchmarks tab draws this frontier.
- **Whole-state models work but are not yet worth it on simple boxes.** The full-state generators run a 1/60 s step in 0.8–5.3 ms on grids up to 32³, with velocity errors from about 9 % down to 2·10⁻⁶. The solver's own arithmetic as a fixed GPU network (the "F64 composition") is faster (0.25–0.6 ms) and more accurate on these boxes. A learned model pays off only where the classical solve is expensive: larger grids, free surfaces, several fluids, obstacles.
- **Freezing accurate parts does not make an accurate whole.** Generators that reused exactly calibrated viscosity or pressure components still erred through their learned transport and fallback paths. Compare the complete fields, not the parts.
- **The data was narrow.** Every scene was a sealed box of water driven by sinusoidal forcing, a few seconds long, with only short windows of states kept for the larger scenes. A model can fit that well and still not generalise.

## What changes now

The [roadmap](../surrogate/ROADMAP.md) moves to the current Rheon, forces of many kinds, several fluids with free surfaces, open and periodic boundaries, a live playground, and two model branches, speed and accuracy, judged on the same benchmarks.
