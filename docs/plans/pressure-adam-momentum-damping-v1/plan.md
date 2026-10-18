# First-moment damping screen

Test the user's proposed partial momentum damping without resetting Adam's
second moment or step counter. This is an independent one-step CPU experiment
on consumed training states, not a training trajectory or deployment.

Predeclare saved updates 1, 10, 100, 150 and 200; retain first-moment factors
0, 1/4, 1/2, 3/4 and 1. Factor 0 discards historical first moment only; factor 1
is the preserved-state control. Before the same Adam update use m_before*f;
retain v_before, t_before, saved clipped CUDA gradient, LR, betas and epsilon.
Do not damp the new gradient contribution. Update 1 has no saved history and
all factors must coincide. Also evaluate the literal before and observed CUDA
after parameters. No endpoint interpolation or magnitude normalization.

Evaluate all eight original train-only analytical X-fundamental cases with
the captured full loss and relative pressure-equation residual at 1/2/4/8
passes. Use the unchanged <=1e-6 pressure qualification. Guards: no case worsens
at 1/2 passes; no fitted 4/8-pass case loses qualification; weak case 7 loss and
four-pass residual improve beyond twice observed CPU/GPU pre/post gaps; worst
four-pass residual decreases. The gaps are empirical, not rounding bounds.
Record lengths and direction cosine relative to the preserved endpoint; changed
length alone does not isolate a direction benefit.

Bound to three separate one-thread CPU workers, five queued snapshots, 120s
global elapsed cap, 12GiB host reserve, 2GiB RSS per worker, 64MiB vector
workspace. Qualify literal scalar recurrences before launching measurements.
Retain source identities, all case results, guards, elapsed time, memory and
child exit/reaping evidence; exercise cancellation at startup and worker phase.
No CUDA training arm unless later snapshots support a useful guarded policy.
No weights or datasets committed. Existing Tuldok-linked activity is the progress
owner; no separate catalog or Project is created.

Owner: lab integration agent for the user. Private review branch
experiment/pressure-adam-momentum-damping-20261018 starts at
f39c065950dd3df7db7f5590f72eb5df7f505125. Integration target is the prior
experiment/pressure-adam-step-screen-20261018 line. Retain the new branch for
user review until integrated or explicitly retired; no worktree, push or rewrite.
Write set is the new experiment/docs directories and an optional discovery link
only. Concurrent chart, spectrogram and THIRD_PARTY edits remain outside scope.
