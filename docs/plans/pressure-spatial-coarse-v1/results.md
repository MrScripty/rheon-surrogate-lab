# Spatial coarse correction: results

The new coarse path passed architecture and conversion qualification, but its
CUDA fit failed the existing finite-gradient-norm guard. It has no final learned
checkpoint, CPU accuracy result or promotion. The independently trained control
reproduced the retained current-0.1 weights exactly and passed all 74 development
trajectories and 233 independent sparse pressure checks. Eight unused whole scenes
remain untouched. This experiment demonstrates no accuracy improvement.

## Intervention and controls

The candidate adds 4,096 parameters to the actual configured linear, bias-free
370,016-parameter network: restrict into 64 spatial bins, apply a learned 64×64 matrix,
interpolate to the original grid, remove its mean and scale by
`(longest_axis_cell_count / 16)^2`. Initial new weights are zero; surviving tensors, RNG, inputs,
normalization and sampling remain identical. This addresses global spatial
communication without a native or spectral inverse during inference. Prototype
and nonzero synthetic conversion checks establish implementation behavior;
they do not establish learned pressure accuracy.

Both managed captures froze source 17c787da99aa454f82ae80ad290bf0085f9ddeec.
The narrow deployment mapping/conversion was qualified at b492c84. The original
Tuldok catalog owns both Studies and all Runs; selected Pumas runtime is
Python 3.14.7, Torch 2.14.0+cu132, NumPy 2.5.3, RTX 5090 Laptop GPU. Network tensors
and Adam moment state are float32; deterministic algorithms on, AMP/TF32 off. Existing
physical arithmetic remains float64. See[method.md](method.md) for physical
restrictions, gates, input reconstruction and measurement boundaries.

| Outcome | Fresh control | Coarse candidate |
|---|---:|---:|
| Parameters |370016|374112|
| Completed optimizer updates |5000|Unknown; last logged1300/5000|
| Actual optimizer forwards |50000|No completed work total|
| Optimizer elapsed |498.238s|Unavailable|
| Trainer elapsed |502.045s|Unavailable|
| Final learned weights |1485452bytes; exact retained SHA|None published|
| Fresh CPU restore |2.905s; five exact artifact pairs|Not run|
| Physical development trajectories |74/74passed|Not run|
| Independent sparse pressure checks |233; all bounds cover oracle|Not run|
| Eligible for untouched-scene evaluation |No, unchanged population gates|No completed fit|

The control's 50,000 forwards cover 425,703,026 cells. Its optimizer clock is faster
than the retained control's 631.954 s, but source and final weights are numerically
identical: this is a repeat under different machine conditions, not evidence of
an implementation speedup. Complete control Study elapsed 1542.295 s includes
queue/admission/handoff/CPU restoration/readback and must not be called GPU
training time. Candidate Run start-through-drainage 207.359 s includes bootstrap;
Study command 962.555 s includes queued waiting. No optimizer-only candidate clock
was recovered. Exact source/stream/work/weight proof is in
[control-training-reproducibility.json](reports/control-training-reproducibility.json).

## Failure and limits

The failed Run is d0b98276805b40a2942e2b123fe35330, in Study
ddf97abc6fa54ebc8fa750146baed543. PyTorch clip_grad_norm_ with
error_if_nonfinite=True reported a nonfinite total gradient norm. Exact failing
update and individual gradient cause are unknown. Sparse logged losses use
different samples and do not prove a steadily growing loss trend. Custody
confirms exit 1/nonzero_exit, drained/ECHILD and closed spawn. No final artifacts
were published; only the unsuccessful attempt's initial local checkpoint exists.
The owned Study was closed once through its original journal. No guard or
physical threshold was relaxed and no numerical retry started.

Retained[stdout](reports/candidate-training-stdout.log),
[stderr](reports/candidate-training-stderr.log),
[failure summary](reports/candidate-failure-summary.json) and
[custody](reports/candidate-failure-custody.json) identify the attempt. The
sequential qualification controller stopped at this known failed marker only
after successful control qualification; its exit 1 is not a control physics
failure. The separate control-only completion owner finished diagnostics/timings.

Independent CPU/CUDA probes show a possible mechanism: ordinary float32 norm
reduction overflows for finite gradients [3e20, 4e20], while stable scaling yields
the representable expected norm 5e20 and unit direction [0.6, 0.8]. This does not
prove that the failed candidate had finite components. No production clipping
change followed. The[next proposal](next-proposal.md) first captures exact
update/stream/component and stage diagnostics in a capped managed CUDA screen;
no diagnostic replay has started.

## Control accuracy and timing

| Development population | Qualified | Mean/max calls | Worst paired pressure relative L2 |
|---|---:|---:|---:|
| Original58 |58/58|7.19315/11|3.316357140e-6|
| Late amended8 |8/8|7.73009/8|1.089717514e-6|
| Hard minimum8 |8/8|8.34071/15|1.657885439e-6|

Across 8,523 AI steps: mean 7.371817 calls, max 15. Same-RHS sparse pressure checks
and paired trajectory error are distinct measurements. Every population still
uses all original comparator gates. Passing physics is insufficient for
promotion. The 144 shared-RHS pair summaries and stationary transfer summaries
exactly match retained current 0.1; these are fresh measurements, not reused
campaign evidence. The[regression](reports/regression-summary.json),
[transfer](reports/transfer-summary.json) and
[frequency/amplitude](reports/common-rhs-summary.json) records retain tails,
absolute energies, calls and source identities.

Fresh full 74 campaign: 691.829977 s, 16 one-thread workers on cores 5–20, all children
reaped. Observed maximum worker RSS sum 10.10 GiB; minimum available 46.32 GiB.
Accounting is 2 GiB/worker plus 8 GiB available reserve; broad pools run sequentially.
The 144-pair diagnostic took 11.457571 s. Minute resource-monitor samples are not
peak RAM measurements; the completed trainer reports CUDA peak allocation
870,009,344 bytes. The candidate has no completed trainer memory receipt.

| Cube | Raw network forward median, rep1/rep2 | Paired diagnostic AI-branch mean, rep1/rep2 |
|---|---:|---:|
|16³|5.705/5.867ms|62.967/68.029ms|
|24³|32.399/35.505ms|340.833/352.875ms|
|32³|36.703/37.700ms|572.312/619.079ms|
|40³|120.172/119.152ms|1475.521/1479.265ms|

One core 22; raw timing has 31 samples/grid after warm-up. Two paired repetitions
have exact physical signatures/call sequences, 15 native warm-up steps and 8 AI steps
per grid. Paired cost includes an additional same-request native diagnostic
solve, capture, correction, IPC and logging; raw forwards exclude all of those
and physical reconstruction/stopping. These columns are different scopes and
neither is standalone end-to-end AI solver latency. Do not subtract the evolving
native branch time. No learned candidate latency comparison exists.

The interval from first managed Study intent through final control timing
summary is 2426.879656 s (40 m 26.880 s), excluding prototype preparation and final
review/closure. Overlapping phase clocks are not summed. Retain
[phase commands](reports/cpu-phase-clocks.json),
[completion commands](reports/control-completion-phase-clocks.json),
[raw samples rep1](reports/control-raw-timing-1.json),
[raw samples rep2](reports/control-raw-timing-2.json),
[paired timing](reports/serial-timing.json) and
[completion summary](reports/completion-summary.json).

## Retention

Local branch: experiment/spatial-coarse-correction-20261017, based on
fe00afb79f8cc3cef09d5a552ca700526037f0e5. No push or merge. Tuldok remains at
225f8e862f6ee7a6f778522c6e15a5fbc97fe5ca and Rheon at
759d928448dec8b160f4c113e35bf2bf2bd1b9d3; neither source was changed here.
Weights, datasets and orchestration helpers remain under
runs/pressure-spatial-coarse-v1 outside Git. Small committed receipts bind their
identities; historical evidence and root untracked user assets are preserved.
Original application interfaces remain the observation targets; the preview is
recorded pressure, not live tracer flow. The final
[catalog readback](reports/final-catalog-readback.json),
[host process check](reports/host-process-closure.json) and
[application check](reports/live-apps-preserved.json) retain closure evidence.
