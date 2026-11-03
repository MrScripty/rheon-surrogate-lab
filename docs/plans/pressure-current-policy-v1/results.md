# Current-policy feedback results

Measurement complete. Neither candidate meets every unchanged population accuracy/work gate; no model promoted and eight unused scenes remain untouched.

The frozen retained 0.03 network generated native-amplitude early/prefix-8 pairs from 32 recorded training frames belonging to 16 simulation parents, plus 112 manufactured states across 14 admitted grids. These 144 pressure examples are not new CFD simulations. Independent labels and stencil identities qualified every pair in 31.263 seconds and 4,500 network calls. Maximum worker RSS was 0.880 GB and minimum available host memory was 46.762 GB; all workers were reaped. Median late correction RMS was 9.37004e-8 Pa. Directions differ from the older scaled bank, with median unit-direction difference 1.14065. Rescaling the same RHS changes predictions by median 4.2754e-7 relative, which argues against small absolute amplitude alone explaining late error. This does not establish temporal aliasing or a causal noise floor.

Both v11 configurations add the same current-policy stream at weight 0.1 and retain the original low-band weight 0.03, streams, initialization, seeds and 5,000-update schedule. Only the selected early/late payload differs. Actual source, inputs, teacher, normalization, runtime, initial checkpoint, sample order and work budgets match. Each arm executes 50,000 optimizer network forwards over 425,703,026 cells. The historical 0.03 fit lacks this new work counter; equal updates do not mean equal compute. Inference stays at 370,016 parameters and 1,485,452 weight bytes, with no added input or layer.

| Arm | Optimizer s | Trainer s | Complete managed Study s | Peak CUDA allocation GB |
|---|---:|---:|---:|---:|
| early | 504.952 | 508.683 | 1480.719 | 0.870 |
| late | 631.954 | 643.475 | 1368.268 | 0.870 |

Complete Study clocks include admission, queueing, verification, fresh CPU restore and artifact readback. The fits overlapped different CPU activity windows, so their wall-time difference cannot be attributed to the selected feedback stage. The network stays float32 with AMP and TF32 disabled; physical assembly and accumulation stay float64. All five artifact pairs match exactly after each fresh CPU restore. Each candidate passed its first-frame check before the full campaign.

| Model | Physical scenes | AI steps | Mean/max calls | Independent checks | CPU campaign s |
|---|---:|---:|---:|---:|---:|
| reference003 | 74/74 | 8523 | 7.500997 / 11 | 233 | 710.910 |
| current-early | 74/74 | 8523 | 8.211193 / 16 | 233 | 742.814 |
| current-late | 74/74 | 8523 | 7.371817 / 15 | 233 | 694.736 |

The retained 0.03 campaign is explicitly reused, supported by exact package equivalence and a fresh fixture; it was not reexecuted. Each fresh candidate ran 74 episodes with 16 single-thread workers, and all children were reaped. Sealed water, cubic cells, valid previous-pressure history, 15 native warm-up steps, physical limits, stopping criteria, the 32-call ceiling and model-only inference stay fixed. Concurrent campaign clocks include startup, native controls and oracles; they are not isolated inference latency.

| Population | Prior accurate worst paired pressure | Retained0.03 | Early | Late |
|---|---:|---:|---:|---:|
| original58 | 3.14479467e-06 | 3.61037514e-06 | 2.94041996e-06 | 3.31635714e-06 |
| late-amended8 | 1.34895491e-06 | 1.08573912e-06 | 1.22631536e-06 | 1.08971751e-06 |
| minimum8-failed8 | 1.41516687e-06 | 1.6874652e-06 | 1.90452316e-06 | 1.65788544e-06 |

Each population is tested separately against all three historical comparators. The [decision](reports/qualification-decision.json) records every failed pressure, mean-call and maximum-call gate. Physical qualification alone does not select a model. The [full summary](reports/regression-summary.json) and [CSV](reports/regression-summary.csv) retain every scene. Paired trajectory drift and same-RHS pressure error are distinct measurements.

current-early: max same-RHS mean-free pressure relative error 7.08803947e-07, independent operator defect1.66634313e-13; every recorded pressure bound covers its oracle = true. Peak summed worker/child RSS 11.658 GB, minimum available host memory 49.396 GB.

current-late: max same-RHS mean-free pressure relative error 6.95180734e-07, independent operator defect1.63870128e-13; every recorded pressure bound covers its oracle = true. Peak summed worker/child RSS 11.270 GB, minimum available host memory 49.900 GB.

Bounded CPU preparation compared five checkpoints on exactly 144 paired RHS inputs. Each model uses 1,152 calls across native and rescaled two-call early/late probes. Independently qualified labels and pair hashes match. These training-only diagnostics do not select a model or consume held-out scenes.

| Model | Early median/max two-call pressure error | Late median/max two-call pressure error | CPU diagnostic s |
|---|---:|---:|---:|
| control | 0.01314725 / 0.1086961 | 0.0254078 / 0.3018384 | 12.705 |
| previous01 | 0.008364216 / 0.1704104 | 0.02515178 / 0.2986909 | 12.409 |
| weight003 | 0.009085401 / 0.1757488 | 0.02293114 / 0.2526953 | 11.939 |
| current-early | 0.007003463 / 0.07395667 | 0.02216356 / 0.2107506 | 11.933 |
| current-late | 0.01070911 / 0.1963598 | 0.02048052 / 0.1694017 | 11.309 |

[Common-RHS frequency/amplitude bins](reports/trained-common-rhs-summary.json) retain absolute signal/error energies in Pa² and coefficient/full-L2 intervals. Low spatial modes have indices 0–3 in each axis, excluding DC. Aggregate energies are weighted by grid and case. These are spatial modes, not temporal Hz or evidence of a causal noise floor. The [manufactured diagnostics](reports/diagnostic-summary.json) retain 84 fixed spatial/amplitude rows per model, separately from trajectory errors.

[Serial timing](timing.md) covers two repetitions per model in reversed order on core 22 after broad CPU work drains. Each uses a fixed one-metre cube at 16³, 24³, 32³ or 40³ resolution, 15 native warm-up steps and eight AI steps. The AI-side diagnostic timer includes an additional same-request native reference solve, request capture, learned correction, IPC and protocol logging. Startup is separate. External sparse capture and spectral-oracle work are absent from timed episodes. These branch ratios do not measure standalone model or solver speed; no native-time estimate is subtracted. Repeated physical frames and call sequences match exactly. The [24 raw rows](reports/serial-timing.json) retain clocks, memory, hashes and scope. Two repetitions under desktop/cache variability do not establish a scaling law.

Preparation for a subsequent experiment checked the parameter-free global-average shortcut on six analytical fields across three consumed shapes: 12 float32 forwards in 2.018 seconds, with 0.845 GB peak RSS and no optimizer or simulation. Removing the pressure gauge leaves relative differences of 2.253e-7 to 3.748e-6, consistent with rounding. In exact arithmetic, the selected linear decoder with replicated boundaries carries this shortcut as a constant removed by the gauge. The spatial encoder/decoder still communicates information. This does not prove the overall low-frequency cause, and no saved checkpoint or inference package changed. See the [probe](reports/global-mean-gauge-probe.json).

Late Study 02 reached 5,000 updates but drained with output_limit before publication. Stderr was empty and retained files were below 32 MiB. A real profile.json rename between listing and lstat reproduces a monitor failure compatible with the receipt; the exact historical exception was not retained. No unpublished weights were reused. Separate Tuldok fix 709d0dd passed 39 tests and both reviews, then deployed as current-06 in 27.722 seconds with zero active Runs and all 20 prior states, owner, catalog and capacity settings preserved. The failed Study was explicitly closed. A client postcondition shape error was resolved from its acknowledgment without replay. Fresh late Study 03 uses identical capture 02 and initialization. The failed attempt remains separate from the two successful fits.

The original Tuldok catalog owns the actual Models, Studies and Runs. Existing Rheon project paths show recorded pressure previews, not live tracer or velocity flow. See the [ledger](execution-ledger.md), [issues](issues.md), [environment and commands](reports/command-environment.json), [matched training](reports/matched-training.json), and [next proposal](next-proposal.md).

The timing scope correction also limits interpretations of older timing records
produced by this comparison-engine path. Their raw measurements and historical
reports remain preserved. The [scope review](reports/timing-scope-review.json)
records the current engine source and proves the corrected summary retains every
numeric timing row and source record. No standalone speedup or slowdown is inferred.
