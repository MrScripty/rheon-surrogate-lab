# Lab acceptance evidence

[Results](results.json), [baseline](baseline.md), [build](build.json),
[cold audit](cold-audit.json), [output bounds](output-bounds.json),
[browser](viewer.json), [paired CLI contract](paired-cli-handoff.json),
[actual CNN dataset](cnn-dataset.json), [normalization refusal](cnn-negative.json),
[runtime inventory](runtime-inventory.json).

## Pressure result

Three balanced repetitions/four exposed seeds/120frames preserve the original
checkpoint and algorithms. Median complete-rollout Jacobi30.4746s, SGS24.2339s,
learned4.42045s:6.894x/5.482x. All4320timed frames pass native gates; cold
full-field audit confirms worst velocity9.0048e-8 and tracer5.9089e-8 relativeL2.
All480restored pressure predictions exactly match retained native inputs.
Cloud allocation is four CPU equivalents; all methods retain eight native
threads, model8threads/interop1. This is a new Torch2.8 CPU observation, not
the historical hardware/runtime reproduction. OS caches/load are uncontrolled.
Imports/model setup/warmup/capture/compression/rendering are excluded from the
resident speed ratio; the worker-wide scope is separately recorded.

The outer local wrapper refused a17,322,003-byte viewer payload at its16MiB cap
after successful numerical computation. The cap is now32MiB, and read-only
inspection proves every retained output fits corrected per-file/aggregate bounds.
This repair changes no numerical code or measurements and does not retroactively
turn that refused attempt into a managed completed Run.

Clean CPU training produced the exact historical training.npz SHA3ca87734...
b539184, completed native validation/actual physics ablation, and verified
restored/refitted parameter equality. It performs zero optimizer steps.

## Separate neural diagnostic

The captured original8train/4validation/4test pressure pairs fit the unchanged
1760-parameter NumPy CNN with200actual Adam updates. Loss ratio0.0365122056
reproduces the historical decrease; fresh restore exactly reproduces retained
train/validation predictions, with zero new optimizer updates. Test relative
pressure error7.6643 and native0/16qualification reproduce its negative result.
Actual changed-normalization restore exits1 before outputs. Do not present this
neural diagnostic as pressure-model parity or an acceleration result.

## Supporting checks and remaining boundary

20Rust contract tests pass.30Python tests run:29pass,1optional operator-export
skip. Root/new-probe fmt and Clippy(-Dwarnings), Python compilation and JS syntax
pass. Actual CPU browser load/scrub/playback/orbit/desktop/mobile checks pass.
Original source/checkpoint/freeze and pressure dependency are unchanged.

The separate paired owner still needs real catalog/runtime/actor/material/grant,
managed Runs, verified CLI downloads/restore and reopening evidence. Local
training/restore and schema fixtures do not certify those boundaries. Runtime
inventory was actually validated with explicit optional exclusions, but no
managed snapshot registration is claimed. Cross-cloud-thread message tools are
unavailable; precise inputs and recipes are published in draftPR1.

## Review and CI continuation

[Review/CI metadata](review-ci.json) records successful Actions38531223801
on source21cd8b0, the fresh clone regeneration and the unchanged CNN negative
physics result. Local32Python tests=31pass/1optional skip; the Actions run/job
step API confirms success, but its log archive returned403. No log-derived
counts or completed independent review are invented. CodeRabbit manual review
is acknowledged in progress. [Fresh regeneration](../../../../tuldok/REGENERATION.md)
lets the separate executor generate its own immutable inputs and measure its
new source-dependent split/recipe hashes.

Independent review subsequently completed with two valid lifecycle findings.
Repairc34e248 preserves original launch errors while attempting all cleanup,
and bounds trusted same-group children for direct local timeouts. Local34tests
=33pass/1optional skip; Actions38533860305 passed after those repairs. Both
review threads are resolved. Managed Runs keeps its separate custody authority.
