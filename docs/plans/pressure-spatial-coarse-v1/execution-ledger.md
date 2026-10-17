# Execution ledger

Created local branch from fe00afb. Original root untracked user assets preserved.
Fresh Standards MCP selected 22 policies without unresolved facts. The bounded
prototype uses the existing captured current-0.1 checkpoint and selected Pumas
runtime. Inspection of the constructed model proves no active GroupNorm/SiLU or
bias tensors, correcting the inherited-source reading in the prior proposal.
One initial read-only inspection used nonexistent project-03 instead of actual
project-02 and failed before model import; no mutation or result was produced.

Four focused tests pass: volume/constant identities, zero initialization and RNG,
nonconstant far-field response/linearity, and input/parameter autodiff. CPU
prototype passed in 12.353403 seconds; CUDA deterministic forward/backward passed
in 3.194819 seconds, no optimizer or simulation advancement. The zero branch
reproduces the retained checkpoint outputs exactly on all seven tested shapes.
Parameter count 370016 -> 374112. Median standalone raw-network CPU overhead on
16/24/32/40 cubes is 5.14/1.48/0.63/0.80 percent. Rectangular grids 33x11x9 and
25x16x23 cost 10.07/1.56 percent; degenerate 1x3x7 costs 28.15 percent and is
outside the preregistered four-cube cost gate. These short timings are
architecture admission evidence, not production solver latency.

Freeze control/candidate configs from the exact current-0.1 recipe. Candidate
alone uses an explicit captured model overlay; all physical/loss/data/stream
owners stay unchanged. No training has started at this source-freeze boundary.

Source review caught missing inference packaging for the new tensor before any
training admission. Extended only current lab packaging/numerical reconstruction:
closed v2 mapping carries an explicit coarse flag; no new-reader v1 fallback.
Historical package readers remain unchanged in their own directories. Nonzero
synthetic coarse weights prove exact raw and two-call pressure conversion across
three consumed grids. Two negative fixtures reach the intended tensor-inventory
and coarse-flag errors; existing nine protocol tests pass. No learned accuracy
claim is made from synthetic conversion coefficients. Both captures preserve all
shared source/input/normalization identities, original seeded tensors and RNG;
the only candidate tensor starts at zero. SDK preflight initially used system
Python lacking jsonschema; corrected to the existing application environment for
that read-only check, without admission/mutation replay.

Both review axes accept source/capture/conversion admission at b492c84. Original
owner identity/catalog/capability negotiation passed, but runs.list exceeded the
300-second HTTP limit. A separate bounded read-only transaction in the same
catalog found26Runs, none active, in0.431seconds; no broad read or mutation was
replayed. All actual admission/custody checks remain with the original HTTP/SDK.
Submitted finite control and candidate Study controllers with their own durable
journals. Existing one-slot CUDA scheduling orders training. Sequential CPU
qualification waits for exact managed restore and first-frame success. Started
only a bounded external preview publisher on the existing project paths.

Prepared matched actual-training comparison and a separate raw CPU-forward timer.
The timer calibrated on a retained model with four consumed cubes and31samples
per grid; only its execution/calibration is accepted here, not final latency.
After broad pools drain, final raw forwards and paired worker timings run in
reversed order twice, with their different measurement scopes retained.

Report review found two provenance gaps before final diagnostics: explicitly bind
new overlay bytes to the qualified CPU prototype and exact64x64float32zero tensor;
bind timer artifact/source hashes before loading and recheck through completion.
Four focused comparison fixtures pass, including changed source/shape/precision.
The revised raw timer calibrated in9.030596seconds; preliminary calibration is
preserved. Stopped only the idle completion waiter before any child work, then
replaced it to pass the explicit prototype identity. CUDA/Study/CPU-qualification
owners were unchanged, and no scientific phase or mutation was replayed.

Shared-RHS report review required binding the actual producer/source closure,
zero execution counters, one early/late row per unique ordered case, canonical
spectral bins and identical input signal energies. Added these checks before
new-model summary use. Actual retained-data self-comparison remains exact;
six isolated malformed pair/source/stage/bin/signal/counter fixtures reach the
intended errors in0.166020seconds, with zero network calls. Earlier calibration
remains immutable and is not evidence for the revised source. Calibration
helper is retained locally under command-sources/coarse-common-calibration.py.

Candidate Run d0b98276805b40a2942e2b123fe35330 failed/drained after last logged
completion1300/5000. Custody confirms exit1/nonzero_exit, ECHILD and closed spawn;
stdout/stderr hashes match. Existing clip_grad_norm_ error_if_nonfinite=True
reported a nonfinite total gradient norm. This does not identify the individual
gradient cause or exact failing update. No final artifacts were published; no
candidate CPU restore, trajectory accuracy or matched completed-fit result exists.
Run wall207.358649seconds includes runtime startup; optimizer-only time unavailable.
Study command wall962.555611seconds includes admission/queue/readback. Guards and
thresholds remain unchanged; no numerical retry or replacement training was made.

Stopped only the idle paired-completion waiter before any child phase. Retained
failed marker and closed the owned candidate Study through its original journal.
The first closure helper attempted to read an internal custody_identity field
from the public SDK projection and stopped before any mutation; the remaining
closure verified the retained internal identity against that public terminal Run
and the custody receipt, then journaled one Study stop. Both review axes verify
failure classification and the common-report corrections. The independent control
CUDA Run completed and remains on its original restore/CPU qualification path.
Finish its physical/diagnostic/timing evidence; all eight unused holdouts remain
untouched and no candidate can be promoted.

Fresh control completed5000CUDAupdates in498.237518optimizer seconds and
502.045229trainer seconds; full Study1542.295358seconds and fresh CPU restore
2.905135seconds. Actual50000forwards/425703026cells, all five exact artifact pairs,
shared data/source/sampling and both initial/final checkpoint identities match the
retained current0.1 control (qualified custody monitor change remains separate).
First-frame fixture passed9.023094seconds, transfer diagnostic8.973660seconds;
full74scene campaign follows. New control is qualified independently, not reused.

Prepared next diagnostic by testing a finite3-4vector in installed float32 Torch.
Both CPU and CUDA ordinary norms overflow, while a scaled float32 calculation
returns the independent expected5e20norm and0.6/0.8unit direction. Probes take
0.951540/1.590016seconds internally, with no network/optimizer/simulation calls.
This demonstrates a possible norm-reduction mechanism, not the actual candidate
cause: its failing gradient components were not captured. No clipping algorithm,
model precision or finite/physical guard was changed. Probe sources stay local
under command-sources; small receipts retain the exact identities.
