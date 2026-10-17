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
