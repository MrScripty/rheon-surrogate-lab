# Next proposal: observe the optimizer before changing it

Keep C53 deployed. The completed CPU diagnosis does not establish a benefit from
per-case capping or removing early feedback. Capping equalizes gradient lengths,
but leaves case 7 opposed at C53/C66 and the same count of conflicting cases at
all three checkpoints. At C69 both final and early losses already align with the
raw mean. Endpoint Euclidean geometry cannot explain the actual stalled training
trajectory or Adam's transformed updates.

Propose one bounded diagnostic replay of the unchanged C69 training policy,
retaining optimizer evidence rather than testing a new policy immediately.
Keep C53 initialization, samples/order, objective, 200 aggregate updates and
1,600 examples, rates, float32 and physical target unchanged. This is a replay,
not a longer training run or a new accuracy claim.

Before admission, qualify instrumentation that observes without changing the
optimizer/RNG path. At a small predeclared set of updates (for example 1, 10,
100, 150 and 200), retain exact network state, Adam moments/step counters,
accumulated and clipped gradients, and actual parameter deltas. Record per-case
loss/residual before and after those updates. Declare output limits through the
existing Tuldok recipe; target <=64 MiB of additional retained diagnostics.
Do not write unbounded per-update vectors or change current custody/runtime owners.

Then run one journaled CUDA replay, fresh CPU restore and final weight/quality
comparison to retained C69. Preserve any reproducibility discrepancy explicitly;
do not assume bitwise agreement without checking it. Record diagnostic overhead
separately from optimizer work and retain full runtime/memory/cessation evidence.

Use the saved states for bounded CPU analysis: compare the actual Adam delta with
the weak case's full/final/early gradients and the proposed capped direction;
check update magnitude around LR decay, and compare predicted local change with
measured pre/post change. This can separate wrong-direction updates, small steps
and nonlinear finite-step effects. Qualify any CPU Adam reconstruction against
the captured delta using a declared numerical tolerance; never invent moments.

Choose one subsequent correction from that evidence: optimizer balance, a guarded
early-pass objective, or schedule/step magnitude. Do not combine them in one test
or expand to 24 cases while the eight-case gate fails. No replay or new training
is started under the completed CPU diagnostic plan.
