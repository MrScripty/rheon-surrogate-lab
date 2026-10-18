# Execution ledger

2026-10-18: Jeremy admitted the recorded CPU-only aggregate-gradient diagnosis.
Isolated a new branch from 04b5c39, preserving unrelated dirty files. No training,
optimizer updates or new simulations are authorized under this diagnostic plan.


Implemented aggregation from streamed dot products, with an independent direct
float64 vector reconstruction for each mean and component projection. The literal
2D oracle covers negative signed influence, zero/undefined direction, cross-term
derivatives and a cap that reverses target alignment. Actual C69 joint-derivative
mutation retains scalar values and is rejected at the exact expected guard;
valid control matches exactly. The guard worker completed/reaped in 2.570 s.

Cancellation exercised the actual three numerical workers before publishing any
result. After the initial passing cancellation check, tightened the pool cleanup
to signal all groups before one shared ten-second grace; the final cancellation
check passes on that final source. Both receipts retain their own source hashes;
cancellation-final.json is acceptance evidence. Three children and parent reaped,
return codes -15 and no partial published result. No user processes affected.

Actual pool completed in 5.711 s. C53/C66/C69 worker times 4.970/4.995/4.989 s,
peak RSS 732.69/729.51/735.08 MiB. One CPU/Torch/BLAS thread on CPUs 21/22/23;
CUDA hidden and uninitialized. Fitting tensors 297,152 bytes, live vector bound
14,800,640 bytes; raw vector payload 59,202,560 bytes per checkpoint under the
64 MiB disk cap including headers. Raw vectors remain ignored. All children reaped.
Existing Pumas runtime CPython 3.14.7 / Torch 2.14.0+cu132 / NumPy 2.5.3;
exact runtime manifest identity and worker commands are retained in execution.json.

Matched captured numerical objectives/normalization/amplitude and exact CPU fitted
pressure replay. Direct captured versus reconstructed joint gradients pass 24/24;
all independent operator labels <1e-10. C53/C66 component geometry and full losses
replay their historical diagnostic exactly. Separate float32 backwards remain
limited on 1/8 C53, 7/8 C66 and 6/8 C69 cases; worst 0.2204% relative L2 at C66,
0.004446% at C69. Limits are retained, not silently passed or a changed pressure
threshold. Checkpoint archives match network arrays only; Adam replay unavailable.

Finding: case 7's full-gradient cosine with the raw mean is -0.703992 at C53,
-0.835282 at C66, +0.597549 at C69. With per-case cap 1 these become -0.692812,
-0.646318, +0.664738. At C69, final and weighted-early cosines are both positive
(+0.596962/+0.618353 raw; +0.664544/+0.670085 capped). Thus the final C69 raw mean
locally descends those objectives; the hypothesis that it still directly opposes
the weak case at that checkpoint is contradicted. This does not reconstruct the
training path or Adam's preconditioned/momentum direction.

C69 cases 0 and 2 supply 61.293% of summed raw gradient lengths; capping reduces
their combined norm budget to ~25%. Nevertheless cases 0/1 oppose both means.
C53 opposed cases remain 6/7; C66 opposed cases change 3/6/7 -> 1/6/7. Counts
stay 2/3/2. No cap removes target opposition at C53/C66, and at C69 target
alignment was already positive. C69 has zero cases with early-gradient norm
exceeding final, versus five at C66. These endpoint observations do not prove
which mechanism stalled C69 or that any proposed policy improves pressure.

Raw/global-clipped mean L2 at C69 is ~1; capped common direction L2 ~0.357191.
Comparing their directional derivative magnitudes without that scale difference
would be misleading. The report shows both norms and explains negative derivative
versus positive gradient alignment, raw norm budget versus signed contribution.
No parameter, buffer, training state, physical target, deployed model or dataset
changed. C53 remains deployed; no holdouts consumed or training started.

Evidence verification and all-checkpoint/all-term browser checks pass, including
retained plotted/table scalar values, keyboard target details, explicit absent
Adam state, 14 precision limitations and 430px layout. Browser reaped. New offline
report linked from the cohort page; only that owned link slice is staged from
otherwise dirty chart/helper files. Other chart work changed during this task;
its C69 import/link updates are preserved outside this slice. Next recommendation
is a bounded unchanged-policy C69 replay capturing actual optimizer state/deltas,
not blind longer fitting or an unqualified cap/loss change. Not executed here.
