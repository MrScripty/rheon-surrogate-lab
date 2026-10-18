# Next proposal: balanced shape aggregation

Keep C53 deployed. No new training has started. The completed CPU diagnosis
supports a bounded aggregation test before changing the model or feedback loss.

C66 has 16/28 opposing full-objective shape pairs; canonical + final, differentiated
jointly without early feedback, still has 13/28, including two below -0.5. C67 has
17/28 full versus 12/28 without early, including five below -0.5. Thus removing
early feedback does not remove strong shared final-solve disagreement. Early
feedback nevertheless dominates the gradient on five of eight cases at both
trained checkpoints, and opposes final feedback on C67 case 1 (cosine -0.597).
These observations support more than one optimization mechanism; they do not
prove that any proposed optimizer policy will improve fitting.

Propose one matched CUDA screen on C66's eight X-fundamental cases, C53
initialization, normalization, architecture, seeds, four passes and unchanged
loss/early weight. For each complete balanced eight-case cycle, accumulate the
mean of its eight losses' gradients, clip once, then make one Adam update.
Backpropagate each example separately and release its graph before the next;
do not hold eight graphs or retain an unbounded dataset cache.

Match example work: 200 exposures per case, 1,600 loss examples, 6,400 network
forwards and 7,424,000 forward cells, excluding the same diagnostic overhead.
This makes **200 optimizer updates rather than C66's 1,600**. It therefore changes
update frequency and Adam's moment history; it is not an equal-update comparison
or proof that gradient conflict alone caused a difference. Do not multiply the
example budget by eight merely to restore the old update count.

Keep C66's learning-rate schedule by exposure, not old update index: cycles 1–100
at 1e-4, 101–150 at 1e-5, 151–175 at 1e-6, 176–190 at 1e-7, 191–200 at 1e-8.
Use the same seeded order within every cycle. Record example count, optimizer
update count and exposure independently; diagnostic observations remain every ten
exposures per case. Reusing the old single-example update counter or scheduler
without conversion would invalidate this comparison.

Before CUDA admission, qualify one aggregate cycle on CPU against explicitly
averaged captured per-example gradients, global clipping once, no per-case Adam
step, exact unchanged train-only labels, and bounded memory. Name float32
summation tolerance and numerical limitations explicitly. The present component
study's exact joint-gradient oracle passed all 24 cases, but 14 separate-gradient
sum checks exceeded 5e-6 (worst 0.2204% of full-gradient L2); do not erase that
finding or interpret small alignment differences as exact mathematical facts.
Exact historical Adam moments were not saved, so no frozen-Adam replay is claimed.

Then one journaled Tuldok CUDA Run, fresh CPU restore and artifact parity. Retain
the all-case four/eight-pass fitting gate, no one/two-pass regression against C53,
128 consumed real requests, 240 manufactured regressions, actual trainer/workflow/
CPU time and memory. Stop if the fitting gate fails. If it passes, propose the
24-case fitting screen and separate real-frame retention; do not promote directly.
The earlier conditional C68 remains skipped; select a fresh model ID when this
separate experiment is admitted.
