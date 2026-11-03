# Prepared next steps

These steps are prepared, not running. Keep the retained learned control.
Targeted checkpoint 5bbdc315… fixes the two original shape regressions and reduces
mean calls, but higher worst pressure difference prevents promotion. The next
work should target accuracy at comparable work, rather than add updates alone.

1. Diagnose the remaining 31×11×7 and 31×15×23 pressure witnesses on consumed
   cases. Compare pressure/gradient error and residual modes against an independent
   exact solve on the same RHS; distinguish isolated solve error from autonomous
   state differences. Pressure difference is already present after valid warm-up
   and is non-monotonic over time. Do not assume either cold start or runaway
   instability. A CPU experiment with extra fixed corrections or stricter stopping
   can map accuracy versus actual calls/cost, under a separately named inference
   policy and the same cap. Keep that distinct from a checkpoint improvement.
2. Run a controlled late-residual training comparison. Current training unrolls
   two corrections, while inference needs more. Thirty-two independent late labels
   are qualified and source bound for the retained control; prepare matching
   late residual directions for the targeted checkpoint and larger shapes before
   its fit. Use fixed sampling/refresh stages and a relative correction/gradient
   objective appropriate to small labels. Preserve an unchanged recipe control,
   original normalization and real/analytical exposure. Keep float32 neural values
   and existing physical gates. CPU preparation can overlap one CUDA optimizer.
3. Integrate the 16 prepared larger real water parents through an explicit,
   versioned packet/view contract. Preserve their CPU-teacher/native paired
   provenance, whole-scene splits, cubic spacing and force metadata. Start with
   matched post-warm-up states per parent; adding all 49 correlated states would
   also change temporal exposure. Audit train-only normalization/amplitude and
   either freeze the original scales or test that change separately. Compare
   old 16 versus combined 32 parents without mixing this data effect into the
   late-residual objective test. Pressure-only exports cannot stand in for full
   velocity/tracer or restart snapshots.
4. Retain the eight newly consumed final scenes as development data. The prepared
   independent-seed confirmation recipe/population remains unexecuted because
   the full promotion criterion failed. Preserve the unused population until
   a new prospective protocol admits it. Repeat a robust complete improvement
   across independent seeds before definitive promotion; do not search seeds
   until a failing recipe happens to look favorable.
5. Keep the 127×5×5 failure attached to every claim about resolution support.
   Larger/elongated operator coverage or global/coarse communication remains
   a separate investigation if late data cannot address it. Input shape admission
   does not prove arbitrary-shape accuracy. GPU inference, application-adapter
   migration, broader fluids/boundaries and isolated timing/crossover benchmarks
   remain separate work.

Reserve whole final simulations before each new fit, freeze candidate selection,
then evaluate once. Record updates, sample/cell work, actual network calls,
pressure/velocity consequences, physical failures, all elapsed boundaries and
process/custody outcomes as in this study.
