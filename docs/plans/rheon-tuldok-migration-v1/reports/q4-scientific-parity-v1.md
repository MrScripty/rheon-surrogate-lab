# Q4 scientific parity

Passed for the frozen current packages: control `89361859195666963962effb05b9f7484f4192a1345dfdbcbc0e0685bb60cb4a` and original desktop `2398b86e41053a52647ce9b9dc58528da47e3094d2b7cc721a54589c4b68445c`.

Fresh old references reproduce the fixed saved control frame and retained aa82 prediction exactly. Current packages match input mapping, float32 preprocessing and raw output, complete float64 pressure reconstruction, and zero-residual bypass bit for bit. The control retains its exact six-call stopping trace. Unsupported physical/history contexts are rejected before network entry.

The source-bound current public native probe completes 15 native steps and exactly one model step. Its actual input arrays and prediction match the bound archived frame exactly. The matched native control passes, both children are reaped, and there is no native refinement or fallback.

Independent two-cell and neighbor checks pass. The same-RHS pressure relative error is 2.7527900530109385e-7, covered by the fixed 7.192272482718904e-7 bound; this differs from independently advanced native–AI trajectory drift. Numerical asset bytes remain unchanged: 25,765 files / 5,744,383,179 bytes. The new current runtime descriptor additionally measures 1,047 file links.

[Scalar results and exact receipt references](q4-scientific-parity-v1.json) retain command/source/package identities, elapsed times and RSS. Raw evidence remains under the primary lab's `runs/dependency-migration-v1/numerical-qualification-v1/`. Seven harness tests pass. Timing is advisory and uses a warm, uncontrolled OS cache.

This is numerical migration parity, with zero optimizer updates and no CUDA initialization. It does not promote either model, waive aa82's retained accuracy failures, consume holdout data, or satisfy Q7's full migrated cohort or timing procedures. Desktop lifecycle and visual qualification are separate.
