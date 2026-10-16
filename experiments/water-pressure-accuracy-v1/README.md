# Pressure accuracy diagnostic

Use `configs/study.json` to select recorded checkpoints, consumed scenes,
policies, interpreter, native binary and bounded resources. No new training,
deployment or final-population evaluation occurs here. Numerical/process owners
remain in the existing experiment modules.

`diagnose.py capture` uses the actual native paired engine and existing CPU
correction loop, retaining selected input arrays locally. `replay` evaluates both
checkpoints on the exact same captured fields through 32 actual calls. Its
independent cosine solve is checked by the existing enumerated-neighbor action;
it is an analysis oracle, not an inference refinement. Pressure metrics compare
to the same-input exact solve. Native trajectory metrics compare the independently
evolving AI/reference simulations, a different question.

`repro.py` exits 1 for the original same-input pressure regression and can compare
a stricter candidate policy to the retained original policy. `run.py` delegates
fresh named native trajectories and replays with bounded single-core child
processes. The existing timed command and native rollout owners retain shutdown,
termination, reaping and failure evidence. Never reuse an output attempt name.

All arrays/checkpoints stay under ignored `runs/`. Small methods, per-step
metrics, timings, failures and decisions belong in
`docs/plans/water-pressure-accuracy-v1/`. Stricter stopping is a separate CPU
inference experiment; it does not change the float32 network, physical gates,
retained model or advertised arbitrary-grid accuracy.
