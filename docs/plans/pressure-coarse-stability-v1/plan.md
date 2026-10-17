# Overflow-only clipping integration and capped CUDA stability

Status: accepted; clipping integration and1500-update finite stability screen complete. Owner: root. Base: 84525d986cdaba1b5252802e07dc6b43dee9b701.
User authorized integration, qualification and one capped 1500-update CUDA screen.
This plan executes the preceding pressure-coarse-gradient-v1/next-proposal.md.

Write set: this plan/reports; experiments/pressure-coarse-stability-v1; the
clipping owner and gradient_clipping.py in experiments/water-conditioning-v1/training; and
bounded entry-source selection in pressure-throughput-feedback-v1/assemble.py.
Review added the directly affected generic training/project.py source inventory
and its isolated import regression test; shared trainer packages must include
the new numerical dependency.
Historical captures/results, model/loss/physics/configuration and Tuldok/Rheon
sources remain unchanged. Ignored execution artifacts live under
runs/pressure-coarse-stability-v1 in the workspace root.

Integrate the qualified float32 max-scaled norm arithmetic at the trainer's
clipping owner. Call native Torch unchanged first. Only its specific nonfinite
L2 norm guard, with present colocated finite float32 gradients and representable
true total, may recover. Preserve clip1/epsilon1e-6. Re-raise the original error
for nonfinite components, unrepresentable total or unrelated RuntimeError.
No wider model/Adam tensors, new data, gain/rate, loss or physical gate changes.

Acceptance: real CUDA one_update differential checks on consumed fields must
preserve ordinary losses, weights, gradients, Adam and CPU/CUDA RNG exactly.
A loss hook at the same seam supplies the two minimized observed gradient norms:
native guard must fail there before optimizer.step; integrated repair must
match independent direction/unit-norm expectations and complete one finite Adam
step. NaN, Inf, unrepresentable total and unrelated errors preserve rejection
without changing gradients or stepping Adam. Arithmetic checks also cover CPU.
These injected gradients are explicit seam fixtures, not recovered model tensors.

Freeze qualified source before capture. Compare original trainer to the repaired
trainer with only the documented helper insertion and call-site replacement.
Preserve all other original sources, exact5000 configuration, input hashes,
normalization, initialized tensors and five full5000 sampling arrays/1500prefixes.
Use the existing capped observer, journalled admission, custody collector and
terminal Study closure; strengthen qualification binding for the repair.

Then admit exactly one managed CUDA Run in the original Tuldok catalog. Cap
before1501, require1500 successful original CUDA gradient/Adam checks and retain
every repair's exact norms, failure context, losses, elapsed scopes and resources.
Compare all13 historical loss checkpoints through1300; expect first repair1398
only if the original prefix reproduces. Label cap as diagnostic cessation, not
completed training. No final model, CPU restore, full fit, promotion or holdouts.
If actual nonfinite state occurs, preserve it and stop without retry.

Original Tuldok and Rheon apps remain visible with truthful activity and recorded
qualified control preview; no claim of new simulation/model-preview results.
Use bounded one-shot observations, no detached monitoring service. Retain Run,
source/custody, drainage/reaping, Study closure and owned-controller cessation.
Keep optimizer-only timing unavailable unless directly measured; distinguish
instrumented loop, managed Run and full command clocks.

Standards MCP snapshot:v1:aae1fe79-f421-40a7-a45b-bcf37671dc97 selects22 policies,
zero unresolved facts; all exact texts match the preceding accepted snapshot.
The diagnosing-bugs workflow reuses the minimized red-capable norm reproduction;
integration regression precedes managed replay. Independent review before
admission and after results follows the code-review skill, fixed base above.

No implementation remains in this plan. Both independent final reviews passed.
The next proposal covers a separate full fit/CPU qualification; it is unstarted.
