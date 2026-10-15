# Next experiments supported by these results

Prepared, not executed. The completed sequence remains in results.md and the
immutable per-trial reports. Do not treat this document as another completed run.
Keep CUDA float32 training, CPU inference and the same native physical gates.

1. Address the measured instability before increasing corpus size. The recurrent
   candidate improves saved-state error but amplifies all three finest grid
   modes, and actual independent simulations hard-stop after 4–6 AI steps.
   Retain the physical-residual input and compare an explicitly mode-informed
   objective with the current objective. Manufactured compatible Neumann modes
   have analytical pressure labels; include low, intermediate, finest and mixed
   modes at positive/negative amplitudes. Measure the residual-update gain
   `||r-A M(r)||/||r||`, and penalize amplification alongside actual pressure and
   gradient error. Passing sampled modes is evidence for those modes, not proof
   of global convergence. Zero correction alone can satisfy a non-amplification
   bound, so actual accuracy must remain a separate gate.
2. Continue the bounded known-case optimization control. All fitted candidates
   remain above 1e-6. Compare correction-RMS output units against the existing
   previous-pressure RMS, holding architecture/objective/schedule fixed, then
   compare optimizer/schedule only after representation is fixed. Retain the
   rounded float32 ideal correction as the representability control. Do not
   relax the physical thresholds or change precision to hide the learning gap.
3. Test a stable smoothing/coarse decomposition as a separately labelled control.
   Existing full analytic spectral correction solves the equation; it is an
   algorithmic control, not AI accuracy. A known high-frequency smoother plus a
   learned coarse correction may prevent the measured checkerboard amplification.
   If tested, identify all algorithmic work and measure total CPU cost; do not
   report native refinement or fallback as a successful raw model prediction.
4. Re-run independent native/AI trajectories on the three retained scenarios
   after source/CPU qualification. Preserve 15 native warm-up steps, actual
   previous pressure and hard physical stops. The criterion is complete qualified
   trajectories, not a lower teacher-state median. The unchanged UI deployment
   adapter requires an explicit new contract before consuming these checkpoints.
5. Admit the prepared 16-simulation/32-state varied cohort only after known-case
   fitting and stability are credible, then larger populations and several seeds.
   Preserve equal update/sample exposure comparisons and shape-aware batching.
   Reserve a new untouched population before final candidate selection; all
   populations/scenarios evaluated in this tranche are development evidence.

The highest-value result here is a concrete failure mechanism to test: a learned
approximate inverse can be linear, improve recorded residuals and still amplify
unseen high-frequency errors during real simulation. More training data alone
has not been shown to remove that mechanism.
