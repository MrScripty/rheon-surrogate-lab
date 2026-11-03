# Prepared follow-ups

These are prepared subsequent experiments; this document admits no new jobs.
Keep the cohort16/two-evaluation checkpoint and its exact normalization as the
new development reference after final evidence is sealed. Preserve every earlier
checkpoint and the measured per-grid tradeoffs. Training remains CUDA; inference
qualification remains CPU with the existing model 1e-6/native 1e-8 profiles.

1. Reproduce the cohort improvement with additional declared initialization and
   sampling seeds. Reserve fresh whole simulations with new physical-setup/seed
   identities for final evaluation. Development results here guide experimentation;
   they do not establish unseen-population or long-term accuracy.
2. Compare prepared cohort16 one-evaluation training against the current two-
   evaluation control, using identical normalization, real states, analytical
   bank, exposure and learning-rate schedule. Then consider a separately named
   four-evaluation trial. Count actual neural/voxel work as well as updates/time.
   Judge fixed-small inference budgets and adaptive counts, not training loss
   alone. A successful raw one-evaluation pressure solve remains unqualified.
3. Compare a separately named tighter stopping profile when pressure-field drift
   matters, retaining the existing acceptance target and float32 calibration.
   Current trajectory drift improves on some scenes and worsens on another.
   Inspect remaining residuals by spatial frequency and shape, using the retained
   signed mode probes. Test feedback on intermediate corrections or hard remaining
   modes as a separate objective change. Preserve broad-frequency checks alongside
   low modes so one improvement cannot hide another regression. Require better
   equation/gradient error per actual network evaluation before changing capacity.
4. CPU preparation can qualify new elongated and odd-count grids within the
   existing 65,536-cell contract, with explicit cubic spacing, force histories and
   native controls. Increase trajectory duration under native-qualified forcing/
   timestep. Retain pressure-field drift and per-step physical failures. Verify
   actual native axis/message limits before admitting each new grid.
5. Larger-cell-count scaling needs a separate capture/IPC/resource contract:
   qualified native fixture, bounded message/data/history budgets, CPU memory and
   thread/worker limits, and representability checks. Current40-cubed measurements
   cannot establish a crossover at 48/64-cubed or larger.
6. Use the final matched-resource CPU timings to select execution policy. The
   channels-last microprobe showed mixed results; it has no rollout migration.
   Tensor-fill/convolution and managed preflight/custody overhead are distinct
   future profiling concerns. Keep execution changes separate from learning.
7. Coordinate deployment as its own integration. The lab's one-channel residual
   representation, train-only scales, iterative policy/counts, previous pressure,
   native warm-up, fallback and restricted physical settings must be represented
   explicitly by the Rheon adapter. Source and result custody here do not change
   the application's deployed model or solver architecture.
