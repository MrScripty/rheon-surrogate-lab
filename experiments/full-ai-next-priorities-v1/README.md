# Full AI pressure and temporal experiments

This isolated study executes `docs/plans/full-ai-next-priorities-v1/plan.md`.
The original plan and registration remain in the documentation directory;
raw experimental evidence is under `runs/full-ai-next-priorities-v1`.

The temporal `bundle_model.py`, `bundle_model_v2.py` and `bundle_model_v3.py`
files are retained untrained architecture prototypes. Their contract and CPU
feasibility measurements are recorded in `bundle-contract-01/02/03.json`.
They are not qualified training results or application models.

`bundle_model_v4.py` addressed adaptive pooling; its CUDA qualification exposed
the same limitation in fixed-window pooling. `bundle_model_v5.py` preserves
the compact third prototype's architecture and uses equivalent separable mean
pools for both CUDA backward operations. CPU inference retains the reference
pooling kernels. Forward/gradient equivalence and the failing reference
regressions are retained as scientific/debug evidence. The managed temporal
trainer uses this fifth implementation.

Primary temporal fits now use `temporal_train_v2.py` and
`temporal_feedback.py`. They reconstruct the last predicted physical state and
apply the deployed mean-free pressure encoding before the next block, preserving
gradients throughout. The original trainer fed cell-zero pressure directly into
the next block. Its completed B1 fit is diagnostic only; the original queue was
stopped before B2 admission. Both registrations and all original receipts remain
available. Corrected primary Projects and fit/restore pairs have generation 02.

The conditional pressure architecture screen is registered separately. It maps
verified C107 tensors into redistributed channels or one learned correction with
more learned global modes. These mapped initializations are not claimed to be
functionally identical to C107. Candidate source and initialization inventories
are captured in each new Project.

Candidate inference contains no numerical pressure inverse. Numerical pressure
work in `native_client.py` belongs to reference capture and evaluation only.
The temporal model consumes current physical state and prescribed forcing;
future reference pressure equations and state fields are supervision only.

No file here promotes an application model. Run controllers must keep their
durable mutation journals and must never replay an unresolved admission.
