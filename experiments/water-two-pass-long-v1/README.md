# Two-pass longer control

`run.py` delegates a single managed CUDA fit and existing CPU qualification,
then evaluates consumed whole-simulation regressions, freezes development
selection and compares the checkpoint with the qualified control on eight new
whole simulations. It does not implement a numerical solver or new trainer.

From the lab root, after committed clean source capture:

```
.venv/bin/python experiments/water-two-pass-long-v1/run.py \
  --sequence experiments/water-two-pass-long-v1/configs/sequence.json \
  --reports docs/plans/water-two-pass-long-v1/reports \
  --torch-python /path/to/qualified/pumas/python \
  --managed-controller /path/to/existing-authenticated-workflow-adapter.py
```

The adapter is the existing local-owner Tuldok workflow adapter; credentials
stay in memory. Source capture and normal existing-catalog custody are required.
Attempts are fresh and never silently replayed. See the canonical plan for
selection, limits, retention, physics and lifecycle obligations. The separate
mixed-odd bank configuration is preparation only and does not affect this fit.

A failed comparison caused by a missing retained reference has an explicit
`--resume-development` operation. It verifies the original sequence/source
identity, completed managed custody and sealed CPU qualification, accepts only
corrected reference bindings, and never resubmits training. A separate recovery
status and recovered drift paths preserve the original failure. Other failures
require reconciliation rather than this recovery operation.
