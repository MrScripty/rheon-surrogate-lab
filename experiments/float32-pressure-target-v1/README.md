# Float32 model-output target

The current development acceptance target is a relative full pressure-equation
residual of **1e-6**, with the original **1e-12 absolute floor**. The model's
features, weights and raw correction remain float32. This addresses rounding
precision, not float32's largest representable number.

The target is an explicit model-output profile in configs/development.json.
It does not rewrite historical native thresholds or reinterpret old results.
The reference/corrector continues to use its existing stricter accuracy. All
physical failures remain blocking: predicted divergence 1e-7 s⁻¹, actual rounded
velocity divergence 1e-5 s⁻¹, zero wall-normal velocity, net-flux limits and
Courant <= 0.5. The profile replaces only the original-RHS residual flag when
assessing a candidate. Original native results are retained alongside that new
assessment, including their actual solver iterations and strict qualification.

Calibration uses the ideal reference-minus-base correction through the exact
current float32 output conversion, pressure rescale and gauge operations. The
96 training frames have maximum relative residual 4.7665255695e-7. A twofold
margin, rounded up to a decimal power, selects 1e-6. That selection is written
before evaluating the 48 validation frames. All 144 rounded ideal corrections
pass the new profile and the unchanged physical limits; all 144 fail the old
strict residual check. This is empirical calibration on the current development
family, not a universal accuracy guarantee for float32 or arbitrary grids.

The twelve actual restored model predictions still qualify 0/12. Their errors
are much larger than rounding error. The learning loss is continuous and had
no hard 1e-8 stopping target; the changed number belongs to acceptance feedback.
No retraining, network precision change or checkpoint modification is needed
to apply this profile. Teacher-state calibration does not establish autonomous
rollout accuracy or solver acceleration.

Run from the lab root with fresh output and explicitly selected material:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/float32-pressure-target-v1/run.py \
  --packet "$PWD/runs/variable-pressure-v2-material-1/packet" \
  --binary "$PWD/runs/variable-pressure-v2-material-1/native-build/rheon-variable-pressure-evaluation" \
  --training "$PWD/runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts" \
  --restored "$PWD/runs/variable-pressure-v2-workspace-1/execution/stage-01/artifacts" \
  --output "$PWD/runs/new-float32-target-check"
```

`--profile` selects another explicit profile. It must agree with the declared
train-only calibration rule; validation or learned-prediction failures never
silently change it. `targets.assess` consumes quality already checked by the
unchanged native-response validator and produces separate acceptance feedback.
The driver revalidates source-bound native identity, packet, normalization,
typed restore identity and whole artifact parity. It performs 156 fresh native
diagnostics: 144 quantization cases plus 12 actual predictions. Native reference
and solver arithmetic remain unchanged; this is a float32 learned-output profile,
not a claim that every calculation in the simulator uses float32.

Calibration and real model fields/progress appear in the existing Tuldok/Rheon
observers, with ideal references explicitly labelled as calibration rather than
learned predictions. The accepted run is in runs/float32-pressure-target-v1-1;
its profile, train-only selection, per-case native records and report are retained.
The report measures source/data/weight hashes; no historical hash constant changes.

```sh
.venv/bin/python -m unittest discover -s experiments/float32-pressure-target-v1 -p test_targets.py -v
```
