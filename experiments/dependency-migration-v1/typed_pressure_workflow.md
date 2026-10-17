# Prospective typed pressure workflow

`typed_pressure_workflow.py` reads the actual prepared one-update water training
project. It verifies its schema-2 snapshot, recipe/source/input hashes, measured
managed runtime, train-only fitting packet, float32 one-channel architecture,
float64 pressure reconstruction, and one-update schedule. It emits inert Task,
reader, Model, trainer, evaluator and view declarations. Scalar density, timestep
and cubic spacing remain explicit physical context in retained Model policies;
residual/base/native/predicted pressure remain actual ZYX tensor ports.

The prospective Study budget is two attempts, 1,600 admitted wall seconds,
512 MiB retained data and 1 GiB staging. These reserve the CUDA training attempt
and a separate ordinary CPU restore; they are admission limits rather than OS
quotas. Training plans exactly one optimizer update. Restore plans zero updates,
with CUDA hidden. Neither a declaration nor an inert Model/Study draft grants
execution authority. Historical Run ancestry remains unknown. Fixed original
source/normalization references record provenance rather than a fabricated fit
or new-parent relationship.

`consumed_validation_probe(root, control_project, absent_output)` selects the
first original control validation frame and verifies its presence in the actual
completed original training receipt and managed artifact bookend. It checks the
original packet hashes and copies its residual, solved base and native pressure
payloads exactly into a bounded NPZ. It uses only the standard library, opens no
reserved-test inputs, and imports no NumPy/Torch or producer code. This is an
already-consumed validation diagnostic; it is not untouched evaluation data or a
claim of independent model quality.

The extracted reference currently identifies original Run
`e20126d732a24013aafa6cc09ebbc0a5`, trajectory
`baseline--water-validation-0000`, frame 15, grid 16³, h=0.125 m,
rho=1000 kg/m³ and dt=1/60 s. Its complete metadata, original receipt identity,
payload hashes and extracted NPZ identity are retained outside source under the
migration receipt directory. This Run is provenance only; no new Trial is
attached to its history.

`typed_pressure_evidence.py` is captured execution glue for a future approved
bound training recipe. Import is inert. In training mode it calls the existing
one-update trainer with the original numerical model/loss and Q3 train-only
diagnostic hooks. The separately captured consumed-validation probe is removed
only from the original trainer's input-evidence projection; the complete host
composition still retains it. The large runtime manifest is an explicit Project
data input in both modes, with its receipt checked against the actual captured
bytes, original source witness and fitting-input snapshot before omission from
the trainer's numerical fit slots. The managed host separately owns its runtime
descriptor/bootstrap verification. After training, the wrapper loads the actual final
float32 checkpoint on CPU, reconstructs full float64 pressure with the original
model implementation, and emits the existing SDK checkpoint, fitted transform,
fixed validation field pair and mean-free same-request pressure relative L2.
Constant-reference undefined errors fail explicitly. This metric is separate
from autonomous trajectory drift, and asserts no scientific acceptance or model
promotion.

The original trainer's `profile.json` conflicts with the SDK's required
`profile.json`. The wrapper retains the original bytes as
`training-profile.json`, while preserving the original training receipt byte for
byte. Ordinary CPU restore maps that one declared source artifact name through
the original custody validator. Actual artifact name, role, hash, complete
drained Run identity and shared origin still must match. The original receipt's
`outputs["profile.json"]` continues to verify the exact original profile bytes;
its contents are never rewritten to manufacture custody. Restore executes the
ordinary existing trainer path, copies its original outputs, verifies exact
weight/prediction parity and emits no bound model-evidence claim.

`retain_prospective_intent` exercises current Project opening and inert
`models.create` / `studies.create` operations in an explicitly authenticated
disposable catalog. It deliberately returns `unsupported` with no Trial, Run,
approval, admission or optimizer action. A schema check of a generic recipe is
not pressure evidence qualification. The separately owned bound-project
assembler must provide all exact contribution references, captured SDK/source,
validation input/target hashes, model/policy/task identities and finite material
before `studies.prepare_trial` can be qualified. `prepare_current_trial` consumes the separately assembled current pressure
project through the actual current owners: snapshot inspection, exact bound
recipe inspection, Model creation/publication, graph capture, closed Study
creation and one prospective member Trial. An exact unchanged preparation body
reconciles to that Trial. It never grants approval, opens admission or submits a
Run. The full measured runtime must already be registered in the caller's
isolated catalog. Actual Q10 execution remains gated on M3 and its own
source-frozen approval/admission receipt.

The focused tests use explicit actual material roots and isolated PostgreSQL:

```sh
RHEON_PRESSURE_MATERIAL_ROOT=/absolute/lab-artifact-root \
RHEON_PRESSURE_TRAINING_MANIFEST=runs/dependency-migration-v1/training-project-3/project.json \
python -B -m unittest discover -s experiments/dependency-migration-v1 \
  -p test_typed_pressure_workflow.py -v
```

They validate all six current contribution declarations, exact consumed probe
extraction without numerical imports, changed-source/heldout-input refusal,
gauge-independent metric arithmetic, truthful artifact aliasing, owner-only
inert drafts, actual Model/material/closed single-Trial preparation and receipt
re-entry, zero execution grants/reservations/Runs, and exact runtime input receipt
checks against wrong bytes, another snapshot and an unsupported receipt kind. They do not execute the
evidence wrapper, fit a model, perform inference or qualify A11 acceptance.
