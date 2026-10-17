`qualify_typed_pressure.py` is the A11 qualification client for the single
prospective pressure smoke that also supplies the Q8 live lifecycle check. Its
execution mode requires the root-issued structural cutover receipt and exact
launch witness. Ordinary admission remains closed while this specific smoke is
authorized. Root completes M3 only after machine and browser readback pass.

`--prepare-only` measures the assembled schema2 Project, both recipes, all declared
source/data/provider/contribution bytes, the complete current managed runtime,
and the current Tuldok source inventory. It imports no numerical implementation,
connects to no catalog, makes no HTTP request and runs no optimizer. Use this
mode again after the application source freeze; execution requires an identical
source-verification receipt and target.

The execution command requires explicit `--endpoint`, `--catalog-id`,
`--catalog-root`, `--project-root`, `--project-alias`, `--tuldok-root`,
`--assembly` and an existing owned private `--session-file`. The configured
`pressure` root must point to the primary artifact root. The server exposes root
aliases and measured relative bytes; the launch receipt additionally proves the
absolute alias mapping. No implicit owner bootstrap or SQL authorization is used.

The cutover receipt must contain:

```json
{
  "schema": "rheon-tuldok-migration-cutover-v1",
  "m3_service_switched": true,
  "m3_complete": false,
  "ordinary_admissions_open": false,
  "smoke_authorized": {
    "kind": "typed_pressure_one_update_cpu_restore",
    "max_optimizer_updates": 1
  },
  "endpoint": "http://127.0.0.1:EXPLICIT_PORT",
  "catalog_id": "EXACT_CURRENT_CATALOG_ID"
}
```

The same document can supply `--app-launch-receipt`. That witness must bind the
same endpoint/catalog, `project_roots` mapping, `app_source_sha256` equal to the
prepared Tuldok source inventory, `app_sha256` equal to current `app.py`, and
`rheon_binary_sha256`. Both receipt arguments require their exact SHA256.

Before execution, register the measured runtime through the existing current
owner CLI if the live catalog does not already retain it:

```sh
"$APP_PYTHON" -B "$TULDOK/torch_runtimes.py" register \
  --data "$LIVE_CATALOG" \
  --manifest "$ROOT/runs/dependency-migration-v1/runtime/current-manifest-v2.json" \
  --expected-sha256 b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682
```

This owner CLI performs its own legitimate local-owner login/exchange, verifies
the full installation and audits registration. It runs no model. The
qualification client only reads the resulting exact registry manifest.

The ready driver invocation, after root supplies the actual target and gate, is:

```sh
"$APP_PYTHON" -B "$LAB/experiments/dependency-migration-v1/qualify_typed_pressure.py" \
  --execute-after-m3 --project-root "$ROOT" --project-alias pressure \
  --assembly "$ROOT/runs/dependency-migration-v1/typed-pressure-project-02/assembly.json" \
  --tuldok-root "$TULDOK" --endpoint "$LIVE_ENDPOINT" \
  --catalog-id "$LIVE_CATALOG_ID" --catalog-root "$LIVE_CATALOG" \
  --session-file "$OWNER_SESSION_FILE" \
  --prepared-receipt "$PREPARED_RECEIPT" --prepared-sha256 "$PREPARED_SHA256" \
  --m3-receipt "$CUTOVER_RECEIPT" --m3-sha256 "$CUTOVER_SHA256" \
  --app-launch-receipt "$CUTOVER_RECEIPT" --app-launch-sha256 "$CUTOVER_SHA256" \
  --output "$ROOT/runs/dependency-migration-v1/typed-pressure-live-01"
```

The ignored operational launcher already binds the actual prospective target
`http://127.0.0.1:8091`, catalog `a7da4a8de53541a2a0e2aad08005bb71`, and
`/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres`.
After root publishes `ops/m3-cutover.json` and `ops/m3-app-launch.json`, the exact
ready command is:

```sh
"$ROOT/runs/dependency-migration-v1/app-env/bin/python" -B \
  "$ROOT/runs/dependency-migration-v1/ops/run_typed_pressure_smoke.py" \
  --run-authorized-smoke
```

It uses the existing private `ops/migration-owner-session.private.txt`; an
explicit `--session-file` can select a legitimately renewed current owner
session. It runs final source verification into `typed-pressure-live-source-01`,
then invokes this client once into `typed-pressure-live-01`, hashing the gate,
launch and preparation receipts directly. It refuses prior launcher/output
state. It performs no runtime registration or credential bootstrap.

For inert preparation, use the same assembly, source/root, endpoint/catalog and
alias arguments with `--prepare-only` and a fresh output. No session, catalog
root or cutover receipt is needed for this mode.

Execution uses current registered HTTP operation contracts and bounded existing
transport. It opens the exact snapshot, publishes its captured Model, prepares
one closed Trial, checks exact Model composition and zero prior reservations,
obtains the owner's exact material grant, opens that finite Study and admits
one attempt. After completed/drained custody it closes the Study, verifies the
single settled reservation, reads the SDK checkpoint and fixed field slice,
then creates an ordinary managed CPU restore graph. Every trained restore slot
is an actual artifact of that new training Run. The sole filename mapping is
`trained_profile_json` to unchanged `training-profile.json`; private custody,
roles and complete origin are validated by the existing trainer/Run owners.

The ordinary restore starts through the existing Run supervisor in a fresh
managed CPU process with CUDA hidden. The client checks its zero-update receipt,
same source/input closure, original training origin, and exact weight,
checkpoint, normalization, history and prediction artifact hashes. A separate
fresh hidden-CUDA CPU diagnostic loads those restored weights using the captured
model and train-only normalization. It calls only prediction, forbids optimizer
construction and checks exact float64 pressure bytes on the predeclared consumed
validation frame. That pressure must equal the SDK field pair and the selected
API/CLI slice. It cannot establish scientific acceptance or quality promotion.

Readback uses an existing `--agent-config`, or a legitimate owner-issued new
`inspect` agent. Its private token/config live only in a mode0700 `private`
directory as mode0600 files, never in a receipt/log. The official `AgentClient`,
artifact verifier/downloader and actual current `project_cli.py` read immutable
Run/checkpoint/field identities. The final machine receipt explicitly leaves
browser readback pending and includes those exact selectors for root's browser
check. A passed machine receipt alone does not complete A11 or M3.

Each mutation has one persisted dispatch intent and one acknowledged/unresolved
record. There are no mutation retry loops. A command cannot reuse an output or
the cutover receipt's exclusive smoke claim, even with a different output path.
If a transport, validation or unknown-commit failure occurs, inspect the journal
and current durable objects through read operations. An unresolved admission may
already have started the one update. Do not create a new command/output, mint a
new grant, admit a second attempt or manufacture a parent Run to recover. Root
must determine the actual state and authorize any further action explicitly.

Focused checks:

```sh
/usr/bin/python3 -B -m unittest discover -s experiments/dependency-migration-v1 \
  -p test_qualify_typed_pressure.py -v
```

These checks cover source-only imports, exact smoke gates, no replay after an
uncertain commit, credential redaction, actual Run-artifact mapping and truthful
zero-update restore provenance. They perform no training or inference.

With the explicitly selected disposable PostgreSQL test connection in
`RHEON_PRESSURE_DRIVER_TEST_DATABASE_URL`, the final test uses the actual
HTTP transport/registry and current Project/Model/Study owners to prepare the
real bound pressure material. It stops before the first grant and forbids
`recipe_process.launch` and `studies.admit`. It asserts one closed Trial and
zero Runs, execution grants/consumptions and reservations, then removes only
its own unique disposable catalog schema. The original/candidate catalog
connection must never be supplied to this fixture.
