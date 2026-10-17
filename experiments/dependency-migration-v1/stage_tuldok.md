# Current Tuldok deployment artifact

`stage_tuldok.py` assembles the current application's exact bytes into an absent
attempt directory. It does not start an application service, open a catalog,
read an owner credential, submit a Run or load a model. The application environment
remains the separately provisioned `app-env`; no virtual environment is copied.

The payload includes all tracked top-level Python owner and CLI modules,
requirements pins, canonical extension schemas, SDK plus its authoring schema
copies, static assets, prompts and current contract documents. Known current
untracked owner additions are explicitly named. Current `candidate_packages.py`
reads two built-in inference closures relative to its own module: their exact
`infer.py`, `contract.json` and `NOTICE.txt` are included. One authored source
folder is named `inference-v2-private-qa`; these three tracked code/contract/notice
files are needed by the current owner and contain no task private payload.

The allowlist excludes the entire `tools` tree (including
`tools/migrations/rheon-lab-once`), all tests and CI configuration, Git metadata,
scripts, adapters, non-runtime examples, caches, environments, catalogs,
credentials, weights, datasets and task-run data. Nested SDK caches and test
files are excluded too. An unknown untracked resource is an error, rather than
an implicit addition to the deployed source closure. Selected symlinks and
non-regular or multiply linked source files are refused.

Example, with an explicitly fresh attempt and pre-existing receipt parent:

```sh
ROOT="$PWD"
"$ROOT/runs/dependency-migration-v1/app-env/bin/python" -B \
  "$ROOT/runs/dependency-migration-v1/lab/experiments/dependency-migration-v1/stage_tuldok.py" \
  --source "$ROOT/runs/dependency-migration-v1/Tuldok" \
  --destination "$ROOT/runs/dependency-migration-v1/Tuldok-deployed/current-03" \
  --python "$ROOT/runs/dependency-migration-v1/app-env/bin/python" \
  --receipt "$ROOT/runs/dependency-migration-v1/receipts/tuldok-deployment/current-03.json"
```

The builder first copies to a private staging sibling. It preserves each source
file's exact bytes and mode, retains before/after identity and content witnesses,
then verifies the selected source set and Git origin again. A no-follow component
walk rejects aliases and detects replacement while reading. The limits are
2,048 selected files, 8 MiB per file and 64 MiB aggregate payload; the guarded
smoke has a 45-second deadline. Linux `renameat2(RENAME_NOREPLACE)` publishes the
verified directory atomically and cannot replace a concurrent empty directory.
Any failure before that publish removes staging and leaves no deployment root.

`deployment-manifest.json` records the Git HEAD/tree/branch/status witness, every
payload path/hash/byte count/mode, the aggregate source content identity and the
actual smoke result. Payload file records exclude the generated manifest itself;
the external staging receipt binds its exact SHA-256, completing artifact custody
without a self-referential hash. The app SHA-256 is recorded separately. No owner
source is rewritten, so `recipe_contract.py`'s relative host-file fingerprints
remain equal to the prepared current application. Every deployed file is either
in the payload inventory or is this separately bound manifest.

The smoke uses `app-env` Python with `-I -B` in an empty temporary working directory.
It imports all deployed top-level modules from that deployment root, validates
all current operation schemas, checks built-in registries, exact recipe host
fingerprints, validation import closure, canonical/SDK schema identity, bundled
inference sources, static routes and HTML script/style URL aliases, and the
corners prompt. It patches PostgreSQL connection entry points and adds Python
audit guards against sockets, child processes and file writes/mutations. No
Dataset, server, owner credential or execution object is constructed. This is
an inert smoke of trusted source, not an OS sandbox or a full execution test.

Do not overwrite or edit a deployment attempt. A later application source change
requires a new attempt, followed by new host/source comparison and smoke. If the
directory was published but the receipt write failed or its acknowledgement is
unknown, preserve it. Reconcile its exact manifest hash and payload using
`stage_tuldok.verify(path)` and the source witness; do not blindly retry at that
destination. The builder's focused tests exercise this actual receipt-failure
boundary as well as source mutation, smoke failure, aliases, unexpected resources,
concurrent destination creation and artifact tampering.

The converter remains only in source Git and migration receipts. This deployment
contains no converter or migration tests, and no CI payload. Retiring the
migration-only tests from the source repository's ongoing CI is a separate
post-cutover owner action: its current full-source unittest discovery still
finds those tests. Source CI was not silently changed under the application
freeze. After successful cutover and verification, keep the retained converter
history/receipts as required and prevent its return to deployed runtime or
ongoing CI. This artifact does not perform catalog conversion or service cutover.
