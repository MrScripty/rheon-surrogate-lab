# Migration operation source snapshots

These are exact copies of one-time operational scripts and offline fixtures.
Their index binds the original paths and bytes, including the temporary publisher
restart wrappers and scripts under `runs/dependency-migration-v1/ops/`.
They are archived for review, outside application execution and default tests.
They retain their original path assumptions: **do not execute these archive copies**.

Use each actual receipt for the command, selected inputs, environment, elapsed
time, outcomes and limitations. A source snapshot does not assert successful
execution. The immutable cutover authorization and later completion receipt have
separate meanings; no script retries an uncertain mutation or restores the original
catalog in place. Numerical workloads use the committed experiment entrypoints;
these scripts operate the migration and its bounded acceptance checks.

No database connection values, credentials, SQL dumps, model weights or simulation data
are included here. The offline source-format converter belongs to the separate
Tuldok commit under `tools/migrations/rheon-lab-once/`, outside deployed payloads.

Multiple versions can retain the same original path. Their exact hashes and
individual attempt receipts identify what actually ran; an archived snapshot
does not assert that its original path still contains those bytes. The explicit
Node-selection versions retain the fix after browser attempt01 selected Node18.
