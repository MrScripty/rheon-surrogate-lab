# Backup retention

The user authorized removal of identical backup files on 23 October 2026.
The cleanup compares actual SHA-256 content against each backup's original
receipt; it does not infer duplication from filenames or model labels.

The complete retained backup is:

`runs/dependency-migration-v1/completed-current-backup-04/`

The following older directories retain their original, different database
snapshots and receipts. Their duplicate catalog payloads are removed:

- `runs/dependency-migration-v1/original-schema8-backup-01/`
- `runs/dependency-migration-v1/original-schema8-backup-02/`

They are **compacted snapshots, not standalone restore inputs**. Each contains
`COMPACTED-BACKUP.json`, identifying the retained source and exact manifest.
Do not pass them directly to legacy restore scripts. Keep the complete retained
backup and manifest for as long as either historical snapshot is needed.
No active catalog, deployment, runtime, training output or unique model weight is
included in this cleanup. Existing historical receipts remain unmodified.

The ignored manifest lists every removed relative path, byte count and SHA-256:

`runs/model-portfolio-migration-v1/discovery/backup-compaction-01.json`

The small [completion receipt](reports/backup-compaction-01.json) records totals
and post-removal verification. The complete retained backup is verified against
its metadata hash and every payload entry before and after deletion. Unique
metadata and remaining payloads in older directories are also checked afterward.

To reconstruct either older snapshot, from the repository root:

```bash
python3 -B experiments/model-portfolio-migration-v1/compact_backups.py materialize \
  --manifest runs/model-portfolio-migration-v1/discovery/backup-compaction-01.json \
  --backup-name original-schema8-backup-01 \
  --destination /tmp/rheon-schema8-backup-01-restored
```

Choose a destination that does not already exist, with sufficient free space.
Use `original-schema8-backup-02` for the other snapshot. Materialization creates
ordinary independent files, retains that snapshot's exact metadata and original
receipt, and verifies the result against the original backup receipt. No hard
links or symlink substitutes are used. This reconstructs the filesystem backup;
the original Tuldok restore procedure still owns database restoration.

## Native history rehearsal backup 03

The complete standalone native restore input for this snapshot is now:

`runs/model-portfolio-migration-v1/native-backup-compaction-03/private/backup`

The old snapshot at
`runs/model-portfolio-migration-v1/isolated-history-metadata-restore-01/private/extraction-exchange-native-history-extraction-rehearsal-03/stopped-copy`
retains its unique metadata, original receipt and `COMPACTED-BACKUP.json` pointer.
Its duplicate `catalog` tree was retired only after an actual native restore
proved all 43 tables, 22,803 rows, sequence positions and 11,160 independent
catalog files exact. Use the complete keeper above for native restoration.

The compact keeper retains all 11,549,686,016 logical bytes in 11,162 files;
allocated file bytes fell from 11,573,395,456 to 1,675,128,832. Shared inodes exist
only among verified identical copies inside this private backup. Restoration
creates independent files, and the source catalog was never linked. After
verification, 11,053,054,076 duplicate logical bytes were removed from the old
rehearsal backup; approximately 28.25 GB free was observed afterward. The test
restore database is stopped and its verified temporary catalog is removed.

See [actual native restore](reports/native-backup-compaction-03.json) and
[duplicate retirement](reports/native-backup-retirement-03.json). This does not
complete the full original live-catalog backup or authorize unique weight
retirement. The later rehearsal04 compaction is documented below.

## Rehearsal backup 04 compacted after disk-space follow-up

The user reiterated the shortage of disk space. All11,159duplicate catalog
payloads in rehearsal04 are verified against the restored-and-qualified compact
rehearsal03 keeper above, then removed:11,053,053,888logical bytes (10.3GiB).
Rehearsal04 retains its distinct original metadata.json, original receipt and
one188byte coordinator journal. No unique payload or model weight is deleted.
The keeper is verified again afterward. The completion receipt is
[rehearsal04 compaction](reports/rehearsal04-payload-compaction-01.json).

Rehearsal04 is now a compacted historical snapshot, not a standalone native
restore input. Keep its COMPACTED-BACKUP.json and the ignored per-file manifest
at runs/model-portfolio-migration-v1/discovery/rehearsal04-payload-compaction-01.json.
The original rehearsal03 keeper must remain available. To reconstruct the exact
original rehearsal04 files as ordinary independent copies, run:

```bash
runs/dependency-migration-v1/app-env/bin/python -B \
  experiments/model-portfolio-migration-v1/compact_rehearsal04_backup.py materialize \
  --destination /tmp/rheon-rehearsal04-materialized
```

Choose an absent destination with sufficient space. The command verifies all
original bytes against the unchanged receipt; the native restore procedure still
owns SQL restoration. Rehearsal04's distinct SQL snapshot is retained, and a new
SQL restore of that particular snapshot is not claimed by file compaction.


## Stopped rehearsal source catalog compacted

The stopped working source at
`runs/model-portfolio-migration-v1/isolated-history-metadata-restore-01/private/catalog`
is now compacted and cannot be restarted as a complete catalog. Its 15,340 files
were exact byte/length/mode matches in the complete retained native backup:
`runs/model-portfolio-migration-v1/isolated-history-metadata-restore-01/private/extraction-exchange-full-current-history-extraction-03/stopped-copy`.
The keeper was fully verified before and after removal. Its original SQL snapshot,
receipt and complete catalog remain. The stopped physical SQL cluster is retained
separately; no live service or unique payload was modified. This cleanup removed
12,278,416,312 logical bytes (11.4 GiB). See
[completion receipt](reports/stopped-source-compaction-01.json).

The original source file/directory modes and hashes are in
`runs/model-portfolio-migration-v1/discovery/stopped-source-compaction-01.json`;
`private/COMPACTED-SOURCE.json` pins the manifest. Reconstruct the source catalog
as independent files at a new external path before considering source reuse:

```bash
python3 -B runs/model-portfolio-migration-v1/discovery/compact_stopped_source_copy.py \
  materialize /tmp/rheon-stopped-source-catalog-restored
```

This requires an absent destination and about 13 GiB of free space. Full-data
materialization has not been rerun merely to create another large duplicate;
the original complete native backup remains the recovery input.


## Inactive dependency-migration rehearsal catalogs compacted

The three old working restore catalogs `runs/dependency-migration-v1/catalog-rehearsal-01`,
`catalog-rehearsal-02` and `catalog-rehearsal-03` are now compacted, not directly
restartable catalogs. No process selected them or held an open file under them.
Every removed file matched a fully verified file in the unchanged complete backup
`runs/dependency-migration-v1/completed-current-backup-04`. Native source projects,
SQL databases, original services, unique payloads and root metadata were not
removed. Each catalog retains four original root metadata/restore/migration files
and a COMPACTED-SOURCE.json that pins its exact reconstruction manifest.

The cleanup removes 100,533 duplicate files / 22,322,363,907 logical bytes (20.8 GiB)
across the three copies. Every keeper payload is verified again after each pass.
See [summary and per-catalog proofs](reports/retired-catalog-compaction-summary-01.json).
Three tiny real-filesystem fixtures verify independent reconstruction, modes,
retained headers, changed-keeper refusal and destination containment. Full-size
reconstruction is deliberately not run merely to recreate the removed duplicates.

To reconstruct one catalog as ordinary independent files, choose an absent
external directory with at least 8 GiB free and substitute index 1, 2 or 3:

```bash
python3 -B experiments/model-portfolio-migration-v1/compact_retired_catalog.py \
  materialize --catalog-index 2 --destination /tmp/rheon-rehearsal02-materialized
```

The original catalog01 execution script is additionally retained under
runs/model-portfolio-migration-v1/discovery/compact_retired_catalog01.py. Keep
all ignored manifests and the complete keeper. Do not launch legacy tools against
the compacted directories without materialization. These are filesystem-copy
retirements; no change to the original SQL snapshots or a new SQL restore is claimed.


## Four inactive archived-catalog backup copies compacted

On 24 October, the additional disk-space audit removed134,056 exact duplicate
catalog payload files /29,763,292,860logical bytes (27.7GiB) from the
`original-archive` backups inside these dependency-migration stages:

- `domain-original-stage-01`
- `domain-stage-02`
- `domain-stage-03`
- `domain-stage-04`

All original metadata.json and backup-receipt.json snapshots remain. Each
removed payload matches its original receipt and the unchanged complete keeper
`runs/dependency-migration-v1/completed-current-backup-04`, with matching file
modes. No process selected the inactive stages or held their files open.
All keeper payloads and retained original metadata were verified again after
removal. Active migration inputs, services and unique weights were untouched.
Free space was61GiB afterward. See the
[completion receipt](reports/domain-archive-backup-compaction-01.json).
The existing five compaction/reconstruction/refusal fixtures pass; the wrapper
also rejects reconstruction inside this repository. Full-size reconstruction
was not performed merely to recreate the deleted duplicates.

These four directories are now compacted backups, not standalone restore inputs.
Keep their COMPACTED-BACKUP.json markers, the complete keeper, and the manifest
`runs/model-portfolio-migration-v1/discovery/domain-archive-backup-compaction-01.json`.
To reconstruct a complete independent backup, choose an absent external path
with at least8GiB free and substitute the stage name as needed:

```bash
python3 -B experiments/model-portfolio-migration-v1/compact_domain_archive_backups.py materialize \
  --manifest runs/model-portfolio-migration-v1/discovery/domain-archive-backup-compaction-01.json \
  --backup-name domain-stage-04/original-archive \
  --destination /tmp/rheon-domain-stage04-backup-restored
```

Materialization restores ordinary independent files and verifies the original
backup receipt. Native Tuldok restoration still owns SQL restoration; file
compaction does not claim a new database restore. Legacy absolute paths into
these compacted catalog copies need materialization before reuse.

## Repeated disk-pressure follow-up: no further full-size fixtures

The user's renewed instruction is to avoid duplicating data. The next cold-load
qualification must reuse the closed complete project in a confined fresh
installation, rather than create another approximately 16 GB payload copy.
Publication, custody, closure and relocation safety must be established first.
Do not run the old full-copy fixture preparation for the combined project.
Retain unique source metadata and payloads until their replacement has been
verified. The later, explicit artifact-only retirement decision below permits
removing verified duplicate source payloads after native publication and custody;
it does not waive the subsequent cold/UI acceptance gate.

The inactive `catalog-current-cold-04` has now been compacted against the unchanged
complete `completed-current-backup-04`: 33,511 exact duplicate files,
7,440,787,969 logical bytes (6.9 GiB). Five unmatched or root metadata files remain.
See [the completion receipt](reports/inactive-cold-catalog-compaction-01.json).
Each removed file matched the keeper's actual SHA256, size and permission mode;
the entire keeper was verified before and after removal. No unique payload was
deleted. Reconstruct this retired catalog only when legacy reuse actually needs it:

```bash
python3 -B experiments/model-portfolio-migration-v1/compact_inactive_cold_catalog.py \
  materialize --destination /tmp/rheon-cold04-materialized
```

The old historical inventories remain unchanged. A
[current locator audit](reports/retained-model-locations-after-compaction-01.json)
rehashed all 524 checkpoint byte states, all 205 fit receipts and all 127 named
pressure material/source bindings. It selected surviving exact locations for
292 retired checkpoint locators and explicitly mapped three evidence paths.
Use `runs/model-portfolio-migration-v1/discovery/retained-model-locations-01.json`
for current locations. This audit copied no payloads.

Four inactive `external/` trees in the same domain stages listed above have also
been compacted against keeper04. They contributed 4,932 exact duplicate files /
919,761,952 bytes (0.86 GiB). All 8,656 unmatched or root files remain; an unmatched
file is not asserted to be globally unique. No process selected these trees or
held them open, and the checked current model/pressure/full-state bindings did
not select them. Each tree retains its COMPACTED-SOURCE.json and exact mapping.
See [the combined cleanup receipt](reports/storage-cleanup-followup-01.json).

```bash
python3 -B experiments/model-portfolio-migration-v1/compact_external_stages.py \
  materialize --stage domain-stage-02 --destination /tmp/rheon-stage02-external
```

Keep all compaction manifests and keeper04. No large materialization was run just
to recreate removed duplicates. The established three focused compaction tests
pass, including exact independent reconstruction, changed-keeper refusal and
destination containment.

The host reboot interrupted the native composition. Native recovery subsequently
published the existing committed directory at
`projects/rheon-model-portfolio-unified-04`, retaining its checkpoint, directory
inode and all 8,705 payload inodes, sizes and modes. No payload was copied during
recovery. [Publication receipt](reports/committed-composition-recovery-01.json).
The temporary archive source remains absent; the independent
[post-publication custody audit](reports/published-composition-custody-01.json)
checks original archive facts and all 53,873 retained records without claiming
a fresh comparison to that absent source's inodes or modes.

## Closed composition-input artifact retirement

The renewed disk-space instruction authorizes removing the artifact duplicates
in these two closed migration inputs only:

- `runs/model-portfolio-migration-v1/native-composition-primary-01/root/project`
- `runs/model-portfolio-migration-v1/full-current-history-cold-01/root/project`

Their keeper is the published project above. The storage tool does not merge,
save or publish a native project. It retains all original canonical records,
manifests and working files, plus a per-file reconstruction map. Before removal
it verifies the native publication/custody receipts, complete source membership,
the keeper's complete payload population, exact file SHA256/size/mode matches,
source inactivity and exclusive native directory locks. The keeper is rehashed
after removal. Original standalone backups, lab weights and live user UI are
outside this cleanup. Cold loading and final UI acceptance remain outstanding.

The reconstruction map is
`runs/model-portfolio-migration-v1/discovery/composition-input-payload-compaction-01.json`.
Each source receives `COMPACTED-PAYLOADS.json` **before** any deletion. These
compacted inputs must not be reopened, recomposed or used as standalone native
restore fixtures until their artifacts have been reconstructed. Old custody
reports describe their original complete state; rerunning a source-preservation
reader against a compacted input requires reconstruction first.

Only if legacy source reuse actually needs another payload copy, choose
`primary` or `current` and explicitly reconstruct ordinary independent files:

```bash
python3 -B experiments/model-portfolio-migration-v1/compact_composition_inputs.py \
  reconstruct --source current
```

This operation requires free space for the selected original payload population
plus 1 GiB. It does not run automatically. Interrupted removal uses the existing
map via the explicit `resume` command; it never creates a second cleanup map.
Keep the published keeper and map while either source snapshot is retained.
The focused real-filesystem checks include exact reconstruction, changed keeper,
unique payload and busy-root refusal, and interruption immediately after unlink:
[initial cleanup qualification](reports/composition-input-payload-compaction-code-01.json)
and [current reconstruction qualification](reports/composition-input-payload-compaction-code-02.json).

Actual retirement completed: **5,083 files / 15,746,636,174 logical bytes
(14.7 GiB)**. All 8,705 keeper payloads passed the final rehash; no unique
payload was deleted and no payload copy was created. About 24 GiB was free
at completion. [Actual completion receipt](reports/composition-input-payload-compaction-01.json).
Reconstruction permits later native checkpoints/additional artifacts in the same
keeper project while verifying every originally retained payload; compaction
itself still requires the original pinned checkpoint. The temporary cold-test
relocation must return the keeper to its recorded path before these tools run.

## Complete keeper04: internal duplicate storage shared

The inactive `runs/dependency-migration-v1/completed-current-backup-04` remains
a complete standalone backup at all original paths. Internal hardlink
deduplication reclaimed **5,055,508,480 allocated bytes (4.71 GiB)** while retaining
all 33,515 files, exact original bytes, permission modes, metadata and receipt.
Only 7,860 identical private paths were relinked, entirely within that backup.
No links were made to the original catalog, live project or another backup.

Existing compaction maps and materializer paths above remain valid. The native
copy reconstruction fixture proves independent restored files and modes; the
actual whole backup passed its original receipt inventory and content checks
before and after. No full-size SQL restore was repeated merely to recreate data.
Keep this backup and all existing maps. In-place editing of a shared backup member
would affect its internal aliases; use the native independent-copy restore path
before modifying restored catalog files. See the
[actual storage receipt](reports/private-backup04-deduplication-01.json) and
[focused code qualification](reports/private-backup-dedup-code-qualification-01.json).
