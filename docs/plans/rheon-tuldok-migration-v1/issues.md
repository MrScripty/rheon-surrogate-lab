# Migration findings

| ID | Finding | Disposition / owner / revisit gate |
| --- | --- | --- |
| E1 | Primary Rheon and old live viewer contain unrelated uncommitted work | Preserved. Migration task uses fetched isolated worktree; no stash/reset/merge into those edits |
| E2 | GitHub Tuldok lacks required CUDA/CPU execution and live activity | Required M1 current-contract implementation and one-time conversion; preserve original evidence offline, not obsolete runtime/manifest readers |
| E3 | Production catalog schema 8; new Tuldok requires10 | Required M3 explicit backup/restore rehearsal and quiesced upgrade; existing Metadata/storage tooling owns transition |
| E4 | Existing app environment lacks jsonschema4.26.0 | New application environment required; current environment preserved. Nine tests passed with already provisioned lab environment; no new install here |
| E5 | Main lab checkout lacks later water training code | Preserve experiment/water-pressure-recovery-v1 at 1548266. Recommended implementation base is that complete lineage; main integration remains separately reviewable |
| E6 | Native probe imports changed private desktop engine fields/modules | M2 owns replacement through an adequate supported boundary. Do not infer compilation from source path selection or copy an engine/process implementation |
| E7 | Accepted Session native fallback can conceal learned-model rejection in aggregate pass counts | Required distinct raw diagnostic and accepted execution evidence; report actual dispatch, rejection, accepted native fallback and network calls separately |
| E8 | Higher upstream density/count limits exceed checkpoint qualification | Keep current cubic spacing and existing model envelope. Larger-grid generalization and expanded physical settings remain separate model experiments |
| E9 | Upstream measured aa82 input availability was blocked in a different environment | Local bundle found; selected hashes match. Full runtime rehash, retained prediction parity and real lifecycle qualification pending M2 |
| E10 | Current project action surface differs from the local live format | Convert the project once and qualify live import/inspection/activity through current interfaces before cutover; no legacy-reader obligation |
| E11 | Pressure warm-start primitive remains a lab patch | Textual apply-check passed on target; patch not applied or qualified. M2 inventories real consumers before selecting a separately identified variant or replacement |

These findings describe migration work, not failed physical qualifications or
permission denials. No auto-review rejection occurred. The original catalog and
running service remain unchanged.

## Standards review dispositions

R1–R5 from the retained [review](reports/standards-review.md) were addressed by
revision 53605af; its [resolution](reports/standards-review-resolution.md) remains
historical. The user's [one-time migration direction](reports/one-time-migration-decision.md)
supersedes the compatibility-reader/API policy while retaining the requirement to
account for all prior results. Neither report is a runtime pass. E2/E3/E6/E9/E10/E11 remain
implementation/qualification work. Current numerical/provenance gates are pending;
no gate blocks beginning M1 source work and no service/solver switch is qualified.
