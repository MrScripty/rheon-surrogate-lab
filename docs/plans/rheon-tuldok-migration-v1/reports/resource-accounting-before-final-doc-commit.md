# Resource accounting before final documentation commit

Snapshot: 2026-10-17T03:27:42.346913+00:00. Browser, M3 and retirement verification are complete; final documentation commit and ignored-ops final accounting remain pending.

This is a read-only resource inventory, not cleanup or merge authorization. The public report intentionally records the pre-documentation lab HEAD and dirty state; it cannot contain its own eventual commit hash.

| Repository/worktree | Branch | HEAD | Dirty entries |
| --- | --- | --- | --- |
| lab: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab` | `refs/heads/plan/rheon-tuldok-migration-20261016` | `6795ac21bce7e00c759d0b6a1eb51b1fee7c8045` | 9 |
| lab: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/dependency-migration-v1/lab` | `refs/heads/integration/current-dependencies-20261016` | `01b9e5bd5b8603204979dda656e189b6938582ad` | 122 |
| Rheon: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/Rheon` | `refs/heads/main` | `43ed26b645dd42de906d0fac7d1ec7d5ab6e351a` | 18 |
| Rheon: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/dependency-migration-v1/Rheon` | `refs/heads/integration/lab-current-contracts-20261016` | `759d928448dec8b160f4c113e35bf2bf2bd1b9d3` | 0 |
| Rheon: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/live-ai-comparison/Rheon` | `refs/heads/feat/live-ai-pressure-comparison` | `66e50388ec7a788dbda0ad1227e330996edf506e` | 0 |
| Rheon: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/live-integration/Rheon` | `refs/heads/feat/live-experiment-view` | `a15e160feddeefb1d2b0718b4012405495107f61` | 2 |
| Rheon: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/water-learning-sequence-v1/Rheon` | `refs/heads/integration/ai-pressure-comparison-20261015` | `0fbadbe733b5b521e192dc8a2aa980d6d69a1e01` | 0 |
| Tuldok: `/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok` | `refs/heads/main` | `300f12df6f48e028fd9379911703f7c89337952a` | 0 |
| Tuldok: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/dependency-migration-v1/Tuldok` | `refs/heads/integration/rheon-lab-current-20261016` | `8af5a6a1e288e5b2f86a75b53cab93ca71222a04` | 0 |
| Tuldok: `/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab/runs/live-integration/Tuldok` | `refs/heads/feat/managed-torch-cuda` | `d3514c60d36c978cefefee14bca6ecdb65b3d86e` | 0 |
| Tuldok: `/media/jeremy/OrangeCream/passeur_cache/tuldok-tasks/6240622ff4d89463e085c67b/219a705e-2561-4710-914a-fc0cb8fccac9` | `refs/heads/muse-bridge/219a705e-2561-4710-914a-fc0cb8fccac9` | `6e9e6ec32d4dd719af0e051baafacecd190864c2` | 2 |
| Tuldok: `/media/jeremy/OrangeCream/passeur_cache/tuldok-tasks/6240622ff4d89463e085c67b/a06f60b6-3ee0-4019-9cde-b75f085e9854` | `refs/heads/muse-bridge/a06f60b6-3ee0-4019-9cde-b75f085e9854` | `6e9e6ec32d4dd719af0e051baafacecd190864c2` | 4 |

The Tuldok application PID516904/start5889611 and watcher PID536521/start6106702 match host `/proc` proof. Only command hashes and public paths are retained. Requested Rheon PID493890/start5608030 was unavailable; cessation is not inferred.

Migration tree allocated size: 95.15 GiB. Workspace available: 122.97 GiB; `/tmp`: 46.03 GiB. Host available RAM: 50.49 GiB. These are advisory snapshots, not reservations.

Retain all registered worktrees and unrelated dirty/untracked primary work; deployed current-03, app-env and immutable managed runtime; stages/backups/catalog fixtures; qualification outputs and exact receipts. Source merges and completed M3 do not authorize removing any path consumer.

The working evidence index contains 141 exact receipts. HEAD currently has 41 small migration JSON blobs. Final docs/evidence commit is pending; protected review/CLI reports exist in both primary and isolated docs.

No SQL/API, tests, builds, numerical execution, source/index changes, cleanup, merge/push, process or lease changes were performed. Only resource report artifacts and the separately authorized evidence index/review additions were written.
