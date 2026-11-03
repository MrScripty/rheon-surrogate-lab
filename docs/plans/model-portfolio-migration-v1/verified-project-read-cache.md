# Reuse verified decoding inside a single portable store

Owner/integration owner: /root. Private proposal branch
codex/verified-project-read-cache-20261024, based on dd1feca. Status: isolated
implementation passes 41 store tests after the recorded failing reproducer;
all 173 affected native/SQL tests now pass. Candidate commit
1f1e3bc5cf6210ed9658256b869389d60d8d3898 is retained on the private proposal.
Commit bf2746e adds active receipt selection, with all 226 affected checks passing.
Both are now fast-forwarded into the composition source after the first complete
cold lifecycle and restoration. The fixed-source cold run02 is live; final UI
acceptance remains required. The private proposal stays retained-protected for
qualification/review until that acceptance, then clean removal with accepted
commits reachable from the composition integration branch.

The user's nearly immediate Open expectation is unsatisfied by the actual
53,873-record cold Open, still CPU-active after more than 50 minutes. The current
bounded validator profile spends 88.6 of 90 profiled seconds in composition
membership decoding; canonical serialization takes most of that CPU. This
locates one contributor, not a complete allocation of live Open time. The same
canonical encoding/decoding is repeated by every portable-store snapshot. The
ranking to investigate is repeat snapshot validation, within-call membership
hashing, then copying/domain reconstruction. No end-to-end speed claim follows
from the isolated tests or partial profile.

First mechanism: after a fully successful load, retain only that load's exact
manifest bytes, decoded manifest and decoded records in the private Store
instance. A later load may reuse decoded objects only when the current manifest
bytes are identical AND every referenced canonical file is freshly read through
the existing anchored reader and matches its declared length/SHA256. Both
manifest pointer observations and root/metadata physical-binding checks remain.
Any mismatch fails with the existing error; there is no fallback to old values.
Identical canonical bytes already passed JSON/receipt/root validation, so
reparsing and re-encoding them provides no new integrity evidence.

The cache must never cross Store instances, persist to disk, treat timestamps as
content identity, publish caller-owned mutable objects, or replace per-payload
verification. Public snapshots retain their independent deep-copy contract.
Private root borrowing can use the verified current header/checkpoint without
making an unused full public snapshot copy. Close releases the private cache.
Commit and changed-manifest paths still perform their existing full validation.

Write set: project_store.py and its focused real-filesystem tests; the existing
artifact tests must observe the verified `_current` closure rather than the
unused public snapshot copy, retaining both before/after check counts and all
literal payload assertions. Native/SQL tests and bounded measurement tools,
plus reports/plan/ledger/state, are also in scope.
Do not change operation schemas, project formats, receipts, domain validation,
artifact roles, current cold/UI source, or project contents. The proposal
worktree remains protected until serial integration and real UI acceptance;
then remove it cleanly with accepted commits reachable from the integration line.
Integration target: codex/composition-source-artifacts-20261024 in
`runs/model-portfolio-migration-v1/sources/Tuldok-composition-source-artifacts`.
Do not advance that target while the original cold controller uses dd1feca.
The proposal is retained-protected, with its exact head reachable from
`codex/verified-project-read-cache-20261024`; it supports only isolated
qualification until the cold owner's terminal outcome permits integration.
If target authority changes first, re-evaluate integration and verification;
do not rewrite the live source or automatically cherry-pick.

Next isolated slice, admitted before implementation: the exact combined profile
also shows repeated canonical envelope hashing in composition membership
validation. Extend this private proposal with project_composition.py and its
existing real composition tests, plus one read-only exact-checkpoint codec
measurement. Demonstrate repeated hashing at the complete decode_records seam
before changing it. Reuse a digest only for the same indexed envelope inside one
membership validation invocation; preserve every source membership comparison,
Run-observation distinction, canonical numeric spelling and all later-call
mutation checks. No cache may survive an invocation or admit caller-supplied
digests. This is a separate measured contributor; the prior Store qualification
remains pinned to 1f1e3bc, not automatically to a later proposal head.

Acceptance of the mechanism requires a fast deterministic red/green reproduction
of repeat parsing, actual reads on every repeat, unchanged values and independent
snapshots, corruption/root replacement/different-manifest refusal, canonical
commit/Save/reopen behavior and appropriate native integration tests. Measure
actual retained metadata without a full project/payload fixture; any generic
canonical-store-only measurement of closed compacted source metadata is strictly
read-only, excludes ProjectProjection/Session/SQL/model use, and cannot qualify a
complete native Open or authorize reuse of missing source artifacts. This is an
explicit diagnostic exception to the compacted-input reuse prohibition.

Final acceptance still requires the complete native project and user UI Open
path, with measured latency and exact preservation. A faster helper or a
successful small fixture alone is insufficient. Continue diagnosis if measured
whole-project behavior remains too slow; do not describe this cache as making
Open instant before observing it.
