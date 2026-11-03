# Adopt the remaining portfolio notes through one native owner

Status: Implementation qualified at native contract/HTTP/fork scope in isolated
commit fecd3fca2f383ddd11d88440ab259af3a33bceba; actual211note import pending.
The first real adoption passed; its complete
SDK/status/adoption wall time was 199.166 seconds. This is metadata-operation
time, not model inference. Repeating it for 211 notes is unnecessary project
validation and publication work.

The working project is `projects/rheon-model-portfolio-unified-03`, logical ID
`26122f0d30304598b64dcb814dc411ea`. All nine pressure packets are already admitted.
`notes/portfolio-overview.md` is adopted at revision 1. The other 211 working files
are prepared from `runs/model-portfolio-migration-v1/native-portfolio-cards-04`.
Their producer and independent verification preserve 205 original receipts, 51
exact receipt-bound configurations, 379 aliases and 524 checkpoint identities.
Do not adopt the superseded hashed-filename cards02/cards03 instead.

Implement the batch at `ProjectFiles`, using the same adoption construction as
the single-file operation. Add a native `project_files.files_adopt` operation,
POST `/files/adopt-many`, with an explicit list of 1–40 ordinary adoption intents:
kind, relative path, expected revision and expected SHA256. Keep each existing
1 MiB file limit and the existing 1,000-file project limit. Bound total captured
UTF-8 bytes to 8 MiB per batch. The metadata request contains paths and hashes,
not file content. The result contains ordinary per-file adoption results and a
changed count; it does not duplicate full text in the HTTP response.

The caller retains the native binding, expected checkpoint and request key.
Validate shape, kinds/paths, duplicates and bounds before file reads. Require
the existing edit permission for every target file. Check expected revision and
SHA for every input; a stale/missing/unsafe/changed file prevents publication of
the entire group. Capture files under one before/after-verified root scope.
Reuse the existing anchored UTF-8 ordinary-file reader and its checks. Publish
one canonical ProjectSession mutation after every input is accepted. Do not
loop nested single-file mutations, skip durability, bypass SQL/session owners,
or change old scientific records.

Extract a private adoption constructor only where both single and batch paths
can share their actual semantics. Preserve no-op and exact-request replay,
deleted-note revival/history, original actor/time metadata, fork provenance,
and existing single-file API behavior. The new batch operation receipt must
validate against its exact retained file revisions, including all item IDs,
project IDs, metadata and changed count. Older operation shapes stay valid.

Wire the operation through the authoritative API registry, native dispatcher,
SDK/CLI/MCP surfaces that consume that registry, and affected contract fixtures.
Inspect those existing paths first; do not build a parallel contract registry.
No additional UI workflow is required to run this authorized migration, but the
existing native Files UI must still read every adopted note normally.

Acceptance evidence must cover complete multi-file adoption through the real
owner, permission-before-read, duplicate paths, one changed/missing/unsafe input
with no checkpoint/partial history publication, file/aggregate/project bounds,
no-op/replay, retained history and immutable fork originals. Real HTTP or native
SDK evidence must traverse the new operation. Use the coding standards MCP.

After qualification, close/stop user controller07 before changing application
source or project protocol support. Resume the same installation at the same port
with --resume --keep-open on the new clean source. Reopen the working project;
keep the user UI serving at ordinary turn end. A separate restart qualification
preserves the endpoint, catalog and pre-restart owner session.
Open the actual working project, exclude the already-adopted overview, and adopt
all remaining intents in bounded groups. Retain each request before dispatch;
stop and reconcile unknown outcomes rather than retrying automatically. Verify
all 212 native file records and exact receipt/configuration bytes. Then Save,
Close and independently cold Open under the existing inertness audit. Original
live Tuldok/Rheon and unique weights remain untouched.

The broader goal remains open: combined native charts over all compatible
partitions, curated architecture/purpose/outcome/lessons, honest Trial-free
relationships, remaining old-reference benchmark families and intermediate
states, fresh timings/drift, clean reproduction, original backup capacity and
the separate later new-Rheon epoch. Batch adoption is a means to that project.
