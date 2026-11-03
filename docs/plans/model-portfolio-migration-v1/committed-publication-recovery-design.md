# Recover the committed composition without another payload copy

Status: native implementation qualified; real recovery pending. Owner: `/root`.

The reboot left a committed, unselected project beside an absent final path.
The original publication call cannot reconcile that path and would create a new
stage if replayed. Rebuilding or Save As would copy approximately 16 GB again.
An unvalidated migration-script rename would bypass the native publication
owner. The selected change is explicit committed-stage recovery in that owner.

`project_fork.recover_committed_publication` accepts the configured root
capability, explicit sibling stage/final paths, expected project/checkpoint,
original publication input and proof of cessation of the former owner. It
recovers the frozen historical snapshot; it does not promise to re-read today's
source projects or reactivate their permissions. This distinction is necessary
because the temporary archive input disappeared in the reboot. It is a local
operator/library capability, with no new HTTP operation or UI claim.

The store's exclusive physical-root lock and full domain validator accept the
project. Its exact original request must agree with both provenance and the
canonical commit receipt. All copied payloads must be present and verified;
captured working files must be clean. Save establishes the existing durability
barriers, then native renameat2 NOREPLACE publishes the same directory. No records,
identity, payloads or SQL projection are synthesized. Existing destinations are
not replaced. A post-rename uncertain acknowledgement is reconciled by verifying
the exact final project on another explicit recovery, without reconstruction.

The write set is the task-owned native source's project_fork.py,
project_projection.py, project_files.py and tests/test_project_publication_recovery.py.
The existing complete projection validator and working-file observation loop are
extracted for reuse; their AST bodies are unchanged. Current UI/service code is
not edited in place. Native source is retained for acceptance and subsequent
serial integration; no remote push, worktree removal or history rewrite is implied.

All 66 affected native filesystem/SQL tests pass, including eight new recovery
cases, complete projection/file checks, composition and a distinct-role cold
database. The initial SQL run lacked the required cold-role fixture for one old
test; the corrected run provisions it and has no failures/skips. See
[qualification](reports/committed-publication-recovery-qualification-01.json).
Real custody, recovery, later native cold lifecycle and final UI remain separate.

Composed-design review: applicable. The publication owner retains filesystem
transition/durability responsibility; domain owners retain semantic validation;
the operator owns current root authority/cessation; the lab's independent reader
owns original evidence comparison. Request/checkpoint checks, locking,
validation and no-overwrite publication are required interleavings. Callers do
not copy payloads, choose conflicts, or inspect canonical filenames. Domain
changes affect the shared domain validator, working-file changes its shared
observer, and filesystem recovery changes the publication owner. Stable request
values and current root descriptors cross those boundaries; no source runtime
or SQL permission is inferred. These concerns have separate filesystem/domain
and original-evidence tests. Removing the recovery entry point would force a
payload recopy or unsafe caller-owned rename, so it supplies necessary behavior;
no new registry, adapter framework, format or second validator is introduced.
The retained complexity is one explicit frozen-stage recovery transition and its
unknown-outcome handling within the existing publication lifecycle.

The archive reader independently compares original scientific observations
against the retained schema8 metadata, checks complete source-envelope bindings
and rehashes the old copied payloads. It also compares the two surviving source
trees literally. It cannot freshly compare inodes/modes against the absent
archive tree; that limit is recorded. An expanded registration-population check
found 58 archived registrations outside the original 62-registration extraction
scope. Those original documents remain in the preserved SQL snapshot; account
for them before claiming the whole portfolio is organized. This does not imply
58 missing fits or missing weight states.
