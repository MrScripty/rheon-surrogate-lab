# Design verification

Scope: documents and metadata only. No new-experiment numeric arrays, model source,
weights, training, native compilation, runtime provisioning or GPU execution.
No existing lab/Tuldok source, prior reproduction, freeze or checkpoint was changed.

Public repository and supplied PR merges were inspected through authenticated
GitHub reads. Local lab clone is based on20dd90f6; Tuldok merged contracts were read
atf7543fb7. Source and receipt inspection is not numerical reproduction.

Focused document checks completed:

- Adopted standards188beda `standards_verifier.checks.plan_contract.validate_plan`
  accepted current design-plan structure. This is a structural function check,
  not a Standards Engine route or policy-compliance certificate.
- Reservation JSON parses strictly as finite metadata.112 trajectory IDs/seeds are
  unique and whole histories have one split. Smoke counts12/4/4/4/4; full48/12/8/8/8.
- g4/g5 geometry holdout is disjoint from train/validation; trajectory-test geometry,
  forcing, amplitude and dt tuples are represented in train. Pulse/dt/amplitude
  regime is absent from train/validation. Smoke/full seeds are disjoint.
- 41 existing committed JSON documents/reservations contain35 explicit historical
  seed values, with no intersection with the112 new seeds. Limited to available
  seed-key metadata; implementation must audit all generated lineage/source rules.
- Sizing independently computed from declared arrays: smoke134144bytes/frame,
  raw train38633472bytes; full1060864bytes/frame, all raw23423877120bytes.
  Storage and complete quality/resident/cold workload are prospective cap admission,
  not execution measurements. Full is705600method-frames; smoke56448.
- Local Markdown targets resolve; new files are confined to the design directory.
  Final staged whitespace/scope and public draft publication checks passed.

Independent review required three repairs (storage, full cold workload, full GPU
acceptance mapping); [final recheck](design-review.md) closed all three. All experimental E gates remain
pending. This design does not claim accelerated solver qualification or benchmark
gains. Existing workflows/tests are not rerun as numerical proof for docs-only work.

Publication verified2026-10-14:

- Source/design commit:`74a7a5e061647fb4f626a976dce577b4cdbfefbb`, pushed to
  `origin/design/global-pressure-correction-v1` with no force/history rewrite.
- [Draft PR2](https://github.com/MrScripty/rheon-surrogate-lab/pull/2) is open,
  draft=true, base main20dd90f6, inspected initial head74a7a5e and12changed files.
- Initial staged diff review:12new UTF-8 design/metadata files,113216bytes, no existing
  tracked changes, binary/data/model payloads or observed credential patterns.
  `git diff --cached --check` passed; index bytes equalled reviewed working bytes.
- Adopted plan structure/local links/reservation/scope checks passed. Main checked
  all10 independent reviewer subject hashes before evidence/lifecycle updates.

Subsequent changes record actual publication and design acceptance only; no model,
data, protocol, tolerance, budget or numerical source changes. D1–D6 close the design
task. E1–E9 and full-specific gates remain prospective. This does not merge/admit
implementation or full training. Repository CI is a separate supporting check;
no CI success or numerical rerun is inferred from draft publication.
