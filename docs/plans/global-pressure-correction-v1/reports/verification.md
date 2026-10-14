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
  Final staged whitespace/scope and public draft publication checks remain pending.

Independent review required three repairs (storage, full cold workload, full GPU
acceptance mapping); [final recheck](design-review.md) closed all three. All experimental E gates remain
pending. This design does not claim accelerated solver qualification or benchmark
gains. Existing workflows/tests are not rerun as numerical proof for docs-only work.

Commit/push/draft PR publication: pending.
