# One-time conversion, no backward compatibility

Authority: the user's direction after revision 53605af: "We do not want backwards
compatibility. Our migration is a one time thing that you can handle."

This supersedes the ongoing-compatibility recommendations in the earlier review
resolution, especially R4's retained-format/API design. Those documents remain
historical evidence, not current implementation requirements. The current
canonical policy is in [plan.md](../plan.md); procedures are in
[acceptance-protocol.md](../acceptance-protocol.md).

The earlier policies meant keeping old project/material readers, a versioned
inspection API for the local activity manifest, old discovery contracts and
deployment loaders working alongside the new application. That exceeded the
requested migration. They are removed as requirements.

Current decision:

- Convert active projects, activity, recipes, package descriptors and affected
  consumers together into the chosen current contracts during one coordinated
  cutover. No old-client overlap, dual read/write, compatibility aliases or
  obsolete runtime/material/recording loaders in the migrated application.
- Use one temporary offline source decoder/converter. Rehearse it, validate
  complete source→destination accounting and publish while writers are stopped.
  Retire it from deployed code and ongoing compatibility CI after cutover.
- Preserve exact source backups, original scientific results, model weights,
  failures and provenance. Converted representations can have new IDs/hashes;
  their mapping states that fact. Do not claim old work ran under new material.
- If a source Run cannot truthfully become a current Run, preserve and expose
  its original result as a historical import through existing current evidence
  inspection. Do not add a legacy validator or invent a current execution receipt.
  Every source record must have a verified disposition before publication.
- Keep numerical parity, 74-scene regression, CUDA lifecycle, component-specific
  deployment gates, rollback backups and timing/memory records. These verify the
  one-time migration; they do not create ongoing format compatibility.

The coding standards MCP supports this choice: `topic.contracts.evolution` /
Coordinated Breaking Replacement calls for coordinated producer/consumer changes
and removal of obsolete paths when the deployment is owned together. Persisted
Evolution still requires an explicit verified data migration and recoverable
originals; it does not require permanent old-format readers. MCP snapshot:
`snapshot:v1:1d80959f-e983-499c-aa41-ed47dbc1a82f`, 20 selected policies and zero
unresolved routing fact categories. Routing itself is not an acceptance pass.

The composition probe was revised: source decoding is temporary converter work,
normal execution depends on current owners only, and the converter's retirement
leaves no missing runtime capability. No new history framework/catalog is added.

This change revises documents only. No conversion, application source edit,
database mutation, service restart, model training or simulation was performed.
The [verification receipt](one-time-plan-verification.json) binds this revision's
plan/protocol and the checks performed; original review and baseline evidence
remain unchanged.
