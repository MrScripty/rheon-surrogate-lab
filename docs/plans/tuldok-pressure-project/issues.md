# Findings

- Existing trajectory Cargo manifests use sibling Rheon paths; clean reproduction
  can silently build another revision. Preserve them; new project manifest pins
  historical tested revision and has its own lockfile.
- Tuldok saved checkout was stale; fetched main read-only. Current generic external
  CPU contracts suffice; old sequence-only docs do not describe current capacity.
- No cloud-thread messaging tool is callable; parent/paired thread IDs are not
  collaboration agents. Contract handoff is documented here and in status updates.
- Managed Tuldok Run/UI acceptance requires separate owner/catalog/snapshot/grant.
  Local recipe and viewer evidence cannot substitute for those boundaries.

- Actual viewer output17.3MB exceeded the original16MiB declaration. Numerical
  computation passed; local outer completion was refused. Corrected32MiB cap and
  actual retained-output checks pass. No retroactive Run completion or recomputed
  timing is invented. See reports/output-bounds.json.
- Cargo's empty .cargo-ok completion marker initially caused an overly strict
  clean-source refusal. Exact empty marker exemption is tested; all other source
  changes remain refused, and successful build receipt records the distinction.
- Actual CNN native quality remains0/16, matching its historical negative result;
  it is a separate train/restore diagnostic, never a solver parity claim.

- Automatic approval review denied a prepared comment on TuldokPR58: separate
  public destination lacked explicit disclosure authorization for the detailed
  handoff. No comment posted; destination approval requested. Lab artifacts and
  the in-conversation handoff remain available. This action is not retried or
  routed through another tool while approval is pending.
