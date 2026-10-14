# Live repo project in Tuldok and Rheon

Status: active. Acceptance: pending.
Owner request: keep this repo visible as a Tuldok project while work proceeds,
and display real generation/evaluation in Rheon's egui desktop beside Tuldok.
The root agent owns implementation and verification; no delegated agents.

Branches: lab feat/live-rheon-tuldok-project from 5e8533b;
Tuldok feat/live-project-activity and Rheon feat/live-experiment-view from their
fetched main refs. External dirty working trees remain untouched. Clean source
worktrees remain under ignored runs/live-integration for review and serving.
No model change, GPU support, altered physics gate or historical result rewrite.

Apply pinned Coding Standards188beda Core/Router, implementation, planning,
architecture/contracts, verification including GUI, dependencies and commit.
Use the codebase-design skill to keep live observation separate from scientific
acceptance and immutable Tuldok execution/custody.

Write set: this plan; lab tuldok-project.json; experiments/live-project-v1;
root README; new ignored observer/host outputs. Tuldok projects.py, project UI,
focused tests and contract/runbook. Rheon desktop live observer module, desktop
entrypoint, optional desktop serialization dependencies, tests and README.
Existing CPU model/capture/train/evaluation executable bytes remain frozen.

Sequence:
1. Normal Tuldok project rooted in the actual lab repo; bounded live activity
   inspection and automatic refresh using existing authenticated project routes.
2. Optional external experiment view in Rheon's egui app; physical field geometry,
   actual published frames, progress, errors and terminal state. No app-owned
   simulation is substituted for external work.
3. Lab observer publishes atomic status/previews from actual generation,
   training stdout and native diagnostic progress. Immutable scientific outputs
   and normal managed Runs retain their own authority.
4. Connect to the user's existing Tuldok catalog and desktop session, preserve
   local modifications, verify actual browser/native GUI updates and commit.

Acceptance claims / evidence:
- Repo appears in existing Tuldok UI; files/status change without reopening.
  Authenticated HTTP plus real browser workflow, actual existing catalog.
- Desktop shows real generated/evaluated fields with correct geometry, stage,
  trajectory/frame and clear stale/failed/completed state. Native egui tests
  plus actual desktop observation.
- Bounds, malformed input, atomic publication and cancellation/error state are
  tested at observer/project interfaces; no new numerical model is introduced.
- A bounded real generation and evaluation exercise updates both displays;
  historical executable/results remain byte unchanged.
- Focused affected tests pass; authored changes are committed in all changed repos.

Integration is explicitly authorized by the owner. Preserve existing catalog,
credentials, local app.py edits, in-progress Rheon merge and unrelated processes.
Changes remain local review branches; no push/merge or cleanup of retained evidence.

Owner clarification during integration: future domains use one shared physical
cell size h. Per-axis cell counts control extent, L_axis = N_axis*h; enlarging or
shrinking an axis adds/removes cubic cells rather than stretching fixed counts.
The frozen 16-cubed historical rectangular-cell experiments remain historical.
Their operators account for per-axis spacing, so this is a new grid requirement,
not retroactive evidence that those operators warp space. New numerical profiles
must qualify changed counts independently, including the neural network's shape
constraints. The viewer must display actual counts/extents and support rectangular
count arrays; the observation layer must not resample science to pretend compliance.
