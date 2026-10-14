# Live lab project

`tuldok-project.json` opens the actual repo in Tuldok. The live project panel
shows producer activity and recent file changes. Selected unpinned text/source
files refresh automatically; declared hashes and immutable execution snapshots
still retain their original identity and rejection behavior. Activity is external
observation, not a fabricated completed Run or scientific qualification.

The current host serves the repo through the user's existing Tuldok catalog on
127.0.0.1:8091 with `--project-root lab=/absolute/lab`. Use `/workspaces`, choose
**Rheon Surrogate Lab · cubic cells**, and inspect its Data/Pipeline/Runs views.
A registered project also supports `/workspaces?project=PROJECT_ID#data`.
Existing catalog data, credentials, sessions and local source edits are preserved.
The local integration branches are `feat/live-project-activity` in Tuldok and
`feat/live-experiment-view` in Rheon. Their clean source worktrees are retained
under ignored `runs/live-integration`; they share the original repositories'
Git history and do not change the dirty worktrees or in-progress merge.

Open Rheon's **actual egui desktop**, using the qualified integration build:

```sh
runs/live-integration/native-target/debug/rheon-desktop \
  --live-project "$PWD/runs/live-project/frame.json"
```

This optional view observes external publications. The normal app-owned worker,
archive inspection and numerical solver remain separate. Tracer points use the
physical cell centres and box aspect ratio. Pressure slices retain physical
axis scales, field labels and Pa units. Follow the latest publication or scrub
the latest 64 retained frames. Display history is bounded; full scientific data
remains in the capture/evaluation directories. Missing/malformed data and a
producer with no new running frames are visible, rather than replacing it with
an unrelated app simulation. Close the observer window when finished.

## Historical fixed-count capture and evaluation

For current variable-domain generation, use the [cubic-cell setup](../cubic-pressure-v1/README.md).
The commands below preserve the earlier fixed-count capture/evaluation profile.

Launch with the local qualified scientific interpreter and explicit matching
configuration/binary. Never reuse an existing output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/live-project-v1/live.py capture \
  --config CONFIG.json --binary /absolute/qualified-capture-binary \
  --output "$PWD/runs/new-live-capture"

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python experiments/live-project-v1/live.py evaluate \
  --workspace /absolute/completed-cpu-workspace \
  --output "$PWD/runs/new-live-evaluation"
```

Capture first independently qualifies one train frame, then publishes observations
of real native frames during complete generation and verification. A candidate
preview is labelled pending qualification until the complete capture passes.
Evaluation uses the existing independently restored prediction bytes and frozen
native evaluator. Its journal drives the display. The scientific worker remains
single-threaded, with live observation in its parent; the original native owner
controls its own descendants and physics gates. Cancellation asks the worker to
drain before the observer's outer lifetime ends. Interrupted/failed outputs remain
inspectable; the adapter never silently retries a numerical attempt.

The evaluation view explicitly shows **teacher-state diagnostic** scope and
**reference tracer**. Its pressure slice can show a failed learned proposal,
labelled accordingly; this is not an autonomous learned fluid rollout. No model
architecture, weights, historical numeric outputs or physical tolerance changes
are introduced by the observation adapter. Live display overhead is additional
work and establishes no acceleration benchmark.

Actual managed training can be followed with Tuldok's existing scoped agent client:

```sh
.venv/bin/python experiments/live-project-v1/live.py watch-run \
  --tuldok-source /absolute/Tuldok --agent-config /private/agent-config.json \
  --run-id RUN_ID
```

This reads normal authenticated Run progress, including its training updates,
and publishes it in the project panel and current desktop status. It neither
submits nor cancels a Run. The configured catalog owns Run identity/custody;
watching does not copy a catalog or its credentials. Observation has a bounded
600-second default lifetime, configurable up to 3600 seconds.

## Inspect exact fields in Data

`bundle.py` exports one whole independently verified trajectory to Tuldok's
scientific-field format. It wraps original f32/f64 bytes in NPY1 headers, retains
SI geometry/units and copies original capture/verification evidence. It performs
no resampling, pressure inference or dtype conversion. Tuldok's own parser
checks the complete ZIP before publication. Import it through a normal project
Data reference; the new record is a draft and human review/release remain separate.

```sh
.venv/bin/python experiments/live-project-v1/bundle.py \
  --capture /absolute/verified-capture --trajectory TRAJECTORY_ID \
  --tuldok-source /absolute/Tuldok --output "$PWD/runs/new-fields.zip"
```

## Grid requirement

Future experimental domains use one shared cell size. At 16 cells per metre,
`h = 1/16 m`, and lengths `Lx,Ly,Lz` require counts `Lx/h,Ly/h,Lz/h`.
For 1.5×1×0.75 m, counts are 24×16×12; cells stay cubic. Noninteger cell counts
must be resolved explicitly, not by stretching cells. The frozen older CPU
profile used 16³ counts with axis-dependent spacings and remains historical.
The current GN model accepts only 16³/32³ arrays; rectangular-count support needs
an independently qualified model/data profile before training. The observer
supports rectangular count arrays up to its 65,536-cell/2-MiB publication bounds.
