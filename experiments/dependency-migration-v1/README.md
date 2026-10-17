# Current dependency selection and migration producers

`config.template.json` selects source roots, retained artifacts, current runtime
manifests, toolchain and output roots independently. It contains no moving Git
revision or source-content pin. Paths resolve from the template, independently
of the process working directory. `artifact_root` is the primary lab root;
existing scientific artifacts remain in its `runs/`, outside the lab worktree.

Freeze a fresh selection after committing or finishing source changes:

```sh
python3 experiments/dependency-migration-v1/config.py --output /absolute/path/to/runs/dependency-migration-v1/selection-attempt.json
export RHEON_LAB_DEPENDENCY_CONFIG=/absolute/path/to/runs/dependency-migration-v1/selection-attempt.json
```

The generator resolves actual full revisions and source-content hashes, verifies
retained artifact hashes, measures the selected current runtime-manifest hash,
and resolves Cargo through the selected Rust toolchain. `--cargo` selects an
explicit executable with the same required version. Generated selections live
outside measured source files (including ignored `runs/`), avoiding a
self-referential source hash or commit pin.
They are exclusive files: use a new attempt path after any source change. The
default output/selection is `runs/dependency-migration-v1/selection-current.json`.
No production path reads the archived host configuration or historical pins.

`dependency_config.load(path=None)` also accepts an explicit path; otherwise it
uses the environment variable above, then the generated default. Its public
operations are:

- `source(name)` returns a root after verifying revision and frozen content.
- `source_root(name)` verifies the selected base revision and path. Build/material
  owners then bind their actual compiled/copied input closure before and after
  work; unrelated documentation or independent workflow edits do not invalidate
  those inputs. The full checkout witness remains separate provenance.
- `source_witness(name)` measures revision, Git tree and actual source files,
  including modified, untracked and deleted files. A dirty integration tree is
  truthfully identified and never represented as its unmodified base revision.
- `artifact(name)` verifies bytes and, when declared, its separate source-content
  binding. A build hash cannot qualify a different source tree.
- `runtime(name)` verifies the current manifest through selected Tuldok's
  explicitly pinned `torch_runtime` source hash, including all installed files
  and link topology. It verifies that owner source again after verification.
- `path_for(name)` resolves custody/output/tool paths without claiming execution.
- `witness()` derives fresh source, artifact and runtime witnesses independently.

Source selection does not confer numerical, model, runtime or catalog acceptance.
Each owner retains its own contract. Selected manifests can change identity while
unchanged installed numerical assets retain their original comparison binding.

The current live project producer is:

```sh
python3 experiments/dependency-migration-v1/dependency_project.py --output /absolute/fresh/current-project
```

It accepts only the current schema-2 `tuldok-project.json`, with `extensions: []`
and mutable activity as ordinary evidence `id: activity`. It checks all retained
reference hashes, resolves source evidence from the selected lab and scientific
artifacts from `artifact_root`, and records the original-to-current manifest and
reference mapping. Bind its project root to primary `artifact_root`; the output
receipt supplies the root-relative manifest path. The byte-exact original
manifest is `dependency-original-project.json`. Mutable activity observation
does not establish scientific custody or acceptance.

The Q3 material producer is:

```sh
python3 experiments/dependency-migration-v1/dependency_training.py --output /absolute/fresh/training-project
python3 experiments/dependency-migration-v1/dependency_training_check.py --project /absolute/fresh/training-project --output /absolute/fresh/material-check.json
```

Preparation creates no catalog, approval, Run, optimizer or simulation. It uses
the independently fixed control's exact captured model/trainer/loss sources and
original training parents. It copies selected `.npy` members byte for byte into
two uncompressed shards, removing all validation parents, preserves train-only
normalization, and changes only the bounded schedule to one update with states
0 and 1. The direction bank's unchanged numeric bytes receive a truthful new
packet-reference binding. Fresh material captures current runtime and custody
owners and records old-to-new provenance. A Q3-only entrypoint replaces heldout
diagnostics with explicitly identified train-only diagnostics; actual optimizer,
loss and CPU restore semantics remain in the unchanged trainer. Receipts state
zero heldout evaluation and no scientific acceptance/model promotion. The
current Tuldok owner handles approval, CUDA execution, cancellation, drainage
and CPU restore; this directory introduces no runner or catalog.

`local-pressure-v1/bootstrap.py` and `project.py` now expose this current material
path. Their superseded producer implementations are frozen offline references
named `dependency-reference-*.py` and are not imported by current runtime code.
The historical numerical model/trainer and all original results/weights remain
unchanged. Former captured-CPU recipe-construction tests were retired with that
producer; current recipe validation runs against real freshly assembled material.

For native builds use `water-residual-learning-v1/build_rollout.py
[--dependency-config FILE] --output NEWDIR`, or `local-pressure-v1/pipeline.py
[--dependency-config FILE] capture-build --config EXPERIMENT --output NEWDIR`.
Both select source and tool paths through this configuration and bind fresh build
witnesses. Old `--prepared`, `--rheon-source` and `--cargo` selectors are removed.

`dependency_runtime.py` is an explicit offline migration utility that produces
current runtime material from a fixed baseline manifest only after every old
field/asset still matches and fresh topology/inventory verification passes.
Normal runtime code does not parse that obsolete baseline representation.

Run the focused configuration regressions with:

```sh
python3 -m unittest discover -s experiments/dependency-migration-v1 -p dependency_tests.py -v
```

These cover source/build mismatch, deleted/new source files, root independence,
duplicate/obsolete selection rejection, and refreezing after a commit without
editing the template. They do not qualify GPU, service, catalog or physics gates.
