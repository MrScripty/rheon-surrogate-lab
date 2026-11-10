# Training recipes

One file per model in this project, named like its card in `../fits/`. Each says how the model was trained:

- `command` and `entrypoint`: what was run.
- `settings`: the training settings (updates, learning rate, schedule, seeds, architecture choices).
- `inputs`: the prepared files the run read. Each has a `kind`:
  - `training data`, with `training_set` naming an entry of `training-sets.json`;
  - `earlier model weights`, with `from_model` naming the model when the file is exactly that model's output;
  - `not training data`, for the list of files in the Python environment.
- `sources`: the code and settings files the run used.

`scripts/extract_model_recipes.py` writes these from the project's run records, which remain the exact account. Two models were trained before runs were recorded here; their files are read from the run's own folder and have no command.

## How the training data was produced

`training-sets.json` says, for each prepared training set: the script that produced it, its settings file, the command (and whether that command was recorded at the time or reconstructed), what it was built from, and which earlier models' weights it depends on. Its `recordings` section covers the simulated scenes at the bottom of every chain. All scripts and settings files it names are in this repository; the check in `extract_model_recipes.py --check` fails if one is not.

It was traced by hand from manifests, receipts and the scripts, so it is the place to correct if a step is found to be wrong. The prepared files themselves sat under `runs/`, which is not in Git.

Three things limit rebuilding the earlier training sets from this account:

- Ten of the 26 training sets were computed with an earlier model's weights, so that model has to be retrained first.
- Only five have the exact command recorded; the rest are reconstructed from the scripts' arguments.
- The scripts read and write fixed paths under `runs/` and were never run from a fresh checkout.
