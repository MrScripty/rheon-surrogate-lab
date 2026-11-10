# Training recipes

One file per model in this project, named like its card in `../fits/`. Each says how the model was trained:

- `command` and `entrypoint`: what was run.
- `settings`: the training settings (updates, learning rate, schedule, seeds, architecture choices).
- `inputs`: the prepared files the run read. An input with `from_model` is the weights of an earlier model this one started from.
- `sources`: the code and settings files the run used.

`scripts/extract_model_recipes.py` writes these from the project's run records, which remain the exact account. Two models were trained before runs were recorded here; their files are read from the run's own folder and have no command.

The inputs are prepared training files under `runs/`, which is not in Git. A recipe names what was read; it does not yet say how each prepared file was built from simulated scenes.
