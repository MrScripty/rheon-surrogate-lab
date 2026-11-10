# surrogate

Learned fluid steppers for Rheon scenes, with one benchmark that scores every model the same way.

This package is new code. The studies under `experiments/` and `docs/plans/` are the earlier work and stay as they are; two of their solvers are wrapped here as benchmark baselines.

## What is here

| Path | Purpose |
| --- | --- |
| `fields.py` | Staggered-grid layout and operators (divergence, curl, sealed potential). |
| `scenes.py` | Builds a scene's full parameters from a seed, a grid and a forcing type. |
| `datasets/` | Dataset definitions (small scene tables) and the driver that simulates them. |
| `solver.py` | Runs the Rheon build behind the reference data, one output step at a time. |
| `forcing.py` | Rebuilds a scene's forcing as the velocity it adds per output interval. |
| `data.py` | Reads trajectory banks: generated datasets and the earlier recordings under `runs/`. |
| `models/` | A 3D U-Net and the stepper built on it. Velocity changes are the curl of a predicted potential, so every output is divergence free and sealed whatever the weights are. |
| `train.py` | Trains a stepper on its own multi-pass rollouts with noisy starts. |
| `bench/` | The benchmark: cases, subjects, metrics, timing, result documents and Tuldok packets. |
| `configs/` | Training configurations. |
| `tests/` | Unit tests; they use small synthetic data and run on a CPU. |

## Environments

Tests and anything on the CPU run in the lab's `.venv`. Training and GPU subjects need the CUDA runtime that `scripts/pumas_trajectory.py` resolves (PyTorch 2.14 from Pumas Library).

```sh
.venv/bin/python -m unittest discover -s surrogate -t .
```

Set `TULDOK_CONTRACT_ROOT` to a Tuldok checkout to also run Tuldok's own packet validator over the packets this package writes.

## Datasets

A dataset is a definition, not stored fields: a table of scenes, each built from a seed, a grid and a forcing type. The simulated fields are produced by running the solver and are never committed.

```sh
python -m surrogate.datasets list
python -m surrogate.datasets plan diversity-v1                       # scenes, states and disk needed; runs nothing
python -m surrogate.datasets generate diversity-v1 --split development
```

`generate` writes every output step of each scene under `data/reference/<dataset>/` (ignored by Git; change it with `--root` or `$SURROGATE_DATA`). It runs one scene after another on one CPU thread, skips scenes already on disk, and can be stopped and restarted. Reserved scenes are only generated with `--include-reserved`.

| Definition | Scenes | What it is |
| --- | --- | --- |
| `diversity-v1` | 148 train, 24 development, 52 reserved | Scheduled forcing, six cell densities from 8 to 100 per metre, grids up to 128³. The lab's earlier recordings of these scenes kept only short windows; the reserved scenes have never been simulated. |
| `sinusoidal-v1` | 16 train, 8 development, 88 reserved | The older scenes on four grid shapes. All 88 reserved scenes have already been used to check an earlier model. |

Thirty `diversity-v1` scenes need eight solver substeps per output; the driver skips those until that solver build is wired in.

## Benchmark

```sh
python -m surrogate.bench suites
python -m surrogate.bench run --suite dense-development-240 \
    --subject persistence --subject rheon-native --subject spectral-composition:1 \
    --subject checkpoint:runs/<training run>/model.pt --output runs/<name>/result.json
python -m surrogate.bench packets runs/<name>/result.json --output-directory runs/<name>/packets
```

Each subject starts from a case's first recorded state and is fed its own outputs from then on. Every output is scored against the recording, and the first pass is timed after warm-up. `bench/metrics.py` defines each number. The result document records the Git commit of the code; `run` refuses uncommitted changes unless told otherwise.

The packets are Tuldok `retained_measurements_v1` files. Importing them into a Tuldok project is done from a running Tuldok and is not automated here yet.

Subjects:

- `persistence` returns its input. A model that does not beat it has learned nothing.
- `rheon-native` is Rheon's own solver through the lab's probe, on one CPU thread.
- `spectral-composition[:K]` is the earlier GPU solver composition (the "F64 cell composition"). It reproduces the reference to rounding and is listed as a solver, not as a learned model.
- `checkpoint:PATH` is a stepper trained here.

## Training

```sh
python -m surrogate.train surrogate/configs/pilot-dense-k1.json runs/<name> --device cuda
```

The run writes `model.pt` (what the benchmark loads), `resume.pt`, `config.json` and `log.jsonl` into the output directory.

## What needs local files

The Rheon probe, the solver composition's kernel builds and the earlier recordings live under `runs/`, which is not in Git. On a fresh checkout the tests pass, but generating data, the suites, training and the two solver subjects need those files first.

The probe is built from a Rheon source tree with the lab's own changes. Those changes are not yet in the Rheon repository or in this one, so the probe cannot be rebuilt from a fresh checkout today.

Of the earlier recordings, only the sinusoidal scenes keep every output step. The diversity recordings keep short windows of at most four outputs, which is what the earlier models were fitted on.
