# Dependency provenance

Author: Puma. Project license: Apache-2.0, as [LICENSE](LICENSE). This repository does not redistribute compiled binaries, model weights or a vendored solver.

| Dependency | Where | Pinned source | Purpose | Terms |
| --- | --- | --- | --- | --- |
| Rheon | `probes/reference/` | MrScripty/Rheon `bbe01bd3` (branch `lab/reference-solver-v1`) | The solver behind the earlier data and the native baseline | Apache-2.0 |
| Rheon | `probes/cubic-pressure-v1/`, `probes/sealed-water-v1/` | MrScripty/Rheon `43ed26b6` | The capture drivers the earlier datasets cite | Apache-2.0 |
| anyhow, serde, serde_json, sha2 | the probes | crates.io, versions in each `Cargo.lock` | Errors, manifests, checksums | MIT OR Apache-2.0 |
| PyTorch | `requirements-cpu.lock` | 2.8.0+cpu (CUDA builds come from Pumas Library at run time) | Models and training | BSD-3-Clause |
| NumPy and the rest of `requirements-cpu.lock` | Python environment | exact versions in the lock file | Arrays and support | Their own licenses |

Each probe's `Cargo.lock` holds its complete resolved graph and registry checksums. Dependency licenses do not choose a license for this project's own source.
