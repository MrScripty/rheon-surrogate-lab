# Concurrent pressure experiments

The existing Tuldok catalog owns Models, Studies, Trials, approvals and managed
CUDA training/CPU restore Runs. `run_study.py` is an authenticated client with a
durable per-mutation journal. `benchmark.py` owns standalone, independent CPU
qualification children; its source observation is visible in the existing Project
and Rheon preview. It does not create competing catalog Run records.

Use explicit artifact roots, selected source/runtime paths and configuration
files. No historical checkpoint hash is replaced with a new hard-coded one.
`assemble.py --help`, `run_study.py --help`, `package.py --help` and
`benchmark.py --help` describe the finite operations. Captured inputs and runtime
manifests remain external artifacts; no weights or datasets belong in Git.

The assembly inputs are a verified train-only template, its typed declarations,
an already-consumed probe, the current Tuldok source and a measured Pumas runtime
manifest. The builder measures them before capture, validates current Projects,
recipes and contributions, and constructs a unique provider identity from the
actual material. It never imports project code in the catalog or executes a fit.
The selected Pumas interpreter performs an inert architecture/configuration probe.
The captured trainer owns optimization; current Tuldok SDK owns final checkpoint
and pressure-field evidence. No closed migration gate or converter is reopened.

`configs/control.json` and `configs/lowband.json` use identical data, seeds,
sampling, learning-rate schedule, two training corrections and5000updates. Only
band-feedback weight differs scientifically. Both request1800s CUDA execution;
ordinary CPU restore remains600s. A600s control attempt was retained after a
verified custody timeout, last reported update4500. A fresh Study starts from
initialization; no optimizer resume or automatic mutation replay is permitted.
A failed or uncertain submission must be inspected before any new attempt.

After a known terminal readback failure, `run_study.py --collect-existing` reads
only acknowledged completed training/restore Runs. It never admits work. Existing
regular nonsymlink artifacts are reused only after exact byte-size/SHA verification;
mismatches are preserved and rejected. Network reads stream through Tuldok's
existing full-hash verifier. The private owner credential requires an absolute,
owned mode0600 path and is never included in public receipts.

The measured CPU configuration uses16independent one-thread processes,2GiB
reservation per worker and8GiB observed available-memory reserve. Exclude the
currently assigned CUDA/restore cores explicitly; the retained host configuration
excludes2,3,4 and reserves the first two affinity cores for the desktop. This is
standalone phase accounting, not an OS memory sandbox or a cross-process global
lease. Managed Runs independently use Tuldok's tested reservations/affinity.
Whole-scene tasks are dispatched dynamically rather than fixed serial shards.
A memory/deadline/custody/source failure stops the pool. Only an explicitly
recognized drained native numerical rejection may remain a scientific case failure.
Every parent owns and reaps its worker; rollout owns and drains its native child.

`episode.py` uses the existing current adapter/rollout, exact compiled Rheon input
closure and independent cosine/neighbor-stencil pressure oracle. Physical gates,
32-call maximum, conjunctive pressure/residual stopping, cubic spacing and native
warm-up remain unchanged. Complete74development cases precede any final untouched
scenes. `summarize.py` applies the three population gates independently against
the retained reference, prior accurate late model and harder-data26b control.
A rejected challenger is a retained result. No automatic promotion occurs.

`spatial.py` reuses the controlled analytic cosine/amplitude diagnostic at fixed
network-call counts. `spectral.py` audits233consumed sparse states with bounded
frequency/amplitude bins; it never builds a large state-by-mode tensor. Spatial
frequency is not temporal Hz, and autonomous-state correlations do not establish
a causal noise floor. Pure-low and high-only manufactured fields separate scaling
from cross-band leakage. Same-RHS pressure error and paired trajectory drift must
be reported separately.

`equivalence.py` can reuse this turn's complete control campaign only if a freshly
restored package has exactly the same numerical/executable/runtime bytes and
physical/stopping contracts, and a fresh first-frame input/output is byte-exact.
Only descriptor id/version and measured provenance metadata may differ. Any
numerical difference requires a full new campaign. Reused evidence must be labelled
as reuse, never an additional execution or accuracy improvement.

Methods, failures, timings, memory, source identities and small receipts are retained
in `docs/plans/pressure-throughput-feedback-v1`; raw material remains under the
selected artifact root. `observe.py` publishes bounded recorded pressure previews
through the already-open Tuldok/Rheon paths. Those previews are external observation,
not live velocity/tracer animation or scientific acceptance authority.
