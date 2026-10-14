# Frozen pressure model diagnosis and next proposal

**Plan status:** `Implemented`

**Current phase:** Reviewed lifecycle repair and exact-head CI publication

**Next slice:** Parent review of diagnosis and lifecycle repair after exact-head CI

**Acceptance status:** `pending`

**Execution ledger:** [execution-ledger.md](execution-ledger.md)

**Issues:** [issues.md](issues.md)

## Objective

Parent accepted exact PR4 1a976d6 and authorized normal merge followed by diagnosis
of the frozen final weights. Explain observed zero corrections without training,
final-test access or modification of original scientific configuration. Propose
one narrowly scoped next experiment from measured development evidence, with
independent numerical/standards review and explicit uncertainty.

## Objective Acceptance

| ID | Criterion | Kind | Environment | Mode | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Reconfirmed accepted head/checks and normal merge receipt | release-artifact | required-real public GitHub PR4 | automated | satisfied | execution-ledger.md |
| D2 | Frozen source/data/checkpoint identities and all384 existing development frames; no optimizer/state mutation/final access | contract | required-real existing qualified CPU data and weights | automated | satisfied | reports/diagnosis.json |
| D3 | Initial reconstruction versus final pressure/loss, activation/head spatial variation, gauge behavior, bounded gradients and retained history measured | focused | required-real deterministic one-thread CPU Torch2.8 | automated | satisfied | reports/diagnosis.md |
| D4 | Numeric facts separated from architecture/scaling/optimization hypotheses; proposed next experiment and acceptance gates, no training launched | contract | representative primary sources plus actual development measurements | either | satisfied | next-experiment.md |
| D5 | Independent numerical/standards review and source/evidence preservation, publication handoff | release-artifact | representative independent GPT6.1SolHigh and real Git | either | satisfied | reports/review.md |

## Binding Decisions

Tracked writes to new `experiments/global-pressure-v1/diagnosis/**` and
`docs/plans/global-pressure-diagnosis-v1/**`. Parent subsequently authorized the
identified lifecycle fixture repair in `scripts/test_tuldok_pressure.py` only,
with regressions and CI; this is an explicit extension of the prior write set.
Production cleanup/executor and all scientific training/model/tests/config,
all receipts/negative results, split reservations, native code, Tuldok and prior
plans remain unchanged. Main owns plan/research/publication; diagnosis worker owns
new diagnostic script and generated report in this new directory only. Reviewers
are read-only. No optimizer is constructed or step invoked, no checkpoint is
rewritten, no final/full/GPU arrays are generated or inspected. No scientific
Run or training submission is authorized in this slice.

Read exact packet2 five already admitted files and existing final weights/norm/
receipt/prediction bytes, with their frozen hashes and old reader. Initial weights
are reconstructed from original seed20261014 and exact architecture/package;
they are not a retained initial checkpoint. Check reconstructed initial and final
validation loss against original receipt to establish supported reproducibility.
Population is all384 existing development frames (288train/96validation), with
all32 original witnesses identified separately. Bound one-thread CPU diagnosis to
180s work and4GiB observed process peak RSS. Report aggregate and per-row raw head
spatial variation/gauge-fixed correction/loss components/activation ranges. On the
four validation parents at frames0/23, compare initial/final autograd gradients,
raw head gradient versus parameter gradient, and fixed-scale finite differences;
never alter persisted weights or create an optimizer. Ephemeral precision probes
may cast copies to f64 to distinguish representational rounding; they do not
constitute a candidate trained model or a deployment configuration. Hash all
input/source/artifact bytes before and after measurements; finite/range checks
apply. Retained history is only every20updates; intermediate weights/Adam moments
must not be inferred if absent.

## Simplicity And Ownership Review

**Applicability:** `applicable`

- Independent concepts and dimensions: Existing model/data remain authoritative; diagnosis owns measurements and proposed scientific policy.
- State, identity, value, time, policy, and mechanism: SHA selects immutable bytes; derivatives are measurements, not updates; historical logged steps do not supply missing checkpoints.
- Caller and composition-root knowledge: Diagnostic explicitly binds existing sources/packet/weights; no catalog/grant/optimizer dependency.
- Representative change paths and forced owners: Any later architecture/scale/loss change needs a separately reviewed scientific design.
- Stable Interfaces versus hidden knowledge: Reuse exact reader, model, normalization and inventory; report measurement scope and tolerances.
- Independent evolution, testing, failure, and replacement: Reviewers verify actual read-only measurements and distinguish causal facts from hypotheses.
- Necessary complexity and containment: Hooks/autograd/precision comparison confined to new diagnostic script and bounded original development population.
- Deletion and cumulative machinery result: No copied trainer, scheduler, executor or new generic Tuldok contract.

## Milestones

| Unit | Owner | State |
| --- | --- | --- |
| Merge and preserve | Main | Complete |
| Measurements | Diagnosis worker; main integration | Complete |
| Proposal/review | Main and independent reviewers | Complete; parent acceptance pending |

## Blockers

No source, evidence or independent review blocker remains. Both independent reviews cleared
frozen observations, compact/archive preservation and the proposed next experiment.
Draft publication/parent integration acceptance are next; new scientific training
and final-test exposure remain outside scope. Merge completed normally.

## Re-Plan Triggers

Hash drift, missing trained bytes, package mismatch, runtime/RSS budget or unexpected
nonfinite values stops affected measurements. Record unavailable history honestly;
no regeneration of training or arbitrary configuration search.

## Final Acceptance

- Acceptance status: `pending`
- Final status: `Implemented`


The diagnosis is implemented and independently reviewed. Plan acceptance remains
pending parent exact-head review. The next-experiment document is a proposal, not
an approved scientific configuration or authorization to train.
