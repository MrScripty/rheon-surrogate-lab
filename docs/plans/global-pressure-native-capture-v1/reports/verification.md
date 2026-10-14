# Integration verification and implementation handoff

The implementation branch derives from reviewed design merge
`4743a16a083040fa32da24903484705c72bd2ce9`. Only the new experiment, probe,
implementation-plan and dedicated workflow namespaces change. Prospective global-pressure model training,
held-out final arrays, upstream Rheon/Tuldok writes and release tags are absent.
The reservation SHA256 remains
`1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5`.

Implementation source commit `56b2507b763fe603b0206cd23592d488a58b30d0` was
pushed and GitHub verified [draft PR3](https://github.com/MrScripty/rheon-surrogate-lab/pull/3)
OPEN/draft, base main, exact source head56b2507. Evidence documentation is a
follow-up commit; the numerical source closure remains unchanged. Publication
is not permission to merge this implementation. Native CI at source56b2507 passed [run38548035723](https://github.com/MrScripty/rheon-surrogate-lab/actions/runs/38548035723)
and the existing lab CPU workflow passed [run38548035762](https://github.com/MrScripty/rheon-surrogate-lab/actions/runs/38548035762).
The documentation-only final PR head is verified separately before handoff.

Actual required affected local checks pass:23 native pressure/operator/API tests,
11 preparation tests,45 capture tests with the qualified native binary (56 total),
3 probe tests, Rust fmt, and native/probe all-target Clippy with warnings as errors.
Plan-contract structure passes the adopted standards verifier. All nonpatch
source passes whitespace check; immutable unified-diff blank context lines are
explicit artifact-format exceptions. No overall full native material suite or
experimental parallel qualification is claimed. Independent source and complete
numerical evidence reviews are clear; final standards source/evidence/documentation review is clear.

Next implementation requires parent review of this slice's source and evidence.
Then use complete development train/validation references for the reviewed
small global pressure-correction model, implement the generic Tuldok recipe and
CLI/UI configuration paths, and qualify trained-weight restore before candidate
rollouts. Tuldok managed capture needs separate train/validation plus bounded flat
shards with metadata/transient-copy admission under the selected output profile.
Keep final trajectory/geometry/regime reservations sealed until checkpoint and
all candidate/temporal/native/analytic methods are frozen. Full training stays a
separate conditional local-GPU protocol. No model or speedup claim follows from
this reference/API qualification.

The unchanged historical lab CPU workflow automatically reran its200-step
CNN/fresh-restore regression diagnostic. That is separate historical evidence;
it neither trains this global-pressure model nor changes the prior0/16 native
physics failure or qualifies that CNN as an accelerated solver.
