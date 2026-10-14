# Integration verification and implementation handoff

The implementation branch derives from reviewed design merge
`4743a16a083040fa32da24903484705c72bd2ce9`. Only the new experiment, probe,
implementation-plan and dedicated workflow namespaces change. Model training,
held-out final arrays, upstream Rheon/Tuldok writes and release tags are absent.
The reservation SHA256 remains
`1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5`.

Source/review checks and complete capture evidence are pending final source freeze.
Publication evidence will identify the exact pushed implementation revision and
verified draft PR; publication is not permission to merge this implementation.

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
