# Frozen development model diagnosis

Run from the lab root with the qualified local CPU environment:

```sh
/workspace/task-torch/bin/python -m unittest discover -s experiments/global-pressure-v1/diagnosis -p 'test_*.py'
/workspace/task-torch/bin/python experiments/global-pressure-v1/diagnosis/diagnose.py \
  --output docs/plans/global-pressure-diagnosis-v1/reports/new-local-observation.json
```

The default inputs are original packet2 and the accepted training-stage artifacts.
The script requires their exact frozen hashes and the exact original reader/model/
trainer/test bytes. It constructs no optimizer, performs no state update, and
accesses no final data. Initial weights are reconstructed from seed20261014, not
a retained initial checkpoint. It checks original batch2 validation evaluation
against the training receipt and bitwise agreement with all32 retained witnesses.

The immutable default output is
`docs/plans/global-pressure-diagnosis-v1/reports/diagnosis.json`. For independent
reproduction, pass `--output` with a new path in this diagnosis directory. Existing
reports cannot be overwritten. The work has a180s wall deadline, one CPU thread
and a4GiB observed peak-RSS guard. RSS observation is not adversarial containment.

The first complete17,921,455-byte observations are preserved locally at
`runs/global-pressure-v1-diagnosis/full-report.json`, SHA256
`27a820e6486e1eb455d0006b5e7975ada689014e9e1d79b54721165f558e7d89`.
The published report is a compact derivative made by `summarize.py`, without
rerunning model arithmetic. It keeps every row's loss/input/head and final
decoder-stage measurements, all16 derivative cases, identities/reproduction
proofs and aggregate ranges/zero counts for other stages. The complete archive
is ignored development evidence; `diagnose.py --output` can reproduce its
measurements on the pinned original inputs if the archive is unavailable. A rerun
creates a NEW observation/hash because timing/RSS and publication metadata vary;
it cannot recover the exact original archive bytes. The compact report's archive
availability phrase refers to regenerating measurements, not exact artifact
recovery. Existing full and compact observation bytes remain preserved.
The summarizer is a one-time publisher for the exact first report, and refuses
to overwrite any report that is not that exact full observation.

The reported16.31s measurement counter ends after final identity checks and
excludes imports and final JSON encoding/publication. The compact report records
this scope and a separate postprocessing clock; neither is an enclosing workflow
cost claim. The180s wall alarm remains active through full-report publication.

Per-row stage statistics distinguish within-channel spatial range from variation
between channels. Hooks observe pre-convolution values, convolution/SiLU outputs,
global pooling, and mathematical SiLU derivatives evaluated in f64 at the actual
preactivation. The latter are arithmetic observations, not f32-autograd claims.
Population forward measurements use batch2, matching the original validation
evaluation. Fixed derivative cases and the original witness check use batch1;
f32 convolution associations can differ between batch sizes. Receipt validation
is checked with the original evaluator, and witness pressure bytes are checked
with the original witness inference routine, rather than inferred from the
population-stage statistics.
Fixed8 validation cases receive component-loss derivatives, gauge/raw/input/stage/
parameter gradients and fixed-direction finite differences. Head-only f64
arithmetic uses the same f32 decoder features; full f64 arithmetic uses ephemeral
copied weights and explicitly replayed stages after checking exact f32 replay.
Bias-free head evaluation is an algebraic observation, not a candidate model.
No intermediate weights or optimizer moments are recovered from every20-update
logs, and these measurements do not establish the historical cause of training
behavior.
