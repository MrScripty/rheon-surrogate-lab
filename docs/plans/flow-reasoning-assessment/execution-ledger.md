# Detached correction experiment execution

The user authorized beginning the nine-fit campaign on 2026-10-18. All nine
Projects passed CPU qualification before managed admission. The order is C53
U/A/D, C44 U/A/D, then C37 U/A/D. Each fit uses 1,600 updates and a separate
managed CUDA Run followed by fresh managed CPU restore.

Qualification verifies the exact starting weights, training-only calibration,
analytical operator labels, unchanged case order, balanced depth order, identical
attached/detached pressure and loss values, finite parameter derivatives, and the
detached prefix boundary. Historical helpers agree at 1/2/4/8 corrections on all
eight fitting cases. Captured common network sources match each original model.
No optimizer updates occur in qualification.

[Project and qualification inventory](execution/prepared.json) records all nine
Project identities and local evidence hashes. C53 U uses `c53-u-project-02`;
its first prospective capture failed local dependency validation before admission
and is preserved. The remaining arms use their respective `project-01` captures.

The first controller attempt stopped during session authentication. Inspection
confirmed no study directories, mutation journals or active Run records. Its
[failure record](execution/execution-attempt-01.json) is preserved. Automatic
approval review rejected probing alternate credentials. The user then explicitly
approved the saved owner session at
`runs/tuldok-performance-v1/private/owner-session.txt`. An explicit continuation
verified the zero-mutation failure and used a fresh C53 U controller directory.
Credential contents are not included in experiment evidence.

The first admitted training Run is `d4211bfd98de484d941682eb93a8212c`, under
Project `af9e6df7326a41b0a266ca938f177726` and Study
`d018822753ca41fc9abfe6067df885d9`. The existing
[Tuldok model workspace](http://127.0.0.1:8091/?workspace=models) owns admission,
CUDA execution, CPU restore, bounded resources and artifact custody.

The [live campaign state](execution/execution.json) records completed arms and the
active arm. Per-arm training receipts are published there after successful
restore. C53 U must reproduce retained C66 weights exactly before later fits.
All arms must preserve their expected update, example, depth and forward counts.
Execution or numerical failures stop the campaign without automatic retries.
Accuracy failures remain reportable outcomes and do not remove the other groups.

## Completed training and evaluation

All nine CUDA training and fresh CPU restore workflows completed with exact
artifact parity. The C53 U control reproduced retained C66 weights exactly.
Actual optimizer work was 14,400 updates, 67,200 network forwards and 77,952,000
forward cells. Summed trainer elapsed time was 737.05 seconds; summed managed
workflow elapsed time was 2,383.66 seconds. These measure training and workflow
costs, not solver acceleration.

CPU evaluation initially stopped at the original C44 fitting evaluator: its
historical input view includes 16 training and 64 validation parents, so the
strict train-only calibration guard rejected it. Those validation parents supply
exactly the same 128 consumed regression requests; the view contains no reserved
test parents. No nontraining value entered calibration or any optimizer.

The corrected evaluator freezes the exact common eight-case bank from the C53
train-only calibration used by all actual fits, then evaluates every checkpoint
on those tensors. Its 32 C53 case/pass measurements exactly reproduce the
original fitting evaluator. The failed attempt and successful existing outputs
are preserved; only missing CPU measurements were resumed. No training or remote
admission was repeated. [Recovery evidence](execution/evaluation-recovery.json).

All 12 checkpoints now have fitting and consumed-real measurements. The nine
candidates and retained C53 also have the 240-case manufactured diagnostic.
[Evaluation lifecycle](execution/evaluation.json) records each new worker's
exit and reaping, with prior failure evidence retained separately. No candidate
passes the full fitting or independent-scene gate, and no model is promoted.
[Complete results](execution/results.md), [interpretation](outcome.md).

After training, the campaign performs isolated CPU evaluation on
the eight fitting cases, the 128 consumed real requests and the retained 240-case
manufactured diagnostic. Each starting checkpoint gets its own untouched fitting
and real-request reference measurements; C53 defines the existing manufactured
deployment guard. Evaluation status and final scalar summaries are written under
`execution/`. Original data, model checkpoints, validation membership and physical
gates remain bound to the recorded experiment. No model promotion or independent
holdout consumption occurs in this campaign.

Implementation and command usage are in the
[experiment README](../../../experiments/pressure-detached-correction-v1/README.md).
