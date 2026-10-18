# Tuldok performance qualification — 2026-10-18

The scheduling repair is active at http://127.0.0.1:8091 in the existing Tuldok
catalog. Model training remains paused. No scientific inputs, weights, model
settings or historical results were replaced.

The coordinator repeatedly loaded and validated all 55 historical Runs while
holding the catalog lock, even with an empty queue. This occupied one CPU core
(about 4% of this 24-CPU system) and blocked normal reads. The repair selects
queued/active candidates in PostgreSQL, fully validates each selection, and waits
for committed admission when idle. Dispatch, authorization, artifact integrity,
startup recovery, capacity and queued resource retries retain their checks.

| Measurement | Old coordinator running | Repaired, after a real attempt |
| --- | --- | --- |
| Same completed Run, two reads | 7.351–8.271 s | 0.443–0.446 s |
| Same 189-byte artifact, two reads | 7.674–7.889 s | 0.393–0.394 s |
| Idle application CPU in ~3 seconds | 2.950 CPU seconds | 0.000 CPU seconds |
| Same restore-graph preflight | 15.104 s | 14.263 s |

The original read reproduction uses the same script hash and existing owner,
catalog, Run and artifact. Two seconds is a diagnostic trigger, not a product SLA.
An additional three-sample after measurement gives Run median .527 s (range
.521–.527) and artifact median .476 s (.456–.495); its first reads were near the
end of restore-artifact collection. All numbers are observations, not an exact
paired speedup or a claim about whole-training throughput.

The old coordinator subsequently exited on an advisory-lock timeout. The direct
full-history diagnostic held that lock longer than its 15-second timeout and may
have triggered the exit. Later old-source reads were .46–.55 s because the
coordinator was dead; that receipt is explicitly retained as
reads-before-coordinator-exited.json and is excluded from healthy comparisons.
An exited thread is not a performance success. Arbitrary coordinator-exception
recovery remains a separate existing resilience finding.

Verification used isolated PostgreSQL 16: 7 focused scheduling checks pass,
including admission wake, timed resource retry, capacity and corrupt selection.
The broader 40-test run has 39 passes and one unavailable legacy image-trainer
runtime, failing identically on the unchanged base. The focused checks are
included in that broader total, not additional tests.

Our actual Pumas path was separately qualified through the live owner API with
one journaled approval/submission of the existing C67 zero-update CPU restore.
Run 29472d8e679a48b1933d4683e401ad02 completed with drained custody. Approval took
13.937 s, admission 27.849 s, completion from controller start 78.097 s, and all
artifact readback finished in 100.273 s. Ten artifacts whole-hash verified; nine
expected unchanged outputs match exactly, including weights, normalization,
history and predictions. The receipt differs only in elapsed/inference timings
and peak RSS. The deployed coordinator then remained quiet through the read
benchmarks, with three app threads and no thread exception in its log.

Preparation remains substantial: the graph preflight still takes ~14 s;
new restore artifacts take ~2.1–2.3 s each because their captured input closure
differs from the old selected Run. Historical C67 admission was 210.781 s, but it
is context, not a controlled timing pair. The next performance investigation
should profile current material validation in preflight, approval and admission,
while preserving authority and integrity. No broader speed claim is made here.

The canonical completed implementation plan and ledger are in Tuldok at
`docs/plans/request-latency-v1/` on branch
`performance/catalog-request-latency-20261018`, final commit
`de0f1b833f1922b4a86c12036405fe8cd795b4a0`.
Product repair/deployment source: `eed2b84d21050943d211defffc3133609422950a`.
Admitted base: `225f8e862f6ee7a6f778522c6e15a5fbc97fe5ca`.
The two-commit range was reviewed; only runs.py changes product behavior, with
regression tests and qualification documentation. Origin/main 300f12d was fetched
and is already an ancestor of that integration; existing CUDA work was preserved.
No push, merge or history rewrite was performed.

The lab branch `performance/tuldok-request-latency-20261018` starts at
`3ae583618605c36c26b5bf710cbb40172335b67f`. It owns reusable experiment tooling and
retained integration receipts, not a second implementation plan. Jeremy owns
review/integration into this existing lab integration line. Retain the branch
unchanged for review, with no automatic synchronization; retire after accepted
integration or explicit disposition. Consumers are the local qualification and
Tuldok proposal review. Unrelated working-tree edits were preserved.

The clean task-created Tuldok worktree was removed after protecting its exact
final head on the named branch; [disposition](reports/worktree-disposition.json).
The running deployment is an independent immutable current-07 copy. The isolated
test PostgreSQL service was stopped; no test/profiler process remains. Credentials,
model weights, datasets and binary profiles are excluded from commits.

[Reports](reports/) retain timings, CPU, test outcomes, deployment identity and
numerical comparisons. [Command sources](command-sources/) retain the exact
one-time diagnostic/deployment sources for audit, including their historical
process IDs; these are not automatic deployment/retry tools. Reusable read and
zero-update restore probes live in `experiments/tuldok-performance-v1/` with
workload selection in config.json. API mutations are journaled and never retried.
