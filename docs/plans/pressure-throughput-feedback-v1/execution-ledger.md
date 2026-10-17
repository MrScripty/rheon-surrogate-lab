# Execution ledger

## Admission and scheduler discovery

The user authorized concurrent CPU/CUDA execution followed by the proposed
pressure-feedback sequence. Created separate Tuldok and lab branches from the
completed migration heads. Protected the original checkouts and existing PR89.
CPU preparation used four processes and regression had an explicit two-worker
cap; Tuldok's catalog coordinator executed one attempt synchronously.

Built `test_run_concurrency.py` through the existing real PostgreSQL/ExternalCase
and actual neutral subprocess custodians. Before the fix, the first attempt was
running/go_intent and the second stayed queued/prepared after5 seconds; the
test failed in9.659s. An initial isolated cluster used SQL_ASCII, rejected by
Tuldok's UTF8 contract before the test; created a separate UTF8 test database in
the same new private Unix-socket cluster. No original catalog was used.

The first receipt harness incorrectly resolved the virtual-environment Python
symlink into system Python, losing psycopg. Preserved scheduler-tests-01.json;
corrected interpreter selection to retain its absolute venv path. Tests-02 and
04 each retain an unavailable legacy TinyCNN-runtime test, failing at preflight
before scheduling. No runtime package was installed or substituted. All12 other
built-in lifecycle tests and all11 external lifecycle tests passed in those runs.

Added the standard-library executor behind the existing Runs owner, per-device
admission budgets, CPU affinity inherited from each executor thread, current
host/cgroup memory observation and optional captured recipe resource requests.
No numerical worker, model weight, physical threshold or database schema changed.
Unknown custody fences dispatch; executor futures retain ownership through drain.

Tests-03:5 capacity,3 concurrency,7 progress tests pass,46.689s total. GPU device
classification in the concurrency test is explicitly simulated; CPU custodians,
job contexts, cancellation and PostgreSQL are real. Tests-05:6 capacity,3
concurrency,9 recipe,3 bound-Torch and4 captured-CPU contract tests pass. Its CPU
concurrency test also observes disjoint actual custodian affinity. Final
source-frozen test selection is ongoing; original receipts are retained unchanged.

## Complete CPU workload benchmark

Used the current migrated fixed control package and native binary through the
existing rollout owner,16 consumed development scenes,15 native warm-up steps
and8 AI steps per scene. All worker processes use one CPU thread/core and fresh
model startup with complete installed-runtime verification. No training or unused
validation scene is involved. Raw outputs are under
`runs/pressure-throughput-feedback-v1/cpu-benchmark` in the artifact root.

Two workers:110.374s, conservative summed process-tree RSS peak1,786,155,008B.
Eight workers:71.309s,4,762,517,504B. Initial comparison failed because the
signature included each attempt's intentionally fresh request SHA. Inspection of
all frame differences found only reference/AI seconds and diagnostic request SHA;
all physical values and network-call sequences were exact. Retained the failed
raw receipt and a separate corrected comparison (`cpu-throughput-02-vs08.json`).
The corrected projection excludes precisely those metadata fields; a focused
regression proves physical pressure differences still fail the comparison.

Sixteen workers:53.366s,7,544,217,600B summed RSS;49,210,724,352B minimum
available RAM. All16 numerical frame projections and all128 AI call-count entries
match the two-worker baseline exactly. All workers/native children reaped.
One repetition per configuration so far; these are complete short-workload
throughput observations, not full74 regression timing or isolated inference
latency. Startup hashing slows individually with concurrency, but aggregate
completion improves. Reverse-order repetitions are in progress.

## Scheduler acceptance and repeated throughput

Two repetitions in order 2,8,16,8,2,16 gave median complete times
112.2521252905s /67.3790247255s /54.3725686680s. Sixteen independent
one-thread workers are 2.064499214x faster than two on this shortened workload.
All physical frame values and call sequences match exactly. See
reports/cpu-throughput-summary.json for every raw receipt hash, resource scope
and timing. This does not establish full74 speed or isolated inference latency.

Independent standards/spec reviews exposed a real startup fence race. Retained
the failing actual-child regression (scheduler-fence-red.json), then fixed both
startup checks and coordination of drain-failure fence writes with dispatch.
All13 builtin lifecycle tests pass (scheduler-tests-10.json); actual external
lifecycle11 and concurrent cancellation/shutdown3 also pass (tests-09). The
unavailable legacy TinyCNN numerical runtime is explicitly excluded, not passed.
Both reviews close with zero remaining findings. Tuldok commits are
5e18837e1b580b5b831118adb0b0dc8b1dbba4e7,
7fbc82d9abe80182cfb3186a6691649298e98eb7 and
a9e99d28fb88320323dbcbfb63ae8224b89043ff.

Actual-device qualification03 completed in270.023226304s with1.255602121s of
measured CPU-computation/CUDA-optimizer interval overlap, disjoint assigned
cores, one float32 CUDA update and exact fresh-CPU checkpoint restore. Peak CUDA
allocation767402496B. All three Runs completed/drained; executor joined. Attempts
01(relative path) and02(obsolete Project method) failed before any optimizer
update and remain retained. This is qualification, not a scientific fit or
model accuracy result. See reports/actual-device-overlap-03.json.

Low-band feedback source passes five meaningful CPU tests in0.646s (2.070s
process total), including an analytic odd-grid cosine, bounded weak-mode loss,
zero-weight loss/gradient parity, first/final derivatives and strict policy
validation. No extra network passes. See reports/band-feedback-tests-01.json.
CUDA qualification and full matched fits are pending.
