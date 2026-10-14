# Controlled-child procfs lifecycle race repair

Parent authorized this narrow PR5 update after the observed legacy CPU failure.
At headc1828de, job115859083081 failed in the process cleanup fixture: the child
/proc/5417/stat entry existed at the precheck and disappeared before read_text.
The trace was FileNotFoundError at scripts/test_tuldok_pressure.py:52, not a
physics assertion failure. The old code already accepted an absent entry, but
its existence/read pair had a reaping window. The prior failure remains linked
and its filtered log is preserved in ignored diagnosis evidence storage.

The fixture now performs one direct read. Missing entry (FileNotFoundError,
ENOENT) and missing process during read (ProcessLookupError, ESRCH) are the only
read exceptions accepted as stopped/reaped. Linux proc_single_show returns ESRCH
when the PID task no longer exists; Python maps the two errors to those specific
exception classes. See [Linuxv6.17 procfs source](https://github.com/torvalds/linux/blob/v6.17/fs/proc/base.c)
and [Python3.12 exceptions](https://docs.python.org/3.12/library/exceptions.html#ProcessLookupError).
The actual observed failure was ENOENT; ESRCH is the adjacent supported read-window
case, not an additional observed CI incident.

A successfully read state must remain zombieZ or deadX. A live sleepingS child
still fails the assertion. Permission errors, other I/O errors and malformed
snapshots still fail; no general OSError/Exception catch, retry or polling was
added. This is the controlled same-group Python child's fixture probe, not a new
arbitrary-descendant custody guarantee or general procfs parser. Production
process-group killing, timeout propagation and parent reaping are unchanged.

Three deterministic regressions inject disappearance directly at read with a
forbidden exists() call, distinguish terminal/live states, and ensure unexpected
errors/malformed data escape. The original real controlled-child timeout/nonzero-
exit test remains. Main scoped ConfigContracts run:13tests,0skips,2.031s,pass.
Independent scoped repair review and exact-head CI are recorded in the draft PR
and final handoff after publication; no blind rerun of the failed head is used.

Writes extend only to the parent-authorized legacy test file and new diagnosis
repair documentation. Scientific training/model/test sources, config, final model
weights, all original receipts/negative results, full/compact diagnosis evidence
and GroupNorm proposal remain unchanged. The GroupNorm candidate remains entirely
unimplemented/untrained. CI's existing historical tinyCNN contract is distinct
from the frozen-model diagnosis and any future candidate training.


Independent GPT-6.1 Sol High SOURCE/REGRESSION review is CLEAR. The reviewer
independently ran only13ConfigContracts in2.031s withzero skips, verified accepted
versus unexpected errors and the original real child-timeout/nonzero-exit checks,
and confirmed unchanged scientific/diagnosis/proposal bytes. No diagnostic or
scientific model training was rerun. New exact-head CI is required after this code
repair; historical failedc1828 is preserved rather than rerun unchanged untilgreen.
