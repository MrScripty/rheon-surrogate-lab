# Issues

No open blockers. Owner: this study. Local Euclidean gradient alignment is not
Adam's update direction or proof of causal training interference. Exact saved
optimizer state is required for that optional analysis; no substitute optimizer
state may be invented. Revisit after checkpoint inventory inspection.


LC-1 (measurement limitation, owner: diagnostic study): separate float32 backwards
are not bitwise linear; the initial 5e-6 summation check failed. Fix-now by checking
captured versus reconstructed joint backward exactly on every case and retaining
the original separate-gradient precision verdict. Do not present that verdict as
universally passed or infer Adam/causal training behavior. Trigger for further
work: joint equivalence or historical full-gradient replay fails, or a proposed
conclusion depends on tiny alignment differences comparable to this discrepancy.


LC-2 (optional analysis unavailable, owner: future training study): all three
selected checkpoint archives match only their network inventories; Adam moments
were not retained. Disposition: report unavailable, do not reconstruct fictitious
history. Revisit only with an explicitly retained and verified optimizer state.

LC-1 closure: captured-order reconstruction passes exact joint equivalence on
24/24 cases. The original separate-gradient check remains limited on 14/24 cases,
worst 0.2204% of full-gradient L2. The report exposes those values. This is a
measurement precision limitation, not a changed pressure target. No open execution
blocker remains; all success, failure and cancellation test workers were reaped.
