# Coarse gradient diagnostic

Status: completed diagnostic; norm-reduction overflow reproduced, repair arithmetic qualified separately.
Owner: root. Base 765e79815dc8912bb159691013a3b0c8b4b22c02; local branch
experiment/coarse-gradient-diagnostic-20261017. User continuation adopts the
completed coarse experiment's next-proposal.md. One managed CUDA replay, capped
at 1500 optimizer updates, no automatic full replacement fit or promotion.

Preserve the exact original 5000-update configuration, including schedule and
all full-length sampling arrays. Shortening updates changes balanced-bank RNG
draw positions, so the diagnostic cap is separate and stops before update 1501.
Bind all five stream order hashes and initial tensors to the retained original
captures/control history before executing the first optimizer update. Preserve
all original numerical/model/loss/physical owners and guards. Added detached
observations do not alter gradients, RNG, parameters, optimizer or decisions.

Write set: experiments/pressure-coarse-gradient-v1, this plan/reports, and one
explicit bounded diagnostic-entry option in the existing shared assembler.
No Tuldok/Rheon changes or historical rewrites. Model float32 CUDA, original
physical tensors float64, no CPU optimizer or holdout access. Selected Pumas
runtime and original Tuldok catalog own execution; original UIs stay open.

The diagnostic entrypoint is intentionally incomplete training. If the original
finite-gradient guard fires, retain its original exception/nonzero Run outcome.
If the cap is reached, raise a separately labelled diagnostic-stop exception;
that nonzero diagnostic Run does not mean a numerical model failure. Neither
outcome publishes a completed pressure checkpoint or starts CPU restore.
Original journalled Study ownership handles submission, readback and closure.

At failure retain exact update, active sample identities, learning rates, each
correction's residual-input/coarse/local/combined norms, finite outputs and
losses, every parameter's gradient finiteness/max/norm and a wider diagnostic
total norm. Bounded structured logs retain only every 100th update plus failure,
not tensors or datasets. Timing overhead is explicit; this is not a training
throughput benchmark. Separate optimizer work, Run/queue/readback and local
preparation clocks; preserve failed/uncertain admissions without replay.

First verify the observation seam preserves actual update results and RNG on
consumed small fields, and test the known finite 3–4 norm-overflow case plus actual
nonfinite components. Then freeze sources, verify capture/stream closure, and
run the bounded original trajectory. The original replay can take minutes until
its failure is located; a fast minimization comes afterward at the clipping
seam if captured values support it. Do not call a synthetic norm toy proof of
the actual candidate cause. Rank falsifiable explanations only after observing
the real failure, and qualify any fix separately before a full fit.

Fresh Standards MCP snapshot:v1:2d323269-2d90-4dd1-a7d0-3785f26b0bd7 selects 22
policies, no unresolved facts; all texts match the prior accepted 22-policy route.
The diagnosing-bugs skill guides the replay/measurement loop. Diagnostic files
are explicitly retained as experiment evidence; no instrumentation is installed
into the shared production trainer or inference package.

Admission review tightened the controller boundary: require a qualified captured
gradient entrypoint, unique explicit cap 1..1500 and bound reference plus matching
capture-verification/observer receipts before any catalog mutation. One Study
attempt remains; its conservative 3600 s wall budget includes Tuldok overhead
(actual prior 1800 s recipe reserved 1937 s), not a second training attempt. The
recipe's 1800 s execution deadline and separate 1500-update cap are unchanged.
