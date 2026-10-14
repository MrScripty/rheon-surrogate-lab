# Execution ledger

2026-10-14: verified public repo via authenticated integration, cloned clean main,
read standards and inspected unchanged pressure/trajectory sources and receipts.
Fetched clean stale Tuldok refs without modifying its working branch. Created
feature branch and scoped plan before implementation. Isolated Rust1.92.0 and
CPU Torch2.8.0+cpu/NumPy2.3.5 provisioned; no GPU execution.

Project source and generic contract fixtures implemented. Supporting evidence:
20Rust tests pass;24Python tests run,23pass/1optional operator-export skip;
root/newprobe fmt and Clippy(-Dwarnings) pass; JS syntax passes. Actual sealed
native READY/frame/STOP succeeds. Original serial-test prerequisite built against
a clean task-created detached Rheon2209012b checkout. Original Python suite had
an initial missing-binary obstruction, then passed after that prerequisite.
Managed runtime inventory still under qualification; local results are pending.

Paired CLI contract read at TuldokPR58/cdbe5136. Added a separate captured-data
CNN diagnostic: original NumPy Adam, persisted family/seed splits, training-only
normalization, fresh restore and checkpoint/prediction identity. Ten focused
lab/generic-reader tests pass. This does not claim modal fitting is neural
optimization or repair the original CNN's global-pressure quality limit.
Added dedicated actual-CPU-viewer browser procedure; execution pending fields.

2026-10-14 lab evidence closure: see reports/verification.md and results.json.
Full original-model CPU comparison qualified4320timed frames;6.894x/5.482x; cold
velocity9.0048e-8/tracer5.9089e-8;480restored pressures exact. Clean CPU training
regenerated historical NPZ bytes and restored/refitted state exactly. Separate
CNN200Adam/fresh restore passes with expected native0/16negative quality. Actual
existing CPU viewer passes. Final Python30tests=29pass/1optional skip;20Rust pass;
fmt/Clippy/JS supporting gates pass. Repaired viewer cap after exact observed
refusal; corrected retained-output review is distinct from managed publication.
Managed Tuldok acceptance remains pending its paired owner, with input/source
contracts in draftPR1. No Tuldok code or final release tags changed.

Review/CI continuation: parent instructed no retry of the denied cross-repository
comment and will coordinate internally. CodeRabbit's successful status was a
skipped draft review, not completed independent review. No Actions workflow
previously existed. Source review repaired verified-manifest reread drift and
receipt publication overwrites; isolated regressions and actual existing-data
repackaging pass. Python32run=31pass/1optional skip. Added fresh regeneration
commands and CPU supporting CI with verified checkout action pin, fixed Rheon/
Tuldok source pins, native gates and separate real Adam/fresh restore. CI remains
pending an actual run; numerical source/checkpoint/architecture is unchanged.

Actual GitHub Actions38531223801 at21cd8b0889050dacc8e9697e4104e1017f8f4727
completed successfully, including named native, Python and actual Adam/fresh
restore steps. Run/job/step API metadata is retained in reports/review-ci.json;
log archive download returned403, so no log-derived test counts are asserted.
A separate clean clone rebuilt/generate/verified/packaged/assembled and trained/
restored: tensor NPZ38cd3176...ebbd9cf reproduced; split hash changed with source
provenance; restored predictions exact; nativeCNN0/16negative reproduced.
Regeneration commands are in tuldok/REGENERATION.md. Independent CodeRabbit
review was explicitly requested on labPR1 and acknowledged in progress; no
review completion is inferred yet. Parent directed no cross-repository comment
retry and owns internal integration coordination. No larger model or tag.
