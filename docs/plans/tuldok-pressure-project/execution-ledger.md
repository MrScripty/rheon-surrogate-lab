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
