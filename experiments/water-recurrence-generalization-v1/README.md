# Recurrence and generalization

This experiment varies training recurrence and initialization/sampling seeds while
reusing the existing cohort trainer, analytical bank, Tuldok execution and CPU
qualification tools. The [plan](../../docs/plans/water-recurrence-generalization-v1/plan.md)
owns admission; `configs/sequence.json` names inputs, trials and selection policy.
Configurations identify material by paths rather than substituting historical
hashes into executable code. Managed captures retain their exact source witness.

Use `water-conditioning-v1/training/pipeline.py assemble` with the sequence's
training, validation, reserved and runtime paths and the chosen trial config.
Commit source before assembly. Execute the captured train/restore/validation
recipes through the existing authenticated Tuldok workflow. Do not run an
optimizer in the controller or create another catalog.

For every completed checkpoint, use `water-multigrid-learning-v1/evaluate.py`
on the training and validation packets with default adaptive32 policy, then on
validation with `--network-budget 8 --fixed-budget`. Use `mode_probe.py` and
`study.py` from that same experiment for signed operator probes and physical
trajectories. Record unsuccessful physical outcomes as results, not missing data.

Fresh development, final holdout and512-step studies declare cubic spacing and
force parameters explicitly. Final holdouts run only after the development-based
selection is frozen. Native controls must qualify before judging the AI track.
Different recurrence depths have different compute cost at equal update counts.
Concurrent observations measure correctness/throughput; isolated timings require
their own resource-controlled experiment.

Full outputs stay under ignored `runs/water-recurrence-generalization-v1`.
Small receipts, commands, comparisons and decisions are committed in the plan's
reports directory. Historical results and the application deployment are separate.
