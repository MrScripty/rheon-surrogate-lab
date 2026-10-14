# Frozen GroupNorm native development diagnostic

This artifact-only adapter consumes the accepted final200 cached predictions and
unchanged qualified native6 operator. It never imports a model/trainer, computes
new predictions, updates weights, advances state, opens Tuldok access or reads
reserved final arrays. The local result is development evidence only.

Scientific authority is the reviewed [PR7 protocol](../../../docs/plans/global-pressure-groupnorm-native-dev-v1/protocol.md).
The [execution plan](../../../docs/plans/global-pressure-groupnorm-native-dev-run-v1/plan.md)
owns current authorization, exact source/input freeze, review and measured results.
Original evaluators and negative results remain unchanged.

`admission.py` owns typed GN train/restore artifact/custody and raw/packet/operator
lineage. `process.py` owns one Linux child session, bounded IO, observed RSS and
complete descendant drainage. `evaluate.py` owns fixed-order orchestration, the
whole-workflow watchdog and bounded durable publication. The new modules are
compiled from their verified source bytes; stale Python bytecode is not authority.

The CLI supports only `verify` and `evaluate`. Both require explicit artifact
roots, the reviewed source-freeze file and its SHA256. `verify` reads bytes only;
`evaluate` performs exactly two identity calls and the fixed32 serial teacher pass
through native6, which evaluates all declared raw/corrected/baseline policies.
No population/checkpoint/budget selection or retry option exists. A prior attempt
or an output directory is never permission to repeat a scientific run.

```
/workspace/task-torch/bin/python experiments/global-pressure-v1/groupnorm-native-dev-v1/evaluate.py verify \
  --training-artifacts /workspace/rheon-surrogate-lab/runs/global-pressure-groupnorm-v1/training-stage/artifacts \
  --restored-artifacts /workspace/rheon-surrogate-lab/runs/global-pressure-groupnorm-v1/restore-stage/artifacts \
  --capture /workspace/rheon-surrogate-lab/runs/global-pressure-v1-capture/committed-development-complete \
  --packet /workspace/rheon-surrogate-lab/runs/global-pressure-v1-training/development-packet2 \
  --binary /tmp/global-pressure-training-native-build6/rheon-global-pressure-training \
  --freeze /workspace/rheon-surrogate-lab/docs/plans/global-pressure-groupnorm-native-dev-run-v1/reports/source-freeze.json \
  --freeze-sha256 <independently-reviewed-exact-SHA256> \
  --output /tmp/<fresh-owned-verification-directory>
```

The separately admitted action uses `evaluate` with its predeclared fresh output
and the same freeze. Current-machine bulk arrays are ignored archives, not public
Git content. Another machine must obtain the already declared licensed archive
and native build closure; it must report unavailable rather than regenerate,
retrain, renew credentials or choose another binary.

Outputs occupy one new private direct child of `/tmp`: input proof <=1MiB,
population <=64KiB, durable cases journal <=8MiB, final report <=16MiB and failure
<=1MiB, <=32MiB aggregate. Final-report pending/incomplete names belong to the same
16MiB family. Only `evaluation.json` after successful command completion is an
accepted diagnostic; a failed final publication revokes that name and retains
`evaluation.incomplete.json`. Private staging has one writer, no concurrent path
mutation and no consumers before terminal completion. Existing output paths and
symlink aliases are rejected before admission; final publication never overwrites.

The600s watchdog starts at module `STARTED`, after Python interpreter/stdlib
bootstrap. It covers freeze/source admission, module imports, input checks,
identity/case children, drainage and artifact publication. Work reserves1s for
failure publication; each child owns its smaller timeout/drain reserve. Deadline
and observed4GiB process high-water are fail-stop policies, not OS containment.
The final artifact records elapsed time before final publication; terminal CLI
summary records time through publication and final-report publication time.
Interpreter/stdlib bootstrap and terminal stdout delivery are outside that clock.
A complete candidate-specific enclosing solver timer is unavailable.

Report every admitted case, actual quality/failures, known completed iterations,
unknown breakdown counts, wasted warm work, fresh p_base reset and all available
native/process times. Earlier cold-restore inference is a separate historical
ledger; its sum with new stages is not a paired end-to-end speedup. PCG-final
pressure/velocity fields, rollout, final generalization and GPU results remain
unavailable under the preserved wire.

Run fixture-only tests with the one-thread environment above:

```
/workspace/task-torch/bin/python -m unittest discover \
  -s experiments/global-pressure-v1/groupnorm-native-dev-v1 -p 'test_*.py' -v
```

Those fixtures test admission, policy interpretation and process/publication
failure; they neither train nor invoke the scientific native executable.
