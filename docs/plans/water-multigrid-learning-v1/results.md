# Multi-grid learning results

Status: verifying; the completed measurements below are retained, while the
second fit and real-cohort qualification still have pending managed/CPU stages.
Historical results and the current default remain preserved.

## Completed comparisons

The earlier checkpoint needs13.4375 evaluations and the preceding latest
checkpoint12.6015625 on128 saved development states under identical per-evaluation
stopping:6.22% less work from changed weights/training. Both pass128/128. The
larger15.81% improvement reported previously also included stopping granularity.

The new one-evaluation-trained multi-grid checkpoint passes128/128, with mean
13.46875 and maximum23 evaluations. The preceding latest mean/max are12.6015625/26.
It improves the tail but regresses mean work on these small saved states.

| Cube slices | Latest single-grid-trained weights | Multi-grid one-evaluation-trained weights |
| --- | ---: | ---: |
| 16³ | 10 | 12.75 |
| 24³ | 10.875 | 13.25 |
| 32³ | 17.875 | 13.5 |
| 40³ | 22.875 | 13.625 |

These are mean actual network evaluations on fresh23-step physical trajectories,
after15 native history steps. Both fixed-volume and fixed-spacing sweeps pass;
work counts happen to match across the two sweeps. Native controls qualify all
seven distinct scenes. Maximum allowed work remains32 evaluations per solve.
Concurrent execution makes these numerical/work observations, not latency claims.
The multi-grid weights reduce40-cubed work40.44%, with a27.5% regression at16-cubed.
No single candidate dominates every measured workload.

Multi-grid one-evaluation training uses5,000 real-frame draws plus5,000 analytical
draws, exactly1,000 at each grid, at44.3782 optimizer updates/s. Initialization,
normalization and real-frame sampling match the single-grid control exactly.
Training includes3.4125 times its voxel-evaluation proxy because auxiliary grids
are larger; equal update/example counts do not imply equal compute. Raw one-pass
output remains physically unqualified (fit0/1, development0/128). Adaptive
iteration passes the fitting frame in11 evaluations and development128/128.

The retained baseline and multi-grid candidate both pass all1,491 AI steps in
three512-step trajectories (8.5333s), after native warm-up. Quarter-strength
forcing keeps native controls valid; this does not extend the original full-force
claim. The trial record preserves source, CUDA execution, CPU restore, all
managed Run IDs/custody, native checks, mode probes, grid results and lifecycle.

## Prepared cohort and mechanism evidence

The existing next cohort contains16 distinct water-filled sealed domains and32
actual simulation snapshots. Counts, cubic spacing, forces and real history are
retained in reports/prepared-cohort-audit.json. Schema-v5 feedback uses the full
real training population's cell-weighted correction RMS2.853573464078259Pa;
validation statistics are excluded. Its640 analytical labels pass independent
operator checks (worst6.44257e-14 relative mismatch). These manufactured labels
are additional operator examples, not additional fluid simulations.

CPU profiles on16/24/32/40-cubed identify 3D convolution and tensor-fill cost.
Four threads help the sampled mechanism; eight regress relative to four on32³.
These profiles are distinct from whole-track timing. A one-AI-step fixture
qualifies matched native/AI four-core execution. No native-solver speedup or
production deployment is claimed from these observations.

See [methods](reports/methods.md), [ledger](execution-ledger.md),
[trial record](reports/trial-multigrid-single-pass.json), and the active [plan](plan.md).
