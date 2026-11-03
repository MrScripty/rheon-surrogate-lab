# Materialized learned pressure execution: bounded investigation

The frozen C107 spatial path is bias free and linear. A small-grid execution
variant materializes its learned CNN coefficients from unit-vector responses,
then uses a matrix product for each fine correction. It does not assemble or
invert a pressure operator and receives no reference pressure or RHS labels
during construction. The learned global blocks and acceptance gates remain.

The raw CNN pilot shows lower warm cost on three geometries, but the first
complete historical pressure qualification failed its registered numerical
comparison. This variant is unqualified and is not a candidate frontier point.
The completed 128-request diagnostic finds maximum pressure relative difference
1.72e-7 and two failures of the tighter equation-error comparison. All physical
target classifications remain unchanged. Worst equation error changes from
2.25506e-4 to 2.25570e-4. This is a failed numerical-equivalence gate, not a
failed 7e-4 physical target, and it remains excluded from candidate frontiers.

Float64 matrix construction fixes those historical failures in the pilot, but
float32 matrix storage subsequently fails a fresh-bank comparison. That attempt
also remains unqualified. The next registered version retains float64 learned
coefficients and dot products for grids with at most 2,048 voxels, while larger
small grids use the already qualified compiled CNN. The qualification keeps
the original pressure and equation comparison tolerances; no speed or new
accuracy claim follows before that full qualification and balanced timing pass.

Dense matrices scale quadratically in voxel count. This experiment is limited
to grids with at most 16 cells per axis; larger grids keep the unchanged parent
path. A 16³ matrix alone uses 64 MiB. The 64 tested small geometries require
approximately 389 MiB of float32 coefficients. The precise routed alternative
uses float64 matrices for 58 geometries, totaling approximately 419 MiB, and
retains the CNN for the other six. Matrix construction and tracing are cold
costs and must be reported separately from warm inference. The cache has a
512 MiB coefficient limit and uses least recently used eviction.

[Registration](materialized-pressure-execution-registration.json) ·
[Raw pilot](../../../runs/full-ai-next-priorities-v1/materialized-fine-pilot-01.json) ·
[Failed qualification record](../../../runs/full-ai-next-priorities-v1/materialized-pressure-failed-qualification-01/failure.json)

[Historical numeric diagnosis](../../../runs/full-ai-next-priorities-v1/materialized-pressure-diagnosis-01.json) ·
[Failed mixed-precision record](../../../runs/full-ai-next-priorities-v1/mixed-materialized-pressure-failed-qualification-01/failure.json) ·
[Precise routed registration](precise-routed-pressure-execution-registration.json)

The precise routed version passes the full 2,176-request qualification. Maximum pressure relative difference is 5.2919e-8 and maximum equation-error difference is 3.2154e-8, both inside the unchanged tolerances. All target classifications are preserved. Balanced CPU timing and a new independent confirmation are complete: the fixed primary reaches 2.133 ms median at 4.6963e-4 worst residual on the new small-grid bank. See [the independent result](pressure-confirmation-v2/results.md) for per-grid, cold and native limits.
