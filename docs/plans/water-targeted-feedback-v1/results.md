# Targeted feedback results

Both 5,000-update CUDA fits are trained. Supplemental control completes managed
CPU restore/validation and standard CPU qualification; targeted CPU restore and
qualification remain pending. No new model is promoted on these interim results.

Supplemental control passes 128/128 familiar states with mean/max 6.390625/10
calls, versus retained control 9.4140625/12. Exactly eight calls pass 121/128.
It still needs 25 calls on the identical captured 16×23×25 solve, versus baseline
11 and preceding feedback 28. All 18 standard development trajectories pass.
Whole consumed/final comparisons remain pending.

Training intervals are 373.062 s (extra-example control) and 336.582 s (targeted).
Both retain 625 primary exposures per grid. Supplemental exposures are 625 per
reused grid for control and [833,833,833,834,834,833] for targeted's six grids.
This matches example count, not cell work or isolated runtime conditions.

Independent CPU preparation qualifies 32 late labels (worst independent operator
error 1.10620e-13), a one-solve native fixture and 16 larger varied water parents.
All 1,024 parent steps and 784 post-warm-up pressure states pass native paired
physics gates. Sixteen pressure/base timelines are imported/reopened in the same
Tuldok catalog, with explicit CPU spectral-teacher labels and full force/source
lineage. These are future inputs; neither current fit uses them. No velocity or
tracer export or neural inference success is claimed for that preparation.

See reports/methods.md, diagnosis-findings.md, configuration.json,
cpu-preparation.json, future-collection.json and reconciliations.json for details.
The ledger records actual methods, durations and both setup failures/recoveries.
