# Water learning sequence results

All four declared CUDA runs completed 10,000 updates within their 600-second
limits. Fresh CPU restore and validation succeeded through the existing Tuldok
catalog. The unchanged float32 model still fails pressure qualification.
Increasing training parents from 256 to 1,024 reduced median validation equation
error by **15.5%**, while increasing the worst error by **9.9%**. The selected
1,024-parent candidate qualified **0/128** fresh reserved pressure proposals.

| Trial | Training simulations / states | Training elapsed seconds | Median validation relative residual | Worst validation relative residual | Qualified validation |
|---|---|---:|---:|---:|---:|
| One-case fit | 1 / 1 | 238.93 | 0.0716866 | 0.275699 | 0/128 |
| Longer original cohort | 256 / 512 | 377.37 | 0.0391099 | 0.197603 | 0/128 |
| Matched small cohort | 256 / 256 | 386.55 | 0.0456552 | 0.189139 | 0/128 |
| Matched large cohort | 1,024 / 1,024 | 384.14 | 0.0385793 | 0.207926 | 0/128 |

Every trial uses the original 64 validation simulations, both retained states.
The one-case model also fails its fitted training snapshot: relative residual
**0.00177824**, versus the unchanged **1e-6** model-output target. Lower training
loss therefore does not establish that even one pressure problem is solved.
All native reference checks passed. Previous native pressure was supplied for
every learned proposal; these failures are not cold-start testing errors.

The data-size pair holds architecture, loss, initialization/sampling policy,
batch size, schedule, update count and validation population fixed. The original
256 parents are nested within 1,024. One state per parent fits Tuldok's existing
64 MiB captured-input bound; original native captures retain both saved states.
The longer-original-cohort run uses two states per parent, so comparing it to
the large cohort changes two factors. Training-only normalization naturally
differs across populations; normalized loss across those populations is not an
absolute accuracy comparison. The table reports the same physical equation
metric for every candidate.

The longer original-cohort run reproduces the historical first 2,000-update
tensor inventory exactly. Its validation loss falls from the frozen historical
**0.00354147** at 2,000 updates to **0.00257260** at 10,000, approximately 27.4%.
The historical reserved-test median was 0.0436711 on different simulations;
that value cannot establish a controlled gain against the new reserved set.
Historical source, weights, captures and results are preserved.

Selection used validation only: qualified count, lower median residual, lower
worst residual, then the declared fixed tie order. `scale-1024` won and its final
update-10,000 weights were frozen before fresh reserved inference. No intermediate
checkpoint or reserved result was used to choose it. The preserved
`reports/selection-before-test.json` is that pre-test snapshot; its
`reserved_inference_started: false` records the state at selection time. The
completed test is recorded separately.

Fresh reserved test: 64 independent new water simulations, two snapshots each:

- Qualified raw AI proposals: **0/128**.
- Median relative residual: **0.0387903**; range **0.00444333–0.208442**.
- Native reference and native fallback solutions: **128/128 qualified**.
- Learned warm-start/refinement path used 16,553 native iterations, including
  4,096 wasted warm iterations and fallback on every case. Native temporal PCG
  used 12,457 iterations; zero-initialized PCG used 13,272.

Fallback success belongs to the native solver. These iteration counts do not
show AI solver acceleration. This study checks individual teacher-state pressure
proposals with preceding native pressure; it does not establish stable autonomous
AI trajectories. Each trial uses one seed, so the measured median difference is
a bounded experiment rather than a statistical confidence claim.

The selected weights are
`fac9f59be57f95d9112f623006fd93e1cf26eaaad63c553a86a571cf0ce0aac2`.
All optimization used the NVIDIA GeForce RTX 5090 Laptop GPU, float32 parameters,
batches, gradients and Adam moments, with AMP and TF32 disabled. CPU restores used
zero optimizer updates and never initialized CUDA. Restored weight hashes and
saved CPU witnesses matched; device arithmetic is not claimed bit-identical
between CUDA and CPU. Model/loss/physical gates remain unchanged.

All scientific stages are completed/drained in catalog
`a7da4a8de53541a2a0e2aad08005bb71`:

| Trial | CUDA training Run | CPU restore Run | CPU validation Run |
|---|---|---|---|
| One-case | bc1ea77166d444dc8f791cc532a49541 | 1fa2fa3fbf044a4795bc41c8b9845bc4 | 841260809ad2459e802c351ca00649c0 |
| Longer 256 | a63a03c360ab4dc993a4add962f86834 | cac7599102a54fc09b6ef1c48860b319 | 182782998e654ce3a2b5d81b3ce861fc |
| Matched 256 | a9c6b01204874a1e8d12fe863f216e8d | 702a1bfd6c1d490a982c8cba02048e2b | 42d9f60c1541467288b7cb10e445299f |
| Matched 1,024 | c9c0cc20ce4849f8af6b0c9bf841154f | 00b82369780c47cdae4db1b416eb5cdf | db16914bd906467a8a6bc623e2a26573 |

Fresh reserved CPU Run: `b51a71ef0ff94c4ca00b3102a8e4fdbc`. A separate two-update
CUDA smoke and fresh CPU restore qualified the new packet path before the four
scientific runs. Exact receipts, histories, validation summaries, selection,
fitted-case and fresh-test results are retained in `reports`; `acceptance.json`
records the complete measured summary and artifact hashes. Full artifacts remain
in the managed Runs and local `runs/water-learning-sequence-v1` directories.

The observation work is separate: [continuous simulation and tracer evidence](observation.md).
Rheon's viewer now pins an actual native timeline, with translucent water voxels,
the recorded force widget and explicit tracer playback. All 768 new training
simulations and both continuous preview sequences are visible in the same Tuldok
data collection. The existing AI-comparison deployment remains the earlier frozen
model; this experiment's selected candidate has been evaluated through its own
managed CPU path.
