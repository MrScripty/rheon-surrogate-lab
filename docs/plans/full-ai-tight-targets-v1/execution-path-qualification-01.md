# Execution path qualification 01

All checks below are zero-update component controls, using consumed development scenes at states 1 and 121. They do not predict future states or establish complete-generator timing/accuracy. Each timing population uses two warmups and five serial timed repeats, one CPU thread/core23.

| Execution control at 32 cubed | K | Median block ms | Maximum block ms | Maximum ms/output |
| --- | --- | --- | --- | --- |
| Double transform, estimate plans | 1 | 0.6551 | 0.7346 | 0.7346 |
| Double transform, estimate plans | 2 | 1.2321 | 1.3892 | 0.6946 |
| Double transform, estimate plans | 3 | 1.8185 | 2.0914 | 0.6971 |
| Double transform, estimate plans | 4 | 2.3992 | 3.0033 | 0.7508 |
| Double transform, bounded measured plans | 1 | 0.5293 | 0.6105 | 0.6105 |
| Double transform, bounded measured plans | 2 | 0.9713 | 1.1268 | 0.5634 |
| Double transform, bounded measured plans | 3 | 1.4211 | 1.6777 | 0.5592 |
| Double transform, bounded measured plans | 4 | 1.8725 | 2.4756 | 0.6189 |
| Compact Torch codec + factorized known forcing | 1 | 1.2372 | 1.7806 | 1.7806 |
| Compact Torch codec + factorized known forcing | 2 | 1.7306 | 1.7808 | 0.8904 |
| Compact Torch codec + factorized known forcing | 3 | 2.2582 | 2.7217 | 0.9072 |
| Compact Torch codec + factorized known forcing | 4 | 2.7922 | 2.9867 | 0.7467 |
| Native compact codec + factorized known forcing | 1 | 0.7999 | 1.2384 | 1.2384 |
| Native compact codec + factorized known forcing | 2 | 1.2479 | 1.6277 | 0.8139 |
| Native compact codec + factorized known forcing | 3 | 1.7474 | 1.9635 | 0.6545 |
| Native compact codec + factorized known forcing | 4 | 2.1895 | 2.2593 | 0.5648 |

The transform controls include finite/alignment guards, full transform/multiplier/inverse and pressure materialization. Their coefficients are identity or random diagnostics, not fitted predictors. Nontrivial multipliers agree with the double Torch implementation to less than 1e-11; native identity pressure residual is below 1e-8. All plan owners close after the finite job. The measured-plan policy allows approximately 0.05 seconds per plan and reports actual setup separately.

The wire controls include all current-state encoding, prescribed forcing for K horizons and complete identity-state pressure/velocity/tracer reconstructions. The compact layout omits only the known zero upper normal face of each staggered velocity. It averages no interior faces. The native codec preserves double pressure; carrier round-trip error is at most 2.31e-8 in these cases. Factorized Gaussian forcing matches the existing float32 forcing bitwise on this test population, including its diagnostic divergence. These properties are measured on declared cases, not universal bitwise equivalence.

The component timings have different scopes; do not add their medians/maxima as a measured complete model result. The current wire alone still exceeds 1.2 ms in a one-step observation, before neural work. It therefore does not justify a full-generator speed pass or an expensive full-generator training campaign yet. Further fusion and forcing work remain required. A separately bounded pressure-component capacity fit can answer the frozen-pressure accuracy question while full-model execution is redesigned.

The [pressure component contract](modal-pressure-component-protocol-01.md) uses one fixed 800-update managed CUDA fit on all 16 training scenes (144 native double pressure/RHS pairs), followed by exact typed double CPU restoration. Actual CPU/CUDA constructor and objective controls passed with zero updates; typed save/load and physical decoder checks passed. The component uses free learned modal gains, with no analytic inverse initialization or runtime numerical solve. Finite geometry tables do not establish arbitrary-shape generalization.

Primary backend API rules: [advanced real transforms](https://www.fftw.org/doc/Advanced-Real_002dto_002dreal-Transforms.html), [normalization and transform kinds](https://www.fftw.org/doc/Real_002dto_002dReal-Transform-Kinds.html), [new-array alignment](https://www.fftw.org/doc/New_002darray-Execute-Functions.html), [planner flags and time limits](https://www.fftw.org/doc/Planner-Flags.html). The loaded serial FFTW binary and version are recorded and hashed in the raw qualification; no FFTW thread API is initialized.

Raw results:
- [fftw-pressure-qualification-01.json](../../../runs/full-ai-tight-targets-v1/fftw-pressure-qualification-01.json) — SHA-256 `1933bb2fbbc8dcc84ec0b793cc5cbffa6df2d26686961fc30aafbc118176791e`.
- [measured-fftw-qualification-01.json](../../../runs/full-ai-tight-targets-v1/measured-fftw-qualification-01.json) — SHA-256 `306089d905b1f0457621dc7c4128bf86a5e1bd57c33ffd73c9aa2ecf4776f905`.
- [compact-io-qualification-01.json](../../../runs/full-ai-tight-targets-v1/compact-io-qualification-01.json) — SHA-256 `4331e95cfcf8ae6f26a6a80bdaae628883a88ca936882159921163d663f3d10d`.
- [native-compact-io-qualification-01.json](../../../runs/full-ai-tight-targets-v1/native-compact-io-qualification-01.json) — SHA-256 `1d774d04217e75ed6caa38bf7bc212c85f3ba1663d12e39c5c18a12852283b39`.
