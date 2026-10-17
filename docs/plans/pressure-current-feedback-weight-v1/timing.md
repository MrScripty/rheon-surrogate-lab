# Serial CPU branch timing

One core22 and one episode at a time after broad pools drain. Two reversed-order repetitions per model, same15native warm-up plus8AIsteps. Physical arrays and call counts repeat exactly within each model.

One core and one episode at a time after broad work drains. ai_seconds is the current Rheon paired diagnostic AI-branch time, including its additional same-request native reference solve, request capture, learned correction, IPC and protocol logging. External sparse capture and spectral-oracle work are absent from timed episodes. Startup is separate. These branch ratios do not measure standalone AI versus native solver or model inference speed; do not subtract the independently evolving reference_seconds as an estimate of the diagnostic solve.

| Cubic grid | Retained current0.1 ms | New current0.01 ms | New current0.03 ms | Native branch ms (retained/0.01/0.03) | Mean NN calls (retained/0.01/0.03) |
|---|---:|---:|---:|---:|---:|
| 16³ | 66.677 | 66.544 | 73.245 | 14.623 / 14.042 / 14.840 | 6.000 / 6.500 / 7.000 |
| 24³ | 349.449 | 404.154 | 389.397 | 63.031 / 62.751 / 65.408 | 7.375 / 8.625 / 8.000 |
| 32³ | 600.554 | 604.868 | 669.568 | 187.730 / 187.775 / 187.858 | 8.500 / 9.000 / 10.000 |
| 40³ | 1508.072 | 1568.307 | 1535.893 | 447.501 / 440.949 / 445.611 | 7.500 / 8.000 / 8.000 |

The table reports paired diagnostic branch cost, including the extra same-request native diagnostic solve, capture, correction, IPC and logging. It is not standalone model inference or solver speed. Do not subtract native branch time as an estimate. Two repetitions under desktop/cache variation do not establish a scaling law.

Raw frame/callback hashes, model startup, full child/campaign clocks, memory and drainage remain in [serial timing](reports/serial-timing.json).
