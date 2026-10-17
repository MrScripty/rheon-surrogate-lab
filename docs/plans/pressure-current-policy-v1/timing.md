Two serial repetitions per model ran on core 22 after both broad CPU pools drained.
Each episode uses 15 native warm-up steps followed by eight AI steps. Paired
diagnostic branch time includes an additional native solve on the AI state for
its same-request reference, request capture, learned correction, IPC and protocol
logging. Startup is separate. External sparse capture and spectral-oracle work
are absent from timed episodes. The second repetition
matches the first physical frames and network call sequences exactly.

Each row is a consumed scene in a one-metre cube with cubic cells. Values are the
mean of the two per-repetition means. Models are matched within each scene; four
scenes do not establish an isolated resolution-scaling law.

| Grid | Retained 0.03 diagnostic ms/step | Early diagnostic ms/step | Late diagnostic ms/step | Native ms (retained/early/late) | Diagnostic/native ratios |
|---|---:|---:|---:|---:|---:|
| 16³ | 72.720 | 65.035 | 63.717 | 14.471 / 14.720 / 14.481 | 5.03 / 4.42 / 4.40 |
| 24³ | 380.225 | 403.775 | 359.296 | 63.201 / 65.902 / 64.739 | 6.02 / 6.13 / 5.55 |
| 32³ | 600.317 | 596.401 | 585.807 | 184.171 / 186.549 / 183.723 | 3.26 / 3.20 / 3.19 |
| 40³ | 1535.753 | 1547.050 | 1479.355 | 431.104 / 438.042 / 441.264 | 3.56 / 3.53 / 3.35 |

The six complete batches took 72.335, 72.381, 71.233, 71.280, 72.387, 71.686 seconds.
Raw frame/callback, campaign/config/source hashes, startup, memory and drainage
are retained in [serial timing](reports/serial-timing.json). The diagnostic AI branch costs more than the independent native branch in these
episodes. These values do not establish standalone AI inference or solver speed;
the independently evolving native branch is not a valid subtraction estimate for
the extra same-request solve. Two repetitions under uncontrolled desktop
and cache conditions do not establish a scaling law or isolated solver speedup.
