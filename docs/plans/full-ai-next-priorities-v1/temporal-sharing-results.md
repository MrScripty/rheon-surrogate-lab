# Endpoint and retained-feature full-state comparison

Both architectures run the spatial encoder once per block. The endpoint model feeds its predicted endpoint to a small intermediate decoder. The reuse model retains first-prediction features through cheap latent updates. Every intermediate pressure, velocity and tracer state is supervised and evaluated. All six managed fits and exact CPU restores are complete; controls verify the common data, sampling, objective, feedback curriculum and paired encoder initialization. Parameter counts and arithmetic work differ, so the screen is not FLOP matched.

Complete-block timings include physical encoding, prescribed forcing, all internal updates and reconstruction of every output. K2/3/4 budgets are below4/6/8ms with no separate internal-stage cap. Worst errors below include all scenes and every physical horizon.

| Model | Grids | Interval | Block ms | ms / step | Pressure | Equation | Velocity | Tracer | Combined pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| b2-endpoint | small | single_block | 2.949 | 1.475 | 1.63989 | 1.19849 | 0.999585 | 0.0189023 | False |
| b2-endpoint | small | own_history12 | 2.949 | 1.475 | 1.55082 | 0.693105 | 0.999585 | 0.00960183 | False |
| b2-endpoint | small | own_history48 | 2.949 | 1.475 | 1.5736 | 0.705704 | 0.999585 | 0.15467 | False |
| b2-endpoint | small | own_history240 | 2.949 | 1.475 | 1.83554 | 1.09106 | 0.999585 | 2.51743 | False |
| b2-endpoint | 32cubed | single_block | 39.328 | 19.664 | 1.17716 | 1.19864 | 0.998478 | 0.00495946 | False |
| b2-endpoint | 32cubed | own_history12 | 39.328 | 19.664 | 1.18379 | 1.02983 | 0.998543 | 0.0085964 | False |
| b2-endpoint | 32cubed | own_history48 | 39.328 | 19.664 | 1.18379 | 1.02983 | 0.998681 | 0.0401073 | False |
| b2-endpoint | 32cubed | own_history240 | 39.328 | 19.664 | 1.18379 | 1.02983 | 0.998681 | 0.671634 | False |
| b2-reuse | small | single_block | 2.864 | 1.432 | 1.74979 | 1.0992 | 0.985982 | 0.0186923 | False |
| b2-reuse | small | own_history12 | 2.864 | 1.432 | 1.54487 | 0.825517 | 0.985982 | 0.00776989 | False |
| b2-reuse | small | own_history48 | 2.864 | 1.432 | 1.56997 | 0.825517 | 0.985982 | 0.149353 | False |
| b2-reuse | small | own_history240 | 2.864 | 1.432 | 1.86403 | 1.19227 | 0.985982 | 2.43611 | False |
| b2-reuse | 32cubed | single_block | 38.815 | 19.407 | 1.25247 | 1.53961 | 0.979753 | 0.00481542 | False |
| b2-reuse | 32cubed | own_history12 | 38.815 | 19.407 | 1.1117 | 0.964474 | 0.979753 | 0.00722421 | False |
| b2-reuse | 32cubed | own_history48 | 38.815 | 19.407 | 1.14598 | 0.964474 | 0.979753 | 0.0363458 | False |
| b2-reuse | 32cubed | own_history240 | 38.815 | 19.407 | 1.22399 | 0.993221 | 0.979753 | 0.646141 | False |
| b3-endpoint | small | single_block | 3.113 | 1.038 | 1.91647 | 1.05028 | 0.999843 | 0.0294382 | False |
| b3-endpoint | small | own_history12 | 3.113 | 1.038 | 1.78892 | 0.991262 | 0.999899 | 0.0100248 | False |
| b3-endpoint | small | own_history48 | 3.113 | 1.038 | 1.80866 | 0.991262 | 0.999927 | 0.157278 | False |
| b3-endpoint | small | own_history240 | 3.113 | 1.038 | 2.10411 | 1.22132 | 0.999927 | 2.57936 | False |
| b3-endpoint | 32cubed | single_block | 41.113 | 13.704 | 1.18816 | 1.17379 | 0.999063 | 0.00814273 | False |
| b3-endpoint | 32cubed | own_history12 | 41.113 | 13.704 | 1.17929 | 0.960164 | 0.999442 | 0.00294605 | False |
| b3-endpoint | 32cubed | own_history48 | 41.113 | 13.704 | 1.18531 | 0.962289 | 0.999528 | 0.0272034 | False |
| b3-endpoint | 32cubed | own_history240 | 41.113 | 13.704 | 1.20638 | 1.06588 | 0.999528 | 0.640158 | False |
| b3-reuse | small | single_block | 3.202 | 1.067 | 1.89237 | 1.06678 | 0.997005 | 0.0296566 | False |
| b3-reuse | small | own_history12 | 3.202 | 1.067 | 1.76361 | 0.791708 | 0.997005 | 0.00848551 | False |
| b3-reuse | small | own_history48 | 3.202 | 1.067 | 1.7802 | 0.791708 | 0.997005 | 0.135345 | False |
| b3-reuse | small | own_history240 | 3.202 | 1.067 | 2.13534 | 1.17927 | 0.997005 | 2.24286 | False |
| b3-reuse | 32cubed | single_block | 42.175 | 14.058 | 1.10391 | 1.56478 | 0.99642 | 0.00637365 | False |
| b3-reuse | 32cubed | own_history12 | 42.175 | 14.058 | 1.07982 | 0.928322 | 0.996563 | 0.00683546 | False |
| b3-reuse | 32cubed | own_history48 | 42.175 | 14.058 | 1.07982 | 0.928322 | 0.996563 | 0.028982 | False |
| b3-reuse | 32cubed | own_history240 | 42.175 | 14.058 | 1.07982 | 0.986793 | 0.996563 | 0.587842 | False |
| b4-endpoint | small | single_block | 3.419 | 0.855 | 1.68954 | 1.20959 | 0.999854 | 0.0364071 | False |
| b4-endpoint | small | own_history12 | 3.419 | 0.855 | 1.66411 | 1.05823 | 0.999854 | 0.00892053 | False |
| b4-endpoint | small | own_history48 | 3.419 | 0.855 | 1.68398 | 1.05823 | 0.999854 | 0.139069 | False |
| b4-endpoint | small | own_history240 | 3.419 | 0.855 | 1.99052 | 1.2042 | 0.999854 | 2.22623 | False |
| b4-endpoint | 32cubed | single_block | 45.775 | 11.444 | 1.41446 | 1.36951 | 0.99901 | 0.00862035 | False |
| b4-endpoint | 32cubed | own_history12 | 45.775 | 11.444 | 1.3404 | 1.11403 | 0.999148 | 0.00536498 | False |
| b4-endpoint | 32cubed | own_history48 | 45.775 | 11.444 | 1.3404 | 1.11976 | 0.999282 | 0.023969 | False |
| b4-endpoint | 32cubed | own_history240 | 45.775 | 11.444 | 1.3404 | 1.11976 | 0.999282 | 0.598266 | False |
| b4-reuse | small | single_block | 3.645 | 0.911 | 1.75717 | 1.12788 | 0.996137 | 0.0381367 | False |
| b4-reuse | small | own_history12 | 3.645 | 0.911 | 1.71616 | 0.971207 | 0.996137 | 0.00757854 | False |
| b4-reuse | small | own_history48 | 3.645 | 0.911 | 1.7429 | 0.971207 | 0.996137 | 0.138327 | False |
| b4-reuse | small | own_history240 | 3.645 | 0.911 | 2.02908 | 1.18453 | 0.996137 | 2.27101 | False |
| b4-reuse | 32cubed | single_block | 49.559 | 12.390 | 1.2606 | 1.25997 | 0.996159 | 0.00960351 | False |
| b4-reuse | 32cubed | own_history12 | 49.559 | 12.390 | 1.24563 | 1.0132 | 0.996473 | 0.00580296 | False |
| b4-reuse | 32cubed | own_history48 | 49.559 | 12.390 | 1.24563 | 1.02545 | 0.996473 | 0.0252985 | False |
| b4-reuse | 32cubed | own_history240 | 49.559 | 12.390 | 1.24563 | 1.02545 | 0.996473 | 0.579848 | False |

No accuracy-qualified sharing finalist; use retained fitting/startup diagnostics before another bounded architecture/data change.
No reserved scenes were consumed. Single-block accuracy and240-step own-history are separate requirements; latency alone does not qualify a model. Cold setup, per-repeat spread, tensor-only timing, peak memory and sequential one-step controls are retained in the timing evidence.

[Results and hashes](temporal-sharing-results.json) · [Control audit](../../../runs/full-ai-next-priorities-v1/temporal-sharing-comparison-audit-01.json) · [Timing evidence](../../../runs/full-ai-next-priorities-v1/temporal-sharing-timing-01/summary.json)
