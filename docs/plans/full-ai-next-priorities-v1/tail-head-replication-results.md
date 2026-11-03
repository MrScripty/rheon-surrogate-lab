# Learned final block: three continuation seeds

All three 10,000-update managed CUDA fits, exact independent CPU restores and consumed physical evaluations are complete. Every seed passes 7e-4 on all tested small-grid requests. These are calibrations from one fixed parent, not independent pretraining runs.

| Seed | Historical worst | Initial bank worst | Recent bank worst |
| --- | ---: | ---: | ---: |
| a-tail8 | 0.0002255063 | 0.0004745284 | 0.0005561435 |
| a-tail8-s2 | 0.0002255128 | 0.0004745222 | 0.0005561006 |
| a-tail8-s3 | 0.0002255153 | 0.0004745301 | 0.0005561372 |

All 94,956 parent parameters remain frozen; 33,982 final-head parameters are fitted. All seven declared model/diagnostic artifacts restore bitwise. Precise execution qualification for both continuation seeds passes all 2,176 requests. The 2.075 ms timing belongs to the primary only; each frozen model will receive its own five-repeat timing on the new bank.

The primary remains `a-tail8`. New scenes are reserved before capture and are disjoint from previous pressure and all temporal scenes. The [new independent confirmation](pressure-confirmation-v2/results.md) passes the prepared-pressure target for all three seeds. Native acceptance remains unmet, and no application model is promoted.

[Run identities, parity and error evidence](tail-head-replication-results.json) · [Primary precise execution result](precise-routed-pressure-results.md)
