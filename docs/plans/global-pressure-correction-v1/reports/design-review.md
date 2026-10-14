# Independent design review

Reviewer: independent GPT-6.1 Sol, high reasoning, read-only delegated review.
Date:2026-10-14. Subject: new design documents/reservation on lab20dd90f6.
Disposition after repairs: **no remaining material blocker to D5 design acceptance**.
D6 publication is outside the review; all numerical/environment E gates stay pending.

## Reviewed scope and evidence limits

Reviewer read all ten initial design artifacts, inspected actual CNN train/encode/
Adam/packaging/restore sources, modal model, native trajectory probe, older sealed
warm-start patch and committed negative/global/analytic comparison receipts.
Standards: Core/planning/verification-oracles/architecture/contracts/performance at
188beda1fa477d21d576c233dd8c7f4c4c267d23. Checked merged Tuldok CPU runtime and
external-recipe contracts atf7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6.
The recheck inspected revised material and checked metadata/hashes.

No numerical test, array generation, training, source modification or publication
was performed by the reviewer. Document arithmetic and source inspection do not
prove native API support, reference physics, runtime/hardware, restore execution,
conservation, stability or performance. Full historical lineage separation is not
proved by112 unique reserved seeds.

## Complete review round and dispositions

| Finding | Observed impact | Repair and independent disposition |
| --- | --- | --- |
| R1/P1 full disk cap | Declared32³ canonical frames require23423877120raw bytes (~21.8GiB), exceeding12GiB before overhead |32GiB conditional cap plus container/working/transient/free-space admission; closed |
| R2/P2 full timing budget | Cold repetitions unspecified;403200resident method-frames excluded cold;4hour cap unqualified | Three resident+three cold,705600full/56448smoke method-frames,16hour conditional cap and measured train-only per-method projection; closed |
| R3/P2 full acceptance | Full10k localGPU procedure could not satisfy smoke200-update managedCPU E3 or managedGPU reading of E6 | Explicit FullE3/E4/E6 mapping, sameGPU restore/CPU parity and bounded managedCPU+externalGPU evidence; closed |

Reviewer also rechecked that immutable model/data identity corruption fails before
execution; only numerical inference failures use declared fallback. No further
blocker was observed in global context, original-b residual scaling, bounded
fallback, whole-history split/geometry/regime separation, train-only statistics,
actual trained-weight restore protocol, autonomous rollout/conservation accounting
or analytic/temporal controls.

## Reviewer-bound working-tree identities

These hashes bind the rechecked documents before later status/evidence-only
updates. Domain protocol/model/data/compute semantics were not changed after review.
Plan/criteria lifecycle projections and ledger/issues record actual review/publication
state separately; a hash of those earlier projections is not a final-commit claim.

```text
8a3d0048054693ad7ea73324645d25cca14af491088c9a7ad08127b1106e82b9  plan.md
670b4a5c9cf67ed78ef2a571ea3be55d933e7c08763f4ee348674568978f1290  inspection.md
5c32d6a3a623f5595e76ca910f4711886689fd23763918f9e88631c36018008c  research.md
933dbb8df52b19c011dfb1e5f0d07c5789938111703fc7011a3670d45264ba00  data-and-model.md
32390c055b81284e33b0f48b346f5fafd40e6cc1ece9a6372c8f2df5760ffb59  acceptance.md
283ec5d0e800dda731db17b6061ed48f13b40fb570c4694c4ac86bc811678cdf  execution-and-handoff.md
5f20231b7a7ef2fddf6edee0b894de2dabf7940278d6e42df2479baf54ba7364  local-gpu.md
1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5  reservation.json
891af67f339dfe492de73c8e80ae8fe6e2bebccdf25d916978537b92a720db8e  execution-ledger.md
ef67278f5187fc41d68e2be824dba5f1ce1156794ec857f770ae699f6ccb212a  issues.md
```

The main agent checked these hashes against actual working-tree bytes before
changing only lifecycle/evidence projections. Detailed repairs are in the ledger.
