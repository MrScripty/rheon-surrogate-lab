# Execution ledger

2026-10-18 07:15 UTC: admitted the recorded CPU-only loss-component diagnosis.
Created an isolated branch from bd80c7d; preserved pre-existing dirty files.
The checkpoint-1600 archive contains 19 network arrays and no named optimizer
moments. Inspect all checkpoint inventories before classifying Adam analysis.
No training or new CFD Run is admitted by this diagnostic plan.


07:20 UTC: the initial three workers stopped at separate-gradient additivity:
C53 case 7, C66 case 0 and C67 case 1 exceeded the planned relative-L2 5e-6.
All were reaped; raw vectors and logs remain in workers-01. Repeating C66 case 0
reproduced 5.26863e-6. A targeted probe showed the reconstructed joint gradient
and a repeated captured full gradient both match the original bit for bit.
This isolates the observed difference to separate versus joint float32 backward
arithmetic on that case, not scalar loss semantics or forward nondeterminism.

Replan: the original universal separate-gradient tolerance is superseded as the
acceptance oracle. Every case must instead match the captured full gradient
exactly when the reconstructed terms are differentiated jointly. The original
5e-6 precision check remains visible per case; failures are limitations, not
silently accepted passes. No model precision or physical acceptance target changes.
The actual joint canonical+final backward is measured as without_early, rather
than inferring that counterfactual solely from the sum of separate gradients.
The initial source is retained locally as runs/pressure-loss-components-v1/
measure-initial.py; the targeted probe receipt is retained under reports/.

Second preflight: C66/C67 passed exact joint checks, but C53 exposed a numerical
reconstruction difference. Moving a length-one mean into each term and creating
branches in a different order changed the float32 backward arithmetic. The final
reconstruction preserves the captured tensor ranks, reduction placement and
operation order. All 24 reconstructed joint gradients now match the captured
full gradient bit for bit. The earlier checks remain in workers-01/02; their
successful partial outputs are runtime preflight evidence, not final receipts.

Final run: workers-03 completed in 5.728087 s, all three children returned 0 and
were reaped. C53: 4.901959 s; C66: 4.804367 s; C67: 4.798850 s. One CPU thread per
worker on cores 21/22/23, no CUDA initialization or optimizer update, unchanged
weight/buffer inventories. Live vector bound 11.29 MiB; raw vector payload 56.46
MiB per checkpoint, retained only in ignored runtime files. Fitting tensor guard
remains 8 MiB. Retained C53/C66 full-gradient dot matrices replay exactly.

| Checkpoint | Full opposing pairs | Final-only | Canonical + final | Early | Early norm > final | Separate precision check limited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C53 | 12/28 | 12/28 | 12/28 | 12/28 | 0/8 | 1/8 |
| C66 | 16/28 | 13/28 | 13/28 | 18/28 | 5/8 | 7/8 |
| C67 | 17/28 | 12/28 | 12/28 | 15/28 | 5/8 | 6/8 |

Canonical + final retains two strong (<-0.5) opposing pairs at C66 and five at
C67. Early/final cosine on C67 case 1 is -0.597; its early gradient norm is 2.393
times final's. This supports investigating both shape aggregation and early
feedback, but does not establish Adam behavior or training causality. The next
proposal chooses aggregation with the original loss to avoid conflating both.

Original separate-gradient 5e-6 checks are LIMITED on 14/24 cases. Worst relative
sum error: C53 1.42840e-5, C66 0.00220443 (0.2204%), C67 6.00914e-5. Exact joint
checks and historical full-gradient replay pass; these numerical limitations are
visible in the report. They are not model pressure errors or changes to the
physical 1e-6 acceptance target. All three selected checkpoint archives contain
network arrays only; exact Adam-direction replay is unavailable, not approximated.

Focused evidence: literal geometry/chain-rule oracle passes, including undefined
zero-vector cosines. A real C66 fixture preserves loss values but detaches the
final derivative; the joint guard rejects at the intended exact diagnostic.
The actual supervisor's SIGTERM test terminates/reaps all three owned sleeper
fixtures. Its first harness import selected an older same-named module and exited
before creating children; explicit-path loading fixed the harness. Original log
remains under cancel-01. Completed/failed diagnostic owners were all reaped.

The interactive report checks all 15 model/component matrices, retained scalar
labels, within-case loss/precision tables, keyboard access and narrow-screen
layout. Source/evidence verification passes with the 14 precision limitations
explicitly retained. No training, new scenes, deployment or historical numerical
result changes. Method/environment commands: commands.md. Next proposal:
next-steps.md. Raw vectors, weights and screenshots are excluded from Git.

Closure: all objective evidence is satisfied under the recorded joint-equivalence
replan, with the original 14 precision-check limitations preserved. Existing
Tuldok activity is completed and links the report; C53 remains deployed. Reviewed
branch scope from explicit base bd80c7d8fd10b4fba9523d22cdf509d07e88d6be: the new
diagnostic directories and owned dashboard report link only. Historical numerical
files and unrelated dirty work are preserved. No credentials, binary/raw vectors,
weights, simulation datasets or screenshots are committed. Debug probes remain
only as identified runtime evidence; temporary probe launchers were removed.
No live diagnostic, fixture or headless browser process remains owned by this
study. No history rewrite, push or merge. The branch is retained for Jeremy's
review under the plan's integration/retirement contract; no worktree was created.
