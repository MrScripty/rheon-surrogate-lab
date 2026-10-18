# Implementation and execution findings

1. Initial preparation used the numerical interpreter for Tuldok schema
   validation; that runtime lacks the extension schema validator dependency.
   Use the existing app environment for assembly; its architecture probe and
   numerical qualification still use Pumas. Retained under preparation-failed-01.
2. The new recipe output initially omitted required=true. Contract validation
   rejected it. Corrected before admission (preparation-failed-02).
3. A single 48 MiB trace exceeded Tuldok's 40 MiB input declaration bound during
   restore-recipe validation. Split into two 32 MiB declared artifacts containing
   3/2 selected updates. Retained under preparation-failed-03. Complete actual
   output remains bounded at 64 MiB, including ordinary artifacts.
4. First catalog attempt opened the Project snapshot, then recipes.inspect
   rejected the inherited provider's 32 MiB output envelope. No model/study/Run
   was created. Increase the captured provider output declaration to 64 MiB;
   retain acknowledged open-project mutation and the rejection under
   managed-rejected-01. Controller elapsed 10.000635 s; child reaped.
5. Run 7002ae0601464f85ad6878b13ed81a07 was admitted but exited in policy parsing,
   before training: captured portable_policy.py still enforced 32 MiB. Cessation
   is failed/drained/echild, return code 1; study closed by its existing owner.
   The exact captured-policy CLI reproduced the typed `resources` error in
   0.826 s. Increase only the captured output-budget cap to 64 MiB and add this
   policy preflight to qualification. The same CLI then passed. Historical
   policies are untouched. Controller elapsed 80.004956 s; child reaped. Source,
   terminal event and small custody evidence retained in reports/managed-failed-02.json.
6. Reusing the old controller path with a newly sealed Project encountered the
   durable request-key guard: Request key is bound to different inputs. No Run
   was admitted. Preserve that rejection and use an entirely fresh attempt-03
   namespace. Controller elapsed 10.000570 s; child reaped. No request journal
   was edited, deleted or silently replayed.
7. Report generation had a dictionary-construction syntax error caught by local
   AST parsing before execution. Corrected without changing captured training
   sources. All study Python sources subsequently parsed.

Preparation-only attempt durations were not measured; do not infer them from
mtime. Some early disposable numerical transparency tests passed while later
recipe validation was incomplete; those were never sufficient for admission.
The final qualifier checks captured policy, source identities, provider/recipe
budgets and numerical transparency together. Non-numerical budget declarations
are explicitly separate from unchanged model/training calculations.

8. Attempt-03 completed its CUDA child with exit 0, drained/echild, but managed
   publication rolled back. Tuldok's evidence decoder limits every declared
   artifact to profile.shard_bytes (8 MiB); the 26.7/17.8 MB trace files exceeded
   it. Catalog state is interrupted/go_intent; do not describe this as a completed
   published Run. Owner runs.progress verified custody and update 200; study
   closure is journaled. Controller elapsed 150.009626 s; child controller reaped.
   The actual evidence reader reproduced the precise bounded-file diagnostic.
   Five per-update compressed artifacts passed the same reader, at 4.1–5.6 MB
   each. Also raise model-evidence aggregate limit from 32 to 64 MiB consistently.
   No Tuldok code, database rows or historical Run state was edited. No scientific
   result from this unpublished attempt is used for model evaluation. The byte
   compression fixture inspects format/capacity only, not model accuracy.
   Fresh attempt-04 was admitted only after custody verification, study closure,
   full numerical transparency and declaration checks. All failures remain local
   evidence; no automatic replay of ambiguous mutations occurred.

The resource-policy negative fixture initially ran in the app interpreter and
could not import NumPy; that was not the target failure and was not counted.
The correct Pumas command reached the typed resources diagnostic. The evidence
reader fixture initially expected the message without its standard Extension
evidence prefix; it was corrected to match the complete message and rerun.
9. The first CPU analysis compared historical rows with extra grid labels against
   measurement-only rows, falsely flagging quality mismatch despite identical
   metric values. Compare explicit case/pass/residual keys and separately require
   geometry equality. Preserve the first analysis/source/log; rerun the bounded
   CPU analysis. No training rerun was needed for this reporting correction.
