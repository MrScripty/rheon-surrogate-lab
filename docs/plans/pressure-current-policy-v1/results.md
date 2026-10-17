# Current-policy feedback results

Active experiment; the matched late-state arm and serial timing are pending.
The early-state arm completed measurement and is ineligible under unchanged
population gates. No model promoted; eight unused scenes remain untouched.

The frozen retained0.03 network generated early/prefix8 paired correction states
on32recorded training frames from16parents plus112manufactured states across14
admitted grids. These are144pressure examples, not144newCFDsimulations. Independent
labels/neighbor-stencil identities qualified all144pairs. Native late correction
RMS median9.37004e-8Pa; directions differ from the older scaled bank. Rescaling
the same RHS changes predictions by median4.2754e-7relative, so small absolute
amplitude alone does not explain the observed late error. This is spatial
analysis and does not establish temporal aliasing or a causal noise floor.

The two configurations add the same train-only0.1-weight stream, vary only selected
early/late payload, and retain lowband0.03, all original streams, initialization,
seeds and5000-update schedule. Actual early optimizer work:50,000networkforwards,
425,703,026forwardcells. The old0.03 fit did not record this new counter; equal
updates alone is not equal compute. Inference architecture remains370,016parameters,
1,485,452weightbytes; the train-only stream adds no inference input or layer.

Early CUDA optimizer504.952s, complete trainer508.683s, complete managed Study
including admission/queue/CPUrestore/readback1480.719s. Five CPU artifact pairs
match exactly. First-frame physical qualification passed before all74episodes.
The16worker campaign took742.814s, peak summed worker/child RSS11.66GB and
minimum available host memory49.40GB; all children reaped. Concurrent campaign
clocks are not isolated inference latency. All8523AIsteps and233independent
pressure checks passed, with all pressure-error bounds covering their oracles.
Maximum independent oracle defect1.667e-13; maximum same-RHS mean-free pressure
relative error7.0881e-7. These quantities differ from paired trajectory drift.

| Population | Prior accurate worst trajectory pressure | Retained0.03 | New early | New early mean/max calls |
|---|---:|---:|---:|---:|
| Original58 | 3.14479e-6 | 3.61038e-6 | 2.94042e-6 | 7.913 /11 |
| Late-amended8 | 1.34895e-6 | 1.08574e-6 | 1.22632e-6 | 9.721 /15 |
| Minimum8-failed8 | 1.41517e-6 | 1.68747e-6 | 1.90452e-6 | 8.919 /16 |

Early overall mean/max calls8.211/16 versus retained0.03's7.501/11. It improves
original58pressure but fails call-work gates and minimum-population pressure.
[Preliminary population report](reports/early-population-check.json) retains every
case and gate against all three historical comparators. Retained0.03's campaign
is explicitly reused with current-package equivalence and a fresh fixture; it
was not reexecuted. Both fresh candidates still receive their own74episodes.

During CUDA work, common-RHS CPU preparation compared retained checkpoints and
then the completed early model on identical144paired inputs. Early2call median
error0.007003 versus retained0.03's0.009085; late2call median0.022164 versus0.022931.
This local improvement coexists with greater autonomous call work. Frequency/
amplitude energies and84fixed manufactured rows remain recorded separately from
trajectory metrics. No validation member or training choice was changed in response.

Late Study02 reached5000updates but drained with output_limit before successful
publication. Retained files were below32MiB, stderr empty. A real profile.json
rename between live listing/lstat reproduces a Tuldok monitor failure compatible
with this receipt; the old exact observation exception was not retained. No
unpublished weights reused. Narrow fix709d0dd passed39tests and both reviews,
then deployed separately as current-06 in27.722s with zero active Runs, all20prior
Run states, owner/catalog and scheduler settings preserved. A fresh lateStudy03
uses identical capture02and initialization. The failed Study was explicitly closed;
a postcondition shape error was resolved from its acknowledgment without replay.

Original Tuldok Models/Studies/Runs remain in the existing catalog. Rheon continues
to show recorded pressure previews through the existing project paths; this is
not live tracer or velocity flow. Managed run state belongs to Tuldok. See the
[execution ledger](execution-ledger.md), [issues](issues.md) and [plan](plan.md).
