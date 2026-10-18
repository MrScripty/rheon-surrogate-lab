# Fundamental-mode fit and conditional frequency curriculum

Status: Accepted as a completed experiment. Jeremy authorized continuing the
recorded C60/C61 decision; numerical promotion gates failed as recorded.
Branch: `experiment/pressure-single-mode-fit-20261017`, private, based on
`53a2921058057aca605bef5d2d4636ba971802a9`. Jeremy owns eventual integration;
retain for experiment review, no automatic rebase/merge, retire after integration
or explicit disposition. C53 remains the experiment baseline.

Canonical settings: `experiments/pressure-single-mode-fit-v1/config.json`.

1. C62 tests fitting the fundamental X mode (ZYX index 0,0,1) on counts 13x15x16,
   cubic spacing 1/16 m. Labels are new analytical Neumann eigenmodes; amplitude
   and water density/timestep come only from original training states. Fresh
   Adam, exact C53 initialization, 2,000 CUDA float32 updates, learning rate 1e-4
   tapering to 1e-8, four shared corrections. Retain early feedback 0.05, remove
   the frozen teacher and other fitting examples. This is a capacity/optimization
   diagnostic, not a deployable specialist or evidence of generalization.
2. Require independent label/operator defect <=1e-10, finite nonzero autograd,
   exact initialization/recurrence and zero optimizer updates in CPU qualification.
   Use existing visible Tuldok catalog, current Pumas runtime, journalled managed
   CUDA training and fresh CPU restore with exact artifact parity. No private
   training, new catalog, output-precision escalation or physics relaxation.
3. On restored CPU weights measure the fitted mode at 1/2/4/8 passes. Capacity
   success requires four- and eight-pass relative equation residual <=1e-6 and
   no worse one/two-pass fitted-case residual than C53. Retain actual per-100-update
   fitted-mode curves separately from inherited real-frame diagnostics. Measure
   128 consumed real requests and the existing 240 manufactured cases as regression
   guards; specialist regression prevents promotion, not the capacity conclusion.
4. Only after capacity success, C63 tests a frequency-balanced supplemental bank
   from C53. Match C60's 1,500 updates, seeds, LR schedule, objective, frozen teacher,
   primary bank, real/direction/current samples and grid sampling order. Change only
   supplemental pattern content: 12 fundamental-axis, 12 other low, 8 high patterns
   per grid, each with the same four signed amplitudes. C60 is the retained matched
   control; verify data/order/scales/runtime/budget identities before attribution.
5. C63 must qualify all 128 real requests at four and eight passes, preserve C53's
   earlier-pass/worst-case limits and at least 230/240 manufactured eight-pass
   cases. No independent scenes or longer fit unless those guards pass. Original
   untouched holdouts remain unopened. If C62 cannot fit, retain failure and
   investigate optimization/objective interference before changing architecture.

Write set: new `experiments/pressure-single-mode-fit-v1/` and matching docs;
one optional trainer-transform seam in the existing typed preparation owner;
candidate CPU material resolves its Project through the retained assembly;
narrow ID reservation/report-link updates in the existing chart helper. Preserve
all preexisting dirty chart work and all historical results. No weights, complete
simulation arrays or credentials enter Git. Keep workflow and model changes in
separate commits.

Record sources/config/input/weight identities, exact device/updates, trainer and
whole-workflow elapsed time, memory, retained CPU timing, Run custody and failures.
CPU workers use one thread on distinct cores, max three, with 12 GiB host reserve
plus 2 GiB per worker. Teacher cache <=256 MiB; CUDA wall cap 1,800 s per arm.

CPU source/math/autograd qualification passed without updates. C62 completed CUDA
training, fresh CPU restore and the full recorded CPU screen. Its fitted-case gate
passed: four-pass residual 8.82325286189385e-9, eight-pass 2.545420251713197e-12.
Its broader regression guards failed; it remains a diagnostic specialist.

C63 also completed CUDA training, fresh CPU restore and the recorded CPU screen.
Matched-source, actual bank/order/work and artifact-parity checks passed. C63
improved low-mode errors against C60 but failed the unchanged C53 guards: 8/128
real requests qualified at four passes, 128/128 at eight, and 210/240 manufactured
eight-pass cases versus C53's 230. No candidate, independent scene or longer fit
was admitted after that screen. Reports, timings and findings are retained here.

Next proposed slice: a 24-case fundamental-only fit (eight grids, three axes),
without competing losses, to separate shared fitting capacity from mixed-loss
interference. Keep C53, the cubic operator and current precision. This proposal
has not started; retain this branch for Jeremy's review until integration or
explicit disposition, without automatic synchronization or history rewriting.
