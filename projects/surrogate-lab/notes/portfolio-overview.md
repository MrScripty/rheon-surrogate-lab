# Rheon model portfolio

## Pressure benchmarks now cover 150 fits

The unified “150 pressure fits · complete GPU requests · 128 consumed cases” view joins the 127
named fits with 23 original-training-output fits. All 185 measured default fit identities now have
GPU accuracy and timing: 150 pressure proposals and 35 full-state generators. Twenty other
components still need appropriately scoped benchmarks. This does not complete all 524 parameter
states or intended solver variants.

For the 23 newly measured pressure fits, median case latency ranges from 2.08794 to 8.65121 GPU ms
per pressure request and worst mean-free pressure error ranges from 0.00602154 to 1.58280. These are
retained experiments, not promoted application-quality models. A pressure request does not advance a
full simulation step; compare it separately from the complete F64 generator results below.

Select both unified pressure partitions to compare all 150 subjects; filter cells per metre
separately from grid X/Y/Z and input dt. Primary timing excludes forward-validation hooks; separate
paired overhead for the additional 23 fits is retained without subtraction. The native maximum chart
shows the maximum over selected case statistics; cards also report the median across case medians.
Original process captures remain at their exact bound paths, and historical measurement tables stay
unchanged.

This project retains 205 recorded fits/components and 524 parameter byte states. All 205 have source
reviews; 35 full-state generator fits now have matched GPU benchmarks. The four F64 baseline
executions are not four new fits. The 127 named pressure fits have a separate 128-request GPU
comparison.

## Matched full-state results

On the same 64 retained native cases, the original F64 cell composition has better worst pressure
error and 7.645–31.085 times lower matched median GPU time than every one of the 35 other generator
fits. This compares distinct retained designs; the best checkpoint has not disappeared or
deteriorated.

| Physical outputs per pass | Worst pressure relative L2 | GPU ms/output, median case | GPU ms/output, 32 cubed |
| --- | --- | --- | --- |
| 1 | 1.55386e-16 | 0.25839 | 0.58552 |
| 2 | 2.79755e-16 | 0.17926 | 0.47411 |
| 3 | 2.79755e-16 | 0.14110 | 0.43128 |
| 4 | 2.79755e-16 | 0.12893 | 0.40099 |

Every output includes pressure, all three staggered velocities and tracer, and advances a full 1/60
second. The F64 implementation executes sequential N1 advances; packing K outputs amortizes
transfers and graph-call overhead, rather than learning a shortcut across intermediate states. The
N8-only coordinate change is unused in this regime.

The best pressure tails among the 35 other fits are about 5.24e-4. A controlled K1 diagnostic
locates the fixed-viscosity series' shared 32-cubed worst pressure case mainly in its CNN pressure
fallback. Other pressure-map and request errors remain. The diagnostic does not replace a causal
generator with future reference input.

## Reading the measurements

Choose the retained source titled “Matched GPU generators and F64 baseline”, then select K, grid
dimensions and cells per metre. Plot IDs use model names; b1-v1 and b1-v2 are distinct fits.
Original fit identities and exact process captures remain inspectable. Times cover native CPU input,
packing, known controls, GPU prediction and all owned CPU outputs with synchronization.

The table above reports medians of per-case medians. A native maximum plot reports the maximum of
the selected per-case statistic. Thus “Median GPU time per physical output” on that plot means the
largest case median, not a median across cases. Each fit card includes density/domain tables and the
exact retained source identity.

The population is 16 consumed fitting scenes at four starts, densities 8 and 16 cells/m, and grids
15×6×8, 9×11×11, 16 cubed and 32 cubed. It is not independent holdout evidence or a 600-step
own-history rollout. Resource observations found no competing CUDA compute PID; external host work
was present and continuous hardware exclusivity is not established.

## Historical timing correction

The four earlier “Complete GPU generators” tables retain ten fits measured with a forward-validation
hook inside the timing interval. Their scalar values and original packets stay unchanged. The new
matched tables exclude that hook and preserve all prior output hashes. Paired calibration measures
median hook overhead of 0.0593–0.4997 ms per block across fits. That instrumentation mismatch does
not explain the numerical error gap.

Historical instrumented packet identities:
```text
f6969ec40ad9153a88a474ea7eb4e77d3652b441a7c535763de4253328817f70
ee6a1e516cf9efd181c9f5427037dece7e7c46ec5fe1064ccafaefa9ed325f10
1bd75131441638b77c5509232b2b8afb9f27a068c618a8dd6bb2fc19715a13b9
19954f1816a8748269628cd0728d52c7a53bc61c3c67bc65015ad2da6e827470
```

The quoted near-exact cell and temporal anchors also reproduce on their original populations. Those
use different rectangles, densities or cadences. The original temporal entry requires fine-clock
force schedules absent from these legacy scenes; its cadence results stay separate. No fine
histories were invented for this comparison.

## Model history and remaining work

Pressure-only error/time and complete simulation-output error/time are separate tasks. Guide
summaries explain each method, fitting exposure, purpose and outcome. Authored comparisons organize
evidence; all 94 genuine Trials retain their original unknown ancestry.

Clean GPU training/publication reproduction, remaining component/checkpoint benchmarks and long
drift still need qualification. Thirty-five parameter states remain outside native declarations, and
936 registrations still need disposition. No unique weights are approved for deletion by these
results alone.

Old-Rheon references remain the benchmark authority. The updated Rheon changes fluid behavior and
needs a separate physics epoch with training/test regeneration scripts. Final portable save/load
acceptance and practical fast Open remain unfinished.
