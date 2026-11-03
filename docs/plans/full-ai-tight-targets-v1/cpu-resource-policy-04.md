# CPU resource allowance and efficiency measurements

The owner permits a qualifying generator to use up to four physical CPU cores.
This is a prospective change to the earlier single-thread resource requirement;
earlier measurements and registrations retain their original meaning.

Single-core complete inference latency remains a primary measure of compute
efficiency. Report it alongside two-core and four-core latency; additional cores
are an option, not an assumed improvement. Shared caches, memory bandwidth and
thread management may make a larger allocation slower. Select the smallest
allocation that achieves the best measured complete latency, retaining all
measured configurations, including regressions.

For a controlled scaling comparison, use the same physical CPU type, learned
weights, inputs, intermediate outputs and full inference clock. Include one
performance core (CPU 6), two performance cores (6 and 4), and four performance
cores (6, 4, 5 and 7). Retain CPU 23, an efficiency core used in historical
measurements, as a separately labelled single-core control. Never attribute a
performance-core versus efficiency-core difference to parallel scaling.

Record affinity, actual worker placement, thread limits, full block median,
95th percentile and maximum latency, amortized latency per generated simulation
step, and speedup and core-normalized efficiency against the matching one-core
run. Measure complete K=1,2,3,4 outputs. Thread launch and synchronization costs
inside inference count. Setup and warmup are reported separately. Do not run
competing numerical jobs during measurements. A matched Rheon comparison must
use and report the same core allowance and comparable affinity.

The target remains a maximum of 1.2 ms per generated simulation step and worst
equation, pressure, velocity and tracer error at most 1e-5, with the existing
own-history and independent verification gates. The block budgets are therefore
1.2, 2.4, 3.6 and 4.8 ms. An improvement in timing alone does not qualify a model.
Managed CUDA learning, exact CPU restores, complete outputs and the ordered
higher-density and 600-step phases remain required.

First experiment: parallelize independent rows in the existing compiled learned
transport and sparse stages, preserving each voxel's arithmetic order. Keep
pressure transforms and prescribed forcing serial initially so their resource
use is explicit. Require byte-exact agreement with the serial implementation
before using the timing results. This experiment changes execution only and
performs no parameter learning.
