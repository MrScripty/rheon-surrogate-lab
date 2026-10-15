# Pressure conditioning results

Status: schedule controls and scalar-feature fitting are being evaluated; managed
GPU throughput controls are being qualified. Scientific accuracy remains unqualified.
Historical source, captures and results are preserved.

The historical one-frame fit reproduces relative residual 0.00177824. The same
frame's ideal correction rounded through actual float32 output reconstruction
has residual 9.73352e-8 and passes every native physical gate under the existing
1e-6 model profile. All 1,024 actual training states and 128 validation states
also pass the equation-only rounded-reference control. This does not substitute
for full physical qualification of all rounded states.

On the selected historical 1,024-parent checkpoint, median residual is 0.0353517
on its actual training states and 0.0385793 on validation. Doubling residual has
median correction-linearity error 3.38%/3.88% respectively; sign reversal has
12.31%/14.01%. These diagnostics motivate amplitude-preserving conditioning but
do not identify GroupNorm as the sole cause. Eight repeated raw corrections
reach median validation residual 0.00429424; an operator-based minimum-residual
step reaches 0.00373089. Neither is a qualified pressure solve or measured native
simulation acceleration.

The constant-1e-4 one-frame run completed 10,000 CUDA updates and fresh managed CPU
restore. Its native fitted-case relative residual is 0.00103740, approximately
41.7% below the historical fit but still over 1,000 times the equation target.
All physical gates remain unchanged. Other controlled results are pending final
native evaluation; loss reduction alone is not success.

CPU diagnostics used all assigned cores 1–23 on the 1,024-state training audit,
finishing in 23.42 seconds while CUDA optimizer progress advanced. Every worker
used one Torch thread and zero CUDA initialization. Pressure inference on one
16-cubed frame is fastest among measured thread counts at eight threads:
3.26 ms versus 541.07 ms at 24, on the actual host. This is an isolated
microbenchmark, not end-to-end simulation timing. Profiling shows slowdown across
small convolutions, normalization, padding and fills; it does not establish a
single universal cause.

Persistent-worker replay of all 1,024 recorded states gives approximately
1,455–1,468 states/s with 23 single-threaded workers versus 460–462 states/s with
one eight-threaded worker. Startup is reported separately and can dominate small
batches. Three eight-threaded workers have little gain and one severe stall;
a passive wait-policy control removes that stall but gives only 435–490 states/s.
These are pressure-state throughput measurements, not three time-evolving native
simulations. Existing CPU deployment threading/witnesses are unchanged.

Parallel native checking reproduces all scientific outputs from the historical
128-case serial validation exactly, excluding elapsed_s, solver_s and
verification_s. Each native process group drained and ownership was released.
The current Tuldok supervisor executes one managed Run at a time; additional
CUDA submissions queue. Concurrent managed training requires a separate runtime
scheduler/lifecycle change, not a second submission or private optimizer.
