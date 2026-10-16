# Execution ledger

- Start the authorized matched early/late direction comparison on a new branch
  from a1b84ff593034e747417e063cb261e12d9c429aa. Keep previous results/model and
  unrelated cohort-progress files. The prior accuracy study remains Accepted.
- Eight fixed patterns per existing analytical grid retain all three fundamental
  directions, mixed directions and high-frequency controls without duplicating
  the four amplitude variants of a homogeneous model. This bounds each extra
  numeric bank to one32 MiB typed input and keeps the managed input boundary
  small. Both fits receive exactly the same selected cases/shapes/order.
- The next architecture/loss ablations remain conditional; first isolate extra
  early versus late directions with identical objective and amplitude rules.
- The first fixture failed at the preprocessing call boundary (wrong signature),
  before a bank was emitted; fixed and retained separately. Both real/analytical
  fixture pairs then qualified in4.066s, no CUDA/optimizer work.
- At2026-10-16T09:54 UTC, the24-worker full preparation caused a system-wide OOM
  followed by host reboot. Kernel evidence shows large Python resident sets and
  global OOM killing desktop browser processes. No new managed Run had started.
  Original timing stays `running` as last durable observation; interruption
  reconciliation records host restart, not normal process reaping/completion.
- Safe reproduction gives the exact cause: the existing spectral oracle's
  unoptimized four-operand einsum requests32,768,000,000bytes for40^3. A disposable
  process with512 MiB additional address-space cap rejects it in0.033s, without
  exhausting the host. No model call is needed, ruling out autograd as the cause;
  concurrency amplifies the per-solve allocation.
- Replace both reachable production four-operand cosine contractions with
  explicit axiswise two-operand products. Same40^3 probe now completes in0.007s,
  pressure error4.38e-16 and11.1 MiB additional peak RSS. Two tests pass, covering
  dense small-grid parity/roundtrip and bounded real40^3 allocation. No precision,
  pressure equation, training recipe or historical result changes in this fix.
- Reduce preparation to4 workers, stream a bounded queue, cap worker/parent
  address-space growth, require8 GiB available, and construct one small analytical
  grid bank at a time. Revised real/analytical fixture qualifies in3.190s. Retain
  all earlier fixture manifests and failed source; next full attempt gets a fresh
  output/timing identity. Memory bounds take precedence over maximizing worker count.
- User clarified step size means simulation dt and requested an analytic test
  without many new simulations/refits. Add sampling_probe.py as an isolated CPU
  diagnostic. Ideal sinusoid samples/cycle=1/(f*dt), Nyquist requires f<1/(2*dt);
  period=1/f. Slow variation needs a longer observation window, not inherently a
  smaller dt. Translating wavelength to time requires a propagation/advection
  model; frozen aligned advection gives f=U/lambda, not a universal fluid rule.
- Probe completed in2.443s, peak758,595,584bytes, one CPU thread, no CUDA,
  no optimizer or advanced simulations. Six already consumed saved states at
  five dt factors(.25,.5,1,2,4), four network calls per solve: input and pressure
  bits identical in every comparison. Doubling residual changes predictions in
  all six negative controls. Divergence estimates vary with dt as the existing
  stopping equation specifies. Holding residual fixed is a contract check;
  actual changed-dt trajectories need not have the same residual or pressure.
- Three stationary manufactured fundamental modes on the first saved grid have
  relative pressure errors0.00744%,0.06043%,0.76741% after four calls (Z,Y,X).
  Independent neighbor-action defects <=5.95e-16; spectral inverse agrees with
  each known pressure <1e-9. Spatial solve error thus exists at temporal frequency
  zero, without temporal aliasing. This does not quantify the temporal error in
  evolving flows or establish the cause of every historical trajectory error.
  Command/environment and replay results: reports/temporal-sampling-command.json
  and reports/temporal-sampling-probe.json. No new data enters fitting or holdout
  selection; matched early/late training protocol is unchanged.
- User correctly identifies a separate hypothesis: slowly evolving signals may
  be contaminated by accumulated step error despite many samples/cycle. The
  fixed-state probe does NOT reject this hypothesis. For N=1/(f*dt) updates per
  cycle, coherent local velocity bias b accumulates as N*b only under neutral
  propagation; uncorrelated zero-mean errors scale in RMS as sqrt(N)*sigma under
  that assumption, and coupled dynamics may damp or amplify errors. Neither is
  a measured growth law for Rheon. A local pressure error injects velocity error
  -dt/rho*grad(delta_p); hence if that gradient bias stays fixed and coherent,
  neutral accumulation over a cycle is approximately -grad(delta_p)/(rho*f).
  dt cancels in this restricted example; more steps alone do not imply that a
  smaller dt improves or worsens total error. Existing saved trajectories can
  measure total pressure separation over time, but cannot identify a changed-dt
  causal effect or add per-step relative pressure errors as velocity drift.
