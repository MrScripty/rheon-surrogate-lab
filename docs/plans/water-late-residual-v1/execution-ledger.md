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
