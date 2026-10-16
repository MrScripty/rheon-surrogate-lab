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
- User proposes postponing low-frequency loss to reduce other-band error first,
  then restoring low-frequency training. Treat this as a curriculum hypothesis,
  not permission to hide low-frequency evaluation error or change the currently
  matched early/late fit protocol. Spatial band balancing and a staged objective
  are possible separate ablations; neither has been trained or qualified here.
- CPU band_probe.py reuses six consumed targeted-checkpoint residuals, zero-base
  correction map, original h/rho/dt/scales and4/8 actual single calls. Partition
  by the existing cosine index cube0..2, excluding gauge, into low and remainder;
  evaluate each input separately and their sum. Qualify partition <1e-14 and
  prediction superposition discrepancy <1e-5 relative to exact low-mode pressure.
  This index cutoff is spatial, not temporal, and is not a common physical
  wavenumber cutoff across different domain proportions.
- Probe passes in2.718s, no CUDA, advanced simulations or optimizer updates.
  Leakage from the other input band has4.04%–59.63% of the intrinsic low-input
  error norm across12 comparisons. Removing the other input band improves low
  accuracy5/12 times and worsens7/12, where useful cancellation is removed.
  Thus cross-band contamination exists but intrinsic low-input error is larger
  on every sampled case/call count. Neither a universal noise floor nor a benefit
  from high-first training is established. Masking low output error would also
  cease penalizing leakage into that band; it need not reduce that contamination.
  Preserve unchanged full-band physical evaluation even for a future masked-loss
  warm-up. Timings, inputs, calls, decomposition and identities are retained in
  reports/spatial-band-probe.json and its command receipt. No model promotion.
- User confirms direct low-frequency response first and reiterates durable
  methods/results retention. Keep curricula deferred. Completed bounded bank:
  144 paired directions (32 real train states plus112 manufactured directions),
  no exclusions, worst independent operator defect4.071e-13,12.158s, parent peak
  886,439,936bytes. Matching5,000 extra draws and54,460,289 cells/correction;
  combined logical residency706,219,584bytes. Reports/bank-preparation.json and
  direction-loader.json retain exact lineage, bounds and sampling identities.
- Final source review caught packet-receipt verification comparing original
  packet slots against added direction slots; check original packet before
  extending the typed inputs. Reuse existing trial record owner, adding direction
  draws/cell work only when present; previous record accounting stays unchanged.
  All32 focused trainer tests pass after source changes (1.176s test interval).
- Before source capture, reserve8 new whole final scenes with unique seeds and
  physical setups, disjoint from136 prior configured scenes. Config file hash
  c2f4e824331d4ee8cc45971cc8a3045292d3cb3143e77598de87668462c063db.
  Prior independent-seed confirmation remains unconsumed. No optimizer yet.
- Register live snapshot2cce354d8b724a1dbce76989e470a437, late residual training,
  in the same catalog a7da4a8de53541a2a0e2aad08005bb71; reopen it and confirm old
  pressure-accuracy project retained. GPU is available on the host RTX5090;
  sandbox nvidia-smi lacks device access, host query succeeds. Use real managed
  host CUDA execution, never substitute CPU fitting.
- Capture smoke/early/late project materials from clean source7df75591de53e0f2da012e30d58aba057fb30ca1.
  Verify copied witness identities, then remove only the clean task worktree,
  confirming exact OID retention on the experiment branch and absence of its
  path/registration. Source-capture and worktree-cleanup receipts retained.
- Managed smoke qualifies1 CUDA optimizer update followed by fresh CPU restore
  with0 updates/no CUDA initialization. Both Runs complete and drain in the same
  catalog; restored weight bytes and every artifact's second download match.
  Run IDs/execution details and total managed elapsed boundary are in
  reports/managed-smoke.json. Full controlled sequence starts only after this.
- Prepare fixed4/8/12-call common-input comparison against six consumed saved
  inputs, independently enumerating the operator against spectral labels. Both
  retained/targeted baseline probes pass. At8 calls: retained0/6 equation checks,
  targeted6/6; worst pressure errors8.516e-6 and1.526e-5 respectively. At12 both
  pass6/6, worst errors1.824e-8 and1.618e-7. These are same-RHS pressure metrics,
  not autonomous trajectory accuracy or isolated timing benchmarks.
- Describe train-only paired-bank spectra: median reference correction energy
  in the existing low-mode cube rises90.012%→99.655% for real states and
  50.0%→99.666% for manufactured directions. RHS low-mode medians23.226%→78.326%
  and50.0%→67.683%. Description0.140s, no optimizer/CUDA; this characterizes the
  extra direction effect, not a newly trained model result. Matched full fits and
  their measured CPU/native outcomes remain pending.
- Expand the fixed-call comparison before new-checkpoint outcomes to all15
  unique existing captured inputs, including historical peak frames and both
  reference/targeted trajectories. Exact-byte deduplication removes the three
  shared first states. Preserve the six-input fixtures and verify their numerical
  results are exactly unchanged in the expanded reports. Targeted passes15/15
  at8 calls; retained2/15. Both pass15/15 at12 calls. Commands and source identity
  are in fixed-all-retained.json and fixed-all-targeted.json. These reused
  consumed inputs do not become training examples or candidate-selection labels.
