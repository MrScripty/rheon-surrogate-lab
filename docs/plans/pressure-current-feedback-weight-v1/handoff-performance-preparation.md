# Separate handoff-performance investigation

Status: baseline/preparation only. No Tuldok code, authorization, runtime,
custody behavior or numerical model changed. This does not select a checkpoint.

The first completed managed Study took1922.802seconds, versus547.501seconds in
its CUDA trainer and2.896seconds in its CPU restore trainer. Full Study time also
contains admission, queue, staging, evidence, drainage and all artifact readback;
the difference cannot be attributed entirely to one component. Recorded restore
approval took161.441seconds and restore admission199.938seconds.

A minimal read-only request for one byte of the189byte published CPU profile
artifact reproduced slow handoff:33.214seconds wall,33.201seconds in transport,
0.016745seconds client CPU. Client output validation took about0.013seconds.
The actual negotiated owner request, segment identity/hash checks and a2second
diagnostic latency bound are retained locally. The command deliberately returned
nonzero for the observed latency; this is a performance reproduction, not a
model/physics failure. No credential or returned payload is retained in the report.

After both managed Study controllers completed, the identical one-byte read still
took27.535seconds, including27.499seconds in transport and0.039646seconds client
CPU. The original Tuldok process used27.15seconds aggregate CPU and made
4352053375logical read bytes/55591read calls during that interval, with zero
physical read_bytes. Linux clock ticks were independently checked at100/s.
These process-wide counters can include UI/background work; they are not a
function-level server profile. They weaken a purely concurrent-Study or physical
disk-wait explanation and support profiling repeated cached validation first.
No request-local reuse or other optimization was implemented.

Ranked hypotheses for a later isolated performance change:

1. Repeated server-side Run/context/lineage validation. Actual restore material
   has nine artifact references to one completed training Run. A counting replay
   through production authorize_transitive_inputs confirms nine parent loads and
   nine current-authority checks. This proves duplication, not its causal share
   of HTTP latency. Profile _load, _extension_context and _external_inventory
   before choosing an optimization.
2. Contention with other Study readbacks or UI requests. Repeat the same tiny
   request after managed controllers drain; preserve the original catalog and
   open desktop apps. Measure lock wait separately from service CPU.
3. Storage/metadata access. Measure server CPU and I/O, and compare payload sizes
   at the same lineage depth. A tiny returned payload alone does not exclude
   expensive validation of other retained files.

The counting replay uses actual retained child/parent documents and counting
load/authority seams; it performs no HTTP request, optimizer update or simulation.
The first replay correctly refused an earlier running-state snapshot. A fresh
read of the authoritative completed parent supplied the valid replay; the old
snapshot remains preserved. The first HTTP probe setup selected a nonexistent
restore artifact and stopped before transport; the corrected probe used an
actual published profile artifact. Neither preparation error mutated a Run.

If repeated parent validation dominates, consider request-local reuse of validated
parent facts, with explicit ownership and no cache surviving the request. Preserve
current per-input authorization, actor/catalog binding, source/hash and population
checks, cycle/depth bounds, custody verification, artifact inventories and final
whole-file hashes. Verify revoked authority on a later request, changed/missing
parent evidence, shared DAGs, cycles and distinct source hashes. Do not assume
that memoization is safe or sufficient before those tests and server profiling.

Keep any implementation on a separate Tuldok branch/commit and qualify against
the same original catalog/CPU checkpoint restore. Current scientific configurations
and results stay immutable. This document is preparation for that work; it makes
no throughput-improvement claim.
