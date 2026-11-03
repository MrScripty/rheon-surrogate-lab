"""Publish native transport representation checks without a forecasting claim."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'tracer-representation-results.json'
    if output.exists():raise ValueError('Fresh tracer representation report required')
    path=BASE/'tracer-representation-qualification-01.json';d=read(path)
    protocol=DOCS/'tracer-representation-protocol-01.json'
    if d['status']!='passed_offline_representation_qualification'or len(d['rows'])!=256:
        raise ValueError('Complete registered tracer representation probes required')
    if not d['native_primitive_matches_captured_next_tracer_bitwise']or not d['at_most_eight_active_donors']:
        raise ValueError('Native primitive and convex support not qualified')
    if d['protocol_sha256']!=sha(protocol):raise ValueError('Tracer protocol changed')
    for name,digest in d['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified tracer source changed')
    for name,digest in d['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Native input/target changed')
    evidence={str(p.relative_to(ROOT)):sha(p)for p in [path,protocol,BASE/'tracer-teacher-build-01/build-receipt.json']}
    write(output,dict(status='completed',summaries=d['summaries'],maximum_CFL=d['maximum_CFL'],source_evidence=evidence,
        source_sha256=sha(__file__),candidate_frontier_point=False,CPU_latency_claim=False,new_optimizer_updates=0,
        candidate_inference_implemented=False,training_queries_only=True,reserved_scenes_consumed=False,
        reference_displacements_only_offline=True,original_goal_complete=False,
        next_decision='If full-state tracer accuracy remains inadequate, qualify a learned displacement decoder and convex tensor-product readout; no native backtrace may enter deployed inference'))
    lines=['# Native tracer representation controls','',
        'A convex displacement-based readout reproduces all 256 native primitive outputs exactly at float32 precision '
        'when supplied native displacement labels. The 64 captured next-tracer states also match the native primitive bitwise. '
        'The probes cover all 16 training scenes, four starting times, and captured, zero, constant and random tracer fields.','',
        '| Tracer pattern | Native displacement worst relative error | Offline first-order median | Offline first-order worst |',
        '| --- | ---: | ---: | ---: |']
    for row in d['summaries']:
        lines.append(f'| {row["pattern"]} | {row["native_displacement_worst"]:.7g} | {row["offline_Euler_median"]:.7g} | {row["offline_Euler_worst"]:.7g} |')
    lines.extend(['',f'Maximum observed CFL is {d["maximum_CFL"]:.6g}. The qualified local support contains at most eight active donors. '
        'Zero and constant tracer fields are preserved. This supports a small learned-displacement representation in place of 27 unconstrained softmax weights.','',
        'These are offline numerical and representation controls. The displacements are supplied by the native sampler; '
        'no learned predictor, optimizer update, complete-state forecast, rollout pass or CPU inference timing is established. '
        'A candidate must predict its displacements from causal model inputs, qualify gradients and state behavior, and pass every full-state horizon and own240 check. '
        'Native backtracing or reference displacements cannot be inserted into candidate inference.','',
        'The active matched provisional-supervision continuations keep the existing tracer architecture. '
        'Assess their actual outcomes before selecting the next representation comparison.','',
        '[Evidence and source hashes](tracer-representation-results.json) · [Frozen protocol](tracer-representation-protocol-01.json) · '
        '[Full-state target](plan.md)',''])
    (DOCS/'tracer-representation-results.md').write_text('\n'.join(lines))
    print('Published offline tracer representation controls; no model target claim',flush=True)


if __name__=='__main__':run()
