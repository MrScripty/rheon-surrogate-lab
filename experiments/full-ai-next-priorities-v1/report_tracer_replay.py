"""Publish scalar own-history controls with supplied reference velocity."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'tracer-reference-velocity-replay-results.json'
    if output.exists():raise ValueError('Fresh scalar replay report required')
    path=BASE/'tracer-reference-velocity-replay-01.json';d=read(path)
    protocol=DOCS/'tracer-reference-velocity-replay-protocol-01.json'
    if d['status']!='completed_offline_reference_velocity_control'or len(d['rows'])!=2048 or len(d['center_velocity_qualification'])!=64:
        raise ValueError('Complete registered replay and center qualification required')
    if not d['reference_velocity_supplied_at_every_step']or d['full_state_prediction']:raise ValueError('Scalar control scope changed')
    if d['protocol_sha256']!=sha(protocol):raise ValueError('Replay protocol changed')
    for name,digest in d['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified scalar source changed')
    for name,digest in d['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Native reference history changed')
    evidence={str(p.relative_to(ROOT)):sha(p)for p in [path,protocol]}
    write(output,dict(status='completed',summaries=d['summaries'],source_evidence=evidence,source_sha256=sha(__file__),
        scalar_own_history_steps=128,reference_velocity_supplied_at_every_step=True,
        candidate_frontier_point=False,CPU_latency_claim=False,new_optimizer_updates=0,reserved_scenes_consumed=False,
        full_state_prediction=False,original240step_requirement_unchanged=True,original_goal_complete=False))
    lines=['# Scalar transport replay under reference velocity','',
        'The first-order displacement control advances its own tracer for 128 steps while receiving native reference velocity at every step. '
        'It covers all 16 training scenes and 2,048 outputs. The offline center-velocity interpolation matches 64 independent native sampler cases exactly.','',
        '| Grid group | Interval | Worst relative tracer error |',
        '| --- | ---: | ---: |']
    for r in d['summaries']:lines.append(f'| {r["regime"]} | {r["interval_steps"]} | {r["worst_relative_tracer_error"]:.8g} |')
    lines.extend(['','This is a scalar transport control, with supplied reference fluid history and no learned displacement predictor. '
        'It does not meet the full-state or 240-step forecasting objective, establish CPU model latency, or trigger reserved capture. '
        'It supports testing a compact learned displacement decoder with convex readout; an actual model must use its own predicted fluid history and pass every original gate.','',
        '[Source-bound results](tracer-reference-velocity-replay-results.json) · '
        '[Single-step representation checks](tracer-representation-results.md) · [Original scope](plan.md)',''])
    (DOCS/'tracer-reference-velocity-replay-results.md').write_text('\n'.join(lines))
    print('Published reference-velocity scalar replay; original full-state/240step requirements unchanged',flush=True)


if __name__=='__main__':run()
