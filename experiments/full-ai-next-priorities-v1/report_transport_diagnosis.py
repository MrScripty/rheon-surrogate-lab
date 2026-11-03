"""Publish offline transport controls without treating reference inputs as forecasts."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-coupled-transport-results.json'
    if output.exists():raise ValueError('Fresh transport diagnostic report required')
    path=BASE/'temporal-coupled-transport-diagnosis-01.json';d=read(path)
    qpath=BASE/'provisional-teacher-qualification-01.json';q=read(qpath)
    protocol=DOCS/'temporal-coupled-transport-diagnostic-protocol-01.json'
    if d['status']!='completed'or q['status']!='passed'or not d['normal_metrics_bitwise_equal_to_prior']or not d['candidate_coefficients_unchanged']:
        raise ValueError('Complete unchanged-candidate diagnostics required')
    if d['teacher_qualification_sha256']!=sha(qpath)or d['protocol_sha256']!=sha(protocol)or q['protocol_sha256']!=sha(protocol):
        raise ValueError('Diagnostic qualification/protocol changed')
    for name,digest in d['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified diagnostic source changed')
    for name,digest in {**d['source_evidence'],**q['source_evidence']}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Native diagnostic input changed')
    if len(d['rows'])!=896 or len(q['rows'])!=128 or len(q['fixtures'])!=24:raise ValueError('Complete registered populations required')
    evidence={str(p.relative_to(ROOT)):sha(p)for p in [path,qpath,protocol,BASE/'provisional-teacher-build-01/build-receipt.json',
        BASE/'b2-coupled-pair-01/receipt.json',BASE/'b2-coupled-pair-01/restore/receipt.json',BASE/'temporal-coupled-fitting-diagnosis-01.json']}
    write(output,dict(status='completed',summaries=d['summaries'],transport_summaries=d['transport_summaries'],
        source_evidence=evidence,source_sha256=sha(__file__),training_queries_only=True,new_optimizer_updates=0,
        candidate_inference_changed=False,candidate_frontier_point=False,CPU_latency_claim=False,reserved_scenes_consumed=False,
        native_pressure_solves=0,native_steps_executed=0,original_goal_complete=False,
        next_decision='Qualify direct provisional-velocity labels from the actual training representation; compare matched continuation control and auxiliary-supervision arms'))
    def value(phase,mode,field,stat='worst'):
        return next(r[stat]for r in d['summaries']if r['phase']==phase and r['mode']==mode and r['field']==field)
    lines=['# Coupled-model transport diagnosis','',
        'The train-only controls identify provisional transport as the immediate fluid limitation. '
        'Replacing pressure alone leaves warm worst velocity error at '
        f'{value("warm","offline_native_pressure","velocity"):.7g}, compared with '
        f'{value("warm","normal","velocity"):.7g} normally. Replacing the provisional velocity instead reduces it to '
        f'{value("warm","offline_native_provisional","velocity"):.7g}. '
        'These reference-field substitutions are offline controls and are excluded from candidate frontiers.','',
        '| Warm training control | Worst pressure | Worst equation | Worst velocity | Worst tracer |',
        '| --- | ---: | ---: | ---: | ---: |']
    for mode in read(protocol)['modes']:
        lines.append('| '+mode+' | '+' | '.join(f'{value("warm",mode,f):.7g}'for f in ('pressure','equation','velocity','tracer'))+' |')
    lines.extend(['','The learned transport improves on the saved initialization and zero-transport controls. '
        'Its warm provisional error is nevertheless about 33% of the native transport increment on median and up to 106% in this probe. '
        'Direct provisional-velocity supervision is the next controlled objective comparison, alongside the full-state loss. '
        'Tracer changes little under either fluid substitution and needs its own transport/generalization diagnosis.','',
        'The offline reconstruction passes 24 independent native fixtures and 128 captured targets. '
        f'Corrected velocities round-trip exactly; maximum RHS disagreement is {q["maximum_captured_RHS_relative_error"]:.7g}. '
        'That qualification uses original float64 pressure. Labels reconstructed from the actual normalized float32 training targets still require a separate check.','',
        'All 896 diagnostic outputs are retained. The unchanged model reproduces its previous fitting errors bitwise and its coefficients remain unchanged. '
        'No optimizer update, native pressure solve, native timestep, latency claim or reserved capture occurs. '
        'Every physical output, 240-step accuracy gate and combined 4/6/8 ms budget remains unchanged.','',
        '[Machine-readable summaries and evidence](temporal-coupled-transport-results.json) · '
        '[Diagnostic protocol](temporal-coupled-transport-diagnostic-protocol-01.json) · '
        '[Actual full-state result](temporal-coupled-results.md)',''])
    (DOCS/'temporal-coupled-transport-results.md').write_text('\n'.join(lines))
    print('Published train-only transport diagnosis; no candidate target or latency claim',flush=True)


if __name__=='__main__':run()
