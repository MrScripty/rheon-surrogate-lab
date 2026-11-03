"""Publish CPU diagnostics separately from serial benchmark latency."""
import statistics
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'coupled-CPU-stage-profile-results.json'
    if output.exists():raise ValueError('Fresh profile report required')
    path=BASE/'coupled-CPU-stage-profile-01.json';d=read(path);protocol=DOCS/'coupled-CPU-stage-profile-protocol-01.json'
    if d['status']!='completed_diagnostic_only'or len(d['parity'])!=32 or len(d['stages'])!=48:raise ValueError('Complete registered profile required')
    if d['protocol_sha256']!=sha(protocol)or not d['weights_unchanged']or d['optimizer_updates']!=0:raise ValueError('Unchanged profile context required')
    for name,digest in d['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Profile source changed')
    weights=BASE/'b2-provisional-aux-pair-01/restore/weights.npz'
    if sha(weights)!=d['model']['weights_sha256']:raise ValueError('Profiled weights changed')
    for r in d['stages']:
        if r['module_calls']['network']!=1 or r['module_calls']['fluid']!=1 or any(v!=2 for k,v in r['module_calls'].items()if k not in('network','fluid')):
            raise ValueError('Counted profile path differs from deployed K2 model')
    preparation=[]
    for regime in ('small','32cubed'):
        values=[r['preparation_and_reconstruction_seconds']for r in d['stages']if(max(r['counts_xyz'])<=16)==(regime=='small')]
        preparation.append(dict(regime=regime,median_outside_network_ms=statistics.median(values)*1000))
    evidence={str(p.relative_to(ROOT)):sha(p)for p in [path,protocol,BASE/'b2-provisional-aux-pair-01/receipt.json',weights,
        BASE/'coupled-CPU-stage-profile-batch-01/execution.json']}
    write(output,dict(status='completed',stage_summaries=d['stage_summaries'],input_forcing_and_output=preparation,
        preliminary_denormal_timings=d['preliminary_variant_timings'],source_evidence=evidence,source_sha256=sha(__file__),
        denormal_variant_adopted=False,denormal_variant_full_qualification_not_triggered=True,
        interpretation='No observed denormal-flush speed benefit; prioritize tracer readout, field/forcing work and measured stage costs',
        profile_overhead_is_not_benchmark_latency=True,candidate_frontier_point=False,new_optimizer_updates=0,
        reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Coupled CPU stage profile','',
        'This diagnostic uses the restored auxiliary checkpoint, one CPU thread on core23, eight consumed development scenes and starting steps0/63. '
        'Inclusive module measurements use hooks and are not the serial benchmark latency.','',
        '| Module | Small-grid inclusive block ms | 32³ inclusive block ms |','| --- | ---: | ---: |']
    names=[r['module']for r in d['stage_summaries']if r['regime']=='small']
    for name in sorted(names,key=lambda n:next(r['median_inclusive_block_ms']for r in d['stage_summaries']if r['regime']=='small'and r['module']==n),reverse=True):
        values=[next(r['median_inclusive_block_ms']for r in d['stage_summaries']if r['regime']==regime and r['module']==name)for regime in ('small','32cubed')]
        lines.append(f'| {name} | {values[0]:.4f} | {values[1]:.4f} |')
    lines.extend(['',f'Input encoding, prescribed forcing and output reconstruction outside the network take a median {preparation[0]["median_outside_network_ms"]:.4f} ms on the selected small-grid profile cases. '
        'The tracer decoder is the largest individually measured small-grid module. These medians are inclusive diagnostics and do not sum into the benchmark median.','',
        'The preliminary denormal screen shows no speed benefit and is not adopted. '
        'Selected-output classifications remain unchanged, but no full-window/own240 execution-variant qualification is claimed. '
        'The original weights and execution policy remain unchanged.','',
        '[Profile evidence](coupled-CPU-stage-profile-results.json) · [Five-repeat measured CPU costs](temporal-provisional-cost-results.md) · '
        '[Matched accuracy results](temporal-provisional-results.md)',''])
    (DOCS/'coupled-CPU-stage-profile-results.md').write_text('\n'.join(lines))
    print('Published CPU stage profile; denormal variant not adopted',flush=True)


if __name__=='__main__':run()
