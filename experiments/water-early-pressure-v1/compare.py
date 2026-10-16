"""Compare matched fits with frozen consumed baselines, retaining failed cases."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

LAB=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pressure_population_owner',LAB/'experiments/water-pressure-bound-regression-v1/analyze.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
read,digest,write,write_csv=owner.read,owner.digest,owner.write,owner.write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--reports',type=Path,required=True)
    args=p.parse_args();config=read(args.config);root=LAB/config['root']
    report=read(root/'development/report.json')
    assert report['state'] in ('completed','failed') and report['config_sha256']==digest(args.config)
    admission=read(root/'admission.json')
    assert admission['config_sha256']==digest(args.config)
    assert report['models']==admission['models'] and report['binary_sha256']==admission['binary_sha256']
    for item in admission['sources']:
        assert digest(LAB/item['path'])==item['sha256']
    old=read(LAB/admission['baseline_summary'])
    assert old['source_report_sha256']==digest(LAB/'runs/water-pressure-bound-regression-v1/development/report.json')
    assert config['scenes']==read(LAB/'experiments/water-pressure-bound-regression-v1/configs/regression.json')['scenes']
    cases=list(old['cases']);rows=[];sparse=[];failures=[]
    metadata={r['scene']:r for r in cases if r['model']=='late-pressure'}
    models=[m['name'] for m in config['models']]
    for info in report['models']:
        assert digest(LAB/info['artifacts']/'weights.npz')==info['weights_sha256']
        assert digest(LAB/info['artifacts']/'normalization.json')==info['normalization_sha256']
    native={r['scene']:r for r in report['records'] if r['model']=='native'}
    assert all(native[i]['qualified'] for i in range(74))
    for model in models:
        observed={r['scene']:r for r in report['records'] if r['model']==model}
        for i,expected in enumerate(config['scenes']):
            row=observed.get(i)
            directory=LAB/row['output'] if row else None
            receipt=read(directory/'native/receipt.json') if directory and (directory/'native/receipt.json').exists() else None
            frames=[json.loads(s) for s in (directory/'native/frames.jsonl').read_text().splitlines()] if receipt else []
            frames=[f for f in frames if f.get('ai_quality')]
            callbacks=[json.loads(s) for s in (directory/'native/callbacks.jsonl').read_text().splitlines()] if receipt else []
            complete=bool(receipt and receipt['result'] and not receipt['result']['hard_stop'] and frames
                and frames[-1]['step']==expected['steps']
                and len(frames)==len(callbacks)==expected['steps']-expected['warmup_steps'])
            qualified=bool(complete and row['qualified'] and receipt['native_process']['reaped'])
            calls=[c['network_passes'] for c in callbacks]
            peak=max(frames,key=lambda f:f['relative_pressure_difference']) if frames else None
            info=metadata[i]
            cases.append({'scene':i,'population':info['population'],'group':info['group'],'model':model,
                'qualified':qualified,'complete_window':complete,'calls':calls,
                'worst_paired_pressure_relative_l2':peak['relative_pressure_difference'] if peak else None,
                'peak_step':peak['step'] if peak else None,
                'source_receipt_sha256':digest(directory/'native/receipt.json') if receipt else None,
                'observed_native_reaped':receipt['native_process']['reaped'] if receipt else None})
            if not qualified:failures.append({'scene':i,'model':model,'complete':complete})
            bystep={c['step']:c for c in callbacks}
            for f in frames:
                c=bystep.get(f['step']);last=c['trace'][-1] if c else {}
                rows.append({'scene':i,'group':info['group'],'population':info['population'],'model':model,
                    'step':f['step'],'time_s':f['time_s'],'paired_pressure_relative_l2':f['relative_pressure_difference'],
                    'relative_residual':f['ai_quality']['relative_residual'],
                    'actual_divergence':f['ai_quality']['actual_divergence'],
                    'network_evaluations':c['network_passes'] if c else None,
                    'pressure_bound':last.get('mean_free_pressure_relative_error_bound'),
                    'stopping_passed':last.get('equation_passed'),'physical_passed':f['ai_quality']['qualified']})
            diagnostic=directory/'diagnostics.json' if directory else None
            if diagnostic and diagnostic.exists():
                value=read(diagnostic)
                assert value['config_sha256']==digest(args.config)
                assert value['weights_sha256']==next(m for m in report['models'] if m['name']==model)['weights_sha256']
                for sample in value['samples']:
                    assert sample['pressure_bound_covers_oracle']
                    sparse.append({'scene':i,'model':model,'group':info['group'],**{k:v for k,v in sample.items() if k!='top_modes_zyx'}})
            else:failures.append({'scene':i,'model':model,'reason':'missing sparse diagnostics'})
    groups={};table=[]
    for group in ('original58','late-amended8','minimum8-failed8'):
        chosen=[r for r in cases if r['group']==group]
        expected=len([r for r in chosen if r['model']=='late-pressure'])
        summaries={m:owner.describe([r for r in chosen if r['model']==m]) for m in ['reference-current','late-pressure',*models]}
        checks={m:{baseline:owner.gates(summaries[baseline],summaries[m],expected)
                   for baseline in ['reference-current','late-pressure']} for m in models}
        groups[group]={'models':summaries,'gates':checks}
        table.extend({'group':group,'model':m,**s} for m,s in summaries.items())
    eligible=[m for m in models if not any(f['model']==m for f in failures) and all(
        all(value for checks in g['gates'][m].values() for value in checks.values()) for g in groups.values())]
    def rank(m):
        selected=[r for r in cases if r['model']==m]
        v=owner.describe(selected)
        return [v[k] if v[k] is not None else 10**9 for k in
                ('maximum_network_evaluations','worst_paired_pressure_relative_l2','mean_network_evaluations')]
    diagnostic_choice=min(eligible or models,key=rank)
    result={'schema':'rheon-earlier-pressure-comparison-v1','groups':groups,'cases':cases,'failures':failures,
        'eligible_for_fresh_validation':eligible,'frozen_challenger':diagnostic_choice,
        'choice_scope':'conditional fresh validation if eligible; otherwise representative consumed timing only',
        'all_consumed_gates_passed':bool(eligible),'candidate_promoted':False,
        'independent_bound_checks':len(sparse),'elapsed_seconds':report['elapsed_seconds'],
        'sources':{'admission_sha256':digest(root/'admission.json'),'report_sha256':digest(root/'development/report.json'),
                   'analysis_sha256':digest(__file__)},
        'scope':'Three consumed populations individually; require no regression versus retained reference AND frozen pressure-aware late model. Paired trajectories and sparse same-RHS errors are distinct.'}
    write(args.reports/'regression-summary.json',result)
    write_csv(args.reports/'population-comparison.csv',table)
    write_csv(args.reports/'trajectory-metrics.csv',rows)
    write_csv(args.reports/'sparse-metrics.csv',sparse)
    write_csv(args.reports/'case-comparison.csv',[{**{k:v for k,v in r.items() if k!='calls'},
        'mean_calls':statistics.mean(r['calls']) if r['calls'] else None,'max_calls':max(r['calls'],default=None)} for r in cases])
    print(json.dumps({'groups':groups,'eligible':eligible,'frozen_challenger':diagnostic_choice,'failures':failures},indent=2))


if __name__=='__main__':main()
