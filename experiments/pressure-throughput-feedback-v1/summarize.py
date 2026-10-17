"""Retain per-population accuracy, work and timing without promoting candidates."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pressure_population_summary',HERE.parent/'water-pressure-bound-regression-v1/analyze.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


def summarize(baseline, campaigns, destination):
    historical=owner.read(baseline)
    cases=[r for r in historical['cases'] if r['model'] in ('reference-current','late-pressure','control')]
    sources={'historical_summary':{'path':str(baseline),'sha256':owner.digest(baseline)},'campaigns':{}}
    sparse=[];times={};failures=[]
    for name,path in campaigns.items():
        report=owner.read(path)
        if report['status'] not in ('passed','failed_cases_retained') or len(report['records'])!=74:
            raise ValueError('complete74 development records required')
        if sorted(r['scene'] for r in report['records'])!=list(range(74)):raise ValueError('exact unique scene membership required')
        sources['campaigns'][name]={'path':str(path),'sha256':owner.digest(path)}
        times[name]={'complete_elapsed_s':report['elapsed_seconds'],'workers':report['workers'],'memory':report['memory'],
            'scope':'complete concurrent CPU development episodes with fresh startup, native controls and sparse independent oracles; not inference latency'}
        for record in report['records']:
            if record.get('status')=='failed':
                failures.append({'model':name,'scene':record['scene'],'error':record['episode']['error']})
                meta=next(r for r in cases if r['model']=='control' and r['scene']==record['scene'])
                cases.append({**meta,'model':name,'qualified':False,'complete_window':False,'calls':[],'worst_paired_pressure_relative_l2':None})
            else:
                metric=record['metrics'];cases.append({**metric,'model':name})
                sparse.extend({'model':name,'scene':record['scene'],'group':metric['group'],**row} for row in record['sparse'])
    groups={};models=list(campaigns)
    for group in ('original58','late-amended8','minimum8-failed8'):
        chosen=[r for r in cases if r['group']==group];expected={'original58':58,'late-amended8':8,'minimum8-failed8':8}[group]
        descriptions={m:owner.describe([r for r in chosen if r['model']==m]) for m in ('reference-current','late-pressure','control',*models)}
        checks={m:{b:owner.gates(descriptions[b],descriptions[m],expected) for b in ('reference-current','late-pressure','control')} for m in models}
        groups[group]={'models':descriptions,'gates':checks}
    eligible=[m for m in models if all(all(v for gate in group['gates'][m].values() for v in gate.values()) for group in groups.values())]
    result={'format':'rheon_pressure_band_comparison_v1','sources':sources,'groups':groups,'cases':cases,'failures':failures,
        'timing':times,'sparse':sparse,'independent_checks':len(sparse),'eligible_for_unused_eight':eligible,
        'untouched_scenes_consumed':0,'candidate_promoted':False,
        'scope':'Each population independently versus retained reference, prior accurate late model and harder-data26b control. Same-RHS sparse pressure error and paired trajectory drift remain distinct.'}
    owner.write(destination,result)
    owner.write_csv(destination.with_suffix('.csv'),[{'group':g,'model':m,**v} for g,row in groups.items() for m,v in row['models'].items()])
    print(json.dumps({'eligible':eligible,'failures':failures,'timing':times,'groups':groups},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--campaign',action='append',required=True,help='NAME=complete report path');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();campaigns={}
    for item in a.campaign:
        name,separator,path=item.partition('=')
        if not separator or name in campaigns or name in ('control','reference-current','late-pressure'):raise ValueError('unique fresh model label required')
        campaigns[name]=Path(path)
    summarize(a.baseline,campaigns,a.output)

if __name__=='__main__':main()
