"""Select across frozen policy and training comparisons without spending fresh scenes."""
import importlib.util
from pathlib import Path

LAB=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('population_metrics',LAB/'experiments/water-pressure-bound-regression-v1/analyze.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
read,write,digest=owner.read,owner.write,owner.digest


def main():
    reports=LAB/'docs/plans/water-pressure-recovery-v1/reports'
    candidates=[]
    for phase in ('policy','training'):
        summary=read(reports/phase/'regression-summary.json')
        config=LAB/f'experiments/water-pressure-recovery-v1/configs/{phase}-regression.json'
        for model in read(config)['models']:
            name=model['name']
            metrics=owner.describe([r for r in summary['cases'] if r['model']==name])
            candidates.append({'phase':phase,'name':name,'eligible':name in summary['eligible_for_fresh_validation'],
                'metrics':metrics,'model':model,'config':str(config.relative_to(LAB)),
                'config_sha256':digest(config),'summary_sha256':digest(reports/phase/'regression-summary.json')})
    def rank(row):
        m=row['metrics']
        return m['maximum_network_evaluations'],m['worst_paired_pressure_relative_l2'],m['mean_network_evaluations']
    eligible=[r for r in candidates if r['eligible']]
    qualified=[r for r in candidates if r['metrics']['qualified_scenes']==r['metrics']['complete_scenes']==r['metrics']['scenes']==74]
    chosen=min(eligible or qualified,key=rank) if (eligible or qualified) else None
    result={'candidates':candidates,'eligible':bool(eligible),'selected':chosen,
            'scope':'frozen challenger for seed replication if eligible; otherwise timing diagnostic only',
            'replication':'pending' if eligible else 'deferred: comparative consumed gates failed',
            'fresh_validation':'pending successful seed replication' if eligible else 'deferred: comparative consumed gates failed',
            'independent_scenes_used':0,'candidate_promoted':False,'source_sha256':digest(__file__)}
    write(reports/'selection.json',result)
    print(__import__('json').dumps({'eligible':bool(eligible),'selected':chosen['name'] if chosen else None},indent=2))


if __name__=='__main__':
    main()
