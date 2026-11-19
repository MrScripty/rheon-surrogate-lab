"""Freeze dashboard selection and explicitly locate each retained CPU checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population(cohort):
    cases=cohort.get('timing_cases')
    if cohort['cases']!=128 or not cases or any(c['split']!='validation' for c in cases):
        return None
    return sorted((c['trajectory_id'],c.get('native_frame',c.get('frame')),tuple(c['counts_xyz']),c['cell_size_m']) for c in cases)


def selection(cohorts,top):
    key=next(population(c) for c in cohorts if population(c))
    candidates=[c for c in cohorts if population(c)==key]
    frontier=[c for c in candidates if not any(q['worst_error']<=c['worst_error'] and q['inference_median_ms']<=c['inference_median_ms'] and (q['worst_error']<c['worst_error'] or q['inference_median_ms']<c['inference_median_ms']) for q in candidates)]
    accurate=sorted(candidates,key=lambda c:(c['worst_error'],c['order']))[:top]
    fast=sorted(candidates,key=lambda c:(c['inference_median_ms'],c['order']))[:top]
    groups=[('frontier',frontier),('accurate',accurate),('fast',fast)]
    chosen={c['weights_sha256'] for c in frontier+accurate+fast}
    selected=[];seen=set()
    for cohort in candidates:
        identity=cohort['weights_sha256']
        if identity not in chosen or identity in seen:continue
        seen.add(identity)
        selected.append((cohort,[name for name,group in groups if any(c['weights_sha256']==identity for c in group)]))
    return selected


def prepare(root,config_path,output):
    config=read(config_path);base=root/'docs/plans';lab=root/config['source_lab']
    passes=config['passes']
    if not passes or any(type(n)is not int or n<1 for n in passes)or passes!=sorted(set(passes)):
        raise ValueError('Neural pass counts must be unique, increasing positive integers')
    if config['timing_repetitions']<1 or config['top_count']<1:
        raise ValueError('Positive timing repetitions and model ranking count required')
    page=root/config['chart']
    chart=json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>',page.read_text(),re.S)[1])
    models=[]
    for cohort,tags in selection(chart['cohort_points'],config['top_count']):
        timing=(base/cohort['timing_source']).resolve()
        if not timing.exists():timing=(lab/'docs/plans'/cohort['timing_source']).resolve()
        expected=chart['history_sources'].get(cohort['timing_source'])
        if expected and sha(timing)!=expected:raise ValueError('Changed timing evidence '+cohort['id'])
        record=read(timing)
        if record.get('replay_project'):
            project=(base/record['replay_project']).resolve();restore=(base/record['replay_artifacts']).resolve()
        elif timing.parent.name=='artifacts':
            project=timing.parents[3]/'project'
            choices=[p.parent for p in timing.parents[2].glob('stage-*/artifacts/receipt.json') if read(p).get('mode')=='restore']
            if len(choices)!=1:raise ValueError('Ambiguous CPU restore '+cohort['id'])
            restore=choices[0]
        else:
            admission=(base/record['admission_source']).resolve()
            if not admission.exists():admission=(lab/'docs/plans'/record['admission_source']).resolve()
            entry=next(e for e in read(admission)['models'] if e['weights_sha256']==cohort['weights_sha256'])
            project=root/entry['project'];restore=root/entry['artifacts']
        if sha(restore/'weights.npz')!=cohort['weights_sha256']:raise ValueError('Wrong weights '+cohort['id'])
        models.append(dict(id=cohort['id'],name=cohort['name'],weights_sha256=cohort['weights_sha256'],tags=tags,
            original_worst_error=cohort['worst_error'],original_inference_median_ms=cohort['inference_median_ms'],
            project=str(project.relative_to(root)),restore=str(restore.relative_to(root)),
            identities={str(p.relative_to(root)):sha(p) for p in [project/'source-witness.json',project/'config.json',restore/'weights.npz',restore/'normalization.json',restore/'receipt.json',restore/'history.json']}))
    reference=read(root/config['reference_adapter'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'cohort-progress-selection-snapshot.html').write_bytes(page.read_bytes())
    result=dict(schema=config['schema'],root=str(root),config=config,models=models,chart_sha256=sha(page),
        config_sha256=sha(config_path),binary_sha256=sha(root/config['binary']),
        support={k:reference[k] for k in ['support','context','inputs','output']},
        validation_identities=next(c['timing_cases'] for c,t in selection(chart['cohort_points'],config['top_count'])),
        validation_sources={name:sha(root/config['validation_inputs']/name) for name in ['validation_dataset.npz','validation_dataset_extra.npz','validation_splits.json']})
    (output/'material.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m['id']:m['tags'] for m in models},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(Path(__file__).resolve().parents[2],args.config.resolve(),args.output.resolve())
