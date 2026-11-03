"""Attach the separately retained inference-policy campaign to the offline page."""
import json
import hashlib
import os
import statistics
from pathlib import Path
from four_pass_study import completed_four_pass_studies
from detached_correction_chart import completed_detached_study
from itertools import chain


def enrich_pass_sweep(data,base):
    artifact=Path(base)/'pressure-pass-sweep-v1/chart-data.json'
    if artifact.exists():
        result=json.loads(artifact.read_bytes())
        if result['schema']!='rheon-inference-pass-sweep-chart-v1':
            raise ValueError('Unknown inference-pass study schema')
        data['pass_sweep']=result
        lab=Path(base).parent.parent
        known={p['id']:p for p in data['cohort_points']}
        added=0
        from current_cohort_records import SOURCE_LAB
        detached_models, detached = completed_detached_study(base,lab,SOURCE_LAB)
        extra = [('pressure-detached-correction-v1','Detached correction study','detached-correction-study',
                  Path(base)/'flow-reasoning-assessment/execution/results.json',detached)] if detached else []
        for study_name, _, tag, study_path, study in chain(completed_four_pass_studies(base),extra):
            for fit in study['training']:
                identifier=fit['id']
                if identifier not in known:
                    continue
                evidence=study['snapshots'][identifier]
                if 'source' in evidence:
                    path=lab/evidence['source']
                    expected=evidence['sha256']
                else:
                    jobs=study['cpu_execution']['rows']+study.get('curriculum_cpu_execution',{}).get('rows',[])
                    job=next(row for row in jobs if row['label']==identifier+'-snapshots')
                    if job['returncode']!=0 or not job['reaped']:
                        raise ValueError('Completed drained snapshot evaluation required')
                    command=job['command']
                    path=Path(command[command.index('--output')+1])/'snapshots.json'
                    expected=hashlib.sha256(path.read_bytes()).hexdigest()
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Fine-tuning snapshot evidence changed')
                snapshots=json.loads(path.read_text())
                if ('source' not in evidence and snapshots!=evidence
                        or snapshots['weights_sha256']!=known[identifier]['weights_sha256']):
                    raise ValueError('Fine-tuning snapshot report/checkpoint mismatch')
                source=os.path.relpath(path,base)
                result['sources'][source]=expected
                point=known[identifier]
                result['models'].append(dict(id=identifier,name=point['name'],tags=[tag],
                    weights_sha256=point['weights_sha256'],original_worst_error=point['worst_error'],
                    original_inference_median_ms=point['inference_median_ms']))
                for row in snapshots['summary']:
                    cases=[r for r in snapshots['rows'] if r['passes']==row['passes']]
                    identity=lambda r:(r['trajectory_id'],r.get('native_frame',r.get('frame')),tuple(r['counts_xyz']),r['cell_size_m'])
                    if (len(cases)!=128 or sorted(map(identity,cases))!=sorted(map(identity,point['timing_cases']))
                            or max(r['relative_residual'] for r in cases)!=row['worst_equation_residual']
                            or statistics.median(r['relative_residual'] for r in cases)!=row['median_equation_residual']):
                        raise ValueError('Fine-tuning snapshot population/statistics differ')
                    result['points'].append(dict(model=identifier,passes=row['passes'],cases=row['cases'],
                        worst_error=row['worst_equation_residual'],median_error=row['median_equation_residual'],
                        inference_median_ms=row['median_inference_ms'],
                        inference_best_ms=min(r['inference_s'] for r in cases)*1000,
                        inference_worst_ms=max(r['inference_s'] for r in cases)*1000,
                        equation_passes=row['equation_passes'],mean_actual_calls=row['passes'],
                        source=source,timing_source=source,study=study_name))
                    added+=1
        result['additional_snapshot_settings']=added
        if detached:
            next(m for m in result['models'] if m['id']=='C66')['tags'].append('detached-correction-study')
            result['repeated_control']=detached['repeated_control']
            result['timing_scope']+=' The detached-correction C53/U control exactly reproduces C66; C66 keeps its earlier measured timings, and the linked study retains the replay timings.'
        if added:
            result['timing_scope']+=' Fine-tuned checkpoints add isolated 1/2/4/8-pass measurements on the same 128 states. No trajectory on this tab’s original setup was recorded for these checkpoints. Available fresh simulation results are in Drift over time and the linked studies.'
