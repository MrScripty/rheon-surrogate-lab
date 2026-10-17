"""Retain a failed qualification without treating cancellation as model accuracy."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-throughput-feedback-v1'))
from benchmark import numerical_signature
from rejection import classify


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('cpu','control','oracle-replay','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    a.cpu=a.cpu.resolve(strict=True);a.control=a.control.resolve(strict=True);a.oracle_replay=a.oracle_replay.resolve(strict=True)
    campaign=a.cpu/'regression'
    report=read(campaign/'receipt.json');state=read(a.cpu/'status.json');clocks=read(a.cpu/'phase-clocks.json')
    proof=read(a.cpu/'measurement-sources.json');replay=read(a.oracle_replay)
    assert report['status']=='failed'and report['all_worker_children_reaped']and state['status']=='failed'
    assert len(clocks)==1 and clocks[0]['phase']=='regression'and clocks[0]['child_custody']['echild']
    assert not clocks[0]['child_custody']['signals_sent']
    assert state['measurement_sources_sha256']==clocks[0]['measurement_sources_sha256']==digest(a.cpu/'measurement-sources.json')
    assert all(digest(Path(path))==sha for path,sha in proof['sources'].items())
    assert all(digest(Path(path))==sha for path,sha in replay['sources'].items())
    assert not replay['all_original_checks_passed']and not replay['cuda_initialized']and replay['optimizer_updates']==replay['simulations_advanced']==0
    assert all(r['operator_passed']and r['bound_covers_oracle']for r in replay['rows'])
    control=read(a.control);assert control['status']=='passed'and len(control['records'])==74
    assert proof['sources'][str(HERE.parent/'pressure-throughput-feedback-v1/benchmark.py')]==control['source_sha256']
    config=read(a.cpu/'regression-config.json');assert config['scene_indices']==list(range(74))and digest(a.cpu/'regression-config.json')==report['config_sha256']
    old={r['scene']:r for r in control['records']};rows=[];inputs={str(path):digest(path)for path in(a.cpu/'status.json',campaign/'receipt.json',a.cpu/'phase-clocks.json',a.cpu/'measurement-sources.json',a.control,a.oracle_replay,Path(__file__))}
    for index in range(74):
        files=list(campaign.glob(f'case-*-scene-{index:02d}/episode.json'));assert len(files)<=1
        row=dict(scene=index,group=old[index]['metrics']['group'],disposition='not_started')
        if files:
            path=files[0];folder=path.parent;episode=read(path);inputs[str(path)]=digest(path)
            row.update(raw_failure_class=episode.get('failure_class'),raw_error=episode.get('error'))
            if episode['status']=='passed':
                row.update(disposition='passed_complete_window',mean_calls=episode['metrics']['mean_calls'],maximum_calls=episode['metrics']['maximum_calls'],
                    worst_paired_pressure_relative_l2=episode['metrics']['worst_paired_pressure_relative_l2'],independent_checks=len(episode['sparse']))
            elif episode.get('failure_class')=='scientific_rejection':
                assert classify(ValueError('sparse capture schedule mismatch'),folder)['failure_class']=='scientific_rejection'
                for name,sha in episode['rejection_evidence'].items():assert digest(Path(name))==sha
                attempt=json.loads((folder/'native/rejected-attempts.jsonl').read_text())['attempt']
                callbacks=[json.loads(s)for s in(folder/'native/callbacks.jsonl').read_text().splitlines()];last=callbacks[-1]
                row.update(disposition='native_candidate_rejection',rejection_step=attempt['generation'],rejection_calls=last['network_passes'],
                    relative_residual=attempt['outcome']['relative_residual'],actual_divergence=attempt['outcome']['actual_divergence'],
                    same_request_native_pressure_relative_l2=attempt['same_request_pressure_comparison']['relative_l2'])
            elif episode['error']=='ValueError: independent same-RHS operator/pressure bound failed':
                assert index==7 and any(str(folder)in key for key in replay['sources'])
                row.update(disposition='independent_pressure_bound_failure',oracle_replay_sha256=digest(a.oracle_replay))
            elif episode['error'].startswith('KeyboardInterrupt: benchmark owner requested cancellation'):
                row.update(disposition='cancelled_after_pool_stop')
            else:raise ValueError('Unclassified retained failure: '+str(path))
            if row['disposition']!='cancelled_after_pool_stop':
                previous=a.control.parent/f"case-{old[index]['slot']:02d}-scene-{index:02d}"
                assert numerical_signature(folder/'native-control/frames.jsonl')==numerical_signature(previous/'native-control/frames.jsonl')
                row['native_control_physical_frames_exact']=True
        rows.append(row)
    counts=dict(collections.Counter(r['disposition']for r in rows));assert counts['independent_pressure_bound_failure']==1
    result=dict(format='pressure-coarse-partial-qualification-v1',status='analyzed',qualification='failed',campaign_complete=False,
        scheduled_scenes=74,dispositions=counts,rows=rows,population_gates_evaluated=False,global_call_or_accuracy_improvement_claimed=False,
        completed_independent_checks=sum(r.get('independent_checks',0)for r in rows),scheduled_independent_checks=233,
        campaign_wall_s=report['elapsed_seconds'],phase_wall_s=clocks[0]['elapsed_s'],cpu_owner_wall_s=state['elapsed_s'],memory=report['memory'],
        all_phase_children_drained=True,holdouts_consumed=0,candidate_promoted=False,analysis_elapsed_s=time.perf_counter()-started,sources=inputs,
        scope='Preserve actual incomplete74-case campaign. Complete passing windows, explicit native rejections, independent bound failure, pool-cancellation aftermath and unstarted cases are distinct. No pass percentage or complete-population/call-speed claim. Pressure bound failure is distinct from measured pressure error; replay shows independent operator/coverage valid. Serial/raw timing was not admitted after failed campaign.')
    with a.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    with a.output.with_suffix('.csv').open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k]for k in('qualification','campaign_complete','dispositions','completed_independent_checks','analysis_elapsed_s')}))


if __name__=='__main__':main()
