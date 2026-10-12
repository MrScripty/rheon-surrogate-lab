"""Audit timed parallel frames and compare full fields at selected CPU settings."""
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

import trajectory_experiment as exp
from trajectory_model import ModalPressure


def metrics(teacher,predicted):
    errors=[]
    for t,p in zip(teacher,predicted,strict=True):
        tv=np.concatenate(t['velocity']).astype(np.float64)
        pv=np.concatenate(p['velocity']).astype(np.float64)
        tracer=t['tracer'].astype(np.float64)
        guess=p['tracer'].astype(np.float64)
        errors.append({'velocity_relative_l2':float(np.linalg.norm(tv-pv)/np.linalg.norm(tv)),
                       'tracer_relative_l2':float(np.linalg.norm(tracer-guess)/np.linalg.norm(tracer))})
    return {'max_velocity_relative_l2':max(r['velocity_relative_l2'] for r in errors),
            'max_tracer_relative_l2':max(r['tracer_relative_l2'] for r in errors),'frames':errors}


def main():
    run=Path(sys.argv[1]); result=json.loads((run/'parallel/result.json').read_text())
    config=result['config'];freeze=json.loads((run/'freeze.json').read_text())
    exp.BINARY=exp.ROOT/'target/release/rheon-trajectory-parallel'
    assert exp.sha(exp.BINARY)==config['binary_sha256']
    assert exp.sha(run/'model.pt')==config['checkpoint_sha256']==freeze['checkpoint_sha256']
    assert len(result['records'])==36
    count=0
    for record in result['records']:
        assert record['threads']==config['chosen_threads'][record['method']]
        assert len(record['reports'])==120
        for r in record['reports']:
            assert r['actual_divergence_max']<=1e-5 and r['relative_residual']<=1e-8 and r['courant']<=1
            if record['method']=='learned': assert r['iterations']==0
            count+=1
    for method,median in result['median_120_frame_wall_ms'].items():
        assert median==statistics.median(r['wall_ms_including_native_startup_teardown'] for r in result['records'] if r['method']==method)
    torch.set_num_threads(1)
    model=ModalPressure(freeze['n'],freeze['device'])
    model.load_state_dict(torch.load(run/'model.pt',weights_only=True,map_location=freeze['device'])['state_dict'])
    model.eval()
    accuracy=[]
    for seed in config['test_seeds']:
        os.environ['RAYON_NUM_THREADS']=str(config['chosen_threads']['learned'])
        _,predicted=exp.rollout(freeze['n'],'learned',seed,120,model,capture=True)
        case={'seed':seed}
        for method in ['jacobi','sgs']:
            os.environ['RAYON_NUM_THREADS']=str(config['chosen_threads'][method])
            _,teacher=exp.rollout(freeze['n'],method,seed,120,capture=True)
            case[method]=metrics(teacher,predicted)
            assert case[method]['max_velocity_relative_l2']<1e-5
            assert case[method]['max_tracer_relative_l2']<1e-5
            print(seed,method,case[method]['max_velocity_relative_l2'],flush=True)
        accuracy.append(case)
    audit={'status':'verified','qualified_benchmark_frames':count,'accuracy':accuracy,
           'checkpoint_sha256':config['checkpoint_sha256'],'binary_sha256':config['binary_sha256'],
           'result_sha256':exp.sha(run/'parallel/result.json'),'precision':'float64 norms over full stored float32 fields'}
    exp.write_json(run/'parallel/verification.json',audit)


if __name__=='__main__': main()
