"""Parallelize independent native pressure checks with one process owner per case."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

from context import LAB, data, model, policy
import diagnostics
import numpy as np


def initialize(binary, cpus):
    global BINARY, ORACLE, VALIDATOR, PROFILE, PROCESS
    index=multiprocessing.current_process()._identity[0]-1
    os.sched_setaffinity(0,{cpus[index%len(cpus)]})
    model.configure('cpu')
    BINARY=Path(binary)
    ORACLE,VALIDATOR=diagnostics.checks.historical_oracles()
    PROFILE=diagnostics.targets.load(LAB/'experiments/float32-pressure-target-v1/configs/development.json')
    PROCESS=policy.load_module('conditioning_native_owner',LAB/'experiments/local-pressure-v1/process.py')


def evaluate(task):
    case,request=task
    response,lifecycle=PROCESS.native_call(BINARY,'--evaluate',request,time.monotonic()+63,60)
    independent=diagnostics.checks.check_response(request,response,ORACLE,VALIDATOR)
    return {**case,**response,'float32_acceptance':diagnostics.targets.assess(response['raw_learned'],PROFILE),
            'independent_candidate_residual_l2':independent,'process':lifecycle,
            'worker_pid':os.getpid(),'worker_affinity':sorted(os.sched_getaffinity(0))}


def run(args):
    import torch
    model.configure('cpu')
    receipt=policy.decode(policy.read(args.inference/'receipt.json'))
    if receipt['schema']=='rheon-water-learning-training-v1':
        policy.require(receipt['mode']=='restore' and receipt['execution']['device']=='cpu'
                       and receipt['execution']['cuda_initialized'] is False,'cpu_restore','fresh completed CPU restore')
        weights=receipt['outputs']['weights.npz']['sha256'];predictions_sha=receipt['outputs']['predictions.npz']['sha256']
    elif receipt['schema']=='rheon-learned-iteration-diagnostic-v1':
        source=Path(receipt['source_cpu_restore']);raw_source=policy.read(source/'receipt.json')
        restored=policy.decode(raw_source)
        policy.require(policy.sha(raw_source)==receipt['source_cpu_restore_sha256']
            and restored['schema']=='rheon-water-learning-training-v1' and restored['mode']=='restore'
            and restored['optimizer_updates']==0 and restored['execution']['device']=='cpu'
            and restored['execution']['cuda_initialized'] is False,
            'iteration_source','exact fresh managed CPU restore source')
        policy.require(receipt['runtime']=={'device':'cpu','cuda_initialized':False}
            and restored['closure']['source_witness_sha256']==receipt['source_witness_sha256']
            and restored['outputs']['weights.npz']['sha256']==receipt['weights_sha256']
            and restored['outputs']['normalization.json']['sha256']==receipt['normalization_sha256'],
            'iteration_source','same measured model and preprocessing')
        policy.require(receipt['inference_policy']['name'] in ('fixed-repeated-learned-correction','adaptive-learned-correction')
            and type(receipt['inference_policy']['proposals']) is int and 1<=receipt['inference_policy']['proposals']<=32
            and receipt['inference_policy']['nn_passes_per_pressure_solve']==receipt['inference_policy']['proposals']*restored['architecture']['correction']['steps']
            and receipt['inference_policy']['nn_passes_per_pressure_solve']<=32
            and receipt['inference_policy']['native_pressure_refinement'] is False,
            'iteration_policy','explicit bounded learned-only diagnostic')
        if receipt['inference_policy']['name']=='adaptive-learned-correction':
            owner=policy.load_module('learned_stopping_owner',LAB/'experiments/water-stability-learning-v1/learned_iteration.py')
            owner.stopping_policy(receipt['inference_policy']['stopping'])
            policy.require(len(receipt['case_iterations'])==len(receipt['cases'])
                and all(c['case']==case and type(c['proposals']) is int
                    and 0<=c['proposals']<=receipt['inference_policy']['proposals']
                    and c['network_passes']==c['proposals']*restored['architecture']['correction']['steps']
                    for c,case in zip(receipt['case_iterations'],receipt['cases'])),
                'iteration_policy','same selected cases and bounded adaptive work')
        weights=receipt['weights_sha256'];predictions_sha=receipt['predictions_sha256']
    else:
        policy.require(receipt['schema']=='rheon-water-learning-inference-v1' and receipt['runtime']['device']=='cpu'
                       and receipt['runtime']['cuda_initialized'] is False,'cpu_inference','completed managed inference')
        weights=receipt['weights_sha256'];predictions_sha=receipt['predictions_sha256']
    policy.require(receipt['optimizer_updates']==0,'cpu_inference','no CPU optimization')
    raw=policy.read(args.inference/'predictions.npz',2*policy.MIB)
    policy.require(policy.sha(raw)==predictions_sha,'predictions','complete restored artifact')
    arrays=data.helpers().read_npz(raw,receipt['prediction_inventory'],2*policy.MIB)
    _reservation,parents=data.read_view(args.packet)
    if receipt['schema']=='rheon-learned-iteration-diagnostic-v1':
        policy.require(policy.sha(policy.canonical(_reservation))==receipt['packet_experiment_sha256'],
            'iteration_population','exact selected diagnostic packet')
    cases=[(i,c) for i,c in enumerate(receipt['cases']) if not args.training_only or c['split']=='train']
    policy.require(cases,'population','nonempty exact restored cases')
    measured=diagnostics.admit(args.binary)
    captures={};tasks=[]
    for index,case in cases:
        parent=parents[case['parent']];root=Path(parent['capture_root'])
        if root not in captures:
            capture_raw=policy.read(root/'manifest.json',4*policy.MIB)
            policy.require(policy.sha(capture_raw)==parent['capture_manifest_sha256'],'capture','exact original states')
            captures[root]=(policy.sha(capture_raw),policy.decode(capture_raw))
        trajectory=next(p for p in captures[root][1]['trajectories'] if p['trajectory_id']==parent['scene']['trajectory_id'])
        policy.require(trajectory['entry']==parent['scene'] and case['native_frame']==parent['frames'][case['frame']]['index']
                       and case['trajectory_id']==parent['entry']['trajectory_id'],'case','exact parent/frame lineage')
        frame=next(f for f in trajectory['frames'] if f['index']==case['native_frame']);slot=case['frame']
        velocity=[np.frombuffer(policy.read(root/frame['arrays']['pre_'+axis]['path'],policy.MIB),dtype='<f4').astype(float).tolist() for axis in 'uvw']
        request={'schema':'rheon-variable-pressure-teacher-case-v1','counts_xyz':parent['counts_xyz'],
            'spacing_m':[parent['cell_size_m']]*3,'density_kg_m3':parent['density_kg_m3'],'dt_s':parent['entry']['dt_s'],
            'original_rhs':parent['arrays']['rhs'][slot].reshape(-1).tolist(),
            'reference_pressure':parent['arrays']['reference'][slot].reshape(-1).tolist(),
            'previous_pressure':parent['arrays']['base'][slot].reshape(-1).tolist(),
            'pre_velocity':velocity,'candidate_pressure':arrays[f'case_{index:03d}'].reshape(-1).tolist(),'candidate_error':None}
        tasks.append((case,request))
    output=policy.fresh(args.output)
    cpus=sorted(os.sched_getaffinity(0));available=cpus[1:] or cpus
    workers=min(args.workers or len(available),len(available),len(tasks))
    started=time.time();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(str(args.binary),available)) as pool:
        for record in pool.map(evaluate,tasks):
            records.append(record)
            with (output/'cases.jsonl').open('ab') as stream:stream.write(policy.canonical(record)+b'\n')
            print('NATIVE',len(records),'/',len(tasks),flush=True)
    policy.require(diagnostics.admit(args.binary)==measured,'binary_drift','same admitted native evaluator')
    for root,(digest,_manifest) in captures.items():
        policy.require(policy.sha(policy.read(root/'manifest.json',4*policy.MIB))==digest,'capture_drift','unchanged captures')
    values=[r['raw_learned']['relative_residual'] for r in records]
    finite=[v for v in values if v is not None and math.isfinite(v)]
    result={'status':'complete_native_diagnostic','cases':len(records),
        'candidate_policy':receipt.get('inference_policy'),
        'raw_native_qualified':sum(r['raw_learned']['qualified'] for r in records),
        'float32_qualified':sum(r['float32_acceptance']['qualified'] for r in records),
        'relative_residual':{'minimum':min(finite) if finite else None,'median':float(np.median(finite)) if finite else None,'maximum':max(finite) if finite else None},
        'methods':diagnostics.checks.summary(records),'weights_sha256':weights,
        'workers':workers,'used_cpus':sorted({c for r in records for c in r['worker_affinity']}),
        'elapsed_s':time.time()-started,'all_native_children_drained':all(r['process']['drained'] and r['process']['group_empty'] and r['process']['ownership_released'] for r in records),
        'cuda_initialized':torch.cuda.is_initialized(),
        'scope':'unchanged native equation/physical gates and oracle; independent teacher-state cases, no cold start or autonomous rollout'}
    policy.write(output/'evaluation.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','inference','output','binary'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--training-only',action='store_true');parser.add_argument('--workers',type=int)
    args=parser.parse_args()
    if args.workers is not None and args.workers<1:parser.error('--workers must be positive')
    run(args)


if __name__=='__main__':main()
