"""Derive immutable small trial records from retained managed and CPU receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--managed',type=Path,help='Explicit custody index after a recorded postprocessing recovery')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.trial;managed=args.managed or root/'managed-full'
    receipt=read(managed/'receipt.json');witness=read(root/'project/source-witness.json');config=read(root/'project/config.json')
    if receipt['status']!='completed_and_drained':raise ValueError('normal reopened/redownloaded managed completion required')
    stages=[]
    for index,stage in enumerate(receipt['stages']):
        source=managed/f'stage-{index:02d}';run=read(source/'run.json')
        if run['state']!='completed' or run['custody_phase']!='drained':raise ValueError('every stage must drain')
        for item in run['artifacts']:
            path=source/'artifacts'/item['name'];again=managed/f'redownloaded-stage-{index:02d}'/item['name']
            if digest(path)!=item['sha256'] or path.read_bytes()!=again.read_bytes():raise ValueError('artifact custody/parity mismatch')
        stages.append({k:stage[k] for k in ('name','run_id','state','custody_phase','material_sha256')})
    trained=read(managed/'stage-00/artifacts/receipt.json');restored=read(managed/'stage-01/artifacts/receipt.json')
    history=read(managed/'stage-00/artifacts/history.json');control=read(args.control/'history.json');old=read(args.control/'receipt.json')
    if trained['execution']['device']!='cuda:0' or restored['execution']['cuda_initialized'] or restored['optimizer_updates']!=0:
        raise ValueError('actual CUDA training and fresh CPU restore required')
    if trained['outputs']['weights.npz']['sha256']!=restored['outputs']['weights.npz']['sha256']:raise ValueError('restore weights differ')
    feedback=history['analytical_feedback'];parents=read(root/'project/inputs/splits.json')['parents'];eligible=[i for i,x in enumerate(parents) if x['entry']['split']=='train']
    rng=np.random.Generator(np.random.PCG64(config['training']['sampling_seed']));order=[]
    for _ in range(config['training']['updates']):
        batch=[]
        for _ in range(config['training']['batch_size']):
            parent=eligible[int(rng.integers(len(eligible)))];slot=int(rng.integers(len(parents[parent]['frames'])));batch.append((parent,slot))
        order.append(batch)
    order=np.asarray(order,dtype='<i8')
    if hashlib.sha256(order.tobytes()).hexdigest()!=trained['closure']['sample_order_sha256']:raise ValueError('real exposure reconstruction differs')
    real_cells=sum(math.prod(parents[int(parent)]['counts_xyz']) for batch in order for parent,_ in batch)
    analytical_cells=sum(n*math.prod(g['grid']['counts_xyz']) for n,g in zip(feedback['grid_exposures'],feedback['grids']))
    supplement=history.get('supplemental_feedback')
    extra_draws=supplement['additional_examples'] if supplement else 0
    extra_cells=sum(n*math.prod(g['grid']['counts_xyz']) for n,g in zip(supplement['grid_exposures'],supplement['grids'])) if supplement else 0
    direction=history.get('direction_feedback')
    direction_draws=direction['additional_examples'] if direction else 0
    direction_cells=direction['forward_cells_per_correction'] if direction else 0
    result={'schema':'rheon-multigrid-trial-record-v1','trial':root.name,'configuration':config,'source_commit':witness['source_commit'],
        'source_witness_sha256':digest(root/'project/source-witness.json'),'catalog_id':receipt['catalog_id'],'managed_stages':stages,
        'cuda_optimizer_updates':trained['optimizer_updates'],'cuda_execution':trained['execution'],'cpu_restore_execution':restored['execution'],
        'optimizer_updates_per_s':history['optimizer_updates_per_s'],'optimizer_interval_seconds':history['training_finished_unix']-history['training_started_unix'],
        'weights_sha256':trained['outputs']['weights.npz']['sha256'],'weights_file_bytes':trained['outputs']['weights.npz']['bytes'],
        'normalization_sha256':trained['closure']['normalization_sha256'],'analytical_training':{k:v for k,v in feedback.items() if k!='grids'},
        'training_exposure':{'real_simulations':len(eligible),'real_states':sum(len(parents[i]['frames']) for i in eligible),
            'real_draws':int(order.shape[0]*order.shape[1]),'analytical_draws':feedback['additional_examples'],
            'forward_network_evaluations':(order.shape[0]*order.shape[1]+feedback['additional_examples']+extra_draws+direction_draws)*config['correction']['steps'],
            'forward_cell_evaluation_proxy':(real_cells+analytical_cells+extra_cells+direction_cells)*config['correction']['steps'],
            'scope':'sample/evaluation/voxel accounting, not FLOPs or backward-work equivalence'},
        'control_parity':{'initial_parameter_inventory':history['checkpoints']['checkpoint-000.npz']['inventory']==control['checkpoints']['checkpoint-000.npz']['inventory'],
            'normalization':trained['closure']['normalization_sha256']==old['closure']['normalization_sha256'],
            'real_sampling_order':trained['closure']['sample_order_sha256']==old['closure']['sample_order_sha256']},
        'final_training_trace':history['trace'][-1],'cpu_pipeline':read(root/'cpu-complete.json'),
        'adaptive_fit':read(root/'fit/report.json'),'adaptive_development':read(root/'development/report.json'),
        'mode_probe':{k:v for k,v in read(root/'mode-probe/report.json').items() if k!='cases'},
        'grid_study':{k:v for k,v in read(root/'grid-study/report.json').items() if k not in ('records','configuration')},
        'raw_native':{name:read(root/name/'evaluation.json') for name in ('raw-fit-native','raw-validation-native') if (root/name/'evaluation.json').exists()},
        'scope':'development, iterative CPU inference, unchanged physical targets; no deployment or final-test claim'}
    if supplement:
        result['supplemental_training']={k:v for k,v in supplement.items() if k!='grids'}
        result['training_exposure'].update(supplemental_draws=extra_draws, supplemental_forward_cells=extra_cells*config['correction']['steps'])
    if direction:
        result['direction_training']=direction
        result['training_exposure'].update(direction_draws=direction_draws,
            direction_forward_cells=direction_cells*config['correction']['steps'])
    if (root/'long-study/report.json').exists():result['long_study']={k:v for k,v in read(root/'long-study/report.json').items() if k not in ('records','configuration')}
    # Validate every native trajectory receipt, including failed scientific cases.
    native=[read(path) for path in root.glob('*-study/scene-*/*/receipt.json')]
    if not native or not all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in native):raise ValueError('trajectory lifecycle not drained normally')
    result['native_trajectory_children_reaped']=len(native)
    if args.managed:
        result['managed_receipt_path']=str(managed/'receipt.json')
        result['managed_recovery']=receipt.get('recovery')
    raw=json.dumps(result,indent=2,allow_nan=False)+'\n';args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if args.output.read_text()!=raw:raise ValueError('immutable record already differs')
    else:args.output.write_text(raw)
    print(json.dumps({'trial':root.name,'updates':trained['optimizer_updates'],'native_children_reaped':len(native),'record':str(args.output)}),flush=True)


if __name__=='__main__':main()
