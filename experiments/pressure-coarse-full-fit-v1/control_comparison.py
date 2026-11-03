"""Bind a retained full control without claiming a fresh control rerun."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_campaign(campaign,config_path,weights_sha256,benchmark_source):
    config=json.loads(config_path.read_text())
    if campaign['config_sha256']!=digest(config_path):raise ValueError('Control campaign configuration differs')
    manifest_path=Path(config['manifest']);manifest=json.loads(manifest_path.read_text())
    if campaign['manifest_sha256']!=digest(manifest_path):raise ValueError('Control campaign package manifest differs')
    if manifest['files']['weights.npz']['sha256']!=weights_sha256:
        raise ValueError('Control campaign packaged weights differ from measured Study')
    sources={str(config_path):digest(config_path),str(manifest_path):digest(manifest_path)}
    for name,row in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Flat captured package file required')
        path=manifest_path.parent/name
        if not path.is_file()or path.is_symlink()or path.stat().st_size!=row['bytes']or digest(path)!=row['sha256']:
            raise ValueError('Control campaign package file differs')
        sources[str(path)]=row['sha256']
    if (campaign['runtime_manifest_sha256']!=digest(Path(config['runtime_manifest']))
        or campaign['binary_sha256']!=digest(Path(config['binary']))
        or campaign['source_sha256']!=digest(benchmark_source)
        or manifest['files']['installed-runtime.json']['sha256']!=campaign['runtime_manifest_sha256']):
        raise ValueError('Control campaign runtime/binary/source differs')
    for row in campaign['records']:
        command=row['command']
        if (Path(command[command.index('--config')+1]).resolve()!=config_path.resolve()
            or command[command.index('--scene')+1]!=str(row['scene'])
            or Path(command[command.index('-B')+1]).resolve()!=benchmark_source.resolve()):
            raise ValueError('Control campaign worker command differs')
    sources.update({str(path):digest(path)for path in (Path(config['runtime_manifest']),Path(config['binary']),benchmark_source)})
    return sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','control-project','control-study','qualified-project','control-campaign','control-campaign-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pressure-throughput-feedback-v1'))
    from compare_training import measured
    receipt,metrics=measured(args.control_study,args.root)
    control=json.loads((args.control_project/'source-witness.json').read_text())
    qualified=json.loads((args.qualified_project/'source-witness.json').read_text())
    reference=json.loads((args.qualified_project/'gradient-reference.json').read_text())
    history=json.loads((args.control_study/'train/history.json').read_text())
    left=json.loads((args.control_project/'config.json').read_text());right=json.loads((args.qualified_project/'config.json').read_text())
    assert {k:v for k,v in left.items()if k!='id'}=={k:v for k,v in right.items()if k!='id'}
    for project,witness in ((args.control_project,control),(args.qualified_project,qualified)):
        for name,identity in witness['sources'].items():assert digest(project/name)==identity,name
    old=reference['original_sources'];shared=control['sources']
    assert set(old)==set(shared)|{'base_model.py','coarse.py'}
    assert old['base_model.py']==shared['model.py']
    assert all(old[name]==identity for name,identity in shared.items()if name not in('model.py','config.json'))
    closure=receipt['training_receipt']['closure']
    assert closure['source_witness_sha256']==digest(args.control_project/'source-witness.json')
    for key in ('inputs','normalization_sha256','runtime_manifest_sha256'):assert closure[key]==control[key]==qualified[key]
    assert metrics['optimizer_updates']==5000 and all(receipt['artifact_parity'].values())
    orders=dict(real=closure['sample_order_sha256'],**{label:history[name]['sampling_order_sha256']for label,name in (
        ('analytical','analytical_feedback'),('supplemental','supplemental_feedback'),('direction','direction_feedback'),('current','current_feedback'))})
    assert orders==reference['sampling_order_sha256']
    initial=history['checkpoints']['checkpoint-000.npz']['inventory'];candidate=reference['initial_inventory']
    assert set(candidate)==set(initial)|{'spatial_coarse.weight'}
    assert all(candidate[name]==dict(shape=row['shape'],sha256=row['sha256'])for name,row in initial.items())
    import numpy as np
    zero=np.zeros((64,64),dtype='<f4')
    assert candidate['spatial_coarse.weight']==dict(shape=[64,64],sha256=hashlib.sha256(zero.tobytes()).hexdigest())
    logged=[row['gradient_norm_before_clip']for row in history['trace']]
    assert len(logged)==50 and all(math.isfinite(value)for value in logged)
    campaign=json.loads(args.control_campaign.read_text());assert campaign['status']=='passed'and campaign['all_worker_children_reaped']
    assert sorted(row['scene']for row in campaign['records'])==list(range(74))
    assert sum(len(row['sparse'])for row in campaign['records'])==233
    campaign_sources=bind_campaign(campaign,args.control_campaign_config,receipt['weights_sha256'],
        Path(__file__).resolve().parents[1]/'pressure-throughput-feedback-v1/benchmark.py')
    sources={str(path):digest(path)for path in(args.control_project/'source-witness.json',args.control_project/'config.json',
        args.qualified_project/'source-witness.json',args.qualified_project/'gradient-reference.json',args.control_study/'receipt.json',
        args.control_study/'train/history.json',args.control_study/'train/receipt.json',args.control_campaign,Path(__file__))}
    sources.update(campaign_sources)
    result=dict(format='pressure-coarse-full-fit-retained-control-v1',status='passed',retained_control=True,fresh_control_run=False,
        control_study_id=receipt['study_id'],control_run_id=receipt['training_run_id'],control_restore_run_id=receipt['restore_run_id'],
        weights_sha256=receipt['weights_sha256'],qualified_reference_sha256=digest(args.qualified_project/'gradient-reference.json'),qualified_project_witness_sha256=digest(args.qualified_project/'source-witness.json'),
        original_numeric_config_except_id_exact=True,all_other_shared_numerical_sources_exact=True,
        inputs_normalization_runtime_exact=True,all_five_full_orders_exact=True,all_surviving_initial_tensors_exact=True,
        campaign_package_weights_bound=True,sampled_native_norms_finite=len(logged),optimizer_updates=5000,control_trajectories=74,control_independent_checks=233,
        training_metrics=metrics,sources=sources,optimizer_updates_executed_by_report=0,
        scope='Retained completed native-clipping control and exact CPU restore, original config except id, shared sources excluding intentional coarse model/clipping replacement, inputs/init/full orders/runtime. Fifty logged norms are sampled evidence, not stored per-update gradients. No fresh control or accuracy campaign executed; inference costs include coarse arithmetic only in candidate.')
    with args.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k]for k in('status','retained_control','sampled_native_norms_finite','control_trajectories','control_independent_checks')}))


if __name__=='__main__':main()
