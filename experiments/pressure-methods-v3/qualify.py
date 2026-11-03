"""Qualify physical recurrence, exact replay, and actual 128³ autograd before fits."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def paths(project):
    rows=read(project/'project.json')['data'];witness=read(project/'source-witness.json');result={}
    for row in rows:
        if row['id'] in ['dataset','dataset_extra','splits','initial_weights'] or row['id'].startswith('view_'):
            path=ROOT/row['path']
            if sha(path)!=row['sha256'] or sha(path)!=witness['inputs'][row['id']]['sha256']:raise ValueError('Input identity changed')
            result[row['id']]=path
    return result

def qualify(args):
    if args.output.exists():raise ValueError('Fresh qualification required')
    os.sched_setaffinity(0,{args.cpu});args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();project=args.project
    extra=sorted(k for k in paths(project)if k.startswith('view_'))
    if not args.cuda:
        text=(ROOT/'experiments/pressure-methods-v1/qualify_hybrid_base.py').read_text()
        text=text.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path('+repr(str(ROOT))+')')
        text=text.replace('paths = source_inputs(project)', 'paths = {r["id"]:ROOT/r["path"] for r in read(project/"project.json")["data"]}')
        text=text.replace("['dataset','dataset_extra']",repr(['dataset','dataset_extra',*extra]))
        capture=args.output.with_name(args.output.stem+'-recurrence-qualifier.py');capture.write_text(text)
        spec=importlib.util.spec_from_file_location('v3_recurrence_qualifier',capture)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        preliminary=args.output.with_name(args.output.stem+'-recurrence.json');module.qualify(project,preliminary)
        result=read(preliminary)
    else:result=dict(status='passed',project_sha256=sha(project/'project.json'),source_witness_sha256=sha(project/'source-witness.json'),optimizer_updates=0)
    sys.path.insert(0,str(project))
    import torch,model,data,fine_tuning,mixed_order,train
    model.configure('cuda'if args.cuda else'cpu');torch.set_num_threads(1)
    config=read(project/'config.json');train.experiment(config);model.select(config)
    inputs=paths(project)
    _,parents=data.load(tuple(inputs[k].read_bytes()for k in ['dataset','dataset_extra',*extra]),read(inputs['splits']))
    normalization=model.fit_normalization(parents,data.fit_normalization)
    network=model.PressureCorrectionNet().to('cuda'if args.cuda else'cpu')
    fine_tuning.initialize(network,inputs['initial_weights'],data.helpers())
    inventory=data.helpers().state_inventory(network)
    actual=mixed_order.order(parents,config);stats=dict(mixed_order.STATS)
    small=[p for p in parents if max(p['counts_xyz'])<=16]
    control=mixed_order.order(small,dict(config,training=dict(config['training'],updates=1600)))
    replay=actual[[max(parents[int(row[0,0])]['counts_xyz'])<=16 for row in actual]]
    if replay.shape!=control.shape or not bool((replay==control).all()):raise ValueError('Original small replay changed')
    largest=max(parents,key=lambda p:p['arrays']['base'][0].size)
    if args.cuda:torch.cuda.reset_peak_memory_stats()
    losses=[]
    # Both cold/warm fields participate when the largest native trajectory is present.
    for frame in range(len(largest['frames'])):
        network.zero_grad(set_to_none=True)
        sample=data.sample(largest,frame,device='cuda'if args.cuda else'cpu')
        loss=model.training_loss(network,sample,normalization).mean()
        if not bool(torch.isfinite(loss)) or not loss.requires_grad:raise ValueError('Nonfinite or nondifferentiable objective')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):raise ValueError('Incomplete finite gradient ownership')
        losses.append(float(loss.detach()))
    if inventory!=data.helpers().state_inventory(network):raise ValueError('Zero-update qualification modified parameters')
    if not args.cuda:
        import io
        predicted={}
        with torch.no_grad():
            for index,case in enumerate(train.cases(parents)):
                sample=data.sample(parents[case['parent']],case['frame'])
                pressure=model.predict_pressure(network,sample,normalization)
                if not bool(torch.isfinite(pressure).all()):raise ValueError('Publication inference produced nonfinite pressure')
                predicted[f'case_{index:03d}']=pressure[0,0].numpy().copy()
        result['cpu_publication_inference_verified']=True
        if hasattr(train,'publication_arrays'):
            packet=io.BytesIO();data.np.savez(packet,**train.publication_arrays(predicted))
            if len(packet.getvalue())>8*1024**2:raise ValueError('Diagnostic export exceeds evidence shard cap')
            result['prediction_export_smoke']=dict(bytes=len(packet.getvalue()),max_bytes=8*1024**2,
                full_prediction_inventory=data.inventory(predicted),sampled_export_inventory=data.inventory(train.publication_arrays(predicted)),
                full_resolution_cpu_restore_hash_check_retained=True)
    observed=data.helpers().observed_peak_rss_bytes()
    result['observed_peak_rss_bytes']=observed
    if not args.cuda and torch.cuda.is_initialized():raise ValueError('CPU qualification initialized CUDA')
    result.update(status='passed',entrypoint_configuration_validated=True,exact_initial_inventory=inventory==fine_tuning.SPEC['initial_inventory'],
        small_replay=dict(matching_updates=1600,exact_sample_subsequence=True),sampling=stats,
        largest_counts_xyz=largest['counts_xyz'],largest_native_frames=[f['index']for f in largest['frames']],largest_losses=losses,
        finite_largest_grid_backward=True,parameters=sum(p.numel()for p in network.parameters()),optimizer_updates=0,
        source_sha256=sha(__file__),cuda_initialized=torch.cuda.is_initialized(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()if args.cuda else 0,
        elapsed_s=time.perf_counter()-started,scientific_acceptance_or_promotion=False)
    write(args.output,result);print('Qualified',args.project.name,'CUDA'if args.cuda else'CPU',largest['counts_xyz'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=22)
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args();args.project=args.project.resolve();qualify(args)
