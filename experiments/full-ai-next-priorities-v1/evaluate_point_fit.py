"""Check managed learned coefficients against native points on target geometries."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from point_sampling_model import PointSamplingNet
from prepare_flux_teacher import fields
from transport_teacher_support import native,packed_native_points,point_mask
from managed_fit_artifacts import inventory,load_weights
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh restored-point evaluation required')
    protocol_path=DOCS/'point-sampling-evaluation-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered point evaluator source changed')
    pair=read(args.pair/'receipt.json');receipt=read(args.pair/'restore/receipt.json')
    history=read(args.pair/'restore/history.json')
    if pair['status']!='completed'or not all(pair['artifact_parity'].values())or receipt['optimizer_updates']!=0:
        raise ValueError('Completed exact managed CPU restore required')
    if receipt['originating_training_run']['run_id']!=pair['training_run_id']or not receipt['full_cpu_probe_bitwise_verified']:
        raise ValueError('Restore provenance differs')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    network=PointSamplingNet()
    load_weights(network,args.pair/'restore/weights.npz',history['final_inventory']);network.eval()
    initial=inventory(network);rng=np.random.Generator(np.random.PCG64(protocol['data_seed']));cases=[]
    for counts in protocol['counts_xyz']:
        for cpm in (8,16):
            for amplitude in (0.,.01,1.):
                value,physical=fields(rng,counts,1);value=value*amplitude
                physical=[v*amplitude for v in physical]
                cases.append((value,dict(counts=counts,cell_size_m=1/cpm,dt_s=1/60,point_only=True,
                    velocity=[v[0].astype('f8').tolist()for v in physical],tracer=[0.]*int(np.prod(counts))),
                    dict(counts_xyz=counts,cells_per_metre=cpm,amplitude=amplitude)))
    rows=[]
    for start in range(0,len(cases),2):
        selected=cases[start:start+2];references=native([c[1]for c in selected])
        for (value,case,identity),reference in zip(selected,references):
            target=-packed_native_points(reference,case['counts'],'velocity_samples').reshape(1,12,*value.shape[-3:])
            mask=point_mask(case['counts']).expand(1,4,3,*value.shape[-3:]).reshape_as(target)
            with torch.inference_mode():prediction=network(value).double()
            difference=(prediction-target)*mask
            error=float(difference.norm())/max(float((target*mask).norm()),1e-30)
            zero_exact=not bool((prediction*mask).any())if identity['amplitude']==0 else None
            rows.append(dict(**identity,relative_error=error,zero_exact=zero_exact,
                pass_tolerance=error<=protocol['relative_tolerance']and zero_exact is not False))
    if inventory(network)!=initial or torch.cuda.is_initialized():raise ValueError('CPU evaluation changed weights or initialized CUDA')
    passing=bool(history['coefficient_accuracy_pass']and all(r['pass_tolerance']for r in rows))
    evidence={str(p.relative_to(ROOT)):sha(p)for p in (args.pair/'receipt.json',args.pair/'restore/receipt.json',
        args.pair/'restore/history.json',args.pair/'restore/weights.npz',protocol_path,BASE/'transport-teacher-build-01/build-receipt.json')}
    write(args.output,dict(status='passed_component_qualification'if passing else'failed_component_qualification',rows=rows,
        source_evidence=evidence,source_sha256=protocol['source_sha256'],maximum_native_relative_error=max(r['relative_error']for r in rows),
        parameters=972,optimizer_updates=0,managed_training_updates=pair['optimizer_updates'],
        learned_float32_coefficients_native_qualified=passing,no_analytic_coefficient_assignment=True,
        native_sampling_in_predictor=False,full_state_forecast=False,candidate_frontier_point=False,
        CPU_latency_claim=False,reserved_scenes_consumed=False))
    print('Learned point component:', 'passed'if passing else'failed',len(rows),'native cases; worst',max(r['relative_error']for r in rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
