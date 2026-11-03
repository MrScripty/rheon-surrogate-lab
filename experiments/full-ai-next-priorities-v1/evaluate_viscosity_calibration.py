"""Validate restored neural viscosity on native operators and physical updates."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from viscosity_calibration_model import ViscosityNet
from prepare_flux_teacher import fields
from viscosity_teacher_support_v2 import native,from_carrier,packed_output
from temporal_features_v2 import geometry
from managed_fit_artifacts import load_weights,inventory
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    args.pair=args.pair.resolve()
    if args.output.exists():raise ValueError('Fresh neural viscosity evaluation required')
    protocol_path=DOCS/'viscosity-calibration-registration-01.json';reg=read(protocol_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered viscosity source changed')
    pair=read(args.pair/'receipt.json');restored=read(args.pair/'restore/receipt.json');history=read(args.pair/'restore/history.json')
    if pair['status']!='completed'or not all(pair['artifact_parity'].values())or restored['optimizer_updates']!=0 or not restored['full_cpu_probe_bitwise_verified']:
        raise ValueError('Exact completed managed CPU restore required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);network=ViscosityNet()
    load_weights(network,args.pair/'restore/weights.npz',history['final_inventory']);network.eval();before=inventory(network)
    rng=np.random.Generator(np.random.PCG64(reg['evaluation_seed']));cases=[]
    for counts in reg['evaluation_counts_xyz']:
        for cpm in (8,16):
            raw,_=fields(rng,counts,1)
            quantized=(raw*16).round().clamp(-16,16)/16
            quantized=torch.where(quantized==0,torch.zeros_like(quantized),quantized)
            for pattern,value in (('zero',torch.zeros_like(raw)),('dyadic',quantized),('physical',.01*raw)):
                dyadic=pattern=='dyadic';case=from_carrier(value,counts,cpm,density=1. if dyadic else 1000.,mu=1. if dyadic else .001,
                    dt=(1/cpm)**2/64 if dyadic else 1/60)
                cases.append((case,value,dict(counts_xyz=counts,cpm=cpm,pattern=pattern)))
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(cases),2):
            selected=cases[start:start+2]
            for (case,value,identity),reference in zip(selected,native([c[0]for c in selected])):
                counts=case['counts'];mask=geometry(tuple(counts),'cpu')[1][1:4].unsqueeze(0).double()
                target=packed_output(reference,counts).double();predicted_operator=network(value).double()*mask
                output=(value.double()+reference['beta']*predicted_operator).float().double()*mask
                absolute=(output-target)*mask;relative=float(absolute.norm())/max(float((target*mask).norm()),1e-30)
                operator_error=None
                if identity['pattern']=='dyadic':
                    truth=((target-value.double())/reference['beta'])*mask
                    operator_error=float(((predicted_operator-truth)*mask).norm())/max(float(truth.norm()),1e-30)
                zero_exact=bool((output==0).all())if identity['pattern']=='zero'else None
                passed=relative<=reg['published_tolerance']and (operator_error is None or operator_error<=reg['published_tolerance'])and zero_exact is not False
                rows.append(dict(**identity,updated_velocity_relative_error=relative,operator_relative_error=operator_error,zero_exact=zero_exact,pass_tolerance=passed))
    if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Native evaluation changed weights or initialized CUDA')
    passed=bool(history['coefficient_accuracy_pass']and all(r['pass_tolerance']for r in rows))
    paths=[args.pair/'receipt.json',args.pair/'restore/receipt.json',args.pair/'restore/history.json',args.pair/'restore/weights.npz',protocol_path,
        BASE/'viscosity-teacher-build-02/build-receipt.json']
    write(args.output,dict(status='passed_component_qualification'if passed else'failed_component_qualification',rows=rows,
        maximum_updated_velocity_relative_error=max(r['updated_velocity_relative_error']for r in rows),
        maximum_operator_relative_error=max(r['operator_relative_error']for r in rows if r['operator_relative_error']is not None),
        parameters=243,optimizer_updates=0,managed_training_updates=1000,source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},
        source_sha256=reg['source_sha256'],native_viscosity_in_candidate_inference=False,analytic_coefficient_assignment=False,
        CPU_latency_claim=False,full_state_accuracy_claim=False,reserved_scenes_consumed=False))
    print('Neural viscosity restored-component native check:',passed,len(rows),'cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
