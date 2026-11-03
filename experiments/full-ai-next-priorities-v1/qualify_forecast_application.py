"""Qualify real neural/native handoff and event accounting without speed claims."""
import argparse
import os
from pathlib import Path
import signal
import numpy as np
import torch
import temporal_displacement_objective as teaching
from fixed_viscosity_evaluation_support import load_model
from temporal_train_fixed_viscosity import inventory
from forecast_application_episode import episode
from native_client import FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh application qualification required')
    protocol_path=DOCS/'forecast-application-qualification-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered application source changed')
    transport=read(BASE/'bounded-native-client-qualification-01/receipt.json')
    if transport['status']!='passed'or sha(BASE/'bounded-native-client-qualification-01/receipt.json')!=protocol['transport_qualification_sha256']:
        raise ValueError('Exact bounded native transport/restart qualification required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    def stop(*_):raise InterruptedError('Application qualification interrupted; reaping native consumer')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    pair=BASE/'b2-fixed-viscosity-pair-01';network,norm,identity=load_model(pair);before=inventory(network)
    bank_path=BASE/'temporal-development-240-01/bank.json'
    if sha(bank_path)!=protocol['bank_sha256']:raise ValueError('Application qualification scene changed')
    row=next(r for r in read(bank_path)['trajectories']if r['identifier']==protocol['scene'])
    path=ROOT/row['receipt']
    if row['split']!='development'or sha(path)!=row['receipt_sha256']:raise ValueError('Consumed scene receipt required')
    receipt=read(path);spec=receipt['states'][0];source=path.parent/spec['path']
    if sha(source)!=spec['sha256']:raise ValueError('Causal initial native state changed')
    with np.load(source,allow_pickle=False)as a:initial={n:a[n].copy()for n in FIELDS}
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached application inference')
    teaching.displacement_labels=forbidden
    rows=[]
    try:
        for k in (1,2,3,4):
            network.horizons=k
            for scenario in ('steady','forcing','unsupported_boundary'):
                result,states=episode(network,norm,receipt['scene'],initial,args.output/f'k{k}-{scenario}',steps=48,scenario=scenario)
                expected=13 if scenario=='unsupported_boundary'else 48
                if result['physical_steps']!=expected or len(states)!=expected:
                    raise ValueError('Application exposes an incorrect physical-state interval')
                if scenario!='steady'and result['event_discarded_states']!=k-1:
                    raise ValueError('Every unused pre-event state must be invalidated')
                if scenario=='unsupported_boundary'and(result['completed_physical_interval']or result['first_unavailable_step']!=14):
                    raise ValueError('Unsupported boundary continuation must not fabricate future states')
                write(args.output/f'k{k}-{scenario}'/'receipt.json',dict(result,model=identity,
                    qualification_unroll_horizons=k,trained_horizons=2,actual_independent_K_fit=False,
                    performance_claim=False,accuracy_claim=False))
                rows.append(dict(horizons=k,scenario=scenario,physical_steps=expected,
                    event_discarded_states=result['event_discarded_states'],endpoint_discarded_states=result['endpoint_discarded_states'],
                    network_calls=result['network_calls'],native_final_state_bytes_exact=True,native_processes=result['native_processes']))
                print('Qualified neural/native application mechanics:',k,scenario,flush=True)
    finally:network.horizons=2
    if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Qualification changed neural tensors or initialized CUDA')
    write(args.output/'receipt.json',dict(status='passed',rows=rows,model=identity,source_sha256=protocol['source_sha256'],
        protocol_sha256=sha(protocol_path),source_evidence={str(p.relative_to(ROOT)):sha(p)for p in
            (pair/'receipt.json',pair/'restore/weights.npz',path,source)},
        trained_horizons=2,diagnostic_horizon_views_only=True,no_independent_K1_K3_K4_fit_claim=True,
        optimizer_updates=0,performance_claim=False,accuracy_claim=False,application_physical_acceptance_claim=False,
        no_future_reference_input=True,native_numerical_steps_in_forecast=0,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
