"""Check declared deployments and deferred metrics on actual trained bundles."""
import argparse
import os
from pathlib import Path
import signal
import torch
import temporal_displacement_objective as teaching
from application_model_support import load_model
from forecast_application_episode import episode
from application_episode_metrics import evaluate_forecast
from evaluate_coupled import references
from native_client import FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh deployment-interface qualification required')
    protocol_path=DOCS/'application-deployment-interface-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered deployment/evaluator source changed')
    registry_path=ROOT/protocol['registry']
    if sha(registry_path)!=protocol['registry_sha256']:raise ValueError('Declared actual trained checkpoints changed')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    def stop(*_):raise InterruptedError('Deployment qualification interrupted; reaping native consumer')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached application inference')
    teaching.displacement_labels=forbidden
    cases=[]
    for scenario,bank_path in (('steady',BASE/'temporal-development-240-01/bank.json'),
                               ('forcing',BASE/'application-event-reference-01/bank.json')):
        if sha(bank_path)!=protocol['bank_sha256'][scenario]:raise ValueError('Fixed application references changed')
        row=next(r for r in read(bank_path)['trajectories']if r.get('original_identifier',r['identifier'])==protocol['scene'])
        if row['split']!='development':raise ValueError('Consumed application scene required')
        scene,truth=references(row);cases.append((scenario,scene,truth))
    rows=[]
    for spec in read(registry_path)['models']:
        network,norm,identity=load_model(spec)
        for scenario,scene,truth in cases:
            directory=args.output/(spec['identifier']+'-'+scenario)
            result,outputs=episode(network,norm,scene,{n:truth[0][n]for n in FIELDS},directory,steps=48,scenario=scenario)
            diagnostics=evaluate_forecast(result,outputs,truth,scene,norm)
            if not result['completed_physical_interval']or not diagnostics['complete']or len(diagnostics['rows'])!=48:
                raise ValueError('Every accepted physical output must receive deferred metrics')
            if not diagnostics['accuracy_pass']:raise ValueError('Prototype deployment misses the original research accuracy criterion')
            write(directory/'receipt.json',dict(result,model=identity,diagnostics=diagnostics,
                performance_claim=False,accuracy_scope='One declared development scene through48steps; not all8/own240qualification'))
            rows.append(dict(identifier=spec['identifier'],trained_horizons=spec['horizons'],policy=spec['policy'],scenario=scenario,
                physical_steps=48,all48_outputs_scored=True,worst=diagnostics['worst'],native_processes=result['native_processes'],
                zero_native_numerical_steps=result['native_numerical_steps_executed']==0))
            print('Qualified declared deployment and deferred metrics:',spec['identifier'],scenario,flush=True)
    if torch.cuda.is_initialized():raise ValueError('Application deployment must remain on CPU')
    write(args.output/'receipt.json',dict(status='passed',rows=rows,source_sha256=protocol['source_sha256'],
        protocol_sha256=sha(protocol_path),registry_sha256=sha(registry_path),optimizer_updates=0,
        actual_independently_trained_horizons=[2,3,4],accuracy_scope='One development scene,steady/event48only',
        full_eight_scene_own240_accuracy_claim=False,performance_claim=False,application_physical_acceptance_claim=False,
        reference_states_only_in_deferred_metrics=True,native_numerical_steps_in_forecast=0,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
