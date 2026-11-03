"""Qualify numerical/C107 event controls and deferred checkpoint diagnostics."""
import argparse
import os
from pathlib import Path
import signal
import torch
from native_application_episode import episode
from application_episode_metrics import evaluate_native
from evaluate_coupled import references
from native_client import FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha,load_checkpoint


def run(args):
    if args.output.exists():raise ValueError('Fresh native application control qualification required')
    protocol_path=DOCS/'native-application-controls-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered application control source changed')
    for relative,digest in protocol['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified control artifact changed')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    def stop(*_):raise InterruptedError('Control qualification interrupted; reaping native child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state=load_checkpoint('C107');norm=read(BASE/'temporal-training-view-02/normalization.json');rows=[]
    for scenario,bank_path in (('steady',BASE/'temporal-development-240-01/bank.json'),
                               ('forcing',BASE/'application-event-reference-01/bank.json')):
        if sha(bank_path)!=protocol['bank_sha256'][scenario]:raise ValueError('Fixed control references changed')
        bank=read(bank_path);row=next(r for r in bank['trajectories']if r.get('original_identifier',r['identifier'])==protocol['scene'])
        if row['split']!='development':raise ValueError('Consumed development control scenes only')
        scene,truth=references(row);initial={n:truth[0][n]for n in FIELDS}
        for method in ('native-pcg','spectral','C107'):
            called=[]
            def predict(sample):
                called.append(tuple(sample['rhs'].shape))
                return state.model.predict_pressure(state.network,sample,state.normalization)
            result,outputs=episode(method,predict if method=='C107'else None,scene,initial,
                args.output/(scenario+'-'+method),steps=48,scenario=scenario)
            diagnostics=evaluate_native(result,outputs,truth,scene,norm)
            if method=='C107':
                if not called or result['completed_physical_interval']or result['physical_steps']!=0 or result['native_failure']is None:
                    raise ValueError('Actual C107 dispatch and unchanged first-step rejection required')
            elif not result['completed_physical_interval']or result['physical_steps']!=48 or not diagnostics['diagnostic_accuracy_pass']:
                raise ValueError('Numerical controls must complete the full interval and match declared reference checkpoints')
            write(args.output/(scenario+'-'+method)/'receipt.json',dict(result,diagnostics=diagnostics,
                performance_claim=False,role='Control qualification only; no candidate inference component'))
            rows.append(dict(method=method,scenario=scenario,physical_steps=result['physical_steps'],
                completed_physical_interval=result['completed_physical_interval'],diagnostic_accuracy_pass=diagnostics['diagnostic_accuracy_pass'],
                native_gates_unchanged=True,C107_callback_calls=len(called),native_processes=result['native_processes']))
            print('Qualified native application control:',scenario,method,'accepted',result['physical_steps'],'of48',flush=True)
    if torch.cuda.is_initialized():raise ValueError('CPU controls must not initialize CUDA')
    write(args.output/'receipt.json',dict(status='passed',rows=rows,source_sha256=protocol['source_sha256'],
        protocol_sha256=sha(protocol_path),reference_banks=protocol['bank_sha256'],
        numerical_methods_are_controls_only=True,native_gates_unchanged=True,fallback=False,
        optimizer_updates=0,performance_claim=False,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
