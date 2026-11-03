"""Same-scene numerical and native C107-pressure application controls without fallback."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint
from native_client import NativeClient,FIELDS
from temporal_metrics import metrics


def run(args):
    if args.output.exists():raise ValueError('Fresh native control episodes required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23})
    setup=time.perf_counter();state=load_checkpoint('C107');state.torch.set_num_threads(1)
    model_setup=time.perf_counter()-setup
    norm=read(BASE/'temporal-training-view-02/normalization.json');bank=read(args.bank)
    episodes=[]
    for repeat in range(3):
        methods=['native-pcg','spectral','C107'];methods=methods[repeat:]+methods[:repeat]
        for method in methods:
            for row in bank['trajectories']:
                if row['split']!='development':raise ValueError('Consumed development controls only')
                receipt_path=ROOT/row['receipt']
                if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Reference receipt changed')
                receipt=read(receipt_path);scene=receipt['scene'];directory=args.output/f'r{repeat}-{method}-{row["identifier"]}'
                started=time.perf_counter();client=NativeClient(scene,directory);native_setup=time.perf_counter()-started
                clocks=[];frames=[];diagnostics=[];validation_seconds=0.;failed=None
                with np.load(receipt_path.parent/receipt['states'][0]['path'],allow_pickle=False)as source:
                    initial_mass=float(source['tracer'].astype('f8').sum()/scene['cells_per_metre']**3)
                try:
                    for step in range(1,241):
                        tick=time.perf_counter()
                        frame=client.advance(method=method,predict=(lambda sample:state.model.predict_pressure(state.network,sample,state.normalization))if method=='C107'else None)
                        elapsed=time.perf_counter()-tick;clocks.append(elapsed);frames.append(frame)
                        if not frame['frame']['advanced']:
                            failed=dict(attempted_step=step,native_frame=frame);break
                        if step in (12,48,240):
                            tick=time.perf_counter();packet,physical=client.inspect()
                            spec=receipt['states'][step];source=receipt_path.parent/spec['path']
                            if sha(source)!=spec['sha256']:raise ValueError('Reference state changed')
                            with np.load(source,allow_pickle=False)as arrays:reference={name:arrays[name].copy()for name in (*FIELDS,'rhs')}
                            if packet['step']!=step:raise ValueError('Native control timestamp differs')
                            value=metrics(physical,reference,reference['rhs'],scene,norm,initial_mass)
                            diagnostics.append(dict(step=step,metrics=value));validation_seconds+=time.perf_counter()-tick
                    custody=client.close()
                finally:
                    if client.child.poll()is None:client.close()
                accepted_steps=sum(bool(frame['frame']['advanced'])for frame in frames)
                result=dict(status='completed',method=method,repeat=repeat,identifier=row['identifier'],counts_xyz=scene['counts'],
                    cells_per_metre=scene['cells_per_metre'],physical_steps_requested=240,accepted_steps=accepted_steps,
                    completed_physical_interval=accepted_steps==240,failure=failed,per_attempt_seconds=clocks,
                    application_seconds=sum(clocks),application_seconds_per_physical_second=(sum(clocks)/(240/60)if accepted_steps==240 else None),
                    native_setup_seconds=native_setup,model_setup_seconds=model_setup if method=='C107'else 0.,
                    offline_export_and_validation_seconds=validation_seconds,diagnostics=diagnostics,native_process=custody,
                    fallback=False,native_acceptance_gates_unchanged=True,source_sha256=sha(__file__))
                write(directory/'receipt.json',result);episodes.append({k:v for k,v in result.items()if k not in ('per_attempt_seconds','failure','diagnostics')})
                write(args.output/'progress.json',dict(status='running',episodes=episodes))
                print('Native control',method,row['identifier'],'repeat',repeat,'accepted',accepted_steps,'of240',flush=True)
    write(args.output/'summary.json',dict(status='completed',episodes=episodes,bank_sha256=sha(args.bank),
        source_sha256=sha(__file__),candidate_C107_entry=state.entry,
        timing_scope='Actual native advances, pressure transport, solver or model work and native physical gates; setup and offline exported comparisons separate',
        cpu=23,threads=1,cuda_initialized=state.torch.cuda.is_initialized(),model_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
