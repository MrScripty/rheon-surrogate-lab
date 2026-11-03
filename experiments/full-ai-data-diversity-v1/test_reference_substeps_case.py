"""Reference-only full-duration replay of a known sampled-control stability failure."""
import argparse
from pathlib import Path
import os
import time
from support import ROOT, read, write, digest
from macro_native_client import MacroNativeClient
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh reference stability comparison required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Registered reference case changed: '+p)
    qualification=read(ROOT/reg['qualification'])
    if qualification['status']!='passed_reference_substep_clock_and_forcing':
        raise ValueError('Actual fine-clock qualification required')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True);results=[]
    for factor in reg['substeps']:
        scene=dict(reg['scene'],reference_substeps=factor)
        client=MacroNativeClient(scene,args.output/f'substeps-{factor}')
        started=time.perf_counter();frames=[];passed=True;endpoint=None
        try:
            for step in range(1,scene['steps']+1):
                frame=client.advance();frames.append(frame)
                if not frame['frame']['advanced']:
                    passed=False;break
                if frame['frame']['step']!=step:raise ValueError('Actual macro advance mismatch')
            if passed:
                packet,fields=client.inspect()
                if packet['step']!=240 or packet['native_step']!=240*factor or packet['time_s']!=4.:
                    raise ValueError('Complete physical interval and fine-step count required')
                endpoint=args.output/f'endpoint-substeps-{factor}.npz'
                np.savez_compressed(endpoint,**fields)
        finally:owner=client.close()
        if not owner['reaped']or owner['returncode']!=0 or not owner['graceful_protocol_complete']:
            raise ValueError('Every stability trial must reap its reference child')
        if factor==1 and (passed or len(frames)!=reg['baseline_rejection_macro_step']):
            raise ValueError('Exact previously rejected macro-step boundary must reproduce')
        row=dict(substeps=factor,macro_timestep_s=1/60,reference_timestep_s=1/(60*factor),
            complete_240_macro_steps=passed,accepted_macro_steps=sum(f['frame']['advanced']for f in frames),
            accepted_fine_steps=sum(sum(v['frame']['advanced']for v in f['fine_frames'])for f in frames),
            worst_courant=max(f['frame']['quality']['courant']for f in frames),
            worst_actual_divergence=max(f['frame']['quality']['actual_divergence']for f in frames),
            last_frame=frames[-1],native_process=owner,elapsed_s=time.perf_counter()-started,
            endpoint_sha256=digest(endpoint)if endpoint else None,
            frames=frames,reference_only=True,model_target_pass=False)
        results.append(row)
        write(args.output/'results.json',dict(status='completed_reference_stability_trials'if factor==reg['substeps'][-1]else'running',
            identifier=reg['identifier'],results=results,source_sha256=reg['source_sha256'],
            protocol_sha256=digest(args.protocol),optimizer_updates=0,
            continuous_high_frequency_forcing_qualified=False,model_target_pass=False))
        print('Reference substeps',factor,'full240',passed,'accepted macro',row['accepted_macro_steps'],
              'max Courant',row['worst_courant'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
