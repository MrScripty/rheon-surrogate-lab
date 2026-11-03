"""Real native comparisons for forcing, legacy compatibility and state restart."""
import argparse
from pathlib import Path
import os
from support import ROOT, BASE, read, write, digest
from native_forcing_client import NativeForcingClient
from bounded_native_client import BoundedNativeClient
from force_program import make_scene, PROFILES, MOTIONS
from scheduled_force import force
from recurrent_sampler_native import dimensions
import numpy as np
import torch


def same(a,b):
    return set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a)


def run(args):
    if args.output.exists():raise ValueError('Fresh force/capture qualification required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered forcing evidence changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed']);args.output.mkdir(parents=True)
    rows=[];custody=[];serial=0
    def accepted(client):
        frame=client.advance()
        if not frame['frame']['advanced']:
            raise ValueError('Actual native physical advance required')
        return frame
    def close(client):
        value=client.close();custody.append(value)
        if not value['reaped'] or value['returncode']!=0 or not value['graceful_protocol_complete']:
            raise ValueError('Every native child must complete and reap')
    with torch.inference_mode():
        for density in reg['densities']:
            for profile in PROFILES:
                for motion in MOTIONS:
                    for counts in reg['counts_xyz']:
                        scene=make_scene(reg['seed']+serial,counts,density,profile,motion,steps=8)
                        serial+=1
                        client=NativeForcingClient(scene,args.output/f'force-{serial:03d}')
                        try:
                            _,before=client.inspect()
                            for step in (0,1,7):
                                increments,_=force(scene,step)
                                bases=[('zero',[torch.zeros(s,dtype=torch.float32)for s in dimensions(counts)[:3]]),
                                    ('random',[torch.randn(s,dtype=torch.float32)*.01 for s in dimensions(counts)[:3]]),
                                    ('canceling',[(-v).float()for v in increments])]
                                for kind,base in bases:
                                    _,merged=force(scene,step,base)
                                    actual=client.preview_force(step,[v.numpy()for v in base])
                                    if any(a.tobytes()!=v.numpy().tobytes()for a,v in zip(actual,merged)):
                                        raise ValueError('Known forcing differs from native: '+str((density,profile,motion,counts,step,kind)))
                                    rows.append(dict(cells_per_metre=density,counts_xyz=counts,
                                        profile=profile,motion=motion,step=step,base=kind,
                                        all_staggered_fields_bitwise_equal=True))
                            _,after=client.inspect()
                            if not same(before,after):raise ValueError('Force preview changed physical state')
                        finally:close(client)
        legacy=[]
        for density in reg['densities']:
            scene=make_scene(2026102039,[8,7,5],density,'continuous','stationary',steps=8)
            scene={k:v for k,v in scene.items()if k not in ('forcing_program','forcing_schedule')}
            old=BoundedNativeClient(scene,args.output/f'legacy-old-{density}')
            new=NativeForcingClient(scene,args.output/f'legacy-new-{density}')
            try:
                for step in range(5):
                    if step:accepted(old);accepted(new)
                    if not same(old.inspect()[1],new.inspect()[1]):raise ValueError('Optional force extension changed legacy states')
                legacy.append(dict(cells_per_metre=density,complete_states=5,all_fields_bitwise_equal=True))
            finally:close(old);close(new)
        restarts=[]
        for index,profile in enumerate(PROFILES):
            scene=make_scene(2026102047+index,[8,7,5],(10,25,50,100)[index],profile,MOTIONS[index%3],steps=8)
            original=NativeForcingClient(scene,args.output/f'restart-original-{index}')
            restored=None
            try:
                for _ in range(4):accepted(original)
                packet,state=original.inspect()
                if packet['step']!=4:raise ValueError('Actual restart prefix must reach step4')
                restored=NativeForcingClient(scene,args.output/f'restart-restored-{index}')
                restored.restore(packet['step'],state)
                for _ in range(4):
                    accepted(original);accepted(restored)
                    if not same(original.inspect()[1],restored.inspect()[1]):raise ValueError('Scheduled force lost actual time at restart')
                if original.inspect()[0]['step']!=8 or restored.inspect()[0]['step']!=8:
                    raise ValueError('Both restored histories must reach actual step8')
                restarts.append(dict(profile=profile,own_future_steps=4,all_fields_bitwise_equal=True))
            finally:
                close(original)
                if restored is not None:close(restored)
    if len(rows)!=reg['expected_force_comparisons'] or torch.cuda.is_initialized():
        raise ValueError('Qualification population or device changed')
    write(args.output/'qualification.json',dict(status='passed_native_forcing_and_capture_contract',
        force_comparisons=rows,legacy=legacy,restarts=restarts,native_processes=custody,
        all_staggered_forcing_bytes_exact=True,optimizer_updates=0,
        model_target_pass=False,reference_capture_only=True,
        source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol)))
    print('Qualified',len(rows),'exact forcing comparisons, legacy compatibility and scheduled restart',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
