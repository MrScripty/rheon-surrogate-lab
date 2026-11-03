"""Qualify fine forcing, unchanged default results and exact macro-state restart."""
import argparse
from pathlib import Path
import os
from support import ROOT, read, write, digest
from native_forcing_client import NativeForcingClient
from macro_native_client import MacroNativeClient
from scheduled_force_substeps import force
from force_program import make_scene, PROFILES, MOTIONS
from recurrent_sampler_native import dimensions
import torch


def equal(a,b):
    return set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes()for k in a)


def run(args):
    if args.output.exists():raise ValueError('Fresh reference clock qualification required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Registered reference source changed: '+p)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed']);args.output.mkdir(parents=True)
    rows=[];owners=[];legacy=[];restarts=[];serial=0
    def close(client):
        result=client.close();owners.append(result)
        if not result['reaped']or result['returncode']!=0 or not result['graceful_protocol_complete']:
            raise ValueError('Every native reference child must complete and reap')
    def advance(client):
        frame=client.advance()
        if not frame['frame']['advanced']or not frame['frame']['quality']['qualified']:
            raise ValueError('Actual qualified physical advance required')
        if len(frame.get('fine_frames',[frame]))!=getattr(client,'reference_substeps',1):
            raise ValueError('Every actual internal reference step required')
    with torch.inference_mode():
        for factor in reg['substeps']:
            for density in reg['densities']:
                for profile in PROFILES:
                    for motion in MOTIONS:
                        counts=[5,7,9]
                        scene=make_scene(reg['seed']+serial,counts,density,profile,motion,8)
                        scene['reference_substeps']=factor;serial+=1
                        client=MacroNativeClient(scene,args.output/f'force-{serial:03d}')
                        try:
                            before_packet,before=client.inspect()
                            for step in sorted({0,factor-1,factor,8*factor-1}):
                                increments,_=force(scene,step)
                                fixtures=[('zero',[torch.zeros(s,dtype=torch.float32)for s in dimensions(counts)[:3]]),
                                    ('random',[torch.randn(s,dtype=torch.float32)*.01 for s in dimensions(counts)[:3]]),
                                    ('canceling',[(-v).float()for v in increments])]
                                for kind,base in fixtures:
                                    _,merged=force(scene,step,base)
                                    expected=client.preview_force(step,[v.numpy()for v in base])
                                    if any(a.tobytes()!=b.numpy().tobytes()for a,b in zip(expected,merged)):
                                        raise ValueError('Fine prescribed forcing changed bytes: '+str((factor,density,profile,motion,step,kind)))
                                    rows.append(dict(substeps=factor,cells_per_metre=density,counts_xyz=counts,
                                        profile=profile,motion=motion,native_step=step,base=kind,all_fields_bitwise_equal=True))
                            after_packet,after=client.inspect()
                            if before_packet!=after_packet or not equal(before,after):
                                raise ValueError('Read-only force preview changed physical state or clock')
                        finally:close(client)
            print('Qualified prescribed forcing at internal factor',factor,flush=True)
        for density in reg['densities']:
            scene=make_scene(2026102039,[8,7,5],density,'continuous','stationary',8)
            scene={k:v for k,v in scene.items()if k not in ('forcing_program','forcing_schedule')}
            old=NativeForcingClient(scene,args.output/f'legacy-old-{density}')
            new=MacroNativeClient(scene,args.output/f'legacy-new-{density}')
            try:
                for step in range(5):
                    if step:advance(old);advance(new)
                    op,oa=old.inspect();np,na=new.inspect()
                    if not equal(oa,na)or op['step']!=np['step']or op['time_s']!=np['time_s']:
                        raise ValueError('Default reference clock changed legacy physical results')
                legacy.append(dict(cells_per_metre=density,complete_states=5,all_fields_bitwise_equal=True))
            finally:close(old);close(new)
        for factor in reg['substeps']:
            for index,profile in enumerate(PROFILES):
                scene=make_scene(2026102047+index,[8,7,5],(10,25,50,100)[index],profile,MOTIONS[index%3],8)
                scene['reference_substeps']=factor
                original=MacroNativeClient(scene,args.output/f'restart-original-{factor}-{index}')
                restored=None
                try:
                    for _ in range(4):advance(original)
                    packet,state=original.inspect()
                    if packet['step']!=4 or packet['native_step']!=4*factor or packet['time_s']!=4/60:
                        raise ValueError('Actual macro/fine restart prefix mismatch')
                    restored=MacroNativeClient(scene,args.output/f'restart-restored-{factor}-{index}')
                    restored.restore(4,state)
                    for step in range(5,9):
                        advance(original);advance(restored)
                        op,oa=original.inspect();rp,ra=restored.inspect()
                        if not equal(oa,ra)or op!=rp or op['step']!=step or op['native_step']!=step*factor:
                            raise ValueError('Fine reference restart changed complete macro states or time')
                    restarts.append(dict(substeps=factor,profile=profile,
                        complete_future_macro_states=4,actual_future_native_steps=4*factor,all_fields_bitwise_equal=True))
                finally:
                    close(original)
                    if restored is not None:close(restored)
    if len(rows)!=reg['expected_force_comparisons']or torch.cuda.is_initialized():
        raise ValueError('Reference qualification population or device changed')
    write(args.output/'qualification.json',dict(status='passed_reference_substep_clock_and_forcing',
        force_comparisons=rows,legacy=legacy,restarts=restarts,native_processes=owners,
        every_internal_reference_step_counted=True,macro_output_timestep_s=1/60,
        forcing_semantics='Sampled macro controls; continuous high-frequency program contract remains separate',
        model_target_pass=False,optimizer_updates=0,source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol)))
    print('Qualified',len(rows),'exact fine-force cases, default compatibility and macro restart',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
