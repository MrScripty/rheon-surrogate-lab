"""Stream native states without retaining trajectories; verify old typed labels."""
import argparse
import hashlib
import os
import resource
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from support import ROOT,read,write,digest
from continuous_reference_client import ContinuousReferenceClient
FIELDS=('pressure','vx','vy','vz','tracer')

def resident_bytes(pid):
    try:
        for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines():
            if line.startswith('VmHWM:'):return int(line.split()[1])*1024
    except FileNotFoundError:return 0
    return 0

def run(args):
    if args.output.exists():raise ValueError('Fresh streamed reference proof required')
    reg=read(args.protocol)
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen streamed native source changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True);began=time.monotonic();active=None
    result=dict(status='running',active=None,rows=[],native_processes=[],optimizer_updates=0,cpu_model_forward=False,
        candidate_inference_executed=False,native_reference_only=True,all_steps_streamed_without_restore=True,
        retained_physical_state_archives=0,expanded_reserved52_references_opened=0,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'])
    try:
        for case in reg['cases']:
            if shutil.disk_usage(args.output).free<reg['minimum_free_disk_bytes']:raise ValueError('Native streaming disk floor reached')
            if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Original native scene lineage changed')
            receipt=read(ROOT/case['receipt']);scene=dict(receipt['scene'],reference_substeps=case['reference_substeps'],forcing_control_substeps=1)
            specs={s['step']:s for s in case['states']};directory=args.output/case['identifier'];directory.mkdir()
            result['active']=dict(identifier=case['identifier'],step=0);write(args.output/'results.json',result)
            active=ContinuousReferenceClient(scene,directory,io_timeout_s=120.)
            try:
                for step in range(reg['steps']+1):
                    if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite native streaming qualification deadline')
                    if step:
                        frame=active.advance()
                        if not frame['frame']['advanced'] or frame['frame']['step']!=step:raise ValueError('Native physical reference rejected actual advance')
                    packet,values=active.inspect()
                    if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Native physical clock mismatch')
                    for name in FIELDS:
                        if values[name].dtype.str!=('<f8' if name=='pressure' else '<f4') or not np.isfinite(values[name]).all():raise ValueError('Exact finite reference field types required')
                    values['rhs']=np.zeros_like(values['pressure']) if step==0 else active.last_request['rhs']
                    values['native_input_residual']=np.zeros_like(values['pressure']) if step==0 else active.last_request['native_input_residual']
                    values['pressure_base']=np.zeros_like(values['pressure']) if step==0 else active.last_request['base']
                    matches=None
                    if step in specs:
                        spec=specs[step];path=ROOT/spec['path']
                        if digest(path)!=spec['sha256']:raise ValueError('Original archive changed')
                        with np.load(path,allow_pickle=False) as old:
                            matches={name:values[name].dtype==old[name].dtype and values[name].shape==old[name].shape and values[name].tobytes()==old[name].tobytes() for name in values}
                        if not all(matches.values()):raise ValueError('Streamed uninterrupted reference differs from previously qualified typed labels: '+str((case['identifier'],step,matches)))
                    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+resident_bytes(active.child.pid)
                    if peak>reg['maximum_combined_peak_rss_bytes']:raise ValueError('Native streamed reference memory ceiling reached')
                    result['rows'].append(dict(identifier=case['identifier'],split=case['split'],step=step,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],reference_substeps=case['reference_substeps'],
                        actual_native_internal_advances=step*case['reference_substeps'],old_field_byte_matches=matches,all_typed_fields_finite=True,
                        actual_field_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()},observed_combined_peak_rss_bytes=peak))
                    result['active']['step']=step
                custody=active.close();active=None
                if not custody['reaped'] or custody['returncode']!=0 or not custody['graceful_protocol_complete']:raise ValueError('Owned native child must exit normally and reap')
                result['native_processes'].append(dict(identifier=case['identifier'],**custody))
            finally:
                if active is not None:
                    result['native_processes'].append(dict(identifier=case['identifier'],**active.close()));active=None
            print('Native bounded streamed field proof',case['identifier'],reg['steps'],'outputs',flush=True)
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Native proof source changed during execution: '+path)
        result.update(status='passed_streamed_native_reference_typed_byte_and_custody_qualification',active=None,
            inspected_states=len(result['rows']),old_checkpoint_states_byte_verified=sum(r['old_field_byte_matches'] is not None for r in result['rows']),
            actual_internal_native_advances=sum(reg['steps']*r['reference_substeps'] for r in reg['cases']),
            all_native_children_reaped=True,all_native_children_returncode_zero=True,elapsed_diagnostic_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        if active is not None:result['native_processes'].append(active.close())
        result['active']=None;write(args.output/'results.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
