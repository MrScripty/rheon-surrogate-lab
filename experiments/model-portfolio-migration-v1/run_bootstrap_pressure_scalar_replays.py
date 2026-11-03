"""Four fresh processes for the two original selected GroupNorm checkpoints."""
import os
from pathlib import Path
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require


def run():
    path=RUN/'bootstrap-pressure-CUDA-protocol-01.json';protocol=read(path)
    folder=RUN/'bootstrap-pressure-CUDA-owner-01';require(not folder.exists(),'Fresh owner required')
    folder.mkdir()
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    runtime=read(ROOT/protocol['runtime_manifest'])
    first,second=protocol['fit_ids'];order=[(first,0),(second,0),(second,1),(first,1)]
    state=dict(status='running_bootstrap_pressure_CUDA_workers',pid=os.getpid(),created_unix=time.time(),
        protocol=binding(path),producer=binding(Path(__file__)),pending=[dict(fit_item_id=fid,phase=phase) for fid,phase in order],
        active_child=None,qualified=[],new_parameter_fits=0,prediction_arrays_saved=False,latency_qualified=False)
    save(folder/'status.json',state)
    child=None
    try:
        for fid,phase in order:
            with (folder/(fid+'-phase'+str(phase)+'.log')).open('xb') as log:
                child=subprocess.Popen([runtime['python'],'-B',str(ROOT/protocol['worker']),'--protocol',str(path),
                    '--fit-id',fid,'--phase',str(phase)],cwd=ROOT,env=environment(runtime,'cuda'),stdout=log,stderr=subprocess.STDOUT)
                state['active_child']=dict(fit_item_id=fid,phase=phase,pid=child.pid,started_unix=time.time())
                save(folder/'status.json',state);code=child.wait(timeout=protocol['worker_deadline_seconds']+120)
            require(code==0,'Bootstrap worker failed: '+fid+' phase'+str(phase)+' exit'+str(code))
            result_path=ROOT/protocol['output_root']/(fid+'-phase'+str(phase))/'result.json';result=read(result_path)
            require(result['status']=='bootstrap_pressure_CUDA_all_applicable_cases_qualified' and result['cases']==770,
                'Complete guarded bootstrap qualification required')
            state['qualified'].append(dict(fit_item_id=fid,phase=phase,result=binding(result_path)))
            state['pending'].remove(dict(fit_item_id=fid,phase=phase));state['active_child']=None
            save(folder/'status.json',state);print(fid,'phase',phase,'770requests qualified',flush=True)
        state.update(status='all_four_bootstrap_pressure_CUDA_workers_terminal',finished_unix=time.time())
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=30)
        state.update(status='bootstrap_pressure_owner_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:save(folder/'status.json',state)


if __name__=='__main__':run()
