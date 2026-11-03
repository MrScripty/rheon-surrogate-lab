"""Bounded two-worker owner; all four task jobs retain scalars only."""
import os
from pathlib import Path
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require


def run():
    protocol_path=RUN/'recent-component-scalar-replay-protocol-01.json';protocol=read(protocol_path)
    folder=RUN/'recent-component-scalar-replay-owner-01';require(not folder.exists(),'Fresh scalar owner required')
    folder.mkdir()
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    runtime=read(ROOT/protocol['runtime_manifest'])
    pending=list(protocol['jobs']);active={}
    state=dict(status='running_recent_component_scalar_replays',pid=os.getpid(),created_unix=time.time(),
        protocol=binding(protocol_path),producer=binding(Path(__file__)),pending=[j['name'] for j in pending],active=[],qualified=[],
        maximum_concurrent_workers=2,prediction_arrays_saved=False,primary_timing_samples=0,latency_qualified=False)
    save(folder/'status.json',state)
    try:
        while pending or active:
            while pending and len(active)<2:
                job=pending.pop(0);log=(folder/(job['name']+'.log')).open('xb')
                child=subprocess.Popen([runtime['python'],'-B',str(ROOT/protocol['worker']),'--protocol',str(protocol_path),'--job',job['name']],
                    cwd=ROOT,env=environment(runtime,'cuda'),stdout=log,stderr=subprocess.STDOUT)
                active[job['name']]=dict(child=child,log=log,job=job,started=time.monotonic())
                state['pending'].remove(job['name'])
            state['active']=[dict(name=name,pid=value['child'].pid) for name,value in active.items()]
            save(folder/'status.json',state)
            for name,task in list(active.items()):
                code=task['child'].poll()
                require(time.monotonic()-task['started']<protocol['worker_deadline_seconds']+120,'Scalar job timeout: '+name)
                if code is None:continue
                task['log'].close();require(code==0,'Scalar worker failed: '+name+' exit'+str(code))
                path=ROOT/protocol['output_root']/name/'result.json';value=read(path)
                require(value['status']=='all_original_component_outputs_exact_expanded_scalars_retained'
                    and value['all_original_output_hashes_exact'] and value['all_original_accepted_metrics_exact'],
                    'Every accepted output andmetric must remain exact')
                state['qualified'].append(dict(name=name,result=binding(path),records=sum(p['records'] for p in value['scalar_populations'].values())))
                del active[name];print(name,'qualified',flush=True)
            time.sleep(.25)
        state.update(status='all_four_recent_component_scalar_jobs_terminal',active=[],finished_unix=time.time())
        require(sum(q['records'] for q in state['qualified'])==17648,'Complete eightfit output population required')
    except BaseException as error:
        state.update(status='recent_component_scalar_owner_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        for task in active.values():
            if task['child'].poll() is None:task['child'].terminate();task['child'].wait(timeout=30)
            task['log'].close()
        raise
    finally:save(folder/'status.json',state)


if __name__=='__main__':run()
