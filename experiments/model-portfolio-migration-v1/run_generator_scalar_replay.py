"""Sequential owner of35fresh scalar-only CUDA accuracy workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require


def run():
    protocol_path = RUN/'generator-scalar-replay-protocol-01.json'
    protocol = read(protocol_path)
    folder = RUN/'generator-scalar-replay-owner-01'
    require(not folder.exists(), 'Fresh worker owner required; no retry')
    folder.mkdir()
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    runtime = read(ROOT/protocol['runtime_manifest'])
    models = protocol['models']
    first = [next(m for m in models if m['constructor_kind'] == kind) for kind in ('plain_captured','inherited_CUDA')]
    order = first+[m for m in models if m not in first]
    state = dict(status='running_generator_scalar_replays',pid=os.getpid(),created_unix=time.time(),
        protocol=binding(protocol_path),producer=binding(Path(__file__)),pending=[m['fit_item_id'] for m in order],
        active_child=None,qualified=[],new_parameter_fits=0,new_reference_generation=0,prediction_arrays_saved=False,
        latency_qualified=False,primary_timing_samples=0,native_mutations=0)
    save(folder/'status.json',state)
    child = None
    try:
        for model in order:
            fid = model['fit_item_id']
            with (folder/(fid+'.log')).open('xb') as log:
                child = subprocess.Popen([runtime['python'],'-B',str(ROOT/protocol['worker']),
                    '--protocol',str(protocol_path),'--fit-id',fid],cwd=ROOT,env=environment(runtime,'cuda'),stdout=log,stderr=subprocess.STDOUT)
                state['active_child'] = dict(fit_item_id=fid,pid=child.pid,started_unix=time.time())
                save(folder/'status.json',state)
                code = child.wait(timeout=protocol['worker_deadline_seconds']+120)
            require(code == 0, 'Scalar replay child failed: '+fid+' exit '+str(code))
            result_path = ROOT/protocol['output_root']/fid/'result.json'
            value = read(result_path)
            require(value['status'] == 'all64_generator_cases_exact_original_outputs_expanded_scalars_retained'
                and value['all_original_physical_output_hashes_exact'] and value['all_original_metric_scalars_exact']
                and value['device_observation']['closed'] and not value['device_observation']['forbidden_forward_attempts'],
                'Exact fields/metrics and guarded CUDA lifecycle required')
            state['qualified'].append(dict(fit_item_id=fid,result=binding(result_path),physical_outputs=value['physical_outputs']))
            state['pending'].remove(fid)
            state['active_child'] = None
            save(folder/'status.json',state)
            print(len(state['qualified']),'/35',model['identifier'],value['physical_outputs'],'outputs qualified',flush=True)
        state.update(status='all35_generator_scalar_replays_terminal',finished_unix=time.time())
        require(sum(r['physical_outputs'] for r in state['qualified']) == 5184, 'All physical outputs required')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        state.update(status='generator_scalar_owner_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error),finished_unix=time.time())
        raise
    finally:
        save(folder/'status.json',state)


if __name__ == '__main__':
    run()
