"""One retained sequential owner for two fresh original-viscosity CUDA phases."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT,RUN,read,save,binding,require
from component_material import current_publication
from viscosity_component_benchmark import verify_sources


def run():
    current_publication(ROOT)
    protocol_path = RUN/'viscosity-component-GPU-protocol-01.json'
    protocol = read(protocol_path)
    verify_sources(protocol)
    require(binding(ROOT/protocol['runtime_manifest']['path']) == protocol['runtime_manifest'],'Runtime manifest changed')
    runtime = read(ROOT/protocol['runtime_manifest']['path'])
    folder = ROOT/protocol['output_root']
    require(not folder.exists() and folder.is_relative_to(RUN),'Fresh viscosity batch required;reconcile original attempts')
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    folder.mkdir()
    state = dict(status='viscosity_GPU_benchmark_running',pid=os.getpid(),created_unix=time.time(),protocol=binding(protocol_path),
        pending=[0,1],qualified=[],active_child=None,automatic_retries=False,new_model_fits=0,full_portfolio_complete=False)
    save(folder/'status.json',state)
    began = time.monotonic()
    try:
        for phase in protocol['phase_order']:
            remaining = protocol['aggregate_deadline_seconds']-(time.monotonic()-began)
            require(remaining > 0 and shutil.disk_usage(folder).free >= protocol['minimum_free_disk_bytes'],'Aggregate time/disk bound exceeded')
            log = folder/f'phase-{phase}.log'
            with log.open('x') as stream:
                command = [runtime['python'],'-B',protocol['worker'],'--protocol',str(protocol_path.relative_to(ROOT)),'--phase',str(phase)]
                child = subprocess.Popen(command,cwd=ROOT,env=environment(runtime,'cuda'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(active_child=child.pid,active_phase=phase)
                save(folder/'status.json',state)
                try:
                    code = child.wait(timeout=min(remaining,protocol['worker_deadline_seconds']+30))
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            state['active_child'] = None
            result_path = folder/f'phase-{phase}'/'result.json'
            require(code == 0 and result_path.exists(),'Original viscosity worker failed;inspect retained log/result;no retry')
            result = read(result_path)
            require(result['status'] == 'original_viscosity_GPU_phase_qualified' and result['cases'] == 80
                    and result['learned_tensors_and_architecture_unchanged'] and not result['device_observation']['forbidden_forward_attempts'],
                    'Original viscosity phase qualification differs')
            state['qualified'].append(dict(phase=phase,pid=child.pid,result=binding(result_path)))
            state['pending'].remove(phase)
            save(folder/'status.json',state)
            print('Original viscosityphase',phase,'qualified80patterns',flush=True)
        state.update(status='both_original_viscosity_GPU_phases_terminal',active_child=None,active_phase=None,finished_unix=time.time())
    except BaseException as error:
        state.update(status='viscosity_benchmark_stopped_requires_reconciliation',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(folder/'status.json',state)


if __name__ == '__main__':
    run()
