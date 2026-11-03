"""One sequential owner for the original eight modal-pressure GPU workers."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication
from modal_component_benchmark import verify_sources


def run():
    current_publication(ROOT)
    protocol_path = RUN/'modal-component-GPU-protocol-01.json'
    protocol = read(protocol_path);verify_sources(protocol)
    require(binding(ROOT/protocol['runtime_manifest']['path']) == protocol['runtime_manifest'], 'Runtime manifest changed')
    runtime = read(ROOT/protocol['runtime_manifest']['path'])
    folder = ROOT/protocol['output_root']
    require(not folder.exists() and folder.is_relative_to(RUN), 'Original owner exists;reconcile it instead of restarting')
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    folder.mkdir()
    state = dict(status='modal_GPU_benchmark_running',pid=os.getpid(),created_unix=time.time(),protocol=binding(protocol_path),
        pending=protocol['phase_order'].copy(),qualified=[],active_child=None,automatic_retries=False,full_portfolio_complete=False)
    save(folder/'status.json',state);began = time.monotonic()
    try:
        for item in protocol['phase_order']:
            remaining = protocol['aggregate_deadline_seconds']-(time.monotonic()-began)
            require(remaining > 0 and shutil.disk_usage(folder).free >= protocol['minimum_free_disk_bytes'], 'Time/disk bound exceeded')
            fid,phase = item['fit_item_id'],item['phase']
            with (folder/f'{fid}-phase{phase}.log').open('x') as stream:
                child = subprocess.Popen([runtime['python'],'-B',protocol['worker'],'--protocol',str(protocol_path.relative_to(ROOT)),
                    '--fit-id',fid,'--phase',str(phase)],cwd=ROOT,env=environment(runtime,'cuda'),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(active_child=child.pid,active_item=item);save(folder/'status.json',state)
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
            path = folder/f'{fid}-phase{phase}'/'result.json'
            require(code == 0 and path.exists(), 'Original modal worker failed;retain logs and reconcile,no retry')
            result = read(path)
            require(result['status'] == 'original_modal_GPU_phase_qualified' and result['cases'] == 144
                    and result['learned_tensors_and_architecture_unchanged'] and not result['device_observation']['forbidden_forward_attempts'],
                    'Original modal worker qualification differs')
            state['qualified'].append(dict(**item,pid=child.pid,result=binding(path)))
            state['pending'].remove(item);save(folder/'status.json',state)
            print(fid,'phase',phase,'qualified144native requests',flush=True)
        state.update(status='all_eight_original_modal_GPU_workers_terminal',active_child=None,active_item=None,finished_unix=time.time())
    except BaseException as error:
        state.update(status='modal_benchmark_stopped_requires_reconciliation',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(folder/'status.json',state)


if __name__ == '__main__':
    run()
