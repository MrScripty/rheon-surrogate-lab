"""One sequential process owner; terminal gaps are retained without retry."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT,RUN,read,save,binding,require
from original_train_pressure_accuracy_02 import publication_gate
from original_train_pressure_timing import verify_sources


def run(protocol_path):
    publication_gate()
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rheon.original-train-pressure-GPU-timing.v1' and protocol['device'] == 'cuda',
            'Explicit original pressure timing protocol required')
    verify_sources(protocol)
    require(binding(ROOT/protocol['runtime_manifest']['path']) == protocol['runtime_manifest'],'Frozen runtime manifest changed')
    folder = ROOT/protocol['output_root']
    require(folder.is_relative_to(RUN) and not folder.exists(),'Fresh batch output required; reconcile original owner first')
    runtime = read(ROOT/protocol['runtime_manifest']['path'])
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    folder.mkdir()
    state = dict(status='original_pressure_GPU_timing_running',pid=os.getpid(),created_unix=time.time(),protocol=binding(protocol_path),
        pending=list(protocol['block_order']),qualified=[],adapter_gaps=[],active_block=None,active_child=None,
        inference_device='cuda',timing_units='ms/pressure_request',full_simulation_step=False,full_portfolio_complete=False)
    save(folder/'status.json',state)
    began = time.monotonic()
    try:
        for block in protocol['block_order']:
            require(shutil.disk_usage(folder).free >= protocol['minimum_free_disk_bytes'],'Free disk below registered reserve')
            remaining = protocol['aggregate_deadline_seconds']-(time.monotonic()-began)
            require(remaining > 0,'Aggregate pressure timing deadline reached')
            fit_id,phase = block['fit_item_id'],block['phase']
            command = [runtime['python'],'-B',protocol['worker'],'--protocol',str(protocol_path.relative_to(ROOT)),
                       '--fit-id',fit_id,'--phase',str(phase)]
            log = folder/f'{fit_id}-phase-{phase}.log'
            state.update(active_block=block,active_child=None)
            save(folder/'status.json',state)
            with log.open('x') as stream:
                process = subprocess.Popen(command,cwd=ROOT,env=environment(runtime,'cuda'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child'] = process.pid
                save(folder/'status.json',state)
                try:
                    code = process.wait(timeout=min(remaining,protocol['worker_deadline_seconds']+30))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL)
                            process.wait(timeout=10)
            state['active_child'] = None
            result_path = folder/fit_id/f'phase-{phase}'/'result.json'
            result = read(result_path) if result_path.exists() else None
            if code == 0 and result and result['status'] == 'original_train_pressure_GPU_timing_qualified':
                require(result['cases'] == 128 and result['all_output_hashes_exact'] and result['primary_observer_calls'] == 0
                        and result['device_observation']['closed'] and not result['device_observation']['forbidden_forward_attempts'],
                        'Timing worker must retain every numerical/device invariant')
                state['qualified'].append(dict(**block,result=binding(result_path),pid=process.pid,
                    case_median_median_ms=result['case_median_median_ms']))
                print(fit_id,'phase',phase,'qualified128GPUpressure timings',flush=True)
            else:
                state['adapter_gaps'].append(dict(**block,process_exit=code,log=binding(log),
                    result=binding(result_path) if result_path.exists() else None))
                print(fit_id,'phase',phase,'terminal gap retained; no retry',flush=True)
                raise RuntimeError('Timing gap requires reconciliation before additional work')
            state['pending'].remove(block)
            save(folder/'status.json',state)
        state.update(status='all_original_pressure_GPU_timing_phases_terminal',active_block=None,active_child=None,
            finished_unix=time.time(),all_qualified=len(state['qualified']) == len(protocol['block_order']) and not state['adapter_gaps'])
    except BaseException as error:
        state.update(status='original_pressure_GPU_timing_stopped_requires_reconciliation',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(folder/'status.json',state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve())
