"""One retained sequential owner for the22remaining CUDA accuracy attempts."""
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


def run(protocol_path):
    publication_gate()
    plan = read(protocol_path)
    require(plan['schema'] == 'rheon.original-train-pressure-CUDA-qualification.v1' and plan['device'] == 'cuda'
            and plan['latency_claim'] is False,'Explicit original-train CUDA accuracy protocol required')
    folder = ROOT/plan['output_root']
    require(folder.is_relative_to(RUN) and not folder.exists(),'Fresh batch directory required; reconcile existing attempts')
    for path,expected in plan['source_sha256'].items():
        require(binding(ROOT/path)['sha256'] == expected if not Path(path).is_absolute()
                else __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest() == expected,
                'Frozen qualification source changed')
    require(binding(ROOT/plan['publication_acceptance']['path']) == plan['publication_acceptance'],'Publication acceptance changed')
    runtime = read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    folder.mkdir()
    state = dict(status='remaining_original_train_CUDA_accuracy_running',pid=os.getpid(),created_unix=time.time(),
        protocol=binding(protocol_path),pending=list(plan['fit_ids']),qualified=[],adapter_gaps=[],active_fit=None,active_child=None,
        latency_claim=False,new_parameter_updates=0,new_reference_generation=False,full_portfolio_complete=False)
    save(folder/'status.json',state)
    began = time.monotonic()
    try:
        for fit_id in plan['fit_ids']:
            require(shutil.disk_usage(folder).free >= plan['minimum_free_disk_bytes'],'Insufficient free disk for new result records')
            remaining = plan['aggregate_deadline_seconds']-(time.monotonic()-began)
            require(remaining > 0,'Aggregate qualification deadline reached')
            command = [runtime['python'],'-B',plan['worker'],'--protocol',str(protocol_path.relative_to(ROOT)),'--fit-id',fit_id]
            log = folder/(fit_id+'.log')
            state.update(active_fit=fit_id,active_child=None)
            save(folder/'status.json',state)
            with log.open('x') as stream:
                process = subprocess.Popen(command,cwd=ROOT,env=environment(runtime,'cuda'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child'] = process.pid
                save(folder/'status.json',state)
                try:
                    code = process.wait(timeout=min(remaining,plan['worker_deadline_seconds']+30))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            state['active_child'] = None
            result_path = folder/fit_id/'result.json'
            result = read(result_path) if result_path.exists() else None
            if code == 0 and result and result['status'] == 'original_train_pressure_CUDA_accuracy_qualified':
                require(result['cases'] == 128 and result['learned_tensors_exact_and_unchanged'] and result['captured_architecture_unchanged']
                        and result['device_observation']['CPU_neural_forwards'] == 0
                        and not result['device_observation']['forbidden_forward_attempts'] and not result['unbound_lab_Python_modules'],
                        'Original CUDA qualification result differs from its declared invariant')
                state['qualified'].append(dict(fit_item_id=fit_id,result=binding(result_path),cases=128,worst=result['worst']))
                print(fit_id,'qualified128original pressure cases',flush=True)
            else:
                state['adapter_gaps'].append(dict(fit_item_id=fit_id,process_exit=code,log=binding(log),
                    result=binding(result_path) if result_path.exists() else None,scientific_accuracy_failure_claim=False))
                print(fit_id,'terminal adapter gap retained; no retry',flush=True)
            state['pending'].remove(fit_id)
            save(folder/'status.json',state)
        state.update(status='all22_original_train_CUDA_accuracy_attempts_terminal',active_fit=None,active_child=None,
            finished_unix=time.time(),all22_qualified=len(state['qualified'])==22 and not state['adapter_gaps'])
    except BaseException as error:
        state.update(status='original_train_CUDA_batch_stopped_requires_reconciliation',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(folder/'status.json',state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve())
