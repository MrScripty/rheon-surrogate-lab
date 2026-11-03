"""Own the preregistered joint-generator GPU workers; never replay a directory."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from full_state_gpu_timing import resources
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
PROTOCOL = RUN/'joint-generator-GPU-benchmark-protocol-01.json'
OUTPUT = RUN/'joint-generator-GPU-benchmark-01'


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_bytes(canonical(value)+b'\n')
    pending.replace(path)


def stop(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid,signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=10)


def run():
    require(not OUTPUT.exists(), 'Follow/reconcile the original campaign; do not replay its output directory')
    protocol = read(PROTOCOL)
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec, 'Preregistered dependency changed')
    require(binding(ROOT/protocol['cases']['path']) == protocol['cases'], 'Qualified case selection changed')
    observed = resources()
    if observed['other_compute_pids']:
        print('Not dispatched: another GPU compute process is active:',observed['other_compute_pids'],flush=True)
        return
    runtime = read(ROOT/protocol['runtime_manifest'])
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    OUTPUT.mkdir()
    state = dict(status='starting_original_workers',pid=os.getpid(),created_unix=time.time(),active=None,completed=[],
        protocol_sha256=binding(PROTOCOL)['sha256'],runner=binding(Path(__file__)),resources_before=observed,
        CPU_neural_forwards=0,new_reference_generation=False,native_mutations=0,full_portfolio_complete=False)
    state_path = OUTPUT/'status.json'
    save(state_path,state)
    child = None
    try:
        for block in protocol['process_blocks']:
            result = ROOT/block['result']
            require(result.parent == OUTPUT and not result.exists(), 'Fresh exact original worker result required')
            log = result.with_suffix('.log')
            command = [runtime['python'],'-B',str(ROOT/protocol['worker']),'--protocol',str(PROTOCOL),
                '--identifier',block['identifier'],'--phase',str(block['phase']),'--output',str(result)]
            with log.open('xb') as stream:
                child = subprocess.Popen(command,cwd=ROOT,env=environment(runtime,'cuda'),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='original_worker_running',active=dict(block,pid=child.pid,log=str(log.relative_to(ROOT))),
                             updated_unix=time.time())
                save(state_path,state)
                print('Started',block['identifier'],'phase',block['phase'],'PID',child.pid,flush=True)
                try:
                    code = child.wait(timeout=protocol['worker_deadline_seconds']+60)
                finally:
                    stop(child)
            require(code == 0 and result.exists(), 'Original worker failed; inspect its retained result/log, never replay automatically')
            measured = read(result)
            require(measured['status'] == 'all64_native_cases_measured'
                    and measured['protocol_sha256'] == state['protocol_sha256'], 'Original worker result membership differs')
            state['completed'].append(dict(block,result_binding=binding(result)))
            state.update(active=None,status='original_worker_completed',updated_unix=time.time())
            save(state_path,state)
            child = None
        state.update(status='all8_original_workers_complete_reduction_pending',finished_unix=time.time(),active=None,
                     independent_accuracy_and_timing_reduction_complete=False)
        save(state_path,state)
    except BaseException as error:
        if child is not None:
            stop(child)
        state.update(status='original_campaign_stopped_requires_reconciliation',active=None,
                     error_type=type(error).__name__,error=str(error),updated_unix=time.time())
        save(state_path,state)
        raise


if __name__ == '__main__':
    run()
