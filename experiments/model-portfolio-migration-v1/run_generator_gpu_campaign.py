"""Serial owner for source-bound generator protocols; one output directory/run.

Each process has a deadline and belongs to this owner. A failed or interrupted
campaign requires reconciliation of its retained status; it is never replayed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from full_state_gpu_timing import resources
from run_joint_generator_gpu_benchmark import save, stop
from review_water_learning_sequence import binding, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'


def run(protocol_path):
    protocol = read(protocol_path)
    blocks = protocol['process_blocks']
    require(len(blocks) == 2*protocol['models'] and protocol['cases_per_model'] == 64,
            'Two complete model phases and the shared64cases required')
    directories = {(ROOT/block['result']).parent.resolve() for block in blocks}
    require(len(directories) == 1, 'One owned output directory required')
    output = directories.pop()
    require(output.parent == RUN and not output.exists(), 'Fresh bounded campaign directory required; reconcile existing work')
    for spec in [protocol['cases'],*protocol['dependencies']]:
        require(binding(ROOT/spec['path']) == spec, 'Preregistered input changed')
    before = resources()
    if before['other_compute_pids']:
        print('Not dispatched: another GPU compute process is active:',before['other_compute_pids'],flush=True)
        return
    runtime = read(ROOT/protocol['runtime_manifest'])
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir()
    state = dict(status='starting_original_workers',pid=os.getpid(),created_unix=time.time(),active=None,completed=[],
        protocol=binding(protocol_path),protocol_sha256=binding(protocol_path)['sha256'],runner=binding(Path(__file__)),
        resources_before=before,CPU_neural_forwards=0,new_reference_generation=False,native_mutations=0,
        full_portfolio_complete=False)
    state_path = output/'status.json'
    save(state_path,state)
    child = None
    try:
        for block in blocks:
            result = ROOT/block['result']
            require(result.parent.resolve() == output and not result.exists(), 'Fresh admitted result required')
            log = result.with_suffix('.log')
            command = [runtime['python'],'-B',str(ROOT/protocol['worker']),'--protocol',str(protocol_path),
                '--identifier',block['identifier'],'--phase',str(block['phase']),'--output',str(result)]
            with log.open('xb') as stream:
                child = subprocess.Popen(command,cwd=ROOT,env=environment(runtime,'cuda'),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='original_worker_running',active=dict(block,pid=child.pid,log=str(log.relative_to(ROOT))),updated_unix=time.time())
                save(state_path,state)
                print('Started',block['identifier'],'phase',block['phase'],'PID',child.pid,flush=True)
                try:
                    code = child.wait(timeout=protocol['worker_deadline_seconds']+60)
                finally:
                    stop(child)
            require(code == 0 and result.exists(), 'Worker failed; inspect retained evidence without automatic replay')
            measured = read(result)
            require(measured['status'] == 'all64_native_cases_measured'
                    and measured['protocol_sha256'] == state['protocol_sha256'], 'Worker result differs from protocol')
            state['completed'].append(dict(block,result_binding=binding(result)))
            state.update(status='original_worker_completed',active=None,updated_unix=time.time())
            save(state_path,state)
            child = None
        state.update(status='all_original_workers_complete_reduction_pending',finished_unix=time.time(),active=None,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    run(parser.parse_args().protocol.resolve())
