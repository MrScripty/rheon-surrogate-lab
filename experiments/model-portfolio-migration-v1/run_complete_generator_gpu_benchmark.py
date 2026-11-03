"""Own the admitted70fresh generator workers serially without replay."""
from pathlib import Path
import os
import subprocess
import sys
import time

from full_state_gpu_timing import resources
from review_water_learning_sequence import ROOT, binding, read, require
from run_joint_generator_gpu_benchmark import save, stop

RUN = ROOT/'runs/model-portfolio-migration-v1'
PROTOCOL = RUN/'complete-generator-GPU-protocol-01.json'


def run():
    protocol = read(PROTOCOL)
    output = ROOT/protocol['output']
    require(not output.exists(),'Follow the original campaign owner;never replay an output directory')
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Preregistered dependency changed')
    before = resources()
    require(not before['other_compute_pids'],'Another CUDA compute process is active;no campaign dispatched')
    runtime = read(ROOT/protocol['runtime_manifest'])
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir()
    state = dict(status='original_campaign_starting',pid=os.getpid(),created_unix=time.time(),
        protocol=binding(PROTOCOL),completed=[],active=None,resources_before=before,full_portfolio_complete=False)
    status = output/'status.json'
    save(status,state)
    child = None
    try:
        for block in protocol['process_blocks']:
            path = ROOT/block['result']
            require(path.parent == output and not path.exists(),'Fresh original worker result required')
            log = path.with_suffix('.log')
            with log.open('xb') as stream:
                child = subprocess.Popen([runtime['python'],'-B',str(ROOT/protocol['worker']),
                    '--protocol',str(PROTOCOL),'--fit-id',block['fit_item_id'],'--phase',str(block['phase']),
                    '--output',str(path)],cwd=ROOT,env=environment(runtime,'cuda'),stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='original_worker_running',active=dict(block,pid=child.pid,
                    log=str(log.relative_to(ROOT))),updated_unix=time.time())
                save(status,state)
                print('Started',block['fit_item_id'],block['identifier'],'phase',block['phase'],'PID',child.pid,flush=True)
                try:
                    code = child.wait(timeout=protocol['worker_deadline_seconds']+60)
                finally:
                    stop(child)
            require(code == 0 and path.exists(),'Original worker failed;reconcile retained result/log without replay')
            d = read(path)
            require(d['status'] == 'all64_original_cases_uninstrumented_primary_and_paired_hooks_measured'
                    and d['protocol'] == state['protocol'] and d['fit_item_id'] == block['fit_item_id']
                    and d['phase'] == block['phase'],'Original worker outcome/membership differs')
            state['completed'].append(dict(block,result_binding=binding(path)))
            state.update(active=None,status='original_worker_completed',updated_unix=time.time())
            save(status,state)
            child = None
        state.update(status='all70_original_workers_complete_reduction_pending',finished_unix=time.time(),active=None)
        save(status,state)
    except BaseException as error:
        if child is not None:
            stop(child)
        state.update(status='original_campaign_requires_reconciliation',active=None,error_type=type(error).__name__,
            error=str(error),updated_unix=time.time())
        save(status,state)
        raise


if __name__ == '__main__':
    run()
