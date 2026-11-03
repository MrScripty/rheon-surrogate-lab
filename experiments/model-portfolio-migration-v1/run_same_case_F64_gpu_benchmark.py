"""Own eight matched F64 workers only after the35fit campaign is reduced."""
import os
from pathlib import Path
import subprocess
import sys
import time

from full_state_gpu_timing import resources
from review_water_learning_sequence import ROOT, binding, read, require
from run_joint_generator_gpu_benchmark import save, stop

RUN = ROOT/'runs/model-portfolio-migration-v1'
PROTOCOL = RUN/'same-case-F64-GPU-protocol-01.json'


def run():
    previous = read(RUN/'complete-generator-GPU-benchmark-01/status.json')
    reduced = read(ROOT/'docs/plans/model-portfolio-migration-v1/reports/complete-generator-GPU-benchmark-01.json')
    require(previous['status'] == 'all70_original_workers_complete_reduction_pending' and previous['active'] is None
            and len(previous['completed']) == 70 and not Path(f'/proc/{previous["pid"]}').exists()
            and reduced['status'] == 'all35_original_generators_common_GPU_primary_and_hook_calibration_reduced',
            'Original35fit benchmark must finish and independently reduce before F64dispatch')
    protocol = read(PROTOCOL)
    output = ROOT/protocol['output']
    require(not output.exists(),'Preserve original matched F64 owner;never replay its directory')
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Preregistered F64 dependency changed')
    before = resources()
    require(not before['other_compute_pids'],'Another CUDA compute process is active;no F64dispatch')
    runtime = read(ROOT/protocol['runtime_manifest'])
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir()
    state = dict(status='original_F64_campaign_starting',pid=os.getpid(),created_unix=time.time(),
        protocol=binding(PROTOCOL),completed=[],active=None,resources_before=before,
        preceding_generator_reduction=binding(ROOT/'docs/plans/model-portfolio-migration-v1/reports/complete-generator-GPU-benchmark-01.json'),
        full_portfolio_complete=False)
    status = output/'status.json'
    save(status,state)
    child = None
    try:
        for block in protocol['process_blocks']:
            path = ROOT/block['result']
            require(path.parent == output and not path.exists(),'Fresh original F64worker required')
            log = path.with_suffix('.log')
            with log.open('xb') as stream:
                child = subprocess.Popen([runtime['python'],'-B',str(ROOT/protocol['worker']),
                    '--protocol',str(PROTOCOL),'--K',str(block['K']),'--phase',str(block['phase']),
                    '--output',str(path)],cwd=ROOT,env=environment(runtime,'cuda'),stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='original_F64_worker_running',active=dict(block,pid=child.pid,
                    log=str(log.relative_to(ROOT))),updated_unix=time.time())
                save(status,state)
                print('Started F64 K',block['K'],'phase',block['phase'],'PID',child.pid,flush=True)
                try:
                    code = child.wait(timeout=protocol['worker_deadline_seconds']+60)
                finally:
                    stop(child)
            require(code == 0 and path.exists(),'Original F64worker failed;reconcile result/log without replay')
            d = read(path)
            require(d['status'] == 'all64_original_cases_F64_N1_GPU_measured' and d['protocol'] == state['protocol']
                    and d['K'] == block['K'] and d['phase'] == block['phase'] and d['cleanup_complete'],
                    'Original F64worker outcome/membership differs')
            state['completed'].append(dict(block,result_binding=binding(path)))
            state.update(active=None,status='original_F64_worker_completed',updated_unix=time.time())
            save(status,state)
            child = None
        state.update(status='all8_original_matched_F64_workers_complete_reduction_pending',finished_unix=time.time(),active=None)
        save(status,state)
    except BaseException as error:
        if child is not None:
            stop(child)
        state.update(status='original_F64_campaign_requires_reconciliation',active=None,error_type=type(error).__name__,
            error=str(error),updated_unix=time.time())
        save(status,state)
        raise


if __name__ == '__main__':
    run()
