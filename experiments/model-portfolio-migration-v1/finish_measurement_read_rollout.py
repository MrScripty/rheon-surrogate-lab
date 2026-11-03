"""Follow the existing Open owner, then serially verify custody and native reads.

This never starts or retries a project lifecycle request. A failed/unknown
original Open stops this continuation before its read-only acceptance checks.
"""
import hashlib
import os
from pathlib import Path
import select
import subprocess
import time

from import_common_generator_measurements import ROOT, RUN, CONTROLLER, read, save, binding, require


JOB = RUN/'native-measurement-source-read-rollout-02'
PYTHON = ROOT/'runs/dependency-migration-v1/app-env/bin/python'


def run():
    output = JOB/'finish-status.json'
    require(not output.exists(), 'Follow the original acceptance continuation; do not duplicate it')
    original = read(JOB/'open-status.json')
    pid = original['pid']
    result = dict(status='following_original_Open_owner',created_unix=time.time(),pid=os.getpid(),
        original_Open_pid=pid,producer=binding(Path(__file__)),steps=[],native_lifecycle_requests_started=0,
        project_or_dataset_copies=0,new_model_forwards=0,manual_image_inspection_pending=True)
    save(output,result)
    try:
        if Path('/proc',str(pid)).exists():
            descriptor = os.pidfd_open(pid)
            try:
                command = Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
                require(any(value.endswith(b'rollout_measurement_source_reads.py') for value in command)
                        and b'open' in command and b'--job' in command,
                        'Actual original Open driver identity required')
                poller = select.poll()
                poller.register(descriptor,select.POLLIN)
                deadline = original['created_unix']+7400
                while not poller.poll(1000):
                    if time.time() > deadline:
                        raise TimeoutError('Original Open still needs explicit observation; no request replay permitted')
            finally:
                os.close(descriptor)
        original = read(JOB/'open-status.json')
        require(original['status'] == 'native_open_acknowledged'
                and read(JOB/'open.response.json')['status'] == 200, 'Original Open must acknowledge before acceptance checks')
        def step(name, command, timeout):
            result['status'] = name+'_running'
            result['active_command'] = command
            save(output,result)
            started = time.monotonic()
            with (JOB/(name+'.log')).open('xb') as log:
                process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                result['active_pid'] = process.pid
                save(output,result)
                code = process.wait(timeout=timeout)
            result['steps'].append(dict(name=name,returncode=code,wall_seconds=time.monotonic()-started,
                log=binding(JOB/(name+'.log'))))
            result.pop('active_pid',None)
            result.pop('active_command',None)
            save(output,result)
            require(code == 0, 'Original '+name+' failed; inspect its retained log without replay')
        step('after-custody',[str(PYTHON),'-B',str(ROOT/'experiments/model-portfolio-migration-v1/rollout_measurement_source_reads.py'),
            'custody-after','--job',str(JOB)],300)
        step('native-reads',[str(PYTHON),'-B',str(ROOT/'experiments/model-portfolio-migration-v1/time_native_measurement_reads.py'),
            '--job',str(RUN/'measurement-source-reads-native-candidate-01'),
            '--expected-commit','00a02ee25cb30d254c059a9b372036f9f6f36696'],900)
        step('native-browser',['node',str(ROOT/'experiments/model-portfolio-migration-v1/native_project_browser.cjs'),
            str(CONTROLLER),str(RUN/'native-measurement-source-read-browser-01'),str(JOB/'browser-actions.json'),'300000'],1800)
        browser = read(RUN/'native-measurement-source-read-browser-01/report.json')
        require(browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
                and len(browser['steps']) == 68 and not browser['errors'], 'Complete prepared native UI checks required')
        require(binding(ROOT/'projects/rheon-model-portfolio-unified-04/.tuldok/project.json')['sha256'] ==
                read(JOB/'after-custody.json')['facts']['header_sha256'], 'Read-only UI changed the project header')
        result.update(status='native_custody_reads_and_UI_passed_manual_images_pending',finished_unix=time.time(),
            project_left_open=True,actual_UI_action_count=68,chart_selections=3,numerical_presence_checks=14,
            whole_Open_timing='Mixed contention intervals;not a clean speed comparison',
            candidate_API_timing='Observed after user confirmed CPU contention ended;not a controlled paired speedup')
    except BaseException as error:
        result.update(status='acceptance_continuation_stopped_for_explicit_reconciliation',error_type=type(error).__name__)
        raise
    finally:
        save(output,result)
    print(result['status'],flush=True)


if __name__ == '__main__':
    run()
