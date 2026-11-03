"""Follow the original I57note request,then custody and changed-note UI once."""
import os
from pathlib import Path
import subprocess
import time

from import_common_generator_measurements import ROOT,RUN,DOC,CONTROLLER,read,save,binding,require
from finish_matched_portfolio_publication import identity

HERE = Path(__file__).resolve().parent
JOB = RUN/'pressure-status-publication-finish-01'
NOTE_JOB = RUN/'native-pressure-reproduction-status-adoption-01'
BROWSER = RUN/'native-pressure-status-browser-01'


def terminal(owner):
    require(owner['status'] == 'native_pressure26note_adoption_acknowledged' and owner['adopted_notes'] == 26
            and not Path(f'/proc/{owner["pid"]}').exists(),'Original status-note owner must be terminal and acknowledged')
    response = read(NOTE_JOB/'adopt.response.json')
    require(response['status'] == 200 and response['result']['checkpoint'] == owner['checkpoint']
            and response['result']['value']['changed_count'] == 26,'Native correction response must match owner ACK')


def run():
    require(not JOB.exists() and not BROWSER.exists() and not (NOTE_JOB/'browser-actions.json').exists()
            and not (DOC/'reports/native-pressure-reproduction-status-adoption-01.json').exists(),
            'Fresh status finish owner required;existing stages need reconciliation')
    owner = read(NOTE_JOB/'status.json')
    require(owner['pid'] == 1821421,'Follow the explicitly identified original correction owner')
    started = identity(owner['pid'])
    if started is not None:
        command = [value.decode() for value in Path(f'/proc/{owner["pid"]}/cmdline').read_bytes().split(b'\0') if value]
        require(any((ROOT/value).resolve() == HERE/'adopt_pressure_reproduction_status_notes.py' for value in command)
                and command[-1] == 'adopt','Exact original correction process required')
    else:
        terminal(owner)
    names = ['adopt_pressure_reproduction_status_notes.py','adopt_pressure_outcome_notes.py',
        'adopt_matched_generator_outcome_notes_02.py','adopt_matched_generator_outcome_notes.py',
        'import_common_generator_measurements.py','finish_matched_portfolio_publication.py',
        'prepare_pressure_status_browser.cjs','matched_model_query.cjs','native_project_browser.cjs']
    frozen = [binding(HERE/name) for name in names]+[binding(Path(__file__))]
    JOB.mkdir()
    state = dict(status='waiting_for_original_status_note_owner',pid=os.getpid(),created_unix=time.time(),
        original_note_pid=owner['pid'],original_note_starttime=started,stages=[],active_child=None,frozen_tools=frozen,
        new_model_forwards=0,payload_copies=0,automatic_retries=False,full_portfolio_complete=False)
    save(JOB/'status.json',state)
    try:
        deadline = time.monotonic()+7200
        while identity(owner['pid']) is not None:
            require(identity(owner['pid']) == started,'Original correction PID reused;reconcile')
            require(time.monotonic() < deadline,'Observation bound;no native request replayed')
            time.sleep(2)
        terminal(read(NOTE_JOB/'status.json'))
        state['accepted_original_owner'] = binding(NOTE_JOB/'status.json')
        save(JOB/'status.json',state)
        stages = [('01-status-note-custody',[ROOT/'runs/dependency-migration-v1/app-env/bin/python','-B',HERE/'adopt_pressure_reproduction_status_notes.py','after'],
                   DOC/'reports/native-pressure-reproduction-status-adoption-01.json','native26pressure_outcome_notes_exact_and_scientific_records_unchanged'),
                  ('02-status-browser-preparation',['node',HERE/'prepare_pressure_status_browser.cjs'],None,None),
                  ('03-actual-status-browser',['node',HERE/'native_project_browser.cjs',CONTROLLER,BROWSER,NOTE_JOB/'browser-actions.json','300000'],
                   BROWSER/'report.json','passed_native_project_UI_actions')]
        for name,command,evidence,expected in stages:
            require(all(binding(ROOT/spec['path']) == spec for spec in frozen),'Frozen status-publication tool changed')
            log = JOB/(name+'.log')
            item = dict(name=name,command=[str(arg) for arg in command],started_unix=time.time(),log=str(log.relative_to(ROOT)))
            state['stages'].append(item)
            state.update(status='pressure_status_publication_stage_running',active_stage=name)
            save(JOB/'status.json',state)
            with log.open('x') as stream:
                child = subprocess.Popen([str(arg) for arg in command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child'] = item['pid'] = child.pid
                save(JOB/'status.json',state)
                code = child.wait()
            item.update(exit_code=code,finished_unix=time.time(),log_binding=binding(log))
            state['active_child'] = None
            save(JOB/'status.json',state)
            require(code == 0,'Original status-publication stage failed;retain evidence;no retry')
            if evidence is not None:
                require(read(evidence)['status'] == expected,'Actual stage acceptance differs')
                item['accepted_evidence'] = binding(evidence)
            save(JOB/'status.json',state)
            print('Accepted pressure-status stage:',name,flush=True)
        state.update(status='pressure_status_notes_and_UI_complete_actual_image_inspection_pending',active_stage=None,
            finished_unix=time.time(),actual_UI_image_inspection_pending=True,project_left_open=True)
    except BaseException as error:
        state.update(status='pressure_status_publication_stopped_requires_reconciliation',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(JOB/'status.json',state)


if __name__ == '__main__':
    run()
