"""Follow the existing note owner,then custody and real UI;never replay it."""
import os
from pathlib import Path
import subprocess
import time

from import_common_generator_measurements import ROOT,RUN,DOC,CONTROLLER,read,save,binding,require
from finish_matched_portfolio_publication import identity

HERE = Path(__file__).resolve().parent
JOB = RUN/'pressure-publication-finish-02'
NOTE_JOB = RUN/'native-pressure-outcome-adoption-01'
BROWSER = RUN/'native-pressure-publication-browser-01'


def require_terminal_notes(owner):
    require(owner['status'] == 'native_pressure26note_adoption_acknowledged' and owner['adopted_notes'] == 26
            and not Path(f'/proc/{owner["pid"]}').exists(),'Original26note owner must be terminal and acknowledged')
    response = read(NOTE_JOB/'adopt.response.json')
    require(response['status'] == 200 and response['result']['checkpoint'] == owner['checkpoint']
            and response['result']['value']['changed_count'] == 26,'Original native26note response must match owner ACK')


def run():
    require(not JOB.exists() and not BROWSER.exists() and not (NOTE_JOB/'browser-actions.json').exists()
            and not (DOC/'reports/native-pressure-outcome-adoption-01.json').exists(),
            'Fresh finish owner required; existing stages need explicit reconciliation')
    audit = read(DOC/'reports/native-all150-pressure-import-custody-01.json')
    import_owner = read(ROOT/audit['original_owner']['path'])
    require(audit['status'] == 'all150_pressure_tables_native_custody_exact'
            and binding(ROOT/audit['original_owner']['path']) == audit['original_owner']
            and not Path(f'/proc/{import_owner["pid"]}').exists(),'Accepted terminal pressure import custody required')
    owner = read(NOTE_JOB/'status.json')
    require(owner['pid'] == 1808723,'Follow the explicitly identified original26note process')
    started = identity(owner['pid'])
    if started is not None:
        command = [value.decode() for value in Path(f'/proc/{owner["pid"]}/cmdline').read_bytes().split(b'\0') if value]
        require(any((ROOT/value).resolve() == HERE/'adopt_pressure_outcome_notes.py' for value in command)
                and command[-1] == 'adopt','Exact existing native note process required')
    else:
        require_terminal_notes(owner)
    tools = ['import_all_pressure_measurements.py','adopt_pressure_outcome_notes.py','adopt_matched_generator_outcome_notes_02.py',
        'adopt_matched_generator_outcome_notes.py','import_common_generator_measurements.py','finish_matched_portfolio_publication.py',
        'prepare_pressure_publication_browser.cjs','matched_model_query.cjs','native_project_browser.cjs']
    frozen = [binding(HERE/name) for name in tools]+[binding(Path(__file__))]
    JOB.mkdir()
    state = dict(status='waiting_for_original26note_owner',pid=os.getpid(),created_unix=time.time(),original_note_pid=owner['pid'],
        original_note_starttime=started,stages=[],active_child=None,frozen_tools=frozen,new_model_forwards=0,
        dataset_or_weight_copies=0,automatic_retries=False,full_portfolio_complete=False)
    save(JOB/'status.json',state)
    try:
        deadline = time.monotonic()+7200
        while identity(owner['pid']) is not None:
            require(identity(owner['pid']) == started,'Original note PID reused; reconcile identity')
            require(time.monotonic() < deadline,'Observation bound reached; no native request replayed')
            time.sleep(2)
        require_terminal_notes(read(NOTE_JOB/'status.json'))
        state['accepted_note_owner'] = binding(NOTE_JOB/'status.json')
        save(JOB/'status.json',state)
        stages = [('01-note-after-custody',[ROOT/'runs/dependency-migration-v1/app-env/bin/python','-B',HERE/'adopt_pressure_outcome_notes.py','after'],
                    DOC/'reports/native-pressure-outcome-adoption-01.json','native26pressure_outcome_notes_exact_and_scientific_records_unchanged'),
                  ('02-browser-preparation',['node',HERE/'prepare_pressure_publication_browser.cjs'],None,None),
                  ('03-actual-native-browser',['node',HERE/'native_project_browser.cjs',CONTROLLER,BROWSER,NOTE_JOB/'browser-actions.json','300000'],
                    BROWSER/'report.json','passed_native_project_UI_actions')]
        for name,command,evidence,expected in stages:
            require(all(binding(ROOT/spec['path']) == spec for spec in frozen),'Frozen publication tool changed')
            log = JOB/(name+'.log')
            item = dict(name=name,command=[str(arg) for arg in command],started_unix=time.time(),log=str(log.relative_to(ROOT)))
            state['stages'].append(item)
            state.update(status='pressure_publication_stage_running',active_stage=name)
            save(JOB/'status.json',state)
            with log.open('x') as stream:
                child = subprocess.Popen([str(arg) for arg in command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child'] = item['pid'] = child.pid
                save(JOB/'status.json',state)
                code = child.wait()
            item.update(exit_code=code,finished_unix=time.time(),log_binding=binding(log))
            state['active_child'] = None
            save(JOB/'status.json',state)
            require(code == 0,'Original publication stage failed; inspect its retained evidence;no retry')
            if evidence is not None:
                require(read(evidence)['status'] == expected,'Stage acceptance differs')
                item['accepted_evidence'] = binding(evidence)
            save(JOB/'status.json',state)
            print('Accepted pressure-publication stage:',name,flush=True)
        state.update(status='pressure_notes_custody_and_browser_complete_actual_image_inspection_pending',finished_unix=time.time(),
            active_stage=None,actual_UI_image_inspection_pending=True,project_left_open=True)
    except BaseException as error:
        state.update(status='pressure_publication_stopped_requires_reconciliation',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(JOB/'status.json',state)


if __name__ == '__main__':
    run()
