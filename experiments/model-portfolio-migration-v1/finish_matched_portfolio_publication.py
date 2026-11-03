"""Serially finish the prepared publication without replaying any native owner.

Wait for the existing import process, verify custody, publish the qualified
notes, then exercise the actual browser. Human/assistant image inspection and
its explicit acceptance receipt remain required after this owner finishes.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require

HERE = Path(__file__).resolve().parent
JOB = RUN/'matched-portfolio-publication-finish-01'
STATUS = JOB/'status.json'
IMPORT_OWNER = RUN/'native-matched-generator-GPU-import-reconciliation-01/status.json'
NOTE_JOB = RUN/'native-matched-generator-outcome-adoption-01'
BROWSER = RUN/'native-matched-portfolio-browser-01'


def require_terminal_import(state):
    require(state['status'] == 'four_readable_matched_GPU_imports_native_metadata_and_maxima_passed'
            and state['active_request'] is None and len(state['imports']) == 4
            and all(r.get('status') == 200 for r in state['requests'])
            and not Path(f'/proc/{state["pid"]}').exists(),
            'Original reconciled import must be terminal with all native checks acknowledged')


def identity(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    # A process name can contain spaces or parentheses. Fields after the last
    # closing parenthesis start with field3; starttime is field22.
    return raw.rsplit(')',1)[1].split()[19]


def run():
    require(not JOB.exists() and not NOTE_JOB.exists() and not BROWSER.exists()
            and not (DOC/'reports/native-matched-generator-GPU-import-custody-01.json').exists()
            and not (DOC/'reports/native-matched-generator-outcome-adoption-01.json').exists()
            and not (DOC/'reports/matched-preparation-source-byte-correction-01.json').exists(),
            'Fresh serial-publication owner required; existing stages need reconciliation')
    owner = read(IMPORT_OWNER)
    require(owner['pid'] == 1658248,'Follow the explicitly identified original continuation owner')
    started = identity(owner['pid'])
    if started is not None:
        command = [v.decode() for v in Path(f'/proc/{owner["pid"]}/cmdline').read_bytes().split(b'\0') if v]
        require(any((ROOT/a).resolve() == HERE/'reconcile_matched_generator_import.py' for a in command)
                and command[-1] == 'run','Exact original native import process required')
    else:
        require_terminal_import(owner)
    tools = ['reconcile_matched_generator_import.py','import_matched_generator_measurements.py',
        'import_common_generator_measurements.py','adopt_matched_generator_outcome_notes.py',
        'reconcile_matched_preparation_source_totals.py','prepare_matched_portfolio_browser.cjs','native_project_browser.cjs']
    frozen = [binding(HERE/name) for name in tools]+[binding(Path(__file__))]
    JOB.mkdir()
    state = dict(status='waiting_for_original_import_owner',pid=os.getpid(),created_unix=time.time(),
        import_owner_path=str(IMPORT_OWNER.relative_to(ROOT)),original_import_pid=owner['pid'],
        original_import_process_starttime=started,stages=[],active_child=None,frozen_tools=frozen,
        new_model_forwards=0,payload_copies=0,automatic_retries=False,full_portfolio_complete=False)
    save(STATUS,state)
    try:
        deadline = time.monotonic()+7200
        while identity(owner['pid']) is not None:
            require(identity(owner['pid']) == started,'Original import PID was reused; reconcile its identity')
            require(time.monotonic() < deadline,'Observation deadline; no native request replayed')
            time.sleep(2)
        owner = read(IMPORT_OWNER)
        require_terminal_import(owner)
        state['accepted_import_owner'] = binding(IMPORT_OWNER)
        save(STATUS,state)

        def stage(name,arguments,expected_path=None,expected_status=None):
            require(all(binding(ROOT/spec['path']) == spec for spec in frozen),'A frozen publication tool changed')
            log = JOB/(name+'.log')
            item = dict(name=name,arguments=[str(a) for a in arguments],started_unix=time.time(),log=str(log.relative_to(ROOT)))
            state['stages'].append(item)
            state.update(status='serial_publication_stage_running',active_stage=name)
            save(STATUS,state)
            with log.open('x') as output:
                process = subprocess.Popen([str(a) for a in arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child'] = process.pid
                item['pid'] = process.pid
                save(STATUS,state)
                code = process.wait()
            item.update(exit_code=code,finished_unix=time.time(),log_binding=binding(log))
            state['active_child'] = None
            save(STATUS,state)
            require(code == 0,'Original publication stage failed; inspect its retained log and owner; no retry')
            if expected_path is not None:
                require(read(expected_path)['status'] == expected_status,'Publication stage acceptance differs')
                item['accepted_evidence'] = binding(expected_path)
                save(STATUS,state)
            print('Accepted stage:',name,flush=True)

        python = ROOT/'runs/dependency-migration-v1/app-env/bin/python'
        def py(name,phase=None):
            return [python,'-B',HERE/name]+([phase] if phase else [])
        stage('01-import-custody',py('reconcile_matched_generator_import.py','after'),
            DOC/'reports/native-matched-generator-GPU-import-custody-01.json','four_readable_matched_GPU_tables_native_custody_exact')
        stage('02-advisory-source-totals',py('reconcile_matched_preparation_source_totals.py'),
            DOC/'reports/matched-preparation-source-byte-correction-01.json','advisory_source_totals_corrected_original_admission_and_packets_unchanged')
        stage('03-note-before-custody',py('adopt_matched_generator_outcome_notes.py','before'),
            NOTE_JOB/'before-custody.json','compact_matched_note_before_custody_captured')
        stage('04-note-adoption',py('adopt_matched_generator_outcome_notes.py','adopt'),
            NOTE_JOB/'status.json','native_matched38note_adoption_acknowledged')
        stage('05-note-after-custody',py('adopt_matched_generator_outcome_notes.py','after'),
            DOC/'reports/native-matched-generator-outcome-adoption-01.json','native38matched_outcome_notes_exact_and_scientific_records_unchanged')
        stage('06-browser-preparation',['node',HERE/'prepare_matched_portfolio_browser.cjs'])
        stage('07-actual-native-browser',['node',HERE/'native_project_browser.cjs',RUN/'native-portfolio-ui-accepted-01',
            BROWSER,NOTE_JOB/'browser-actions.json','300000'],BROWSER/'report.json','passed_native_project_UI_actions')
        state.update(status='native_custody_and_browser_complete_actual_image_inspection_pending',finished_unix=time.time(),
            active_stage=None,actual_UI_image_inspection_pending=True,project_left_open=True)
    except BaseException as error:
        state.update(status='serial_publication_stopped_requires_inspection',error_type=type(error).__name__,error=str(error),
            stopped_unix=time.time(),child_may_remain_live=state.get('active_child') is not None
                and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(STATUS,state)


if __name__ == '__main__':
    run()
