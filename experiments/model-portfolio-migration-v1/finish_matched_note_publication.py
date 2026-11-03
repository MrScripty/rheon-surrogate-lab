"""Continue after the reconciled, mutation-free composed-note precheck failure."""
from pathlib import Path
import os
import subprocess
import time

from import_common_generator_measurements import ROOT,RUN,DOC,CONTROLLER,read,save,binding,require,header
from finish_matched_portfolio_publication import require_terminal_import

HERE = Path(__file__).resolve().parent
JOB = RUN/'matched-note-publication-finish-02'
NOTE_JOB = RUN/'native-matched-generator-outcome-adoption-01'
BROWSER = RUN/'native-matched-portfolio-browser-01'


def run():
    require(not JOB.exists() and NOTE_JOB.exists() and not (NOTE_JOB/'status.json').exists()
            and not list(NOTE_JOB.glob('*.request.json')) and not BROWSER.exists(),
            'Fresh continuation from accepted before-custody required; existing note requests need reconciliation')
    correction_path = DOC/'reports/composed-matched-note-selection-correction-01.json'
    correction = read(correction_path)
    require(correction['status'] == 'composed_file_owner_selection_corrected_before_any_note_mutation'
            and correction['corrected_fixture_checks_passed'] == 3 and correction['actual_before_custody_passed'],
            'Qualified composed-file selection correction required')
    for spec in [correction['original_serial_owner'],correction['original_tool_preserved'],correction['corrected_selector'],
                 correction['regression_test'],correction['before_custody']]:
        require(binding(ROOT/spec['path']) == spec,'Precheck reconciliation evidence changed')
    prior = read(ROOT/correction['original_serial_owner']['path'])
    require(not Path(f'/proc/{prior["pid"]}').exists() and prior['status'] == 'serial_publication_stopped_requires_inspection'
            and prior['active_child'] is None and not prior['child_may_remain_live'],'Original serial owner must be terminal')
    audit = read(DOC/'reports/native-matched-generator-GPU-import-custody-01.json')
    require_terminal_import(read(ROOT/audit['original_owner']['path']))
    raw,current = header()
    before = read(NOTE_JOB/'before-custody.json')
    require(current['checkpoint'] == correction['checkpoint'] == audit['checkpoint'] == before['checkpoint'],
            'Accepted native checkpoint changed before note continuation')
    tools = ['adopt_matched_generator_outcome_notes_02.py','adopt_matched_generator_outcome_notes.py',
        'import_common_generator_measurements.py','finish_matched_portfolio_publication.py',
        'prepare_matched_portfolio_browser.cjs','native_project_browser.cjs']
    frozen = [binding(HERE/name) for name in tools]+[binding(Path(__file__))]
    JOB.mkdir()
    state = dict(status='starting_reconciled_note_publication',pid=os.getpid(),created_unix=time.time(),stages=[],active_child=None,
        correction=binding(correction_path),starting_checkpoint=current['checkpoint'],frozen_tools=frozen,
        new_model_forwards=0,payload_copies=0,automatic_retries=False,full_portfolio_complete=False)
    save(JOB/'status.json',state)
    try:
        python = ROOT/'runs/dependency-migration-v1/app-env/bin/python'
        stages = [('01-note-adoption',[python,'-B',HERE/'adopt_matched_generator_outcome_notes_02.py','adopt'],
                    NOTE_JOB/'status.json','native_matched38note_adoption_acknowledged'),
                  ('02-note-custody',[python,'-B',HERE/'adopt_matched_generator_outcome_notes_02.py','after'],
                    DOC/'reports/native-matched-generator-outcome-adoption-01.json','native38matched_outcome_notes_exact_and_scientific_records_unchanged'),
                  ('03-browser-preparation',['node',HERE/'prepare_matched_portfolio_browser.cjs'],None,None),
                  ('04-actual-native-browser',['node',HERE/'native_project_browser.cjs',CONTROLLER,BROWSER,
                    NOTE_JOB/'browser-actions.json','300000'],BROWSER/'report.json','passed_native_project_UI_actions')]
        for name,args,evidence,expected in stages:
            require(all(binding(ROOT/s['path']) == s for s in frozen),'Frozen continuation source changed')
            log = JOB/(name+'.log')
            item = dict(name=name,command=[str(a) for a in args],started_unix=time.time(),log=str(log.relative_to(ROOT)))
            state['stages'].append(item)
            state.update(status='reconciled_note_publication_stage_running',active_stage=name)
            save(JOB/'status.json',state)
            with log.open('x') as stream:
                child = subprocess.Popen([str(a) for a in args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child'] = item['pid'] = child.pid
                save(JOB/'status.json',state)
                code = child.wait()
            item.update(exit_code=code,finished_unix=time.time(),log_binding=binding(log))
            state['active_child'] = None
            save(JOB/'status.json',state)
            require(code == 0,'Original continuation stage failed; inspect retained evidence; no retry')
            if evidence is not None:
                require(read(evidence)['status'] == expected,'Continuation acceptance differs')
                item['accepted_evidence'] = binding(evidence)
            save(JOB/'status.json',state)
            print('Accepted reconciled stage:',name,flush=True)
        state.update(status='matched_notes_custody_and_browser_complete_actual_image_inspection_pending',finished_unix=time.time(),
            active_stage=None,actual_UI_image_inspection_pending=True,project_left_open=True)
    except BaseException as error:
        state.update(status='reconciled_note_publication_stopped_requires_inspection',error_type=type(error).__name__,error=str(error),
            child_may_remain_live=state['active_child'] is not None and Path(f'/proc/{state["active_child"]}').exists())
        raise
    finally:
        save(JOB/'status.json',state)


if __name__ == '__main__':
    run()
