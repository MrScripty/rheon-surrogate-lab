"""Accept actual chart/card UI only after recorded manual screenshot inspection."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,header,read,save,binding,require

BROWSER = RUN/'native-pressure-publication-browser-01'
JOB = RUN/'native-pressure-outcome-adoption-01'
REPORT = DOC/'reports/native-pressure-publication-UI-01.json'
REQUIRED_IMAGES = (
    'all150-pressure.png','all150-cpm16-GPU-time.png','selected7-pressure.png','selected7-GPU-time.png',
    'selected7-cpm8-grid16-pressure.png','pressure-original-source.png','pressure-205-model-guide.png',
    'fit-8992e19d389ea016-pressure-outcome.png','pressure-card-original-receipt.png','pressure-updated-overview.png')


def run():
    require(not REPORT.exists(),'Preserve original native pressure UI acceptance')
    browser_path = BROWSER/'report.json'
    browser = read(browser_path)
    expectations = read(JOB/'browser-expectations.json')
    actions = read(JOB/'browser-actions.json')
    inspection_path = BROWSER/'visual-inspection.json'
    inspection = read(inspection_path)
    note_path = DOC/'reports/native-pressure-outcome-adoption-01.json'
    notes = read(note_path)
    imports_path = DOC/'reports/native-all150-pressure-import-custody-01.json'
    imports = read(imports_path)
    raw,current = header()
    require(browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and not browser['errors'] and browser['steps'] == actions,
            'All actual original browser actions must pass and its isolated browser must be stopped')
    require(expectations['numeric_presence_assertions'] == 1092 and len(expectations['queries']) == 13
            and expectations['updated_model_summaries'] == 23 and expectations['summary_paragraph_assertions'] == 138,
            'Full declared150subject/7subject comparison and23summary workload required')
    require(notes['status'] == 'native26pressure_outcome_notes_exact_and_scientific_records_unchanged'
            and current['checkpoint'] == notes['checkpoint'] and notes['canonical_header_sha256'] == binding(PROJECT/'.tuldok/project.json')['sha256']
            and notes['changed_note_records'] == 26 and notes['pressure_fit_cards'] == 23 and notes['guide_revision'] == 8
            and imports['native_sources'] == 31,'Current exact published pressure checkpoint required')
    require(inspection['status'] == 'actual_pressure_UI_captures_inspected'
            and inspection['browser_report'] == binding(browser_path)
            and not inspection['new_blocking_findings'],'Actual screenshot inspection receipt with no blocking finding required')
    viewed = {Path(spec['path']).name:spec for spec in inspection['captures_actually_viewed']}
    require(set(REQUIRED_IMAGES) <= viewed.keys(),'All ten declared workflow/summary/source screenshots must be inspected')
    for name,spec in viewed.items():
        require(ROOT/spec['path'] == BROWSER/name and binding(ROOT/spec['path']) == spec,'Actually inspected screenshot bytes differ')
    api = browser['api_requests']
    require(api and all(item.get('status') == 200 and item.get('complete') for item in api),
            'Every recorded browser API read must finish HTTP200')
    guide = read(PROJECT/'notes/00-models.portfolio.json')
    require(len(guide['models']) == 205 and len(guide['relationships']) == 26,'205models/26authored relationships required')
    save(REPORT,dict(status='all150pressure_comparison_and23outcome_notes_actual_native_UI_accepted',created_unix=time.time(),
        checkpoint=current['checkpoint'],canonical_header_sha256=notes['canonical_header_sha256'],native_measurement_sources=31,
        pressure_fits_in_unified_comparison=150,updated_model_summaries=23,guide_revision=8,models=205,authored_comparisons=26,
        actual_UI_actions=len(actions),complete_HTTP200_reads=len(api),requested_chart_comparisons=13,
        displayed_numeric_presence_assertions=1092,six_paragraph_summary_checks=23,captures=len(list(BROWSER.glob('*.png'))),
        captures_actually_viewed=list(viewed.values()),every_capture_manually_viewed=False,browser_stopped=True,project_left_open=True,
        all_import_and_note_custody_exact=True,raw_process_bodies_copied=0,model_or_dataset_copies=0,new_model_forwards=0,
        numerical_UI_assertion_scope='Numeric textpresence in the correct native maximum row;not captured HTTPbody equality or fresh physical reexecution. Exact scalar bits/metric/case/filter identities are qualified independently and under native custody.',
        visualization_scope='All150subjects remain selectable;the7subject filtered comparisons provide readable labels. Existing linear-axis near-zero compression remains a nonblocking follow-up.',
        evidence=[binding(path) for path in (browser_path,JOB/'browser-expectations.json',inspection_path,note_path,imports_path)],
        producer=binding(Path(__file__)),near_instant_Open_achieved=False,full_portfolio_complete=False))
    print('Actual150pressure charts,23outcome summaries and ten viewed captures accepted;full migration remains active.',flush=True)


if __name__ == '__main__':
    run()
