"""Combine unchanged scientific chart evidence with corrected current-note UI."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,header,read,save,binding,require

OLD_BROWSER = RUN/'native-pressure-publication-browser-01'
BROWSER = RUN/'native-pressure-status-browser-01'
JOB = RUN/'native-pressure-reproduction-status-adoption-01'
REPORT = DOC/'reports/native-pressure-publication-UI-01.json'
OLD_IMAGES = ('all150-pressure.png','all150-cpm16-GPU-time.png','selected7-pressure.png','selected7-GPU-time.png',
              'selected7-cpm8-grid16-pressure.png','pressure-original-source.png')
NEW_IMAGES = ('pressure-status-guide9.png','fit-8992e19d389ea016-current-reproduction.png',
              'pressure-status-card-original-receipt.png','pressure-status-readable-overview.png')


def complete_browser(folder,actions_path):
    report = read(folder/'report.json')
    require(report['status'] == 'passed_native_project_UI_actions' and report['browser_stopped'] and not report['errors']
            and report['steps'] == read(actions_path),'Original actual browser actions must pass and browser stop')
    require(report['api_requests'] and all(row.get('status') == 200 and row.get('complete') for row in report['api_requests']),
            'Every recorded API request must finish HTTP200')
    return report


def run():
    require(not REPORT.exists(),'Preserve original pressure UI acceptance;never overwrite a prior acceptance')
    old = complete_browser(OLD_BROWSER,RUN/'native-pressure-outcome-adoption-01/browser-actions.json')
    current_UI = complete_browser(BROWSER,JOB/'browser-actions.json')
    old_expect = read(RUN/'native-pressure-outcome-adoption-01/browser-expectations.json')
    new_expect = read(JOB/'browser-expectations.json')
    require(len(old['steps']) == 1666 and len(old_expect['queries']) == 13
            and old_expect['numeric_presence_assertions'] == 1092 and old_expect['updated_model_summaries'] == 23,
            'Complete original150subject/7subject chart workload required')
    require(new_expect['guide_revision'] == 9 and new_expect['updated_model_summaries'] == 23
            and new_expect['summary_paragraph_assertions'] == 138 and new_expect['overview_exact_full_text_checked'],
            'Current corrected23summary/overview workload required')
    old_custody_path = DOC/'reports/native-pressure-outcome-adoption-01.json'
    old_custody = read(old_custody_path)
    custody_path = DOC/'reports/native-pressure-reproduction-status-adoption-01.json'
    custody = read(custody_path)
    before = read(ROOT/custody['before']['path'])
    require(custody['previous_note_custody'] == binding(old_custody_path)
            and before['previous_note_custody'] == binding(old_custody_path)
            and before['checkpoint'] == old_custody['checkpoint']
            and custody['all_non_note_scientific_records_exact'] and custody['measurement_sources_and_payloads_unchanged']
            and custody['all23five_scientific_summary_fields_unchanged'] and custody['overview_words_unchanged'],
            'Original13chart evidence may carry forward only across exact scientific/measurement custody')
    _,project = header()
    require(project['checkpoint'] == custody['checkpoint'] == new_expect['current_checkpoint']
            and binding(PROJECT/'.tuldok/project.json')['sha256'] == custody['canonical_header_sha256']
            and custody['guide_revision'] == 9 and custody['current_reproduction_status_corrected'],
            'Current native checkpoint must equal accepted correction custody and actual browser')
    receipt_path = BROWSER/'visual-inspection.json'
    inspection = read(receipt_path)
    require(inspection['status'] == 'actual_corrected_pressure_UI_captures_inspected'
            and inspection['current_browser_report'] == binding(BROWSER/'report.json')
            and inspection['prior_chart_browser_report'] == binding(OLD_BROWSER/'report.json')
            and not inspection['new_blocking_findings'] and inspection['I57_resolved'],
            'Current four note captures and unchanged six chart/source captures must actually be inspected')
    seen = {spec['path']:spec for spec in inspection['captures_actually_viewed']}
    required = [folder/name for folder,names in ((OLD_BROWSER,OLD_IMAGES),(BROWSER,NEW_IMAGES)) for name in names]
    require(all(str(path.relative_to(ROOT)) in seen for path in required),'All ten declared actual image inspections required')
    for spec in seen.values():
        require(binding(ROOT/spec['path']) == spec,'Actually inspected screenshot bytes changed')
    guide = read(PROJECT/'notes/00-models.portfolio.json')
    require(len(guide['models']) == 205 and len(guide['relationships']) == 26,'205models/26authored relationships required')
    save(REPORT,dict(status='all150pressure_comparison_and23corrected_outcome_notes_actual_native_UI_accepted',created_unix=time.time(),
        checkpoint=project['checkpoint'],canonical_header_sha256=custody['canonical_header_sha256'],native_measurement_sources=31,
        pressure_fits_in_unified_comparison=150,updated_model_summaries=23,guide_revision=9,models=205,authored_comparisons=26,
        prior_actual_chart_and_note_actions=1666,current_corrected_note_actions=len(current_UI['steps']),
        prior_complete_HTTP200_reads=len(old['api_requests']),current_complete_HTTP200_reads=len(current_UI['api_requests']),
        requested_chart_comparisons=13,displayed_numeric_presence_assertions=1092,current_six_paragraph_summary_checks=23,
        chart_evidence_checkpoint=old_custody['checkpoint'],scientific_chart_evidence_carried_across_exact_non_note_custody=True,
        captures_actually_viewed=list(seen.values()),every_capture_manually_viewed=False,both_browsers_stopped=True,project_left_open=True,
        I57_resolved=True,overview_words_preserved_and_prose_reflowed=True,model_or_dataset_copies=0,new_model_forwards=0,
        numerical_UI_assertion_scope='Correct native maximum-row numeric textpresence,not capturedHTTPbody equality or fresh physical reexecution. Independent exact scalar/metric/case/filter proof and native custody are separate.',
        visualization_scope='All150subjects remain selectable in a tall chart;captured all150viewports show only middle portions. Seven-subject comparisons provide readable complete labels/axes. Existing number-word spacing/table-header wrapping and near-zero linear compression remain separate follow-ups.',
        evidence=[binding(path) for path in (OLD_BROWSER/'report.json',BROWSER/'report.json',receipt_path,old_custody_path,custody_path)],
        producer=binding(Path(__file__)),near_instant_Open_achieved=False,full_portfolio_complete=False))
    print('Actual150pressure comparisons and23corrected summaries accepted;I57resolved;full migration remains active.',flush=True)


if __name__ == '__main__':
    run()
