"""Accept matched charts/outcome notes after actual UI and image inspection.

The operator must inspect the captures named by visual-inspection.json before
running this recorder. Browser assertions establish text presence and native
interaction; they do not independently reexecute physical predictions.
"""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,CONTROLLER,read,save,header,binding,require,digest

FOLDER = RUN/'native-matched-portfolio-browser-01'
JOB = RUN/'native-matched-generator-outcome-adoption-01'
REPORT = DOC/'reports/native-matched-portfolio-UI-01.json'


def run():
    require(not REPORT.exists(),'Preserve original matched portfolio UI acceptance')
    imported_path = DOC/'reports/native-matched-generator-GPU-import-custody-01.json'
    notes_path = DOC/'reports/native-matched-generator-outcome-adoption-01.json'
    imported,notes = read(imported_path),read(notes_path)
    require(imported['status'] == 'four_readable_matched_GPU_tables_native_custody_exact'
            and notes['status'] == 'native38matched_outcome_notes_exact_and_scientific_records_unchanged'
            and notes['changed_note_records'] == 38 and notes['original_note_histories_exact']
            and notes['all_non_note_scientific_records_exact'],'Both native custody gates must pass')
    browser_path = FOLDER/'report.json'
    browser,actions = read(browser_path),read(JOB/'browser-actions.json')
    expected = read(JOB/'browser-expectations.json')
    require(browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and not browser['errors'] and browser['steps'] == actions
            and all(r.get('status') == 200 and r.get('complete') for r in browser['api_requests']),
            'Every planned real native UI action and complete HTTP200 read must pass')
    queries = expected['queries']
    require(len(queries) == 40 and {q['K'] for q in queries} == {1,2,3,4}
            and expected['numeric_presence_assertions'] == 390 and expected['matched_model_summaries'] == 35
            and expected['summary_paragraph_assertions'] == 210,'Complete planned UI populations required')
    numeric = [a for a in actions if a['kind'] == 'wait_text' and a.get('selector','').startswith('#project-measurements-maxima tbody tr:')]
    summary = [a for a in actions if a['kind'] == 'wait_text' and a.get('selector') == '#project-portfolio-detail']
    require(len(numeric) == 390 and len(summary) == 215,'Actual chart and summary assertion populations differ')
    reads = [r for r in browser['api_requests'] if r['path'].endswith('/measurements/compare')]
    require(len(reads) == 44,'Forty chart queries and four initial-load comparisons required')
    screenshots = [a['name'] for a in actions if a['kind'] == 'screenshot']+['native-project.png']
    require(len(screenshots) == len(set(screenshots)) == 83
            and {p.name for p in FOLDER.glob('*.png')} == set(screenshots),'Complete screenshot population required')
    inspection_path = FOLDER/'visual-inspection.json'
    inspection = read(inspection_path)
    viewed = inspection['captures_actually_viewed']
    require(inspection['status'] == 'representative_native_captures_actually_viewed' and len(set(viewed)) == len(viewed)
            and len(viewed) >= 10 and set(viewed) <= set(screenshots) and inspection['findings'],
            'Record actual representative image inspection without claiming all images were viewed')
    for k in range(1,5):
        require(any(name.startswith(f'K{k}-') for name in viewed),'Each K must have an inspected capture')
    require(any('-cpm' in name for name in viewed) and any('source-inspection' in name for name in viewed)
            and 'matched-205-model-guide.png' in viewed and any(name.endswith('-matched-outcome.png') for name in viewed)
            and 'matched-card-original-receipt-and-grid-table.png' in viewed
            and 'matched-overview-timing-scope-correction.png' in viewed,'Inspect filters/source/guide/card/overview')
    raw,current = header()
    require(current['checkpoint'] == notes['checkpoint'] and digest(raw) == notes['canonical_header_sha256'],
            'Actual UI checkpoint must match accepted native note custody')
    guide = read(PROJECT/'notes/00-models.portfolio.json')
    require(len(guide['models']) == 205 and len(guide['relationships']) == 26,'Model/relationship population changed')
    controller = read(CONTROLLER/'controller.json')
    require(controller['status'] == 'ready' and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid')),
            'Keep the original user UI running')
    save(REPORT,dict(status='matched39configurations_and35outcome_notes_actual_native_UI_accepted',created_unix=time.time(),
        checkpoint=current['checkpoint'],original_generator_fits=35,F64_assembly_executions=4,measured_scalars=24960,native_sources=29,
        actual_UI_actions=len(actions),requested_chart_comparisons=40,initial_load_comparisons=4,displayed_numeric_presence_assertions=390,
        all35_six_paragraph_summaries_asserted=True,guide_revision=7,total_models=205,authored_relationships=26,
        complete_HTTP200_reads=len(browser['api_requests']),screenshots_captured=83,captures_actually_viewed=viewed,
        every_screenshot_manually_viewed=len(viewed)==83,visual_findings=inspection['findings'],
        historical_instrumented_table_scope_correction_adopted_and_read=True,
        numerical_scope='Custody verifies all24960scalars; owner checks12unfiltered maxima; browser asserts390maximum numeric text strings across40queries. Browser does not retain complete response bodies or reexecute physical predictions.',
        statistic='Native plots show maximum selected per-case statistic; outcome-table timing summaries report median of per-case medians.',
        timing_device='NVIDIA GeForce RTX5090LaptopGPU; primary forward-validation hook disabled; separate paired calibration retained.',
        browser_stopped=True,project_left_open=True,near_instant_Open_achieved=False,new_neural_forwards=0,payload_copies=0,full_portfolio_complete=False,
        producer=binding(Path(__file__)),evidence=[binding(p) for p in [imported_path,notes_path,browser_path,JOB/'browser-actions.json',
            JOB/'browser-expectations.json',inspection_path,*[FOLDER/name for name in viewed]]]))
    print('Matched GPU charts,35outcomes,original receipts and historical scope correction accepted in actual native UI.',flush=True)


if __name__ == '__main__':
    run()
