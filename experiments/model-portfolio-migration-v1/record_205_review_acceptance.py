"""Record complete source-review publication after actual UI/image inspection.

Inspect all eight listed captures before invoking this recorder. This accepts
method-summary publication only; benchmark, reproduction and portable lifecycle
requirements remain open.
"""
import os
from pathlib import Path
import time

from record_182_review_acceptance import DOC, RUN, PROJECT, binding, read, require, write

VIEWED = ['205-reviewed-guide.png',
    'fit-c2d989595d367aad-summary.png',
    'fit-d1c66c9301a5b890-summary.png',
    'fit-d6d89efe3c45e76a-summary.png',
    'fit-e50811f9ce86666b-summary.png',
    'fit-f07ea413add20163-summary.png',
    'fit-fb8ce15096322ec9-summary.png',
    'retained-comparison-and-receipts.png']


def run():
    output = DOC/'reports/native-205-review-UI-01.json'
    coverage_path = DOC/'reports/review-coverage-95.json'
    state_path = DOC/'work-state-95.json'
    require(not any(p.exists() for p in (output,coverage_path,state_path)),'Preserve complete-review acceptance')
    audit_path = DOC/'reports/native-205-method-review-adoption-01.json'
    audit = read(audit_path)
    owner_path = RUN/'method-review-native-adoption-16/status.json'
    owner = read(owner_path)
    require(owner['status'] == 'native_twentysix_note_adoption_acknowledged'
            and not Path(f'/proc/{owner["pid"]}').exists()
            and audit['reviewed_fit_cards'] == 205 and audit['changed_note_records'] == 26
            and audit['all_non_note_scientific_records_exact'] and audit['original_note_histories_exact'],
            'Terminal acknowledged owner and complete canonical custody required')
    folder = RUN/'native-205-review-browser-01'
    browser_path = folder/'report.json'
    browser = read(browser_path)
    actions_path = RUN/'method-review-native-adoption-16/browser-actions.json'
    actions = read(actions_path)
    require(browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and not browser['errors'] and browser['steps'] == actions and len(actions) == 393
            and all(r['status'] == 200 and r['complete'] for r in browser['api_requests']),
            'All393real UI actions and complete HTTP200 reads required')
    require(len(list(folder.glob('*.png'))) == 26 and len(set(VIEWED)) == 8
            and all((folder/name).is_file() for name in VIEWED),'Exact captures and inspected image population required')
    require(sum(a['kind'] == 'wait_text' and a.get('selector') == '#project-portfolio-detail' for a in actions) == 140,
            'All23six-paragraph summaries plus retained comparison/evidence must be asserted')
    index_path = PROJECT/'notes/portfolio-index.json'
    index = read(index_path)
    fits = index['fits']
    require(len(fits) == len({f['item_id'] for f in fits}) == 205
            and all(f.get('method_review_status') == 'captured_source_and_receipt_reviewed' for f in fits),
            'All205native fit summaries must be source-reviewed')
    header_path = PROJECT/'.tuldok/project.json'
    current = read(header_path)
    require(current['checkpoint'] == audit['checkpoint'] == owner['checkpoint']
            and binding(header_path)['sha256'] == audit['canonical_header_sha256'],
            'Native checkpoint changed after custody/UI')
    coverage_before_path = DOC/'reports/review-coverage-93.json'
    coverage = read(coverage_before_path)
    require(coverage['source_reviewed'] == 205 and len(coverage['unpublished']) == 23,
            'Exact previous source/native coverage required')
    for pid in (13702,13722):
        os.kill(pid,0)
    now = time.time()
    result = dict(status='native205guide_and_all23new_summaries_actual_UI_passed',created_unix=now,
        checkpoint=current['checkpoint'],native_note_revisions=26,new_reviewed_model_summaries=23,native_reviewed_total=205,
        guide_revision=6,literal_summary_paragraphs=1230,authored_comparisons=26,
        actual_UI_actions=393,complete_HTTP200_API_reads=len(browser['api_requests']),
        all_new_six_paragraph_summaries_asserted=True,native_comparison_and_original_receipt_disclosure_passed=True,
        retained_Experiment_progression_coexists=True,screenshots_captured=26,
        representative_screenshots_actually_viewed=VIEWED,all26screenshots_manually_viewed=False,
        visual_findings=['Inspected method summaries remain wrapped and readable.',
            'Authored spacing and long graph-node label cleanup remain separately tracked as I41/I45.'],
        verification_browser_stopped=True,project_left_open=True,native_UI_closed=False,
        new_model_forwards=0,payload_copies=0,full_portfolio_complete=False,
        producer=binding(Path(__file__)),helper=binding(Path(__file__).with_name('record_182_review_acceptance.py')),
        evidence=[binding(audit_path),binding(browser_path),binding(actions_path),*[binding(folder/n) for n in VIEWED]])
    write(output,result)
    coverage.update(created_unix=now,status='all205_source_reviews_native_published_and_actual_UI_accepted',
        native_published=205,unpublished=[],full_portfolio_complete=False,
        evidence=[binding(coverage_before_path),binding(index_path),binding(audit_path),binding(output)])
    write(coverage_path,coverage)
    previous_path = DOC/'work-state-94.json'
    state = read(previous_path)
    state.update(created_unix=now,previous_state='work-state-94.json',previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress',phase='all205_method_reviews_native_custody_and_actual_UI_accepted')
    state['review_status'].update(native_adopted_and_audited=205,native_UI_accepted=205,unpublished_source_reviews=0,
        native_checkpoint=current['checkpoint'],guide_revision=6,literal_summary_paragraphs=1230,
        actual_UI_actions=393,complete_HTTP200_API_reads=len(browser['api_requests']),screenshots_captured=26,
        representative_screenshots_actually_viewed=8,working_index_reviewed_after_owner_staging=205)
    state['native_publication'].update(status=owner['status'],original_owner_verified_live=False,original_owner_terminal=True,
        native_ACK_and_after_custody_pending=False,actual_UI_pending=False,exact_canonical_custody_passed=True,
        native_operation_seconds=owner['native_operation_wall_seconds'],checkpoint=current['checkpoint'],
        last_accepted_checkpoint=current['checkpoint'],payload_preservation=audit['payload_preservation'],payload_bodies_rehashed=False)
    state['project'].update(checkpoint=current['checkpoint'],canonical_records=audit['canonical_records'],
        canonical_record_bytes=audit['canonical_record_bytes'])
    state['verification_this_continuation']=dict(new_native_reviews_accepted=23,actual_UI_actions=393,screenshots_inspected=8,
        new_source_reviews=0,CPU_neural_forwards=0,CUDA_neural_forwards=0,new_reference_generation=False,
        service_restarts=0,model_or_dataset_copies=0,unique_weights_deleted=0)
    state['remaining_objective'][0]='Ingest and visually qualify the four native common-generator GPU tables; all205source/method reviews are now natively accepted.'
    state['next_source_review_slice']='All205source reviews and native summaries are complete; no source-review backlog remains.'
    state['next_slice']='Run the prepared common-generator import owner before/import/after phases once, follow original request handles, then actual chart/filter/GPU-label UI checks. Continue remaining model/state benchmarking, GPU-only constructor qualification and retention/reproduction/portable work.'
    state['evidence']=[binding(previous_path),binding(audit_path),binding(output),binding(coverage_path),
        binding(Path(__file__).with_name('import_common_generator_measurements.py'))]
    write(state_path,state)
    print('All205method summaries now native/custody/UI accepted; benchmarking and full migration remain incomplete.')


if __name__ == '__main__':
    run()
