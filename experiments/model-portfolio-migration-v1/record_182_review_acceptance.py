"""Record the first bounded final-review batch after real UI/image inspection.

The operator must inspect the listed captures before invoking this recorder.
It performs no API, neural or reference-solver work and changes no project data.
"""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
RUN = ROOT/'runs/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
VIEWED = ['182-reviewed-guide.png',
    'fit-01fc69dbfaf391dd-summary.png',
    'fit-1ccc04176f26bb5a-summary.png',
    'fit-29706aa9da6fd913-summary.png',
    'fit-2dd3c2d95b4c0233-summary.png',
    'fit-385e70a586037324-summary.png',
    'fit-3a5fac7e44249154-summary.png',
    'fit-4f5ce39a5ba8aace-summary.png',
    'fit-4fa8cfbcce378b75-summary.png',
    'fit-5b3b1aef2efcee62-summary.png',
    'fit-74f4d0fc1a4c22b4-summary.png',
    'retained-comparison-and-receipts.png']


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def run():
    output = DOC/'reports/native-182-review-UI-01.json'
    coverage_path = DOC/'reports/review-coverage-93.json'
    state_path = DOC/'work-state-93.json'
    require(not any(p.exists() for p in [output,coverage_path,state_path]),'Preserve original 182-review acceptance')
    audit_path = DOC/'reports/native-182-method-review-adoption-01.json'
    audit = read(audit_path)
    owner = read(RUN/'method-review-native-adoption-15/status.json')
    require(owner['status'] == 'native_forty_note_adoption_acknowledged'
            and not Path(f'/proc/{owner["pid"]}').exists()
            and audit['reviewed_fit_cards'] == 182 and audit['changed_note_records'] == 40
            and audit['all_non_note_scientific_records_exact'] and audit['original_note_histories_exact'],
            'Terminal original owner and complete canonical custody required')
    folder = RUN/'native-182-review-browser-01'
    browser = read(folder/'report.json')
    actions_path = RUN/'method-review-native-adoption-15/browser-actions.json'
    actions = read(actions_path)
    require(browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and not browser['errors'] and browser['steps'] == actions and len(actions) == 617,
            'All 617 original UI actions must complete without errors')
    require(browser['api_requests'] and all(r['status'] == 200 and r['complete'] for r in browser['api_requests']),
            'All observed UI API requests must complete HTTP 200')
    screenshots = list(folder.glob('*.png'))
    require(len(screenshots) == 40 and len(set(VIEWED)) == 12 and all((folder/n).is_file() for n in VIEWED),
            'Exact captured and actually-inspected screenshot populations required')
    require(sum(a['kind'] == 'wait_text' and a.get('selector') == '#project-portfolio-detail' for a in actions) == 224,
            'All 37 six-paragraph summaries and comparison/receipt text must be asserted')
    index_path = PROJECT/'notes/portfolio-index.json'
    fits = read(index_path)['fits']
    native = {f['item_id'] for f in fits if f.get('method_review_status') == 'captured_source_and_receipt_reviewed'}
    require(len(fits) == 205 and len(native) == 182,'Exact current native reviewed count required')
    header_path = PROJECT/'.tuldok/project.json'
    header = read(header_path)
    require(header['checkpoint'] == audit['checkpoint'] == owner['checkpoint']
            and binding(header_path)['sha256'] == audit['canonical_header_sha256'],'Native checkpoint changed after custody/UI')
    source_coverage_path = DOC/'reports/review-coverage-91.json'
    source_coverage = read(source_coverage_path)
    pending = [r for r in source_coverage['unpublished'] if r['fit_item_id'] not in native]
    require(source_coverage['source_reviewed'] == 205 and len(pending) == 23
            and native|{r['fit_item_id'] for r in pending} == {f['item_id'] for f in fits},
            'Complete source/native/remaining set accounting differs')
    for pid in [13702,13722]:
        os.kill(pid,0)
    now = time.time()
    result = dict(status='native182guide_and_all37new_summaries_actual_UI_passed',created_unix=now,
        checkpoint=header['checkpoint'],native_note_revisions=40,new_reviewed_model_summaries=37,native_reviewed_total=182,
        guide_revision=5,literal_summary_paragraphs=1092,authored_comparisons=26,
        actual_UI_actions=617,complete_HTTP200_API_reads=len(browser['api_requests']),
        all_new_six_paragraph_summaries_asserted=True,native_comparison_and_original_receipt_disclosure_passed=True,
        retained_Experiment_progression_coexists=True,screenshots_captured=40,
        representative_screenshots_actually_viewed=VIEWED,all40screenshots_manually_viewed=False,
        visual_findings=['Inspected summary paragraphs remain wrapped and readable.',
            'I41 authored prose spacing and I45 long graph-node title clipping remain open cleanup items.'],
        verification_browser_stopped=True,project_left_open=True,native_UI_closed=False,
        new_model_forwards=0,payload_copies=0,full_portfolio_complete=False,producer=binding(Path(__file__)),
        evidence=[binding(audit_path),binding(folder/'report.json'),binding(actions_path),*[binding(folder/n) for n in VIEWED]])
    write(output,result)
    coverage = dict(source_coverage,created_unix=now,status='205_source_reviews_182_native_23_unpublished_exact',
        native_published=182,unpublished=pending,full_portfolio_complete=False,
        evidence=[binding(source_coverage_path),binding(index_path),binding(audit_path),binding(output)])
    write(coverage_path,coverage)
    prior_path = DOC/'work-state-92.json'
    state = read(prior_path)
    state.update(created_unix=now,previous_state='work-state-92.json',previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress',phase='all205_source_reviewed_182_native_UI_accepted_23_pending')
    state['review_status'].update(native_adopted_and_audited=182,native_UI_accepted=182,unpublished_source_reviews=23,
        native_checkpoint=header['checkpoint'],guide_revision=5,literal_summary_paragraphs=1092,
        actual_UI_actions=617,complete_HTTP200_API_reads=len(browser['api_requests']),screenshots_captured=40,
        representative_screenshots_actually_viewed=12,working_index_reviewed_after_owner_staging=182,
        working_index_is_native_acceptance=False,new_source_reviews_this_continuation=0)
    state['native_publication'].update(status=owner['status'],original_owner_verified_live=False,
        original_owner_terminal=True,native_ACK_and_after_custody_pending=False,actual_UI_pending=False,
        native_operation_seconds=owner['native_operation_wall_seconds'],checkpoint=header['checkpoint'],
        exact_canonical_custody_passed=True,payload_bodies_rehashed=False,payload_preservation=audit['payload_preservation'])
    state['project'].update(checkpoint=header['checkpoint'],canonical_records=audit['canonical_records'],
        canonical_record_bytes=audit['canonical_record_bytes'])
    state['verification_this_continuation']=dict(new_source_reviewed_fits=0,native_notes_accepted=40,
        new_native_reviews_accepted=37,actual_UI_actions=617,screenshots_actually_inspected=12,
        new_model_forwards=0,new_reference_generation=False,new_project_dataset_or_weight_copies=0,
        unique_weights_deleted=0,service_restarts=0)
    state['remaining_objective'][0]='Publish the final 23 completed source reviews / 26 notes with custody and actual UI checks.'
    state['next_slice']='Recheck GPU/host quiet and dispatch the prepared six-model sharing benchmark without heavy concurrent UI/publication work; prepare/qualify the final 23-review batch against this accepted checkpoint and serialize its native write with clean timing.'
    state['evidence']=[binding(prior_path),binding(audit_path),binding(output),binding(coverage_path),
        binding(DOC/'reports/complete-source-batch-02-tool-preparation-01.json')]
    write(state_path,state)
    print('182 native reviews and actual UI accepted; all205 source-reviewed, final23 reviews await publication. Goal remains active.')


if __name__ == '__main__':
    run()
