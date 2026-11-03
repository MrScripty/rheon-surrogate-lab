"""Record completed native publication/UI and exact remaining review coverage.

Run only after the original browser terminates and the listed screenshots have
actually been inspected. No API call, model execution or project mutation.
"""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
RUN = ROOT/'runs/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
VIEWED = [
    '145-reviewed-guide.png',
    'fit-0b2b6b6a8c42a47a-summary.png',
    'fit-10d3a512b41ab2a6-summary.png',
    'fit-12380a9f8e9b0ffe-summary.png',
    'fit-561e02c747b96415-summary.png',
    'fit-6a1cb8a085cb73f0-summary.png',
    'fit-a9e199535346c3d9-summary.png',
    'fit-bec10b73f24cc066-summary.png',
    'fit-c8896a4c9a730de5-summary.png',
    'fit-d106ea978eb8ed52-summary.png',
    'fit-d18b434d52b73791-summary.png',
    'retained-comparison-and-receipts.png',
]
PENDING_REPORTS = [
    'captured-pressure-qualification-method-reviews-01.json',
    'captured-current-pressure-feedback-method-reviews-01.json',
    'captured-full-coarse-pressure-method-review-01.json',
    'captured-pressure-pass-fine-tuning-method-reviews-01.json',
]


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
    ui_path = DOC/'reports/native-145-review-UI-01.json'
    coverage_path = DOC/'reports/review-coverage-89.json'
    state_path = DOC/'work-state-89.json'
    require(not any(p.exists() for p in [ui_path,coverage_path,state_path]),'Preserve original accepted evidence')
    owner = read(RUN/'method-review-native-adoption-14/status.json')
    require(owner['status'] == 'native_thirtytwo_note_adoption_acknowledged'
            and not Path(f'/proc/{owner["pid"]}').exists(),'Original native owner must be terminal')
    audit_path = DOC/'reports/native-145-method-review-adoption-01.json'
    audit = read(audit_path)
    require(audit['reviewed_fit_cards'] == 145 and audit['changed_note_records'] == 32
            and audit['all_non_note_scientific_records_exact'] and audit['original_note_histories_exact'],
            'Completed native note/scientific custody required')
    browser_path = RUN/'native-145-review-browser-01'
    browser = read(browser_path/'report.json')
    actions_path = RUN/'method-review-native-adoption-14/browser-actions.json'
    actions = read(actions_path)
    require(browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and browser['errors'] == [] and browser['steps'] == actions and len(actions) == 489,
            'All original actual UI actions must complete without errors')
    require(browser['api_requests'] and all(r['status'] == 200 and r['complete'] for r in browser['api_requests']),
            'Every observed UI API read must complete HTTP 200')
    shots = sorted(p.name for p in browser_path.glob('*.png'))
    require(len(shots) == 32 and len(VIEWED) == len(set(VIEWED)) == 12 and set(VIEWED) <= set(shots),
            'Exact captured/actually-inspected screenshot populations required')
    require(sum(a['kind'] == 'wait_text' and a.get('selector') == '#project-portfolio-detail' for a in actions) == 176,
            'All 29 six-paragraph cards, retained comparison lesson and original receipt disclosure must be asserted')
    index_path = PROJECT/'notes/portfolio-index.json'
    fits = read(index_path)['fits']
    native = {r['item_id']:r for r in fits if r.get('method_review_status') == 'captured_source_and_receipt_reviewed'}
    require(len(fits) == 205 and len(native) == 145,'Native index must contain exactly 145 reviewed fits')
    header = read(PROJECT/'.tuldok/project.json')
    require(header['checkpoint'] == audit['checkpoint'] == owner['checkpoint']
            and binding(PROJECT/'.tuldok/project.json')['sha256'] == audit['canonical_header_sha256'],
            'Native project changed after custody/UI')
    controller = read(RUN/'native-portfolio-ui-accepted-01/controller.json')
    for pid in [13702,13722]:
        os.kill(pid,0)
    pending = {}
    evidence = [binding(audit_path),binding(index_path)]
    for name in PENDING_REPORTS:
        path = DOC/'reports'/name
        report = read(path)
        require(report['new_model_forwards'] == report['native_mutations'] == 0,'Read-only source reviews required')
        for bound in [report['producer'],*report['helpers']]:
            require(binding(ROOT/bound['path']) == bound,'Executed source-review helper changed')
        for row in report['rows']:
            fid = row['fit_item_id']
            require(fid not in native and fid not in pending,'New source reviews must be unique and unpublished')
            fit = next(f for f in fits if f['item_id'] == fid)
            require(binding(ROOT/row['original_receipt']['path']) == row['original_receipt']
                    and fit['receipt_sha256'] == row['original_receipt']['sha256'],'Original receipt identity differs')
            require(all(row.get(k) for k in ['method','purpose','experiment','outcome','lesson','reproduction']),
                    'All six scientific review paragraphs required')
            pending[fid] = dict(fit_item_id=fid,name=fit['name'],report=str(path.relative_to(ROOT)))
        evidence.append(binding(path))
    require(len(pending) == 22,'Expected 22 additional unpublished reviews')
    reviewed = set(native)|set(pending)
    tasks = {}
    for task in sorted({f['task'] for f in fits}):
        group = [f for f in fits if f['task'] == task]
        n = sum(f['item_id'] in reviewed for f in group)
        tasks[task] = dict(total=len(group),source_reviewed=n,remaining=len(group)-n)
    require(len(reviewed) == 167 and tasks['full_state_generator']['source_reviewed'] == 35
            and tasks['pressure_proposal']['remaining'] == 38,'Exact portfolio coverage differs')
    now = time.time()
    ui = dict(status='native145guide_and_all29new_summaries_actual_UI_passed',created_unix=now,
        checkpoint=audit['checkpoint'],native_note_revisions=32,new_reviewed_model_summaries=29,native_reviewed_total=145,
        guide_revision=4,literal_summary_paragraphs=870,authored_comparisons=26,actual_UI_actions=len(actions),
        complete_HTTP200_API_reads=len(browser['api_requests']),all_new_six_paragraph_summaries_asserted=True,
        native_comparison_and_original_receipt_disclosure_passed=True,retained_Experiment_progression_coexists=True,
        screenshots_captured=len(shots),representative_screenshots_actually_viewed=VIEWED,
        all32screenshots_manually_viewed=False,
        visual_findings=['Inspected guide and representative summary paragraphs remain wrapped and readable.',
                         'Original generic bootstrap labels remain ambiguous; use fit identity and source context until authored-label cleanup.',
                         'I45: the long independent guided-pressure title clips lower identity/review text inside its fixed-height graph node; the selected summary remains readable.',
                         'I41 authored number/word spacing cleanup remains open.'],
        verification_browser_stopped=True,project_left_open=True,native_UI_closed=False,
        new_model_forwards=0,payload_copies=0,full_portfolio_complete=False,
        evidence=[binding(browser_path/'report.json'),binding(actions_path),binding(audit_path),
                  *[binding(browser_path/name) for name in VIEWED]],producer=binding(Path(__file__)))
    coverage = dict(status='167_unique_source_reviews_145_native_22_unpublished_exact',created_unix=now,
        fits=205,source_reviewed=167,native_published=145,source_reviews_remaining=38,task_counts=tasks,
        unpublished=list(pending.values()),remaining=[dict(fit_item_id=f['item_id'],name=f['name'],task=f['task'])
            for f in fits if f['item_id'] not in reviewed],full_portfolio_complete=False,evidence=evidence)
    state = dict(schema='rheon.model-portfolio-migration.work-state.v1',created_unix=now,previous_state='work-state-88.json',
        previous_goal_turn_classification='verified_wait',current_goal_turn_classification='progress',goal_status='active',
        full_objective_achieved=False,phase='167_source_reviews_145_native_22_unpublished_38_remaining',
        review_status=dict(total_fits=205,source_reviewed=167,native_adopted_and_audited=145,
            native_UI_accepted=145,source_reviews_remaining=38,unpublished_source_reviews=22,
            new_source_reviews_since_state88=22,new_source_reviews_this_continuation=6,
            native_checkpoint=audit['checkpoint'],guide_revision=4,literal_summary_paragraphs=870,
            actual_UI_actions=len(actions),complete_HTTP200_API_reads=len(browser['api_requests']),
            screenshots_captured=len(shots),representative_screenshots_actually_viewed=len(VIEWED)),
        native_publication=dict(original_pid=owner['pid'],original_exec_session=10999,original_owner_terminal=True,
            native_status=owner['status'],native_operation_seconds=owner['native_operation_wall_seconds'],
            native_note_revisions=32,exact_canonical_custody_passed=True,payload_bodies_rehashed=False,
            payload_preservation=audit['payload_preservation'],project_left_open=True),
        UI=dict(origin=controller['origin'],launcher_pid=13702,app_pid=13722,keep_open=True,
            project_left_open=True,source_commit=owner['source_commit'],restart_count=7,
            actual_UI_passed=True,verification_browser_stopped=True,near_instant_Open_achieved=False),
        project=dict(path=str(PROJECT.relative_to(ROOT)),project_id=header['project_id'],checkpoint=header['checkpoint'],
            canonical_records=audit['canonical_records'],canonical_record_bytes=audit['canonical_record_bytes']),
        comparison_findings=dict(local_smoother_composition_gap='I43',four_training_vs_two_deployment_corrections='I44',
            C56_completed_fit_failed_simulation_qualification=True,old_physics_preserved=True),
        sharing_generator_benchmark=dict(status='undispatched',blocking_compute_pid=36974,process='Overwatch.exe',
            game_interference=False,new_contended_timing_accepted=False,latest_occupancy='Recheck GPU before original dispatch.'),
        verification_this_continuation=dict(new_source_reviewed_fits=6,new_model_forwards=0,new_reference_generation=False,
            new_project_dataset_or_weight_copies=0,unique_weights_deleted=0,service_restarts=0),
        remaining_objective=['38 pressure-proposal source reviews; publish 22 completed reviews plus subsequent work.',
            'All-model/state/variant consistent old-Rheon accuracy and GPU timing; sharing campaign and constructor qualification.',
            'Same-case near-exact F64 anchor comparison, I43 composed solvers and I44 recurrence variants.',
            '35 retained parameter states need proper native retention; 936 outside registrations need dispositions.',
            'GPU-only clean reproduction and scripts for separate new-Rheon generation; retain old reference epoch.',
            '600-output own-history drift references/qualification and complete portable lifecycle.',
            'Practical fast Open, I45 long-node clipping and final authored-label/prose cleanup.'],
        evidence=[binding(audit_path),*evidence[1:]],state_scope='Compact current authority; previous accepted benchmark evidence remains in state88 and referenced reports.')
    write(ui_path,ui)
    coverage['evidence'].append(binding(ui_path))
    write(coverage_path,coverage)
    state['evidence'].extend([binding(ui_path),binding(coverage_path)])
    write(state_path,state)
    print('Accepted 145 native reviews; 167 source-reviewed, 22 unpublished, 38 source reviews remain. Full goal remains active.')


if __name__ == '__main__':
    run()
