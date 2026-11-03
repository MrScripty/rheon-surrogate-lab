"""Join the actual restart/custody/API/browser evidence for the read-path fix."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, RUN, PROJECT, CONTROLLER, read, save, binding, require
from component_material import observed_file


JOB = RUN/'native-measurement-source-read-rollout-02'


def run():
    output = DOC/'reports/measurement-source-reads-native-acceptance-01.json'
    require(not output.exists(), 'Preserve original actual native acceptance')
    before_path,after_path = JOB/'before-custody.json',JOB/'after-custody.json'
    before,after = read(before_path),read(after_path)
    require(before['status'] == 'exact_project_custody_before' and after['status'] == 'exact_project_custody_after'
            and before['facts'] == after['facts'] and after['canonical_bodies_freshly_hashed']
            and not after['payload_bodies_rehashed'] and after['project_or_payload_copies'] == 0,
            'Exact canonical/header/root and all prior payload identities must survive the source-only restart')
    facts = after['facts']
    require(facts['canonical_records'] == 53976 and facts['canonical_record_bytes'] == 1789673543
            and facts['payload_files'] == 8734 and facts['payload_bytes'] == 16541094986
            and binding(PROJECT/'.tuldok/project.json')['sha256'] == facts['header_sha256'], 'Current accepted checkpoint required')
    commit_path = DOC/'reports/measurement-source-reads-source-commit-01.json'
    commit = read(commit_path)
    integration_path = JOB/'source-integration.json'
    integration = read(integration_path)
    controller = read(CONTROLLER/'controller.json')
    require(integration['status'] == 'qualified_source_installed_same_catalog_UI_resumed'
            and integration['accepted_commit'] == controller['source_commit'] == commit['commit']
            and integration['old_app_and_PostgreSQL_stopped'] and controller['keep_open']
            and controller['status'] == 'ready' and integration['catalog_id'] == controller['catalog_id']
            and integration['origin'] == controller['origin'], 'Qualified same-installation source rollout required')
    evidence = [binding(path) for path in (before_path,after_path,commit_path,integration_path)]
    for action in ('close','open'):
        state_path = JOB/(action+'-status.json')
        state = read(state_path)
        require(state['status'] == 'native_'+action+'_acknowledged'
                and not Path('/proc',str(state['pid'])).exists()
                and read(JOB/(action+'.response.json'))['status'] == 200, 'Original lifecycle owner must acknowledge and stop')
        evidence.extend(binding(path) for path in (state_path,JOB/(action+'.request.json'),JOB/(action+'.response.json')))
    finish_path = JOB/'finish-status.json'
    finish = read(finish_path)
    require(finish['status'] == 'native_custody_reads_and_UI_passed_manual_images_pending'
            and not Path('/proc',str(finish['pid'])).exists()
            and [row['name'] for row in finish['steps']] == ['after-custody','native-reads','native-browser']
            and all(row['returncode'] == 0 for row in finish['steps']), 'Original serial continuation must complete without replay')
    evidence.append(binding(finish_path))
    paths = [RUN/'measurement-source-reads-native-baseline-01',RUN/'measurement-source-reads-native-candidate-01']
    values,read_states = [],[]
    for job in paths:
        state = read(job/'status.json')
        require(state['status'] == 'two_native_measurement_reads_acknowledged'
                and not Path('/proc',str(state['pid'])).exists()
                and state['checkpoint'] == facts['checkpoint'] and state['native_mutations'] == state['model_forwards'] == 0,
                'Both original native read owners must acknowledge unchanged checkpoint and stop')
        for row in state['calls']:
            observed_file(ROOT,row['response']['path'],row['response'])
        sources = read(job/'sources.response.json')['result']['value']
        require(sources['total'] == len(sources['items']) == 31, 'All original sources required')
        stripped = [{k:v for k,v in row.items() if k != 'availability'} for row in sources['items']]
        metadata = read(job/'metadata.response.json')['result']['value']
        values.append(dict(sources=stripped,metadata=metadata))
        read_states.append(state)
        evidence.extend(binding(job/name) for name in ('status.json','sources.response.json','metadata.response.json'))
    require(values[0] == values[1], 'Every original source document and selected packet metadata must remain literal')
    browser_path = RUN/'native-measurement-source-read-browser-01/report.json'
    browser = read(browser_path)
    expected_path = JOB/'browser-expectations.json'
    expected = read(expected_path)
    actions_path = JOB/'browser-actions.json'
    require(binding(actions_path)['sha256'] == expected['actions_sha256']
            and browser['status'] == 'passed_native_project_UI_actions'
            and browser['steps'] == read(actions_path) and len(browser['steps']) == 68
            and browser['browser_stopped'] and not browser['errors']
            and len(browser['api_requests']) == 11
            and all(row.get('status') == 200 and row.get('complete') for row in browser['api_requests'])
            and expected['chart_selections'] == 3 and expected['numeric_presence_assertions'] == 14,
            'All prepared literal native/browser selections and complete HTTP reads required')
    viewed = []
    observations = {
        'cell-K4-cpm100-anchor-time.png':'Two original cell implementations show GPU ms/physical output with separate exact maxima rows.',
        'temporal-K4-cpm50-120-time.png':'Six original temporal variants show GPU time with18matching observations and visible execution labels.',
        'temporal-K4-cpm50-120-error.png':'Six pressure-error series preserve their scales and rows; near-zero axis/readability limitations remain tracked.'}
    require(set(observations) == set(expected['screenshots']), 'Exactly three actual inspected chart captures required')
    for name,note in observations.items():
        path = browser_path.parent/name
        viewed.append(dict(**binding(path),actually_viewed=True,observation=note))
    ended_path = JOB/'user-reported-resource-contention-ended-01.json'
    ended = read(ended_path)
    require(ended['created_unix'] <= read_states[1]['created_unix'], 'Candidate API reads must follow the reported CPU-contention end')
    evidence.extend(binding(path) for path in (browser_path,expected_path,actions_path,ended_path))
    result = dict(status='measurement_source_read_rollout_actual_native_UI_and_custody_accepted',created_unix=time.time(),
        producer=binding(Path(__file__)),installed_source_commit=commit['commit'],checkpoint=facts['checkpoint'],
        canonical_records=facts['canonical_records'],canonical_record_bytes=facts['canonical_record_bytes'],
        payload_files=facts['payload_files'],payload_bytes=facts['payload_bytes'],
        canonical_payload_root_and_header_custody_exact=True,original_31source_documents_and_packet_metadata_exact=True,
        actual_browser_actions=68,actual_complete_HTTP200_reads=11,original_chart_selections=3,
        exact_numeric_presence_assertions=14,actual_viewed_captures=viewed,
        uninspected_additional_capture='native-project.png',all_original_read_and_browser_owners_terminal=True,
        native_source_list_observed_seconds=dict(baseline=read_states[0]['calls'][0]['wall_seconds'],candidate=read_states[1]['calls'][0]['wall_seconds']),
        native_packet_metadata_observed_seconds=dict(baseline=read_states[0]['calls'][1]['wall_seconds'],candidate=read_states[1]['calls'][1]['wall_seconds']),
        native_Close_seconds=read(JOB/'close-status.json')['native_call_wall_seconds'],
        native_Open_seconds=read(JOB/'open-status.json')['native_call_wall_seconds'],
        whole_Open_resource_scope='Mixed reported CPU contention and post-contention intervals;not a clean comparison.',
        API_resource_scope='Candidate reads occur after user confirmed CPU contention ended. Baseline/candidate are individual observations,not a controlled statistical speedup.',
        near_instant_Open_achieved=False,whole_Open_speedup_qualified=False,I36_resolved=False,
        source_list_still_slow=True,source_list_followup='Existing ProjectArtifacts.inspect_many can avoid one canonical-root verification per listed source;needs actual caller regression and qualification before changing source.',
        project_left_open=True,UI_origin=controller['origin'],new_model_forwards=0,project_dataset_or_weight_copies=0,
        new_parameter_fits=0,new_reference_simulations=0,unique_weights_deleted=0,
        source_qualification=binding(ROOT/commit['qualification']['path']),evidence=evidence,full_portfolio_complete=False)
    save(output,result)
    print(result['status'],'68actions/11HTTP200/3viewed charts/14literal values;project left open.',flush=True)


if __name__ == '__main__':
    run()
