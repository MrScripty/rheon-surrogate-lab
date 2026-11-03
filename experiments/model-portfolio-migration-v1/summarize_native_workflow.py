"""Verify retained native owner/browser receipts and both portable directories.

This reads no credentials, executes no model and calls no service. The relocated
directory is the disposable test copy identified by its original copy receipt.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
AUTHORING = RUN / 'native-authoring-01'


def read(path):
    return json.loads(path.read_text())


def evidence(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def canonical(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*')
            if p.is_file() and '.tuldok-local' not in p.relative_to(directory).parts}


def run():
    copy = read(AUTHORING / 'relocation-copy.json')
    original, relocated = Path(copy['original']), Path(copy['copy'])
    baseline = read(AUTHORING / 'canonical-before-browser.json')
    assert baseline == copy['file_sha256'] == canonical(original) == canonical(relocated)
    assert all((original / name).stat().st_ino != (relocated / name).stat().st_ino
               for name in baseline)
    project = read(original / '.tuldok/project.json')
    controllers = []
    for number in (1, 2, 3):
        path = RUN / f'native-controller-{number:02}/controller.json'
        state = read(path)
        assert state['status'] == 'stopped' and state['app_stopped'] and state['postgres_stopped']
        assert not state['live_catalog_accessed'] and not state['live_credentials_read']
        controllers.append(dict(**evidence(path), catalog_id=state['catalog_id'], stopped=True))
    assert len({s['catalog_id'] for s in controllers}) == 3
    opens = []
    for name in ('40-independent-cold-open', '50-relocated-cold-open'):
        path = AUTHORING / (name + '.response.json')
        response = read(path)
        assert response['status'] == 200 and response['result']['state'] == 'open'
        assert response['result']['project']['project_id'] == project['project_id']
        assert response['result']['project']['checkpoint'] == project['checkpoint']
        opens.append(evidence(path))
    assert read(AUTHORING / '50-relocated-cold-open.response.json')['result']['roots'] == ['portable']
    materials = []
    for identifier, publish, cold, moved in [('C107', '12', '42', '51'), ('C128', '16', '43', '52')]:
        paths = [AUTHORING / f'{publish}-model_publish.response.json',
                 AUTHORING / f'{cold}-cold-{identifier}-material.response.json',
                 AUTHORING / f'{moved}-relocated-{identifier}-material.response.json']
        values = []
        for path in paths:
            response = read(path)
            assert response['status'] == 200
            values.append(response['result']['value'])
        assert values[0] == values[1] == values[2]
        for prefix in (cold + '-cold-', moved + '-relocated-'):
            request = read(AUTHORING / f'{prefix}{identifier}-material.request.json')
            assert request['arguments']['verify'] is True
        materials.append(dict(identifier=identifier, ref=values[0]['ref'],
                              source_blobs=len(values[0]['material']['source_closure']),
                              exact_across_publication_and_both_cold_opens=True,
                              evidence=[evidence(p) for p in paths]))
    browser = []
    for number, count in [(2, 29), (3, 19), (6, 15)]:
        path = RUN / f'native-browser-{number:02}/report.json'
        result = read(path)
        assert result['status'] == 'passed_native_project_UI_actions'
        assert len(result['steps']) == count and result['errors'] == [] and result['browser_stopped']
        assert '--no-sandbox' not in result['flags'] and result['verification_only']
        browser.append(dict(**evidence(path), actions=count, exceptions=0,
                            browser_stopped=True, sandbox_enabled=True,
                            screenshot=evidence(path.parent / 'native-project.png')))
    audits = []
    for number in (2, 3):
        path = RUN / f'native-controller-{number:02}/private/cold-audit/events.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert any(item['event'] == 'audit_installed' for item in events)
        assert not any('blocked' in item for item in events)
        audits.append(dict(controller=number, installed=True, blocked_attempts=0,
                           log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(format='rheon_native_portability_qualification_v1', created_unix=time.time(),
        status='qualified_representative_lifecycle_and_captured_source_portability',
        project_path=str(original.relative_to(ROOT) / '.tuldok/project.json'),
        project_id=project['project_id'], checkpoint=project['checkpoint'],
        canonical_files=len(baseline), canonical_bytes=copy['bytes'],
        canonical_unchanged=True, relocated_independent_inodes=True,
        native_inventory_families=dict(sorted(Counter(x['family'] for x in project['inventory']).items())),
        controllers=controllers, cold_opens=opens, model_materials=materials,
        browser=browser, audit_observations=audits,
        audit_limit='Python absolute-path/event guard only; relative dir_fd reads, symlink resolution and native IO are not comprehensively instrumented. This is not a filesystem sandbox claim.',
        historical_failures=dict(material_schema_error='41-cold-C107-material.error.json',
            card_selector_attempts=['native-browser-04/report.json', 'native-browser-05/report.json'],
            card_selector_disposition='First search match was the overview. Native label selection in attempt06 verified the actual C128 card.'),
        Tuldok_application_source_changed=False, CPU_neural_forward=False, new_training_runs=0,
        new_Rheon_adopted=False, latency_claim=False,
        remaining=['Original checkpoint payload custody and original Run/Trial history not migrated.',
                   'Rich Experiment progression and scientific comparison are not integrated in the native project page.',
                   'All-model benchmark coverage, drift, source regeneration and complete project handoff remain pending.'])
    destination = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/native-portability-01.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(report=str(destination.relative_to(ROOT)), files=len(baseline),
                          browser_actions=sum(b['actions'] for b in browser),
                          published_materials_exact=len(materials), controllers_stopped=3)))


if __name__ == '__main__':
    run()
