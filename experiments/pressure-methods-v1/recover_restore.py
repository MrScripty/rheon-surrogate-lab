"""Reconcile one denied restore graph, then restore the already completed fit.

The original failed dispatch remains immutable. This controller never trains.
A second failure requires inspecting this new journal before any continuation.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/dependency-migration-v1/lab/experiments'
sys.path[:0] = [str(SOURCE / 'dependency-migration-v1'),
               str(SOURCE / 'pressure-throughput-feedback-v1'),
               str(ROOT / 'runs/dependency-migration-v1/Tuldok')]
from dependency_config import decode, digest, require
from qualify_typed_pressure import Journal, OwnerHTTP, await_drained, restore_bindings, sha, write
import agent_api
import run_study


def recover(study_path, recovery):
    require(not recovery.exists(), 'Fresh recovery journal required; never replay mutations')
    intent = decode((study_path / 'intent.json').read_bytes())
    assembly_path = Path(intent['assembly'])
    require(digest(assembly_path) == intent['assembly_sha256'], 'Original assembly changed')
    assembly = decode(assembly_path.read_bytes())
    require(not (study_path / 'receipt.json').exists(), 'Study already collected')
    failed_path = study_path / 'mutations/11-restore-graph.json'
    failed = decode(failed_path.read_bytes())
    require(failed['status'] == 'unresolved' and failed['http_status'] == 401
            and failed['code'] == 'unauthenticated'
            and failed['operation'] == 'graphs.create_external', 'Exact denied restore dispatch required')

    def acknowledged(name):
        row = decode((study_path / 'mutations' / name).read_bytes())
        require(row['status'] == 'acknowledged', 'Original mutation must be acknowledged')
        return row['result']

    project = acknowledged('01-open-project.json')
    published = acknowledged('03-publish-model.json')
    study = acknowledged('05-create-study.json')
    trial = acknowledged('06-prepare-trial.json')
    fit_id = acknowledged('09-admit-training.json')['run']['id']
    acknowledged('10-close-study.json')
    owner = OwnerHTTP(agent_api, intent['endpoint'], intent['catalog_id'],
                      ROOT / 'runs/tuldok-performance-v1/private/owner-session.txt')
    training = await_drained(owner, fit_id, timeout=30)
    evidence = owner.call('runs.model_evidence', dict(run_id=fit_id, split='validation'))
    require(evidence['state'] == 'verified', 'Completed training evidence required')
    require(not owner.call('studies.get', dict(study_id=study['id']))['study']['open'],
            'Original training study must remain closed')
    document = decode((ROOT / assembly['project']['path']).read_bytes())
    require(digest(ROOT / assembly['project']['path']) == assembly['project']['sha256'], 'Project changed')
    reference = next(row for row in document['recipes'] if row['id'] == 'restore')
    require(digest(ROOT / reference['path']) == reference['sha256'], 'Restore recipe changed')
    definition = decode((ROOT / reference['path']).read_bytes())
    require(restore_bindings(definition, project, training) == failed['arguments']['inputs'],
            'Current published training artifacts differ from denied restore bindings')
    graphs = owner.call('graphs.list', {})
    matches = [graph for graph in graphs if graph['name'] == failed['arguments']['name']
               or any(node.get('parameters', {}).get('project_id') == project['id']
                      and node.get('parameters', {}).get('recipe_id') == 'restore'
                      for node in graph.get('nodes', []))]
    require(not matches, 'A restore graph already exists; reconcile it instead of creating another')
    recovery.mkdir()
    (recovery / 'mutations').mkdir()
    write(recovery / 'reconciliation.json', dict(status='restore_not_admitted',
        original_failure_sha256=digest(failed_path), original_failure_http_status=401,
        original_study=str(study_path), training_run_id=fit_id,
        training_state=training['state'], training_custody=training['custody_phase'],
        catalog_graphs_examined=len(graphs), matching_restore_graphs=0,
        optimizer_updates_to_execute=0, automatic_retry=False))
    journal = Journal(recovery / 'mutations')

    def mutate(phase, operation, arguments):
        return journal.mutation(phase, operation, arguments, lambda: owner.call(operation, arguments))

    graph = mutate('restore-graph', 'graphs.create_external', failed['arguments'])
    preflight = owner.call('graphs.preflight_external', dict(graph_id=graph['id'], revision=graph['revision']))
    body = dict(graph_id=graph['id'], graph_revision=graph['revision'],
                expected_digest=preflight['material_sha256'],
                consumer_actor_id=owner.identity['actor_id'], ttl_seconds=3600)
    approval = journal.mutation('approve-restore', 'runs.approve_external', body,
        lambda: owner.raw('POST', '/api/workspace/runs/approve-external', body))
    restored_admit = mutate('admit-restore', 'runs.submit_external', dict(
        request_key='recovered-' + sha(dict(study=str(study_path), recovery=str(recovery)))[:24],
        graph_id=graph['id'], graph_revision=graph['revision'],
        expected_digest=preflight['material_sha256'], approval_id=approval['id']))
    print('CPU restore admitted: ' + restored_admit['id'], flush=True)
    restored = await_drained(owner, restored_admit['id'], timeout=2100)
    args = SimpleNamespace(output=study_path, assembly=assembly_path)
    run_study.collect(args, assembly, owner, training, restored, published, study, trial, evidence,
                      time.monotonic() - (time.time() - intent['started_unix']))
    write(recovery / 'receipt.json', dict(status='completed', training_run_id=fit_id,
        restore_run_id=restored['id'], original_study_receipt_sha256=digest(study_path / 'receipt.json'),
        original_failure_preserved_sha256=digest(failed_path), optimizer_updates_executed=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    recover(args.study.resolve(), args.output.resolve())
