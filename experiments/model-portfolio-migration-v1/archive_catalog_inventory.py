"""Select public Run/graph summaries from the verified historical backup.

No database connection or credential records are selected. The compacted backup
still retains its exact original metadata; payload restoration is unnecessary
for this metadata observation. Historical facts do not become current authority.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / 'runs/dependency-migration-v1/original-schema8-backup-02'
PUBLIC_FIELDS = ('id', 'state', 'mode', 'created_at', 'finished_at', 'execution_kind',
                 'custody_phase', 'parent_run_id', 'material_sha256', 'graph_sha256', 'artifacts')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(output, summary_path):
    if output.exists() or summary_path.exists():
        raise ValueError('Fresh archive inventory outputs required')
    receipt = json.loads((BACKUP / 'backup-receipt.json').read_text())
    path = BACKUP / 'metadata.json'
    identity = sha(path)
    if identity != receipt['metadata_sha256']:
        raise ValueError('Original backup metadata differs')
    metadata = json.loads(path.read_text())
    if metadata['dataset_id'] != receipt['dataset_id'] or metadata['schema_version'] != 8:
        raise ValueError('Historical catalog identity differs')

    def documents(table):
        selected = metadata['tables'][table]
        columns = selected['columns']
        id_column, doc_column = columns.index('id'), columns.index('document')
        for row in selected['rows']:
            doc = json.loads(row[doc_column])
            if doc['id'] != row[id_column]:
                raise ValueError('Historical record identity mismatch')
            yield doc

    runs = []
    for document in documents('workspace_runs'):
        row = {k: document.get(k) for k in PUBLIC_FIELDS}
        graph = document.get('graph') or {}
        row.update(graph_id=graph.get('id'), graph_name=graph.get('name'), graph_recipe=graph.get('recipe'))
        runs.append(row)
    graphs = [dict(id=d['id'], revision=d['revision'], name=d.get('name'), recipe=d.get('recipe'))
              for d in documents('workspace_graphs')]
    result = dict(schema='rheon.archived-catalog-metadata-observation.v1', created_unix=time.time(),
                  status='captured_verified_original_metadata', catalog_id=metadata['dataset_id'],
                  archived_schema_version=metadata['schema_version'], runs=runs, graphs=graphs,
                  source_metadata=dict(path=str(path.relative_to(ROOT)), sha256=identity),
                  backup_receipt=dict(path=str((BACKUP / 'backup-receipt.json').relative_to(ROOT)),
                                      sha256=sha(BACKUP / 'backup-receipt.json')),
                  source_sha256=sha(Path(__file__)),
                  credential_records_selected=False, database_opened=False, inference_calls=0,
                  limitations=['This is historical Run/graph metadata, not a current live catalog.',
                               'Recorded artifact declarations require independent payload/custody checks.',
                               'The compacted backup needs materialization before database restoration.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary = {k: v for k, v in result.items() if k not in ('runs', 'graphs')}
    summary.update(output=str(output), output_sha256=sha(output), runs=len(runs), graphs=len(graphs),
                   run_states=dict(Counter(r['state'] for r in runs)),
                   declared_artifacts=sum(len(r.get('artifacts') or []) for r in runs))
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    collect(args.output, args.summary)
