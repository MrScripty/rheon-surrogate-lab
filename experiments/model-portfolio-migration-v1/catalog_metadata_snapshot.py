"""Capture stored Run/graph summaries in an enforced read-only owner snapshot.

The legacy HTTP list verifies transitive artifact custody and timed out. This
one-time inventory uses upstream Tuldok's ReadonlyCatalog instead. It selects
only public metadata fields, never full Run control documents. Stored artifact
declarations remain unverified claims. This is not a backup or an extraction.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
OLD_SERVICE = ROOT / 'runs/dependency-migration-v1/Tuldok-large-input-v3'
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok'
CATALOG = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres')
CATALOG_ID = 'a7da4a8de53541a2a0e2aad08005bb71'
PID = 3802934


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration():
    process = Path('/proc') / str(PID)
    args = [v.decode() for v in (process / 'cmdline').read_bytes().split(b'\0') if v]
    if str(OLD_SERVICE / 'app.py') not in args or str(CATALOG) not in args:
        raise ValueError('Existing service identity differs')
    # Read only the same named connection setting used by the verified service.
    prefix = b'TULDOK_DATABASE_URL='
    with (process / 'environ').open('rb') as stream:
        value = bytearray()
        while True:
            byte = stream.read(1)
            if not byte or byte == b'\0':
                if value.startswith(prefix):
                    return bytes(value[len(prefix):]).decode()
                value.clear()
                if not byte:
                    break
            else:
                value.extend(byte)
    raise ValueError('Verified service database configuration unavailable')


def verify_owner():
    path = ROOT / 'runs/tuldok-performance-v1/private/owner-session.txt'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096 or info.st_mode & 0o077:
            raise ValueError('Existing private session file differs')
        token = stream.read(4096).decode('ascii').strip()
    request = urllib.request.Request('http://127.0.0.1:8091/api/access/session',
                                     headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=10) as response:
        identity = json.loads(response.read())
    if identity.get('kind') != 'owner' or identity.get('catalog_id') != CATALOG_ID:
        raise ValueError('Current owner of exact catalog required')


def snapshot(output, summary_path):
    if output.exists() or summary_path.exists():
        raise ValueError('Fresh metadata snapshot paths required')
    verify_owner()
    descriptor = CATALOG / '.tuldok.json'
    descriptor_sha = sha(descriptor)
    if json.loads(descriptor.read_text())['dataset_id'] != CATALOG_ID:
        raise ValueError('Catalog directory identity differs')
    lock = CATALOG / '.metadata-init.lock'
    lock_before = lock.stat()
    sys.path.insert(0, str(SOURCE))
    from project_transfer import ReadonlyCatalog
    from metadata import connection_info
    db = ReadonlyCatalog(connection_info(configuration()), 'catalog_' + CATALOG_ID, CATALOG_ID, CATALOG)
    try:
        read_only = db.execute("SELECT current_setting('transaction_read_only')").fetchone()[0]
        isolation = db.execute("SELECT current_setting('transaction_isolation')").fetchone()[0]
        if read_only != 'on' or isolation != 'repeatable read':
            raise ValueError('Enforced read-only repeatable-read boundary required')
        # Fixed allowlist: exclude worker/specification/control/credential state.
        runs = []
        query = """SELECT id, jsonb_build_object(
            'id', id, 'state', document::jsonb->'state',
            'mode', document::jsonb->'mode',
            'created_at', document::jsonb->'created_at',
            'finished_at', document::jsonb->'finished_at',
            'execution_kind', document::jsonb->'execution_kind',
            'custody_phase', document::jsonb->'custody_phase',
            'parent_run_id', document::jsonb->'parent_run_id',
            'material_sha256', document::jsonb->'material_sha256',
            'graph_sha256', document::jsonb->'graph_sha256',
            'graph_id', document::jsonb#>'{graph,id}',
            'graph_name', document::jsonb#>'{graph,name}',
            'graph_recipe', document::jsonb#>'{graph,recipe}',
            'artifacts', document::jsonb->'artifacts'
            ) FROM workspace_runs ORDER BY id"""
        for row in db.execute(query):
            value = row[1]
            if not isinstance(value, dict) or value['id'] != row[0]:
                raise ValueError('Stored Run summary representation differs')
            runs.append(value)
        graphs = [dict(id=r[0], revision=r[1], name=r[2], recipe=r[3]) for r in db.execute(
            "SELECT id,revision,document::jsonb->>'name',document::jsonb->>'recipe' FROM workspace_graphs ORDER BY id")]
        tx_snapshot = db.execute('SELECT txid_current_snapshot()::text').fetchone()[0]
    finally:
        db.close()
    after = lock.stat()
    if (lock_before.st_ino, lock_before.st_size, lock_before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns) or sha(descriptor) != descriptor_sha:
        raise ValueError('Catalog descriptor or initialization lock changed during observation')
    result = dict(schema='rheon.stored-catalog-metadata-observation.v1', created_unix=time.time(),
                  status='complete_stored_run_and_graph_metadata_snapshot', catalog_id=CATALOG_ID,
                  transaction_read_only=read_only, transaction_isolation=isolation,
                  transaction_snapshot=tx_snapshot, runs=runs, graphs=graphs,
                  inference_calls=0, database_writes=0, artifact_payloads_opened=0,
                  authorization='User explicitly authorized the read-only metadata snapshot after reviewing the credential-source and direct-database-access question.',
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 (Path(__file__).resolve(), SOURCE / 'project_transfer.py', SOURCE / 'metadata.py')},
                  limitations=['Stored metadata only: no transitive-input, artifact-content or scientific-accuracy verification.',
                               'Not a backup, a stopped-source lease, an extraction or current execution authority.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary = {k: v for k, v in result.items() if k not in ('runs', 'graphs')}
    summary.update(output=str(output), output_sha256=sha(output), runs=len(runs), graphs=len(graphs),
                   run_states=dict(Counter(r['state'] for r in runs)),
                   execution_kinds=dict(Counter(r['execution_kind'] for r in runs)),
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
    try:
        snapshot(args.output, args.summary)
    except Exception as error:
        # Connection exceptions may contain private conninfo; keep them redacted.
        print('Read-only catalog observation failed:', type(error).__name__,
              getattr(error, 'name', '') if isinstance(error, ModuleNotFoundError) else '', file=sys.stderr)
        raise SystemExit(1)
