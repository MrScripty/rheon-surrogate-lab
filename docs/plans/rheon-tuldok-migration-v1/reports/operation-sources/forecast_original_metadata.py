"""Read-only drift forecast, never a replacement for the stopped bookend."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/dependency-migration-v1'
sys.path.insert(0, str(BASE / 'Tuldok'))
from backup_metadata import read_catalog, read_sequences, summaries
from metadata import Metadata

receipt = BASE / 'ops/m3-live-backup-forecast.json'
assert not receipt.exists()
selection = json.loads((BASE / 'ops/live-selection.private.json').read_bytes())
original = json.loads((BASE / 'ops/preflight.json').read_bytes())
backup = json.loads((BASE / 'original-schema8-backup-02/backup-receipt.json').read_bytes())
started = time.monotonic()
db = None
try:
    db = Metadata(Path(original['catalog']), dsn=selection['dsn'], create=False, schema_version=None)
    db.connection.execute('SET default_transaction_read_only = on')
    assert db.dataset_id == backup['dataset_id'] and db.schema_version == 8
    observed, sequences = summaries(read_catalog(db)), read_sequences(db)
    changed = sorted(n for n in observed.keys() | backup['tables'].keys() if observed.get(n) != backup['tables'].get(n))
    result = {'schema': 'rheon-live-backup-metadata-forecast-v1',
              'status': 'metadata_matches_backup02' if not changed and sequences == backup['sequences'] else 'drift_requires_review',
              'changed_tables': changed, 'sequences_equal': sequences == backup['sequences'],
              'catalog_id': db.dataset_id, 'schema_version': db.schema_version,
              'database_default_transaction_read_only': True, 'authoritative_db_writes': 0,
              'is_stopped_bookend_or_gate': False, 'files_and_modes_verified': False,
              'elapsed_s': time.monotonic() - started}
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
except Exception as error:
    print(json.dumps({'status': 'forecast_stopped', 'error_class': type(error).__name__}))
    sys.exit(1)
finally:
    if db is not None:
        db.close()
