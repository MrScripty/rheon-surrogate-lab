"""Back up and cold-restore the quiescent, reconciled current rehearsal catalog."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/dependency-migration-v1/Tuldok"
sys.path.insert(0, str(SOURCE))
from backup_metadata import (backup_catalog, canonical, digest, file_hash, read_catalog,
                             read_sequences, restore_catalog, summaries, tree_manifest)
from metadata import Metadata
from psycopg.conninfo import conninfo_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-receipt", type=Path, required=True)
    parser.add_argument("--reconcile-receipt", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--postgres-directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema_receipt.read_bytes())
    reconciled = json.loads(args.reconcile_receipt.read_bytes())
    assert reconciled["mode"] == "reconcile" and reconciled["result"]["status"] == "verified-completed"
    source = Path(schema["isolated_destination"]).resolve(strict=True)
    assert reconciled["catalog"] == str(source) and not reconciled["application_started"]
    backup, catalog = args.backup.resolve(), args.catalog.resolve()
    pg_directory = args.postgres_directory.resolve()
    assert all(path.is_relative_to(ROOT / "runs/dependency-migration-v1") for path in (source, backup, catalog))
    assert not backup.exists() and not catalog.exists() and not args.receipt.exists()
    assert pg_directory.is_relative_to(Path("/tmp")) and not pg_directory.exists()
    helper = SOURCE / "scripts/local_postgres.py"
    spec = importlib.util.spec_from_file_location("completed_roundtrip_postgres", helper)
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    source_dsn = pg.configuration(Path(schema["postgres_directory"]))
    original = conninfo_to_dict(json.loads((OPS / "live-selection.private.json").read_bytes())["dsn"])
    selected = conninfo_to_dict(source_dsn)
    assert any(selected.get(k) != original.get(k) for k in ("host", "port", "dbname", "user"))
    started = time.monotonic()
    backup_catalog(source, backup, dsn=source_dsn, quiesced=True)
    backed_up_at = time.monotonic()
    saved = json.loads((backup / "backup-receipt.json").read_bytes())
    assert saved["schema_version"] == 10 and saved["dataset_id"] == schema["dataset_id"]
    log_path = OPS / (pg_directory.name + "-init.private.log")
    with os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as log:
        subprocess.run([sys.executable, "-B", str(helper), "init", "--directory", str(pg_directory)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    target_dsn = pg.configuration(pg_directory)
    target = conninfo_to_dict(target_dsn)
    assert all(any(target.get(k) != other.get(k) for k in ("host", "port", "dbname", "user"))
               for other in (selected, original))
    restore_catalog(backup, catalog, dsn=target_dsn)
    restored_at = time.monotonic()
    db = Metadata(catalog, dsn=target_dsn, create=False)
    try:
        assert db.schema_version == 10 and db.dataset_id == saved["dataset_id"]
        assert summaries(read_catalog(db)) == saved["tables"]
        assert read_sequences(db) == saved["sequences"]
        marker_name = ".restore-receipt-" + digest(saved) + ".json"
        marker = {"kind": "postgresql-restore", "backup_sha256": digest(saved), "dataset_id": saved["dataset_id"]}
        assert (catalog / marker_name).read_bytes() == canonical(marker) + b"\n"
        assert tree_manifest(catalog, (marker_name,)) == saved["files"]
        stage = Path(reconciled["stage"]).resolve(strict=True)
        prepared = json.loads((stage / "prepared.json").read_bytes())
        receipt = {"schema": "rheon-completed-current-roundtrip-v1", "status": "passed",
                   "source_backup": str(backup), "source_backup_receipt_sha256": file_hash(backup / "backup-receipt.json"),
                   "isolated_destination": str(catalog), "postgres_directory": str(pg_directory),
                   "dataset_id": db.dataset_id, "target_version": 10,
                   "all_current_rows_sequences_and_managed_files_equal": True,
                   "domain_tables": len(saved["tables"]), "managed_entries": len(saved["files"]),
                   "restore_owner_marker": {"path": marker_name, "sha256": file_hash(catalog / marker_name)},
                   "conversion_sha256": reconciled["result"]["conversion_sha256"],
                   "external_stage_retained_separately": str(stage), "current_root_bindings": prepared["roots"],
                   "backup_seconds": backed_up_at - started, "restore_seconds": restored_at - backed_up_at,
                   "total_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "application_started": False, "production_catalog_accessed": False}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
