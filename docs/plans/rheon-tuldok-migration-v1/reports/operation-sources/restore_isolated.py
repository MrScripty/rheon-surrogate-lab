"""Restore one selected stopped backup into a fresh isolated PostgreSQL service."""
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
from backup_metadata import canonical, digest, file_hash, read_catalog, read_sequences, restore_catalog, summaries, tree_manifest
from metadata import Metadata
from upgrade_metadata import upgrade_catalog
from psycopg.conninfo import conninfo_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--postgres-directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    backup = args.backup.resolve(strict=True)
    catalog = args.catalog.resolve()
    pg_directory = args.postgres_directory.resolve()
    assert not catalog.exists() and not args.receipt.exists()
    assert catalog.is_relative_to(ROOT / "runs/dependency-migration-v1")
    assert pg_directory.is_relative_to(Path("/tmp")) and not pg_directory.exists()
    source = json.loads((backup / "backup-receipt.json").read_bytes())
    assert source["schema_version"] == 8
    started = time.monotonic()
    helper = SOURCE / "scripts/local_postgres.py"
    log_path = OPS / (pg_directory.name + "-init.private.log")
    with os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as log:
        subprocess.run([sys.executable, "-B", str(helper), "init", "--directory", str(pg_directory)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    spec = importlib.util.spec_from_file_location("isolated_restore_postgres", helper)
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    dsn = pg.configuration(pg_directory)
    original_selection = json.loads((OPS / "live-selection.private.json").read_bytes())
    destination_info = conninfo_to_dict(dsn)
    original_info = conninfo_to_dict(original_selection["dsn"])
    assert any(destination_info.get(key) != original_info.get(key) for key in ("host", "port", "dbname", "user"))
    restore_catalog(backup, catalog, dsn=dsn)
    restored_at = time.monotonic()
    first = upgrade_catalog(catalog, dsn=dsn, quiesced=True)
    second = upgrade_catalog(catalog, dsn=dsn, quiesced=True)
    assert first["status"] == "upgraded" and second["status"] == "already_current"
    db = Metadata(catalog, dsn=dsn, create=False)
    try:
        tables = summaries(read_catalog(db))
        for name, expected in source["tables"].items():
            if name != "catalog_version":
                assert tables[name] == expected, name
        assert all(row["count"] == 0 for name, row in tables.items() if name not in source["tables"])
        assert read_sequences(db) == source["sequences"]
        marker_name = ".restore-receipt-" + digest(source) + ".json"
        marker = {"kind": "postgresql-restore", "backup_sha256": digest(source), "dataset_id": source["dataset_id"]}
        assert (catalog / marker_name).read_bytes() == canonical(marker) + b"\n"
        assert tree_manifest(catalog, (marker_name,)) == source["files"]
        receipt = {"schema": "rheon-schema-rehearsal-v1", "status": "passed", "schema_only": True,
                   "source_backup": str(backup), "source_backup_receipt_sha256": file_hash(backup / "backup-receipt.json"),
                   "isolated_destination": str(catalog), "postgres_directory": str(pg_directory),
                   "dataset_id": db.dataset_id, "source_version": 8, "target_version": db.schema_version,
                   "source_domain_tables": len(source["tables"]), "managed_entries": len(source["files"]),
                   "all_original_domain_rows_and_files_equal": True, "sequences_equal": True, "new_tables_empty": True,
                   "upgrade_reentry_verified": True,
                   "restore_owner_marker": {"path": marker_name, "sha256": file_hash(catalog / marker_name), "canonical_bytes_verified": True},
                   "credential_count": tables["access_credentials"]["count"],
                   "restore_wall_seconds": restored_at - started, "total_wall_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "domain_conversion": "pending; no Dataset/service opened before conversion"}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
