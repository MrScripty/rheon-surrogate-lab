"""Explicit isolated domain preparation/publication; never starts the application.

The initial selection comes from source_inventory.py. Publication uses only the
original owner's existing session captured in the selected stopped backup.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/dependency-migration-v1/Tuldok"
sys.path.insert(0, str(SOURCE))
from metadata import Metadata, definition_hash
from psycopg.conninfo import conninfo_to_dict
from source_inventory import converter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "publish", "reconcile"))
    parser.add_argument("--schema-receipt", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--owner-session-file", type=Path, default=OPS / "migration-owner-session.private.txt")
    args = parser.parse_args()
    assert not args.receipt.exists()
    schema = json.loads(args.schema_receipt.read_bytes())
    assert schema["status"] == "passed" and schema["schema_only"] and schema["target_version"] == 10
    catalog = Path(schema["isolated_destination"]).resolve(strict=True)
    assert catalog.is_relative_to(ROOT / "runs/dependency-migration-v1")
    archive = Path(schema["source_backup"]).resolve(strict=True)
    stage = args.stage.resolve()
    assert stage.is_relative_to(ROOT / "runs/dependency-migration-v1") and stage != catalog
    pg_path = SOURCE / "scripts/local_postgres.py"
    pg_spec = importlib.util.spec_from_file_location("domain_rehearsal_postgres", pg_path)
    pg = importlib.util.module_from_spec(pg_spec)
    pg_spec.loader.exec_module(pg)
    dsn = pg.configuration(Path(schema["postgres_directory"]))
    original = json.loads((OPS / "live-selection.private.json").read_bytes())
    selected_info, original_info = conninfo_to_dict(dsn), conninfo_to_dict(original["dsn"])
    assert any(selected_info.get(key) != original_info.get(key) for key in ("host", "port", "dbname", "user"))
    owner = converter()
    selection = json.loads(args.selection.read_bytes())
    assert selection["archive_receipt_sha256"] == schema["source_backup_receipt_sha256"]
    assert selection["archive_receipt_sha256"] == owner.file_hash(archive / "backup-receipt.json")
    assert selection["dataset_id"] == schema["dataset_id"]
    started = time.monotonic()
    db = Metadata(catalog, dsn=dsn, create=False)
    try:
        if args.mode == "prepare":
            assert not stage.exists() and not args.spec.exists()
            spec = {**selection, "target_definition_sha256": definition_hash(),
                    "target_connection_sha256": owner.target_identity(db),
                    "target_application_sha256": owner.application_identity()}
            args.spec.write_text(json.dumps(spec, indent=2) + "\n")
            prepared_sha256 = owner.prepare(db, archive, stage, spec, quiesced=True)
            prepared = json.loads((stage / "prepared.json").read_bytes())
            result = {"status": "prepared_read_only", "prepared_sha256": prepared_sha256,
                      "source_key_count": len(prepared["source_keys"]),
                      "accounting_count": len(prepared["accounting"]),
                      "dispositions": dict(Counter(item["disposition"] for item in prepared["accounting"])),
                      "planned_current_project_count": len(prepared["projects"]),
                      "removed_active_rows": dict(Counter(item["table"] for item in prepared["deletes"])),
                      "roots": prepared["roots"], "converter_sha256": prepared["converter_sha256"],
                      "target_application_sha256": spec["target_application_sha256"],
                      "authoritative_db_writes": 0}
        else:
            assert args.prepared_sha256 and args.spec.exists()
            supplied_spec = json.loads(args.spec.read_bytes())
            prepared = json.loads((stage / "prepared.json").read_bytes())
            assert supplied_spec == prepared["spec"]
            assert set(supplied_spec) == set(selection) | {
                "target_definition_sha256", "target_connection_sha256", "target_application_sha256"}
            assert {key: supplied_spec[key] for key in selection} == selection
            assert supplied_spec["target_connection_sha256"] == owner.target_identity(db)
            assert supplied_spec["dataset_id"] == db.dataset_id == schema["dataset_id"]
            private = args.owner_session_file.stat()
            assert private.st_mode & 0o077 == 0 and private.st_uid == __import__("os").getuid()
            token = args.owner_session_file.read_text().strip()
            assert 32 <= len(token) <= 256
            result = owner.publish(db, stage, args.prepared_sha256, quiesced=True,
                                   token=token, verify_only=args.mode == "reconcile")
        receipt = {"schema": "rheon-domain-rehearsal-v1", "mode": args.mode, "result": result,
                   "catalog": str(catalog), "archive": str(archive), "stage": str(stage),
                   "spec": str(args.spec.resolve()), "source_selection_sha256": owner.file_hash(args.selection),
                   "dataset_id": db.dataset_id, "target_connection_sha256": owner.target_identity(db),
                   "production_catalog_accessed": False, "application_started": False,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    except Exception as error:
        failure = {"schema": "rheon-domain-rehearsal-failed-attempt-v1", "mode": args.mode,
                   "status": "failed_before_publication" if args.mode == "prepare" else "requires_exact_reconciliation",
                   "error_class": type(error).__name__, "catalog": str(catalog), "archive": str(archive),
                   "stage": str(stage), "spec": str(args.spec.resolve()),
                   "source_selection_sha256": owner.file_hash(args.selection),
                   "production_catalog_accessed": False, "application_started": False,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "automatic_retry_performed": False}
        if args.mode == "prepare":
            failure["authoritative_db_writes"] = 0
        if not args.receipt.exists():
            args.receipt.write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
