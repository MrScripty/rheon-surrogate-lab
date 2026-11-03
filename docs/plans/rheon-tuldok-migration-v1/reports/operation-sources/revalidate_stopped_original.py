"""Compare a fresh stopped original bookend to the retained complete backup.

This helper never stops/resumes services or rewrites source records. Any difference
requires a new upstream complete backup; no partial refresh is supported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runs/dependency-migration-v1/Tuldok"))
from backup_metadata import digest, file_hash, read_catalog, read_sequences, summaries, tree_manifest, tree_modes
from metadata import Metadata, definition_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--schema-receipt", type=Path, required=True)
    parser.add_argument("--schema-receipt-sha256", required=True)
    parser.add_argument("--stopped-pid", type=int, required=True)
    parser.add_argument("--quiesced", action="store_true", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists() and args.stopped_pid > 1
    assert not Path("/proc", str(args.stopped_pid)).exists(), "Root must stop the selected original first"
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8091)) != 0, "Original endpoint is still listening"
    selected = json.loads((OPS / "live-selection.private.json").read_bytes())
    expected_command = b"\0".join(item.encode() for item in selected["args"]) + b"\0"
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                assert (entry / "cmdline").read_bytes() != expected_command, "Original source command remains active"
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    catalog = Path(json.loads((OPS / "preflight.json").read_bytes())["catalog"]).resolve(strict=True)
    backup = args.backup.resolve(strict=True)
    assert backup.is_relative_to(ROOT / "runs/dependency-migration-v1")
    verified_raw = args.schema_receipt.read_bytes()
    verified_sha256 = hashlib.sha256(verified_raw).hexdigest()
    assert verified_sha256 == args.schema_receipt_sha256
    verified = json.loads(verified_raw)
    assert verified["status"] == "passed" and verified["schema_only"]
    assert verified["all_original_domain_rows_and_files_equal"] and verified["sequences_equal"]
    assert Path(verified["source_backup"]).resolve(strict=True) == backup
    assert file_hash(backup / "backup-receipt.json") == verified["source_backup_receipt_sha256"]
    saved = json.loads((backup / "backup-receipt.json").read_bytes())
    assert saved["schema_version"] == 8 and len(saved["tables"]) == 31
    assert file_hash(backup / "metadata.json") == saved["metadata_sha256"]
    started = time.monotonic()
    db = Metadata(catalog, dsn=selected["dsn"], create=False, schema_version=None)
    try:
        assert db.dataset_id == saved["dataset_id"] and db.schema_version == 8
        db.connection.execute("SET default_transaction_read_only = on")
        ledger = [list(row) for row in db.execute(
            "SELECT version,dataset_id,definition_sha256,catalog_sha256 FROM catalog_version")]
        assert len(ledger) == 1 and ledger[0][:3] == [8, db.dataset_id, definition_hash(8)]
        tables_before, sequences_before = summaries(read_catalog(db)), read_sequences(db)
        observed = tree_manifest(catalog)
        modes_equal = tree_modes(catalog, observed) == tree_modes(backup / "catalog", saved["files"])
        tables_after, sequences_after = summaries(read_catalog(db)), read_sequences(db)
        ledger_after = [list(row) for row in db.execute(
            "SELECT version,dataset_id,definition_sha256,catalog_sha256 FROM catalog_version")]
        stable = tables_before == tables_after and sequences_before == sequences_after and ledger == ledger_after
        tables_equal, sequences_equal = tables_before == saved["tables"], sequences_before == saved["sequences"]
        files_equal = observed == saved["files"]
        equal = all((stable, tables_equal, sequences_equal, files_equal, modes_equal))
        receipt = {"schema": "rheon-fresh-stopped-original-bookend-v1",
                   "status": "exact_backup_revalidated" if equal else "new_complete_backup_required",
                   "complete_backup": str(backup), "backup_receipt_sha256": file_hash(backup / "backup-receipt.json"),
                   "schema_rehearsal_receipt": {"path": str(args.schema_receipt.resolve()), "sha256": verified_sha256},
                   "dataset_id": db.dataset_id, "schema_version": db.schema_version, "schema_ledger": ledger,
                   "current_schema_ledger_and_structure_validated_by_owner": True,
                   "catalog_schema_signature_sha256": ledger[0][3],
                   "domain_tables": len(tables_before), "domain_rows": sum(item["count"] for item in tables_before.values()),
                   "managed_entries": len(observed), "all_domain_tables_equal": tables_equal,
                   "all_sequences_equal": sequences_equal, "all_managed_hashes_equal": files_equal,
                   "all_file_and_directory_modes_equal": modes_equal, "metadata_pre_post_stable": stable,
                   "fresh_inventory_sha256": digest(observed), "retained_inventory_sha256": digest(saved["files"]),
                   "stopped_original_pid": args.stopped_pid, "original_port_closed": True,
                   "original_source_command_absent": True, "quiescence_attested_by_caller": args.quiesced,
                   "database_default_transaction_read_only": True, "authoritative_db_writes": 0,
                   "files_rewritten_or_replaced": 0, "services_changed_by_helper": False,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
