"""Bounded source-Project selection from an exact stopped backup's SQL rows."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "runs/dependency-migration-v1/Tuldok"
sys.path.insert(0, str(SOURCE))


def converter():
    path = SOURCE / "tools/migrations/rheon-lab-once/convert.py"
    spec = importlib.util.spec_from_file_location("offline_rheon_lab_conversion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(backup):
    started = time.monotonic()
    owner = converter()
    backup = Path(backup).resolve(strict=True)
    receipt_path = backup / "backup-receipt.json"
    receipt = owner.decode(receipt_path.read_bytes())
    assert receipt["schema_version"] == 8 and receipt["kind"] == "postgresql"
    assert owner.file_hash(backup / "metadata.json") == receipt["metadata_sha256"]
    data = owner.decode((backup / "metadata.json").read_bytes())
    assert (data["dataset_id"], data["schema_version"], data["format_version"]) == (receipt["dataset_id"], 8, 1)
    assert set(data["tables"]) == set(owner.TABLES_V8)
    for state in data["tables"].values():
        assert owner.table_state(state["columns"], [[owner.unpack(value) for value in row] for row in state["rows"]]) == state
    assert owner.summaries(data["tables"]) == receipt["tables"] and data["sequences"] == receipt["sequences"]
    roots = {"lab": {"path": str(ROOT)}}
    evidence_path = ROOT / "runs/dependency-migration-v1/ops/source-reference-inventory.json"
    assert owner.file_hash(evidence_path) == "41c73b31e838a9ec085cb7d2407b1234261cc818149609dc594dbb63139c593b"
    evidence = json.loads(evidence_path.read_bytes())["bounded_recovery"]
    recoveries = []
    unique = set()
    for row in evidence["git_historical_exact_matches"]:
        key = (row["path"], row["sha256"])
        if key not in unique:
            recoveries.append({"original_path": row["path"], "sha256": row["sha256"],
                "source": {"kind": "git", "repository": str(ROOT), "revision": row["git_revision"], "path": row["path"]}})
            unique.add(key)
    lineage_root = Path(evidence["root"]).resolve()
    for row in evidence["exact_matches"]:
        key = (row["path"], row["sha256"])
        path = Path(row["recovery_path"]).resolve(strict=True)
        assert path.is_relative_to(lineage_root)
        if key not in unique:
            recoveries.append({"original_path": row["path"], "sha256": row["sha256"],
                "source": {"kind": "file", "root": str(lineage_root), "path": path.relative_to(lineage_root).as_posix()}})
            unique.add(key)
    roots["lab"]["recoveries"] = recoveries
    observed = owner.selected_inventory(ROOT, owner.reference_paths(data, "lab", str(ROOT)))
    roots["lab"].update(observed, selection_sha256=owner.digest(observed))
    selection = {"dataset_id": data["dataset_id"], "archive_receipt_sha256": owner.file_hash(receipt_path),
                 "target_schema_version": 10, "roots": roots}
    report = {"schema": "rheon-bounded-source-selection-v1", "status": "selected_from_exact_stopped_sql_archive",
              "source_backup": str(backup), "source_metadata_sha256": receipt["metadata_sha256"],
              "source_archive_receipt_sha256": selection["archive_receipt_sha256"],
              "source_project_count": data["tables"]["workspace_projects"]["count"],
              "source_run_count": data["tables"]["workspace_runs"]["count"],
              "source_credential_count": data["tables"]["access_credentials"]["count"],
              "selected_file_count": sum(row["kind"] == "file" for row in observed["files"].values()),
              "selected_directory_count": sum(row["kind"] == "directory" for row in observed["files"].values()),
              "selected_bytes": sum(row.get("bytes", 0) for row in observed["files"].values()),
              "preexisting_unavailable_count": len(observed["unavailable"]), "exact_pin_recovery_count": len(recoveries),
              "whole_root_walk_or_copy": False, "authoritative_db_writes": 0,
              "archive_asset_verification": "owned by stopped-backup verification, restore, and mandatory converter prepare; not repeated by bounded source selection",
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return selection, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    selection, report = inventory(args.backup)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    report["selection_file"] = str(args.output.resolve())
    report["selection_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
