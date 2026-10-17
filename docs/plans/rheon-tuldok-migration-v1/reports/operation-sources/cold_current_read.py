"""Fresh current-owner inspection with original inputs and converter inaccessible.

Only this process receives filesystem/spawn guards. Shared repositories and
services are neither renamed nor modified. No Dataset/Run worker is constructed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/dependency-migration-v1/Tuldok"
sys.path.insert(0, str(SOURCE))
from access import Access
from metadata import Metadata
from projects import Projects, TEXT_LIMIT
from runs import Runs
from backup_metadata import digest, file_hash


def current_import_origins(numerical):
    origins = {name: str(Path(module.__file__).resolve()) for name, module in tuple(sys.modules.items())
               if getattr(module, "__file__", None) and not str(module.__file__).startswith("<")}
    forbidden = (SOURCE / "tools/migrations/rheon-lab-once", ROOT / "runs/live-integration/Tuldok",
                 ROOT / ".venv", Path(numerical["python_root"]), Path(numerical["site_packages"]))
    assert all(not Path(path).is_relative_to(prefix) for path in origins.values() for prefix in forbidden)
    owners = {name: {"path": path, "sha256": file_hash(path)} for name, path in origins.items()
              if Path(path).is_relative_to(SOURCE)}
    assert all(name in owners for name in ("access", "metadata", "projects", "runs"))
    return {"loaded_module_count": len(origins), "module_origins_sha256": digest(origins),
            "current_owner_origins": owners, "forbidden_original_or_numerical_imports": 0}


def forbid_original_inputs(prepared, backup, original_catalog, numerical, original_backup):
    # Current catalogs/stage are descendants of the original ROOT/runs. Opening
    # ancestor directory handles is necessary to reach them; deny original file
    # payloads, not shared ancestor traversal.
    denied_files = {str((Path(selected["path"]) / name).absolute())
                    for selected in prepared["spec"]["roots"].values()
                    for name, item in selected["files"].items() if item["kind"] == "file"}
    denied_prefixes = [str(Path(backup).resolve()), str(Path(original_backup).resolve()), str(Path(original_catalog).resolve()),
                       str(SOURCE / "tools/migrations/rheon-lab-once"), str(ROOT / ".git"),
                       str(ROOT / "runs/live-integration/Tuldok"),
                       str(ROOT / ".venv"), numerical["python_root"], numerical["site_packages"]]
    blocked = {"old_input_attempts": 0, "spawn_attempts": 0}
    original_open = os.open
    descriptor_open = False

    def guard(path, directory=None):
        if isinstance(path, int):
            return
        path = os.fsdecode(path)
        if not os.path.isabs(path) and directory is not None:
            path = os.path.join(os.readlink("/proc/self/fd/" + str(directory)), path)
        path = os.path.abspath(path)
        if path in denied_files or any(path == prefix or path.startswith(prefix + os.sep) for prefix in denied_prefixes):
            blocked["old_input_attempts"] += 1
            raise PermissionError("Original migration inputs and converter are unavailable in this cold reader")

    def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal descriptor_open
        guard(path, dir_fd)
        # CPython's nested open audit event omits dir_fd. The wrapper has just
        # checked its actual absolute target; checking that relative name again
        # against cwd would falsely deny a staged path such as "runs".
        descriptor_open = True
        try:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            descriptor_open = False

    def audit(event, arguments):
        if event == "open" and not descriptor_open:
            guard(arguments[0])
        elif event in ("subprocess.Popen", "os.system", "os.fork", "os.posix_spawn", "os.exec"):
            blocked["spawn_attempts"] += 1
            raise PermissionError("Cold current inspection cannot spawn work")

    os.open = guarded_open
    sys.addaudithook(audit)
    # Check both absolute Python reads and the directory-FD path used by Projects.
    probes = [Path(backup) / "metadata.json", Path(original_catalog) / ".tuldok.json",
              Path(original_backup) / "metadata.json",
              SOURCE / "tools/migrations/rheon-lab-once/convert.py", Path(numerical["python"]),
              ROOT / "runs/live-integration/Tuldok/projects.py",
              Path(sorted(denied_files)[0])]
    for probe in probes:
        try:
            probe.read_bytes()
        except PermissionError:
            continue
        raise AssertionError("Old-input guard did not reject its probe")
    root_descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            descriptor = os.open(str(Path(backup) / "metadata.json").lstrip("/"), os.O_RDONLY, dir_fd=root_descriptor)
        except PermissionError:
            pass
        else:
            os.close(descriptor)
            raise AssertionError("Directory-FD old-input guard did not reject its probe")
    finally:
        os.close(root_descriptor)
    blocked["guard_probe_attempts"] = blocked["old_input_attempts"]
    return blocked


def main():
    reader_source_sha256 = file_hash(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-receipt", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--original-backup", type=Path, required=True)
    parser.add_argument("--owner-session-file", type=Path, default=OPS / "migration-owner-session.private.txt")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    schema = json.loads(args.schema_receipt.read_bytes())
    stage = args.stage.resolve(strict=True)
    prepared = json.loads((stage / "prepared.json").read_bytes())
    original_backup = args.original_backup.resolve(strict=True)
    assert file_hash(original_backup / "backup-receipt.json") == prepared["spec"]["archive_receipt_sha256"]
    original_catalog = json.loads((OPS / "preflight.json").read_bytes())["catalog"]
    token = args.owner_session_file.read_text().strip()
    numerical = json.loads((ROOT / "runs/dependency-migration-v1/runtime/current-manifest-v2.json").read_bytes())
    numerical = {name: numerical[name] for name in ("python", "python_root", "site_packages")}
    preguard_imports = current_import_origins(numerical)
    application_files = {path.name: file_hash(path) for path in sorted(SOURCE.glob("*.py"))}
    assert digest(application_files) == prepared["spec"]["target_application_sha256"]
    helper = SOURCE / "scripts/local_postgres.py"
    spec = importlib.util.spec_from_file_location("cold_current_pg", helper)
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    dsn = pg.configuration(Path(schema["postgres_directory"]))
    started = time.monotonic()
    blocked = forbid_original_inputs(prepared, schema["source_backup"], original_catalog, numerical, original_backup)
    db = Metadata(Path(schema["isolated_destination"]), dsn=dsn, create=False)
    # Enforce read-only semantics in the real database as well as in Python.
    db.connection.execute("SET default_transaction_read_only = on")
    access = Access(db, db.path)
    dataset = SimpleNamespace(db=db, lock=db.lock, access=access)
    projects = Projects(dataset, prepared["roots"])
    runs = Runs.__new__(Runs)
    runs.db, runs.lock, runs.access, runs.path = db, db.lock, access, db.path / "workspace-runs"
    try:
        before = {name: db.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                  for name in ("workspace_projects", "workspace_runs", "access_actors", "access_credentials", "access_audit")}
        project_count = evidence_count = data_ref_count = unverified_large_count = 0
        with access.bind(token):
            for saved in prepared["projects"]:
                result = projects.inspect_snapshot(saved["id"])
                assert result["manifest_status"] == "unchanged"
                assert result["manifest_sha256"] == saved["manifest_sha256"]
                for group in ("data", "recipes", "evidence"):
                    for ref in result[group]:
                        assert ref["status"] == "available" and ref.get("integrity") != "mismatch", {
                            "project_id": saved["id"], "group": group,
                            "reference": {key: ref.get(key) for key in ("id", "path", "status", "integrity", "reason", "error")}}
                        if group == "data":
                            data_ref_count += 1
                            if ref.get("integrity") == "unverified":
                                unverified_large_count += 1
                        if group == "evidence" and (saved["root"] == "historical" or ref.get("bytes", 0) <= TEXT_LIMIT):
                            preview = projects.evidence(saved["id"], ref["id"])
                            assert preview["external"] and preview["scientific_verification"] == "not_performed"
                            if ref.get("sha256"):
                                assert preview["sha256"] == ref["sha256"]
                            evidence_count += 1
                project_count += 1
            run_ids = [row[0] for row in db.execute("SELECT id FROM workspace_runs ORDER BY id")]
            artifact_count = artifact_bytes = 0
            for run_id in run_ids:
                result = runs.get(run_id)
                assert result["state"] in ("completed", "failed", "cancelled", "unknown")
                if result["state"] == "completed":
                    for item in result["artifacts"]:
                        raw, media_type = runs.artifact(run_id, item["name"])
                        assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
                        artifact_count += 1
                        artifact_bytes += len(raw)
        after = {name: db.execute("SELECT COUNT(*) FROM " + name).fetchone()[0] for name in before}
        assert before == after
        assert blocked["old_input_attempts"] == blocked["guard_probe_attempts"] and blocked["spawn_attempts"] == 0
        assert application_files == {path.name: file_hash(path) for path in sorted(SOURCE.glob("*.py"))}
        postguard_imports = current_import_origins(numerical)
        assert reader_source_sha256 == file_hash(Path(__file__))
        receipt = {"schema": "rheon-cold-current-inspection-v1", "status": "passed", "dataset_id": db.dataset_id,
                   "catalog": str(db.path), "current_project_count": project_count,
                   "current_evidence_previews_verified": evidence_count, "current_data_references_available": data_ref_count,
                   "bounded_hash_inspection_unverified_refs": unverified_large_count,
                   "retained_current_runs_verified": len(run_ids), "retained_current_artifacts_verified": artifact_count,
                   "retained_current_artifact_bytes_verified": artifact_bytes,
                   "filesystem_guards": blocked, "old_catalog_backup_and_converter_unavailable": True,
                   "installed_numerical_runtime_unavailable": True,
                   "read_only_database_counts_unchanged": True, "counts_before_after": before,
                   "database_default_transaction_read_only": True, "workers_or_application_constructed": False,
                   "prepared_sha256": file_hash(stage / "prepared.json"),
                   "reader_source_sha256": reader_source_sha256, "reader_source_bytes_pre_post_identical": True,
                   "original_backup": str(original_backup),
                   "original_backup_receipt_sha256": prepared["spec"]["archive_receipt_sha256"],
                   "current_application_sha256": digest(application_files), "application_bytes_pre_post_identical": True,
                   "preguard_imports": preguard_imports, "postguard_imports": postguard_imports,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    finally:
        projects.close()
        db.close()


if __name__ == "__main__":
    main()
