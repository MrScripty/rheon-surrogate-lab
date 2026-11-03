"""Launch a disposable current server only after its exact cold roundtrip passed."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/dependency-migration-v1/Tuldok"


def catalog_counts(catalog, dsn, project_id):
    sys.path.insert(0, str(SOURCE))
    from metadata import Metadata
    db = Metadata(catalog, dsn=dsn, create=False)
    try:
        db.connection.execute("SET default_transaction_read_only = on")
        counts = {name: db.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                  for name in ("workspace_projects", "workspace_runs", "access_actors", "access_credentials", "access_audit")}
        row = db.execute("SELECT document FROM workspace_projects WHERE id=%s", (project_id,)).fetchone()
        return counts, None if row is None else hashlib.sha256(row[0].encode()).hexdigest()
    finally:
        db.close()


def verify_deployment(application):
    manifest = application / "deployment-manifest.json"
    raw = manifest.read_bytes()
    value = json.loads(raw)
    declared = set()
    for row in value["files"]:
        path = application / row["path"]
        assert not path.is_symlink() and path.resolve().is_relative_to(application)
        assert path.stat().st_size == row["bytes"] and path.stat().st_mode & 0o777 == row["mode"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        declared.add(row["path"])
    actual = {path.relative_to(application).as_posix() for path in application.rglob("*") if path.is_file()}
    assert actual == declared | {"deployment-manifest.json"}
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roundtrip-receipt", type=Path, required=True)
    parser.add_argument("--cold-receipt", type=Path, required=True)
    parser.add_argument("--current-project-provenance", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8993)
    args = parser.parse_args()
    assert 1024 <= args.port <= 65535 and args.port != 8091
    current = json.loads(args.roundtrip_receipt.read_bytes())
    cold = json.loads(args.cold_receipt.read_bytes())
    assert current["status"] == cold["status"] == "passed" and not current["application_started"]
    catalog = Path(current["isolated_destination"]).resolve(strict=True)
    assert cold["catalog"] == str(catalog) and catalog.is_relative_to(ROOT / "runs/dependency-migration-v1")
    assert cold["old_catalog_backup_and_converter_unavailable"] and cold["read_only_database_counts_unchanged"]
    stage = Path(current["external_stage_retained_separately"]).resolve(strict=True)
    prepared = json.loads((stage / "prepared.json").read_bytes())
    application = args.application.resolve(strict=True)
    assert application.is_relative_to(ROOT / "runs/dependency-migration-v1/Tuldok-deployed")
    deployment_sha256 = verify_deployment(application)
    application_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(application.glob("*.py"))}
    application_sha256 = hashlib.sha256(json.dumps(application_files, ensure_ascii=True, sort_keys=True,
                                                  separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
    assert application_sha256 == prepared["spec"]["target_application_sha256"]
    assert not (application / "tools/migrations/rheon-lab-once").exists()
    provenance = json.loads(args.current_project_provenance.read_bytes())
    assert provenance["identity_relation"] == "new_current_project"
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "runs/dependency-migration-v1") and not output.exists()
    output.mkdir(mode=0o700)
    helper = SOURCE / "scripts/local_postgres.py"
    spec = importlib.util.spec_from_file_location("browser_rehearsal_postgres", helper)
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    dsn = pg.configuration(Path(current["postgres_directory"]))
    counts_before, prior_project = catalog_counts(catalog, dsn, provenance["project_id"])
    assert prior_project is None
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    environment.update(TULDOK_DATABASE_URL=dsn, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    executable = ROOT / "runs/dependency-migration-v1/app-env/bin/python"
    endpoint = "http://127.0.0.1:" + str(args.port)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    roots = {**prepared["roots"], "current-lab": str(ROOT)}
    command = [str(executable), "-B", str(application / "app.py"), "--data", str(catalog),
               "--host", "127.0.0.1", "--port", str(args.port)]
    for alias, path in sorted(roots.items()):
        command.extend(("--project-root", alias + "=" + path))
    started = time.monotonic()
    private_log = output / "server.private.log"
    with os.fdopen(os.open(private_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as log:
        process = subprocess.Popen(command, cwd=application, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            for attempt in range(150):
                assert process.poll() is None, "Isolated server exited; inspect its private log"
                if ("Tuldok: " + endpoint + "\n").encode() not in private_log.read_bytes():
                    time.sleep(0.2)
                    continue
                try:
                    with urllib.request.urlopen(endpoint + "/workspaces", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise RuntimeError("Isolated server did not become ready")
            config = {"endpoint": endpoint, "stage": str(stage),
                      "application": str(application), "deployment_manifest_sha256": deployment_sha256,
                      "application_sha256": application_sha256,
                      "current_project_provenance": str(args.current_project_provenance.resolve()),
                      "owner_session_file": str(OPS / "migration-owner-session.private.txt"),
                      "chromium": "/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell",
                      "receipt": str(output / "browser-receipt.json"),
                      "current_screenshot": str(output / "current-project.png"),
                      "activity_screenshot": str(output / "current-activity.png"),
                      "historical_screenshot": str(output / "historical-evidence.png")}
            config_path = output / "browser-config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            subprocess.run(["node", str(OPS / "rehearsal_browser.cjs"), str(config_path)],
                           check=True, timeout=180,
                           env={key: value for key, value in environment.items() if key != "TULDOK_DATABASE_URL"})
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=60)
        assert process.returncode == 0
    counts_after, registered_project = catalog_counts(catalog, dsn, provenance["project_id"])
    assert registered_project is not None
    expected_counts = dict(counts_before)
    expected_counts["workspace_projects"] += 1
    expected_counts["access_audit"] += 1
    assert counts_after == expected_counts
    assert application_files == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(application.glob("*.py"))}
    assert verify_deployment(application) == deployment_sha256
    receipt = {"schema": "rheon-isolated-browser-server-v1", "status": "passed",
               "source": str(application), "current_application_sha256": application_sha256,
               "application_bytes_pre_post_identical": True, "converter_present": False,
               "deployment_manifest": {"path": str(application / "deployment-manifest.json"), "sha256": deployment_sha256},
               "executable": str(executable), "pid": process.pid,
               "endpoint": endpoint, "catalog": str(catalog), "roots": roots,
               "production_service_changed": False, "server_exited": True,
               "original_owner_authority_reused": True, "fresh_login_or_credential_minted": False,
               "browser_receipt": str(output / "browser-receipt.json"),
               "intentional_clone_write": {"operation": "projects.open", "project_id": provenance["project_id"],
                                           "registered_document_sha256": registered_project,
                                           "counts_before": counts_before, "counts_after": counts_after,
                                           "exactly_one_project_and_registration_audit_added": True,
                                           "run_actor_and_credential_counts_unchanged": True},
               "orchestration_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "largest_reaped_child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               "elapsed_seconds": time.monotonic() - started}
    (output / "server-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
