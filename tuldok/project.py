"""Inspect/assemble a pressure project; build and execute only explicit commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
NATIVE = "lab/target/release/rheon-trajectory-parallel"
RHEON_REVISION = "2209012bc5f902693d0f695ec1c2d854126af317"
SCRIPT_NAMES = ("trajectory_cpu.py", "trajectory_experiment.py", "trajectory_model.py",
                "verify_parallel_results.py", "trajectory_cpu_viewer.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(*arguments, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def cargo_source_clean(root):
    status = git("status", "--porcelain", root=root)
    if not status:
        return True
    # Cargo creates this empty cache-completion marker after fetching a Git
    # checkout. It is not source; every other untracked/modified path is refused.
    marker = Path(root) / ".cargo-ok"
    return (status == "?? .cargo-ok" and not marker.is_symlink()
            and marker.is_file() and marker.read_bytes() == b"")


def build(cargo, receipt):
    destination = Path(receipt)
    if os.path.lexists(destination):
        raise ValueError("Build receipt destination already exists")
    if git("status", "--porcelain"):
        raise ValueError("Commit source work before recording a reproducible build")
    manifest = "probes/tuldok-pressure/Cargo.toml"
    lock = ROOT / "probes/tuldok-pressure/Cargo.lock"
    before = sha(lock)
    environment = {**os.environ, "CARGO_TARGET_DIR": str(ROOT / "target")}
    command = [str(cargo), "build", "--release", "--locked", "--manifest-path", manifest]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    metadata = json.loads(subprocess.check_output(
        [str(cargo), "metadata", "--locked", "--format-version", "1", "--manifest-path", manifest],
        cwd=ROOT, env=environment, text=True))
    rheon = next(package for package in metadata["packages"] if package["name"] == "rheon")
    rheon_root = Path(rheon["manifest_path"]).parent
    rheon_commit = git("rev-parse", "HEAD", root=rheon_root)
    if (rheon_commit != RHEON_REVISION or not cargo_source_clean(rheon_root)
            or rheon["source"] != "git+https://github.com/MrScripty/Rheon.git?rev=" + RHEON_REVISION + "#" + RHEON_REVISION
            or before != sha(lock) or git("status", "--porcelain")):
        raise ValueError("Build source/lock identity changed or was not pinned and clean")
    rustc = str(Path(cargo).with_name("rustc"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination, {"schema": "rheon-tuldok-build-v1", "status": "built",
          "lab_revision": git("rev-parse", "HEAD"), "lab_status": "clean",
          "rheon_revision": rheon_commit, "rheon_status": "clean", "rheon_source": rheon["source"],
          "cargo_cache_marker": "empty .cargo-ok allowed; all other untracked or modified source refused",
          "rustc": subprocess.check_output([rustc, "--version"], text=True).strip(),
          "cargo": subprocess.check_output([str(cargo), "--version"], text=True).strip(),
          "command": command[1:], "lock_sha256": before,
          "manifest_sha256": sha(ROOT / manifest),
          "native_source_sha256": sha(ROOT / "probes/trajectory/src/main.rs"),
          "binary_sha256": sha(ROOT / "target/release/rheon-trajectory-parallel")})


def outputs(command):
    entries = [("model.pt", "checkpoint", 1024**2),
               ("recipe-receipt.json", "evaluation_receipt", 2*1024**2),
               ("result-summary.json", "result_summary", 65536)]
    if command == "train":
        entries += [("training.npz", "training_arrays", 52*1024**2),
                    ("freeze.json", "training_freeze", 2*1024**2),
                    ("training-result.json", "training_result", 2*1024**2)]
    else:
        entries += [("comparison-freeze.json", "comparison_configuration", 1024**2),
                    ("input-freeze.json", "source_freeze", 2*1024**2),
                    ("timing.json", "comparison_timing", 8*1024**2),
                    ("result.json", "comparison_result", 8*1024**2),
                    ("viewport-display.json", "viewer", 32*1024**2)]
        entries += [(f"{method}-{seed}.npz", "full_fields", 128*1024**2)
                    for method in ("jacobi", "sgs", "learned") for seed in range(900, 904)]
    return [{"path": name, "role": role, "max_bytes": maximum, "required": True}
            for name, role, maximum in entries]


def definition(command, sources, runtime_snapshot):
    inputs = {"config": {"max_bytes": 65536}, "build_receipt": {"max_bytes": 65536}}
    arguments = [{"literal": command}, {"literal": "--config"}, {"input": "config"},
                 {"literal": "--build-receipt"}, {"input": "build_receipt"},
                 {"literal": "--native-policy"}, {"native_policy": NATIVE},
                 {"literal": "--output"}, {"output_dir": True}]
    if command == "compare":
        inputs.update(checkpoint={"max_bytes": 1024**2}, freeze={"max_bytes": 2*1024**2})
        arguments += [{"literal": "--checkpoint"}, {"input": "checkpoint"},
                      {"literal": "--freeze"}, {"input": "freeze"}]
    resources = {"wall_seconds": 600 if command == "train" else 2400,
                 "max_output_bytes": 64*1024**2 if command == "train" else 2*1024**3}
    if command == "compare":
        resources["profile"] = "captured_cpu_large_outputs_v1"
    return {"schema_version": 1, "format": "external_recipe_v1",
            "id": "lab-pressure-" + command, "name": "Lab CPU pressure " + command,
            "version": 1, "runtime": "captured_cpu_python_v1",
            "runtime_snapshot_sha256": runtime_snapshot,
            "sources": [{"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
                        for path, raw in sorted(sources.items())],
            "native_sources": [NATIVE], "entrypoint": "worker.py", "arguments": arguments,
            "parameters": {}, "inputs": inputs, "outputs": outputs(command),
            "environment": {}, "resources": resources}


def assemble(destination, build_receipt, runtime_snapshot):
    if (len(runtime_snapshot) != 64 or any(c not in "0123456789abcdef" for c in runtime_snapshot)):
        raise ValueError("Select a measured captured CPU runtime SHA256")
    receipt = json.loads(Path(build_receipt).read_text())
    binary = ROOT / "target/release/rheon-trajectory-parallel"
    if (receipt.get("schema") != "rheon-tuldok-build-v1" or receipt.get("status") != "built"
            or receipt.get("lab_status") != "clean" or receipt.get("rheon_status") != "clean"
            or receipt.get("rheon_revision") != RHEON_REVISION or receipt.get("binary_sha256") != sha(binary)
            or receipt.get("native_source_sha256") != sha(ROOT / "probes/trajectory/src/main.rs")
            or receipt.get("manifest_sha256") != sha(ROOT / "probes/tuldok-pressure/Cargo.toml")
            or receipt.get("lock_sha256") != sha(ROOT / "probes/tuldok-pressure/Cargo.lock")):
        raise ValueError("Build does not match the pinned project sources")
    sources = {"worker.py": (ROOT / "tuldok/worker.py").read_bytes(),
               "native.py": (ROOT / "tuldok/native.py").read_bytes(), NATIVE: binary.read_bytes()}
    for name in SCRIPT_NAMES:
        sources["lab/scripts/" + name] = (ROOT / "scripts" / name).read_bytes()
    files = dict(sources)
    for name in ("LICENSE", "THIRD_PARTY.md"):
        files[name] = (ROOT / name).read_bytes()
    inputs = {"config": ROOT / "tuldok/configs/pressure-v1.json", "build_receipt": Path(build_receipt),
              "checkpoint": ROOT / "experiments/trajectory-v2/model.pt",
              "freeze": ROOT / "experiments/trajectory-v2/freeze.json"}
    references = []
    for name, path in inputs.items():
        target = "inputs/" + path.name
        files[target] = path.read_bytes()
        references.append({"id": name, "label": name.replace("_", " "), "path": target, "sha256": sha(path)})
    recipes = []
    for command in ("train", "compare"):
        name = "recipes/pressure-" + command + "-v1.json"
        raw = (json.dumps(definition(command, sources, runtime_snapshot), indent=2) + "\n").encode()
        files[name] = raw
        recipes.append({"id": command, "label": "CPU pressure " + command, "path": name,
                        "sha256": hashlib.sha256(raw).hexdigest()})
    evidence = []
    for name in ("freeze.json", "result.json", "verification.json"):
        path = ROOT / "experiments/trajectory-cpu-v1" / name
        target = "evidence/historical-cpu-" + name
        files[target] = path.read_bytes()
        evidence.append({"id": "historical-" + name.split(".")[0], "label": "Historical CPU " + name,
                         "path": target, "sha256": sha(path)})
    manifest = {"schema_version": 1, "id": uuid.uuid4().hex, "name": "Rheon lab pressure reproduction",
                "data": references, "recipes": recipes, "evidence": evidence,
                "requirements": [{"id": "runtime", "label": "Captured CPython3.12, Torch2.8.0+cpu, NumPy2.3.5"},
                                 {"id": "execution", "label": "Explicit generic Run material review and execution approval"}]}
    files["project.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o400)
    return manifest


def native_policy(tuldok_root, destination):
    """Use the generic executor's actual ELF/loader reader, without execution."""
    path = Path(destination)
    if os.path.lexists(path):
        raise ValueError("Native policy destination exists")
    sys.path.insert(0, str(Path(tuldok_root).resolve()))
    import recipe_contract
    runtime = {"trusted_install_owner_uid": os.stat("/usr").st_uid, "files": {}}
    policies = recipe_contract.native_policies(
        {"native_sources": [NATIVE]}, {NATIVE: (ROOT / "target/release/rheon-trajectory-parallel").read_bytes()}, runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, policies[NATIVE])


def local_run(project, command, output, python, policy):
    """Direct recipe observation; explicitly does not claim managed Run custody."""
    project, output = Path(project).resolve(), Path(output).resolve()
    process_receipt = output.parent / (output.name + "-process.json")
    if os.path.lexists(process_receipt):
        raise ValueError("Process receipt destination already exists")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((project / "project.json").read_text())
    recipe_ref = next(item for item in manifest["recipes"] if item["id"] == command)
    if sha(project / recipe_ref["path"]) != recipe_ref["sha256"]:
        raise ValueError("Recipe definition changed")
    recipe = json.loads((project / recipe_ref["path"]).read_text())
    for source in recipe["sources"]:
        if sha(project / source["path"]) != source["sha256"]:
            raise ValueError("Captured source changed")
    references = {item["id"]: item for item in manifest["data"]}
    for slot in recipe["inputs"]:
        ref = references[slot]
        if sha(project / ref["path"]) != ref["sha256"]:
            raise ValueError("Input reference changed")
    arguments = []
    for token in recipe["arguments"]:
        kind, value = next(iter(token.items()))
        arguments.append(value if kind == "literal" else
                         str(project / references[value]["path"]) if kind == "input" else
                         str(Path(policy).resolve()) if kind == "native_policy" else str(output))
    started = time.perf_counter()
    execute_local([str(python), str(project / recipe["entrypoint"]), *arguments],
                  recipe["resources"]["wall_seconds"])
    elapsed = time.perf_counter() - started
    validate_outputs(output, recipe)
    write(process_receipt, {
          "schema": "rheon-local-process-v1", "status": "completed", "wall_s": elapsed,
          "scope": "fresh Python launch through worker teardown, including imports, setup, warmup, accuracy capture/compression; excludes assembly/native build",
          "execution": "direct local process, not a managed Tuldok Run",
          "recipe_sha256": sha(project / recipe_ref["path"]),
          "artifacts": {p.name: sha(p) for p in sorted(output.iterdir())}})


def execute_local(command, wall_seconds):
    """Bound a trusted worker and its same-group native children.

    This is direct-local cleanup, not the managed Runs subreaper or a guarantee
    for arbitrary descendants that deliberately leave the process group.
    """
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=wall_seconds)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as cleanup_error:
            error.add_note("Local process-group termination failed: " + type(cleanup_error).__name__)
        try:
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            error.add_note("Local worker reap failed: " + type(cleanup_error).__name__)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def validate_outputs(output, recipe):
    output = Path(output)
    expected = {item["path"] for item in recipe["outputs"]}
    if {p.name for p in output.iterdir()} != expected:
        raise ValueError("Worker outputs differ from the declared recipe")
    for item in recipe["outputs"]:
        if (output / item["path"]).stat().st_size > item["max_bytes"]:
            raise ValueError("Worker output exceeds its declared cap")
    if sum(path.stat().st_size for path in output.iterdir()) > recipe["resources"]["max_output_bytes"]:
        raise ValueError("Aggregate outputs exceed recipe bound")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compiler = sub.add_parser("build")
    compiler.add_argument("--cargo", default="cargo")
    compiler.add_argument("--receipt", required=True)
    assembly = sub.add_parser("assemble")
    assembly.add_argument("--destination", required=True)
    assembly.add_argument("--build-receipt", required=True)
    assembly.add_argument("--runtime-snapshot-sha256", required=True)
    policy = sub.add_parser("native-policy")
    policy.add_argument("--tuldok-root", required=True)
    policy.add_argument("--destination", required=True)
    runner = sub.add_parser("run-local")
    runner.add_argument("--project", required=True)
    runner.add_argument("--recipe", choices=("train", "compare"), required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--python", default=sys.executable)
    runner.add_argument("--native-policy", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.cargo, args.receipt)
    elif args.command == "assemble":
        print(json.dumps(assemble(args.destination, args.build_receipt, args.runtime_snapshot_sha256)))
    elif args.command == "native-policy":
        native_policy(args.tuldok_root, args.destination)
    else:
        local_run(args.project, args.recipe, args.output, args.python, args.native_policy)


if __name__ == "__main__":
    main()
