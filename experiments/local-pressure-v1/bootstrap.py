#!/usr/bin/env python3
"""Fresh local catalog/runtime/project preflight and explicit managed execution."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from contract import HERE, LAB, MIB, canonical, decode, experiment, fresh, load_module, read, require, sha, write


def host(path):
    value = decode(read(path, 65536))
    require(type(value) is dict and set(value) == {"schema", "python", "cargo", "tuldok_source", "rheon_source", "rheon_revision", "postgres_directory"} and value["schema"] == "rheon-local-pressure-host-v1", "host_schema", "explicit isolated host configuration")
    for key in ("python", "cargo", "tuldok_source", "rheon_source", "postgres_directory"):
        supplied = Path(value[key])
        value[key] = str((path.parent / supplied).absolute() if not supplied.is_absolute() else supplied)
    require(Path(value["postgres_directory"]).absolute() == Path(value["postgres_directory"]).resolve(), "host_path", "canonical PostgreSQL directory")
    return value


def postgres(host_config, action):
    return subprocess.run([host_config["python"], str(Path(host_config["tuldok_source"]) / "scripts/local_postgres.py"), action, "--directory", host_config["postgres_directory"]], check=True)


def database_environment(host_config):
    helper = load_module("local_pressure_postgres_configuration", Path(host_config["tuldok_source"]) / "scripts/local_postgres.py")
    os.environ["TULDOK_DATABASE_URL"] = helper.configuration(Path(host_config["postgres_directory"]))


def scientific_environment():
    # Tuldok's immutable snapshot verifier requires privately owned directories,
    # including intermediate parents created by mkdir(parents=True).
    os.umask(0o077)
    os.environ.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def runtime_configuration(output):
    import sysconfig
    # Distro stdlib may contain links outside /usr (Ubuntu's sitecustomize).
    # Materialize the installed bytes before Tuldok's contained-source inventory;
    # do not relax its symlink rules or copy a historical interpreter environment.
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    root = fresh(output / "python-source")
    (root / "bin").mkdir()
    (root / "lib").mkdir()
    shutil.copyfile(Path(sys.executable).resolve(), root / "bin" / ("python" + version))
    (root / "bin" / ("python" + version)).chmod(0o755)
    # Development .o/.a files are link-time artifacts, not runtime ELF images.
    # Keep sysconfig's Makefile/header data but omit those compiler inputs.
    shutil.copytree(sysconfig.get_path("stdlib"), root / "lib" / ("python" + version), ignore=shutil.ignore_patterns("__pycache__", "site-packages", "dist-packages", "*.o", "*.a"))
    for path in (Path(sys.base_prefix) / "lib").glob("libpython" + version + ".so*"):
        shutil.copyfile(path.resolve(), root / "lib" / path.name)
    # These package roots are the supported scientific dependency closure, not
    # the application's environment. Distribution metadata is measured afresh.
    packages = ["torch", "torchgen", "numpy", "numpy.libs", "filelock", "fsspec", "jinja2", "markupsafe", "mpmath", "networkx", "sympy", "isympy.py", "typing_extensions.py", "setuptools", "_distutils_hack"]
    site = Path(sysconfig.get_path("purelib"))
    for name in ("torch", "numpy", "filelock", "fsspec", "Jinja2", "MarkupSafe", "mpmath", "networkx", "sympy", "typing_extensions", "setuptools"):
        metadata = importlib.metadata.distribution(name)
        entries = [p.parts[0] for p in metadata.files if len(p.parts) == 2 and p.parts[0].endswith(".dist-info") and p.name == "METADATA"]
        require(len(entries) == 1, "runtime_metadata", name)
        packages.append(entries[0])
    excludes = ["torch/bin/" + name for name in ("FileStoreTest", "HashStoreTest", "TCPStoreTest", "protoc", "protoc-3.13.0.0", "test_api", "test_cpp_rpc", "test_dist_autograd", "test_jit", "test_lazy", "test_nativert", "test_tensorexpr", "tutorial_tensorexpr")]
    excludes.append("torch/test")
    optional = ["lib-dynload/" + p.name for p in (root / "lib" / ("python" + version) / "lib-dynload").glob("_tkinter*.so")]
    return {"schema_version": 1, "python_root": str(root), "python_version": version, "site_packages": str(site), "packages": sorted(packages), "exclude_paths": excludes, "stdlib_exclude_paths": optional}


def material(config, host_config, output):
    import capture
    import data
    import dependency
    import native
    prepared = output / "prepared-rheon"
    dependency.prepare(Path(host_config["rheon_source"]), host_config["rheon_revision"], prepared)
    binary = native.build("capture", config, prepared, Path(host_config["cargo"]), output / "capture-build")
    capture.generate(config, binary, output / "fixture", fixture=True)
    fixture = capture.verify(output / "fixture")
    require(fixture["status"] == "verified_fixture" and fixture["frames"] == 1, "fixture_required", "one-frame physics first")
    operator = native.build("evaluation", None, prepared, Path(host_config["cargo"]), output / "native-build")
    oracle, _measurement = native.call(operator, "--oracle")
    write(output / "native-oracle.json", oracle)
    capture.generate(config, binary, output / "capture")
    data.pack(output / "capture", operator, output / "packet")
    return {"fixture": str(output / "fixture"), "capture": str(output / "capture"), "packet": str(output / "packet"), "native_binary": str(operator), "capture_binary": str(binary)}


def prepare(config_path, host_path, output, existing_material=None):
    scientific_environment()
    host_config = host(host_path)
    require(Path(host_config["python"]).resolve() == Path(sys.executable).resolve(), "host_python", "launch with configured isolated interpreter")
    config = experiment(decode(read(config_path, MIB)))
    output = fresh(output)
    write(output / "experiment.json", config)
    write(output / "host.json", host_config)
    import project
    tuldok = Path(host_config["tuldok_source"])
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tuldok, text=True).strip() == project.TULDOK_PIN and not subprocess.check_output(["git", "status", "--porcelain"], cwd=tuldok), "tuldok_source", "clean latest qualified generic owner")
    if existing_material is None:
        selected = material(config, host_config, output)
    else:
        import capture
        import data
        import native
        selected = decode(read(existing_material, MIB))
        require(set(selected) == {"fixture", "capture", "packet", "native_binary", "capture_binary"}, "material_schema", "explicit previously qualified inputs")
        require(capture.verify(Path(selected["fixture"]))["status"] == "verified_fixture" and capture.verify(Path(selected["capture"]))["status"] == "verified_development" and canonical(data.load(Path(selected["packet"])).config) == canonical(config), "material_identity", "qualified fixture/capture/packet/config")
        require(native.admit(Path(selected["native_binary"]))["identity"]["native_receipt"]["base_commit"] == host_config["rheon_revision"] and capture.admit_build(config, Path(selected["capture_binary"]))["identity"]["native_receipt"]["base_commit"] == host_config["rheon_revision"], "material_revision", "latest selected native dependency")
    write(output / "material.json", selected)
    # Numerical source qualification is separate from native capture qualification.
    test_env = {**os.environ, "TULDOK_CUSTODY_SOURCE": str(tuldok / "recipe_process.py")}
    with (output / "numerical-tests.log").open("xb") as log:
        subprocess.run([sys.executable, str(HERE / "qualify.py")], env=test_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    pg = Path(host_config["postgres_directory"])
    if not (pg / "cluster/PG_VERSION").exists():
        postgres(host_config, "init")
    else:
        observed = subprocess.run([host_config["python"], str(tuldok / "scripts/local_postgres.py"), "status", "--directory", str(pg)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if observed.returncode:
            postgres(host_config, "start")
    database_environment(host_config)
    sys.path.insert(0, str(tuldok))
    from metadata import Metadata
    from access import Access
    from runtime_snapshots import RuntimeSnapshots
    catalog = output / "catalog"
    db = Metadata(catalog)
    access = Access(db, catalog)
    session = access.exchange(access._local_owner_token())
    try:
        runtime_config = runtime_configuration(output)
        write(output / "runtime-config.json", runtime_config)
        with access.bind(session["token"]):
            owner = RuntimeSnapshots(db, access, catalog)
            inspection = owner.inspect(runtime_config)
            write(output / "runtime-inspection.json", inspection)
            runtime = owner.create(runtime_config, inspection["inventory_sha256"], quiesced=True)
        write(output / "runtime-descriptor.json", runtime)
    finally:
        with access.bind(session["token"]):
            access.revoke(session["identity"]["credential_id"])
        db.close()
    assembled = project.assemble(Path(selected["packet"]), Path(runtime["root"]) / "manifest.json", tuldok, output / "project")
    from app import Dataset
    dataset = Dataset(catalog, project_roots={"local-pressure": output / "project"})
    session = dataset.access.exchange(dataset.access._local_owner_token(require_existing=True))
    try:
        with dataset.access.bind(session["token"]):
            opened = dataset.projects.open({"root": "local-pressure", "manifest_path": "project.json"})
            manifest = decode(read(output / "project/project.json", MIB))
            recipe = next(r for r in manifest["recipes"] if r["id"] == "train")
            graph = dataset.pipelines.create_external({"name": config["id"] + " dry preflight", "project_id": opened["id"], "recipe_id": "train", "expected_sha256": recipe["sha256"], "inputs": {slot: {"reference_id": slot, "expected_sha256": assembled_input["sha256"]} for slot, assembled_input in decode(read(output / "project/source-witness.json", 2 * MIB))["inputs"].items()}, "parameters": {}})
            preflight = dataset.pipelines.preflight_external(graph["id"], graph["revision"])
            require(preflight["valid"], "preflight_failed", str(preflight.get("errors")))
            require(dataset.db.execute("SELECT COUNT(*) FROM workspace_runs").fetchone()[0] == 0, "preflight_executed", "dry preparation creates no Run")
        write(output / "preflight.json", preflight)
        write(output / "prepared.json", {"status": "prepared_without_runs", "experiment_sha256": sha(canonical(config)), "source_witness_sha256": assembled["source_witness_sha256"], "runtime_snapshot_sha256": assembled["runtime_snapshot_sha256"], "graph_id": graph["id"], "graph_revision": graph["revision"], "material_sha256": preflight["material_sha256"]})
        return {"status": "prepared_without_runs", "workspace": str(output)}
    finally:
        with dataset.access.bind(session["token"]):
            dataset.access.revoke(session["identity"]["credential_id"])
        dataset.close()
        postgres(host_config, "stop")


def execute(workspace):
    scientific_environment()
    prepared = decode(read(workspace / "prepared.json", MIB))
    require(prepared["status"] == "prepared_without_runs" and not (workspace / "execution").exists(), "execution_exists", "explicit once-only execution; never replay prior attempts")
    host_config = decode(read(workspace / "host.json", 65536))
    tuldok = Path(host_config["tuldok_source"])
    postgres(host_config, "start")
    database_environment(host_config)
    try:
        subprocess.run([host_config["python"], str(tuldok / "tests/qualify_project_cli.py"), "--catalog", str(workspace / "catalog"), "--project", str(workspace / "project"), "--stages", str(workspace / "project/stages.json"), "--output", str(workspace / "execution")], check=True, timeout=1800)
        import evaluate
        selected = decode(read(workspace / "material.json", MIB))
        result = evaluate.run(Path(selected["capture"]), Path(selected["packet"]), Path(selected["native_binary"]), workspace / "execution/stage-00/artifacts", workspace / "execution/stage-01/artifacts", workspace / "evaluation")
        write(workspace / "completed.json", {"status": "complete_local_cpu_pipeline", "managed_execution_sha256": sha(read(workspace / "execution/receipt.json", MIB)), "evaluation_sha256": sha(read(workspace / "evaluation/evaluation.json", 16 * MIB)), "source_witness_sha256": prepared["source_witness_sha256"], "solver_acceleration_claim": False})
        return result
    finally:
        postgres(host_config, "stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("config", "host", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    prepare_parser.add_argument("--material", type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.config.absolute(), args.host.absolute(), args.output.absolute(), args.material) if args.command == "prepare" else execute(args.workspace.absolute())
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
