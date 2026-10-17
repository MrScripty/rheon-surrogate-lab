"""Qualified builds and calls for the configuration-selected current native source."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from contract import HERE, LAB, MIB, canonical, decode, fresh, load_module, read, require, reservation, sha, write
import dependency
import dependency_config


def sources(kind):
    crate = "local-pressure-v1" if kind == "capture" else "local-pressure-evaluation-v1"
    paths = [HERE / n for n in ("contract.py", "dependency.py", "native.py", "process.py")]
    paths += [LAB / "experiments/dependency-migration-v1/dependency_config.py"]
    paths += [LAB / "probes" / crate / n for n in ("Cargo.toml", "Cargo.lock", "build.rs", "src/main.rs")]
    if kind == "capture":
        paths += [HERE / "capture.py", LAB / "experiments/global-pressure-v1/capture.py", LAB / "experiments/global-pressure-v1/verify_capture.py"]
    return {p.relative_to(LAB).as_posix(): sha(read(p, 2 * MIB)) for p in paths}


def build(kind, config, source, cargo, output):
    selection = dependency.selection()
    receipt = dependency.verify(source, selection)
    require(cargo.resolve() == selection.path_for("cargo"), "native_toolchain", "configuration-selected Cargo")
    output = fresh(output)
    crate = "local-pressure-v1" if kind == "capture" else "local-pressure-evaluation-v1"
    binary_name = "rheon-local-pressure-capture" if kind == "capture" else "rheon-local-pressure-evaluation"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CARGO_", "RHEON_", "RUSTFLAGS", "RUSTC", "RUSTDOC", "RUSTUP_TOOLCHAIN"))}
    env.update(CARGO_HOME=str(selection.path_for("cargo_home")), CARGO_TARGET_DIR=str(selection.path_for("cargo_target")), RUSTUP_TOOLCHAIN="1.92.0", RAYON_NUM_THREADS="1", CARGO_BUILD_JOBS="2", RUSTC=str(cargo.parent / "rustc"), RUSTDOC=str(cargo.parent / "rustdoc"), RUSTC_WRAPPER="", RUSTC_WORKSPACE_WRAPPER="")
    cargo_version = subprocess.check_output([str(cargo), "--version"], env=env, text=True).strip()
    rustc_version = subprocess.check_output([env["RUSTC"], "-vV"], env=env, text=True).strip()
    require(cargo_version.startswith("cargo 1.92.0 ") and "release: 1.92.0" in rustc_version, "toolchain", "Rust/Cargo 1.92.0")
    common = ["--manifest-path", str(LAB / "probes" / crate / "Cargo.toml"), "--config", f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(source))}', "--offline", "--locked"]
    metadata = decode(subprocess.check_output([str(cargo), "metadata", *common, "--format-version", "1"], env=env))
    packages = [p for p in metadata["packages"] if p["name"] == "rheon"]
    require(len(packages) == 1 and packages[0]["source"] is None and Path(packages[0]["manifest_path"]) == source / "Cargo.toml", "native_graph", "actual selected current dependency")
    write(output / "cargo-metadata.json", metadata)
    binding = {"schema": "rheon-current-pressure-build-v1", "kind": kind, "source_root": str(source), "native_receipt": receipt, "source_sha256": sources(kind), "metadata_sha256": sha(read(output / "cargo-metadata.json")), "toolchain": {"cargo": cargo_version, "rustc": rustc_version, "executables": {str(path): dependency_config.digest(path) for path in (cargo, cargo.parent/"rustc", cargo.parent/"rustdoc")}}}
    if kind == "capture":
        write(output / "reservation.json", reservation(config))
        binding.update(experiment_sha256=sha(canonical(config)), reservation_sha256=sha(read(output / "reservation.json")))
        env["RHEON_CAPTURE_RESERVATION"] = str(output / "reservation.json")
    write(output / "binding.json", binding)
    env.update(RHEON_BUILD_BINDING=str(output / "binding.json"), RHEON_BUILD_BINDING_SHA256=sha(read(output / "binding.json")))
    commands = []
    for operation in ("build", "test", "clippy"):
        tail = ["--", "-D", "warnings"] if operation == "clippy" else []
        command = [str(cargo), operation, *common, "--release", *tail]
        started = time.monotonic()
        subprocess.run(command, env=env, check=True, timeout=600)
        commands.append({"command": command, "elapsed_s": time.monotonic()-started, "exit_code": 0})
    binary = output / binary_name
    shutil.copyfile(Path(env["CARGO_TARGET_DIR"]) / "release" / binary_name, binary)
    binary.chmod(0o555)
    identity = decode(subprocess.check_output([str(binary), "--identity"], timeout=10))
    require(identity["binding"] == binding and identity["native_receipt"] == dependency.verify(source), "build_identity", "native closure unchanged")
    write(output / "build-receipt.json", {"status": "qualified_build", "binary_sha256": sha(read(binary)), "identity": identity, "commands": commands})
    return binary


def admit(binary):
    receipt = decode(read(binary.parent / "build-receipt.json", 2 * MIB))
    identity = decode(subprocess.check_output([str(binary), "--identity"], timeout=10))
    binding = identity["binding"]
    require(receipt["status"] == "qualified_build" and receipt["binary_sha256"] == sha(read(binary)) and receipt["identity"] == identity and decode(read(binary.parent / "binding.json", 2 * MIB)) == binding and binding["source_sha256"] == sources(binding["kind"]) and binding["metadata_sha256"] == sha(read(binary.parent / "cargo-metadata.json")) and binding["native_receipt"] == dependency.verify(Path(binding["source_root"])) and identity["native_threads"] == 1, "native_identity", "measured source/graph/dependency/binary")
    return receipt


def call(binary, mode, request=None, timeout=60):
    # Reuse the qualified Linux descendant/stream owner, including its bounds.
    process = load_module("local_pressure_native_process", HERE / "process.py")
    return process.native_call(binary, mode, request, time.monotonic() + timeout + 3, timeout)
