"""Qualified builds and calls for the selected prepared native dependency."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from contract import HERE, LAB, MIB, canonical, decode, fresh, read, require, sha, write
import dependency
import grid


def sources(kind):
    require(kind == "capture", "build_kind", "cubic-cell capture only")
    paths = [Path(__file__).parent / n for n in ("grid.py", "build_native.py", "cubic_capture.py", "pipeline.py")]
    paths += [HERE / n for n in ("contract.py", "dependency.py")]
    paths += [LAB / "probes/cubic-pressure-v1" / n for n in ("Cargo.toml", "Cargo.lock", "build.rs", "src/main.rs")]
    paths += [LAB / "experiments/global-pressure-v1" / n for n in ("capture.py", "verify_capture.py", "prepare_dependency.py")]
    return {p.relative_to(LAB).as_posix(): sha(read(p, 2 * MIB)) for p in paths}


def build(kind, config, prepared, cargo, output):
    receipt = dependency.verify(prepared)
    output = fresh(output)
    require(kind == "capture", "build_kind", "cubic-cell capture only")
    crate = "cubic-pressure-v1"
    binary_name = "rheon-cubic-pressure-capture"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CARGO_", "RHEON_", "RUSTFLAGS", "RUSTC", "RUSTDOC", "RUSTUP_TOOLCHAIN"))}
    env.update(CARGO_HOME=str(LAB / ".cargo-local"), CARGO_TARGET_DIR=str(LAB / "runs/local-pressure-native-target"), RUSTUP_TOOLCHAIN="1.92.0", RAYON_NUM_THREADS="1", RUSTC=str(cargo.parent / "rustc"), RUSTDOC=str(cargo.parent / "rustdoc"), RUSTC_WRAPPER="", RUSTC_WORKSPACE_WRAPPER="")
    cargo_version = subprocess.check_output([str(cargo), "--version"], env=env, text=True).strip()
    rustc_version = subprocess.check_output([env["RUSTC"], "-vV"], env=env, text=True).strip()
    require(cargo_version.startswith("cargo 1.92.0 ") and "release: 1.92.0" in rustc_version, "toolchain", "Rust/Cargo 1.92.0")
    common = ["--manifest-path", str(LAB / "probes" / crate / "Cargo.toml"), "--config", f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(prepared))}', "--locked"]
    metadata = decode(subprocess.check_output([str(cargo), "metadata", *common, "--format-version", "1"], env=env))
    packages = [p for p in metadata["packages"] if p["name"] == "rheon"]
    require(len(packages) == 1 and packages[0]["source"] is None and Path(packages[0]["manifest_path"]) == prepared / "Cargo.toml", "native_graph", "actual selected prepared dependency")
    write(output / "cargo-metadata.json", metadata)
    binding = {"schema": "rheon-cubic-pressure-build-v1", "kind": kind, "prepared_source": str(prepared), "native_receipt": receipt, "source_sha256": sources(kind), "metadata_sha256": sha(read(output / "cargo-metadata.json")), "toolchain": {"cargo": cargo_version, "rustc": rustc_version}}
    if kind == "capture":
        write(output / "reservation.json", grid.native_reservation(config))
        binding.update(experiment_sha256=sha(canonical(config)), reservation_sha256=sha(read(output / "reservation.json")))
        env["RHEON_CAPTURE_RESERVATION"] = str(output / "reservation.json")
    write(output / "binding.json", binding)
    env.update(RHEON_BUILD_BINDING=str(output / "binding.json"), RHEON_BUILD_BINDING_SHA256=sha(read(output / "binding.json")))
    for operation in ("build", "test", "clippy"):
        tail = ["--", "-D", "warnings"] if operation == "clippy" else []
        subprocess.run([str(cargo), operation, *common, "--release", *tail], env=env, check=True, timeout=600)
    binary = output / binary_name
    shutil.copyfile(Path(env["CARGO_TARGET_DIR"]) / "release" / binary_name, binary)
    binary.chmod(0o555)
    identity = decode(subprocess.check_output([str(binary), "--identity"], timeout=10))
    require(identity["binding"] == binding and identity["native_receipt"] == dependency.verify(prepared), "build_identity", "native closure unchanged")
    write(output / "build-receipt.json", {"status": "qualified_build", "binary_sha256": sha(read(binary)), "identity": identity})
    return binary


def admit(binary):
    receipt = decode(read(binary.parent / "build-receipt.json", 2 * MIB))
    identity = decode(subprocess.check_output([str(binary), "--identity"], timeout=10))
    binding = identity["binding"]
    require(receipt["status"] == "qualified_build" and receipt["binary_sha256"] == sha(read(binary)) and receipt["identity"] == identity and decode(read(binary.parent / "binding.json", 2 * MIB)) == binding and binding["source_sha256"] == sources(binding["kind"]) and binding["metadata_sha256"] == sha(read(binary.parent / "cargo-metadata.json")) and binding["native_receipt"] == dependency.verify(Path(binding["prepared_source"])) and identity["native_threads"] == 1, "native_identity", "measured source/graph/dependency/binary")
    return receipt
