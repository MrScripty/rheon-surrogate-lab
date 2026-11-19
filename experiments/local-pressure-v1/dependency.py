"""Verify the current Rheon source selected by the migration configuration."""
import sys
from pathlib import Path

from contract import LAB, MIB, canonical, read, require, sha

sys.path.insert(0, str(LAB / "experiments/dependency-migration-v1"))
import dependency_config


def selection(path=None):
    return dependency_config.load(path)


def verify(root, configuration=None):
    selected = configuration or selection()
    require(selected.source_root("lab") == LAB.resolve(), "dependency_lab", "configuration-selected current lab root")
    source = selected.source_root("rheon")
    require(Path(root).resolve() == source, "dependency_source", "configuration-selected current Rheon root")
    # These are the Cargo/lib inputs for the selected native probes. No lean
    # library include_str/include_bytes refers outside src. Cargo does not compile
    # dependency binaries or their desktop tests for these library consumers.
    # Bind every other src file, including shared diagnostic ownership, so newly
    # added library modules cannot escape the build identity.
    paths = [source/name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    if (source/"build.rs").exists():
        paths.append(source/"build.rs")
    paths += sorted(path for path in (source/"src").rglob("*")
                    if path.is_file() and not path.is_relative_to(source/"src/bin"))
    files = [{"path": path.relative_to(source).as_posix(), "sha256": sha(read(path, 16*MIB))} for path in paths]
    row = selected.value["sources"]["rheon"]
    return {"schema": "rheon-current-pressure-dependency-v1", "base_commit": row["revision"],
            "source_content_sha256": sha(canonical(files)), "build_inputs": files,
            "license_sha256": sha(read(source / "LICENSE", MIB))}
