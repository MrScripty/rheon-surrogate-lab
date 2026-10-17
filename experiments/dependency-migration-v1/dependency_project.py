"""Produce the current live project against explicit source and artifact roots."""
import argparse
import copy
import json
from pathlib import Path
import sys
import uuid

from dependency_config import canonical, decode, digest, load, require


def assemble(selection, output):
    lab = selection.source("lab")
    root = selection.path_for("artifact_root")
    original = lab / "experiments/dependency-migration-v1/dependency-original-project.json"
    current = lab / "tuldok-project.json"
    value = decode(current.read_bytes())
    require(value["schema_version"] == 2 and "activity_path" not in value,
            "only current project manifest is accepted")
    manifest = copy.deepcopy(value)
    source_project_id = manifest["id"]
    manifest["id"] = uuid.uuid4().hex
    references = []
    for group in ("data", "recipes", "evidence", "extensions"):
        for entry in manifest[group]:
            supplied = Path(entry["path"])
            require(not supplied.is_absolute() and ".." not in supplied.parts,
                    "source manifest requires contained relative paths")
            path = (root if supplied.parts[0] == "runs" else lab) / supplied
            path = path.resolve(strict=True)
            require(path.is_relative_to(root), "project reference escapes bound root")
            measured = digest(path)
            if "sha256" in entry:
                require(measured == entry["sha256"], "retained project reference hash mismatch: " + entry["id"])
            entry["path"] = path.relative_to(root).as_posix()
            references.append({"group": group, "id": entry["id"], "old_path": supplied.as_posix(),
                               "current_path": entry["path"], "measured_sha256": measured})
    sys.path.insert(0, str(selection.source("tuldok")))
    from projects import manifest as validate
    validate(manifest)
    output = Path(output).resolve()
    require(output.is_relative_to(root), "current project must belong to configured root")
    require(not output.exists(), "fresh project output required")
    output.mkdir(parents=True)
    raw = canonical(manifest)
    (output / "project.json").write_bytes(raw)
    mapping = {"schema": "rheon-current-project-provenance-v1", "project_id": manifest["id"],
               "source_project_id": source_project_id, "identity_relation": "new_current_project",
               "original_manifest_sha256": digest(original), "original_manifest_archive": str(original),
               "current_source_manifest_sha256": digest(current),
               "current_manifest_sha256": digest(output / "project.json"),
               "root": str(root), "manifest_path": (output / "project.json").relative_to(root).as_posix(),
               "references": references, "execution_or_acceptance_claim": False}
    (output / "provenance.json").write_bytes(canonical(mapping))
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selection = load(args.dependency_config)
    result = assemble(selection, args.output or selection.path_for("migration") / "current-project")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
