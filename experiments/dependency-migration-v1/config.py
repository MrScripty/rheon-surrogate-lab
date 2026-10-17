"""Freeze a reusable template into one attempt's measured dependency selection."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from dependency_config import canonical, decode, digest, git, require, source_identity, default_selection


def freeze(template, output, cargo=None):
    template = Path(template).resolve(strict=True)
    value = decode(template.read_bytes())
    require(value.pop("schema") == "rheon-dependency-template-v1", "current template required")
    toolchain = value.pop("rust_toolchain")
    value["schema"] = "rheon-dependency-selection-v1"
    def resolve(raw):
        path = Path(raw).expanduser()
        return str((template.parent / path).resolve() if not path.is_absolute() else path.resolve())
    if cargo is None:
        rustup = shutil.which("rustup")
        require(rustup is not None, "select --cargo explicitly or install the configured Rust toolchain")
        cargo = subprocess.check_output([rustup, "which", "--toolchain", toolchain, "cargo"], text=True).strip()
    cargo = Path(cargo).expanduser().resolve(strict=True)
    require(subprocess.check_output([str(cargo), "--version"], text=True).startswith("cargo " + toolchain + " "),
            "selected cargo does not match configured toolchain")
    for row in value["sources"].values():
        root = Path(resolve(row["root"]))
        revision = git(root, "rev-parse", "HEAD").decode().strip()
        row.update(root=str(root), revision=revision,
                   content_sha256=source_identity(root, revision)["content_sha256"])
    for row in value["artifacts"].values():
        row["path"] = resolve(row["path"])
        require(digest(row["path"]) == row["sha256"], "selected retained artifact changed")
    for row in value["runtimes"].values():
        row["manifest"] = resolve(row["manifest"])
        row["manifest_sha256"] = digest(row["manifest"])
        row["owner_sha256"] = digest(Path(value["sources"]["tuldok"]["root"]) / "torch_runtime.py")
    value["paths"] = {name: resolve(path) for name, path in value["paths"].items()}
    value["paths"]["cargo"] = str(cargo)
    output = Path(output).resolve()
    require(not output.exists(), "fresh attempt selection required; use a new output path")
    # The selection cannot contain its own source/content identity.
    for row in value["sources"].values():
        root = Path(row["root"])
        if output.is_relative_to(root):
            result = subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", str(output)], check=False)
            require(result.returncode == 0, "selection output must be outside measured source closure")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical(value))
    return {"status": "frozen_dependency_selection", "path": str(output), "sha256": digest(output),
            "source_revisions": {name: row["revision"] for name, row in value["sources"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("config.template.json"))
    parser.add_argument("--output", type=Path, default=default_selection())
    parser.add_argument("--cargo", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.template, args.output, args.cargo)))


if __name__ == "__main__":
    main()
