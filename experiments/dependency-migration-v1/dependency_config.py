"""Select dependencies; measure source, build and runtime identities separately.

Only current runtime manifests are consumed, through Tuldok's runtime owner.
This module does not provision dependencies or start services/process workloads.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess

SCHEMA = "rheon-dependency-selection-v1"
TEMPLATE = Path(__file__).with_name("config.template.json")
ENVIRONMENT = "RHEON_LAB_DEPENDENCY_CONFIG"


def require(condition, message):
    if not condition:
        raise ValueError("Dependency selection: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON"))


def fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value)
            and not set(value) - set(required) - set(optional), "unsupported fields")


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "file changed during measurement")
    return result.hexdigest()


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def source_identity(root, revision):
    """Bind base revision and actual tracked/unignored source bytes, including edits."""
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "full source revision required")
    require(git(root, "rev-parse", "HEAD").decode().strip() == revision,
            "selected source revision mismatch: " + str(root))
    paths = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
    files = []
    for raw in sorted(set(paths) - {b""}):
        relative = raw.decode("utf-8")
        path = root / relative
        if not path.exists():
            require(not path.is_symlink(), "dangling source symlink")
            files.append({"path": relative, "deleted": True})
            continue
        require(path.resolve().is_relative_to(root), "source file escapes root")
        files.append({"path": relative, "sha256": digest(path),
                      "executable": bool(path.stat().st_mode & stat.S_IXUSR)})
    return {"revision": revision, "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "content_sha256": hashlib.sha256(canonical(files)).hexdigest(), "files": files}


class Selection:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        raw = self.path.read_bytes()
        require(len(raw) <= 1024 * 1024, "configuration exceeds 1 MiB")
        self.value = decode(raw)
        fields(self.value, ("schema", "sources", "artifacts", "runtimes", "paths"))
        require(self.value["schema"] == SCHEMA, "unsupported configuration schema")
        for group in ("sources", "artifacts", "runtimes", "paths"):
            require(type(self.value[group]) is dict, "named " + group + " required")
        for row in self.value["sources"].values():
            fields(row, ("root", "revision"), ("content_sha256",))
        for row in self.value["artifacts"].values():
            fields(row, ("path", "sha256"), ("source", "source_content_sha256"))
            require(("source" in row) == ("source_content_sha256" in row), "build source binding incomplete")
        for row in self.value["runtimes"].values():
            fields(row, ("profile", "manifest", "manifest_sha256", "owner_sha256"))

    def resolve(self, value):
        require(type(value) is str and bool(value), "nonempty path required")
        path = Path(value).expanduser()
        return (self.path.parent / path).resolve() if not path.is_absolute() else path.resolve()

    def path_for(self, name):
        """Resolve an output/custody root without claiming it exists or is qualified."""
        return self.resolve(self.value["paths"][name])

    def source_root(self, name):
        """Verify base revision/path; the build owner binds its actual input closure."""
        row = self.value["sources"][name]
        root = self.resolve(row["root"])
        require(git(root, "rev-parse", "HEAD").decode().strip() == row["revision"],
                "selected source revision mismatch: " + name)
        return root

    def source(self, name):
        row = self.value["sources"][name]
        root = self.source_root(name)
        if "content_sha256" in row:
            require(source_identity(root, row["revision"])["content_sha256"] == row["content_sha256"],
                    "selected source content mismatch: " + name)
        return root

    def source_witness(self, name):
        row = self.value["sources"][name]
        return source_identity(self.source(name), row["revision"])

    def artifact(self, name):
        row = self.value["artifacts"][name]
        path = self.resolve(row["path"])
        require(digest(path) == row["sha256"], "artifact hash mismatch: " + name)
        if "source" in row:
            require(self.source_witness(row["source"])["content_sha256"] == row["source_content_sha256"],
                    "artifact source content mismatch: " + name)
        return path

    def runtime(self, name):
        row = self.value["runtimes"][name]
        path = self.resolve(row["manifest"])
        require(digest(path) == row["manifest_sha256"], "runtime manifest hash mismatch: " + name)
        owner = self.source_root("tuldok") / "torch_runtime.py"
        require(digest(owner) == row["owner_sha256"], "runtime owner source hash mismatch: " + name)
        spec = importlib.util.spec_from_file_location("selected_torch_runtime", owner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = module.load(path)
        require(manifest["profile"] == row["profile"], "runtime profile mismatch: " + name)
        module.verify(manifest)
        require(digest(owner) == row["owner_sha256"], "runtime owner changed during verification")
        return {"profile": row["profile"], "manifest": str(path),
                "manifest_sha256": row["manifest_sha256"], "python": manifest["python"],
                "python_sha256": digest(manifest["python"]), "owner_sha256": digest(owner),
                "file_count": len(manifest["files"]), "total_bytes": manifest["total_bytes"]}

    def witness(self):
        return {"schema": "rheon-dependency-witness-v1",
                "selection_sha256": hashlib.sha256(canonical(self.value)).hexdigest(),
                "sources": {name: self.source_witness(name) for name in self.value["sources"]},
                "artifacts": {name: {"sha256": digest(self.artifact(name))} for name in self.value["artifacts"]},
                "runtimes": {name: self.runtime(name) for name in self.value["runtimes"]}}


def default_selection():
    value = decode(TEMPLATE.read_bytes())
    output = Path(value["paths"]["migration"]).expanduser()
    root = (TEMPLATE.parent / output).resolve() if not output.is_absolute() else output.resolve()
    return root / "selection-current.json"


def load(path=None):
    return Selection(path or os.environ.get(ENVIRONMENT) or default_selection())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    witness = load(args.dependency_config).witness()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(witness) + b"\n")
    print(json.dumps({"status": "measured_dependencies", "witness": str(args.output),
                      "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
