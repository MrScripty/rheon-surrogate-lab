"""Prepare the selected clean Rheon revision with the retained pressure patch."""
import io
from pathlib import Path
import subprocess
import tarfile

from contract import LAB, MIB, canonical, decode, fresh, read, require, sha, write

HISTORICAL = LAB / "experiments/global-pressure-v1/dependency"


def inventory(root):
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "dependency_path", str(path))
        if path.is_dir() or path.name == "source-receipt.json" and path.parent == root:
            continue
        require(path.is_file(), "dependency_path", "regular source files")
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha(read(path, 16 * MIB)), "mode": 0o755 if path.stat().st_mode & 0o111 else 0o644})
    return rows


def prepare(source, revision, output):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == revision and not subprocess.check_output(["git", "status", "--porcelain"], cwd=source), "dependency_revision", "clean selected source revision")
    contract = decode(read(HISTORICAL / "manifest.json", MIB))
    patch = read(HISTORICAL / contract["patch_file"], MIB)
    require(sha(patch) == contract["patch_sha256"], "dependency_patch", "retained reviewed pressure patch")
    output = fresh(output)
    archive = subprocess.check_output(["git", "archive", revision], cwd=source)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(output, filter="data")
    # An isolated index prevents git apply from resolving the surrounding lab
    # checkout when a prepared tree lives beneath runs/.
    subprocess.run(["git", "init", "--quiet"], cwd=output, check=True)
    subprocess.run(["git", "apply", "--check", str(HISTORICAL / contract["patch_file"])], cwd=output, check=True)
    subprocess.run(["git", "apply", str(HISTORICAL / contract["patch_file"])], cwd=output, check=True)
    __import__("shutil").rmtree(output / ".git")
    files = inventory(output)
    receipt = {"schema": "rheon-local-pressure-dependency-v1", "base_commit": revision, "base_tree": subprocess.check_output(["git", "rev-parse", revision + "^{tree}"], cwd=source, text=True).strip(), "patch_sha256": sha(patch), "prepared_tree_sha256": sha(canonical(files)), "license_sha256": sha(read(output / "LICENSE", MIB)), "files": files}
    write(output / "source-receipt.json", receipt)
    return verify(output)


def verify(root):
    receipt = decode(read(root / "source-receipt.json", 2 * MIB))
    require(receipt["schema"] == "rheon-local-pressure-dependency-v1" and receipt["files"] == inventory(root) and receipt["prepared_tree_sha256"] == sha(canonical(receipt["files"])), "dependency_identity", "selected source/patch closure unchanged")
    contract = decode(read(HISTORICAL / "manifest.json", MIB))
    require(receipt["patch_sha256"] == contract["patch_sha256"] and receipt["license_sha256"] == sha(read(root / "LICENSE", MIB)), "dependency_contract", "reviewed patch and license")
    return {key: value for key, value in receipt.items() if key != "files"}
