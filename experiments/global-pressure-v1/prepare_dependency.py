#!/usr/bin/env python3
"""Prepare or verify the exact lab-owned native dependency; never edit its source."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


LAB_ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY = Path(__file__).resolve().parent / "dependency"
BASE_COMMIT = "2209012bc5f902693d0f695ec1c2d854126af317"
RECEIPT_NAME = "source-receipt.json"


class PreparationError(ValueError):
    """Invalid source, patch, destination, or prepared content."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def strict_json(data):
    """Decode identity documents without duplicate keys or nonfinite numbers."""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise PreparationError("duplicate JSON key in dependency identity document")
            value[key] = item
        return value

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise PreparationError("nonfinite JSON number in dependency identity document")
        return value

    def reject_constant(_):
        raise PreparationError("nonfinite JSON number in dependency identity document")

    return json.loads(
        data, object_pairs_hook=unique_object,
        parse_float=finite_float, parse_constant=reject_constant,
    )


def git(source, *args):
    # Ambient Git work-tree/config overrides must never redirect the patch into
    # a checkout belonging to another owner. Status must not refresh its index.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CEILING_DIRECTORIES"] = str(Path(source).parent)
    result = subprocess.run(
        ["git", "--no-optional-locks", "--no-replace-objects", "-C", str(source), *args],
        capture_output=True, check=False, env=environment,
    )
    if result.returncode:
        raise PreparationError(
            f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def inventory(source):
    """Hash every relative file and its Git executable bit, excluding only receipt."""
    entries = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise PreparationError(f"symlink in prepared source: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PreparationError(f"unsupported prepared entry: {path}")
        name = path.relative_to(source).as_posix()
        if name == RECEIPT_NAME:
            continue
        entries.append(
            {
                "path": name,
                "mode": "100755" if path.stat().st_mode & 0o111 else "100644",
                "sha256": sha256(path.read_bytes()),
            }
        )
    return entries


def load_contract():
    manifest_bytes = (DEPENDENCY / "manifest.json").read_bytes()
    manifest = strict_json(manifest_bytes)
    if (
        type(manifest) is not dict
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1 or manifest.get("base_commit") != BASE_COMMIT
    ):
        raise PreparationError("unsupported dependency manifest or native base")
    patch_name = manifest["patch_file"]
    if Path(patch_name).name != patch_name:
        raise PreparationError("patch filename must stay inside dependency directory")
    patch = DEPENDENCY / patch_name
    if sha256(patch.read_bytes()) != manifest["patch_sha256"]:
        raise PreparationError("dependency patch hash mismatch")
    expected_bytes = (DEPENDENCY / "source-files.json").read_bytes()
    if sha256(expected_bytes) != manifest["source_files_sha256"]:
        raise PreparationError("dependency file inventory hash mismatch")
    expected = strict_json(expected_bytes)
    if sha256(canonical_json(expected)) != manifest["prepared_tree_sha256"]:
        raise PreparationError("dependency prepared tree identity mismatch")
    return manifest, patch, expected, sha256(manifest_bytes)


def receipt_for(manifest, manifest_hash):
    return {
        "schema_version": 1,
        "base_commit": BASE_COMMIT,
        "base_tree": manifest["base_tree"],
        "patch_sha256": manifest["patch_sha256"],
        "prepared_tree_sha256": manifest["prepared_tree_sha256"],
        "source_manifest_sha256": manifest_hash,
        "source_files_sha256": manifest["source_files_sha256"],
        "license_sha256": manifest["license_sha256"],
    }


def verify_prepared(source):
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise PreparationError("prepared source must be an actual directory")
    manifest, _, expected, manifest_hash = load_contract()
    receipt = strict_json((source / RECEIPT_NAME).read_bytes())
    # Python equality equates True, 1 and 1.0; identity documents must preserve
    # JSON types as well as values and reject every missing or unknown field.
    if (
        type(receipt) is not dict
        or type(receipt.get("schema_version")) is not int
        or canonical_json(receipt) != canonical_json(receipt_for(manifest, manifest_hash))
    ):
        raise PreparationError("prepared source receipt mismatch")
    if inventory(source) != expected:
        raise PreparationError("prepared source content/mode inventory mismatch")
    return receipt


def publish(staging, destination):
    """Claim the final directory exclusively, then publish its receipt last."""
    # mkdir is the no-replace operation: a raced file, link, or directory fails.
    # Only this invocation's freshly claimed directory may be removed on error.
    destination.mkdir()
    try:
        for entry in staging.iterdir():
            if entry.name != RECEIPT_NAME:
                entry.rename(destination / entry.name)
        (staging / RECEIPT_NAME).rename(destination / RECEIPT_NAME)
        verify_prepared(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise


def safe_destination(destination):
    destination = Path(destination).absolute()
    root = LAB_ROOT / "runs" / "global-pressure-v1-dependency"
    if root.resolve() != root:
        raise PreparationError("dependency output root contains a symlink")
    if destination.resolve() != destination:
        raise PreparationError("destination contains a symlink or traversal")
    if destination == root or root not in destination.parents:
        raise PreparationError(
            "destination must be a child of lab runs/global-pressure-v1-dependency"
        )
    if destination.exists() or destination.is_symlink():
        raise PreparationError("refusing to overwrite existing destination")
    return destination


def prepare(source, destination):
    manifest, patch, expected, manifest_hash = load_contract()
    source = Path(source).resolve(strict=True)
    if git(source, "rev-parse", "--show-toplevel").decode().strip() != str(source):
        raise PreparationError("source must be the Git repository root")
    if git(source, "rev-parse", "HEAD").decode().strip() != BASE_COMMIT:
        raise PreparationError("source HEAD differs from exact native base")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise PreparationError("source must be clean, including untracked files")
    if git(source, "rev-parse", "HEAD^{tree}").decode().strip() != manifest["base_tree"]:
        raise PreparationError("source base tree differs from manifest")
    destination = safe_destination(destination)
    # Only committed blobs enter the closure; ignored local files are never read.
    archive = git(source, "archive", "--format=tar", BASE_COMMIT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                name = Path(member.name)
                if (
                    name.is_absolute() or ".." in name.parts
                    or not (member.isdir() or member.isfile())
                ):
                    raise PreparationError("native archive contains an unsafe entry")
                target = staging / name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as stream:
                        target.write_bytes(stream.read())
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
        git(staging, "apply", "--check", str(patch))
        git(staging, "apply", str(patch))
        if inventory(staging) != expected:
            raise PreparationError("applied patch did not produce the qualified source closure")
        receipt = receipt_for(manifest, manifest_hash)
        (staging / RECEIPT_NAME).write_bytes(canonical_json(receipt))
        verify_prepared(staging)
        # Keep source distinct from build products; deny accidental local edits.
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        publish(staging, destination)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="clean Git checkout at the exact native base")
    parser.add_argument(
        "--destination", type=Path,
        default=LAB_ROOT / "runs/global-pressure-v1-dependency/rheon",
    )
    parser.add_argument("--verify", type=Path, metavar="PREPARED_SOURCE")
    args = parser.parse_args()
    if bool(args.source) == bool(args.verify):
        parser.error("provide exactly one of --source or --verify")
    try:
        result = (
            verify_prepared(args.verify) if args.verify
            else prepare(args.source, args.destination)
        )
    except (PreparationError, OSError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
        parser.exit(2, f"dependency preparation failed: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
