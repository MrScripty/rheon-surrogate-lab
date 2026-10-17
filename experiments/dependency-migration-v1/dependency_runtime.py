"""Measure the current runtime contract and prove retained numerical assets.

This explicit migration utility owns old-to-new manifest evidence. Runtime
callers consume only the emitted current contract through the Tuldok owner.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from dependency_config import canonical, decode, digest, load, require


def measure(selection, baseline, expected_sha256, output):
    baseline = Path(baseline).resolve(strict=True)
    require(digest(baseline) == expected_sha256, "fixed baseline runtime manifest changed")
    original = decode(baseline.read_bytes())
    owner_path = selection.source("tuldok") / "torch_runtime.py"
    spec = importlib.util.spec_from_file_location("migration_runtime_owner", owner_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    started = time.monotonic()
    current = owner.inventory({name: original[name] for name in ("python", "python_root", "site_packages")})
    # The original independently fixed numerical-runtime fields remain binding.
    # Added current topology witnesses describe fresh observation only.
    require(all(current.get(name) == value for name, value in original.items()),
            "fixed numerical runtime changed; do not choose a substitute")
    owner.verify(current)
    raw = owner.encoded(current)
    output = Path(output).resolve()
    require(not output.exists(), "fresh runtime output required")
    output.mkdir(parents=True)
    path = output / "manifest.json"
    path.write_bytes(raw)
    receipt = {"schema": "rheon-current-runtime-witness-v1", "status": "verified_fixed_installed_runtime",
               "elapsed_seconds": time.monotonic() - started, "original_manifest": str(baseline),
               "original_manifest_sha256": expected_sha256, "current_manifest": str(path),
               "current_manifest_sha256": digest(path), "owner_sha256": digest(owner_path),
               "all_original_fields_and_asset_bytes_identical": True,
               "added_current_fields": sorted(set(current) - set(original)),
               "file_count": len(current["files"]), "file_link_count": len(current["file_links"]),
               "total_bytes": current["total_bytes"], "fresh_inventory": True, "full_second_verification": True,
               "assumption": current["assumption"]}
    (output / "receipt.json").write_bytes(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(load(args.dependency_config), args.baseline_manifest, args.expected_sha256, args.output)))


if __name__ == "__main__":
    main()
