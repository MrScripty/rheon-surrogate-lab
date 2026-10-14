"""Preparation trust-boundary tests, independent of native numerical tests."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "prepare_dependency", Path(__file__).with_name("prepare_dependency.py")
)
dependency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependency)


class DependencyPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="global-pressure-dependency-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.run_git("init", "-q")
        (self.source / "LICENSE").write_text("fixture license\n")
        (self.source / "input.txt").write_text("original\n")
        (self.source / ".gitignore").write_text("ignored\n")
        self.run_git("add", ".")
        self.run_git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        )
        base = self.run_git("rev-parse", "HEAD").strip()
        dep = self.root / "dependency"
        dep.mkdir()
        patch_bytes = (
            b"diff --git a/input.txt b/input.txt\n--- a/input.txt\n+++ b/input.txt\n"
            b"@@ -1 +1 @@\n-original\n+qualified\n"
        )
        (dep / "fixture.patch").write_bytes(patch_bytes)
        expected = [
            {"path": name, "mode": "100644", "sha256": dependency.sha256(content)}
            for name, content in [
                (".gitignore", b"ignored\n"), ("LICENSE", b"fixture license\n"),
                ("input.txt", b"qualified\n"),
            ]
        ]
        inv = dependency.canonical_json(expected)
        (dep / "source-files.json").write_bytes(inv)
        manifest = {
            "schema_version": 1, "base_commit": base,
            "base_tree": self.run_git("rev-parse", "HEAD^{tree}").strip(),
            "patch_file": "fixture.patch", "patch_sha256": dependency.sha256(patch_bytes),
            "source_files_sha256": dependency.sha256(inv),
            "prepared_tree_sha256": dependency.sha256(inv),
            "license_sha256": dependency.sha256(b"fixture license\n"),
        }
        (dep / "manifest.json").write_bytes(dependency.canonical_json(manifest))
        for name, value in [("BASE_COMMIT", base), ("DEPENDENCY", dep), ("LAB_ROOT", self.root)]:
            mock = patch.object(dependency, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.destination = self.root / "runs/global-pressure-v1-dependency/rheon"

    def run_git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def test_deterministic_closure_ignores_uncommitted_ignored_files(self):
        (self.source / "ignored").write_text("never exported\n")
        first = dependency.prepare(self.source, self.destination)
        second_path = self.destination.with_name("second")
        second = dependency.prepare(self.source, second_path)
        self.assertEqual(first, second)
        self.assertEqual(
            dependency.inventory(self.destination), dependency.inventory(second_path)
        )
        self.assertFalse((self.destination / "ignored").exists())
        self.assertEqual(dependency.verify_prepared(self.destination), first)
        self.assertFalse((self.destination / "input.txt").stat().st_mode & 0o222)

    def test_source_pin_tracked_and_untracked_drift_fail_before_output(self):
        (self.source / "input.txt").write_text("drift\n")
        with self.assertRaisesRegex(dependency.PreparationError, "clean"):
            dependency.prepare(self.source, self.destination)
        self.assertFalse(self.destination.parent.exists())

        self.run_git("checkout", "--", "input.txt")
        (self.source / "extra").write_text("drift\n")
        with self.assertRaisesRegex(dependency.PreparationError, "clean"):
            dependency.prepare(self.source, self.destination)
        self.run_git("add", "extra")
        self.run_git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "drift",
        )
        with self.assertRaisesRegex(dependency.PreparationError, "HEAD"):
            dependency.prepare(self.source, self.destination)
        self.assertFalse(self.destination.parent.exists())

    def test_ambient_git_overrides_cannot_redirect_mutation_into_source(self):
        with patch.dict(os.environ, {
            "GIT_DIR": str(self.source / ".git"),
            "GIT_WORK_TREE": str(self.source),
            "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "apply.whitespace",
            "GIT_CONFIG_VALUE_0": "error",
        }):
            dependency.prepare(self.source, self.destination)
        self.assertEqual((self.source / "input.txt").read_text(), "original\n")
        self.assertEqual((self.destination / "input.txt").read_text(), "qualified\n")

    def test_patch_and_inventory_tampering_fail_before_output(self):
        patch_file = dependency.DEPENDENCY / "fixture.patch"
        old = patch_file.read_bytes()
        patch_file.write_bytes(old + b"drift")
        with self.assertRaisesRegex(dependency.PreparationError, "patch hash"):
            dependency.prepare(self.source, self.destination)
        patch_file.write_bytes(old)
        (dependency.DEPENDENCY / "source-files.json").write_bytes(b"[]\n")
        with self.assertRaisesRegex(dependency.PreparationError, "inventory hash"):
            dependency.prepare(self.source, self.destination)
        self.assertFalse(self.destination.parent.exists())

    def test_unsafe_existing_and_symlink_destinations_fail(self):
        for target in [self.root / "outside", self.root / "runs/global-pressure-v1-dependency"]:
            with self.assertRaisesRegex(dependency.PreparationError, "child"):
                dependency.prepare(self.source, target)
        dependency.prepare(self.source, self.destination)
        with self.assertRaisesRegex(dependency.PreparationError, "overwrite"):
            dependency.prepare(self.source, self.destination)
        linked = self.destination.with_name("linked")
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(dependency.PreparationError, "symlink"):
            dependency.prepare(self.source, linked / "new")

    def test_prepared_content_mode_extra_symlink_and_receipt_drift_reject(self):
        dependency.prepare(self.source, self.destination)
        target = self.destination / "input.txt"
        target.chmod(0o644)
        target.write_text("drift\n")
        with self.assertRaisesRegex(dependency.PreparationError, "inventory"):
            dependency.verify_prepared(self.destination)
        target.write_text("qualified\n")
        target.chmod(0o755)
        with self.assertRaisesRegex(dependency.PreparationError, "inventory"):
            dependency.verify_prepared(self.destination)
        target.chmod(0o444)
        extra = self.destination / "extra"
        extra.write_text("extra\n")
        with self.assertRaisesRegex(dependency.PreparationError, "inventory"):
            dependency.verify_prepared(self.destination)
        extra.unlink()
        extra.symlink_to(target)
        with self.assertRaisesRegex(dependency.PreparationError, "symlink"):
            dependency.verify_prepared(self.destination)
        extra.unlink()
        receipt = self.destination / dependency.RECEIPT_NAME
        receipt.chmod(0o644)
        value = json.loads(receipt.read_text())
        value["base_commit"] = "drift"
        receipt.write_text(json.dumps(value))
        with self.assertRaisesRegex(dependency.PreparationError, "receipt"):
            dependency.verify_prepared(self.destination)

    def test_raced_publication_preserves_foreign_file_symlink_and_directory(self):
        original_publish = dependency.publish
        for kind in ["file", "symlink", "directory"]:
            destination = self.destination.with_name(kind)

            def raced_publish(staging, target):
                if kind == "file":
                    target.write_text("foreign\n")
                elif kind == "symlink":
                    target.symlink_to(self.source, target_is_directory=True)
                else:
                    target.mkdir()
                original_publish(staging, target)

            with patch.object(dependency, "publish", raced_publish):
                with self.assertRaises(FileExistsError):
                    dependency.prepare(self.source, destination)
            if kind == "file":
                self.assertEqual(destination.read_text(), "foreign\n")
            elif kind == "symlink":
                self.assertTrue(destination.is_symlink())
                self.assertEqual(destination.resolve(), self.source)
            else:
                self.assertEqual(list(destination.iterdir()), [])
            self.assertFalse(list(destination.parent.glob(".prepare-*")))

    def test_failed_publication_cleans_only_its_claimed_destination(self):
        original_verify = dependency.verify_prepared
        unrelated = self.destination.with_name("unrelated")
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("preserve\n")

        def reject_final(source):
            if source == self.destination:
                raise dependency.PreparationError("simulated publication failure")
            return original_verify(source)

        with patch.object(dependency, "verify_prepared", reject_final):
            with self.assertRaisesRegex(dependency.PreparationError, "publication failure"):
                dependency.prepare(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.destination.parent.glob(".prepare-*")))
        self.assertEqual(unrelated.read_text(), "preserve\n")

    def test_receipt_types_unknown_and_missing_fields_reject_on_valid_closure(self):
        receipt = dependency.prepare(self.source, self.destination)
        original_read = Path.read_bytes
        bad_receipts = [
            {**receipt, "schema_version": value}
            for value in [True, False, 1.0, "1", None, [], {}]
        ]
        bad_receipts += [
            {**receipt, "unknown": "field"},
            {key: value for key, value in receipt.items() if key != "base_tree"},
            {**receipt, "base_tree": [receipt["base_tree"]]},
            {**receipt, "patch_sha256": True},
            [], None,
        ]
        for bad in bad_receipts:
            with self.subTest(receipt=bad):
                encoded = json.dumps(bad).encode()

                def read_with_bad_receipt(path):
                    if path == self.destination / dependency.RECEIPT_NAME:
                        return encoded
                    return original_read(path)

                with patch.object(Path, "read_bytes", read_with_bad_receipt):
                    with self.assertRaisesRegex(dependency.PreparationError, "receipt"):
                        dependency.verify_prepared(self.destination)
        self.assertEqual(dependency.verify_prepared(self.destination), receipt)

    def test_manifest_schema_types_reject_before_output(self):
        path = dependency.DEPENDENCY / "manifest.json"
        original = path.read_bytes()
        manifest = json.loads(original)
        for value in [True, False, 1.0, "1", None, [], {}]:
            with self.subTest(schema_version=value):
                path.write_text(json.dumps({**manifest, "schema_version": value}))
                with self.assertRaisesRegex(dependency.PreparationError, "manifest"):
                    dependency.prepare(self.source, self.destination)
        for value in [[], None]:
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(dependency.PreparationError, "manifest"):
                dependency.load_contract()
        path.write_bytes(original)
        self.assertFalse(self.destination.parent.exists())

    def test_duplicate_and_nonfinite_identity_json_reject(self):
        receipt = dependency.prepare(self.source, self.destination)
        valid = dependency.canonical_json(receipt).decode().strip()
        documents = [
            (valid[:-1] + ',"schema_version":1}').encode(),
            b'{"schema_version":1,"extra":{"duplicate":1,"duplicate":2}}',
        ]
        documents += [
            ('{"schema_version":1,"extra":' + token + '}').encode()
            for token in ["NaN", "Infinity", "-Infinity", "1e999"]
        ]
        original_read = Path.read_bytes
        for encoded in documents:
            with self.subTest(document=encoded):
                for target in [
                    self.destination / dependency.RECEIPT_NAME,
                    dependency.DEPENDENCY / "manifest.json",
                ]:
                    def read_with_bad_json(path):
                        return encoded if path == target else original_read(path)

                    with patch.object(Path, "read_bytes", read_with_bad_json):
                        with self.assertRaisesRegex(
                            dependency.PreparationError, "duplicate JSON|nonfinite JSON"
                        ):
                            dependency.verify_prepared(self.destination)
        self.assertEqual(dependency.verify_prepared(self.destination), receipt)


if __name__ == "__main__":
    unittest.main()
