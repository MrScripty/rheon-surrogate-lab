"""Focused provenance and root-selection regressions; no scientific execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import dependency_config as config
from config import freeze
from dependency_training import capture_closure, verify_closure


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "independent-source"
        self.source.mkdir()
        for args in (("init", "-q"), ("add", ".")):
            if args[0] == "add":
                (self.source / "numerics.txt").write_text("original numerical policy\n")
            self.git(*args)
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "fixture")
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.directory = self.root / "selection"
        self.directory.mkdir()
        self.path = self.directory / "dependencies.json"
        self.asset = self.root / "build.bin"
        self.asset.write_bytes(b"measured build")
        self.value = {"schema": config.SCHEMA,
                      "sources": {"rheon": {"root": "../independent-source", "revision": self.revision}},
                      "artifacts": {"probe": {"path": "../build.bin", "sha256": config.digest(self.asset)}},
                      "runtimes": {}, "paths": {"runs": "../outputs"}}
        self.save()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def save(self):
        self.path.write_text(json.dumps(self.value))

    def test_paths_belong_to_selected_configuration_not_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir("/tmp")
            selected = config.load(self.path)
            self.assertEqual(selected.source("rheon"), self.source)
            self.assertEqual(selected.artifact("probe"), self.asset)
            self.assertEqual(selected.path_for("runs"), self.root / "outputs")
        finally:
            os.chdir(previous)

    def test_wrong_source_revision_is_rejected(self):
        self.value["sources"]["rheon"]["revision"] = "0" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            config.load(self.path).source("rheon")

    def test_source_root_preserves_base_identity_without_whole_checkout_gate(self):
        identity = config.load(self.path).source_witness("rheon")
        self.value["sources"]["rheon"]["content_sha256"] = identity["content_sha256"]
        self.save()
        (self.source / "notes.md").write_text("unrelated workflow documentation\n")
        selected = config.load(self.path)
        self.assertEqual(selected.source_root("rheon"), self.source)
        with self.assertRaisesRegex(ValueError, "source content mismatch"):
            selected.source("rheon")
        self.value["sources"]["rheon"]["revision"] = "0" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            config.load(self.path).source_root("rheon")

    def test_changed_build_bytes_are_rejected(self):
        self.asset.write_bytes(b"another build")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            config.load(self.path).artifact("probe")

    def test_build_rejects_changed_source_even_if_binary_hash_matches(self):
        identity = config.load(self.path).source_witness("rheon")
        self.value["artifacts"]["probe"].update(source="rheon", source_content_sha256=identity["content_sha256"])
        self.save()
        (self.source / "numerics.txt").write_text("changed numerical policy\n")
        with self.assertRaisesRegex(ValueError, "source content mismatch"):
            config.load(self.path).artifact("probe")

    def test_witness_captures_modified_new_and_deleted_source(self):
        before = config.load(self.path).source_witness("rheon")
        (self.source / "numerics.txt").unlink()
        (self.source / "new.py").write_text("changed = True\n")
        after = config.load(self.path).source_witness("rheon")
        self.assertNotEqual(before["content_sha256"], after["content_sha256"])
        self.assertIn({"path": "numerics.txt", "deleted": True}, after["files"])
        self.assertEqual({r["path"] for r in after["files"]}, {"numerics.txt", "new.py"})

    def test_duplicate_fields_and_historical_configuration_are_rejected(self):
        self.path.write_text('{"schema":"one","schema":"two"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            config.load(self.path)
        self.path.write_text('{"schema":"rheon-local-pressure-host-v1"}')
        with self.assertRaises(ValueError):
            config.load(self.path)

    def test_runtime_manifest_hash_is_checked_before_runtime_owner(self):
        self.value["runtimes"]["cpu"] = {"profile": "managed_torch_python_v1", "manifest": "../build.bin", "manifest_sha256": "0" * 64, "owner_sha256": "0" * 64}
        self.save()
        with self.assertRaisesRegex(ValueError, "runtime manifest hash mismatch"):
            config.load(self.path).runtime("cpu")

    def test_template_freezes_new_commit_without_editing_its_source(self):
        template = self.directory / "template.json"
        value = {**self.value, "schema": "rheon-dependency-template-v1", "rust_toolchain": "1.92.0",
                 "sources": {"rheon": {"root": "../independent-source"}}}
        template.write_text(json.dumps(value))
        original_template = template.read_bytes()
        cargo = self.root / "cargo"
        cargo.write_text("#!/bin/sh\necho 'cargo 1.92.0 (fixture)'\n")
        cargo.chmod(0o755)
        first = self.root / "attempt-1.json"
        freeze(template, first, cargo)
        (self.source / "numerics.txt").write_text("new accepted implementation\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "second")
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            config.load(first).source("rheon")
        second = self.root / "attempt-2.json"
        freeze(template, second, cargo)
        self.assertEqual(config.load(second).source("rheon"), self.source)
        self.assertEqual(template.read_bytes(), original_template)
        with self.assertRaisesRegex(ValueError, "fresh attempt"):
            freeze(template, second, cargo)

    def test_frozen_selection_can_live_in_ignored_runs_without_self_hash(self):
        (self.source / ".gitignore").write_text("/runs/\n")
        template = self.directory / "template.json"
        template.write_text(json.dumps({**self.value, "schema": "rheon-dependency-template-v1",
            "rust_toolchain": "1.92.0", "sources": {"rheon": {"root": "../independent-source"}}}))
        cargo = self.root / "cargo"
        cargo.write_text("#!/bin/sh\necho 'cargo 1.92.0 (fixture)'\n")
        cargo.chmod(0o755)
        output = self.source / "runs/selection.json"
        freeze(template, output, cargo)
        self.assertEqual(config.load(output).source("rheon"), self.source)
        with self.assertRaisesRegex(ValueError, "outside measured source"):
            freeze(template, self.source / "unignored-selection.json", cargo)

    def test_training_capture_checks_actual_inputs_without_unrelated_docs(self):
        trainer = self.root / "train.py"
        model = self.root / "model.py"
        trainer.write_text("actual_update = True\n")
        model.write_text("unchanged_loss = True\n")
        closure = capture_closure([trainer, model])
        (self.root / "independent-A11-doc.md").write_text("separately owned workflow\n")
        verify_closure(closure)
        trainer.write_text("actual_update = False\n")
        with self.assertRaisesRegex(ValueError, "captured training code/input closure changed"):
            verify_closure(closure)


if __name__ == "__main__":
    unittest.main()
