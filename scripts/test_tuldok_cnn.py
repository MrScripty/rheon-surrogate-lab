"""Captured-data regressions; synthetic inputs prove no solver quality claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tuldok"))
import cnn
import cnn_project


def fixture(root):
    rng = np.random.default_rng(44)
    arrays = {}
    rows = []
    for split, families, base in (("train", ("white_noise", "polynomial_mix"), 1000),
                                 ("validation", ("localized_blob",), 3000),
                                 ("test", ("oscillatory_modes",), 4000)):
        count = 4 * len(families)
        arrays[split + "_rhs"] = rng.normal(size=(count, 6, 7, 8))
        arrays[split + "_pressure"] = rng.normal(size=(count, 6, 7, 8))
        for index in range(count):
            name = families[index // 4] + "-" + str(base + index)
            rows.append(dict(id=name, family=families[index // 4], seed=base + index,
                             split=split, trajectory=name))
    np.savez(root / "dataset.npz", **arrays)
    value = {"schema": "rheon-cnn-splits-v1", "shape_xyz": [8, 7, 6], "dtype": "<f8",
             "rheon_revision": "34235d33c886aa05b5bdf95d920b863d1fb2ad6f",
             "dataset_sha256": cnn.sha(root / "dataset.npz"), "manifest_sha256": "1" * 64,
             "generator_revision": "2" * 40, "cases": rows,
             "policy": "whole seeds and scenario families; standalone fields; checks excluded"}
    (root / "splits.json").write_text(json.dumps(value))
    return arrays, value


class CapturedCNNContracts(unittest.TestCase):
    def test_packaging_refuses_manifest_drift_at_native_verification_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            arrays, splits = fixture(root)
            cases = []
            offsets = {name: 0 for name in ("train", "validation", "test")}
            for row in splits["cases"]:
                index = offsets[row["split"]]
                offsets[row["split"]] += 1
                fields = {}
                for field in ("rhs", "pressure"):
                    path = root / (row["id"] + "-" + field + ".bin")
                    arrays[row["split"] + "_" + field][index].tofile(path)
                    fields[field] = {"path": path.name, "sha256": cnn.sha(path),
                                     "dtype": "<f8", "shape_xyz": [8, 7, 6]}
                cases.append({"spec": row, "arrays": fields})
            manifest.write_text(json.dumps({"schema": "rheon-pressure-dataset-v1", "per_family": 4,
                                           "experiment_commit": "2" * 40,
                                           "rheon_commit": splits["rheon_revision"], "cases": cases}))
            reply = json.dumps({"status": "verified", "cases": 18,
                                "tool": {"source_dirty": False, "rheon_commit": splits["rheon_revision"]}})
            # The controlled native reply proves packaging custody only. A
            # separate actual native regeneration qualifies real data/physics.
            with patch.object(cnn_project.subprocess, "check_output", return_value=reply):
                cnn_project.package(root, root / "valid-package")
            self.assertTrue((root / "valid-package/dataset.npz").is_file())
            def verify(_command, **_kwargs):
                manifest.write_text('{"schema":"changed"}')
                return reply
            with patch.object(cnn_project.subprocess, "check_output", side_effect=verify):
                with self.assertRaisesRegex(ValueError, "manifest changed during native verification"):
                    cnn_project.package(root, root / "package")
            self.assertFalse((root / "package").exists())

    def test_training_reader_does_not_open_test_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays, value = fixture(root)
            arrays["test_rhs"] = np.zeros((4, 6, 7, 8), dtype=np.float32)
            np.savez(root / "dataset.npz", **arrays)
            value["dataset_sha256"] = cnn.sha(root / "dataset.npz")
            (root / "splits.json").write_text(json.dumps(value))
            cases, _ = cnn.data(root / "dataset.npz", root / "splits.json", False)
            self.assertEqual(len(cases), 12)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                cnn.data(root / "dataset.npz", root / "splits.json", True)

    def test_family_leakage_checkpoint_shape_and_source_drift_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, splits = fixture(root)
            splits["cases"][0]["family"] = "oscillatory_modes"
            (root / "splits.json").write_text(json.dumps(splits))
            with self.assertRaisesRegex(ValueError, "family custody"):
                cnn.data(root / "dataset.npz", root / "splits.json", False)
            np.savez(root / "weights.npz", kernel=np.zeros((108, 15)), bias=np.zeros(16), output=np.zeros(16))
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                cnn.weights(root / "weights.npz")
            self.assertEqual(cnn_project.MODEL_HASH, cnn.sha(ROOT / "scripts/train_cpu.py"))

    @unittest.skipUnless(os.environ.get("TULDOK_CONTRACT_ROOT"), "explicit generic reader root needed")
    def test_both_diagnostic_recipe_shapes_use_existing_generic_contract(self):
        for command in ("train", "restore"):
            definition = cnn_project.definition(command, {"cnn.py": b"fixture", "train_cpu.py": b"fixture"}, "0" * 64)
            code = "import json,sys,recipe_contract; recipe_contract.decode_definition(sys.argv[1].encode()); print('accepted')"
            result = subprocess.run([sys.executable, "-c", code, json.dumps(definition)],
                                    env={**os.environ, "PYTHONPATH": os.environ["TULDOK_CONTRACT_ROOT"]},
                                    text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.strip(), "accepted")


if __name__ == "__main__":
    unittest.main()
