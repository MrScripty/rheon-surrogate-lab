"""Portability contracts using synthetic numeric fixtures, never scientific Runs."""
import copy
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from contract import HERE, LAB, ContractError, MIB, canonical, decode, experiment, read, sha
import data
import model_profile
import project
import train
import process
import bootstrap


def config():
    return decode(read(HERE / "configs/development.json", MIB))


def numeric_packet():
    """Literal whole-history byte lineage; no claim of physical conformance."""
    value = config()
    value["trajectories"] = value["trajectories"][:2] + value["trajectories"][-1:]
    value = experiment(value)
    reader = data.primitives()
    trajectories = []
    for parent, entry in enumerate(value["trajectories"]):
        geometry = {"counts_xyz": [16] * 3, "extents_m": value["geometries_extents_m"][entry["geometry"]], "origin_m": [0.0] * 3, "density_kg_m3": 1.25, "boundary": "sealed_zero_normal_mac", "units": "SI"}
        trajectories.append({"trajectory_id": entry["trajectory_id"], "reservation_entry": entry, "geometry": geometry, "initial": {"pressure": {"sha256": sha(bytes(32768)), "path": f"p{parent}/initial"}}, "frames": []})
        for frame in range(24):
            pressure = np.full((16, 16, 16), parent + frame + 1.0, dtype="<f8")
            pressure.flat[0] = 0
            rhs = np.full((16, 16, 16), parent + frame + 0.5, dtype="<f8")
            trajectories[-1]["frames"].append({"time_s": (frame + 1) / 60, "arrays": {"pressure": {"sha256": sha(pressure.tobytes()), "path": f"p{parent}/f{frame}-pressure"}, "rhs": {"sha256": sha(rhs.tobytes()), "path": f"p{parent}/f{frame}-rhs"}}})
    origin = {"schema": "rheon-local-pressure-capture-v1", "status": "development_complete", "fixture": False, "experiment_sha256": sha(canonical(value)), "trajectories": trajectories}
    origin_raw = canonical(origin)
    splits = {"schema": "rheon-local-pressure-packet-v1", "experiment": value, "capture_manifest_sha256": sha(origin_raw), "shards": {}, "native_residual_origin": {"capture_manifest_sha256": sha(origin_raw), "native_build_receipt": {"scope": "test-only literal lineage"}, "rows": []}, "fingerprints": {}}
    blobs = {"origin.json": origin_raw}
    for parent, (name, trajectory) in enumerate(zip(data.NAMES, trajectories)):
        arrays = {field: np.empty(shape, dtype=dtype) for field, (shape, dtype, _) in reader.array_spec(24).items()}
        rows = []
        for frame in range(24):
            arrays["reference"][frame].fill(parent + frame + 1.0)
            arrays["reference"][frame].flat[0] = 0
            arrays["base"][frame].fill(parent + frame if frame else 0)
            arrays["base"][frame].flat[0] = 0
            arrays["rhs"][frame].fill(parent + frame + 0.5)
            arrays["native_input_residual"][frame].fill(parent + frame + 0.25)
            geometry = trajectory["geometry"]
            arrays["spacing"][frame] = np.asarray(geometry["extents_m"]) / 16
            arrays["extents"][frame] = geometry["extents_m"]
            arrays["origin"][frame] = 0
            arrays["rho"][frame] = 1.25
            arrays["dt"][frame] = 1 / 60
            arrays["frame"][frame] = frame
            arrays["trajectory"][frame] = parent
            row = reader.original_row(trajectory, frame)
            rows.append(row)
            splits["native_residual_origin"]["rows"].append({"row_id": row["row_id"], "sha256": sha(arrays["native_input_residual"][frame].tobytes())})
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        blobs[name] = stream.getvalue()
        splits["shards"][name] = {"split": "validation" if parent == 2 else "train", "trajectory_ids": [trajectory["trajectory_id"]], "bytes": len(blobs[name]), "sha256": sha(blobs[name]), "arrays": train.inventory(arrays), "rows": rows}
        splits["fingerprints"][name] = [reader.packed_fingerprint(arrays, 0, trajectory)]
    blobs["splits.json"] = canonical(splits)
    return blobs


class ConfigurationTests(unittest.TestCase):
    def test_bootstrap_creates_private_intermediate_directories(self):
        original = os.umask(0o002)
        try:
            bootstrap.scientific_environment()
            with tempfile.TemporaryDirectory() as root:
                target = Path(root) / "packages" / "torch" / "nested"
                target.mkdir(parents=True, mode=0o700)
                for directory in (target, target.parent, target.parent.parent):
                    self.assertEqual(directory.stat().st_mode & 0o077, 0)
        finally:
            os.umask(original)

    def test_two_lineages_need_only_configuration_changes(self):
        first = experiment(config())
        second = copy.deepcopy(first)
        second["id"] = "another-local-cpu"
        for entry in second["trajectories"]:
            entry["seed"] += 100000
            entry["trajectory_id"] += "-new"
        second = experiment(second)
        self.assertNotEqual(sha(canonical(first)), sha(canonical(second)))
        self.assertEqual(first["training"]["model"], second["training"]["model"])

    def test_final_split_rejected(self):
        value = config(); value["trajectories"][-1]["split"] = "test"
        with self.assertRaises(ContractError) as error:
            experiment(value)
        self.assertEqual(error.exception.code, "development_only")

    def test_seed_leakage_rejected_even_with_distinct_ids(self):
        value = config(); value["trajectories"][-1]["seed"] = value["trajectories"][0]["seed"]
        with self.assertRaises(ContractError) as error:
            experiment(value)
        self.assertEqual(error.exception.code, "seed")

    def test_historical_seed_reuse_rejected(self):
        value = config(); value["trajectories"][0]["seed"] = 100000
        with self.assertRaises(ContractError) as error:
            experiment(value)
        self.assertEqual(error.exception.code, "historical_seed_overlap")

    def test_pathlike_parent_rejected(self):
        value = config(); value["trajectories"][0]["trajectory_id"] = "../outside"
        with self.assertRaises(ContractError) as error:
            experiment(value)
        self.assertEqual(error.exception.code, "trajectory_id")

    def test_booleans_are_not_scientific_numbers(self):
        value = config(); value["training"]["updates"] = True
        with self.assertRaises(ContractError) as error:
            experiment(value)
        self.assertEqual(error.exception.code, "training_range")

    def test_full_capture_cli_requires_fixture_before_creating_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "capture"
            result = subprocess.run([sys.executable, str(HERE / "pipeline.py"), "capture", "--config", str(HERE / "configs/development.json"), "--binary", str(Path(root) / "unused"), "--output", str(output)], capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("fixture_required", result.stderr.decode())
            self.assertFalse(output.exists())


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.blobs = numeric_packet()

    def test_whole_history_reader_and_train_only_normalization(self):
        dataset = data.load(blobs=self.blobs)
        self.assertEqual((len(dataset.train_rows), len(dataset.validation_rows)), (48, 24))
        model, _helpers = model_profile.load()
        first = model.fit_normalization(dataset.train, dataset.train_rows)
        dataset.validation["reference"].fill(1e9)
        dataset.validation["native_input_residual"].fill(1e9)
        second = model.fit_normalization(dataset.train, dataset.train_rows)
        self.assertEqual(first, second)
        self.assertEqual(set(first["training_provenance"]["trajectory_ids"]), {r["trajectory_id"] for r in dataset.train_rows})

    def test_parent_cannot_cross_shards(self):
        split = decode(self.blobs["splits.json"])
        split["shards"]["validation.npz"]["trajectory_ids"] = split["shards"]["train-0.npz"]["trajectory_ids"]
        self.blobs["splits.json"] = canonical(split)
        with self.assertRaises(ContractError) as error:
            data.load(blobs=self.blobs)
        self.assertEqual(error.exception.code, "parent_overlap")

    def test_numeric_corruption_rejected(self):
        raw = bytearray(self.blobs["train-0.npz"]); raw[-100] ^= 1
        self.blobs["train-0.npz"] = bytes(raw)
        with self.assertRaises(ValueError) as error:
            data.load(blobs=self.blobs)
        self.assertIn("container_hash", str(error.exception))

    def test_different_native_anchor_rejected_before_numeric_load(self):
        with self.assertRaises(ContractError) as error:
            data.load(blobs=self.blobs, expected_residual="0" * 64)
        self.assertEqual(error.exception.code, "residual_anchor")

    def test_fixture_status_ineligible_for_training(self):
        origin = decode(self.blobs["origin.json"]); origin.update(status="fixture_complete", fixture=True)
        self.blobs["origin.json"] = canonical(origin)
        split = decode(self.blobs["splits.json"]); split["capture_manifest_sha256"] = sha(self.blobs["origin.json"])
        self.blobs["splits.json"] = canonical(split)
        with self.assertRaises(ContractError) as error:
            data.load(blobs=self.blobs)
        self.assertEqual(error.exception.code, "origin_identity")


class TrainingContracts(unittest.TestCase):
    def test_sampling_uses_only_supplied_train_parent_rows(self):
        rows = [{"trajectory_id": "a"}] * 24 + [{"trajectory_id": "b"}] * 24
        sampled = train.order(rows, config())
        self.assertEqual(sampled.shape, (200, 4))
        self.assertTrue(np.all((sampled >= 0) & (sampled < len(rows))))
        self.assertTrue(np.array_equal(sampled, train.order(rows, config())))

    def test_configured_checkpoint_population_controls_restore(self):
        first = config(); second = copy.deepcopy(first)
        second["training"].update(updates=40, checkpoint_interval=10)
        self.assertEqual(len(train.trained_inputs(first)), 17)
        self.assertEqual(len(train.trained_inputs(second)), 11)
        self.assertIn("trained_checkpoint_040_npz", train.trained_inputs(second))
        self.assertNotIn("trained_checkpoint_200_npz", train.trained_inputs(second))

    def test_complete_history_budget_rejects_dense_checkpoint_schedule(self):
        value = config(); value["training"]["checkpoint_interval"] = 1
        with self.assertRaises(ContractError) as error:
            project.output_projection(value, 1500186, 500000)
        self.assertEqual(error.exception.code, "output_projection")

    def test_generic_recipe_parser_accepts_configured_train_and_restore(self):
        custody_source = Path(os.environ["TULDOK_CUSTODY_SOURCE"])
        sys.path.insert(0, str(custody_source.parent))
        from recipe_contract import decode_definition
        for mode in ("train", "restore"):
            definition = project.definition(config(), {name: b"fixture" for name in ("train.py", "config.json", "runtime-manifest.json", "source-witness.json")}, "a" * 64, mode)
            decoded = decode_definition(canonical(definition))
            self.assertEqual(len(definition["outputs"]), 18)
            self.assertEqual(len(definition["inputs"]), 5 if mode == "train" else 22)

    def test_native_evaluation_arithmetic_is_byte_identical(self):
        self.assertEqual(read(LAB / "probes/local-pressure-evaluation-v1/src/main.rs", MIB), read(LAB / "probes/global-pressure-v1-training/src/main.rs", MIB))

    def test_native_capture_changes_only_reservation_selection(self):
        historical = read(LAB / "probes/global-pressure-v1/src/main.rs", MIB).decode()
        current = read(LAB / "probes/local-pressure-v1/src/main.rs", MIB).decode()
        strip = lambda source: re.sub(r"const RESERVATION: &str =.*?;\n", "", source, flags=re.S)
        self.assertEqual(strip(historical), strip(current))


class ResidualProcessTests(unittest.TestCase):
    def test_residual_child_is_bounded_and_completely_drained(self):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root) / "residual-fixture"
            binary.write_text("#!/usr/bin/python3.12\nimport json,sys\nrequest=json.load(sys.stdin)\nprint(json.dumps({'mode':sys.argv[1],'value':request['value']}))\n")
            binary.chmod(0o755)
            result, observed = process.native_call(binary, "--residual", {"value": 3}, time.monotonic() + 13, 10)
            self.assertEqual(result, {"mode": "--residual", "value": 3})
            self.assertEqual(observed["returncode"], 0)
            for field in ("drained", "echild", "group_empty", "streams_closed", "spawn_closed"):
                self.assertTrue(observed[field], field)

    def test_residual_timeout_cannot_exceed_ten_seconds(self):
        with self.assertRaises(process.NativeProcessError) as error:
            process.native_call(Path("/tmp/nonexistent-unused-native"), "--residual", {}, time.monotonic() + 20, 11)
        self.assertEqual(error.exception.code, "invalid_timeout")

    def test_residual_requires_object_before_launch(self):
        with self.assertRaises(process.NativeProcessError) as error:
            process.native_call(Path(sys.executable).resolve(), "--residual", [], time.monotonic() + 13, 10)
        self.assertEqual(error.exception.code, "invalid_request")


if __name__ == "__main__":
    unittest.main()
