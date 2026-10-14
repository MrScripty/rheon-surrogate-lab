"""Numeric admission tests using explicitly synthetic, disposable byte fixtures.

No fixture is a qualified physics capture. Its origin hash is patched only inside
the test, and native outputs are arithmetic witnesses, never scientific evidence.
"""

import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


SPEC = importlib.util.spec_from_file_location("pressure_training_data", Path(__file__).with_name("data.py"))
data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = data
SPEC.loader.exec_module(data)
GRID = np.arange(4096, dtype="<f8").reshape(data.COUNTS)


def numeric(trajectory_index, frame, field):
    if field == "base":
        return np.zeros(data.COUNTS, dtype="<f8") if frame == 0 else numeric(
            trajectory_index, frame - 1, "reference")
    multiplier = (100 * (trajectory_index + 1) + frame + 1) * 1e-5
    return (GRID % 23) * multiplier if field == "reference" else (
        (GRID % 17 - 8) * multiplier
    )


def raw_reader(descriptor):
    pieces = descriptor["path"].split("/")
    trajectory_id = pieces[0]
    index = [e["trajectory_id"] for e in data.reservation_entries()].index(trajectory_id)
    if pieces[1] == "initial-pressure.bin":
        return np.zeros(data.COUNTS, dtype="<f8")
    frame = int(pieces[1].split("-")[1])
    field = "rhs" if pieces[1].endswith("rhs.bin") else "reference"
    return numeric(index, frame, field)


def descriptor(path, values):
    return {"path": path, "sha256": data.digest(values.tobytes()), "bytes": values.nbytes,
            "dtype": "<f8", "shape_xyz": list(data.COUNTS), "layout": "x-fastest",
            "units": "m3/s2" if path.endswith("rhs.bin") else "Pa"}


def synthetic_origin():
    geometries = [[1.0, 1.0, 1.0], [1.2, 1.0, 0.8], [0.8, 1.2, 1.0], [1.0, 0.8, 1.2]]
    trajectories = []
    for index, entry in enumerate(data.reservation_entries()):
        tid = entry["trajectory_id"]
        initial = {"pressure": descriptor(tid + "/initial-pressure.bin",
                                           np.zeros(data.COUNTS, dtype="<f8"))}
        initial.update({name: {"sha256": "a" * 64} for name in ("u", "v", "w", "tracer")})
        frames = []
        for frame in range(data.FRAMES):
            arrays = {
                "rhs": descriptor(f"{tid}/frame-{frame:03d}-rhs.bin", numeric(index, frame, "rhs")),
                "pressure": descriptor(f"{tid}/frame-{frame:03d}-pressure.bin",
                                       numeric(index, frame, "reference")),
                "pre_u": {"sha256": data.digest(f"synthetic-MAC-{index}-{frame}".encode())},
            }
            frames.append({"index": frame, "time_s": (frame + 1) / 60.0, "arrays": arrays})
        trajectories.append({
            "trajectory_id": tid, "reservation_entry": entry,
            "geometry": {"counts_xyz": list(data.COUNTS), "extents_m": geometries[index % 4],
                         "origin_m": [0.0] * 3, "density_kg_m3": 1.25,
                         "boundary": "sealed_zero_normal_mac", "units": "SI"},
            "initial": initial, "frames": frames,
        })
    return {"status": "development_complete", "fixture": False, "preset": "smoke",
            "counts_xyz": list(data.COUNTS),
            "selection": {"split": "development", "trajectory_id": None},
            "provenance": {"lab_commit": data.SOURCE_COMMIT,
                           "native_receipt": {"base_commit": data.NATIVE_BASE}},
            "trajectories": trajectories}


def synthetic_native():
    receipt = {"schema_version": 1, "base_commit": data.NATIVE_BASE, "base_tree": "a" * 40}
    receipt.update({key: "a" * 64 for key in ("patch_sha256", "prepared_tree_sha256",
                    "source_manifest_sha256", "source_files_sha256", "license_sha256")})
    binding = {
        "schema": "rheon-global-pressure-training-build-v1", "prepared_source": "/test-only/native",
        "native_receipt": receipt, "resolved_native_manifest": "/test-only/native/Cargo.toml",
        "metadata_sha256": "b" * 64, "cargo_command": ["test-only-fixture"],
        "source_sha256": {"test-only-fixture": "c" * 64},
        "toolchain": {"cargo": "test-only", "rustc_vV": "test-only", "rustc_path": "test-only",
                      "rustdoc_path": "test-only"}, "configuration": "test-only",
    }
    identity = {"schema": "rheon-global-pressure-training-native-v1", "binding": binding,
                "native_receipt": receipt, "native_threads": 1, "source_sha256": "d" * 64}
    return {"identity": identity, "binary_sha256": "e" * 64}


def write_fixture(origin, path):
    blob = data.canonical(origin)
    with patch.object(data, "ORIGIN_SHA256", data.digest(blob)):
        return data._write_packet(blob, origin, path, {"status": "synthetic_test_only"},
                                  raw_reader, synthetic_native(), "f" * 64,
                                  lambda rhs, base, _geometry, _dt: rhs - 0.5 * base,
                                  lambda: None)


class DataAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pressure-data-test-only-")
        cls.root = Path(cls.temp.name)
        cls.origin = synthetic_origin()
        cls.origin_hash = data.digest(data.canonical(cls.origin))
        cls.packet = cls.root / "valid"
        cls.receipt = write_fixture(cls.origin, cls.packet)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.work = self.root / self._testMethodName
        shutil.copytree(self.packet, self.work)
        self.mock_origin = patch.object(data, "ORIGIN_SHA256", self.origin_hash)
        self.mock_origin.start()
        self.addCleanup(self.mock_origin.stop)

    def load(self, path=None, **kwargs):
        root = path or self.work
        return data.load_data(*(root / name for name in data.SHARDS), root / "splits.json",
                              root / "origin.json", expected_native_residual_origin_sha256=
                              kwargs.get("anchor", self.receipt["native_residual_origin_sha256"]))

    def splits(self):
        return data.strict_json((self.work / "splits.json").read_bytes())

    def save_splits(self, splits):
        (self.work / "splits.json").write_bytes(data.canonical(splits))

    def rewrite_npz(self, name, mutate):
        with np.load(self.work / name, allow_pickle=False) as current:
            values = {field: current[field] for field in data.FIELDS}
        mutate(values)
        with (self.work / name).open("wb") as stream:
            np.savez(stream, **values)
        splits = self.splits()
        declaration = splits["shards"][name]
        blob = (self.work / name).read_bytes()
        declaration.update(bytes=len(blob), sha256=data.digest(blob))
        for field, array in values.items():
            declaration["arrays"][field] = {"shape": list(array.shape), "dtype": array.dtype.str,
                                            "bytes": array.nbytes, "sha256": data.digest(array.tobytes())}
        self.save_splits(splits)
        return values

    def test_lossless_roundtrip_explicit_rows_and_bounded_packet(self):
        dataset = self.load()
        self.assertEqual(dataset.train["rhs"].shape, (288, 16, 16, 16))
        self.assertEqual(dataset.validation["reference"].shape, (96, 16, 16, 16))
        self.assertEqual(dataset.train_rows[144]["row_id"], "smoke-train-100006:frame-000")
        for trajectory in (0, 5, 6, 11):
            for frame in (0, 23):
                row = trajectory * 24 + frame
                for field in ("rhs", "base", "reference"):
                    np.testing.assert_array_equal(dataset.train[field][row], numeric(trajectory, frame, field))
        self.assertEqual((self.work / "origin.json").read_bytes(), data.canonical(self.origin))
        self.assertLess((self.work / "splits.json").stat().st_size, 128 * 1024)
        self.assertTrue(all(p.stat().st_size <= data.FILE_LIMIT for p in self.work.iterdir()))
        self.assertLess(sum(p.stat().st_size for p in self.work.iterdir()), data.INPUT_LIMIT)
        self.assertTrue(dataset.diagnostics["shared_packed_payloads"])

    def test_original_label_drift_cannot_authorize_itself_with_new_hashes(self):
        self.rewrite_npz("validation.npz", lambda a: a["reference"].__setitem__((0, 0, 0, 1), 987.0))
        with self.assertRaisesRegex(data.DataError, "raw_lineage"):
            self.load()

    def test_derived_residual_drift_and_changed_attestation_need_external_authority(self):
        values = self.rewrite_npz("validation.npz", lambda a: a["native_input_residual"].__setitem__((0, 0, 0, 0), 987.0))
        with self.assertRaisesRegex(data.DataError, "native_residual_lineage"):
            self.load()
        splits = self.splits()
        splits["native_residual_origin"]["rows"][288]["sha256"] = data.digest(values["native_input_residual"][0].tobytes())
        self.save_splits(splits)
        with self.assertRaisesRegex(data.DataError, "native_residual_origin_hash"):
            self.load()

    def test_cross_shard_copied_numeric_parent_rejects_original_lineage(self):
        with np.load(self.work / "train-0.npz", allow_pickle=False) as source:
            copied = {field: source[field][:24].copy() for field in data.CELL_FIELDS}
        self.rewrite_npz("validation.npz", lambda a: [a[field].__setitem__(slice(24), values)
                                                       for field, values in copied.items()])
        with self.assertRaisesRegex(data.DataError, "raw_lineage"):
            self.load()

    def test_original_whole_history_duplicate_rejects_across_splits(self):
        origin = copy.deepcopy(self.origin)
        origin["trajectories"][12]["frames"] = copy.deepcopy(origin["trajectories"][0]["frames"])
        with self.assertRaisesRegex(data.DataError, "lineage_duplicate: original whole history"):
            write_fixture(origin, self.root / "duplicate-original")

    def test_packed_history_duplicate_is_id_independent_even_if_MAC_origin_differs(self):
        origin = copy.deepcopy(self.origin)
        frames = copy.deepcopy(origin["trajectories"][0]["frames"])
        for frame in frames:
            frame["arrays"]["pre_u"]["sha256"] = "9" * 64
        origin["trajectories"][12]["frames"] = frames
        with self.assertRaisesRegex(data.DataError, "lineage_duplicate: packed numeric whole history"):
            write_fixture(origin, self.root / "duplicate-packed")

    def test_crop_relabel_final_and_duplicate_row_metadata_fail_before_numpy_load(self):
        original = self.splits()
        variants = []
        cropped = copy.deepcopy(original)
        cropped["shards"]["train-1.npz"]["rows"].pop()
        variants.append(cropped)
        relabelled = copy.deepcopy(original)
        relabelled["shards"]["validation.npz"]["trajectory_ids"][0] = "smoke-test-100100"
        variants.append(relabelled)
        duplicate = copy.deepcopy(original)
        duplicate["shards"]["train-0.npz"]["rows"][1] = duplicate["shards"]["train-0.npz"]["rows"][0]
        variants.append(duplicate)
        for mutated in variants:
            self.save_splits(mutated)
            with patch.object(data.np, "load", side_effect=AssertionError("allocated")):
                # First shard may be valid; metadata admission must be global.
                with self.assertRaises(data.DataError):
                    self.load()

    def test_closed_typed_split_and_native_metadata(self):
        original = self.splits()
        for key, value in (("schema_version", True), ("schema_version", 1.0),
                           ("preset", "full"), ("unexpected", 1)):
            mutated = copy.deepcopy(original)
            mutated[key] = value
            self.save_splits(mutated)
            with self.assertRaises(data.DataError):
                self.load()
        mutated = copy.deepcopy(original)
        mutated["native_residual_origin"]["native_identity"]["native_threads"] = True
        self.save_splits(mutated)
        anchor = data.digest(data.canonical(mutated["native_residual_origin"]))
        with self.assertRaisesRegex(data.DataError, "native_residual_origin_schema"):
            self.load(anchor=anchor)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        original = (self.work / "splits.json").read_bytes()
        for malformed in (original[:-1] + b',"schema_version":1}',
                          b'{"schema_version":NaN}', b'{"schema_version":1e999}'):
            (self.work / "splits.json").write_bytes(malformed)
            with self.assertRaisesRegex(data.DataError, "json_duplicate|json_nonfinite"):
                self.load()

    def test_origin_hash_rejects_final_and_unreserved_parent_before_allocation(self):
        for value in ("full", "test"):
            origin = copy.deepcopy(self.origin)
            origin["preset"] = value
            (self.work / "origin.json").write_bytes(data.canonical(origin))
            with patch.object(data.np, "load", side_effect=AssertionError("allocated")):
                with self.assertRaisesRegex(data.DataError, "origin_hash"):
                    self.load()

    def test_file_and_aggregate_bounds_are_checked_before_allocation(self):
        with (self.work / "validation.npz").open("wb") as stream:
            stream.truncate(data.FILE_LIMIT + 1)
        with patch.object(data.np, "load", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(data.DataError, "file_size"):
                self.load()
        shutil.copyfile(self.packet / "validation.npz", self.work / "validation.npz")
        with patch.object(data, "INPUT_LIMIT", 1), patch.object(data.np, "load", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(data.DataError, "input_size"):
                self.load()

    def test_symlink_and_nonregular_inputs_reject(self):
        path = self.work / "train-0.npz"
        path.unlink()
        path.symlink_to(self.packet / "train-0.npz")
        with self.assertRaisesRegex(data.DataError, "unsafe_path"):
            self.load()
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(data.DataError, "unsafe_path"):
            self.load()

    def test_nonfinite_object_big_endian_and_wrong_shape_arrays_reject(self):
        for kind in ("nonfinite", "object", "endian", "shape"):
            def mutate(values):
                if kind == "nonfinite":
                    values["rhs"][0, 0, 0, 1] = np.nan
                elif kind == "object":
                    values["rhs"] = np.array(["unsafe"], dtype=object)
                elif kind == "endian":
                    values["rhs"] = values["rhs"].astype(">f8")
                else:
                    values["rhs"] = values["rhs"][:1]
            self.rewrite_npz("train-0.npz", mutate)
            with self.assertRaisesRegex(data.DataError, "array_nonfinite|array_schema"):
                self.load()
            shutil.copyfile(self.packet / "train-0.npz", self.work / "train-0.npz")
            self.save_splits(data.strict_json((self.packet / "splits.json").read_bytes()))

    def test_compression_and_npy_header_bomb_reject_before_array_allocation(self):
        original = (self.work / "train-0.npz").read_bytes()
        for bomb in (False, True):
            stream = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED if bomb else zipfile.ZIP_DEFLATED
            ) as target:
                for name in source.namelist():
                    blob = source.read(name)
                    if bomb and name == "rhs.npy":
                        length = int.from_bytes(blob[8:10], "little")
                        header = blob[10:10 + length].replace(b"(144, 16, 16, 16)", b"(999999999, 16, 16, 16)")
                        blob = blob[:8] + len(header).to_bytes(2, "little") + header + blob[10 + length:]
                    target.writestr(name, blob)
            blob = stream.getvalue()
            (self.work / "train-0.npz").write_bytes(blob)
            splits = self.splits()
            splits["shards"]["train-0.npz"].update(bytes=len(blob), sha256=data.digest(blob))
            self.save_splits(splits)
            with patch.object(data.np, "load", side_effect=AssertionError("allocated")):
                with self.assertRaisesRegex(data.DataError, "npz_compression|npy_header"):
                    self.load()

    def test_array_hash_detects_corruption_even_with_valid_outer_container_hash(self):
        original = (self.work / "train-0.npz").read_bytes()
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(stream, "w") as target:
            for name in source.namelist():
                blob = bytearray(source.read(name))
                if name == "rhs.npy":
                    blob[-8] ^= 1
                target.writestr(name, blob)
        blob = stream.getvalue()
        (self.work / "train-0.npz").write_bytes(blob)
        splits = self.splits()
        splits["shards"]["train-0.npz"].update(bytes=len(blob), sha256=data.digest(blob))
        self.save_splits(splits)
        with self.assertRaisesRegex(data.DataError, "array_hash"):
            self.load()

    def test_pack_fresh_verification_failure_creates_no_output(self):
        capture = self.work / "capture"
        capture.mkdir()
        (capture / "manifest.json").write_bytes(data.canonical(self.origin))
        output = self.work / "must-not-exist"
        with patch.object(data, "fresh_verify", side_effect=data.DataError("capture_verification", "refused")) as verifier:
            with self.assertRaisesRegex(data.DataError, "capture_verification"):
                data.pack(capture, output, self.work / "unused-native")
            verifier.assert_called_once_with(capture)
        self.assertFalse(output.exists())

    def test_input_hashes_and_loaded_arrays_share_one_read_snapshot(self):
        original_read = data.read_bounded
        counts = {}

        def once(path, limit=data.FILE_LIMIT):
            path = Path(path)
            counts[path] = counts.get(path, 0) + 1
            self.assertEqual(counts[path], 1, "must not identify loaded arrays by a later reread")
            return original_read(path, limit)

        with patch.object(data, "read_bounded", once):
            loaded = self.load()
        self.assertEqual(len(counts), 5)
        self.assertEqual(loaded.input_hashes["train-0.npz"],
                         data.digest((self.work / "train-0.npz").read_bytes()))

    def test_writer_existing_directory_preserved_and_native_drift_cleans_new_output(self):
        existing = self.work / "foreign"
        existing.mkdir()
        (existing / "preserve").write_text("foreign")
        with patch.object(data.np, "empty", side_effect=AssertionError("allocated")):
            with self.assertRaises(FileExistsError):
                write_fixture(self.origin, existing)
        self.assertEqual((existing / "preserve").read_text(), "foreign")
        output = self.work / "native-drift"

        def fail_native():
            raise data.DataError("native_residual_identity", "test-only simulated drift")

        with self.assertRaisesRegex(data.DataError, "native_residual_identity"):
            data._write_packet(data.canonical(self.origin), self.origin, output,
                               {"status": "synthetic_test_only"}, raw_reader,
                               synthetic_native(), "f" * 64,
                               lambda rhs, base, _geometry, _dt: rhs - 0.5 * base, fail_native)
        self.assertFalse(output.exists())

    def test_native_residual_authority_has_no_self_declared_default(self):
        with self.assertRaises(TypeError):
            data.load_data(*(self.work / name for name in data.SHARDS),
                           self.work / "splits.json", self.work / "origin.json")


if __name__ == "__main__":
    unittest.main()
