"""Experiment policy and measured identity, independent of host paths and models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
MIB = 1024 ** 2
SCHEMA = "rheon-local-pressure-experiment-v1"


class ContractError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(condition, code, detail):
    if not condition:
        raise ContractError(code, detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_key", key)
            result[key] = value
        return result

    def number(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite", text)
        return value

    def constant(text):
        raise ContractError("nonfinite", text)

    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def read(path, limit=32 * MIB):
    path = Path(path).absolute()
    require(path == path.resolve() and stat.S_ISREG(path.lstat().st_mode), "unsafe_path", str(path))
    require(0 < path.stat().st_size <= limit, "file_size", str(path))
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit, "file_size", str(path))
    return raw


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def fresh(path):
    path = Path(path).absolute()
    require(path == path.resolve() and not path.exists(), "output_exists", str(path))
    path.mkdir(parents=True, mode=0o700)
    return path


def load_module(name, path):
    """Load one measured source; Python's cache never replaces selected file bytes."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(read(path, MIB), str(path), "exec"), module.__dict__)
    return module


def experiment(value):
    fields = {"schema", "id", "geometries_extents_m", "trajectories", "training", "resources"}
    require(type(value) is dict and set(value) == fields and value["schema"] == SCHEMA, "config_schema", "closed experiment v1")
    require(type(value["id"]) is str and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value["id"]), "experiment_id", "safe experiment name")
    geometries = value["geometries_extents_m"]
    require(type(geometries) is dict and 1 <= len(geometries) <= 16, "geometry", "bounded geometry map")
    for name, extents in geometries.items():
        require(re.fullmatch(r"g[0-9]+", name) and type(extents) is list and len(extents) == 3
                and all(type(x) in (float, int) and math.isfinite(x) and 0 < x <= 100 for x in extents), "geometry", name)
    entries = value["trajectories"]
    require(type(entries) is list and 3 <= len(entries) <= 16, "population", "3–16 whole development parents")
    ids, seeds, counts = set(), set(), {"train": 0, "validation": 0}
    for entry in entries:
        require(type(entry) is dict and set(entry) == {"trajectory_id", "seed", "split", "geometry", "forcing_regime", "amplitude_m_s2", "dt_s", "frames"}, "trajectory_schema", "closed reservation")
        tid = entry["trajectory_id"]
        require(type(tid) is str and re.fullmatch(r"[a-z][a-z0-9-]{0,95}", tid) and tid not in ids, "trajectory_id", "unique safe parent")
        require(entry["split"] in counts, "development_only", "train/validation only; final suites sealed")
        require(type(entry["seed"]) is int and 0 <= entry["seed"] < 2**63 and entry["seed"] not in seeds, "seed", "unique whole-parent seed")
        require(entry["geometry"] in geometries and entry["forcing_regime"] in ("steady", "moving"), "scenario", tid)
        require(type(entry["frames"]) is int and entry["frames"] == 24, "frames", "complete fixed 24-frame histories")
        for key in ("dt_s", "amplitude_m_s2"):
            require(type(entry[key]) in (int, float) and math.isfinite(entry[key]) and 0 < entry[key] <= (1 / 30 if key == "dt_s" else 1), "physics_range", key)
        ids.add(tid); seeds.add(entry["seed"]); counts[entry["split"]] += 1
    require(counts["train"] >= 2 and counts["validation"] >= 1, "split_coverage", "two train and one validation parent minimum")
    captured_reservation = HERE / "historical-reservation.json"
    historical = decode(read(captured_reservation if captured_reservation.exists() else LAB / "docs/plans/global-pressure-correction-v1/reservation.json", MIB))
    reserved = {e["seed"] for preset in historical["presets"].values() for e in preset["trajectories"]}
    require(not seeds & reserved, "historical_seed_overlap", "use new seeds; historical/final reservations remain sealed")
    training = value["training"]
    require(type(training) is dict and set(training) == {"model", "seed", "updates", "batch_size", "learning_rate", "checkpoint_interval"}, "training_schema", "closed training profile")
    require(training["model"] == "groupnorm-v1", "model_profile", "unchanged historical GN arithmetic")
    require(type(training["seed"]) is int and 0 <= training["seed"] < 2**63, "training_seed", "integer seed")
    for key, low, high in (("updates", 1, 200), ("batch_size", 1, 4), ("checkpoint_interval", 1, 200)):
        require(type(training[key]) is int and low <= training[key] <= high, "training_range", key)
    require(training["updates"] % training["checkpoint_interval"] == 0, "checkpoint_interval", "include final update")
    require(type(training["learning_rate"]) in (int, float) and math.isfinite(training["learning_rate"]) and 0 < training["learning_rate"] <= 1e-3, "learning_rate", "bounded Adam rate")
    resources = value["resources"]
    require(type(resources) is dict and set(resources) == {"wall_seconds", "max_output_bytes"}, "resources", "closed resource policy")
    require(type(resources["wall_seconds"]) is int and 1 <= resources["wall_seconds"] <= 600
            and type(resources["max_output_bytes"]) is int and 0 < resources["max_output_bytes"] <= 32 * MIB, "resources", "existing bounded CPU policy")
    return decode(canonical(value))


def reservation(config):
    return {"geometries_extents_m": config["geometries_extents_m"], "presets": {"smoke": {"trajectories": config["trajectories"]}}}


def source_inventory():
    paths = list(HERE.glob("*.py")) + list((LAB / "probes/local-pressure-v1").glob("*.*"))
    paths += list((LAB / "probes/local-pressure-v1/src").glob("*.rs"))
    paths += list((LAB / "experiments/global-pressure-v1").glob("*.py"))
    for namespace in ("training", "groupnorm-v1"):
        paths += list((LAB / "experiments/global-pressure-v1" / namespace).glob("*.py"))
    return {path.relative_to(LAB).as_posix(): sha(read(path, 2 * MIB)) for path in sorted(paths)}
