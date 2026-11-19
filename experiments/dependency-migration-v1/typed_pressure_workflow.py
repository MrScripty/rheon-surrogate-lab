"""Prospective pressure Model/Study intent; no admission, optimizer or promotion.

Declarations bind the actual bounded pressure trainer and train-only packet.
Current owners may retain drafts, but scientific Trial preparation remains an
explicit unsupported result until the pressure evidence contract is qualified.
"""
import argparse
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

from dependency_config import canonical, decode, digest, load, require

PROFILE = "managed_torch_python_v1"
TASK_ID = "rheon.pressure:sealed-cubic-correction"
TASK_VERSION = "1.0.0"


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def rooted(root, name):
    path = Path(name)
    require(not path.is_absolute() and not any(part in ("..", ".") for part in path.parts),
            "project-relative material path required")
    path = root / path
    require(path.resolve(strict=True).is_relative_to(root) and not path.is_symlink(),
            "material must remain beneath selected root")
    return path


def consumed_validation_probe(root, control_project, output):
    """Copy one previously evaluated native frame, without importing NumPy/Torch."""
    root, control = Path(root).resolve(strict=True), Path(control_project).resolve(strict=True)
    output = Path(output).resolve()
    require(not output.exists(), "fresh validation probe output required")
    witness = decode((control / "source-witness.json").read_bytes())
    project = decode((control / "project.json").read_bytes())
    refs = {ref["id"]: ref for ref in project["data"]}
    raw_inputs = {}
    for slot in ("dataset", "dataset_extra", "splits"):
        path = rooted(root, refs[slot]["path"])
        require(digest(path) == witness["inputs"][slot]["sha256"]
                and path.stat().st_size == witness["inputs"][slot]["bytes"], "original input bytes changed")
        raw_inputs[slot] = path.read_bytes()
    packet = decode(raw_inputs["splits"])
    index, parent = next((i, p) for i, p in enumerate(packet["parents"]) if p["entry"]["split"] == "validation")
    frame = parent["frames"][0]
    # Actual retained completed training evidence proves this frame was consumed.
    original = control.parent / "managed-full"
    train_receipt_path = original / "stage-00/artifacts/receipt.json"
    receipt = decode(train_receipt_path.read_bytes())
    require(receipt["status"] == "complete" and receipt["mode"] == "train"
            and receipt["outputs"]["splits.json"]["sha256"] == hashlib.sha256(raw_inputs["splits"]).hexdigest()
            and any(case["split"] == "validation" and case["trajectory_id"] == parent["entry"]["trajectory_id"]
                    and case["native_frame"] == frame["index"] for case in receipt["cases"]),
            "selected validation frame lacks original consumption evidence")
    bookend = decode((original / "receipt.json").read_bytes())
    train_run = next(stage for stage in bookend["stages"] if stage["name"] == "train")
    require(train_run["state"] == "completed" and train_run["custody_phase"] == "drained"
            and any(item["name"] == "receipt.json" and item["sha256"] == digest(train_receipt_path)
                    for item in train_run["artifacts"]), "original training receipt custody differs")
    shape = list(reversed(parent["counts_xyz"]))
    members, hashes = {}, {}
    for semantic, source in (("initial_residual", "native_input_residual"),
                             ("previous_pressure", "base"), ("native_pressure", "reference")):
        key = f"p{index:04d}_{source}.npy"
        candidates = []
        for slot in ("dataset", "dataset_extra"):
            with zipfile.ZipFile(io.BytesIO(raw_inputs[slot])) as archive:
                if key in archive.namelist():
                    candidates.append(archive.read(key))
        require(len(candidates) == 1, "one exact original array required")
        raw = candidates[0]
        require(raw[:8] == b"\x93NUMPY\x01\x00", "original NPY1 array required")
        length = struct.unpack_from("<H", raw, 8)[0]
        header = ast.literal_eval(raw[10:10 + length].decode("ascii"))
        require(header["descr"] == "<f8" and header["fortran_order"] is False
                and list(header["shape"]) == [len(parent["frames"]), *shape], "exact float64 frame layout required")
        payload = raw[10 + length:10 + length + math.prod(shape) * 8]
        require(hashlib.sha256(payload).hexdigest() == frame["sha256"][source], "original frame payload differs")
        header = repr({"descr": "<f8", "fortran_order": False, "shape": tuple(shape)}).encode("ascii")
        header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
        members[semantic + ".npy"] = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload
        hashes[semantic] = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_STORED) as archive:
        for name, raw in members.items():
            archive.writestr(zipfile.ZipInfo(name), raw)
    h = parent["cell_size_m"]
    return {"path": str(output), "sha256": digest(output), "population": "validation",
            "consumption": "already consumed by original control training; no untouched population opened",
            "member_id": parent["entry"]["trajectory_id"] + "-frame-" + str(frame["index"]),
            "original_training_run_id": train_run["run_id"], "original_training_receipt_sha256": digest(train_receipt_path),
            "original_splits_sha256": witness["inputs"]["splits"]["sha256"], "array_sha256": hashes,
            "geometry": {"shape_xyz": parent["counts_xyz"], "origin_xyz": [0, 0, 0], "spacing_xyz": [h, h, h],
                         "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "units": "m"},
            "density_kg_m3": parent["density_kg_m3"], "dt_s": parent["entry"]["dt_s"],
            "prior": "exact recorded solved base; native frame " + str(frame["index"]),
            "scientific_acceptance": "not_asserted", "optimizer_updates_executed": 0}


def declarations(root, manifest_path):
    """Read existing current material, never import its Python/numerical sources."""
    root = Path(root).resolve(strict=True)
    path = rooted(root, manifest_path)
    project = decode(path.read_bytes())
    require(project["schema_version"] == 2, "current Project snapshot required")
    material = decode((path.parent / "migration-material.json").read_bytes())
    require(material["schema"] == "rheon-migration-training-material-v1"
            and material["root"] == str(root) and material["manifest_path"] == manifest_path
            and material["project_sha256"] == digest(path), "exact current training material required")
    require(material["diagnostic_population"] == "train_only"
            and material["heldout_parent_count"] == 0 and material["heldout_inputs_imported"] is False,
            "train-only diagnostic; no heldout input is admitted")
    recipes = {}
    for mode, device in (("train", "cuda"), ("restore", "cpu")):
        ref = next(item for item in project["recipes"] if item["id"] == mode)
        recipe_path = rooted(root, ref["path"])
        require(digest(recipe_path) == ref["sha256"], "declared recipe changed")
        recipe = decode(recipe_path.read_bytes())
        require(recipe["runtime"] == PROFILE and recipe["runtime_device"] == device
                and recipe["runtime_snapshot_sha256"] == material["runtime"]["manifest_sha256"],
                "actual managed CUDA trainer and fresh CPU restore required")
        for source in recipe["sources"]:
            require(digest(rooted(root, source["path"])) == source["sha256"], "captured pressure source changed")
        recipes[mode] = recipe
    witness_path = path.parent / "source-witness.json"
    require(digest(witness_path) == material["current_source_witness_sha256"], "source witness changed")
    witness = decode(witness_path.read_bytes())
    config = decode((path.parent / "config.json").read_bytes())
    require(config["training"]["updates"] == 1 and config["training"]["batch_size"] == 1
            and config["training"]["checkpoints"] == [0, 1], "one-update prospective budget required")
    architecture = witness["architecture"]
    require(architecture["input_channels"] == 1 and architecture["dtype"] == "float32"
            and architecture["pressure_dtype"] == "float64"
            and architecture["gauge"] == "subtract_first_correction_cell", "actual pressure architecture required")
    refs = {item["id"]: item for item in project["data"]}
    for ref in refs.values():
        require(digest(rooted(root, ref["path"])) == ref["sha256"], "declared train-only input changed")
    splits = decode(rooted(root, refs["splits"]["path"]).read_bytes())
    require(splits["parents"] and all(p["entry"]["split"] == "train" for p in splits["parents"]),
            "diagnostic packet contains a nontraining parent")
    parent = splits["parents"][0]
    require(parent["density_kg_m3"] == 1000 and parent["entry"]["dt_s"] == 1 / 60,
            "frozen water density and timestep required")
    probe = {"population": "train", "purpose": "fixed same-request diagnostic; no validation claim",
             "trajectory_id": parent["entry"]["trajectory_id"], "frame": parent["frames"][0]["index"],
             "array_sha256": parent["frames"][0]["sha256"], "counts_xyz": parent["counts_xyz"],
             "origin_xyz_m": [0, 0, 0], "cell_size_m": parent["cell_size_m"],
             "density_kg_m3": parent["density_kg_m3"], "dt_s": parent["entry"]["dt_s"]}
    ports = []
    for name, direction, axes, units in (
            ("initial_residual", "input", ["z", "y", "x"], "m3/s2"),
            ("previous_pressure", "input", ["z", "y", "x"], "Pa"),
            ("native_pressure", "target", ["z", "y", "x"], "Pa"),
            ("full_pressure", "output", ["z", "y", "x"], "Pa")):
        ports.append({"id": name, "direction": direction,
                      "contract": {"id": "rheon.pressure:" + name, "version": TASK_VERSION},
                      "encoding": "npy", "axes": axes, "units": units})
    task = {"format": "tuldok_contribution_v1", "id": TASK_ID, "version": TASK_VERSION,
            "kind": "task", "label": "Sealed cubic water pressure correction", "requires": [], "ports": ports,
            "stages": [{"id": "train", "version": TASK_VERSION,
                        "inputs": [p["id"] for p in ports if p["direction"] != "output"],
                        "outputs": ["full_pressure"], "clock": "optimizer_update", "failure_policy": "fail"}]}
    preprocessing = {"kind": "rheon_physical_residual_v1", "network_dtype": "float32",
                     "physical_dtype": "float64", "channel_order": ["dimensionless_laplacian_rhs"],
                     "q": "(rho/h)*initial_residual", "scaling": "q / train-only q RMS",
                     "normalization_sha256": material["normalization_sha256"], "fit_population": "train",
                     "zero_residual": "return supplied previous pressure exactly without invoking network",
                     "scalar_context": {"density": {"units": "kg/m3", "value": 1000},
                                        "dt": {"units": "s", "value": 1 / 60},
                                        "cell_size": {"units": "m", "geometry": "one positive cubic h per sample"}}}
    output = {"kind": "full_pressure", "units": "Pa", "dtype": "float64", "axes": ["z", "y", "x"],
              "layout": "x_fast", "centering": "cell", "gauge": "cell_zero",
              "reconstruction": "float32 raw times selected train-only pressure RMS; gauge correction; add supplied base exactly once",
              "correction": config["correction"], "output_scale": config["output_scale"]}
    context = {"geometry": "sealed full uniform-density cubic water", "origin_xyz_m": [0, 0, 0],
               "domain_extent": "counts_xyz * scalar cell_size_m", "density_kg_m3": 1000,
               "viscosity_pa_s": 0.001, "dt_s": 1 / 60, "minimum_native_steps": 15,
               "prior_pressure": "qualified solved pressure at immediately preceding interval",
               "neural_compute": "float32 CUDA train; float32 CPU restore with CUDA hidden",
               "model_promotion": False, "historical_parent_run": None}
    metric = {"id": "rheon.pressure:same-rhs-mean-free-relative-l2", "population": "train",
              "role": "diagnostic", "probe_sha256": sha(probe), "units": "dimensionless",
              "comparison": "native solution of identical supplied pressure request/RHS; subtract difference mean",
              "trajectory_drift": "separate metric; never substituted for same-request pressure error",
              "acceptance": "none; migration physical/stopping gates remain separately owned"}
    definitions = {"task": task}
    for role, directions in (("model", ("input", "output")),
                             ("trainer", ("input", "target", "output")),
                             ("evaluator", ("target", "output"))):
        value = {"format": "tuldok_contribution_v1", "id": "rheon.pressure:" + role,
                 "version": TASK_VERSION, "kind": role, "label": "Pressure " + role + " declaration",
                 "requires": [], "ports": [copy.deepcopy(p) for p in ports if p["direction"] in directions]}
        prefix = Path(recipes["train"]["entrypoint"]).parent
        implementation = {"model": (str(prefix / "model.py"), "PressureCorrectionNet"),
                          "trainer": (str(Path(__file__).with_name("typed_pressure_evidence.py").relative_to(root)), "main"),
                          "evaluator": (str(Path(__file__).with_name("typed_pressure_evidence.py").relative_to(root)), "mean_free_relative_l2")}
        source, entrypoint = implementation[role]
        value["implementation"] = {"source": source, "entrypoint": entrypoint}
        if role == "trainer":
            value["stages"] = copy.deepcopy(task["stages"])
            value["parameters"] = {"type": "object", "properties": {}, "additionalProperties": False}
            value["defaults"] = {}
        definitions[role] = value
    definitions["reader"] = {"format": "tuldok_contribution_v1", "id": "rheon.pressure:reader",
        "version": TASK_VERSION, "kind": "reader", "label": "Original captured pressure packet reader", "requires": [],
        "implementation": {"source": str(Path(recipes["train"]["entrypoint"]).parent / "data.py"), "entrypoint": "load"}}
    definitions["view"] = {"format": "tuldok_contribution_v1", "id": "rheon.pressure:view",
        "version": TASK_VERSION, "kind": "view", "label": "Pa pressure field pairs", "requires": [],
        "ports": [copy.deepcopy(ports[-1])], "view": {"primitive": "orthogonal_slices",
        "query": {"id": "tuldok:runs.model_evidence", "version": "2.0.0"},
        "label": "Same-request native and predicted pressure in Pa"}}
    return {"schema": "rheon-typed-pressure-intent-v1", "status": "prospective_unqualified",
            "root": str(root), "manifest_path": manifest_path, "project": project,
            "material_sha256": digest(path.parent / "migration-material.json"),
            "source_witness_sha256": digest(witness_path), "runtime": material["runtime"],
            "definitions": definitions, "architecture": architecture, "preprocessing_policy": preprocessing,
            "output_policy": output, "physical_context": context, "fixed_probe": probe, "metric": metric,
            "recipes": recipes, "planned_optimizer_updates": 1, "restore_optimizer_updates": 0,
            "limits": {"attempt_slots": 2, "admitted_wall_seconds": 1600,
                       "retained_bytes": 512 * 1024**2, "peak_staging_bytes": 1024**3},
            "trial_intent": {"replicate": 0, "training_recipe": "train", "restore_recipe": "restore",
                             "admission": "closed", "approval": None, "run_id": None,
                             "historical_lineage": "unknown; no historical Run is reparented",
                             "field_evidence": "unsupported until exact pressure profile is qualified"}}


def capability_report(bundle, recipe_contract):
    """Probe the current recipe owner without imports of pressure/numerical code."""
    recipe_contract.decode_definition(canonical(bundle["recipes"]["train"]))
    candidate = copy.deepcopy(bundle["recipes"]["train"])
    candidate.update(schema_version=2, format="external_model_evidence_recipe_v1", model_evidence={})
    try:
        recipe_contract.decode_definition(canonical(candidate))
    except recipe_contract.WorkbenchError as error:
        if getattr(error, "code", None) != "unsupported":
            raise
        observed = {"operation": "recipe_contract.decode_definition", "code": error.code, "reason": str(error)}
    else:
        observed = {"operation": "recipe_contract.decode_definition", "code": "unavailable",
                    "reason": "Runtime syntax alone does not qualify the pressure evidence policy."}
    return {"status": "unsupported", "execution_submitted": False, "observed": observed,
            "required_owner_work": [
                "Current snapshot captured-source Model publication and managed CUDA/CPU bound evidence composition",
                "Pressure task scalar context, solved-prior lineage, cubic geometry, Pa and cell-zero gauge interpretation",
                "Exact already-consumed validation probe bound separately from unchanged train-only fitting inputs",
                "Actual pressure trainer checkpoint/normalization, fresh CPU restore and typed field evidence adapter",
                "Negotiated pressure evidence/profile contracts and permission negatives before any admission"]}


def retain_prospective_intent(dataset, bundle):
    """Current inert domain operations only; caller provides isolated authenticated catalog.

    This intentionally cannot prepare/admit a Trial, grant authority, open a
    Study, associate historical Runs or synthesize an evidence acceptance.
    """
    import recipe_contract
    report = capability_report(bundle, recipe_contract)
    project = dataset.projects.open_snapshot({"root": "pressure", "manifest_path": bundle["manifest_path"]})
    require(project["id"] == bundle["project"]["id"], "registered Project identity differs")
    task = bundle["definitions"]["task"]
    selector = next(ref for ref in project["recipes"] if ref["id"] == "train")
    model = dataset.models.create({"name": "Prospective pressure diagnostic model; unqualified",
        "task": {"kind": "task", "catalog_id": dataset.db.dataset_id, "id": TASK_ID,
                 "revision": 1, "sha256": sha(task)}, "kind": "captured_source",
        "source_selector": {"project_id": project["id"], "recipe_id": "train", "expected_sha256": selector["sha256"]},
        "specification": {"architecture": bundle["architecture"], "physical_context": bundle["physical_context"],
                          "declaration_status": "prospective_unqualified"},
        "preprocessing_policy": bundle["preprocessing_policy"], "output_policy": bundle["output_policy"]})
    study = dataset.studies.create({"name": "Prospective bounded pressure diagnostic; admission closed",
        "limits": bundle["limits"], "policy": {"format": "selection_policy_v1",
        "metric_definition_sha256": sha(bundle["metric"]), "checkpoint_rule": "explicit",
        "replicate_reduce": "individual", "required_capabilities": ["checkpoints", "field_pairs", "fresh_cpu_restore"],
        "numerical_acceptance": "diagnostic", "tie_break": "trial_id"}})
    require(not study["open"] and not study["trial_ids"], "prospective Study must remain closed and empty")
    return {**report, "project_id": project["id"], "model_draft_id": model["id"], "study_id": study["id"],
            "trial": None, "run": None, "approval": None, "optimizer_updates_executed": 0}


def prepare_current_trial(dataset, root, assembly_path, request_key):
    """Publish exact current Model and prepare one inert member of a closed Study.

    Runtime registration and owner authentication belong to the caller. This
    never approves, opens admission, submits, retries or replays an attempt.
    """
    root = Path(root).resolve(strict=True)
    assembly_path = Path(assembly_path).resolve(strict=True)
    require(assembly_path.is_relative_to(root), "assembly must be beneath the selected artifact root")
    assembly = decode(assembly_path.read_bytes())
    require(assembly["format"] == "rheon_pressure_project_assembly_v1"
            and assembly["execution_submitted"] is False and assembly["optimizer_updates_executed"] == 0,
            "prospective current pressure assembly required")
    selected = assembly["current_project"]
    project_path = rooted(root, selected["path"])
    require(digest(project_path) == selected["sha256"], "assembled current Project changed")
    project = dataset.projects.open_snapshot({"root": "pressure", "manifest_path": selected["path"]})
    require(project["id"] == selected["id"], "assembled Project identity differs")
    recipe = next(ref for ref in project["recipes"] if ref["id"] == "train")
    inspected = dataset.pipelines.inspect_external_recipe(project["id"], "train", recipe["observed_sha256"], contract_version=3)
    policy = inspected["definition"]["model_evidence"]
    require(inspected["definition"]["runtime"] == PROFILE
            and inspected["definition"]["runtime_device"] == "cuda"
            and policy["model_specification"] == assembly["model_specification"],
            "exact bound CUDA pressure policy required")
    draft = dataset.models.create({"name": "Pressure diagnostic Model; no quality promotion",
        "task": {"kind": "task", "catalog_id": dataset.db.dataset_id, "id": policy["task"]["id"],
                 "revision": policy["task"]["revision"], "sha256": sha(policy["task"]["definition"])},
        "kind": "captured_source", "specification": assembly["model_specification"],
        "source_selector": {"project_id": project["id"], "recipe_id": "train", "expected_sha256": recipe["observed_sha256"]},
        "preprocessing_policy": assembly["preprocessing_policy"], "output_policy": assembly["output_policy"]})
    model = dataset.models.publish(draft["id"], {"revision": draft["revision"]})
    references = {ref["id"]: ref for ref in project["data"]}
    inputs = {}
    for slot in inspected["definition"]["inputs"]:
        ref = references[slot]
        require(ref["status"] == "available" and ref["integrity"] == "verified", "current pressure input unavailable")
        inputs[slot] = {"reference_id": slot, "expected_sha256": ref["observed_sha256"]}
    graph = dataset.pipelines.create_external({"name": "One prospective CUDA pressure diagnostic",
        "project_id": project["id"], "recipe_id": "train", "expected_sha256": recipe["observed_sha256"], "inputs": inputs})
    study = dataset.studies.create({"name": "Bounded pressure diagnostic; admission closed", "limits": assembly["limits"],
        "policy": {"format": "selection_policy_v1", "metric_definition_sha256": sha(policy["metric_definitions"][0]),
                   "checkpoint_rule": "explicit", "replicate_reduce": "individual",
                   "required_capabilities": policy["profile"]["required_capabilities"],
                   "numerical_acceptance": "diagnostic", "tie_break": "trial_id"}})
    body = {"request_key": request_key, "revision": study["revision"], "graph_id": graph["id"],
            "graph_revision": graph["revision"], "model": model["ref"], "replicate": 0}
    trial = dataset.studies.prepare_trial(study["id"], body)
    current = dataset.studies.get(study["id"])["study"]
    require(not current["open"] and current["trial_ids"] == [trial["id"]], "closed single-Trial membership required")
    return {"status": "prepared_without_execution", "catalog_id": dataset.db.dataset_id,
            "project_id": project["id"], "model": model["ref"], "graph_id": graph["id"], "study_id": study["id"],
            "trial": trial, "prepare_body": body, "assembly_sha256": digest(assembly_path),
            "admission_open": False, "approval": None, "run": None, "optimizer_updates_executed": 0,
            "scientific_acceptance": "not_asserted", "historical_ancestry": "unknown"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = load(args.dependency_config)
    require(Path(__file__).resolve().is_relative_to(selection.source("lab")), "selected current lab producer required")
    bundle = declarations(selection.path_for("artifact_root"), args.training_manifest)
    runtime = selection.runtime("cpu")
    require(runtime["manifest_sha256"] == bundle["runtime"]["manifest_sha256"], "selected current runtime differs")
    sys.path.insert(0, str(selection.source("tuldok")))
    import extension_contract
    import recipe_contract
    for declaration in bundle["definitions"].values():
        extension_contract.declaration(canonical(declaration), "contribution")
    report = capability_report(bundle, recipe_contract)
    output = args.output.resolve()
    require(not output.exists() and output.is_relative_to(selection.path_for("migration")), "fresh contained output required")
    output.mkdir(parents=True)
    for role, declaration in bundle["definitions"].items():
        (output / (role + ".json")).write_bytes(canonical(declaration))
    (output / "prospective-intent.json").write_bytes(canonical(bundle))
    (output / "capability-report.json").write_bytes(canonical(report))
    print(json.dumps({"output": str(output), "status": report["status"], "optimizer_updates_executed": 0}))


if __name__ == "__main__":
    main()
