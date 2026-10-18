"""Refresh the offline HTML chart from retained scientific result records."""
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from drift_pressure_metrics import enrich_drift_pressure
from pass_sweep_chart import enrich_pass_sweep
from precision_chart import enrich_precision, precision_markup
from four_pass_study import four_pass_link, reserved_four_pass_ids

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "cohort-progress.html"
# Scientific records can live in the retained experiment checkout after cutover.
# Extend the embedded history there instead of requiring a branch switch.
if not (ROOT / "water-learning-sequence-v1/acceptance.json").exists():
    import runpy
    runpy.run_path(str(ROOT / "refresh-current-cohorts.py"), run_name="__main__")
    raise SystemExit(0)
sources = {}
rows = []


def read(name):
    path = ROOT / name
    raw = path.read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def add(group, name, source, median=None, worst=None, qualified=None, cases=None,
        population="Not recorded", policy="Raw proposal", **extra):
    rows.append(dict(group=group, name=name, source=source, median=median,
                     worst=worst, qualified=qualified, cases=cases,
                     population=population, policy=policy, **extra))


def native(group, name, source, result, **extra):
    r = result.get("relative_residual", {})
    add(group, name, source, r.get("median"), r.get("maximum"),
        result.get("float32_qualified", result.get("raw_qualified")),
        result.get("cases"), **extra)


def raw_cases(group, name, source, cases, **extra):
    raw = [r["raw_learned"] for r in cases]
    quality = [r.get("quality", r) for r in raw]
    values = [r["relative_residual"] for r in quality]
    add(group, name, source, statistics.median(values), max(values),
        sum(r.get("qualified", q.get("qualified", False)) for r, q in zip(raw, quality)),
        len(raw), **extra)


# The early model families have different populations; preserve their identity.
source = "global-pressure-training-v1/reports/physics.json"
record = read(source)
raw_cases("01 · Early model families", "Global CNN · 200 CPU updates", source,
          record["cases"], population="Early fixed-grid training", updates=200)
source = "global-pressure-groupnorm-native-dev-run-v1/reports/corrective-evaluation.json"
record = read(source)
raw_cases("01 · Early model families", "Global CNN + GroupNorm · 200 updates", source,
          record["cases"], population="Early fixed-grid training", updates=200)
source = "variable-cubic-pressure-model/acceptance.json"
record = read(source)
case_path = ROOT / "../../runs/variable-pressure-v1-workspace-1/evaluation/cases.jsonl"
if case_path.exists():
    cases = [json.loads(line) for line in case_path.read_text().splitlines()]
    raw_cases("01 · Early model families", "Variable-grid CNN · 10 channels", source,
              cases, population="4 simulations / 96 frames", updates=200,
              raw_source="../../runs/variable-pressure-v1-workspace-1/evaluation/cases.jsonl")

source = "sealed-water-pressure-v1/acceptance.json"
record = read(source)
for r in record["historical_comparison"][:2]:
    add("01 · Early model families", r["name"], source,
        r["relative_residual"]["median"], r["relative_residual"]["maximum"],
        r["raw_native_qualified"], r["cases"], population="4 simulations / 96 frames",
        updates=r["updates"], policy="12 observed development snapshots")
native("02 · Water cohort size", "Original water cohort · 256 sims / 2k updates",
       source, record["native"], population="256 simulations / 512 states",
       updates=2000, policy="128 reserved snapshots", training_seconds=record["training_elapsed_s"])

source = "water-learning-sequence-v1/acceptance.json"
record = read(source)
for key, label in [("fit-one", "One-case fit"), ("long-256", "Longer original cohort"),
                   ("scale-256", "Matched 256 cohort"), ("scale-1024", "Matched 1,024 cohort")]:
    r = record["trials"][key]
    native("02 · Water cohort size", label, source, r["validation_native"],
           population=f"{r['training_parents']:,} simulations / {r['training_snapshots']:,} states",
           updates=r["updates"], policy="128 validation snapshots", training_seconds=r["cuda_training_s"])
source = "water-learning-sequence-v1/reports/fresh-reserved.json"
native("02 · Water cohort size", "Selected 1,024 cohort · fresh reserved test",
       source, read(source), population="1,024 simulations / 1,024 states", policy="128 fresh reserved snapshots")

source = "water-conditioning-v1/reports/summary.json"
record = read(source)
for r in record["scientific"]:
    native("03 · Conditioning controls", r["trial"], source, r["validation"],
           population="1 simulation / 1 state", updates=10000,
           training_seconds=r["training_elapsed_s"])

for folder, group in [("water-residual-learning-v1", "04 · Residual learning"),
                       ("water-stability-learning-v1", "05 · Stability learning")]:
    source = f"{folder}/reports/summary.json"
    for r in read(source)["comparison"]:
        if r["trial"] == "retained-residual-recurrent":
            continue  # The identical measured policy already appears above.
        passes = r.get("steps_per_prediction", r.get("network_passes_per_solve"))
        add(group, r["trial"] + (" · fixed 32" if passes == 32 else ""), source,
            r["validation_median_relative_residual"], r["validation_max_relative_residual"],
            r["validation_qualified"], r["validation_cases"], "1 simulation / 1 state",
            f"Fixed {passes} network evaluations", evaluations=passes,
            updates=r.get("updates", r.get("training_updates")),
            trajectory_qualified=r["trajectory_qualified_ai_steps"], trajectory_steps=r["trajectory_ai_steps"])

source = "water-inference-budget-v1/reports/summary.json"
for r in read(source)["comparison"]:
    if r["trial"] in ["stability-modes", "stability-combined", "stability-modes · fixed repeated reference"]:
        continue  # Retained rows from the previous study, not new measurements.
    add("06 · Inference budget", r["trial"], source,
        r["development_median_relative_residual"], r["development_worst_relative_residual"],
        r["development_qualified"], r["development_cases"], "1 simulation / 1 state",
        f"{r['mean_network_evaluations']:.3f} mean evaluations / {r['max_network_evaluations']} max",
        evaluations=r["mean_network_evaluations"], updates=r["cuda_updates"])

source = "water-multigrid-learning-v1/reports/summary.json"
record = read(source)
fixed = read("water-multigrid-learning-v1/reports/fixed-eight.json")
for i, model in enumerate(json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', PAGE.read_text(), re.S).group(1))["models"]):
    # Every checkpoint gets an explicit entry, including its fixed-budget replay.
    a = model["adaptive"]
    add("07 · Multi-grid cohorts", model["name"], source,
        a["relative_residual"]["median"], a["relative_residual"]["maximum"], a["qualified"], 128,
        f"{model['simulations']} simulations / {model['states']} states",
        f"Adaptive · {model['mean_evaluations']:.3f} mean evaluations",
        evaluations=model["mean_evaluations"], timing_key=model["key"])
    if model["fixed"]:
        a = model["fixed"]
        add("07 · Multi-grid cohorts", model["name"] + " · fixed 8", "water-multigrid-learning-v1/reports/fixed-eight.json",
            a["relative_residual"]["median"], a["relative_residual"]["maximum"], a["qualified"], 128,
            f"{model['simulations']} simulations / {model['states']} states", "Fixed 8 network evaluations", evaluations=8)

for filename in ["trial-multigrid-single-pass.json", "trial-multigrid-two-pass.json", "trial-cohort16-two-pass.json"]:
    source = "water-multigrid-learning-v1/reports/" + filename
    r = read(source)
    raw = r.get("raw_native", {}).get("raw-validation-native")
    if raw:
        exposure = r["training_exposure"]
        native("07 · Multi-grid cohorts", r["trial"] + " · raw proposal", source, raw,
               population=f"{exposure['real_simulations']} simulations / {exposure['real_states']} states")

latest = ROOT / "water-recurrence-generalization-v1/reports"
for path in sorted(latest.glob("trial-*.json")):
    source = str(path.relative_to(ROOT))
    r = read(source)
    exposure = r["training_exposure"]
    pop = f"{exposure['real_simulations']} simulations / {exposure['real_states']} states"
    for key, label in [("raw-validation-native", "raw proposal")]:
        if key in r.get("raw_native", {}):
            native("08 · Recurrence and seed repeats", r["trial"] + " · " + label,
                   source, r["raw_native"][key], population=pop)
    a = r["adaptive_development"]
    native("08 · Recurrence and seed repeats", r["trial"] + " · adaptive", source,
           a["native"], population=pop, policy=f"Adaptive · {a['network_evaluations']['mean']:.3f} mean evaluations",
           evaluations=a["network_evaluations"]["mean"])
    fixed_source = f"../../runs/water-recurrence-generalization-v1/{r['trial']}/fixed-eight/report.json"
    if (ROOT / fixed_source).exists():
        f = read(fixed_source)
        native("08 · Recurrence and seed repeats", r["trial"] + " · fixed 8", fixed_source,
               f["native"], population=pop, policy="Fixed 8 network evaluations", evaluations=8)

# Keep pending trials visible; do not turn assembly or training completion into results.
recorded = {json.loads(p.read_text())["trial"] for p in latest.glob("trial-*.json")}
for trial in ["cohort-two-seed2", "cohort-two-seed3", "cohort-one-control", "cohort-four-control"]:
    if trial not in recorded:
        add("08 · Recurrence and seed repeats", trial, "water-recurrence-generalization-v1/execution-ledger.md",
            population="16 simulations / 32 states", policy="Results pending", status="pending")

# Discover subsequent studies using the same retained trial-record contract.
# Adding a completed study no longer requires adding its folder name to this script.
known_trial_studies = {"water-multigrid-learning-v1", "water-recurrence-generalization-v1"}
def study_order(study):
    selection = study / "reports/final-selection.json"
    frozen = read(str(selection.relative_to(ROOT))).get("frozen_at_utc", "") if selection.exists() else ""
    return frozen, study.name


new_studies = sorted({p.parent.parent for p in ROOT.glob("water-*/reports/trial-*.json")
                      if p.parent.parent.name not in known_trial_studies}, key=study_order)
for number, study in enumerate(new_studies, start=11):
    label = study.name.removeprefix("water-").removesuffix("-v1").replace("-", " ").capitalize()
    group = f"{number:02d} · {label}"
    for path in sorted((study / "reports").glob("trial-*.json")):
        source = str(path.relative_to(ROOT))
        r = read(source)
        if "training_exposure" not in r or "adaptive_development" not in r:
            continue
        exposure = r["training_exposure"]
        pop = f"{exposure['real_simulations']} simulations / {exposure['real_states']} states"
        raw = r.get("raw_native", {}).get("raw-validation-native")
        if raw:
            native(group, r["trial"] + " · raw proposal", source, raw, population=pop)
        a = r["adaptive_development"]
        native(group, r["trial"] + " · adaptive", source, a["native"], population=pop,
               policy=f"Adaptive · {a['network_evaluations']['mean']:.3f} mean evaluations",
               evaluations=a["network_evaluations"]["mean"], updates=r["cuda_optimizer_updates"])
        fixed_source = f"../../runs/{study.name}/{r['trial']}/fixed-eight/report.json"
        if (ROOT / fixed_source).exists():
            f = read(fixed_source)
            native(group, r["trial"] + " · fixed 8", fixed_source, f["native"],
                   population=pop, policy="Fixed 8 network evaluations", evaluations=8)

# Historical fixed-operator models remain a separate family, with their own gates.
source = "pressure-data/reports/final-comparison.json"
r = read(source)
q = r["quality_summaries"]["raw_learned"]
add("09 · Historical fixed-operator models", "Pressure-only learned modal model", source,
    worst=q["relative_residual_l2"], qualified=q["qualified"], cases=q["cases"],
    policy="Frozen final population · native target 1e−10", population="Fixed operator / pressure-only")
for source, label in [("../../experiments/trajectory-cpu-v1/result.json", "Trajectory modal model · CPU only"),
                      ("../../experiments/trajectory-v2/parallel-result.json", "Trajectory modal model · GPU inference")]:
    r = read(source)
    add("09 · Historical fixed-operator models", label, source,
        worst=r.get("max_relative_residual"), policy="Independent 120-frame rollouts · target 1e−8",
        population="12 training trajectories / 96 frames", speed=r["speedup_vs_jacobi"],
        speed_sgs=r["speedup_vs_sgs"], speed_scope="Whole simulation vs Jacobi PCG; hardware differs from water studies")
source = "tuldok-pressure-project/reports/results.json"
r = read(source)
add("09 · Historical fixed-operator models", "Trajectory modal model · managed CPU replay", source,
    worst=r["comparison"]["max_relative_residual"], speed=r["comparison"]["speedup_vs_jacobi"],
    speed_sgs=r["comparison"]["speedup_vs_sgs"], population="Same fixed-operator training cohort",
    policy="Managed runtime / different CPU allocation")

# This negative CNN experiment reports pressure-field error, not equation residual.
# Keep its metric explicit instead of silently substituting one error for another.
for split, metrics in r["cnn"]["split_metrics"].items():
    add("10 · Historical CNN field accuracy", "Managed CNN · " + split, source,
        worst=metrics["max_relative_pressure_l2"], cases=metrics["cases"],
        policy="200 updates · pressure-field relative L2", metric="field_l2",
        mean=metrics["mean_relative_pressure_l2"], population="8 train / 4 validation / 4 test cases")

# Frozen-weight policy studies have episode records rather than trial fits.
# Keep their field-error/timing populations separate from raw-proposal cohorts.
policy_studies = []
policy_phases = []
for path in sorted(ROOT.glob("water-*/reports/summary.json")):
    summary = read(str(path.relative_to(ROOT)))
    if summary.get("schema") != "rheon-stopping-floor-completed-sequence-v1" or summary.get("state") != "completed":
        continue
    if summary["new_training_runs"] != 0 or summary["optimizer_updates"] != 0:
        raise ValueError(f"Unexpected training in frozen-policy study: {path}")
    study = path.parent.parent.name
    selection_source = str((path.parent / "selection.json").relative_to(ROOT))
    selection = read(selection_source)
    if selection != summary["selection"]:
        raise ValueError(f"Frozen policy selection disagrees with summary: {path}")
    phases, model_weights, populations = [], {}, {}
    group = f"{11 + len(new_studies) + len(policy_studies):02d} · Stopping floor policies"
    for phase, summary_name, title in [
        ("floor-development", "development-summary", "Consumed development · minimum-pass policies"),
        ("floor-timing", "timing-summary", "Serial timing · minimum-pass policies"),
        ("pressure-development", "pressure-development-summary", "Consumed development · pressure-aware policies"),
        ("pressure-timing", "pressure-timing-summary", "Serial timing · pressure-aware policies"),
        ("fresh-validation", "validation-summary", "Fresh validation · frozen challenger"),
    ]:
        phase_source = str((path.parent / (summary_name + ".json")).relative_to(ROOT))
        metrics = read(phase_source)
        record_source = str((path.parent / (phase + "-episode-records.json")).relative_to(ROOT))
        episodes = read(record_source)
        if (sources[record_source] != metrics["source_report_sha256"]
                or episodes["state"] != "completed" or episodes["optimizer_updates"] != 0
                or not episodes["all_qualified"]):
            raise ValueError(f"Policy episode source mismatch: {record_source}")
        timed = phase in ("floor-timing", "pressure-timing")
        native_times = [t for r in episodes["records"] if r["model"] == "native" for t in r["ai_seconds"]]
        native_ms = statistics.median(native_times) * 1000 if timed else None
        models = []
        for model in episodes["models"]:
            name, weights = model["name"], model["weights_sha256"]
            if name in model_weights and model_weights[name] != weights:
                raise ValueError(f"Policy name changes checkpoint: {name}")
            model_weights[name] = weights
            weights_path = ROOT.parent.parent / model["artifacts"] / "weights.npz"
            if hashlib.sha256(weights_path.read_bytes()).hexdigest() != weights:
                raise ValueError(f"Policy checkpoint mismatch: {weights_path}")
            result = metrics["models"][name]
            rows_for_model = [r for r in episodes["records"] if r["model"] == name]
            if len(rows_for_model) != result["episodes"] or sum(r["result"]["ai_steps"] for r in rows_for_model) != result["ai_steps"]:
                raise ValueError(f"Incomplete policy population: {record_source}/{name}")
            measured_ms = result["steady_ai_seconds_median"] * 1000 if timed else None
            models.append(dict(name=name, weights_sha256=weights,
                               minimum_required_network_evaluations=model.get("minimum_network_evaluations", 0),
                               inference_median_ms=measured_ms, **result))
            add(group, name + " · " + phase, phase_source,
                median=result["median_paired_pressure_relative_l2"], worst=result["worst_paired_pressure_relative_l2"],
                qualified=result["qualified_episodes"], cases=result["episodes"], metric="field_l2",
                population=title, policy="Frozen weights · " + name, weights_sha256=weights,
                speed=native_ms / measured_ms if timed else None,
                speed_baseline="Rheon native-only control",
                speed_scope="Serial uninstrumented pressure-solve timing versus this phase's native-only control; excludes first AI step",
                evaluations=result["mean_network_evaluations"], updates=0)
        populations["final_holdout" if phase == "fresh-validation" else phase.replace("-", "_")] = dict(models={
            m["name"]: dict(scenes=m["episodes"], qualified_scenes=m["qualified_episodes"],
                            attempted_ai_steps=m["ai_steps"], qualified_ai_steps=m["ai_steps"],
                            mean_network_evaluations=m["mean_network_evaluations"],
                            worst_pressure_relative_l2=m["worst_paired_pressure_relative_l2"])
            for m in models
        })
        entry = dict(id=phase, title=title, timed=timed, native_median_ms=native_ms,
                     models=models, source=phase_source, records_source=record_source,
                     timing_scope=metrics["timing_scope"], sparse_band_scope=metrics["sparse_band_scope"])
        phases.append(entry)
        policy_phases.append((entry, episodes))
    policy_studies.append(dict(id=study, title="Stopping floor", phases=phases,
                               selected_policy=summary["selected_policy"], source=str(path.relative_to(ROOT)),
                               selection_source=selection_source, frozen_at_utc=selection["frozen_at_utc"],
                               model_weights=model_weights, populations=populations,
                               candidate_promoted=summary["candidate_promoted"],
                               fresh_numeric_gates_passed=summary["fresh_numeric_gates_passed"]))

def regression_chart_helpers(context):
    """Share receipt/field validation across study-specific record adapters."""
    definitions, phases, populations = (context[k] for k in ("definitions", "phases", "populations"))
    group, episodes, timing = (context[k] for k in ("group", "episodes", "timing"))
    bindings_source, episodes_source, timing_source = (context[k] for k in ("bindings_source", "episodes_source", "timing_source"))
    cache = {}

    def regression_measurement(row, model, receipt_seal=None, frames_seal=None, callbacks_seal=None):
        output = row["output"]
        key = (output, model["name"])
        if key in cache:
            return cache[key]
        directory = ROOT.parent.parent / output / "native"
        receipt_source = "../../" + str((directory / "receipt.json").relative_to(ROOT.parent.parent))
        receipt = read(receipt_source)
        if sources[receipt_source] != (receipt_seal or row["receipt_sha256"]):
            raise ValueError(f"Regression receipt mismatch: {receipt_source}")
        frames_path = directory / "frames.jsonl"
        raw = frames_path.read_bytes()
        frames_source = "../../" + str(frames_path.relative_to(ROOT.parent.parent))
        sources[frames_source] = hashlib.sha256(raw).hexdigest()
        callbacks_path = directory / "callbacks.jsonl"
        callbacks_source = "../../" + str(callbacks_path.relative_to(ROOT.parent.parent))
        raw_callbacks = callbacks_path.read_bytes()
        sources[callbacks_source] = hashlib.sha256(raw_callbacks).hexdigest()
        if ((frames_seal and sources[frames_source] != frames_seal)
                or (callbacks_seal and sources[callbacks_source] != callbacks_seal)):
            raise ValueError(f"Archived regression fields changed: {output}")
        frames = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        ai = [f for f in frames if f.get("ai_quality")]
        callbacks = [json.loads(line) for line in raw_callbacks.decode().splitlines() if line.strip()]
        if (not ai or len(ai) != receipt["result"]["ai_steps"] or len(callbacks) != len(ai)
                or receipt["result"]["hard_stop"] or not row["qualified"]
                or not all(f["ai_quality"]["qualified"] for f in ai)
                or ai[-1]["step"] != receipt["scene"]["steps"]):
            raise ValueError(f"Incomplete broad regression trajectory: {output}")
        sparse = []
        diagnostic = directory.parent / "diagnostics.json"
        if model["name"] != "reference-current" and diagnostic.exists():
            record = read("../../" + str(diagnostic.relative_to(ROOT.parent.parent)))
            if record["weights_sha256"] != model["weights_sha256"]:
                raise ValueError("Sparse diagnostic checkpoint mismatch")
            sparse = record["samples"]
        result = dict(errors=[f["relative_pressure_difference"] for f in ai],
                      calls=[c["network_passes"] for c in callbacks], sparse=sparse,
                      row={**row, "expected_binary_sha256": receipt["binary_sha256"],
                           "chart_source": bindings_source if "reference-links" in output else timing_source if "serial-timing" in output else episodes_source})
        cache[key] = result
        return result

    def regression_phase(phase_id, title, selected, source, expected=None, timed=False, native_ms=None):
        models, rows = [], []
        for name, definition in definitions.items():
            if name not in selected:
                continue
            items = selected[name]
            errors = [v for item in items for v in item["errors"]]
            calls = [v for item in items for v in item["calls"]]
            sparse = [s for item in items for s in item["sparse"]]
            result = dict(name=name, weights_sha256=definition["weights_sha256"], episodes=len(items),
                          qualified_episodes=len(items), ai_steps=len(errors),
                          mean_network_evaluations=statistics.mean(calls), maximum_network_evaluations=max(calls),
                          minimum_network_evaluations=min(calls), worst_paired_pressure_relative_l2=max(errors),
                          median_paired_pressure_relative_l2=statistics.median(errors),
                          inference_median_ms=statistics.median([t for item in items for t in item["row"]["ai_seconds"]])*1000 if timed else None,
                          sparse_worst_pressure_relative_l2=max((s["pressure_relative_l2"] for s in sparse), default=None),
                          sparse_worst_low_band_relative_l2=max((s["mean_free_pressure_relative_l2"]*math.sqrt(s["low_mode_error_energy_fraction"]) for s in sparse), default=None),
                          sparse_worst_other_band_relative_l2=max((s["mean_free_pressure_relative_l2"]*math.sqrt(max(0, 1-s["low_mode_error_energy_fraction"])) for s in sparse), default=None))
            if expected:
                recorded = expected["models"][name]
                if (recorded["scenes"] != len(items) or recorded["ai_steps"] != len(errors)
                        or recorded["maximum_network_evaluations"] != result["maximum_network_evaluations"]
                        or not math.isclose(recorded["worst_paired_pressure_relative_l2"], max(errors), rel_tol=1e-12)
                        or not math.isclose(recorded["mean_network_evaluations"], statistics.mean(calls), rel_tol=1e-12)):
                    raise ValueError(f"Regression population disagrees with frames: {phase_id}/{name}")
            models.append(result)
            rows.extend(item["row"] for item in items)
            add(group, name + " · " + phase_id, source,
                median=result["median_paired_pressure_relative_l2"], worst=result["worst_paired_pressure_relative_l2"],
                qualified=len(items), cases=len(items), metric="field_l2", population=title,
                policy="Frozen weights · " + name, weights_sha256=definition["weights_sha256"],
                speed=native_ms/result["inference_median_ms"] if timed else None,
                speed_baseline="paired Rheon native track",
                speed_scope="Current serial whole Engine.advance timing; first AI timing sample excluded",
                evaluations=result["mean_network_evaluations"], updates=0)
        phase = dict(id=phase_id, title=title, models=models, timed=timed, native_median_ms=native_ms,
                     native_label="Paired native track", source=source, records_source=timing_source if timed else episodes_source,
                     timing_scope=timing["scope"] if timed else episodes["timing_scope"],
                     gates=expected["gates"] if expected else {}, passed=expected["passed"] if expected else None)
        reference_policy = "reference-current" if "reference-current" in selected else "late-pressure"
        candidate_policy = context.get("selected_policy", "late-pressure")
        phase.update(reference_policy=reference_policy, candidate_policy=candidate_policy)
        phases.append(phase)
        populations[phase_id] = dict(reference=reference_policy, candidate=candidate_policy, models={m["name"]: dict(scenes=m["episodes"], qualified_scenes=m["qualified_episodes"],
            attempted_ai_steps=m["ai_steps"], qualified_ai_steps=m["ai_steps"], mean_network_evaluations=m["mean_network_evaluations"],
            worst_pressure_relative_l2=m["worst_paired_pressure_relative_l2"]) for m in models})
        policy_phases.append((phase, dict(models=list(definitions.values()), records=rows, binary_sha256=episodes["binary_sha256"])))

    return regression_measurement, regression_phase


# Broad frozen-weight regression: reuse archived numerical references, but keep
# its consumed populations and current whole-step timing workloads separate.
for path in sorted(ROOT.glob("water-*/reports/summary.json")):
    summary = read(str(path.relative_to(ROOT)))
    if summary.get("schema") != "rheon-pressure-bound-broad-completed-v1" or summary.get("state") != "completed":
        continue
    report_root = path.parent
    study_id = path.parent.parent.name
    regression_source = str((report_root / "regression-summary.json").relative_to(ROOT))
    regression = read(regression_source)
    timing_source = str((report_root / "serial-timing.json").relative_to(ROOT))
    timing = read(timing_source)
    bindings_source = str((report_root / "population-bindings.json").relative_to(ROOT))
    bindings = read(bindings_source)
    episodes_source = str((report_root / "regression-episodes.json").relative_to(ROOT))
    episodes = read(episodes_source)
    if (sources[regression_source] != summary["analysis_sha256"]
            or sources[timing_source] != summary["timing_sha256"]
            or sources[bindings_source] != regression["binding_report_sha256"]
            or sources[episodes_source] != regression["source_report_sha256"]
            or timing["development_report_sha256"] != sources[episodes_source]
            or summary["optimizer_updates"] != 0 or not episodes["all_qualified"]):
        raise ValueError(f"Broad policy regression source mismatch: {path}")
    definitions = {m["name"]: {**m, **bindings["models"][m["name"]]}
                   for m in [bindings["reference_model"], *episodes["models"]]}
    for model in definitions.values():
        weights_path = ROOT.parent.parent / model["artifacts"] / "weights.npz"
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != model["weights_sha256"]:
            raise ValueError(f"Regression checkpoint mismatch: {weights_path}")
    reference_rows = {r["scene"]: r for r in bindings["reference_rows"]}
    candidate_rows = {r["scene"]: r for r in episodes["records"] if r["model"] == "late-pressure"}
    cases = {c["scene"]: c for c in bindings["cases"]}
    if len(cases) != 74 or len(candidate_rows) != 74:
        raise ValueError("Broad regression must retain all 74 declared cases")
    group = f"{11 + len(new_studies) + len(policy_studies):02d} · Pressure bound regression policies"
    phases, populations = [], {}
    cache = {}

    regression_measurement, regression_phase = regression_chart_helpers(dict(
        definitions=definitions, phases=phases, populations=populations, group=group,
        episodes=episodes, timing=timing, bindings_source=bindings_source,
        episodes_source=episodes_source, timing_source=timing_source))

    measured = {}
    for index, info in cases.items():
        # Bind every reused reference to its immutable original report/fields.
        original_source = "../../" + info["source_report"]
        read(original_source)
        if sources[original_source] != info["source_report_sha256"]:
            raise ValueError("Archived reference report changed")
        for name, row in [("reference-current", reference_rows[index]), ("late-pressure", candidate_rows[index])]:
            measured[index, name] = regression_measurement(row, definitions[name],
                info["source_receipt_sha256"] if name == "reference-current" else None,
                info["source_frames_sha256"] if name == "reference-current" else None,
                info["source_callbacks_sha256"] if name == "reference-current" else None)

    for grouping, records in [("group", regression["replacement_groups"]), ("population", regression["subgroups"])]:
        for name, expected in records.items():
            selected = {model: [measured[i, model] for i, info in cases.items() if info[grouping] == name] for model in definitions}
            regression_phase(grouping + "-" + name, "Consumed " + ("replacement group" if grouping == "group" else "subpopulation") + " · " + name,
                             selected, regression_source, expected)
    for scene in sorted({r["scene"] for r in timing["records"]}):
        serial_rows = [r for r in timing["records"] if r["scene"] == scene]
        selected = {name: [regression_measurement({**r, "output": f"runs/{study_id}/serial-timing/scene-{scene:02d}-{name}-{r['repetition']}"}, definition)
                           for r in serial_rows if r["model"] == name] for name, definition in definitions.items()}
        native_ms = statistics.median([t for r in serial_rows for t in r["native_seconds"]])*1000
        regression_phase("serial-" + str(scene), f"Serial whole-step timing · scene {scene} · two repeats", selected, timing_source, timed=True, native_ms=native_ms)
    policy_studies.append(dict(id=study_id, title="Pressure bound regression", phases=phases, populations=populations,
        selected_policy="late-pressure", source=str(path.relative_to(ROOT)), selection_source=bindings_source,
        frozen_at_utc=episodes["started_at_utc"], model_weights={name: m["weights_sha256"] for name, m in definitions.items()},
        candidate_promoted=summary["candidate_promoted"], fresh_numeric_gates_passed=None,
        default_phase="group-minimum8-failed8", default_population="group-minimum8-failed8",
        population_labels={p["id"]: p["title"] for p in phases},
        outcome_note="All 74 consumed scenes qualified. The recent eight-scene group failed the peak-call gate: 17 vs 11. Candidate not promoted; eight reserved fresh-validation scenes remain unused.",
        source_scope="Archived numerical references; current candidate. Current serial Engine.advance timings are separate workloads; sparse reference spectra unavailable."))

# New matched fits can share the broad-regression schema without legacy trials.
for path in sorted(ROOT.glob("water-*/reports/summary.json")):
    summary = read(str(path.relative_to(ROOT)))
    if summary.get("schema") != "rheon-earlier-pressure-completed-experiment-v1" or summary.get("state") != "completed":
        continue
    report_root, study_id = path.parent, path.parent.parent.name
    for filename, expected_sha in summary["sources"].items():
        source = str((report_root / filename).relative_to(ROOT))
        read(source)
        if sources[source] != expected_sha:
            raise ValueError(f"Earlier-pressure source changed: {source}")
    bindings_source = str((report_root / "training-and-admission.json").relative_to(ROOT))
    admission = read(bindings_source)
    regression_source = str((report_root / "regression-summary.json").relative_to(ROOT))
    regression = read(regression_source)
    episodes_source = str((report_root / "regression-episodes.json").relative_to(ROOT))
    episodes = read(episodes_source)
    timing_source = str((report_root / "serial-timing.json").relative_to(ROOT))
    timing = read(timing_source)
    if (regression["sources"]["admission_sha256"] != sources[bindings_source]
            or regression["sources"]["report_sha256"] != sources[episodes_source]
            or not episodes["all_qualified"] or summary["full_training_runs"] != 2):
        raise ValueError("Earlier-pressure admission/episode mismatch")
    for item in admission["sources"]:
        original = ROOT.parent.parent / item["path"]
        if hashlib.sha256(original.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Earlier-pressure frozen source changed: {original}")
    old_root = ROOT / "water-pressure-bound-regression-v1/reports"
    original_bindings = read(str((old_root / "population-bindings.json").relative_to(ROOT)))
    old_episodes = read(str((old_root / "regression-episodes.json").relative_to(ROOT)))
    definitions = {m["name"]: m for m in [
        {**original_bindings["reference_model"], **original_bindings["models"]["reference-current"]},
        *old_episodes["models"], *episodes["models"]]}
    phases, populations = [], {}
    group = f"{11 + len(new_studies) + len(policy_studies):02d} · Early pressure rollout comparisons"
    regression_measurement, regression_phase = regression_chart_helpers(dict(
        definitions=definitions, phases=phases, populations=populations, group=group,
        episodes=episodes, timing=timing, bindings_source=bindings_source,
        episodes_source=episodes_source, timing_source=timing_source, selected_policy=regression["frozen_challenger"]))
    cases = {c["scene"]: c for c in original_bindings["cases"]}
    rows_by_name = {"reference-current": {r["scene"]: r for r in original_bindings["reference_rows"]},
                    "late-pressure": {r["scene"]: r for r in old_episodes["records"] if r["model"] == "late-pressure"}}
    for model in episodes["models"]:
        rows_by_name[model["name"]] = {r["scene"]: r for r in episodes["records"] if r["model"] == model["name"]}
    sealed_cases = {(c["scene"], c["model"]): c for c in regression["cases"]}
    if len(sealed_cases) != 296:
        raise ValueError("Earlier-pressure comparison must retain four models on 74 scenes")
    measured = {}
    for index, info in cases.items():
        for name, definition in definitions.items():
            measured[index, name] = regression_measurement(rows_by_name[name][index], definition,
                receipt_seal=sealed_cases[index, name]["source_receipt_sha256"],
                frames_seal=info["source_frames_sha256"] if name == "reference-current" else None,
                callbacks_seal=info["source_callbacks_sha256"] if name == "reference-current" else None)
    for name, expected in regression["groups"].items():
        checks = {candidate + " vs " + baseline + ": " + gate: passed
                  for candidate, baselines in expected["gates"].items()
                  for baseline, gates in baselines.items() for gate, passed in gates.items()}
        selected = {model: [measured[i, model] for i, info in cases.items() if info["group"] == name] for model in definitions}
        regression_phase("group-" + name, "Consumed replacement group · " + name, selected, regression_source,
                         {"models": expected["models"], "gates": checks, "passed": all(checks.values())})
    for scene in sorted({r["scene"] for r in timing["records"]}):
        serial_rows = [r for r in timing["records"] if r["scene"] == scene]
        available = {r["model"] for r in serial_rows}
        selected = {name: [regression_measurement({**r, "output": f"runs/{study_id}/regression/serial-timing/scene-{scene:02d}-{name}-{r['repetition']}"}, definition)
                          for r in serial_rows if r["model"] == name] for name, definition in definitions.items() if name in available}
        native_ms = statistics.median([v for r in serial_rows for v in r["native_seconds"]])*1000
        regression_phase("serial-" + str(scene), f"Serial whole-step timing · scene {scene} · two repeats", selected, timing_source, timed=True, native_ms=native_ms)
    policy_studies.append(dict(id=study_id, title="Early pressure", phases=phases, populations=populations,
        selected_policy=regression["frozen_challenger"], source=str(path.relative_to(ROOT)), selection_source=bindings_source,
        frozen_at_utc=admission["frozen_at_utc"], model_weights={name: m["weights_sha256"] for name, m in definitions.items()},
        candidate_promoted=summary["candidate_promoted"], fresh_numeric_gates_passed=None,
        default_phase="group-original58", default_population="group-original58",
        population_labels={p["id"]: p["title"] for p in phases},
        outcome_note="Two matched 5,000-update fits completed. Both qualified all 74 consumed scenes, but neither passed every replacement gate against both baselines. Control was timed as the representative; fresh validation remains deferred and neither model was promoted.",
        training_runs=summary["full_training_runs"],
        new_weights=[model["weights_sha256"] for model in admission["models"]],
        source_scope="Control and first-pressure use matched initialization, sampling, normalization and cell work. Archived numerical baselines are separate from current serial timing, which measured control and late-pressure only."))
    for name, training in summary["training"].items():
        diagnostic_source = f"chart-diagnostics/{study_id}/{name}/validation.json"
        if not (ROOT / diagnostic_source).exists():
            continue
        diagnostic = read(diagnostic_source)
        native("19 · Early pressure original-proposal snapshots", name, diagnostic_source, diagnostic,
               population="Same 128 consumed validation snapshots", policy="Original proposal · chart CPU diagnostic",
               weights_sha256=training["weights_sha256"], updates=5000, training_seconds=training["training_seconds"])

# Recovery combines two new fits with stricter policies on retained weights.
# Keep its consumed populations and serial timing separate, and bind all rows
# to the frozen admissions, trajectory seals, and exact checkpoint artifacts.
for path in sorted(ROOT.glob("water-*/reports/summary.json")):
    summary = read(str(path.relative_to(ROOT)))
    if summary.get("schema") != "rheon-pressure-recovery-completed-v1" or summary.get("state") != "completed":
        continue
    study_id, report_root = path.parent.parent.name, path.parent
    selection_source = str((report_root / "selection.json").relative_to(ROOT))
    selection = read(selection_source)
    if summary["full_training_runs"] != 2 or selection["candidate_promoted"] != summary["candidate_promoted"]:
        raise ValueError("Recovery summary and selection disagree")
    old_root = ROOT / "water-pressure-bound-regression-v1/reports"
    original_bindings = read(str((old_root / "population-bindings.json").relative_to(ROOT)))
    old_episodes = read(str((old_root / "regression-episodes.json").relative_to(ROOT)))
    cases = {c["scene"]: c for c in original_bindings["cases"]}
    timing_source = str((report_root / "training/serial-timing.json").relative_to(ROOT))
    timing = read(timing_source)
    if timing["state"] != "completed" or not timing["all_qualified"] or len(cases) != 74:
        raise ValueError("Incomplete recovery timing or declared population")
    raw_group = max(int(row["group"].split(" · ")[0]) for row in rows) + 3
    for variant, label in [("policy", "Stricter stopping"), ("training", "Gentler pressure feedback")]:
        bindings_source = str((report_root / variant / "admission.json").relative_to(ROOT))
        admission = read(bindings_source)
        regression_source = str((report_root / variant / "regression-summary.json").relative_to(ROOT))
        regression = read(regression_source)
        episodes_source = f"../../runs/{study_id}/{variant}/development/report.json"
        episodes = read(episodes_source)
        if (regression["sources"]["admission_sha256"] != sources[bindings_source]
                or regression["sources"]["report_sha256"] != sources[episodes_source]
                or summary["regression"][variant] != regression["groups"]
                or episodes["state"] != "completed" or not episodes["all_qualified"]
                or episodes["models"] != admission["models"]
                or (variant == "training" and timing["development_report_sha256"] != sources[episodes_source])):
            raise ValueError(f"Recovery admission/episode seals disagree: {variant}")
        for item in admission["sources"]:
            original = ROOT.parent.parent / item["path"]
            if hashlib.sha256(original.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Recovery frozen source changed: {original}")
        definitions = {m["name"]: m for m in [
            {**original_bindings["reference_model"], **original_bindings["models"]["reference-current"]},
            *old_episodes["models"], *episodes["models"]]}
        for model in definitions.values():
            if hashlib.sha256((ROOT.parent.parent / model["artifacts"] / "weights.npz").read_bytes()).hexdigest() != model["weights_sha256"]:
                raise ValueError(f"Recovery checkpoint changed: {model['name']}")
        group_number = raw_group - (2 if variant == "policy" else 1)
        group = f"{group_number:02d} · Pressure recovery · {label.lower()}"
        phases, populations = [], {}
        regression_measurement, regression_phase = regression_chart_helpers(dict(
            definitions=definitions, phases=phases, populations=populations, group=group,
            episodes=episodes, timing=timing, bindings_source=bindings_source,
            episodes_source=episodes_source, timing_source=timing_source,
            selected_policy=regression["frozen_challenger"]))
        rows_by_name = {"reference-current": {r["scene"]: r for r in original_bindings["reference_rows"]},
                        "late-pressure": {r["scene"]: r for r in old_episodes["records"] if r["model"] == "late-pressure"}}
        for model in episodes["models"]:
            rows_by_name[model["name"]] = {r["scene"]: r for r in episodes["records"] if r["model"] == model["name"]}
        sealed_cases = {(c["scene"], c["model"]): c for c in regression["cases"]}
        if len(sealed_cases) != 296 or any(len(values) != 74 for values in rows_by_name.values()):
            raise ValueError("Recovery must retain four models on all 74 consumed scenes")
        measured = {}
        for index, info in cases.items():
            for name, definition in definitions.items():
                value = regression_measurement(rows_by_name[name][index], definition,
                    receipt_seal=sealed_cases[index, name]["source_receipt_sha256"],
                    frames_seal=info["source_frames_sha256"] if name == "reference-current" else None,
                    callbacks_seal=info["source_callbacks_sha256"] if name == "reference-current" else None)
                value["row"]["chart_source"] = regression_source if name in ("reference-current", "late-pressure") else episodes_source
                measured[index, name] = value
        for name, expected in regression["groups"].items():
            checks = {candidate + " vs " + baseline + ": " + gate: passed
                      for candidate, baselines in expected["gates"].items()
                      for baseline, gates in baselines.items() for gate, passed in gates.items()}
            selected = {model: [measured[i, model] for i, info in cases.items() if info["group"] == name] for model in definitions}
            regression_phase("group-" + name, label + " · consumed " + name, selected, regression_source,
                             {"models": expected["models"], "gates": checks, "passed": all(checks.values())})
        if variant == "training":
            for scene in sorted({r["scene"] for r in timing["records"]}):
                serial_rows = [r for r in timing["records"] if r["scene"] == scene]
                selected = {name: [regression_measurement({**r, "output": f"runs/{study_id}/training/serial-timing/scene-{scene:02d}-{name}-{r['repetition']}"}, definition)
                                  for r in serial_rows if r["model"] == name]
                            for name, definition in definitions.items() if any(r["model"] == name for r in serial_rows)}
                native_ms = statistics.median([v for r in serial_rows for v in r["native_seconds"]])*1000
                regression_phase("serial-" + str(scene), f"Serial whole-step timing · scene {scene} · two repeats", selected, timing_source, timed=True, native_ms=native_ms)
        policy_studies.append(dict(id=study_id + "-" + variant, title="Pressure recovery · " + label.lower(),
            phases=phases, populations=populations, selected_policy=regression["frozen_challenger"],
            source=str(path.relative_to(ROOT)), selection_source=selection_source,
            frozen_at_utc=admission["frozen_at_utc"], model_weights={name: m["weights_sha256"] for name, m in definitions.items()},
            candidate_promoted=summary["candidate_promoted"], fresh_numeric_gates_passed=None,
            default_phase="group-original58", default_population="group-original58",
            population_labels={p["id"]: p["title"] for p in phases},
            training_runs=2 if variant == "training" else 0,
            new_weights=[m["weights_sha256"] for m in admission["models"]] if variant == "training" else [],
            outcome_note=("Two matched 5,000-update fits completed; gentle-010 and gentle-025 each qualified all 74 consumed scenes. "
                          "Neither passed every comparative pressure/work gate. Serial whole-step timing measured gentle-010 and prior late-pressure only. "
                          if variant == "training" else
                          "Both stricter stopping policies qualified all 74 consumed scenes using the retained control weights. "
                          "Neither passed every comparative pressure/work gate; these are policy variants, not new training cohorts. ")
                         + "No candidate was promoted; eight reserved independent scenes remain unused.",
            source_scope="Original58, amended8 and recent8 are separate consumed populations. Archived baselines are receipt-bound. "
                         "Pressure-field trajectory deviation differs from raw-proposal equation error; serial timing is whole Engine.advance time."))
    for name, training in summary["training"].items():
        diagnostic_source = f"chart-diagnostics/{study_id}/{name}/validation.json"
        if (ROOT / diagnostic_source).exists():
            diagnostic = read(diagnostic_source)
            if diagnostic["weights_sha256"] != training["weights_sha256"]:
                raise ValueError(f"Recovery diagnostic checkpoint disagrees: {name}")
            native(f"{raw_group:02d} · Pressure recovery original-proposal snapshots", name, diagnostic_source, diagnostic,
                   population="Same 128 consumed validation snapshots", policy="Original proposal · chart CPU diagnostic",
                   weights_sha256=training["weights_sha256"], updates=5000, training_seconds=training["optimizer_seconds"])

text = PAGE.read_text()
match = re.search(r'(<script id="chart-data" type="application/json">)(.*?)(</script>)', text, re.S)
data = json.loads(match.group(2))
data["history"] = rows
data["policy_studies"] = policy_studies
data["pending_studies"] = []
for path in sorted(ROOT.glob("water-*/reports/current.json")):
    progress = read(str(path.relative_to(ROOT)))
    if progress.get("state") not in ("running", "preparing", "in_progress"):
        continue
    data["pending_studies"].append(dict(
        title=path.parent.parent.name.removeprefix("water-").removesuffix("-v1").replace("-", " ").capitalize(),
        source=str(path.relative_to(ROOT)), state=progress["state"], phase=progress.get("phase", "Results pending"),
        native_controls_completed=progress.get("native_controls_completed"),
        candidate_episodes_completed=progress.get("candidate_episodes_completed"),
    ))
data["history_sources"] = sources
refreshed_at = datetime.now(timezone.utc)
data["history_refreshed_utc"] = refreshed_at.isoformat()
snapshot_date = refreshed_at.astimezone(ZoneInfo("America/Vancouver"))
data["snapshot_date"] = snapshot_date.date().isoformat()
recurrence_summary = "water-recurrence-generalization-v1/reports/summary.json"
if (ROOT / recurrence_summary).exists():
    summary = read(recurrence_summary)
    selection = read("water-recurrence-generalization-v1/reports/final-selection.json")
    data["recurrence_populations"] = summary["populations"]
    data["recurrence_selection"] = dict(
        candidate=selection["candidate"]["name"],
        reference=selection["reference"]["name"],
        frozen_at_utc=selection["frozen_at_utc"],
        source=recurrence_summary,
    )
trajectory_studies = []
trial_names_by_weights = {}
for path in ROOT.glob("water-*/reports/trial-*.json"):
    trial = read(str(path.relative_to(ROOT)))
    if trial.get("weights_sha256") and trial.get("trial"):
        trial_names_by_weights[trial["weights_sha256"]] = trial["trial"]
for path in ROOT.glob("water-*/reports/summary.json"):
    selection_path = path.parent / "final-selection.json"
    if not selection_path.exists():
        continue
    summary = read(str(path.relative_to(ROOT)))
    selection = read(str(selection_path.relative_to(ROOT)))
    candidate, reference = selection.get("candidate", {}), selection.get("reference", {})
    if not isinstance(candidate, dict) or not candidate.get("name") or not isinstance(reference, dict):
        continue
    study = path.parent.parent.name
    decision_path = path.parent / "qualification-decision.json"
    decision = read(str(decision_path.relative_to(ROOT))) if decision_path.exists() else None
    populations = summary.get("populations")
    if populations is None:
        # Later studies keep their trajectory results in a role-keyed decision.
        decision_path = path.parent / "decision.json"
        if not decision_path.exists():
            continue
        decision = read(str(decision_path.relative_to(ROOT)))
        if not decision.get("populations") and not decision.get("development"):
            continue
        reference_weights = (selection.get("weights", {}).get("reference")
                             or selection.get("scores", {}).get("reference", {}).get("weights_sha256")
                             or decision.get("retained_weights_sha256"))
        reference_name = trial_names_by_weights.get(reference_weights)
        if reference_name is None:
            raise ValueError(f"No retained trial matches reference checkpoint for {study}")
        reference = {"name": reference_name}
        population_names = {"final": "final_holdout", "fresh": "fresh_development",
                            "prior": "prior_grids", "long": "long_development",
                            "consumed-0": "consumed_batch_0", "consumed-1": "consumed_batch_1",
                            "consumed-2": "consumed_batch_2",
                            "elongated": "exploratory_elongated"}
        populations = {}

        def merge_population(name, models, candidate_name):
            aliases = {"reference": reference_name, "candidate": candidate_name}
            target = populations.setdefault(population_names.get(name, name), {"models": {}})["models"]
            for role, metrics in models.items():
                model_name = aliases[role]
                if model_name in target and target[model_name] != metrics:
                    raise ValueError(f"Conflicting trajectory measurements for {study}/{name}/{model_name}")
                target[model_name] = metrics

        if decision.get("populations"):
            for name, models in decision["populations"].items():
                merge_population(name, models, candidate["name"])
        else:
            for candidate_name, development in decision["development"].items():
                for name, models in development.items():
                    merge_population(name, models, candidate_name)
            if decision.get("final"):
                merge_population("final", decision["final"], decision.get("challenger", candidate["name"]))
        summary_source = str(decision_path.relative_to(ROOT))
    else:
        summary_source = str(path.relative_to(ROOT))
    final_models = set(populations.get("final_holdout", {}).get("models", {}))
    all_models = {model for population in populations.values() for model in population["models"]}
    trajectory_studies.append(dict(
        id=study, title=study.removeprefix("water-").removesuffix("-v1").replace("-", " ").capitalize(),
        populations=populations, candidate=candidate["name"], reference=reference["name"],
        frozen_at_utc=selection["frozen_at_utc"], source=summary_source,
        selection_source=str(selection_path.relative_to(ROOT)), decision=decision,
        unexecuted_final_models=sorted(all_models - final_models) if final_models else [],
    ))
for study in policy_studies:
    trajectory_studies.append(dict(
        id=study["id"], title=study["title"] + (" · trained checkpoint rollouts" if study.get("training_runs") else " · frozen stopping policies"), kind="stopping-policy",
        populations=study["populations"], candidate=study["selected_policy"], reference="reference-current",
        model_weights=study["model_weights"], frozen_at_utc=study["frozen_at_utc"], source=study["source"],
        selection_source=study["selection_source"], decision=dict(candidate_promoted=study["candidate_promoted"]),
        population_labels=study.get("population_labels", {"final_holdout": "Fresh validation · frozen challenger"}),
        default_population=study.get("default_population"), outcome_note=study.get("outcome_note"),
    ))
data["trajectory_studies"] = sorted(trajectory_studies, key=lambda s: s["frozen_at_utc"])

# Joint plots use one entry per trained checkpoint, with the same output policy
# and population for the accuracy bounds and the saved CPU prediction timings.
cohort_points = []
seen_weights = set()


def relative_source(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "../../" + str(path.relative_to(ROOT.parent.parent))


def add_cohort(name, family, source, result, run_root, weights, started=None, direct_times=None):
    if weights in seen_weights:
        return
    residual = result.get("relative_residual", {})
    if not residual or not result.get("cases"):
        return
    timings = []
    timing_source = None
    timing_cases = None
    training_source = None
    run_root = (ROOT / run_root).resolve() if run_root else None
    if run_root and run_root.exists():
        candidates = sorted(set(run_root.glob("managed*/stage-*/artifacts/receipt.json")) |
                            set(run_root.glob("execution/stage-*/artifacts/receipt.json")))
        for path in candidates:
            receipt_source = relative_source(path)
            receipt = read(receipt_source)
            if receipt.get("mode") == "train" and receipt.get("optimizer_updates", 0) > 0:
                run_path = path.parent.parent / "run.json"
                if run_path.exists():
                    run = read(relative_source(run_path))
                    if run.get("started_at"):
                        started = run["started_at"]
                        training_source = relative_source(run_path)
            samples = receipt.get("per_case_inference_s", [])
            cases = receipt.get("cases", [])
            if not samples or len(samples) != len(cases):
                continue
            if receipt.get("execution", {}).get("device", "cpu") != "cpu" or receipt.get("mode") == "train":
                continue
            receipt_weights = receipt.get("weights_sha256", receipt.get("closure", {}).get("weights_sha256"))
            # Older portable receipts bind the immutable checkpoint through the
            # accepted restore, and store only tensor inventories in this file.
            if receipt_weights and receipt_weights != weights:
                continue
            target_split = "reserved_test" if "reserved" in result.get("scope", "").lower() else "validation"
            selected = [(c, t) for c, t in zip(cases, samples) if c.get("split") == target_split]
            if len(selected) != result["cases"]:
                selected = list(zip(cases, samples)) if len(cases) == result["cases"] else []
            if selected and (not timings or len(selected) > len(timings)):
                timings = [t for _, t in selected]
                timing_cases = [c for c, _ in selected]
                timing_source = receipt_source
    if direct_times is not None:
        timings = direct_times
        timing_source = source
    seen_weights.add(weights)
    cohort_points.append(dict(
        name=name, family=family, weights_sha256=weights, trained_at=started,
        best_error=residual.get("minimum"), worst_error=residual["maximum"],
        median_error=residual.get("median"), cases=result["cases"],
        qualified=result.get("float32_qualified", result.get("raw_qualified")),
        inference_median_ms=statistics.median(timings) * 1000 if timings else None,
        inference_best_ms=min(timings) * 1000 if timings else None,
        inference_worst_ms=max(timings) * 1000 if timings else None,
        timed_cases=len(timings), timing_cases=timing_cases,
        source=source, timing_source=timing_source, training_source=training_source,
        policy="Original model proposal, before adaptive repetition or native refinement",
        timing_scope="CPU pressure prediction, including model correction/operator work; excludes setup, sample assembly, native verification and fallback",
    ))


source = "water-learning-sequence-v1/acceptance.json"
for name, r in read(source)["trials"].items():
    add_cohort(name, "Cohort size", source, r["validation_native"],
               f"../../runs/water-learning-sequence-v1/{name}", r["training_weights_sha256"])
source = "water-conditioning-v1/reports/summary.json"
for r in read(source)["scientific"]:
    add_cohort(r["trial"], "Conditioning", source, r["validation"],
               f"../../runs/water-conditioning-v1/{r['trial']}", r["weights_sha256"])
for folder, family, names in [
    ("water-residual-learning-v1", "Residual learning", ["residual-linear-5000", "residual-linear", "residual-nonlinear", "residual-physical-loss", "residual-recurrent"]),
    ("water-stability-learning-v1", "Stability", ["stability-combined", "stability-modes", "stability-scale"]),
    ("water-inference-budget-v1", "Inference budget", ["stability-modes-single-pass", "stability-combined-single-pass-lr1e4"]),
]:
    for name in names:
        source = f"{folder}/reports/{name}.json"
        r = read(source)
        add_cohort(name, family, source, r["native_evaluations"]["validation-evaluation"],
                   f"../../runs/{folder}/{name}", r["weights_sha256"],
                   datetime.fromtimestamp(r["training_started_unix"], timezone.utc).isoformat()
                   if r.get("training_started_unix") else None)
for path in sorted(ROOT.glob("water-*/reports/trial-*.json")):
    source = str(path.relative_to(ROOT))
    r = read(source)
    raw = r.get("raw_native", {}).get("raw-validation-native")
    if raw:
        folder = path.parent.parent.name
        family = folder.removeprefix("water-").removesuffix("-v1").replace("-", " ").capitalize()
        add_cohort(r["trial"], family, source, raw,
                   f"../../runs/{folder}/{r['trial']}", r["weights_sha256"])
# Some completed fits publish admission/episode records rather than trial files.
# Their chart-only snapshot diagnostics preserve the same matched population.
for path in sorted(ROOT.glob("chart-diagnostics/*/*/validation.json")):
    source = str(path.relative_to(ROOT))
    diagnostic = read(source)
    if diagnostic.get("schema") != "rheon-chart-validation-v1":
        continue
    read(diagnostic["admission_source"])
    if (hashlib.sha256((ROOT / diagnostic["case_source"]).read_bytes()).hexdigest() != diagnostic["cases_sha256"]
            or sources[diagnostic["admission_source"]] != diagnostic["admission_sha256"]):
        raise ValueError(f"Chart snapshot diagnostic source mismatch: {path}")
    if diagnostic["weights_sha256"] in seen_weights:
        continue
    add_cohort(diagnostic["name"], diagnostic["family"], source, diagnostic,
               diagnostic["run_root"], diagnostic["weights_sha256"], direct_times=diagnostic["per_case_inference_s"])
    cohort_points[-1].update(timing_cases=diagnostic["timing_cases"], run_root=diagnostic["run_root"],
                             case_source=diagnostic["case_source"], validation_summary_source=source,
                             diagnostic_scope=diagnostic["scope"])

source = "sealed-water-pressure-v1/acceptance.json"
r = read(source)
add_cohort("Original 256-simulation water cohort", "Original water", source, r["native"],
           "../../runs/sealed-water-v1-3", r["weights_sha256"])
for folder, label, run_folder in [
    ("variable-cubic-pressure-model", "Variable grid · 10 channels", "variable-pressure-v1-workspace-1"),
    ("seven-channel-pressure-model", "Variable grid · 7 channels", "variable-pressure-v2-workspace-1"),
    ("cuda-training-cpu-inference", "CUDA portability · 20 updates", "cuda-pressure-v1-workspace-2"),
]:
    source = f"{folder}/acceptance.json"
    r = read(source)
    path = (ROOT / f"../../runs/{run_folder}/evaluation/cases.jsonl").resolve()
    if not path.exists():
        continue
    case_source = relative_source(path)
    raw_bytes = path.read_bytes()
    sources[case_source] = hashlib.sha256(raw_bytes).hexdigest()
    cases = [json.loads(line) for line in raw_bytes.decode().splitlines()]
    quality = [c["raw_learned"].get("quality", c["raw_learned"]) for c in cases]
    errors = [q["relative_residual"] for q in quality]
    result = dict(cases=len(cases), relative_residual=dict(minimum=min(errors), maximum=max(errors), median=statistics.median(errors)))
    weights = r.get("weights_sha256")
    if not weights:
        receipts = sorted((ROOT / f"../../runs/{run_folder}/execution").glob("stage-01/artifacts/receipt.json"))
        if receipts:
            restored = read(relative_source(receipts[0].resolve()))
            weights = restored.get("weights_sha256") or hashlib.sha256((receipts[0].parent / "weights.npz").read_bytes()).hexdigest()
    add_cohort(label, "Early models", case_source, result,
               f"../../runs/{run_folder}", weights)
for source, training_source, label in [
    ("global-pressure-training-v1/reports/physics.json", "global-pressure-training-v1/reports/training.json", "Global CNN · 200 updates"),
    ("global-pressure-groupnorm-native-dev-run-v1/reports/corrective-evaluation.json", "global-pressure-groupnorm-v1/reports/training.json", "Global CNN + GroupNorm"),
]:
    evaluation, training = read(source), read(training_source)
    quality = [c["raw_learned"].get("quality", c["raw_learned"]) for c in evaluation["cases"]]
    errors = [q["relative_residual"] for q in quality]
    result = dict(cases=len(errors), relative_residual=dict(minimum=min(errors), maximum=max(errors), median=statistics.median(errors)))
    if "recipe_receipt" in training:
        weights = training["recipe_receipt"]["weights_sha256"]
        times = training["recipe_receipt"]["per_case_inference_s"]
        started = training["started_at"]
    else:
        weights = training["artifact_inventory"]["weights.npz"]["sha256"]
        times = [c["recorded_model_inference_s"] for c in evaluation["cases"]]
        started = training["cost"]["started_at"]
    add_cohort(label, "Early models", source, result, None, weights, started, times)
    cohort_points[-1]["training_source"] = training_source
    if "recipe_receipt" in training:
        cohort_points[-1]["timing_source"] = training_source
cohort_points.sort(key=lambda c: (c["trained_at"] is None, c["trained_at"] or "", c["name"]))
reserved_ids = reserved_four_pass_ids(ROOT)
number = 1
for cohort in cohort_points:
    while f'C{number:02d}' in reserved_ids:
        number += 1
    cohort["id"] = f"C{number:02d}"
    cohort["order"] = number
    number += 1
data["cohort_points"] = cohort_points

# Resolution difficulty uses the identical validation snapshots behind the
# accuracy/speed comparison, with one worst raw proposal error per grid/spacing.
def validation_identity(case):
    return json.dumps([case["trajectory_id"], case.get("native_frame", case["frame"]),
                       case["counts_xyz"], case["cell_size_m"]], separators=(",", ":"))


resolution_evaluations = []
training_exposures = []
training_exposure_omissions = []
resolution_population = None
for cohort in cohort_points:
    timing_cases = cohort.get("timing_cases")
    if cohort["cases"] != 128 or not timing_cases or not all(c.get("split") == "validation" for c in timing_cases):
        continue
    identities = sorted(validation_identity(c) for c in timing_cases)
    if resolution_population is None:
        resolution_population = identities
    if identities != resolution_population:
        continue
    # These matched inference receipts live under <trial>/<managed>/stage-N/artifacts.
    trial_root = (ROOT / cohort["run_root"]).resolve() if cohort.get("run_root") else (ROOT / cohort["timing_source"]).resolve().parents[3]
    candidates = [trial_root / folder / "cases.jsonl"
                  for folder in ["validation-evaluation", "raw-validation-native"]]
    case_path = ROOT / cohort["case_source"] if cohort.get("case_source") else next((p for p in candidates if p.exists()), None)
    if case_path is None:
        raise ValueError(f"Matched resolution cases unavailable: {cohort['name']}")
    summary_source = cohort.get("validation_summary_source") or relative_source(case_path.with_name("evaluation.json"))
    summary = read(summary_source)
    if summary["weights_sha256"] != cohort["weights_sha256"]:
        raise ValueError(f"Resolution checkpoint mismatch: {case_path}")
    raw = case_path.read_bytes()
    case_source = relative_source(case_path)
    sources[case_source] = hashlib.sha256(raw).hexdigest()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if sorted(validation_identity(c) for c in cases) != resolution_population:
        raise ValueError(f"Resolution population mismatch: {case_path}")
    grouped, errors = {}, []
    for case in cases:
        quality = case["raw_learned"].get("quality", case["raw_learned"])
        error = quality["relative_residual"]
        if not math.isfinite(error) or error < 0:
            raise ValueError(f"Invalid raw resolution error: {case_path}")
        errors.append(error)
        key = (tuple(case["counts_xyz"]), case["cell_size_m"])
        grouped.setdefault(key, []).append((case, error, quality.get("qualified", False)))
    if not (math.isclose(max(errors), cohort["worst_error"], rel_tol=1e-12)
            and math.isclose(min(errors), cohort["best_error"], rel_tol=1e-12)
            and math.isclose(statistics.median(errors), cohort["median_error"], rel_tol=1e-12)):
        raise ValueError(f"Resolution errors disagree with cohort summary: {case_path}")
    domains = []
    for (counts, spacing), measurements in sorted(grouped.items()):
        worst_case, worst_error, _ = max(measurements, key=lambda m: m[1])
        domains.append(dict(
            counts_xyz=list(counts), cell_size_m=spacing, cases=len(measurements),
            worst_error=worst_error, best_error=min(m[1] for m in measurements),
            qualified=sum(bool(m[2]) for m in measurements),
            worst_case={k: worst_case[k] for k in ["trajectory_id", "frame", "native_frame"]},
        ))
    resolution_evaluations.append(dict(
        weights_sha256=cohort["weights_sha256"], source=case_source,
        summary_source=summary_source, cases=len(cases), domains=domains,
    ))
    # Inference receipts list only a diagnostic subset. Count the complete
    # training split bound to the exact checkpoint, never that receipt subset.
    training_receipts = sorted(trial_root.glob("managed*/stage-*/artifacts/receipt.json"))
    bound_training = None
    for receipt_path in training_receipts:
        receipt_source = relative_source(receipt_path)
        receipt = read(receipt_source)
        if (receipt.get("mode") == "train" and receipt.get("optimizer_updates", 0) > 0
                and receipt.get("outputs", {}).get("weights.npz", {}).get("sha256") == cohort["weights_sha256"]):
            bound_training = (receipt_path, receipt)
            break
    if bound_training is None:
        training_exposure_omissions.append(dict(weights_sha256=cohort["weights_sha256"], reason="matching training receipt unavailable"))
        continue
    receipt_path, receipt = bound_training
    splits_path = receipt_path.with_name("splits.json")
    if not splits_path.exists():
        training_exposure_omissions.append(dict(weights_sha256=cohort["weights_sha256"], reason="complete training split unavailable"))
        continue
    if not receipt_path.with_name("history.json").exists():
        training_exposure_omissions.append(dict(weights_sha256=cohort["weights_sha256"], reason="manufactured training accounting unavailable"))
        continue
    splits_source = relative_source(splits_path)
    splits = read(splits_source)
    expected_splits = receipt.get("outputs", {}).get("splits.json", {}).get("sha256") or receipt.get("closure", {}).get("inputs", {}).get("splits", {}).get("sha256")
    if not expected_splits or sources[splits_source] != expected_splits:
        raise ValueError(f"Training split is not bound to checkpoint: {splits_path}")
    training_domains = {}

    def training_domain(counts, spacing):
        key = (tuple(counts), spacing)
        if key not in training_domains:
            training_domains[key] = dict(counts_xyz=list(counts), cell_size_m=spacing,
                                         snapshots=set(), simulations=set(), manufactured_presentations=0)
        return training_domains[key]

    for parent in splits["parents"]:
        if parent["entry"]["split"] != "train":
            continue
        domain = training_domain(parent["counts_xyz"], parent["cell_size_m"])
        identity = (parent.get("capture_manifest_sha256"), parent["entry"]["trajectory_id"])
        domain["simulations"].add(identity)
        for frame in parent["frames"]:
            domain["snapshots"].add((*identity, frame["index"]))
    history_source = None
    history_path = receipt_path.with_name("history.json")
    if history_path.exists():
        history_source = relative_source(history_path)
        history = read(history_source)
        expected_history = receipt.get("outputs", {}).get("history.json", {}).get("sha256")
        if not expected_history or sources[history_source] != expected_history:
            raise ValueError(f"Training history is not bound to checkpoint: {history_path}")
        for key in ["analytical_feedback", "supplemental_feedback"]:
            feedback = history.get(key)
            if not feedback:
                continue
            grids, exposures = feedback.get("grids"), feedback.get("grid_exposures")
            # Earlier single-grid feedback banks store a single grid and count.
            if not grids and feedback.get("grid"):
                grids = [dict(grid=feedback["grid"])]
                exposures = [feedback["additional_examples"]]
            if not grids and feedback.get("schema") == "rheon-analytical-feedback-v1":
                # This version uses the sole fitting frame as the physical
                # template; its evidence records ZYX shape but omits spacing.
                templates = [d for d in training_domains.values() if d["snapshots"]]
                if (len(templates) == 1 and len(templates[0]["snapshots"]) == 1
                        and templates[0]["counts_xyz"] == list(reversed(feedback["shape_zyx"]))):
                    grids = [dict(counts_xyz=templates[0]["counts_xyz"], cell_size_m=templates[0]["cell_size_m"])]
                    exposures = [feedback["additional_examples"]]
            if not grids or exposures is None or len(grids) != len(exposures):
                raise ValueError(f"Manufactured training grid accounting unavailable: {history_path}")
            if sum(exposures) != feedback["additional_examples"]:
                raise ValueError(f"Manufactured training totals disagree: {history_path}")
            for grid, presentations in zip(grids, exposures):
                grid = grid.get("grid", grid)
                spacing = grid.get("cell_size_m") or 1 / grid["cells_per_metre"]
                training_domain(grid["counts_xyz"], spacing)["manufactured_presentations"] += presentations
    domain_counts = []
    for domain in training_domains.values():
        domain_counts.append(dict(counts_xyz=domain["counts_xyz"], cell_size_m=domain["cell_size_m"],
                                  unique_real_snapshots=len(domain["snapshots"]), real_simulations=len(domain["simulations"]),
                                  manufactured_presentations=domain["manufactured_presentations"]))
    training_exposures.append(dict(
        weights_sha256=cohort["weights_sha256"], source=splits_source,
        receipt_source=relative_source(receipt_path), history_source=history_source,
        domains=domain_counts, unique_real_snapshots=sum(d["unique_real_snapshots"] for d in domain_counts),
        real_simulations=sum(d["real_simulations"] for d in domain_counts),
        manufactured_presentations=sum(d["manufactured_presentations"] for d in domain_counts),
    ))
data["resolution_evaluations"] = resolution_evaluations
data["resolution_population_sha256"] = hashlib.sha256(json.dumps(resolution_population).encode()).hexdigest()
data["training_exposures"] = training_exposures
data["training_exposure_omissions"] = training_exposure_omissions

# Recover full time series from the exact trajectories referenced by drift seals.
# Warm-up frames never enter the normalized AI error baseline.
drift_cases = []
for path in sorted(ROOT.glob("water-*/reports/*drift.json")):
    recovered = path.with_name(path.name.replace("-drift.json", "-recovered-drift.json"))
    if recovered != path and recovered.exists():
        continue  # Prefer the explicitly corrected reference binding.
    report = read(str(path.relative_to(ROOT)))
    for case in report.get("cases", []):
        if case.get("source_report_sha256") and case.get("source_receipt_sha256"):
            drift_cases.append((str(path.relative_to(ROOT)), case))
needed_reports = {case["source_report_sha256"] for _, case in drift_cases}
trajectory_reports = {}
run_root = ROOT.parent.parent / "runs"
report_paths = set(run_root.glob("water-*/*/report.json")) | set(run_root.glob("water-*/*/*/report.json"))
for path in sorted(report_paths):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest in needed_reports:
        source = relative_source(path)
        sources[source] = digest
        trajectory_reports[digest] = (path, json.loads(raw))
traces, seen_traces, omitted_traces = [], set(), []
for seal_source, case in drift_cases:
    matched = trajectory_reports.get(case["source_report_sha256"])
    if not matched:
        omitted_traces.append(dict(source=seal_source, reason="source report unavailable"))
        continue
    report_path, report = matched
    row = next((r for r in report["records"] if r.get("receipt_sha256") == case["source_receipt_sha256"]), None)
    if row is None:
        raise ValueError(f"Drift seal has no matching trajectory record: {seal_source}")
    directory = report_path.parent / f"scene-{row['scene']:02d}" / f"{row['model']}-{row['repetition']}"
    receipt_path, frames_path = directory / "receipt.json", directory / "frames.jsonl"
    if not receipt_path.exists() or not frames_path.exists():
        omitted_traces.append(dict(source=seal_source, reason="full trajectory files unavailable"))
        continue
    receipt = read(relative_source(receipt_path))
    if sources[relative_source(receipt_path)] != case["source_receipt_sha256"]:
        raise ValueError(f"Trajectory receipt changed: {receipt_path}")
    if receipt["scene"] != case["scene"]:
        raise ValueError(f"Drift scene mismatch: {receipt_path}")
    model = next(m for m in report["models"] if m["name"] == row["model"])
    if model["weights_sha256"] != case["weights_sha256"]:
        raise ValueError(f"Drift checkpoint mismatch: {receipt_path}")
    raw_frames = frames_path.read_bytes()
    frames = [json.loads(line) for line in raw_frames.decode().splitlines() if line.strip()]
    if not all(f.get("reference_quality", {}).get("qualified") for f in frames):
        omitted_traces.append(dict(source=seal_source, reason="native reference failed qualification"))
        continue
    ai = [f for f in frames if f.get("ai_quality")]
    if len(ai) != case["ai_steps"] or not ai:
        raise ValueError(f"Incomplete AI frame population: {frames_path}")
    errors = [f["relative_pressure_difference"] for f in ai]
    if not all(math.isfinite(e) and e >= 0 for e in errors):
        raise ValueError(f"Invalid pressure drift: {frames_path}")
    if not math.isclose(max(errors), case["worst_pressure_relative_l2"], rel_tol=1e-12, abs_tol=1e-20):
        raise ValueError(f"Drift maximum differs from sealed result: {frames_path}")
    scene = receipt["scene"]
    if ai[0]["step"] <= scene["warmup_steps"]:
        raise ValueError(f"Native warm-up included in AI drift: {frames_path}")
    points = [[f["step"], f["time_s"], f["relative_pressure_difference"], bool(f["ai_quality"]["qualified"]), f["ai_quality"]["relative_residual"]] for f in ai]
    scene_key = hashlib.sha256(json.dumps(dict(scene=scene, binary=receipt["binary_sha256"]), sort_keys=True).encode()).hexdigest()[:20]
    policy = {k: receipt.get(k) for k in ["learned_proposals_per_pressure_solve", "adaptive_stopping", "proposal_network_evaluations"]}
    trace_key = hashlib.sha256(json.dumps([scene_key, case["weights_sha256"], policy, points], sort_keys=True).encode()).hexdigest()
    if trace_key in seen_traces:
        continue
    seen_traces.add(trace_key)
    frame_source = relative_source(frames_path)
    sources[frame_source] = hashlib.sha256(raw_frames).hexdigest()
    traces.append(dict(
        id=trace_key[:20], scene_key=scene_key, scene=scene, model=case["model"],
        weights_sha256=case["weights_sha256"], native_binary_sha256=receipt["binary_sha256"], policy=policy, points=points,
        completed=ai[-1]["step"] == scene["steps"] and not receipt["result"]["hard_stop"],
        hard_stop=receipt["result"]["hard_stop"], qualified_steps=sum(p[3] for p in points),
        source=seal_source, frames_source=frame_source, receipt_source=relative_source(receipt_path),
    ))
# These completed policy episodes carry their own receipt seals. Their model
# weights are reused, but stopping rules remain part of the trace identity.
def policy_evidence_key(scene_key, weights, policy, points):
    canonical = {k: v for k, v in policy.items() if v is not None and not (k == "minimum_network_evaluations" and v == 0)}
    return hashlib.sha256(json.dumps([scene_key, weights, canonical, points], sort_keys=True).encode()).hexdigest()


seen_policy_evidence = {policy_evidence_key(t["scene_key"], t["weights_sha256"], t["policy"], t["points"]) for t in traces}
for phase, report in policy_phases:
    measured_errors = {}
    for row in report["records"]:
        if row["model"] == "native":
            continue
        model = next(m for m in report["models"] if m["name"] == row["model"])
        directory = ROOT.parent.parent / row["output"] / "native"
        receipt_source = relative_source(directory / "receipt.json")
        receipt = read(receipt_source)
        if (sources[receipt_source] != row["receipt_sha256"]
                or receipt["binary_sha256"] != row.get("expected_binary_sha256", report["binary_sha256"])
                or not receipt["native_process"]["reaped"] or receipt["result"]["hard_stop"]
                or receipt["result"]["ai_steps"] != row["result"]["ai_steps"]):
            raise ValueError(f"Policy trajectory receipt mismatch: {receipt_source}")
        if (Path(receipt["cpu_restore"]).resolve() != (ROOT.parent.parent / model["artifacts"]).resolve()
                or Path(receipt["source_project"]).resolve() != (ROOT.parent.parent / model["project"]).resolve()):
            raise ValueError(f"Policy trajectory model mismatch: {receipt_source}")
        frames_source = relative_source(directory / "frames.jsonl")
        raw = (ROOT / frames_source).read_bytes()
        sources[frames_source] = hashlib.sha256(raw).hexdigest()
        frames = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        ai = [f for f in frames if f.get("ai_quality")]
        scene = receipt["scene"]
        if (len(ai) != row["result"]["ai_steps"] or not ai
                or ai[0]["step"] <= scene["warmup_steps"]
                or not all(f["reference_quality"]["qualified"] for f in frames)
                or not all(f["ai_quality"]["qualified"] for f in ai)):
            raise ValueError(f"Incomplete or unqualified policy trajectory: {frames_source}")
        points = [[f["step"], f["time_s"], f["relative_pressure_difference"], bool(f["ai_quality"]["qualified"]),
                   f["ai_quality"]["relative_residual"]] for f in ai]
        if not all(math.isfinite(p[2]) and p[2] >= 0 for p in points):
            raise ValueError(f"Invalid policy pressure error: {frames_source}")
        measured_errors.setdefault(model["name"], []).extend(p[2] for p in points)
        scene_key = hashlib.sha256(json.dumps(dict(scene=scene, binary=receipt["binary_sha256"]), sort_keys=True).encode()).hexdigest()[:20]
        policy = {k: receipt.get(k) for k in ["learned_proposals_per_pressure_solve", "adaptive_stopping",
                  "proposal_network_evaluations", "minimum_network_evaluations", "pressure_bound"]}
        trace_key = hashlib.sha256(json.dumps([scene_key, model["weights_sha256"], policy, points], sort_keys=True).encode()).hexdigest()
        evidence_key = policy_evidence_key(scene_key, model["weights_sha256"], policy, points)
        if evidence_key in seen_policy_evidence:
            continue
        seen_policy_evidence.add(evidence_key)
        seen_traces.add(trace_key)
        traces.append(dict(id=trace_key[:20], scene_key=scene_key, scene=scene, model=model["name"],
                           policy_label=model["name"], weights_sha256=model["weights_sha256"], native_binary_sha256=receipt["binary_sha256"], policy=policy,
                           points=points, completed=ai[-1]["step"] == scene["steps"], hard_stop=False,
                           qualified_steps=sum(p[3] for p in points), source=row.get("chart_source", phase["records_source"]),
                           frames_source=frames_source, receipt_source=receipt_source))
    for model in phase["models"]:
        errors = measured_errors[model["name"]]
        if (len(errors) != model["ai_steps"]
                or not math.isclose(max(errors), model["worst_paired_pressure_relative_l2"], rel_tol=1e-12)
                or not math.isclose(statistics.median(errors), model["median_paired_pressure_relative_l2"], rel_tol=1e-12)):
            raise ValueError(f"Policy pressure summary disagrees with frames: {phase['source']}/{model['name']}")
data["drift_traces"] = traces
data["drift_omissions"] = omitted_traces
data["drift_point_columns"] = ["physical_step", "simulation_time_s", "pressure_relative_l2", "qualified", "equation_relative_residual"]


def pressure_rms_frequency(counts, spacing, components):
    """Frequency of nonconstant cosine pressure modes, weighted by their energy.

    The recorded basis is cos(pi*k*(i+.5)/N). Its physical frequency is
    k/(2*N*h), and its mean-square value is 1/2 per nonzero axis. Combine
    duplicate modes coherently before squaring their coefficients; gauge/DC
    offsets do not describe spatial pressure variation.
    """
    coefficients = {}
    for frequencies, coefficient in components:
        if len(frequencies) != 3 or any(not isinstance(k, int) or k < 0 or k >= n for k, n in zip(frequencies, counts)):
            raise ValueError("Invalid recorded spatial mode")
        if not math.isfinite(coefficient):
            raise ValueError("Invalid recorded mode coefficient")
        key = tuple(frequencies)
        coefficients[key] = coefficients.get(key, 0) + coefficient
    energy = weighted_cycles_squared = 0.0
    for frequencies, coefficient in coefficients.items():
        if not any(frequencies):
            continue
        mode_energy = coefficient ** 2 * 0.5 ** sum(k != 0 for k in frequencies)
        cycles_squared = sum((k / (2 * n)) ** 2 for k, n in zip(frequencies, counts))
        energy += mode_energy
        weighted_cycles_squared += mode_energy * cycles_squared
    if energy <= 0:
        raise ValueError("Frequency undefined for a constant or cancelled pressure pattern")
    cycles_per_cell = math.sqrt(weighted_cycles_squared / energy)
    return cycles_per_cell / spacing, cycles_per_cell


# Preserve a common benchmark separately from pure modes or other probe seeds.
# All cases remain bound to their exact checkpoint and source report.
frequency_probes = []
known_weights = {c["weights_sha256"] for c in cohort_points}
probe_paths = set(run_root.glob("water-*/*/mode-probe/report.json")) | set(ROOT.glob("chart-diagnostics/*/*/mode-probe.json"))
for path in sorted(probe_paths):
    report = read(relative_source(path))
    if report.get("schema") != "rheon-multigrid-mixed-mode-report-v1" or report.get("weights_sha256") not in known_weights:
        continue
    if len(report["cases"]) != report["signed_cases"]:
        raise ValueError(f"Frequency probe population incomplete: {path}")
    cases, sample_keys = [], []
    for i, case in enumerate(report["cases"]):
        counts, spacing = case["counts_xyz"], case["cell_size_m"]
        if len(counts) != 3 or any(n <= 0 for n in counts) or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"Invalid probe grid: {path}")
        frequency_m, frequency_cell = pressure_rms_frequency(counts, spacing, case["components_xyz"])
        trace = [step["relative_residual"] for step in case["trace"]]
        if not trace or not all(math.isfinite(v) and v >= 0 for v in trace):
            raise ValueError(f"Invalid probe error trace: {path}")
        if (len(trace) != case["network_evaluations"]
                or [step["proposals"] for step in case["trace"]] != list(range(1, len(trace) + 1))
                or not math.isclose(trace[0], case["first_relative_residual"], rel_tol=1e-12)
                or not math.isclose(trace[-1], case["final_relative_residual"], rel_tol=1e-12, abs_tol=1e-20)):
            raise ValueError(f"Frequency probe trace disagrees with summary: {path}")
        sample = {k: case[k] for k in ["counts_xyz", "cell_size_m", "population", "components_xyz", "amplitude_pa"]}
        sample_keys.append(json.dumps(sample, sort_keys=True))
        cases.append(dict(
            **sample, id=hashlib.sha256(f"{relative_source(path)}:{i}".encode()).hexdigest()[:20],
            rms_cycles_per_m=frequency_m, rms_cycles_per_cell=frequency_cell,
            first_error=trace[0], final_error=case["final_relative_residual"], trace=trace,
            network_evaluations=case["network_evaluations"], equation_passed=case["equation_passed"],
        ))
    for group in report["aggregate"]:
        population = [c for c in cases if c["counts_xyz"] == group["counts_xyz"] and c["population"] == group["population"]]
        if (len(population) != group["cases"]
                or not math.isclose(max(c["first_error"] for c in population), group["worst_first_relative_residual"], rel_tol=1e-12)
                or not math.isclose(max(c["final_error"] for c in population), group["worst_final_relative_residual"], rel_tol=1e-12, abs_tol=1e-20)
                or not math.isclose(statistics.mean(c["network_evaluations"] for c in population), group["mean_network_evaluations"], rel_tol=1e-12)):
            raise ValueError(f"Frequency groups disagree with source: {path}")
    benchmark_key = hashlib.sha256(json.dumps([report["seed"], report["source_sha256"], sorted(sample_keys)]).encode()).hexdigest()
    frequency_probes.append(dict(weights_sha256=report["weights_sha256"], source=relative_source(path),
                                 benchmark_sha256=benchmark_key, seed=report["seed"], cases=cases))
data["frequency_probes"] = frequency_probes
# This separate artifact comes from CPU replay, not the old scalar-error logs.
spectral_path = ROOT / "pressure-spectrogram-data.json"
if spectral_path.exists():
    spectral = read(spectral_path.name)
    if spectral.get("schema") != "rheon-temporal-pressure-error-spectrogram-v1":
        raise ValueError("Unknown pressure spectrogram schema")
    if spectral["source_sha256"] != hashlib.sha256((ROOT / "build-pressure-spectrogram.py").read_bytes()).hexdigest():
        raise ValueError("Pressure spectrogram builder changed; rebuild its artifact")
    for scene in spectral["scenes"]:
        for kind in ("fields", "receipt", "frames"):
            source = scene[kind + "_source"]
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != scene[kind + "_sha256"]:
                raise ValueError(f"Pressure spectrogram source changed: {source}")
            sources[source] = scene[kind + "_sha256"]
    for model in spectral["models"]:
        if (model["weights_sha256"] not in known_weights
                or hashlib.sha256((ROOT / model["weights_source"]).read_bytes()).hexdigest() != model["weights_sha256"]
                or hashlib.sha256((ROOT / model["project_source"]).read_bytes()).hexdigest() != model["project_sha256"]):
            raise ValueError("Pressure spectrogram checkpoint identity changed")
        for scene in model["scenes"]:
            for spectrum in scene["spectra"]:
                if (len(spectrum["power"]) != len(spectrum["starts"])
                        or any(len(column) != len(spectrum["frequencies_hz"]) for column in spectrum["power"])
                        or any(not math.isfinite(v) or v < 0 for column in spectrum["power"] for v in column)):
                    raise ValueError("Malformed temporal power spectrum")
    data["pressure_spectrogram"] = spectral
enrich_drift_pressure(data, ROOT)
enrich_pass_sweep(data, ROOT)
enrich_precision(data, ROOT)
text = text[:match.start(2)] + json.dumps(data, separators=(",", ":")) + text[match.end(2):]
text = re.sub(r"Evidence snapshot · [^<]+", "Evidence snapshot · " + snapshot_date.strftime("%d %b %Y"), text)
PAGE.write_text(four_pass_link(precision_markup(text, ROOT), ROOT))
print(f"Updated {len(rows)} cohort/policy results across {len(set(r['group'] for r in rows))} study families; {sum(r.get('status') == 'pending' for r in rows)} pending.")
