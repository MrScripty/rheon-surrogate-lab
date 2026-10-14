"""Archive exact original observations and publish a smaller lossless subset.

No model/data reader/Torch import or forward computation is performed. All384
rows and16 derivative cases are preserved; other stage statistics are aggregated
from already measured per-row values. The complete observations remain archived.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs/plans/global-pressure-diagnosis-v1/reports/diagnosis.json"
ARCHIVE = ROOT / "runs/global-pressure-v1-diagnosis/full-report.json"
FULL_SHA = "27a820e6486e1eb455d0006b5e7975ada689014e9e1d79b54721165f558e7d89"
ROW_STAGES = ("head:pre_conv", "decoders.3.layers.0:pre_conv", "decoders.3.layers.0:output",
              "decoders.3.layers.1:output", "decoders.3.layers.1:silu_derivative_f64")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def aggregate_stages(rows):
    result = {}
    for name in rows[0]["stages"]:
        values = [row["stages"][name] for row in rows]
        result[name] = {"rows": len(values), "min": min(v["min"] for v in values),
                        "max": max(v["max"] for v in values), "rms_max": max(v["rms"] for v in values),
                        "spatial_range_max": max(v["spatial_range_max"] for v in values),
                        "spatially_constant_rows": sum(v["spatial_range_max"] == 0 for v in values),
                        "zero_elements": sum(v["zero_count"] for v in values),
                        "elements": sum(v["elements"] for v in values)}
    return result


def compact(full):
    check = full["before"] == full["after"] and full["status"] == "complete"
    if not check:
        raise ValueError("complete immutable measurement proof required")
    result = {name: value for name, value in full.items() if name not in ("initial", "final")}
    result["schema"] = "rheon-global-pressure-frozen-diagnosis-compact-v1"
    for label in ("initial", "final"):
        measured = full[label]
        if len(measured["rows"]) != 384 or len(measured["derivatives"]) != 8:
            raise ValueError("fixed population/cases required")
        result[label] = {"aggregates": measured["aggregates"], "derivatives": measured["derivatives"],
                         "stage_aggregates": aggregate_stages(measured["rows"]),
                         "rows": [{**{key: value for key, value in row.items() if key != "stages"},
                                   "stages": {name: row["stages"][name] for name in ROW_STAGES}}
                                  for row in measured["rows"]]}
    return result


def main():
    started = time.perf_counter()
    original = REPORT.read_bytes()
    if sha(original) != FULL_SHA:
        raise ValueError("exact first observation required; refuses report overwrite")
    result = compact(json.loads(original))
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        if ARCHIVE.is_symlink() or ARCHIVE.read_bytes() != original:
            raise ValueError("immutable archive differs")
    else:
        with ARCHIVE.open("xb") as stream:
            stream.write(original)
    result["full_observation_archive"] = {"path": str(ARCHIVE.relative_to(ROOT)), "sha256": FULL_SHA,
                                          "bytes": len(original), "availability": "local ignored development evidence; regenerate via frozen diagnose.py if absent"}
    result["compaction"] = {"kind": "postprocess measured bytes only; no forward/gradient recomputation",
                             "script_sha256": sha(Path(__file__).read_bytes()), "per_row_retained_stages": list(ROW_STAGES),
                             "other_stage_detail": "complete per-row measurements preserved in exact full archive; compact report records aggregate ranges/zero counts",
                             "original_clock_scope": "original16.31s counter includes file proof/read/forward/derivative/witness and final hash checks; excludes imports and final JSON serialization/publication",
                             "original_rss_scope": "observed peak at end of scientific measurements, before final JSON serialization; not hostile aggregate containment"}
    # Include one serialization pass/archive publication in this separate clock.
    json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["compaction"]["elapsed_s_before_final_encoding_publication"] = time.perf_counter() - started
    result["compaction"]["observed_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if sha(REPORT.read_bytes()) != FULL_SHA:
        raise ValueError("report changed during compaction")
    # Parent explicitly authorized replacing this exact generated full report
    # with its compact derivative after preserving original bytes in ignored runs.
    with tempfile.NamedTemporaryFile(dir=Path(__file__).parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        os.replace(temporary, REPORT)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"report_sha256": sha(raw), "bytes": len(raw), "full_archive_sha256": FULL_SHA,
                      "total_postprocessing_s": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
