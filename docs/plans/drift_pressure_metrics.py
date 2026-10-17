"""Recover bounded pressure magnitudes without rerunning retained simulations.

The independent-trajectory recorder retained relative error e, but not the
reference norm R. Newer native diagnostics also retained same-request absolute
error D and relative error r, giving that native reference norm B = D/r.
Triangle inequalities give max(0, B-D)/(1+e) <= R <= (B+D)/(1-e).
These are estimates of independent drift, never the same-request error itself.
All pressures use the recorder's common cell-zero gauge; norms become RMS Pa
by dividing by sqrt(cell count). Missing or unbounded evidence stays missing.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


PRESSURE_COLUMNS = [
    "estimated_absolute_error_rms_pa", "estimated_reference_pressure_rms_pa",
    "reference_pressure_rms_lower_pa", "reference_pressure_rms_upper_pa",
]


def pressure_magnitudes(frame, counts):
    """Return (bounded estimate, reason); unavailable estimates are None."""
    comparison = (frame.get("diagnostic_attempt") or {}).get("same_request_pressure_comparison") or {}
    if frame.get("pressure_comparison_scope") != "independent_trajectories":
        return None, "independent comparison scope unavailable"
    if comparison.get("status") != "available":
        return None, "same-request norms not recorded"
    error = frame.get("relative_pressure_difference")
    distance, relative = comparison.get("l2"), comparison.get("relative_l2")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (error, distance, relative)):
        return None, "finite comparison norms unavailable"
    if not (0 <= error < 1 and distance > 0 and relative > 0):
        return None, "reference magnitude cannot be bounded from these norms"
    scale = math.sqrt(math.prod(counts))
    reference = distance / relative / scale
    distance /= scale
    lower = max(0, reference - distance) / (1 + error)
    upper = (reference + distance) / (1 - error)
    values = [error * reference, reference, lower, upper]
    if not all(math.isfinite(v) for v in values) or not lower <= reference <= upper:
        return None, "invalid magnitude bounds"
    return values, "bounded estimate"


def enrich_drift_pressure(data, base):
    """Attach aligned magnitude rows from sealed frames; preserve original points."""
    totals = Counter()
    for trace in data.get("drift_traces", []):
        source = trace["frames_source"]
        path = base / source
        expected = data["history_sources"].get(source)
        rows, reasons = [], Counter()
        if not path.is_file() or not expected:
            reasons["sealed frame source unavailable"] = len(trace["points"])
        else:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Drift frame source changed: {source}")
            frames = {f["step"]: f for line in raw.splitlines() if line.strip()
                      if (f := json.loads(line)).get("ai_quality")}
            for point in trace["points"]:
                frame = frames.get(point[0])
                if (frame is None or frame["time_s"] != point[1]
                        or frame["relative_pressure_difference"] != point[2]):
                    raise ValueError(f"Drift point differs from its retained frame: {source}, step {point[0]}")
                row, reason = pressure_magnitudes(frame, trace["scene"]["counts"])
                rows.append(row)
                reasons[reason] += 1
        available = reasons["bounded estimate"]
        # Omit all-null arrays, rather than representing missing measurements as zero.
        trace.pop("pressure_magnitudes", None)
        if available:
            trace["pressure_magnitudes"] = rows
        trace["pressure_magnitude_coverage"] = dict(reasons)
        totals.update(reasons)
    data["drift_pressure_columns"] = PRESSURE_COLUMNS
    data["drift_pressure_method"] = dict(
        kind="triangle_inequality_bounded_estimate", gauge="common cell-zero pressure gauge",
        units="Pa RMS over all cells", coverage=dict(totals),
        reference_estimate="B = same_request_l2 / same_request_relative_l2 / sqrt(cell_count)",
        reference_bounds="max(0, B-D)/(1+e) <= reference_rms <= (B+D)/(1-e), D = same_request_l2/sqrt(cell_count)",
        absolute_error="independent_relative_l2 * reference_rms; bounds scaled by the same relative error",
    )
    return dict(totals)


def main():
    """Refresh only these derived metrics in the current offline chart."""
    page = Path(__file__).with_name("cohort-progress.html")
    text = page.read_text()
    match = re.search(r'(<script id="chart-data" type="application/json">)(.*?)(</script>)', text, re.S)
    data = json.loads(match[2])
    totals = enrich_drift_pressure(data, page.parent)
    page.write_text(text[:match.start(2)] + json.dumps(data, separators=(",", ":"), allow_nan=False) + text[match.end(2):])
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
