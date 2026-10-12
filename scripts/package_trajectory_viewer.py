"""Create a compact display export; full numeric NPZ/JSON evidence stays intact."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    evaluation = Path(sys.argv[1])
    original = evaluation / "viewport.json"
    data = json.loads(original.read_text())
    threshold = 0.025
    for method in ("reference", "learned"):
        for frame in data[method]:
            frame["voxels"] = [[i, round(value, 5)] for i, value in enumerate(frame.pop("tracer"))
                               if value >= threshold]
    data["display"] = {"threshold": threshold, "opacity_decimal_places": 5,
                       "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                       "note": "Only display opacity is rounded; accuracy uses full numeric fields."}
    output = evaluation / "viewport-display.json"
    output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"Display export: {output.stat().st_size:,} bytes; original preserved")


if __name__ == "__main__":
    main()
