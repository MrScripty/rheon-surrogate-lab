"""Observe native left-fold versus Python's float-sum direction normalization."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/full-ai-tight-targets-v1'))
import profile_frozen_components
from temporal_features_v2 import direction_phase as prior
from force_direction import direction_phase


if __name__ == '__main__':
    bank = json.loads((ROOT / 'runs/full-ai-tight-targets-v1/pressure-history-fresh-reference-12/bank.json').read_text())
    for row in bank['trajectories']:
        scene = json.loads((ROOT / row['receipt']).read_text())['scene']
        old, old_phase = prior(scene['seed'])
        new, new_phase = direction_phase(scene['seed'])
        if old != new or old_phase != new_phase:
            print(row['identifier'], 'old', [v.hex() for v in old], 'left_fold', [v.hex() for v in new], flush=True)
