"""Evaluate the frozen learned causal pressure-reuse generator on complete states."""
import argparse
from pathlib import Path
import profile_frozen_components
from warm_pressure_sampler import load_model
from recurrent_sampler_contiguous_v3 import library
import restored_recurrent_sampler as restored
import evaluate_recurrent_sampler as evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('blocks', 'rollout'), required=True)
    args = parser.parse_args()
    restored.library = library
    evaluation.load_model = load_model
    evaluation.run(args)
