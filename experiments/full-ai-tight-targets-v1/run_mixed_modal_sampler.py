"""Qualify, score or time the frozen managed mixed-query full-state composition."""
import argparse
from pathlib import Path
import restored_recurrent_sampler as restored
from recurrent_sampler_contiguous_v3 import library
from mixed_modal_sampler import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('qualify', 'evaluate', 'benchmark'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-mode', choices=('blocks', 'rollout'), default='rollout')
    parser.add_argument('--cpu', type=int, choices=(23,), default=23)
    args = parser.parse_args()
    restored.library = library
    if args.mode == 'qualify':
        import qualify_trained_recurrent_sampler as execution
    elif args.mode == 'evaluate':
        import evaluate_recurrent_sampler as execution
        args.mode = args.evaluation_mode
    else:
        import benchmark_trained_sampler as execution
    execution.load_model = load_model
    execution.run(args)
