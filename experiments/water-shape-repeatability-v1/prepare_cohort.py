"""Prepare a larger train-only cohort from an existing qualified native packet."""
import argparse
from pathlib import Path
import sys

from execution import LAB, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--count', type=int, choices=(32,64), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(LAB/'experiments/water-learning-sequence-v1'))
    import views
    selections = [('baseline', 'train', i, [0,1]) for i in range(args.count)]
    selections += [('baseline', 'validation', i, [0,1]) for i in range(64)]
    receipt = views.create({'baseline': args.packet.absolute()}, selections,
                           'train_validation', args.output.absolute())
    write(args.output/'preparation.json', {'status': 'prepared_not_trained',
        'training_simulations': args.count, 'training_states': 2*args.count,
        'validation_simulations': 64, 'validation_states': 128,
        'source_packet': str(args.packet.absolute()), 'receipt': receipt,
        'reserved_or_final_test_used': False,
        'scope': 'existing native captures only; future training requires a separate controlled configuration and train-only scales'})


if __name__ == '__main__':
    main()
