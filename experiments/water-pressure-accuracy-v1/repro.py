"""Exit nonzero when the paired same-input pressure regression remains."""
import argparse
from diagnose import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--candidate-policy', default='original')
    args = parser.parse_args()
    reference = read(args.reference)
    candidate = read(args.candidate)
    assert reference['sample_sha256'] == candidate['sample_sha256']
    before = reference['selected']['original']
    after = candidate['selected'][args.candidate_policy]
    passed = after is not None and after['pressure_relative_l2'] <= before['pressure_relative_l2']
    print({'same_sample_sha256': reference['sample_sha256'], 'criterion': 'no same-RHS pressure regression',
        'reference_error': before['pressure_relative_l2'],
        'candidate_error': after['pressure_relative_l2'] if after else None,
        'candidate_policy': args.candidate_policy, 'passed': passed})
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
