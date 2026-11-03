"""Select the corrected admitted input for unchanged original F64 producers."""
import argparse

from review_water_learning_sequence import ROOT


def run(mode):
    if mode == 'benchmark':
        import run_same_case_F64_gpu_benchmark as owner
    else:
        import reduce_same_case_F64_gpu_benchmark as owner
        owner.OUTPUT = ROOT/'docs/plans/model-portfolio-migration-v1/reports/same-case-F64-GPU-benchmark-02.json'
    owner.PROTOCOL = ROOT/'runs/model-portfolio-migration-v1/same-case-F64-GPU-protocol-02.json'
    owner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('benchmark','reduce'))
    run(parser.parse_args().mode)
