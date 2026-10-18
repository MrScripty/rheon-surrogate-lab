"""Use the bounded CPU owner with the bank-specific restored-weight evaluator."""
import argparse
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','controller','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('fitting_cpu_owner',ROOT/'experiments/pressure-single-mode-fit-v1/evaluate.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    owner.evaluate(args.config.resolve(),args.projects.resolve(),args.controller.resolve(),args.output.resolve(),
        fitted_script=HERE/'fitted.py')
