"""Evaluate a registered numerical execution policy with preserved learned provenance."""
import argparse
from pathlib import Path
from profile_frozen_components import read,write
import evaluate_recurrent_sampler as evaluation
import restored_recurrent_sampler as restored
from recurrent_sampler_contiguous_v3 import library
from sampler_precision_policies import load_model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reg=read(args.protocol);args.mode='rollout';restored.library=library
    evaluation.load_model=lambda pair,library_path:load_model(pair,library_path,reg['pressure_pair'],reg['precision_policy'])
    evaluation.run(args)
    result=read(args.output/'results.json');result.update(registered_precision_policy=reg['precision_policy'],
        native_pressure_solver_called=False,analytic_inverse_constructed=False,parameter_learning_executed=0,
        consumed_development_diagnostic_only=True,independent_target_pass=False)
    write(args.output/'results.json',result)
