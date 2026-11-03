"""Evaluate/benchmark an exact managed refined-pressure composition, no numerical solve."""
import argparse
from pathlib import Path
from profile_frozen_components import read
import restored_recurrent_sampler as restored
from recurrent_sampler_contiguous_v3 import library
from refined_pressure_sampler import load_model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('qualify','evaluate','benchmark'),required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluation-mode',choices=('blocks','rollout'),default='blocks')
    parser.add_argument('--cpu',type=int,choices=(23,),default=23)
    args=parser.parse_args();reg=read(args.protocol);restored.library=library
    if args.mode=='qualify':import qualify_trained_recurrent_sampler as execution
    elif args.mode=='benchmark':import benchmark_trained_sampler as execution
    else:
        import evaluate_recurrent_sampler as execution
        args.mode=args.evaluation_mode
    execution.load_model=lambda pair,library_path:load_model(pair,library_path,reg['pressure_pair'])
    execution.run(args)
