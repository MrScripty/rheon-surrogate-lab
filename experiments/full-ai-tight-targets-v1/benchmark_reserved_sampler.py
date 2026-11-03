"""Same complete-block clock, explicitly labelled frozen reserved confirmation."""
import argparse
from pathlib import Path
from profile_frozen_components import read,write
from benchmark_trained_sampler import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,choices=(23,),required=True)
    args=parser.parse_args();run(args)
    result=read(args.output);result.update(fresh_holdout_consumed=True,
        reserved_confirmation_of_frozen_candidate=True,no_model_selection_using_reserved_results=True)
    write(args.output,result)
