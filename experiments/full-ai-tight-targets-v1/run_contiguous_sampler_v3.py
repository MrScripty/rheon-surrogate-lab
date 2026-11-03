"""Select only the versioned compiled sampler; leave trained tensors unchanged."""
import argparse
from pathlib import Path
import restored_recurrent_sampler as restored
from recurrent_sampler_contiguous_v3 import library


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('qualify','benchmark','evaluate'),required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,choices=(23,),default=23)
    parser.add_argument('--evaluation-mode',choices=('blocks','rollout'),default='blocks')
    args=parser.parse_args();restored.library=library
    if args.mode=='qualify':
        from qualify_trained_recurrent_sampler import run
    elif args.mode=='benchmark':
        from benchmark_trained_sampler import run
    else:
        from evaluate_recurrent_sampler import run
        args.mode=args.evaluation_mode
    run(args)
