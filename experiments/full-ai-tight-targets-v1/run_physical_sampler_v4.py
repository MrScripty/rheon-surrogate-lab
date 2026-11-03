"""Versioned physical-coordinate arithmetic; original learned tensors preserved."""
import argparse
from pathlib import Path
import restored_recurrent_sampler as restored
from recurrent_sampler_physical_v4 import library


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('qualify','evaluate'),required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluation-mode',choices=('blocks','rollout'),default='blocks')
    args=parser.parse_args();restored.library=library
    if args.mode=='qualify':
        import qualify_recurrent_sampler as tensor_qualification
        import recurrent_sampler_physical_reference as reference
        tensor_qualification.reference=reference
        from qualify_trained_recurrent_sampler import run
    else:
        from evaluate_recurrent_sampler import run
        args.mode=args.evaluation_mode
    run(args)
