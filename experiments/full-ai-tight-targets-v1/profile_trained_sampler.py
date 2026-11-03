"""Profile the actual trained/restored graph with unchanged physical outputs."""
import argparse
from pathlib import Path
import profile_recurrent_sampler as profile
from restored_recurrent_sampler import load_model
from profile_frozen_components import read


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reg=read(args.protocol)
    profile.construct=lambda *unused:load_model(reg['pair'],reg['library_path'])
    profile.run(args)
