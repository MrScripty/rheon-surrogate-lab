"""Qualify the same graph with an inference-context-safe constructor boundary."""
import argparse
from pathlib import Path
import qualify_recurrent_sampler as qualification
from recurrent_sampler_constructor import construct


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    qualification.RecurrentSamplerPrototype=construct
    qualification.run(parser.parse_args())
