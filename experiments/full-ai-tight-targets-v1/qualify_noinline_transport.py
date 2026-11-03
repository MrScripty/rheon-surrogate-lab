"""Reuse the complete boundary/extreme donor-address checks for call-boundary v5."""
import argparse
from pathlib import Path
import qualify_contiguous_transport as qualification
from recurrent_sampler_noinline_v5 import library


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualification.candidate_library=library;qualification.run(args)
