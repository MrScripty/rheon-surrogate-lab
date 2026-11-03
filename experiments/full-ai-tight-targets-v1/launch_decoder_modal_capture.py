"""Establish the existing qualified experiment import path before native capture."""
import argparse
from pathlib import Path
import profile_frozen_components
from capture_decoder_modal_reserved import capture


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-selection',type=Path,required=True)
    capture(parser.parse_args())
