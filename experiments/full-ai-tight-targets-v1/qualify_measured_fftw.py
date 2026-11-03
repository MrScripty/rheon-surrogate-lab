"""Run the frozen transform controls with bounded measured-plan setup."""
import argparse
from pathlib import Path
import qualify_fftw_pressure as qualification
from fftw_pressure_transform_v2 import MeasuredDCTMultiplier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    qualification.DCTMultiplier = MeasuredDCTMultiplier
    qualification.run(parser.parse_args())
