"""Run the frozen IO controls using the source-bound native codec execution."""
import argparse
from pathlib import Path
import qualify_compact_io as qualification
from native_compact_codec import pack, unpack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    qualification.pack = pack
    qualification.unpack = unpack
    qualification.run(parser.parse_args())
