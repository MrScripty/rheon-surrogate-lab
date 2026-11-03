"""Run the frozen managed campaign, then evaluate all restored candidates."""
import argparse
from pathlib import Path
import sys

from campaign import run, process, DIRECTORY, HERE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-file', required=True, type=Path)
    parser.add_argument('--resume-before-admission', action='store_true')
    args = parser.parse_args()
    run(args.session_file.resolve(), resume_before_admission=args.resume_before_admission)
    process([sys.executable, '-B', HERE / 'evaluate.py'], DIRECTORY / 'evaluation.log', timeout=5400)
