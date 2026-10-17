#!/usr/bin/env python3
"""Prepare current water-trainer material without starting a catalog or Run."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dependency-migration-v1'))
from dependency_config import load
from dependency_training import assemble


def scientific_environment():
    os.umask(0o077)
    os.environ.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')})
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scientific_environment()
    print(json.dumps(assemble(load(args.dependency_config), args.output)))


if __name__ == '__main__':
    main()
