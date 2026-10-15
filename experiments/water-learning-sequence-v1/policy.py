"""Current experiment policy with captured immutable CUDA implementation sources."""
from pathlib import Path
from support import MIB, canonical, decode, fresh, load_module, read, require, sha, write
HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]
MAX_CELLS=65536
source=HERE/'portable_policy.py'
if not source.exists():source=LAB/'experiments/cuda-pressure-v1/policy.py'
portable=load_module('water_portable_policy',source)
experiment=portable.experiment
frozen_model=portable.frozen_model
cubic_spacing=portable.cubic_spacing
dimensions=portable.dimensions
