"""Current experiment policy with captured immutable CUDA implementation sources."""
from pathlib import Path
from support import MIB, canonical, decode, fresh, load_module, read, require, sha, write
HERE=Path(__file__).resolve().parent
LAB=HERE.parents[2]
MAX_CELLS=65536
source=HERE/'portable_policy.py'
if not source.exists():source=LAB/'experiments/cuda-pressure-v1/policy.py'
portable=load_module('water_portable_policy',source)
base_experiment=portable.experiment
frozen_model=portable.frozen_model
cubic_spacing=portable.cubic_spacing
dimensions=portable.dimensions


def experiment(value):
    require(type(value) is dict,'conditioning_configuration','object required')
    # Existing physical reservations embed the original training specification.
    if value.get('schema') == 'rheon-cuda-pressure-experiment-v1':
        return base_experiment(value)
    require(type(value) is dict and set(value)=={'schema','id','training','resources','conditioning'}
            and value['schema']=='rheon-water-conditioning-experiment-v1',
            'conditioning_configuration','closed versioned experiment')
    require(value['conditioning'] in ('scalar-units','amplitude-and-units'),
            'conditioning_configuration','declared feature/output transformation')
    original={k:v for k,v in value.items() if k!='conditioning'}
    original['schema']='rheon-cuda-pressure-experiment-v1'
    base_experiment(original)
    return decode(canonical(value))
