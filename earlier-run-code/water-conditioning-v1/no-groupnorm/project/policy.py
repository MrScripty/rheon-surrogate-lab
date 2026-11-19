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
    version=value.get('schema')
    expected={'schema','id','training','resources','conditioning'}
    if version in ('rheon-water-conditioning-experiment-v2','rheon-water-conditioning-experiment-v3'):expected.add('execution')
    if version=='rheon-water-conditioning-experiment-v3':expected.add('normalization')
    require(set(value)==expected and version in ('rheon-water-conditioning-experiment-v1','rheon-water-conditioning-experiment-v2','rheon-water-conditioning-experiment-v3'),
            'conditioning_configuration','closed versioned experiment')
    require(value['conditioning'] in ('scalar-units','amplitude-and-units'),
            'conditioning_configuration','declared feature/output transformation')
    require(value.get('normalization','groupnorm') in ('groupnorm','none'),
            'normalization_configuration','explicit parameter-free block normalization intervention')
    original={k:v for k,v in value.items() if k not in ('conditioning','execution','normalization')}
    original['schema']='rheon-cuda-pressure-experiment-v1'
    base_experiment(original)
    execution(value)
    return decode(canonical(value))


def execution(config):
    selected=config.get('execution',{'resident_inputs':False,'finite_checks':'per-tensor','batch_mode':'serial','profile_updates':0})
    require(type(selected) is dict and set(selected)=={'resident_inputs','finite_checks','batch_mode','profile_updates'}
            and type(selected['resident_inputs']) is bool and selected['finite_checks'] in ('per-tensor','combined')
            and selected['batch_mode'] in ('serial','stacked') and type(selected['profile_updates']) is int
            and 0<=selected['profile_updates']<=20,'execution_configuration','closed measured execution strategy')
    require(not selected['profile_updates'] or config['training']['updates']>=12+selected['profile_updates'],
            'profile_budget','warm-up followed by complete capture window')
    return selected
