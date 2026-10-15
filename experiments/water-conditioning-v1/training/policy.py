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
    if value.get('schema') in ('rheon-water-residual-experiment-v1','rheon-water-residual-experiment-v2','rheon-water-residual-experiment-v3','rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):
        expected={'schema','id','training','resources','conditioning','normalization','execution','correction'}
        if value['schema'] in ('rheon-water-residual-experiment-v2','rheon-water-residual-experiment-v3','rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):expected.add('objective')
        if value['schema'] in ('rheon-water-residual-experiment-v3','rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):expected.update(('analytical_feedback','output_scale'))
        require(set(value)==expected,
                'residual_configuration','closed residual experiment')
        require(value['conditioning']=='physical-residual' and value['normalization']=='none',
                'residual_configuration','one physically scaled residual channel without normalization')
        correction=value['correction']
        require(type(correction) is dict and set(correction)=={'activation','steps'}
                and correction['activation'] in ('silu','linear') and type(correction['steps']) is int
                and 1<=correction['steps']<=4,'residual_correction','explicit activation and one to four shared updates')
        original={k:value[k] for k in ('id','training','resources')}
        original['schema']='rheon-cuda-pressure-experiment-v1'
        base_experiment(original)
        execution(value)
        if value['schema'] in ('rheon-water-residual-experiment-v3','rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):
            feedback=value['analytical_feedback']
            feedback_keys={'weight','seed','patterns'}
            if value['schema'] in ('rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):
                feedback_keys.update(('grids','sampling','cache_max_bytes'))
            if value['schema']=='rheon-water-residual-experiment-v5':feedback_keys.add('amplitude_reference')
            require(type(feedback) is dict and set(feedback)==feedback_keys
                    and type(feedback['weight']) in (int,float) and 0<=feedback['weight']<=1
                    and type(feedback['seed']) is int and 0<=feedback['seed']<2**32
                    and type(feedback['patterns']) is int and 8<=feedback['patterns']<=64,
                    'analytical_feedback','bounded weight, seed and analytical pattern count')
            if value['schema'] in ('rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):
                import analytical
                analytical.validate_grids(feedback)
            if value['schema']=='rheon-water-residual-experiment-v5':
                require(feedback['amplitude_reference']=='train-correction-cell-rms',
                    'analytical_amplitude_reference','all real train cell-states, no validation statistics')
            require(value['output_scale'] in ('previous-pressure-rms','correction-rms')
                    and not value['execution']['prepared_fields'],
                    'residual_scale','explicit output units through ordinary prediction path')
            require(not feedback['weight'] or value['execution']['resident_inputs']
                    and value['execution']['batch_mode']=='serial' and value['training']['batch_size']==1,
                    'analytical_feedback','resident serial fitting; one real sample per update')
        if 'objective' in value:
            require(value['objective'] in ('canonical','correction-gradient-divergence'),
                    'residual_objective','explicit independently compared physical feedback')
            require(value['objective']=='canonical' or not value['execution']['prepared_fields'],
                    'residual_objective','changed objective uses its canonical loss path')
        return decode(canonical(value))
    # Existing physical reservations embed the original training specification.
    if value.get('schema') == 'rheon-cuda-pressure-experiment-v1':
        return base_experiment(value)
    version=value.get('schema')
    expected={'schema','id','training','resources','conditioning'}
    if version in ('rheon-water-conditioning-experiment-v2','rheon-water-conditioning-experiment-v3','rheon-water-conditioning-experiment-v4'):expected.add('execution')
    if version in ('rheon-water-conditioning-experiment-v3','rheon-water-conditioning-experiment-v4'):expected.add('normalization')
    require(set(value)==expected and version in ('rheon-water-conditioning-experiment-v1','rheon-water-conditioning-experiment-v2','rheon-water-conditioning-experiment-v3','rheon-water-conditioning-experiment-v4'),
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
    expected={'resident_inputs','finite_checks','batch_mode','profile_updates'}
    if config['schema'] in ('rheon-water-conditioning-experiment-v4','rheon-water-residual-experiment-v1','rheon-water-residual-experiment-v2','rheon-water-residual-experiment-v3','rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):expected.add('prepared_fields')
    require(type(selected) is dict and set(selected)==expected
            and type(selected['resident_inputs']) is bool and selected['finite_checks'] in ('per-tensor','combined')
            and selected['batch_mode'] in ('serial','stacked') and type(selected['profile_updates']) is int
            and 0<=selected['profile_updates']<=20,'execution_configuration','closed measured execution strategy')
    if 'prepared_fields' in selected:
        require(type(selected['prepared_fields']) is bool and (not selected['prepared_fields'] or
                selected['resident_inputs'] and selected['batch_mode']=='serial'),
                'prepared_inputs','prepared fields require resident serial execution')
    if config['schema'] in ('rheon-water-residual-experiment-v1','rheon-water-residual-experiment-v2','rheon-water-residual-experiment-v3','rheon-water-residual-experiment-v4','rheon-water-residual-experiment-v5'):
        require(not selected['prepared_fields'] or config['correction']['steps']==1,
                'prepared_inputs','recurrent residuals must be recomputed through the ordinary model path')
    require(not selected['profile_updates'] or config['training']['updates']>=12+selected['profile_updates'],
            'profile_budget','warm-up followed by complete capture window')
    return selected
