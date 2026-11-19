"""Keep the primary bank unchanged; balance only supplemental spatial modes."""
import json
import math
from pathlib import Path
import numpy as np
import torch
import original_analytical as original
from policy import canonical,require,sha


def __getattr__(name):
    return getattr(original,name)


def patterns(shape):
    fundamental=[(0,0,1),(0,1,0),(1,0,0)]*4
    low=[(2,0,0),(0,2,0),(0,0,2),(3,0,0),(0,3,0),(0,0,3),
         (1,1,0),(1,0,1),(0,1,1),(1,1,1),(1,2,3),(3,2,1)]
    z,y,x=(n-1 for n in shape)
    high=[(4,0,0),(0,4,0),(0,0,4),(4,4,4),(z,0,0),(0,y,0),(0,0,x),(z,y,x)]
    return [(kind,mode)for kind,group in [('fundamental',fundamental),('other_low',low),('high',high)]for mode in group]


def supplemental_bank(template,specification,updates,primary,primary_spec,amplitude_pa):
    curriculum=json.loads(Path(__file__).with_name('mode-curriculum.json').read_bytes())
    require(curriculum=={'seed':specification['seed'],'fundamental_patterns':12,'other_low_patterns':12,
        'high_patterns':8,'max_low_mode':3},'frequency_curriculum','closed 12/12/8 mode curriculum')
    require(specification['source']=='grids' and specification['patterns']==32,
        'frequency_curriculum','32 independent supplemental patterns')
    raw={k:v for k,v in specification.items()if k!='source'}
    projected=original.validate_grids(raw)
    parent=primary[2]
    require(parent['resident_tensor_bytes']+projected<=min(primary_spec['cache_max_bytes'],raw['cache_max_bytes']),
        'analytical_cache','combined primary and supplemental residency')
    require(math.isfinite(amplitude_pa) and amplitude_pa>0,'analytical_amplitude','original training RMS')
    order,grid_order=original._balanced_order(len(raw['grids']),128,updates,raw['seed'])
    samples=[];banks=[];categories=[]
    for grid in raw['grids']:
        shape=tuple(reversed(grid['counts_xyz']))
        require(min(shape)>=5,'frequency_curriculum','all selected modes fit the grid')
        local={**template,'base':torch.zeros((1,1,*shape),dtype=torch.float64,device=template['base'].device),
            'cell_size_m':torch.tensor([1/grid['cells_per_metre']],dtype=torch.float64,device=template['base'].device)}
        descriptions=[]
        for category,mode in patterns(shape):
            for multiplier in original.AMPLITUDES:
                amplitude=amplitude_pa*multiplier
                samples.append(original.make_sample(local,[(mode,1.)],amplitude));categories.append(category)
                descriptions.append(dict(components_zyx=[(mode,1.)],amplitude_pa=amplitude,category=category))
        banks.append(dict(grid=grid,samples=descriptions))
    actual=sum(v.numel()*v.element_size()for sample in samples for v in sample.values())
    require(actual==projected,'analytical_cache','exact tensor residency')
    exposures={category:sum(categories[int(index)]==category for index in order)
        for category in ['fundamental','other_low','high']}
    evidence=dict(schema='rheon-supplemental-feedback-v1',source_bank_schema='rheon-frequency-curriculum-v1',
        source='Original train-only RMS and common water scalars; explicit cubic grids and 12/12/8 spatial modes',
        grids=banks,sampling=raw['sampling'],grid_exposures=np.bincount(grid_order,minlength=len(banks)).tolist(),
        sampling_order_sha256=sha(order.tobytes()),bank_description_sha256=sha(canonical(banks)),
        loss_weights=original.LOSS_WEIGHTS,weight=raw['weight'],additional_examples=updates,
        resident_tensor_bytes=actual,cache_max_bytes=raw['cache_max_bytes'],reuses_primary=False,seed=raw['seed'],
        combined_resident_tensor_bytes=parent['resident_tensor_bytes']+actual,
        primary_bank_description_sha256=parent['bank_description_sha256'],
        amplitude_reference=raw['amplitude_reference'],amplitude_rms_pa=amplitude_pa,
        curriculum=curriculum,actual_mode_category_exposures=exposures,
        scope='Supplemental manufactured operator examples; not extra CFD trajectories or new normalization')
    return samples,order,evidence
