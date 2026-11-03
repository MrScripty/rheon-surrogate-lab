"""Reproducible physical scenes; splits, units and budgets are explicit."""
import math
import random
from policy import canonical, decode, experiment, require, sha


def specification(value):
    require(type(value) is dict and set(value)=={'schema','id','forcing_profile','seed','population','counts_range','cells_per_metre','steps','snapshot_steps','dt_s','material','force_ranges','training'},'dataset','closed water dataset')
    require(value['forcing_profile']=='compact-gaussian-taper-v1','force_profile','finite four-radius smooth force support')
    require(value['schema']=='rheon-sealed-water-dataset-v1' and value['id']=='sealed-water-forces-v1','dataset','named profile')
    require(type(value['seed']) is int and 0<=value['seed']<2**63,'seed','bounded reservation seed')
    require(type(value['population']) is dict and set(value['population'])=={'train','validation','reserved_test'}
            and all(type(n) is int and 1<=n<=256 for n in value['population'].values()),'population','whole-scene split budgets')
    require(value['counts_range']==[5,16] and value['cells_per_metre']==[8,16],'grids','qualified cubic sampling range')
    require(value['steps']==64 and value['snapshot_steps']==[15,63] and value['dt_s']==1/60,'time','declared bounded trajectory and samples')
    require(value['material']=={'density_kg_m3':1000.0,'dynamic_viscosity_pa_s':.001,'boundary':'sealed_impermeable_free_slip'},'water','fixed engineering water properties')
    ranges=value['force_ranges']
    require(set(ranges)=={'acceleration_m_s2','radius_fraction','fluctuation_fraction','frequency_hz'},'forces','closed range inventory')
    for key,low,high in [('acceleration_m_s2',.01,4),('radius_fraction',.1,.5),('fluctuation_fraction',0,.8),('frequency_hz',0,4)]:
        bounds=ranges[key];require(type(bounds) is list and len(bounds)==2 and all(type(n) in (int,float) and math.isfinite(n) for n in bounds)
            and low<=bounds[0]<=bounds[1]<=high,'force_range',key)
    experiment(value['training'])
    return decode(canonical(value))


def reserve(config):
    config=specification(config);rng=random.Random(config['seed']);rows=[];physical=set()
    for split,count in config['population'].items():
        for number in range(count):
            counts=[rng.randint(*config['counts_range']) for _ in range(3)]
            if number==0:counts=[16,16,16]
            elif number==1:counts=[16,10,6]
            density=config['cells_per_metre'][number%2]
            direction=[rng.uniform(-1,1) for _ in range(3)];norm=math.sqrt(sum(n*n for n in direction))
            direction=[n/norm for n in direction]
            ranges=config['force_ranges'];amplitude=math.exp(rng.uniform(math.log(ranges['acceleration_m_s2'][0]),math.log(ranges['acceleration_m_s2'][1])))
            force={'center_fraction':[rng.uniform(.2,.8) for _ in range(3)],'direction':direction,
                'acceleration_m_s2':amplitude,'radius_fraction':rng.uniform(*ranges['radius_fraction']),
                'swirl_fraction':(-1 if number%2 else 1)*rng.uniform(.15,1),
                'fluctuation_fraction':rng.uniform(*ranges['fluctuation_fraction']),
                'frequency_hz':rng.uniform(*ranges['frequency_hz']),'motion_fraction':rng.uniform(0,.12),
                'phase':rng.uniform(0,2*math.pi),'spatial_modes':[rng.randint(1,max(1,n//4)) for n in counts]}
            row={'trajectory_id':f'water-{split.replace("_","-")}-{number:04d}','seed':rng.randrange(2**53),
                'split':split,'counts_xyz':counts,'cells_per_metre':density,'force':force}
            identity=sha(canonical({key:row[key] for key in ('counts_xyz','cells_per_metre','force')}))
            require(identity not in physical,'duplicate_scene','actual setup identity');physical.add(identity);rows.append(row)
    cells=sum(math.prod(r['counts_xyz']) for r in rows if r['split']!='reserved_test')
    test_cells=sum(math.prod(r['counts_xyz']) for r in rows if r['split']=='reserved_test')
    require(cells*len(config['snapshot_steps'])*32<30*1024**2,'packet_budget','raw train/validation arrays below 30 MiB')
    require(test_cells*len(config['snapshot_steps'])*8<1900*1024,'test_predictions','bounded all-test numeric predictions')
    return {'schema':'rheon-sealed-water-reservation-v1','configuration':config,'scenes':rows}


def validate_reservation(value):
    require(type(value) is dict and set(value)=={'schema','configuration','scenes'} and value['schema']=='rheon-sealed-water-reservation-v1','reservation','closed versioned contract')
    require(value==reserve(value['configuration']),'reservation','exact reproducible scene/split inventory')
    return value
