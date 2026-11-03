"""One physical cell size; domain lengths determine integer per-axis counts."""
import math
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/local-pressure-v1'))
from contract import canonical, decode, experiment, read, require, reservation, sha


def counts(extents, cells_per_metre):
    require(type(cells_per_metre) is int and 1<=cells_per_metre<=128,'resolution','integer cells per metre, 1–128')
    require(type(extents) is list and len(extents)==3,'domain','three physical lengths in metres')
    result=[]
    for length in extents:
        require(type(length) in (int,float) and math.isfinite(length) and length>0,'domain','finite positive length')
        exact=length*cells_per_metre;count=round(exact)
        require(math.isclose(exact,count,rel_tol=0,abs_tol=1e-10),'cell_multiple',f'Length {length} m must be a whole multiple of {1/cells_per_metre} m; do not stretch cells')
        require(2<=count<=128,'cell_count','2–128 cells per axis')
        result.append(count)
    require(math.prod(result)<=65536,'cell_budget','at most 65,536 cells')
    return result


def config(value):
    require(type(value) is dict and set(value)=={'schema','cells_per_metre','experiment'} and value['schema']=='rheon-cubic-pressure-v1','cubic_schema','explicit density and experiment')
    original=experiment(value['experiment'])
    for extents in original['geometries_extents_m'].values():counts(extents,value['cells_per_metre'])
    return value


def plan(value):
    value=config(value);density=value['cells_per_metre'];original=value['experiment']
    grids=[]
    for entry in original['trajectories']:
        extents=original['geometries_extents_m'][entry['geometry']]
        dimensions=counts(extents,density)
        grids.append({**entry,'counts_xyz':dimensions,'extents_m':extents,'spacing_m':[1/density]*3,
                      'model_shape_compatible':dimensions in ([16]*3,[32]*3)})
    return {'cells_per_metre':density,'cell_size_m':1/density,'grids':grids,
            'model_shape_compatible':all(g['model_shape_compatible'] for g in grids)}


def native_reservation(value):
    value=config(value)
    return {**reservation(value['experiment']),'cells_per_metre':value['cells_per_metre']}
