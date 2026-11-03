"""Original local sampler groups and physical metadata; no model execution."""
import math
from pathlib import Path

import numpy as np

from import_common_generator_measurements import ROOT, read, require
from component_material import observed_file

FIELDS = ('point','patch','kind','alpha','middle','end')


def load_groups(entry):
    manifest = read(ROOT/entry['inputs']['manifest']['path'])
    require(manifest['status'] == 'verified_train_only_sampler_view' and manifest['scenes'] == 16
            and manifest['points'] == 77998 and not manifest['development_or_reserved_used'], 'Original complete sampler population required')
    scenes = {}
    for name,sha in manifest['source_evidence'].items():
        if not name.endswith('/receipt.json'):
            continue
        observed_file(ROOT,name,dict(sha256=sha));receipt = read(ROOT/name)
        require(receipt['status'] == 'completed' and receipt['split'] == 'train'
                and receipt['identifier'] not in scenes, 'Original unique native training scene required')
        require(receipt['states'][1]['time_s']-receipt['states'][0]['time_s'] == 1/60., 'Original native timestep differs')
        scenes[receipt['identifier']] = receipt
    require(len(scenes) == 16, 'All16scene metadata owners required')
    groups = []
    for shard in manifest['shards']:
        candidates = [spec for key,spec in entry['inputs'].items() if key.startswith('part_') and Path(spec['path']).name == shard['path']]
        require(len(candidates) == 1 and candidates[0]['sha256'] == shard['sha256'], 'Exact original shard binding required')
        observed_file(ROOT,candidates[0]['path'],candidates[0])
        with np.load(ROOT/candidates[0]['path'],allow_pickle=False) as archive:
            require(set(archive.files) == set(FIELDS), 'Original closed input/target inventory required')
            arrays = {name:archive[name] for name in FIELDS}
        count = shard['points']
        shapes = dict(point=(count,3),patch=(count,3,3,3,3),kind=(count,),alpha=(count,),middle=(count,3),end=(count,3))
        for name,value in arrays.items():
            dtype = '<f4' if name == 'patch' else '<i8' if name == 'kind' else '<f8'
            require(value.dtype.str == dtype and value.shape == shapes[name] and np.isfinite(value).all(), 'Exact original finite sampler array types/shapes required')
        start = 0
        for sample in shard['samples']:
            stop = start+sample['points'];receipt = scenes[sample['identifier']];scene = receipt['scene']
            require(0 <= start < stop <= count and scene['counts'] == shard['counts_xyz']
                    and scene['cells_per_metre'] in (8,16) and sample['kind'] in range(4), 'Original group/geometry metadata differs')
            state = receipt['states'][sample['state']]
            require(state['step'] == sample['state'] and math.isclose(state['time_s'],sample['state']/60.,rel_tol=1e-14,abs_tol=1e-14),
                    'Original native frame cadence differs')
            selected = {name:value[start:stop] for name,value in arrays.items()}
            require(np.all(selected['kind'] == sample['kind'])
                    and np.all(selected['alpha'] == scene['cells_per_metre']/60.), 'Original carrier/timestep-ratio alignment differs')
            identity = dict(shard_key=shard['key'],identifier=sample['identifier'],state=sample['state'],kind=sample['kind'],
                start=start,stop=stop,points=sample['points'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                input_dt_s=1/60.,alpha=scene['cells_per_metre']/60.)
            groups.append(dict(identity=identity,arrays=selected));start = stop
        require(start == count, 'Ordered groups must exhaust each original shard exactly')
    require(len(groups) == 576 and sum(row['identity']['points'] for row in groups) == 77998, 'All576groups/77998points required')
    return groups
