"""Lossless selected-frame views over qualified water captures; no numeric conversion."""
import math
import numpy as np
import torch
from policy import HERE, LAB, MIB, canonical, decode, fresh, read, require, sha, write
import model
import scenes

FIELDS = ('base', 'reference', 'rhs', 'native_input_residual')
DATA = {'dataset': 'dataset.npz', 'dataset_extra': 'dataset-extra.npz', 'splits': 'splits.json'}

def helpers():return model.frozen.load_original('train',model.frozen.ORIGINAL_TRAIN_SHA256)

def inventory(arrays):
    return {name:{'shape':list(v.shape),'dtype':v.dtype.str,'bytes':v.nbytes,'sha256':sha(v.tobytes())} for name,v in arrays.items()}

def sample(parent,index,labels=True,device='cpu'):
    fields=FIELDS if labels else ('base','native_input_residual')
    result={name:torch.from_numpy(parent['arrays'][name][index:index+1].copy()).unsqueeze(1) for name in fields}
    result.update(cell_size_m=torch.tensor([parent['cell_size_m']],dtype=torch.float64),
        rho=torch.tensor([parent['density_kg_m3']],dtype=torch.float64),dt=torch.tensor([parent['entry']['dt_s']],dtype=torch.float64))
    return {name:value.to(device) for name,value in result.items()}

def fit_normalization(parents):
    trains=[p for p in parents if p['entry']['split']=='train'];require(trains,'training_population','train only')
    cells=sum(p['arrays']['base'].size for p in trains);scales={}
    for name,field,units in [('residual','native_input_residual','m3/s2'),('pressure','base','Pa'),('rhs','rhs','m3/s2')]:
        rms=math.sqrt(math.fsum(float(np.square(p['arrays'][field]).sum(dtype=np.float64)) for p in trains)/cells)
        require(math.isfinite(rms) and rms>0,'normalization','positive train-only RMS')
        scales[name]={'value':rms,'units':units,'method':'train_cell_rms','source':field}
    for name,key,units in [('rho','density_kg_m3','kg/m3'),('dt','dt_s','s')]:
        values=[p[key] if key in p else p['entry'][key] for p in trains for _ in p['frames']];constant=all(v==values[0] for v in values)
        scales[name]={'value':1.0 if constant else math.sqrt(math.fsum(v*v for v in values)/len(values)),
            'units':units,'method':'constant_unit_scale' if constant else 'train_frame_rms','source':name}
    return {'schema':'rheon-cubic-pressure-normalization-v2','model_profile':model.ARCHITECTURE['schema'],'channels':model.CHANNELS,
        'coordinates':'actual cell centres in metres; unit scale 1 m','scales':scales,
        'training_provenance':{'trajectory_ids':[p['entry']['trajectory_id'] for p in trains],'frames':sum(len(p['frames']) for p in trains),
            'cells':cells,'grids':[p['counts_xyz'] for p in trains],
            'parent_numeric_sha256':[{f:sha(p['arrays'][f].tobytes()) for f in FIELDS} for p in trains]}}


def load(raw, manifest):
    require(type(manifest) is dict and set(manifest) == {'schema', 'reservation', 'reservation_sha256', 'shards', 'parents', 'arrays', 'population'}, 'view_schema', 'closed selected water view')
    require(manifest['schema'] == 'rheon-water-learning-view-v1', 'view_schema', 'versioned lossless training view')
    reservation = manifest['reservation']
    require(set(reservation) == {'schema', 'sources', 'selection'} and reservation['schema'] == 'rheon-water-learning-reservation-v1', 'view_reservation', 'closed selection authority')
    require(sha(canonical(reservation)) == manifest['reservation_sha256'], 'view_reservation', 'exact measured selection')
    require(manifest['population'] in ('train_validation', 'validation', 'reserved_test'), 'view_population', 'explicit evaluation separation')
    sources = {}
    for name, source in reservation['sources'].items():
        require(set(source) == {'reservation', 'capture_manifest_sha256', 'packet_manifest_sha256'}, 'view_source', 'measured original capture/packet')
        original = scenes.validate_reservation(source['reservation'])
        sources[name] = {row['trajectory_id']: row for row in original['scenes']}
    require(len(raw) == len(manifest['shards']) == 2, 'view_shards', 'two bounded complete numeric inputs')
    arrays = {}
    for payload, shard in zip(raw, manifest['shards']):
        require(set(shard) == {'sha256', 'bytes', 'keys'} and len(payload) == shard['bytes'] and sha(payload) == shard['sha256'], 'view_shard_hash', 'whole input identity')
        require(not set(arrays).intersection(shard['keys']), 'view_inventory', 'no duplicate numeric keys')
        arrays.update(helpers().read_npz(payload, {key: manifest['arrays'][key] for key in shard['keys']}, 32*MIB))
    require(set(arrays) == set(manifest['arrays']), 'view_inventory', 'every declared field exactly once')
    parents = []; identities = set(); declared = set()
    for index, parent in enumerate(manifest['parents']):
        require(set(parent) == {'source', 'scene', 'entry', 'counts_xyz', 'cell_size_m', 'density_kg_m3', 'capture_root', 'capture_manifest_sha256', 'frames'}, 'view_parent', 'closed physical parent')
        source = parent['source']; row = sources[source][parent['scene']['trajectory_id']]
        source_config = reservation['sources'][source]['reservation']['configuration']
        expected_id = source + '--' + row['trajectory_id']
        require(parent['scene'] == row and parent['entry'] == {'trajectory_id': expected_id, 'split': row['split'], 'dt_s': source_config['dt_s']}, 'view_lineage', 'actual source scene and unique namespace')
        require(expected_id not in identities, 'view_duplicate', 'one entry per simulation'); identities.add(expected_id)
        require(parent['counts_xyz'] == row['counts_xyz'] and parent['cell_size_m'] == 1/row['cells_per_metre'] and parent['density_kg_m3'] == 1000.0, 'view_cubic', 'unchanged water and one spacing')
        require(parent['capture_manifest_sha256'] == reservation['sources'][source]['capture_manifest_sha256'], 'view_capture', 'same measured native source')
        allowed = {'train', 'validation'} if manifest['population'] == 'train_validation' else {manifest['population']}
        require(row['split'] in allowed, 'view_split', 'reserved parents never enter training/validation')
        chosen = [frame['index'] for frame in parent['frames']]
        require(chosen and chosen == sorted(set(chosen)) and all(i in source_config['snapshot_steps'] for i in chosen), 'view_frames', 'whole native snapshots with previous pressure')
        values = {}
        for field in FIELDS:
            key = f'p{index:04d}_{field}'; declared.add(key)
            spec = manifest['arrays'][key]
            require(spec['dtype'] == '<f8' and spec['shape'] == [len(chosen), *row['counts_xyz'][::-1]], 'view_shape', 'unaltered full-grid fields')
            values[field] = arrays[key]
        for slot, frame in enumerate(parent['frames']):
            require(set(frame) == {'index', 'sha256'} and set(frame['sha256']) == set(FIELDS), 'view_frame', 'closed original hash inventory')
            for field in FIELDS:
                require(sha(values[field][slot].tobytes()) == frame['sha256'][field], 'view_frame_hash', 'bit-exact original captured field')
        require(bool((values['base'][:,0,0,0] == 0).all() and (values['reference'][:,0,0,0] == 0).all()), 'view_gauge', 'cell-zero gauge retained')
        parents.append({**parent, 'arrays': values})
    require(declared == set(arrays), 'view_inventory', 'no extra numeric fields')
    expected = [{'source': p['source'], 'trajectory_id': p['scene']['trajectory_id'], 'frames': [f['index'] for f in p['frames']]} for p in parents]
    require(reservation['selection'] == expected, 'view_selection', 'all predeclared parent/frame selections')
    return reservation, parents


def inputs(directory):
    return {slot: read(directory/name, 4*MIB if slot == 'splits' else 32*MIB) for slot, name in DATA.items()}


def read_view(directory):
    values = inputs(directory)
    return load((values['dataset'], values['dataset_extra']), decode(values['splits']))
