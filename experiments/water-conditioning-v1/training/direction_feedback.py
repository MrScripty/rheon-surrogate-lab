"""Captured train-only early/late correction directions; never an inference input."""
import math
import numpy as np
import torch
from policy import canonical, decode, require, sha, MAX_CELLS
import data

SLOTS = {'direction_dataset': 'direction.npz', 'direction_manifest': 'direction.json'}


def specification(value):
    require(type(value) is dict and set(value) == {'stage', 'weight', 'seed', 'manifest_sha256'},
            'direction_feedback', 'closed captured direction selection')
    require(value['stage'] in ('early', 'late') and type(value['weight']) in (int, float)
            and math.isfinite(value['weight']) and 0 < value['weight'] <= 1
            and type(value['seed']) is int and 0 <= value['seed'] < 2**32
            and type(value['manifest_sha256']) is str and len(value['manifest_sha256']) == 64
            and all(c in '0123456789abcdef' for c in value['manifest_sha256']),
            'direction_feedback', 'explicit paired stage, finite weight, private seed and manifest identity')
    return value


def bank(inputs, spec, normalization, train_ids, updates, device, resident_bytes, maximum_bytes):
    specification(spec)
    raw = inputs['direction_manifest']
    require(sha(raw) == spec['manifest_sha256'], 'direction_manifest', 'captured paired bank identity')
    manifest = decode(raw)
    require(manifest['schema'] == 'rheon-pressure-direction-bank-v1'
            and set(manifest['stages']) == {'early', 'late'}
            and manifest['source_corrections'] == 8 and manifest['optimizer_updates'] == 0
            and set(manifest['train_trajectory_ids']) == set(train_ids)
            and manifest['packet_splits_sha256'] == sha(inputs['splits'])
            and manifest['source']['normalization_sha256'] == sha(canonical(normalization)),
            'direction_lineage', 'same train-only packet, fixed source policy and original normalization')
    declared = manifest['stages'][spec['stage']]
    numeric = inputs['direction_dataset']
    require(len(numeric) == declared['bytes'] and sha(numeric) == declared['sha256'],
            'direction_payload', 'selected stage bytes')
    arrays = data.helpers().read_npz(numeric, declared['arrays'], 32*1024**2)
    samples, groups = [], {}
    expected = set()
    for index, case in enumerate(manifest['cases']):
        identity = case['identity']
        require(identity['kind'] in ('real', 'manufactured') and
                (identity['kind'] != 'real' or identity['trajectory_id'] in train_ids),
                'direction_lineage', 'training parents only; manufactured fields labelled')
        counts = case['counts_xyz']
        require(type(counts) is list and len(counts) == 3
                and all(type(n) is int and 2 <= n <= 128 for n in counts)
                and math.prod(counts) <= MAX_CELLS, 'direction_shape', 'bounded cubic grid counts')
        fields = {}
        for field in ('rhs', 'reference'):
            key = f'case_{index:03d}_{field}'
            expected.add(key)
            value = arrays[key]
            require(value.dtype.str == '<f8' and list(value.shape) == [1, 1, *counts[::-1]]
                    and np.isfinite(value).all(), 'direction_shape', 'finite full-grid float64 physical fields')
            fields[field] = torch.from_numpy(value.copy()).to(device)
        require(float(fields['reference'].flatten()[0]) == 0
                and float(fields['rhs'].square().sum()) > 0, 'direction_label', 'nonzero equation and cell-zero correction gauge')
        scalars = {}
        for key in ('rho', 'dt', 'cell_size_m'):
            value = case[key]
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    'direction_physics', 'positive finite per-sample scalar')
            scalars[key] = torch.tensor([value], dtype=torch.float64, device=device)
        require(case['stages'][spec['stage']]['operator_relative_error'] < 1e-9,
                'direction_label', 'qualified independent labels')
        samples.append({**fields, **scalars, 'base': torch.zeros_like(fields['reference']),
                        'native_input_residual': fields['rhs']})
        groups.setdefault(identity['group'], []).append(index)
    require(samples and expected == set(arrays), 'direction_inventory', 'all selected fields exactly once')
    actual = sum(t.numel()*t.element_size() for s in samples for t in s.values())
    require(resident_bytes+actual <= maximum_bytes, 'direction_cache', 'combined resident feedback memory')
    rng = np.random.Generator(np.random.PCG64(spec['seed']))
    names = sorted(groups)
    group_order = np.concatenate([rng.permutation(len(names)) for _ in range(math.ceil(updates/len(names)))])[:updates]
    order = np.asarray([rng.choice(groups[names[int(i)]]) for i in group_order], dtype='<i8')
    cells = sum(samples[int(i)]['base'].numel() for i in order)
    evidence = {'schema': 'rheon-direction-feedback-training-v1', 'stage': spec['stage'],
        'manifest_sha256': spec['manifest_sha256'], 'payload_sha256': declared['sha256'],
        'source': manifest['source'], 'weight': spec['weight'], 'seed': spec['seed'],
        'cases': len(samples), 'groups': names,
        'group_exposures': np.bincount(group_order, minlength=len(names)).tolist(),
        'sampling_order_sha256': sha(order.tobytes()), 'additional_examples': updates,
        'forward_cells_per_correction': cells, 'resident_tensor_bytes': actual,
        'combined_resident_tensor_bytes': resident_bytes+actual,
        'amplitude_rms_pa': manifest['amplitude_rms_pa'],
        'loss': 'existing analytical relative pressure/gradient/equation and intermediate contraction',
        'scope': 'additional train-only directions; existing real/analytical streams preserved'}
    return samples, order, evidence
