"""Measure tracer transport in one recorded native timeline, in physical units.

The visibility threshold is a display diagnostic, not a pressure/physics gate.
An independent one-cell translation oracle lives in the native preview tests.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from policy import MIB, decode, read, require, sha, write


def measure(path):
    raw = read(path, 32 * MIB)
    session = decode(raw)
    require(session['schema'] == 'rheon-water-training-timeline-v1',
            'tracer_session', 'one native timeline')
    counts = session['counts_xyz']
    require(isinstance(counts, list) and len(counts) == 3 and
            all(type(n) is int and 5 <= n <= 128 for n in counts) and
            math.prod(counts) <= 65536 and 2 <= len(session['frames']) <= 128,
            'tracer_budget', 'bounded native geometry and timeline before allocation')
    nx, ny, nz = session['counts_xyz']
    h = session['cell_size_m']
    require(np.isfinite(h) and 0 < h <= 1, 'tracer_spacing', 'positive cubic spacing')
    z, y, x = np.indices((nz, ny, nx))
    positions = (np.stack((x, y, z), axis=-1).reshape(-1, 3) + 0.5) * h
    rows = []
    previous_index = -1
    for frame in session['frames']:
        require(frame['index'] > previous_index, 'tracer_order', 'no scene resets')
        previous_index = frame['index']
        concentration = np.asarray(frame['tracer'], dtype=np.float64)
        velocity = np.asarray(frame['velocity'], dtype=np.float64)
        require(concentration.shape == (nx * ny * nz,) and
                velocity.shape == (nx * ny * nz, 3) and
                np.isfinite(concentration).all() and np.isfinite(velocity).all() and
                (concentration >= 0).all() and concentration.sum() > 0,
                'tracer_values', 'finite full-grid nonnegative concentration')
        weight = concentration / concentration.sum()
        center = (positions * weight[:, None]).sum(axis=0)
        sigma = np.sqrt(((positions - center)**2 * weight[:, None]).sum(axis=0))
        require((sigma > 0).all(), 'tracer_spread', 'marker occupies more than one position in each axis')
        rows.append({
            'native_step': frame['index'] + 1, 'time_s': frame['time_s'],
            'centroid_m': center.tolist(), 'axis_sigma_m': sigma.tolist(),
            'tracer_weighted_velocity_m_s': (velocity * weight[:, None]).sum(axis=0).tolist(),
            'maximum_water_speed_m_s': float(np.linalg.norm(velocity, axis=1).max()),
        })
    require(len(rows) >= 2, 'tracer_timeline', 'at least two recorded states')
    displacement = (np.asarray(rows[-1]['centroid_m']) - rows[0]['centroid_m']) / h
    width_change = np.asarray(rows[-1]['axis_sigma_m']) / rows[0]['axis_sigma_m'] - 1
    visible = float(np.linalg.norm(displacement)) >= 0.5 or float(np.abs(width_change).max()) >= 0.1
    return {
        'schema': 'rheon-native-tracer-motion-diagnostic-v1',
        'trajectory_id': session['trajectory_id'], 'session_sha256': sha(raw),
        'cell_size_m': h, 'recorded_states': len(rows),
        'native_steps_qualified': session['provenance']['native_steps_qualified'],
        'retained_frame_parity': session['provenance']['retained_frame_parity'],
        'centroid_displacement_cells_xyz': displacement.tolist(),
        'centroid_displacement_cells_norm': float(np.linalg.norm(displacement)),
        'axis_sigma_change_fraction': width_change.tolist(),
        'visible_motion': visible,
        'visibility_rule': 'centroid travels >= 0.5 cell OR any axis sigma changes >= 10%; display diagnostic only',
        'rows': rows,
        'limitations': 'Native semi-Lagrangian concentration is diffusive and not mass-conservative; width change alone does not prove physical stretching.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-visible', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'tracer_output', 'fresh diagnostic path')
    result = measure(args.session)
    write(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))
    if args.require_visible and not result['visible_motion']:
        raise SystemExit(2)
