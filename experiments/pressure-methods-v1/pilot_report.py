"""Compare matched native teacher times and record qualified larger-grid data."""
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def report():
    runs = ROOT/'runs/pressure-methods-v1'
    reports = ROOT/'docs/plans/pressure-methods-v1'
    execution = read(reports/'native-pilots-execution.json')
    if execution['status'] != 'completed' or not all(w['reaped'] and w['returncode'] == 0 for w in execution['workers']):
        raise ValueError('All native pilots must complete and drain')
    reservation = read(runs/'native-01/reservation.json')
    pilots = {}
    evidence = {}
    for row in reservation['scenes']:
        directory = runs/'native-pilots/capture'/row['trajectory_id']
        receipt = read(directory/'receipt.json')
        if receipt['status'] != 'qualified_pilot' or receipt['row'] != row or not receipt['native_process']['reaped']:
            raise ValueError('Complete matching pilot required')
        if receipt['reservation_sha256'] != sha(runs/'native-01/reservation.json'):
            raise ValueError('Changed pilot reservation')
        for frame in receipt['frames']:
            expected_time = (frame['index']+1)*receipt['native_dt_s']
            if frame['time_s'] != expected_time:
                raise ValueError('Time mismatch')
            for name, field in frame['arrays'].items():
                if sha(directory/field['path']) != field['sha256']:
                    raise ValueError('Changed native field '+name)
        pilots[row['trajectory_id']] = (directory, receipt)
        evidence[str((directory/'receipt.json').relative_to(ROOT))] = sha(directory/'receipt.json')
    temporal = []
    for slot in [0, 1]:
        arrays = {}
        for substeps in [1, 2, 4]:
            directory, receipt = pilots[f'temporal-pilot-substeps{substeps}']
            frame = receipt['frames'][slot]
            fields = {k:np.fromfile(directory/v['path'], dtype=v['dtype']).astype(np.float64)
                      for k, v in frame['arrays'].items() if k == 'pressure' or k == 'tracer' or k.startswith('corrected_')}
            fields['pressure'] -= fields['pressure'].mean()
            fields['velocity'] = np.concatenate([fields.pop('corrected_'+axis) for axis in 'uvw'])
            arrays[substeps] = (frame['time_s'], fields)
        if len({v[0] for v in arrays.values()}) != 1:
            raise ValueError('Teacher endpoint physical times differ')
        for field in ['velocity', 'tracer', 'pressure']:
            # Common finest-reference normalization makes pair distances comparable.
            norm = max(float(np.linalg.norm(arrays[4][1][field])), 1e-300)
            coarse_fine = float(np.linalg.norm(arrays[1][1][field]-arrays[2][1][field]))/norm
            fine_finer = float(np.linalg.norm(arrays[2][1][field]-arrays[4][1][field]))/norm
            temporal.append(dict(time_s=arrays[4][0], field=field, coarse_fine_relative_l2=coarse_fine,
                fine_finer_relative_l2=fine_finer, decreasing_pair_distance=fine_finer < coarse_fine,
                pair_distance_ratio=fine_finer/coarse_fine if coarse_fine else None))
    large = []
    for name, (directory, receipt) in pilots.items():
        if not name.startswith('large-'):
            continue
        large.append(dict(trajectory_id=name, counts_xyz=receipt['row']['counts_xyz'],
            cells_per_metre=receipt['row']['cells_per_metre'], frames=len(receipt['frames']),
            native_steps_qualified=receipt['native_steps_qualified'], payload_bytes=receipt['payload_bytes'],
            elapsed_s=receipt['elapsed_s'],
            worst_reference_relative_residual=max(f['independent']['reference_relative_residual'] for f in receipt['frames']),
            worst_actual_divergence=max(f['independent']['actual_divergence_max_s_inv'] for f in receipt['frames'])))
    result = dict(status='native_pilots_verified', larger_grid_captures=large, temporal_pair_comparisons=temporal,
        pilot_trajectories=len(pilots), completed_native_steps=sum(r['native_steps_qualified'] for _, r in pilots.values()),
        evidence=evidence, source_sha256=sha(__file__),
        limitations=['One force profile and two physical snapshot times; these are training-only qualification pilots.',
            'Temporal pair distances measure timestep sensitivity, not exact-solution error.',
            'Fine-step pressure arrays are not automatically coarse-step pressure labels.',
            'Mixed-domain training, wider temporal coverage, model evaluation and velocity/tracer drift remain required.'],
        model_training_completed=False, all_cohorts_complete=False)
    write(reports/'native-pilots-results.json', result)
    lines = ['# Native domain, timestep and spacing pilots', '',
        'Nine training-only pilot trajectories completed with all native steps qualified.',
        'Retained snapshots passed independent RHS, residual, divergence, flux, Courant',
        'and viscosity checks. No model has been trained on these new captures yet.', '',
        '| Grid | Cells/metre | Snapshots | Native steps | Payload MiB | Capture seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in large:
        lines.append(f"| {row['counts_xyz']} | {row['cells_per_metre']} | {row['frames']} | {row['native_steps_qualified']} | {row['payload_bytes']/1024**2:.2f} | {row['elapsed_s']:.3f} |")
    lines += ['', '| Time s | Field | dt vs dt/2 | dt/2 vs dt/4 |', '| ---: | --- | ---: | ---: |']
    for row in temporal:
        lines.append(f"| {row['time_s']:.6f} | {row['field']} | {row['coarse_fine_relative_l2']:.6g} | {row['fine_finer_relative_l2']:.6g} |")
    lines += ['', 'Temporal differences use the finest field norm for both pairs. Smaller pair',
        'distance supports further temporal study; it does not establish exact-solution',
        'accuracy or that a fine pressure field labels the original coarse request.', '',
        '[Evidence hashes and scalar results](native-pilots-results.json).', '']
    (reports/'native-pilots-results.md').write_text('\n'.join(lines))
    print('Verified', len(pilots), 'native pilots', flush=True)


if __name__ == '__main__':
    report()
