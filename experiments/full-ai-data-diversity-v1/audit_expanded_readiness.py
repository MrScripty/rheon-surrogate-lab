"""Receipt-only coverage audit; never opens reserved references or field archives."""
import argparse
import collections
import hashlib
import math
from pathlib import Path
import time
from support import ROOT, read, write, digest


def run(args):
    if args.output.exists():
        raise ValueError('Fresh data-readiness snapshot required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Readiness audit source changed: '+path)
    reservation = read(ROOT/reg['reservation'])
    expected = {r['identifier']: r for r in reservation['rows']}
    group_splits = collections.defaultdict(set)
    for row in expected.values():
        group_splits[row['group']].add(row['split'])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError('Scene/density group leaked across splits')
    rows, observations, rejected = [], [], []
    for path in reg['controllers']:
        raw = (ROOT/path).read_bytes()
        import json
        controller = json.loads(raw)
        observations.append(dict(path=path, snapshot_sha256=hashlib.sha256(raw).hexdigest(),
            status=controller['status'], active=controller.get('active'),
            observed_completed=len(controller['completed']), observed_unresolved=len(controller['reference_rejections'])))
        rejected.extend(controller['reference_rejections'])
        for item in controller['completed']:
            if item['identifier'] not in expected:
                raise ValueError('Unregistered scene in completed reference population')
            declared = expected[item['identifier']]
            if declared['split'] == 'reserved':
                raise ValueError('Reserved references cannot be opened by this audit')
            receipt_path = ROOT/item['receipt']
            if digest(receipt_path) != item['receipt_sha256']:
                raise ValueError('Completed reference receipt changed')
            receipt = read(receipt_path)
            scene = receipt['scene']
            if receipt['status'] != 'completed' or receipt['split'] != declared['split'] or receipt['group'] != declared['group']:
                raise ValueError('Whole-scene receipt split/group differs')
            if scene['counts'] != declared['scene']['counts'] or scene['cells_per_metre'] != declared['scene']['cells_per_metre']:
                raise ValueError('Grid dimensions or cells/metre differ')
            owner = receipt['native_process']
            if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                raise ValueError('Native reference child was not qualified/reaped')
            if len(receipt['frames']) != 240 or not all(
                f['frame']['advanced'] and f['frame']['quality']['qualified'] and not f['frame']['quality']['hard_stop']
                for f in receipt['frames']):
                raise ValueError('All actual native output transitions must qualify')
            reference_substeps = receipt.get('reference_substeps', 1)
            fine_steps = sum(len(f.get('fine_frames', [f])) for f in receipt['frames'])
            if fine_steps != 240*reference_substeps:
                raise ValueError('Native fine work differs from declared lineage')
            states = {s['step']: s for s in receipt['states']}
            if len(states) != len(receipt['states']) or 0 not in states or 240 not in states:
                raise ValueError('Complete unique initial and terminal states required')
            counts = scene['counts']
            cells = math.prod(counts)
            sizes = dict(pressure=cells, tracer=cells, rhs=cells, native_input_residual=cells, pressure_base=cells,
                **{name: (counts[a]+1)*math.prod(counts[b] for b in range(3) if b != a)
                   for a, name in enumerate(('vx', 'vy', 'vz'))})
            for step, state in states.items():
                if state['time_s'] != step/60 or set(state['arrays']) != set(sizes):
                    raise ValueError('Complete reference clock/field declaration required')
                for name, size in sizes.items():
                    dtype = '<f4' if name in ('vx', 'vy', 'vz', 'tracer') else '<f8'
                    if state['arrays'][name]['shape'] != [size] or state['arrays'][name]['dtype'] != dtype:
                        raise ValueError('Typed staggered field declaration changed')
            windows = {str(k): [s for s in sorted(states) if all(s+lead in states for lead in range(1, k+1))]
                       for k in (1, 2, 3, 4)}
            rows.append(dict(identifier=item['identifier'], group=receipt['group'], split=receipt['split'],
                cells_per_metre=scene['cells_per_metre'], counts_xyz=counts, profile=receipt['profile'], motion=receipt['motion'],
                receipt=item['receipt'], receipt_sha256=item['receipt_sha256'], stored_states=len(states),
                physical_steps=240, native_fine_steps=fine_steps, reference_substeps=reference_substeps,
                reference_scheme_id=receipt.get('reference_scheme_id', 'sampled-controls-native-substeps-1'),
                output_dt_s=1/60, control_semantics='Sampled macro controls held within each output interval',
                available_complete_window_starts_by_K=windows,
                mature_K4_windows_at_requested_starts=[s for s in reg['mature_starts'] if s in windows['4']],
                temporal_convergence_qualified=receipt.get('temporal_convergence_qualified', False)))
    unique = {r['identifier'] for r in rows}
    summaries = []
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row['split'], row['reference_substeps'], row['cells_per_metre'], tuple(row['counts_xyz']))].append(row)
    for (split, substeps, density, grid), values in sorted(grouped.items()):
        summaries.append(dict(split=split, reference_substeps=substeps, cells_per_metre=density, counts_xyz=list(grid),
            scenes=len(values), stored_states=sum(r['stored_states'] for r in values),
            mature_four_step_windows=sum(len(r['mature_K4_windows_at_requested_starts']) for r in values)))
    missing = [dict(identifier=key, split=row['split'], group=row['group'], cells_per_metre=row['scene']['cells_per_metre'],
                    counts_xyz=row['scene']['counts']) for key, row in expected.items()
               if row['split'] != 'reserved' and key not in unique]
    result = dict(status='completed_receipt_only_expanded_readiness_snapshot', created_unix=time.time(),
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'], rows=rows, summaries=summaries,
        controller_observations=observations, registered_scene_counts=reservation['scene_counts_by_split'],
        unique_completed_scenes_by_split=dict(collections.Counter(expected[k]['split'] for k in unique)),
        missing_nonreserved_scenes=missing, original_rejections_with_later_qualified_reference=[
            r['identifier'] for r in rejected if r['identifier'] in unique],
        unresolved_rejections=[r for r in rejected if r['identifier'] not in unique],
        reserved_references_opened=0, field_archives_opened=0,
        field_archive_hashes_and_values_reverified=False, model_fitting_executed=False,
        full_data_training_ready=False, needs=['Finish missing native references',
            'Complete train-only mature K2-4 windows', 'Freeze consistent reference/control semantics for successor',
            'Verify actual field archives before training', 'Use complete240step scoring for future independent confirmation'],
        no_temporal_convergence_claim=True, no_latency_claim=True)
    write(args.output, result)
    print('Unique qualified reference scenes', dict(result['unique_completed_scenes_by_split']),
          'missing', len(missing), 'reserved opened0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
