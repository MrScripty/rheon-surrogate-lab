"""Exercise real audited windows, physical conditioning and label isolation."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import numpy as np

from long_temporal_dataset_v1 import ROOT, FIELDS, LongTemporalDataset, build_index, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(args):
    if args.output.exists():
        raise ValueError('Fresh dataset qualification directory required')
    reg = json.loads(args.protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen dataset dependency changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    args.output.mkdir(parents=True)
    result = dict(status='running', checked_examples=0, complete_target_outputs=0,
        new_parameter_fits=0, learned_forwards=0, cpu_model_forward=False,
        reserved52_references_opened=0, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    try:
        index = build_index([ROOT / p for p in reg['audits']])
        write(args.output / 'index.json', index)
        dataset = LongTemporalDataset(json.loads((args.output / 'index.json').read_text()))
        contract = json.loads((ROOT / reg['sampling_contract']).read_text())
        contracts = {(r['output_steps_per_s'], r['K']): r for r in contract['cases']}
        counts, weights, sample_weights = Counter(), Counter(), Counter()
        allowed = {'current', 'counts_xyz', 'cell_size_m', 'rho_kg_per_m3', 'start_time_s',
                   'output_dt_s', 'K', 'block_duration_s', 'lead_elapsed_s',
                   'normalized_phase', 'reference_dt_s', 'known_force_packet'}
        for example in index['examples']:
            inputs, labels = dataset.inputs(example), dataset.labels(example)
            spec = contracts[example['output_steps_per_s'], example['K']]
            start = example['start_fine_step']
            if set(inputs) != allowed or set(inputs['current']) != set(FIELDS):
                raise ValueError('Physical interface gained a target or scene-ID input')
            if (inputs['output_dt_s'] != spec['output_dt_s']
                    or inputs['block_duration_s'] != spec['block_duration_s']
                    or not np.array_equal(inputs['lead_elapsed_s'], spec['output_lead_times_s'])
                    or example['target_fine_steps'] != [start + n for n in spec['reference_output_offsets']]
                    or len(labels['outputs']) != example['K']):
                raise ValueError('Loader confused output interval with total block horizon')
            packet = inputs['known_force_packet']
            if packet.shape != (spec['reference_endpoint_offset'], 18) or packet.dtype.str != '<f8':
                raise ValueError('Known physical path does not cover complete block')
            if (not np.array_equal(packet[:, 9], np.arange(start, start + len(packet)) * (1 / 240))
                    or not (packet[:, 17] == 1 / 240).all()):
                raise ValueError('Known controls have wrong actual fine times')
            for output in labels['outputs']:
                if set(output) != set(FIELDS):
                    raise ValueError('Incomplete pressure/velocity/tracer target')
                for name in FIELDS:
                    if np.shares_memory(output[name], inputs['current'][name]):
                        raise ValueError('Target aliases deployment input storage')
                result['complete_target_outputs'] += 1
            identity = {n: a.tobytes() for n, a in inputs['current'].items()}
            labels['endpoint']['tracer'][:] = 123.
            if identity != {n: a.tobytes() for n, a in inputs['current'].items()}:
                raise ValueError('Mutating target changed deployment inputs')
            counts[(example['output_steps_per_s'], example['K'])] += 1
            weights[example['split']] += example['example_weight']
            sample_weights[example['physical_scenario_group_id']] += example['example_weight']
            result['checked_examples'] += 1
        if any(not math.isclose(v, 1., rel_tol=1e-12) for v in weights.values()):
            raise ValueError('Scene/cadence/window weights do not sum to one per split')
        if set(counts) != set(contracts) or len(set(counts.values())) != 1:
            raise ValueError('Unbalanced or missing all12 temporal cases')
        example = index['examples'][0]
        states = dataset.receipts[example['identifier']][2]
        entry = states[example['start_fine_step']]
        old = entry['sha256']
        entry['sha256'] = '0' * 64
        try:
            try:
                dataset.inputs(example)
            except ValueError as error:
                if 'Saved native state changed' not in str(error):
                    raise
            else:
                raise ValueError('Changed archive identity was accepted')
        finally:
            entry['sha256'] = old
        result.update(status='passed_real_long_temporal_dataset_conditioning_and_label_isolation',
            index_sha256=digest(args.output / 'index.json'),
            independent_masters_per_split=index['independent_masters_per_split'],
            examples_per_cadence_and_K={f'{r}/s-K{k}': n for (r, k), n in sorted(counts.items())},
            sampling_weight_sum_per_split=dict(weights), scene_weight_sums=dict(sample_weights),
            changed_archive_identity_rejected=True, future_target_mutation_cannot_change_inputs=True,
            trained_model_or_GPU_force_packet_qualification=False,
            full36_master_training_dataset_ready=False,
            note='Small audited-master interface qualification only; no model training or inference.')
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
