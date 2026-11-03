"""Build an audited partial index and qualify exact prescribed event metadata.

Every indexed temporal window is checked at the metadata level. One complete
K4 example per independent master exercises actual typed current/target files
and verifies that target mutation cannot alter model inputs. Existing native
audits cover the full saved archive and exact late replay population.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from long_temporal_dataset_v1 import ROOT, build_index, digest
from long_temporal_dataset_v2 import LongTemporalDataset
from pulse_event_controls_v1 import mean_pulse_gate


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(args):
    if args.output.exists():
        raise ValueError('Fresh audited event-view qualification required')
    reg = json.loads(args.protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen event view dependency changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', checked_window_metadata=0, loaded_complete_examples=0,
        loaded_complete_target_outputs=0, new_parameter_fits=0, learned_forwards=0,
        cpu_model_forward=False, reserved52_references_opened=0, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        full36_master_training_dataset_ready=False, sampled240_labels_unchanged=True)
    try:
        index = build_index([ROOT / p for p in reg['audits']])
        if index['independent_masters_per_split'] != reg['expected_masters_per_split']:
            raise ValueError('Exact declared independent master population required')
        write(args.output / 'index.json', index)
        dataset = LongTemporalDataset(index)
        counts, weights = Counter(), Counter()
        for example in index['examples']:
            if time.monotonic() - started > reg['maximum_wall_seconds']:
                raise TimeoutError('Event-view qualification deadline exceeded')
            packet = dataset.event_inputs(example)
            start, duration = example['start_fine_step'] / 240, example['K'] / example['output_steps_per_s']
            if (packet['start_time_s'] != start or packet['block_duration_s'] != duration
                    or packet['future_physical_fields_used']
                    or packet['requested_lead_times_s'] != [j / example['output_steps_per_s'] for j in range(1, example['K'] + 1)]):
                raise ValueError('Event conditioning confused physical interval or future labels')
            program = dataset.receipts[example['identifier']][1]['scene']['forcing_program']
            integral = mean_pulse_gate(program, start, start + duration) * duration
            cuts = sorted(set([0., duration, *(e[0] for e in packet['events'])]))
            partition = sum(mean_pulse_gate(program, start + left, start + right) * (right - left)
                for left, right in zip(cuts, cuts[1:]) if start + right > start + left)
            if not math.isclose(integral, partition, abs_tol=1e-12, rel_tol=1e-11):
                raise ValueError('Exact event partition lost prescribed gate integral')
            if any(not 0 < e[0] <= duration or not math.isclose(e[1], e[0] / duration, abs_tol=1e-14)
                   or not e[4] > 0 for e in packet['events']):
                raise ValueError('Invalid edge offset, normalized phase or pulse duration')
            counts[example['output_steps_per_s'], example['K']] += 1
            weights[example['split']] += example['example_weight']
            result['checked_window_metadata'] += 1
        for identifier in index['masters']:
            example = next(e for e in index['examples'] if e['identifier'] == identifier
                and e['output_steps_per_s'] == 30 and e['K'] == 4)
            inputs, labels = dataset.inputs(example), dataset.labels(example)
            before = {n: a.tobytes() for n, a in inputs['current'].items()}
            events_before = json.dumps(inputs['known_pulse_events'], sort_keys=True)
            if len(labels['outputs']) != 4 or inputs['known_force_packet'].shape != (32, 18):
                raise ValueError('Full long horizon force path and four complete outputs required')
            for output in labels['outputs']:
                for name, value in output.items():
                    if np.shares_memory(value, inputs['current'][name]):
                        raise ValueError('Future reference target aliases a model input')
                    value[:] = 123.
            if (before != {n: a.tobytes() for n, a in inputs['current'].items()}
                    or events_before != json.dumps(dataset.event_inputs(example), sort_keys=True)):
                raise ValueError('Mutating future labels changed deployment conditioning')
            result['loaded_complete_examples'] += 1
            result['loaded_complete_target_outputs'] += 4
        if len(counts) != 12 or len(set(counts.values())) != 1 or any(not math.isclose(w, 1., rel_tol=1e-12) for w in weights.values()):
            raise ValueError('Unbalanced temporal population or split weighting')
        result.update(status='passed_audited_partial_long_index_exact_event_metadata_and_label_isolation',
            independent_masters_per_split=index['independent_masters_per_split'],
            index_sha256=digest(args.output / 'index.json'),
            window_metadata_per_cadence_K={f'{r}/s-K{k}': n for (r, k), n in sorted(counts.items())},
            sample_weight_sum_per_split=dict(weights),
            native_audits_cover_all_saved_archives=True,
            exact_events_are_known_controls_not_future_reference_fields=True,
            exact_events_do_not_claim_event_exact_reference=True,
            elapsed_seconds=time.monotonic() - started)
    except BaseException as exception:
        result.update(status='failed', error=repr(exception))
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
