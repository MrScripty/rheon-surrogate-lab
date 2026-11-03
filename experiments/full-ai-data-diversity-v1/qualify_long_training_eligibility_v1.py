"""Check real partial expansion metadata and its training-admission rejection."""
import argparse
from collections import Counter
from pathlib import Path

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index
from long_temporal_dataset_v3 import LongTemporalDataset
from reserve_long_scenes_v1 import instantiate


def run(args):
    if args.output.exists():
        raise ValueError('Fresh partial-data qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Eligibility-check source changed: ' + name)
    args.output.mkdir(parents=True)
    index = build_index([ROOT / p for p in reg['audits']])
    reserved = {r['identifier']: r for r in read(ROOT / reg['reservation'])['rows']}
    density = Counter()
    epoch = Counter()
    for identifier, master in index['masters'].items():
        receipt = read(ROOT / master['receipt'])
        if receipt['scene'] != instantiate(reserved[identifier]):
            raise AssertionError('Audited capture differs from frozen original physical scene')
        density[receipt['scene']['cells_per_metre']] += 1
        epoch[receipt.get('reference_numerical_contract', 'original normal-only F32')] += 1
    if len(index['masters']) != reg['expected_masters']:
        raise AssertionError('Unexpected partial master population')
    try:
        LongTemporalDataset(index)
    except ValueError as error:
        if str(error) != 'Full36 training eligibility is not established':
            raise
    else:
        raise AssertionError('Partial mixed-epoch diagnostic index was admitted for training')
    write(args.output / 'index.json', index)
    write(args.output / 'results.json', dict(
        status='passed_real_partial_metadata_and_full36_training_admission_rejection',
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        index_sha256=digest(args.output / 'index.json'),
        audited_masters=len(index['masters']),
        independent_masters_per_split=index['independent_masters_per_split'],
        density_counts=dict(density), reference_epoch_counts=dict(epoch),
        diagnostic_examples=len(index['examples']),
        original_physical_scene_and_prescribed_path_exact=True,
        partial_index_rejected_for_training=True,
        training_ready_full36=False, candidate_references_adopted=False,
        new_parameter_fits=0, optimizer_updates=0, cpu_model_forward=False,
        learned_forwards=0, reserved52_references_opened=0,
        full_generator_target_pass=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
