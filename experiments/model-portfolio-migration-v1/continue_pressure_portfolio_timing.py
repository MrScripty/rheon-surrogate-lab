"""Continue a terminal, specifically diagnosed GPU-contention stop.

Preserve the original 98 exact blocks. Repeat only the known contended block
and the unexecuted suffix, in a new root under an explicit continuation protocol.
No unknown request, live process, or observation timeout permits this procedure.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pressure_gpu_timing import ROOT, read, save, sha, resources
from run_pressure_portfolio_gpu_timing import verify_material

BASE = ROOT / 'runs/model-portfolio-migration-v1'
ORIGINAL = BASE / 'pressure-portfolio-GPU-timing-01'
PROTOCOL = BASE / 'pressure-portfolio-GPU-timing-protocol-02.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(path, output):
    require(not path.exists() and not output.exists() and path.is_relative_to(BASE)
            and output.is_relative_to(BASE), 'Fresh owned protocol/root required')
    state = read(ORIGINAL / 'status.json')
    require(state['status'] == 'original_pressure_GPU_campaign_requires_reconciliation'
            and not Path('/proc', str(state['pid'])).exists(), 'Original parent must be terminal')
    active = state['active']
    require(active['block'] == 99 and active['identifier'] == 'C102'
            and not Path('/proc', str(active['pid'])).exists(), 'Exact known terminal block required')
    failed_path = ROOT / active['output'] / 'results.json'
    failed = read(failed_path)
    require(failed['status'] == 'failed_timing_pilot'
            and failed['error'] == 'Other GPU compute observed; latency is contended'
            and any(r['other_compute_pids'] for r in failed['resource_observations'])
            and not failed['device_qualification']['forbidden_module_calls'],
            'Only the recorded GPU-contention failure is admitted')
    original = read(PROTOCOL)
    require(sha(PROTOCOL) == state['protocol_sha256'], 'Original protocol differs')
    for name, expected in original['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Original registered source changed: ' + name)
    require(len(state['completed']) == 98, 'Exact original prefix required')
    for index, kept in enumerate(state['completed']):
        block = original['block_order'][index]
        raw_path = ROOT / kept['result']
        raw = read(raw_path)
        require(kept['block'] == index + 1 and all(kept[k] == block[k] for k in block)
                and sha(raw_path) == kept['sha256']
                and raw['status'] == 'complete_pressure_GPU_timing_pilot'
                and raw['all_output_hashes_exact'] and raw['cases'] == 128
                and raw['learned_parameters_unchanged'] and raw['architecture_unchanged']
                and not raw['device_qualification']['forbidden_module_calls'],
                'An original completed block is not exact')
    reference = read(ROOT / original['accuracy']['C102']['path'])
    require(len(failed['rows']) == len(reference['rows']) == 128
            and all(actual['output_sha256'] == expected['output_sha256']
                    and actual['input_sha256'] == expected['input_sha256']
                    for actual, expected in zip(failed['rows'], reference['rows'], strict=True)),
            'Contended block must retain every original numerical output')
    require(verify_material(read(ROOT / original['material'])) == original['material_bindings'],
            'Original model/material bindings changed')
    sources = dict(original['source_sha256'])
    own = str(Path(__file__).relative_to(ROOT))
    sources[own] = sha(Path(__file__))
    value = dict(original, schema='rheon.pressure-GPU-timing-contention-continuation.v1',
        created_unix=time.time(), output_root=str(output.relative_to(ROOT)), source_sha256=sources,
        original_protocol=dict(path=str(PROTOCOL.relative_to(ROOT)), sha256=sha(PROTOCOL)),
        original_status=dict(path=str((ORIGINAL / 'status.json').relative_to(ROOT)), sha256=sha(ORIGINAL / 'status.json')),
        accepted_prefix=state['completed'],
        excluded_contended_block=dict(original_block=99, path=str(failed_path.relative_to(ROOT)),
            sha256=sha(failed_path), numerical_output_hashes_exact=True, latency_qualified=False,
            reason=failed['error'], resource_observations=failed['resource_observations']),
        block_order=[dict(original_block=index+1, **block)
                     for index, block in enumerate(original['block_order']) if index >= 98],
        prefix_blocks=98, remaining_blocks=156, expected_remaining_timed_calls=199680,
        heavy_metadata_profiles_must_be_terminal=[
            'docs/plans/model-portfolio-migration-v1/reports/candidate-receipt-validation-profile-01.json',
            'docs/plans/model-portfolio-migration-v1/reports/receipt-validation-pinned-baseline-01.json'])
    save(path, value)
    print('Preserved98blocks; registered156remaining blocks and explicit C102 contention exclusion', flush=True)


def run(path):
    protocol = read(path)
    output = ROOT / protocol['output_root']
    require(not output.exists() and output.is_relative_to(BASE), 'Follow the original continuation; never replay it')
    require(sha(ROOT / protocol['original_status']['path']) == protocol['original_status']['sha256'],
            'Original terminal state changed')
    for name in protocol['heavy_metadata_profiles_must_be_terminal']:
        profile = read(ROOT / name)
        require(profile['status'] == 'fixed_record_validation_completed' and profile['source_unchanged']
                and profile['manifest_unchanged'], 'Heavy metadata profile must finish first')
    for name, expected in protocol['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Registered source changed: ' + name)
    require(not resources()['other_compute_pids'], 'Other GPU compute is still active')
    material = read(ROOT / protocol['material'])
    require(verify_material(material) == protocol['material_bindings'], 'Captured material differs')
    runtime = read(ROOT / protocol['runtime_manifest'])
    sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir()
    state = dict(status='starting', pid=os.getpid(), created_unix=time.time(),
        protocol=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
        completed=list(protocol['accepted_prefix']), active=None, retained_original_prefix=98,
        excluded_contended_block=protocol['excluded_contended_block'],
        CPU_neural_forwards=0, new_fits=0, model_or_dataset_copies=0,
        full_portfolio_complete=False, independent_reduction_pending=True)
    began = time.monotonic()
    try:
        for block in protocol['block_order']:
            remaining = protocol['aggregate_deadline_seconds'] - (time.monotonic() - began)
            require(remaining > 0, 'Original continuation deadline reached')
            index = block['original_block']
            name = f"{index:03}-{block['identifier']}-phase{block['phase']}"
            destination = output / name
            with (output / (name + '.log')).open('x') as log:
                child = subprocess.Popen([runtime['python'], '-B', str(ROOT / protocol['worker']),
                    '--protocol', str(path), '--identifier', block['identifier'], '--output', str(destination)],
                    cwd=ROOT, env=environment(runtime, 'cuda'), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                state.update(status='original_continuation_GPU_worker_in_flight',
                    active=dict(block=index, pid=child.pid, output=str(destination.relative_to(ROOT)),
                                identifier=block['identifier'], phase=block['phase']))
                save(output / 'status.json', state)
                try:
                    returned = child.wait(timeout=min(remaining, protocol['worker_deadline_seconds'] + 60))
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=10)
            result_path = destination / 'results.json'
            result = read(result_path) if result_path.exists() else None
            require(not returned and result is not None
                and result['status'] == 'complete_pressure_GPU_timing_pilot' and result['cases'] == 128
                and result['all_output_hashes_exact'] and result['learned_parameters_unchanged']
                and result['architecture_unchanged'] and not result['device_qualification']['forbidden_module_calls'],
                'Known continuation worker incomplete; retain and reconcile without retry')
            state['completed'].append(dict(block=index, identifier=block['identifier'], phase=block['phase'],
                result=str(result_path.relative_to(ROOT)), sha256=sha(result_path), cases=128))
            state.update(status='continuation_GPU_block_complete', active=None)
            save(output / 'status.json', state)
            print(name, 'complete128requests', flush=True)
        verify_material(material)
        for name, expected in protocol['source_sha256'].items():
            require(sha(ROOT / name) == expected, 'Source changed during original continuation')
        require(len(state['completed']) == 254, 'Complete joined block collection required')
        state.update(status='all254_pressure_timing_blocks_collected_independent_reduction_pending',
                     active=None, finished_unix=time.time())
    except BaseException as error:
        state.update(status='original_continuation_requires_reconciliation',
                     error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.output is None:
            parser.error('prepare requires --output')
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        run(args.protocol.resolve())
