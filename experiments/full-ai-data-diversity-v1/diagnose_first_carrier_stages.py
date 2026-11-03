"""OFFLINE debug: compare the same causal input and the same emitted pressure."""
import argparse
import os
from pathlib import Path
from support import ROOT, read, write, digest
from bounded_native_client import BoundedNativeClient
from legacy_table_forcing import bind_for_diagnostic
from learned_pressure_history import LearnedPressureHistory
from mixed_modal_sampler import load_model
from recurrent_sampler_contiguous_v3 import library
import restored_recurrent_sampler as restored
from diagnose_causal_composition import inventory, FIELDS
from evaluate_recurrent_sampler import identity
from numerics import fine_action
import numpy as np
import torch


def compare(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    delta = a.astype('f8')-b.astype('f8')
    return dict(bitwise_equal=a.dtype == b.dtype and a.tobytes() == b.tobytes(),
        numerical_differences=int(np.count_nonzero(delta)), max_absolute=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta)/max(float(np.linalg.norm(b)), 1e-30)))


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh OFFLINE stage-debug output required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen stage-debug source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    restored.library = library
    receipt = read(ROOT/reg['receipt'])
    scene = receipt['scene']
    with np.load(ROOT/reg['initial_state'], allow_pickle=False) as archive:
        current = {k: archive[k].copy() for k in FIELDS}
    args.output.mkdir(parents=True)
    model = LearnedPressureHistory(load_model(reg['pair'], reg['library_path']), mode='reuse')
    before = inventory(model)
    captured = {}
    rhs_original, action_original = model.rhs, model.action

    def rhs(*a):
        for component, value in enumerate(a[0]):
            captured[f'provisional_{component}'] = value.numpy().copy()
        value = rhs_original(*a)
        captured['learned_rhs'] = value.numpy().copy()
        return value

    def action(*a):
        value = action_original(*a)
        captured['learned_action'] = value.numpy().copy()
        return value

    owners, native_outputs, native_requests = [], {}, {}
    try:
        model.rhs, model.action = rhs, action
        with torch.inference_mode(), bind_for_diagnostic():
            output = model.predict(current, scene, reg['initial_step'], 1)[0]
        if identity(output) != reg['expected_endpoint_identity']:
            raise ValueError('Instrumentation must reproduce the actual frozen first divergence')
        del model.rhs
        del model.action
        for role in ('reference_pressure', 'same_learned_pressure'):
            client = BoundedNativeClient(scene, args.output/role)
            try:
                client.restore(reg['initial_step'], current)
                if role == 'reference_pressure':
                    frame = client.advance()
                else:
                    # Future pressure here is an explicitly OFFLINE stage-isolation input.
                    frame = client.advance(method='OFFLINE_same_learned_pressure',
                        predict=lambda sample: output['pressure'].reshape(sample['base'].shape))
                if not frame['frame']['advanced'] or not frame['frame']['quality']['qualified']:
                    raise ValueError('Actual native diagnostic advance must pass unchanged gates')
                packet, values = client.inspect()
                if packet['step'] != reg['stop']:
                    raise ValueError('Actual single physical transition required')
                native_outputs[role] = values
                native_requests[role] = dict(client.last_request)
            finally:
                owner = client.close()
                owners.append(owner)
                if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                    raise ValueError('Every native diagnostic child must complete and reap')
        shape = (1, 1, *scene['counts'][::-1])
        h = torch.tensor([1/scene['cells_per_metre']], dtype=torch.float64)
        rho = torch.tensor([1000.], dtype=torch.float64)
        with torch.inference_mode():
            captured['offline_edge_action'] = fine_action(
                torch.from_numpy(current['pressure'].reshape(shape)), h, rho).numpy().copy()
        request = native_requests['reference_pressure']
        captured['native_rhs_label'] = request['rhs']
        captured['native_residual'] = request['native_input_residual']
        captured['learned_residual'] = captured['learned_rhs'].reshape(-1)-captured['learned_action'].reshape(-1)
        comparisons = dict(
            rhs=compare(captured['learned_rhs'], request['rhs']),
            pressure_action=compare(captured['learned_action'], captured['offline_edge_action']),
            residual=compare(captured['learned_residual'], request['native_input_residual']),
            native_requests_identical=all(compare(native_requests['reference_pressure'][key],
                native_requests['same_learned_pressure'][key])['bitwise_equal']
                for key in ('base', 'rhs', 'native_input_residual')),
            learned_vs_native_same_pressure={key: compare(output[key].numpy(),
                native_outputs['same_learned_pressure'][key]) for key in FIELDS},
            learned_vs_native_reference_pressure={key: compare(output[key].numpy(),
                native_outputs['reference_pressure'][key]) for key in FIELDS})
        for role, values in native_outputs.items():
            captured.update({role+'_'+key: value for key, value in values.items()})
        captured.update({'learned_'+key: value.numpy() for key, value in output.items()})
        np.savez_compressed(args.output/'stages.npz', **captured)
        if inventory(model) != before or torch.cuda.is_initialized():
            raise ValueError('OFFLINE debug changed learned tensors or device')
        write(args.output/'results.json', dict(status='completed_OFFLINE_first_carrier_stage_debug',
            identifier=receipt['identifier'], initial_step=reg['initial_step'], stop=reg['stop'],
            comparisons=comparisons, native_processes=owners,
            archive_sha256=digest(args.output/'stages.npz'),
            source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
            optimizer_updates=0, independent_confirmation=False, model_target_pass=False,
            native_operators_only_in_offline_diagnostic=True))
        print('[DEBUG-stage15]', comparisons, flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
