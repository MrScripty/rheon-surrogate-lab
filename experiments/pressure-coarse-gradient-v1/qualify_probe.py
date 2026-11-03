"""Qualify the original optimizer seam on an already-consumed real frame."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'artifact-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    sys.path.insert(0, str(args.project.resolve()))
    import numpy as np
    import torch
    import train
    from gradient_screen import Probe, gradient_snapshot, DiagnosticLimit
    config = json.loads((args.project/'config.json').read_text())
    train.model.select(config)
    train.model.configure('cuda')
    document = json.loads((args.project/'project.json').read_text())
    inputs = {row['id']:(args.artifact_root/row['path']).read_bytes() for row in document['data']}
    _, parents = train.data.load((inputs['dataset'], inputs['dataset_extra']), json.loads(inputs['splits']))
    normalization = train.model.fit_normalization(parents, train.data.fit_normalization)
    parent = min((i for i,p in enumerate(parents) if p['entry']['split']=='train'),
                 key=lambda i: np.prod(parents[i]['counts_xyz']))
    sample = train.data.sample(parents[parent], 0, device='cuda')
    indices = np.asarray([[parent, 0]], dtype='<i8')
    def construct():
        torch.manual_seed(config['training']['seed'])
        network = train.model.PressureCorrectionNet().cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=config['training']['learning_rate'],
            betas=(.9,.999), eps=1e-8, weight_decay=0)
        return network, optimizer
    def step(network, optimizer):
        return train.one_update(network, optimizer, parents, indices, normalization,
            cache={(parent,0):sample}, strategy=config['execution'],
            analytical_sample=sample, analytical_weight=.1,
            supplemental_sample=sample, supplemental_weight=.1,
            direction_sample=sample, direction_weight=.1,
            current_sample=sample, current_weight=.1)
    baseline, base_optimizer = construct()
    baseline_results = [step(baseline, base_optimizer) for _ in range(2)]
    baseline_rng = torch.random.get_rng_state().clone()
    baseline_cuda_rng = torch.cuda.get_rng_state().clone()
    network, optimizer = construct()
    initial = {name:dict(shape=list(value.shape),sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest())
               for name,value in network.state_dict().items()}
    order = np.zeros(5000,dtype='<i8')
    real_order = np.tile(indices,(5000,1,1))
    reference = dict(initial_inventory=initial, analytical_seeds={}, direction_seeds={},
        sampling_order_sha256={label:hashlib.sha256((real_order if label=='real' else order).tobytes()).hexdigest()
                               for label in ('real','analytical','supplemental','direction','current')})
    events = []
    probe = Probe(train, 2, reference, events.append)
    for label in reference['sampling_order_sha256']:
        probe.capture_order(label, real_order if label=='real' else order, {})
    probe.install()
    try:
        observed = [step(network, optimizer) for _ in range(2)]
        assert observed == baseline_results
        assert torch.equal(torch.random.get_rng_state(), baseline_rng)
        assert torch.equal(torch.cuda.get_rng_state(), baseline_cuda_rng)
        assert all(torch.equal(value,baseline.state_dict()[key]) for key,value in network.state_dict().items())
        for parameter, original in zip(network.parameters(),baseline.parameters()):
            assert torch.equal(parameter.grad, original.grad)
            for key in ('step','exp_avg','exp_avg_sq'):
                assert torch.equal(optimizer.state[parameter][key],base_optimizer.state[original][key])
        before = {key:value.clone() for key,value in network.state_dict().items()}
        try:
            step(network, optimizer)
        except DiagnosticLimit:
            pass
        else:
            raise AssertionError('Diagnostic cap did not stop')
        assert all(torch.equal(value,before[key]) for key,value in network.state_dict().items())
        assert probe.completed == 2
    finally:
        probe.close()
    small = torch.nn.Linear(2,1,bias=False).cuda()
    small.weight.grad = torch.tensor([[3e20,4e20]],device='cuda')
    snapshot = gradient_snapshot(small)
    assert snapshot['all_gradient_components_finite'] and snapshot['ordinary_float32_total_norm']=='inf'
    assert abs(snapshot['reference_float64_total_norm']/5e20-1)<1e-6
    before = small.weight.grad.clone()
    clip_events = []
    clip_probe = Probe(train, 1, reference, clip_events.append)
    clip_probe.install()
    clip_probe.network = small
    clip_probe.current = dict(scope='Known finite 3-4-vector actual clipping seam')
    try:
        try:
            torch.nn.utils.clip_grad_norm_(small.parameters(),1.,error_if_nonfinite=True)
        except RuntimeError as error:
            assert 'non-finite' in str(error)
        else:
            raise AssertionError('Real original clipping guard did not reject overflowing total norm')
        assert clip_events[-1]['event'] == 'clip_failure'
        assert clip_events[-1]['update']['gradient_snapshot'] == snapshot
    finally:
        clip_probe.close()
    assert torch.equal(small.weight.grad,before)
    small.weight.grad[0,0]=float('nan')
    nonfinite = gradient_snapshot(small)
    assert not nonfinite['all_gradient_components_finite']
    result = dict(format='pressure-coarse-gradient-probe-qualification-v1', status='passed',
        elapsed_s=time.perf_counter()-start, device='cuda', gpu=torch.cuda.get_device_name(0),
        exact_original_update_results=True, exact_weights_gradients_adam_rng=True,
        stopped_before_third_update=True, observer_removed=True,
        optimizer_seam_steps=4, fitting_sequence_steps=0, simulations_advanced=0,
        finite_overflow_snapshot=snapshot, actual_nonfinite_snapshot=nonfinite,
        actual_original_guard_observed_without_change=True,
        source_sha256=hashlib.sha256(Path(__file__).with_name('gradient_screen.py').read_bytes()).hexdigest(),
        scope='Two original vs observed CUDA updates on the same consumed smallest real frame used in five streams. Observation/cap seam qualification only; not original full sampling replay or scientific fit.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','elapsed_s','device','optimizer_seam_steps')}))


if __name__ == '__main__':
    main()
