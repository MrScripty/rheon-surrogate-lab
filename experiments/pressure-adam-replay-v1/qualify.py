"""Prove observer transparency against the retained, uninstrumented C69 owner."""
import argparse
import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write
from qualify import qualify


def check(config_path, project, output):
    started = time.perf_counter()
    if output.exists():
        raise ValueError('Fresh qualification receipt required')
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    config = read(config_path)
    control = ROOT/config['control_project']
    left, right = read(project/'config.json'), read(control/'config.json')
    left.pop('id'); right.pop('id')
    assert left['resources'].pop('max_output_bytes') == 64*1024**2
    assert right['resources'].pop('max_output_bytes') == 32*1024**2
    assert left == right, 'Numerical training policy unchanged'
    left, right = read(project/'fine-tuning.json'), read(control/'fine-tuning.json')
    left.pop('arm'); right.pop('arm')
    assert left == right
    witness, previous = read(project/'source-witness.json'), read(control/'source-witness.json')
    for name, identity in previous['sources'].items():
        if name not in ('train.py','fit_bank.py','fine-tuning.json','config.json','portable_policy.py'):
            assert witness['sources'][name] == identity, name
    expected_budget = (control/'portable_policy.py').read_text().replace("resources['max_output_bytes']<=32*MIB", "resources['max_output_bytes']<=64*MIB")
    assert (project/'portable_policy.py').read_text() == expected_budget
    sys.path.insert(0,str(project))
    from policy import experiment
    experiment(read(project/'config.json'))
    assert witness['normalization_sha256'] == previous['normalization_sha256']
    assert sha(project/'adam_trace.py') == sha(HERE/'adam_trace.py')
    provider = read(project/'provider.json')
    assert provider['resources']['output_bytes'] == 64*1024**2
    for mode in ('train','restore'):
        recipe = read(project/(mode+'-recipe.json'))
        assert recipe['resources']['max_output_bytes'] <= provider['resources']['output_bytes']
        assert recipe['resources']['wall_seconds'] <= provider['resources']['wall_seconds']
        if mode == 'train':
            limits = recipe['model_evidence']['profile']['limits']
            assert limits['shard_bytes']==8*1024**2 and limits['total_output_bytes']==64*1024**2
            assert len(recipe['outputs'])<=32
        assert all(rule['max_bytes'] <= 40*1024**2 for rule in recipe['inputs'].values())
    base = output.with_suffix('.base.json')
    qualify(project, base)
    import numpy as np
    import torch
    import model
    import data
    import fit_bank
    import adam_trace
    import fine_tuning
    spec = importlib.util.spec_from_file_location('retained_balanced_bank',control/'fit_bank.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    assert adam_trace.UPDATES == tuple(config['capture_updates'])
    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ('dataset','dataset_extra')),read(inputs['splits']))
    cache = {(i,j):data.sample(p,j) for i,p in enumerate(parents) for j in range(len(p['frames']))}
    normalization = model.fit_normalization(parents,data.fit_normalization)
    initial = ROOT/next(r['path'] for r in read(project/'project.json')['data'] if r['id']=='initial_weights')
    prototype = model.PressureCorrectionNet()
    fine_tuning.initialize(prototype,initial,data.helpers())
    extras = {name+suffix:None if suffix=='_sample' else 0. for name in
        ('analytical','supplemental','direction','current') for suffix in ('_sample','_weight')}
    rng = torch.get_rng_state().clone()
    outcomes = []
    trace_path = output.parent/'transparency-trace.npz'
    for observed in (False,True):
        network = copy.deepcopy(prototype)
        optimizer = torch.optim.Adam(network.parameters(),lr=config['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
        fit_bank.bind(cache,200)
        torch.set_rng_state(rng)
        metrics = []
        with patch.object(adam_trace,'UPDATES',(1,10)):
            if observed:
                adam_trace.start(trace_path,network)
            for update in range(1,11):
                if observed:
                    adam_trace.before_update(network,optimizer,normalization,update)
                metrics.append((fit_bank if observed else original).one_update(network,optimizer,[],[0],normalization,**extras))
                if observed:
                    adam_trace.after_update(network,optimizer,normalization,update)
            if observed:
                observation = adam_trace.finish()
        outcomes.append(dict(network=network, optimizer=optimizer, metrics=metrics,
            rng=torch.get_rng_state().clone(), stats=copy.deepcopy(fine_tuning.STATS['fundamental_bank'])))
    a,b = outcomes
    assert a['metrics'] == b['metrics'] and a['stats'] == b['stats']
    assert torch.equal(a['rng'],b['rng']) and torch.equal(rng,b['rng'])
    assert all(torch.equal(p,q) and torch.equal(p.grad,q.grad) for p,q in zip(a['network'].parameters(),b['network'].parameters(),strict=True))
    for p,q in zip(a['network'].parameters(),b['network'].parameters(),strict=True):
        assert all(torch.equal(a['optimizer'].state[p][k],b['optimizer'].state[q][k]) for k in ('step','exp_avg','exp_avg_sq'))
    with ExitStack() as stack:
        archives={u:stack.enter_context(np.load(trace_path.with_name('adam-trace-%03d.npz'%u),allow_pickle=False)) for u in (1,10)}
        metadata = json.loads(archives[1]['metadata'].tobytes())
        assert metadata['updates'] == [1,10] and all(len(a.files)==7 for a in archives.values())
        errors = []
        for row in metadata['records']:
            archive = archives[row['update']]
            # Independent replay from observed pre-state, not reusing training's optimizer instance.
            prefix = 'u%03d_'%row['update']
            layer = torch.nn.Parameter(torch.from_numpy(archive[prefix+'parameters_before']).clone())
            group = row['group']
            replay = torch.optim.Adam([layer],lr=group['lr'],betas=tuple(group['betas']),eps=group['eps'],weight_decay=group['weight_decay'])
            assert len(set(row['steps_before'])) == 1
            if any(row['initialized_before']):
                replay.state[layer] = dict(step=torch.tensor(row['steps_before'][0]),
                    exp_avg=torch.from_numpy(archive[prefix+'exp_avg_before']).clone(),
                    exp_avg_sq=torch.from_numpy(archive[prefix+'exp_avg_sq_before']).clone())
            else:
                assert not archive[prefix+'exp_avg_before'].any() and not archive[prefix+'exp_avg_sq_before'].any()
            layer.grad = torch.from_numpy(archive[prefix+'gradient_clipped']).clone()
            replay.step()
            error = float((layer.detach()-torch.from_numpy(archive[prefix+'parameters_after'])).abs().max())
            assert error <= config['adam_reconstruction_absolute_tolerance']
            errors.append(error)
            raw, clipped = archive[prefix+'gradient_raw'], archive[prefix+'gradient_clipped']
            norm = float(np.linalg.norm(raw.astype(np.float64)))
            assert np.allclose(clipped, raw*min(1.,1/(norm+1e-6)),rtol=5e-6,atol=1e-10)
            assert row['steps_after'] == [float(row['update'])]*len(metadata['layout'])
    assert observation['vector_payload_bytes'] < 48*1024**2
    result = read(base)
    result.update(adam_observer=dict(status='passed', unchanged_policy=True,
        original_bank_source_sha256=sha(control/'fit_bank.py'),exact_weights_gradients_moments_steps_rng=True,
        exact_losses_order_exposures=True, disposable_updates_per_path=10,
        selected_disposable_updates=[1,10], production_capture_updates=config['capture_updates'],
        independent_adam_reconstruction_max_abs_errors=errors,
        reconstruction_absolute_tolerance=config['adam_reconstruction_absolute_tolerance'],
        clip_reconstruction_passed=True, additional_vector_payload_bound_bytes=370016*4*6*5,
        archive_cap_bytes=48*1024**2,total_artifact_cap_bytes=64*1024**2,
        trace_sha256={p.name:sha(p) for p in output.parent.glob('adam-trace-*.npz')}), qualification_optimizer_updates=20,
        qualification_only_not_retained_training=True,elapsed_s=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_sha256=sha(__file__))
    assert not torch.cuda.is_initialized()
    write(output,result)
    print('Observer transparency qualified; Adam reconstruction', errors,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','project','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    check(args.config.resolve(),args.project.resolve(),args.output.resolve())
