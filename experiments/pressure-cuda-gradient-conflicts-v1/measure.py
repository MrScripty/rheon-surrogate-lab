"""Sequential CUDA backward diagnostics on published frozen training states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write
from geometry import projection, layers


def measure(config_path, directory, output):
    import numpy as np
    import torch
    started = time.perf_counter()
    config = read(config_path)
    if output.exists() or directory.exists():
        raise ValueError('Fresh diagnostic result and vector directory required')
    directory.mkdir(parents=True)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    project, study = ROOT/config['project'], ROOT/config['study']
    restored = study/'restore'
    receipt = read(study/'receipt.json')
    assert receipt['status'] == 'completed' and all(receipt['artifact_parity'].values())
    for phase in ('training', 'restore'):
        terminal = read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        assert terminal['state'] == 'completed' and terminal['custody_phase'] == 'drained'
    witness = read(project/'source-witness.json')
    for name, identity in witness['sources'].items():
        assert sha(project/name) == identity, name
    sys.path.insert(0, str(project))
    import model
    import data
    import fine_tuning
    import fit_bank
    import physical_residual
    from gradient_clipping import clip_l2_
    model.configure('cuda')
    model.select(read(project/'config.json'))
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(config['gpu_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    network = model.PressureCorrectionNet().to('cuda')
    initial = ROOT/next(r['path'] for r in read(project/'project.json')['data'] if r['id'] == 'initial_weights')
    fine_tuning.initialize(network, initial, data.helpers())
    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ('dataset', 'dataset_extra')), read(inputs['splits']))
    assert all(p['entry']['split'] == 'train' for p in parents)
    cache = {(i,j): data.sample(p,j,device='cuda') for i,p in enumerate(parents) for j in range(len(p['frames']))}
    fit_bank.bind(cache,200)
    samples = fit_bank.SAMPLES
    cases = fine_tuning.STATS['fundamental_bank']['cases']
    assert cases == receipt['training_receipt']['fine_tuning_statistics']['fundamental_bank']['cases']
    normalization = read(restored/'normalization.json')
    assert fine_tuning.SPEC['training_passes'] == 4 and fine_tuning.SPEC['teacher_weight'] == 0
    assert fine_tuning.SPEC['early_weight'] == .05 and fine_tuning.SPEC['relative_equation_target'] == 1e-6
    named = list(network.named_parameters())
    parameters = [p for _,p in named]
    count = sum(p.numel() for p in parameters)
    layout = [dict(name=n, shape=list(p.shape), count=p.numel()) for n,p in named]
    assert 128*count <= config['vector_workspace_limit_bytes']
    assert len(config['updates'])*40*count*4 + 120*256 <= config['vector_disk_limit_bytes']
    checkpoints, sources = [], [config_path, project/'source-witness.json', study/'receipt.json',
                               HERE/'measure.py', HERE/'geometry.py']
    graph_calls = 0

    def flatten(values):
        return torch.cat([v.detach().reshape(-1) for v in values])

    def host(values):
        array = flatten(values).cpu().numpy()
        assert array.dtype == np.float32 and array.shape == (count,) and np.isfinite(array).all()
        return array

    def state_hash():
        return hashlib.sha256(host(parameters).tobytes()).hexdigest()

    for update in config['updates']:
        clock = time.perf_counter()
        trace = restored/('adam-trace-%03d.npz'%update)
        assert sha(trace) == sha(study/'train'/trace.name)
        assert sha(trace) == receipt['training_receipt']['outputs'][trace.name]['sha256'] == receipt['restore_receipt']['outputs'][trace.name]['sha256']
        sources.append(trace)
        with zipfile.ZipFile(trace) as archive:
            assert archive.testzip() is None and sum(i.file_size for i in archive.infolist()) <= 12*1024**2
        with np.load(trace, allow_pickle=False) as archive:
            meta = json.loads(archive['metadata'].tobytes())
            assert meta['layout'] == layout and meta['parameter_count'] == count
            capture = next(r for r in meta['records'] if r['update'] == update)
            prefix = 'u%03d_'%update
            before, after, raw, clipped = [archive[prefix+k].copy() for k in
                ('parameters_before','parameters_after','gradient_raw','gradient_clipped')]
        cursor = 0
        with torch.no_grad():
            for parameter in parameters:
                parameter.copy_(torch.from_numpy(before[cursor:cursor+parameter.numel()]).reshape(parameter.shape))
                cursor += parameter.numel()
        assert cursor == count
        identity = state_hash()
        assert identity == hashlib.sha256(before.tobytes()).hexdigest()
        network.train()
        network.zero_grad(set_to_none=True)
        order = fit_bank.ORDER[(update-1)*8:update*8].tolist()
        assert sorted(order) == list(range(8))
        # Replay the original backward accumulation, including division before backward.
        values = [None]*8
        for index in order:
            value = fit_bank.loss(network, samples[index], normalization)
            values[index] = float(value.detach())
            (value/8).backward()
            graph_calls += 1
            del value
        reconstructed = host([p.grad for p in parameters])
        raw_gap = float(np.linalg.norm(reconstructed.astype('float64')-raw)/np.linalg.norm(raw.astype('float64')))
        total_norm = float(clip_l2_(parameters,1.))
        reconstructed_clipped = host([p.grad for p in parameters])
        clipped_gap = float(np.linalg.norm(reconstructed_clipped.astype('float64')-clipped)/np.linalg.norm(clipped.astype('float64')))
        value_gaps = [abs(a-b)/max(abs(b),1e-30) for a,b in zip(values,capture['before']['losses'],strict=True)]
        replay = dict(raw_relative_l2_gap=raw_gap, clipped_relative_l2_gap=clipped_gap,
                      raw_exact=bool(np.array_equal(reconstructed,raw)), clipped_exact=bool(np.array_equal(reconstructed_clipped,clipped)),
                      maximum_loss_relative_gap=max(value_gaps), observed_clip_norm=total_norm,
                      order=order, losses=values)
        write(directory/('replay-%03d.json'%update),replay)
        if (max(raw_gap,clipped_gap)>config['gradient_relative_l2_tolerance'] or
                max(value_gaps)>config['value_relative_tolerance']):
            raise ValueError('Captured CUDA gradient/loss replay qualification failed')
        network.zero_grad(set_to_none=True)
        del reconstructed, reconstructed_clipped
        rows = []
        for index, sample in enumerate(samples):
            for term in ['full',1,2,3,4]:
                torch.cuda.synchronize()
                term_clock = time.perf_counter()
                if term == 'full':
                    loss = fit_bank.loss(network,sample,normalization)
                    relative = None
                else:
                    prediction = physical_residual.predict_pressure(network,sample,normalization,model.owner,
                        {'activation':'linear','steps':term})
                    remaining = sample['native_input_residual']-model.full_operator(
                        prediction-sample['base'],sample['cell_size_m'],sample['rho'])
                    square = fine_tuning.relative_square(remaining,sample['native_input_residual'])
                    relative = float(square.detach().sqrt().mean())
                    loss = torch.log1p(square/1e-12).mean()
                value = float(loss.detach())
                gradient = torch.autograd.grad(loss,parameters)
                vector = host(gradient)
                graph_calls += 1
                path = directory/('%03d-case-%d-%s.npy'%(update,index,term))
                np.save(path,vector,allow_pickle=False)
                rows.append(dict(case=index,term=str(term),loss=value,relative_residual=relative,
                                 gradient_l2=float(np.linalg.norm(vector.astype('float64'))),
                                 vector=str(path.relative_to(ROOT)),sha256=sha(path),
                                 elapsed_s=time.perf_counter()-term_clock))
                del gradient,vector,loss
                if term != 'full':
                    del prediction,remaining,square
                assert all(p.grad is None for p in parameters)
        def vector(index,term):
            return np.load(directory/('%03d-case-%d-%s.npy'%(update,index,term)),allow_pickle=False)
        # Full per-case gradients were unscaled; separately disclose their mean's gap.
        mean = np.zeros(count,dtype=np.float64)
        for index in order:
            mean += vector(index,'full')/8
        separate_gap = float(np.linalg.norm(mean-raw)/np.linalg.norm(raw.astype('float64')))
        weak = config['weak_case']
        weak_final, weak_full = vector(weak,4), vector(weak,'full')
        delta = after.astype('float64')-before
        for row in rows:
            value = vector(row['case'],row['term'])
            row.update(vs_weak_final=projection(value,weak_final),vs_weak_full=projection(value,weak_full),
                       vs_captured_mean=projection(value,raw),vs_observed_delta=projection(value,delta))
        layer_rows = [dict(case=i,passes=n,layers=layers(weak_final,vector(i,n),layout))
                      for i in config['layer_cases'] for n in config['layer_passes']]
        assert state_hash() == identity and fine_tuning.STATS['teacher_forwards'] == 0
        assert fine_tuning.STATS['fundamental_bank']['optimizer_updates'] == 0
        torch.cuda.synchronize()
        checkpoints.append(dict(update=update,learning_rate=capture['group']['lr'],cases=cases,rows=rows,
                                layer_comparisons=layer_rows,cuda_replay=replay,
                                separate_full_gradient_mean_relative_l2_gap=separate_gap,
                                observed_parameter_sha256=identity,network_unchanged=True,
                                elapsed_s=time.perf_counter()-clock))
        write(directory/'progress.json',dict(completed_updates=[s['update']for s in checkpoints],elapsed_s=time.perf_counter()-started))
        print('Measured CUDA update',update,'replay gap',raw_gap,'seconds',time.perf_counter()-clock,flush=True)
    peak = torch.cuda.max_memory_allocated()
    assert peak <= config['gpu_limit_bytes']
    disk = sum(p.stat().st_size for p in directory.glob('*.npy'))
    assert disk <= config['vector_disk_limit_bytes']
    write(output,dict(schema='rheon-cuda-gradient-conflicts-v1',status='completed',config=config,snapshots=checkpoints,
                      layout=layout,parameter_count=count,graph_calls=graph_calls,maximum_live_forward_graphs=1,
                      elapsed_s=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      gpu_peak_allocated_bytes=peak,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gpu_name=torch.cuda.get_device_name(0),torch=str(torch.__version__),numpy=str(np.__version__),
                      network_dtype='float32',physical_dtype='float64',device='cuda:0',amp=False,tf32=False,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads(),
                      vector_disk_bytes=disk,vector_workspace_bound_bytes=128*count,
                      optimizer_updates=0,model_promoted=False,baseline_retained='C53',
                      sources={str(p.relative_to(ROOT)):sha(p)for p in sources}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','vectors','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    measure(args.config.resolve(),args.vectors.resolve(),args.output.resolve())
