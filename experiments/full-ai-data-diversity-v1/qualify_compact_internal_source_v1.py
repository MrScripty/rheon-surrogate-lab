"""Closed-source teacher startup and one full local target regeneration control."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import zipfile

import numpy as np
import torch

REAL_ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    args.output = args.output.resolve()
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only compact-source qualification required')
    reg = read(args.protocol)
    for p, expected in reg['source_sha256'].items():
        if sha(REAL_ROOT / p) != expected:
            raise ValueError('Frozen compact-source dependency changed: ' + p)
    source = REAL_ROOT / reg['compact_source']
    receipt = read(source / 'receipt.json')
    for name, spec in receipt['outputs'].items():
        path = source / name
        if path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError('Complete compact source changed')
    config = read(source / 'config.json')
    args.output.mkdir(parents=True)
    tree = args.output / 'captured-root'
    tree.mkdir()
    for name in ('teacher-code.zip', 'teacher-inputs.zip'):
        with zipfile.ZipFile(source / name) as archive:
            for entry in archive.infolist():
                p = Path(entry.filename)
                if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError('Unsafe captured member path')
                archive.extract(entry, tree)
    for name, spec in {**config['source_code'], **config['captured_inputs']}.items():
        p = tree / name
        if p.stat().st_size != spec['bytes'] or sha(p) != spec['sha256']:
            raise ValueError('Extracted source/input identity differs: ' + name)
    manifest = read(tree / config['original_manifest'])
    if (tree / Path(config['original_manifest']).parent / manifest['archive']['path']).exists():
        raise ValueError('Precomputed active future labels must be absent')
    # Remove ambient repository module lookup before importing any teacher code.
    sys.path[:] = [p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(REAL_ROOT)]
    sys.path[:0] = [str(tree / 'experiments' / n) for n in
        ('full-ai-data-diversity-v1', 'full-ai-tight-targets-v1', 'full-ai-next-priorities-v1')]
    original_cwd = Path.cwd()
    os.chdir(tree)
    reads, denied = set(), []
    scope = {'oracle_comparison': False}
    oracle = (REAL_ROOT / reg['oracle_archive']).resolve()
    def audit(event, arguments):
        if event != 'open' or not isinstance(arguments[0], (str, bytes)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.is_relative_to(tree):
            reads.add(str(path.relative_to(tree)))
            return
        if not path.is_relative_to(REAL_ROOT):
            return  # Qualified Python/Torch assets and trusted host libraries.
        mode = arguments[1]
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        if writing and path.is_relative_to(args.output):
            return
        if scope['oracle_comparison'] and path == oracle:
            return
        denied.append(str(path))
        raise PermissionError('Ambient repository read is forbidden during closed teacher generation')
    sys.addaudithook(audit)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    result = dict(status='running', optimizer_updates=0, new_parameter_fits=0,
        cpu_model_forward=False, expanded_reserved52_references_opened=0,
        full_generator_target_pass=False, fitting_allowed=False,
        inference_device='cuda', display_label='GPU (CUDA)')
    model = None
    started = time.monotonic()
    try:
        from internal_group_residual_v1 import InternalGroupGenerator, encode
        from internal_residual_network_v1 import wall
        from diagnose_group_transition_v1 import Transitions
        from qualify_group_residual_targets_v2 import provisional
        from recurrent_sampler_native import dimensions
        from cuda_native_transfer_v1 import FIELDS
        from numerics import fine_action
        from capture_memory_telemetry_v1 import process_memory
        from diagnose_causal_composition import inventory
        import support
        if support.ROOT != tree:
            raise ValueError('Teacher module resolved outside captured root')
        teacher = config['teacher_configuration']
        model = InternalGroupGenerator(teacher['transport_pair'], teacher['shared_pair'], teacher['build'], teacher['control_build'])
        before = inventory(model)
        case = next(c for c in config['cases'] if c['identifier'] == reg['identifier'])
        state = next(s for s in case['states'] if s['step'] == reg['start'])
        scene = read(tree / case['receipt'])['scene']
        with np.load(tree / state['path'], allow_pickle=False) as captured:
            fields = {n: torch.from_numpy(captured[n].copy()).to('cuda') for n in FIELDS}
        counts = tuple(scene['counts'])
        shape = tuple(reversed(counts))
        sizes = dimensions(counts)
        h, rho, width = 1. / scene['cells_per_metre'], 1000., 2
        runner = Transitions(model, teacher)
        with torch.no_grad():
            external = model.controls.generate_from_packet(counts, model.packet(scene, reg['start'], 1, 8).to('cuda'))
            fine = fields
            for j in range(width):
                fine, events = runner.advance(fine, scene, external, j, 1)
            phase = model.phase(fields, scene, external, 0, width)
            features = encode(phase)
            base = model.emit(phase, float_carriers=False)
            last = provisional(fine, events, counts)
            p = fine['pressure'].reshape(shape)
            momentum = [wall(model.linear([p], s, 'momentum', c) * (1. / (rho*h)), c) for c, s in enumerate(sizes[:3])]
            delta = [(a + (1./480. - phase['dt'])*m - v.double()) * (phase['dt']/h) for a, m, v in zip(last, momentum, phase['provisional'])]
            delta.append(fine['tracer'].reshape(shape).double() - phase['tracer'].double())
            arrays = {n: np.asarray(v, dtype='<f8') for n, v in [('h', h), ('dt', phase['dt']), ('rho', rho)]}
            arrays['features'] = features.cpu().numpy().copy()
            for n, s in zip(FIELDS, (shape, *sizes)):
                arrays['base_'+n] = base[n].reshape(s).cpu().numpy().copy()
                arrays['target_'+n] = fine[n].reshape(s).cpu().numpy().copy()
            arrays['target_rhs'] = fine_action(p.reshape(1,1,*shape), p.new_tensor([h]), p.new_tensor([rho]))[0,0].cpu().numpy().copy()
            for n, v in zip(('vx','vy','vz','tracer'), delta):
                arrays['target_delta_'+n] = v.cpu().numpy().copy()
        if inventory(model) != before:
            raise ValueError('Teacher generation changed learned assets')
        result['closed_generation_completed_before_oracle_open'] = True
        # The precomputed reference is accessed only after generation, solely by
        # this qualification comparator, never by the compact teacher itself.
        spec = next(x for x in manifest['parts'] if x['identifier'] == reg['identifier'] and x['start'] == reg['start'] and x['group_index'] == 0)
        scope['oracle_comparison'] = True
        with zipfile.ZipFile(oracle) as archive:
            raw = archive.read(spec['member'])
        if hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen comparison member changed')
        scope['oracle_comparison'] = False
        comparisons = []
        with np.load(io.BytesIO(raw), allow_pickle=False) as expected:
            if set(arrays) != set(expected.files):
                raise ValueError('Regenerated complete target inventory differs')
            for n, value in arrays.items():
                old = expected[n]
                exact = value.dtype.str == old.dtype.str and value.shape == old.shape and value.tobytes() == old.tobytes()
                comparisons.append(dict(name=n, dtype=value.dtype.str, shape=list(value.shape), byte_exact=exact,
                    regenerated_sha256=hashlib.sha256(value.tobytes()).hexdigest(), expected_sha256=hashlib.sha256(old.tobytes()).hexdigest()))
        memory = process_memory()
        if memory['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']:
            raise ValueError('Compact source pilot memory budget exceeded')
        if time.monotonic()-started > reg['maximum_wall_seconds']:
            raise ValueError('Compact source pilot deadline exceeded')
        result.update(status='passed_closed_compact_teacher_one_group' if all(r['byte_exact'] for r in comparisons) else 'regenerated_arrays_differ',
            comparisons=comparisons, all_arrays_byte_exact=all(r['byte_exact'] for r in comparisons),
            identifier=reg['identifier'], start=reg['start'], group_index=0,
            Python_ambient_repository_reads_denied=denied, captured_repository_reads=sorted(reads),
            full783sample_regeneration_qualified=False, teacher_parameters_unchanged=True,
            memory_counters=memory, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-started, source_sha256=reg['source_sha256'])
        if not result['all_arrays_byte_exact']:
            raise ValueError('Closed-source target regeneration differs')
    except BaseException as error:
        result.update(status='failed', error=repr(error), captured_repository_reads=sorted(reads),
            Python_ambient_repository_reads_denied=denied)
        raise
    finally:
        if model is not None:
            model.close()
        os.chdir(original_cwd)
        (args.output / 'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
