"""Closed-source teacher startup and one full local target regeneration control."""
import argparse
import hashlib
import io
import json
import math
from types import SimpleNamespace
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
    qualified = read(REAL_ROOT / reg['initial_qualification'])
    args.output.mkdir(parents=True)
    tree = (REAL_ROOT / reg['unpacked_root']).resolve()
    if not tree.is_dir():
        raise ValueError('Previously decoded captured tree required; revalidate every member')
    # The pilot tree is decoded scratch. Source/input identity is revalidated;
    # exported loss assets are decoded from their separately verified capture.
    with zipfile.ZipFile(source / 'loss-assets.zip') as archive:
        for entry in archive.infolist():
            member = Path(entry.filename)
            if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Unsafe loss asset member')
            destination = tree / '.compact-loss-assets' / member
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as handle:
                    handle.write(archive.read(entry))
            spec = config['geometry_and_sparse_assets'][entry.filename]
            if destination.stat().st_size != spec['bytes'] or sha(destination) != spec['sha256']:
                raise ValueError('Decoded loss asset changed')
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
        from internal_residual_network_v2 import InternalGroupResidual
        from internal_residual_loss_v1 import FrozenCoupledTraining, metrics
        from internal_residual_dataset_v1 import TrainingView, METRICS
        from qualify_internal_objective_v2 import inventory as student_inventory
        import support
        if support.ROOT != tree:
            raise ValueError('Teacher module resolved outside captured root')
        teacher = config['teacher_configuration']
        model = InternalGroupGenerator(teacher['transport_pair'], teacher['shared_pair'], teacher['build'], teacher['control_build'])
        before = inventory(model)
        student = InternalGroupResidual(normalization=read(tree / Path(config['original_manifest']).parent / 'network-normalization.json'))
        student_before = student_inventory(student)
        if student_before != qualified['parameter_inventory']:
            raise ValueError('Qualified initial student differs')
        optimization = read(tree / Path(config['original_manifest']).parent / 'optimization-normalization.json')
        fixed = TrainingView.fixed_objective(SimpleNamespace(manifest=manifest,
            fixed={r['identifier'] for r in manifest['scene_rows'] if r['teacher_substeps']==1}), optimization)
        if fixed != qualified['fixed_controls']:
            raise ValueError('All119fixed control scenes differ')
        def asset(member):
            with np.load(tree / '.compact-loss-assets' / member, allow_pickle=False) as captured:
                return {n:captured[n].copy() for n in captured.files}
        sparse = asset(manifest['sparse_part']['member'])
        responses = {}
        specs = {(p['identifier'],p['start'],p['group_index']):p for p in manifest['parts']}
        rows = []
        totals = {m:0. for m in METRICS}
        student.zero_grad(set_to_none=True)
        runner = Transitions(model, teacher)
        generated_before_oracle = 0
        for case in config['cases']:
            scene = read(tree / case['receipt'])['scene']
            counts = tuple(scene['counts']);shape=tuple(reversed(counts));sizes=dimensions(counts)
            h,rho=1./scene['cells_per_metre'],1000.
            for start in config['starts']:
                state=next(s for s in case['states'] if s['step']==start)
                with np.load(tree / state['path'],allow_pickle=False) as captured:
                    fields={n:torch.from_numpy(captured[n].copy()).to('cuda') for n in FIELDS}
                with torch.no_grad():
                    external=model.controls.generate_from_packet(counts,model.packet(scene,start,1,8).to('cuda'))
                fine=fields;offset=0
                for group,width in enumerate((2,2,3)):
                    with torch.no_grad():
                        initial=fine
                        for j in range(width):
                            fine,events=runner.advance(fine,scene,external,offset+j,1)
                        phase=model.phase(initial,scene,external,offset,width)
                        features=encode(phase);base=model.emit(phase,float_carriers=False)
                        last=provisional(fine,events,counts);p=fine['pressure'].reshape(shape)
                        momentum=[wall(model.linear([p],size,'momentum',c)*(1./(rho*h)),c) for c,size in enumerate(sizes[:3])]
                        delta=[(a+(1./480.-phase['dt'])*m-v.double())*(phase['dt']/h) for a,m,v in zip(last,momentum,phase['provisional'])]
                        delta.append(fine['tracer'].reshape(shape).double()-phase['tracer'].double())
                        arrays={n:np.asarray(v,dtype='<f8') for n,v in [('h',h),('dt',phase['dt']),('rho',rho)]}
                        arrays['features']=features.cpu().numpy().copy()
                        for n,size in zip(FIELDS,(shape,*sizes)):
                            arrays['base_'+n]=base[n].reshape(size).cpu().numpy().copy()
                            arrays['target_'+n]=fine[n].reshape(size).cpu().numpy().copy()
                        arrays['target_rhs']=fine_action(p.reshape(1,1,*shape),p.new_tensor([h]),p.new_tensor([rho]))[0,0].cpu().numpy().copy()
                        for n,v in zip(('vx','vy','vz','tracer'),delta):arrays['target_delta_'+n]=v.cpu().numpy().copy()
                    generated_before_oracle+=1
                    spec=specs[case['identifier'],start,group]
                    scope['oracle_comparison']=True
                    with zipfile.ZipFile(oracle) as archive:raw=archive.read(spec['member'])
                    scope['oracle_comparison']=False
                    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:
                        raise ValueError('Frozen target member changed')
                    fingerprints={}
                    with np.load(io.BytesIO(raw),allow_pickle=False) as expected:
                        if set(arrays)!=set(expected.files):raise ValueError('Full target inventory differs')
                        for n,value in arrays.items():
                            old=expected[n]
                            if value.dtype.str!=old.dtype.str or value.shape!=old.shape or value.tobytes()!=old.tobytes():
                                raise ValueError('Regenerated target array differs: '+case['identifier']+':'+str(start)+':'+str(group)+':'+n)
                            fingerprints[n]=hashlib.sha256(value.tobytes()).hexdigest()
                    member=spec['geometry_member']
                    if member not in responses:responses[member]=FrozenCoupledTraining(sparse,asset(member))
                    required={'features','target_rhs',*[prefix+n for prefix in ('base_','target_') for n in FIELDS]}
                    sample={n:torch.from_numpy(arrays[n]).to('cuda') for n in required}
                    sample.update(h=h,dt=phase['dt'],rho=rho)
                    values=metrics(responses[member].emit(sample,student(sample['features'])),sample,optimization)
                    (sum(values.values())*(29./(148*783))).backward()
                    for n,value in values.items():totals[n]+=float(value.detach())/783
                    rows.append(dict(identifier=case['identifier'],start=start,group_index=group,
                        cells_per_metre=scene['cells_per_metre'],counts_xyz=list(counts),arrays_byte_exact=True,
                        array_sha256=fingerprints))
                    offset+=width
                    del arrays,raw,sample,values,features,base,delta,last,momentum
                    if (process_memory()['peak_since_exec_bytes']>reg['maximum_peak_rss_bytes'] or
                            torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or
                            time.monotonic()-started>reg['maximum_wall_seconds']):
                        raise ValueError('Full compact regeneration resource budget exceeded')
                    if len(rows)%27==0:
                        (args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=783))+'\n')
                        print('Closed compact regeneration',len(rows),'/783; exact arrays and full-objective gradients',flush=True)
        if len(rows)!=783 or generated_before_oracle!=783:
            raise ValueError('All783active samples required')
        objective=(29./148)*sum(totals.values())+fixed['objective']
        if not math.isclose(objective,qualified['full148scene_objective'],rel_tol=1e-10,abs_tol=1e-30):
            raise ValueError('Complete compact-source objective differs')
        gradients={}
        for n,parameter in student.named_parameters():
            if parameter.grad is None or not bool(parameter.grad.isfinite().all()):
                raise ValueError('Finite complete student gradients required')
            actual=dict(norm=float(parameter.grad.double().norm()),maximum=float(parameter.grad.abs().max()))
            if any(not math.isclose(v,qualified['gradients'][n][k],rel_tol=1e-6,abs_tol=1e-18) for k,v in actual.items()):
                raise ValueError('Compact-source full-objective gradient differs: '+n)
            gradients[n]=actual
        if inventory(model)!=before or student_inventory(student)!=student_before:
            raise ValueError('Regeneration changed teacher or student weights')
        result.update(status='passed_closed_compact_teacher_all783arrays_objective_and_gradients',
            rows=rows,complete_active_samples=783,all19arrays_per_sample_byte_exact=True,
            samples_generated_before_each_oracle_access=783,
            Python_ambient_repository_reads_denied=denied,captured_repository_reads=sorted(reads),
            full783sample_regeneration_qualified=True,full148scene_objective=objective,
            active_metric_means=totals,fixed_controls=fixed,student_gradients=gradients,
            student_and_teacher_weights_unchanged=True,fitting_allowed=False,
            fitting_note='This qualification does not execute or register a fit. A managed trainer must consume the captured code/input bundles and pass its own source/runtime/objective gates.',
            memory_counters=process_memory(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-started,source_sha256=reg['source_sha256'])
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
