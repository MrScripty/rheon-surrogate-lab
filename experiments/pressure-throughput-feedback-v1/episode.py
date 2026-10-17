"""Scientific development cases via the current adapter and public rollout owner.

The adapter owns neural arithmetic; Rheon owns advancement. Sparse independent
same-RHS oracles run only after the native child has been reaped.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dependency-migration-v1'))
import qualify
from failure import classify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','bindings','binary','manifest','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--scene',type=int,required=True);p.add_argument('--fixture',action='store_true')
    p.add_argument('--native-timeout-seconds',type=int,default=300)
    a=p.parse_args();qualify.require(30<=a.native_timeout_seconds<=1800,'finite native qualification deadline required');a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt={'command':sys.argv,'source_sha256':qualify.digest(__file__),'bindings_sha256':qualify.digest(a.bindings),
             'binary_sha256':qualify.digest(a.binary),'manifest_sha256':qualify.digest(a.manifest),
             'cpu_affinity':sorted(os.sched_getaffinity(0)),'optimizer_updates':0,'fixture':a.fixture,
             'native_timeout_seconds':a.native_timeout_seconds}
    try:
        b=qualify.read(a.bindings);qualify.require(0<=a.scene<74,'bound consumed scene required')
        scene=json.loads(qualify.bound_bytes(a.root,b['regression']['membership_source']))['scenes'][a.scene]
        scene=dict(scene)
        if a.fixture:
            qualify.require(a.scene==0,'fixed scene0 first-frame fixture required');scene['steps']=16
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
        qualify.require(available>=8*1024**3,'8 GiB memory reserve required')
        qualify.require(sum(x.stat().st_size for x in a.output.parent.rglob('*') if x.is_file())<1024**3,'1 GiB raw phase budget')
        lab=Path(__file__).resolve().parents[2];sys.path.insert(0,str(lab/'experiments/water-residual-learning-v1'))
        import rollout
        import model_adapter
        build=qualify.read(a.binary.parent/'build-receipt.json')
        qualify.require(build['status']=='qualified_build' and build['binary_sha256']==receipt['binary_sha256'],'qualified source-bound candidate build required')
        for path,expected in build['source_sha256'].items():
            qualify.require(qualify.digest(path)==expected,'candidate build input changed before comparison')
        sys.path.insert(0,str(lab/'experiments/local-pressure-v1'));import dependency
        selected=dependency.selection(build['dependency_config'])
        source_root=selected.resolve(selected.value['sources']['rheon']['root'])
        native_source=build['native_receipt']
        def verify_native():
            paths=[source_root/name for name in ('Cargo.toml','Cargo.lock','rust-toolchain.toml')]
            if (source_root/'build.rs').exists():paths.append(source_root/'build.rs')
            paths += sorted(p for p in (source_root/'src').rglob('*') if p.is_file() and not p.is_relative_to(source_root/'src/bin'))
            actual=[dict(path=p.relative_to(source_root).as_posix(),sha256=qualify.digest(p)) for p in paths]
            return actual==native_source['build_inputs'] and qualify.digest(source_root/'LICENSE')==native_source['license_sha256']
        qualify.require(verify_native(),'exact compiled Rheon input closure changed')
        receipt['rheon_current_commit']=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
        receipt['native_source_identity_scope']='exact complete compiled input bytes; commit metadata may advance without a byte change'
        active_sources={str(path):qualify.digest(path) for path in (Path(rollout.__file__),Path(model_adapter.__file__),Path(__file__),Path(qualify.__file__),
            lab/'experiments/water-residual-learning-v1/audit.py',lab/'experiments/water-pressure-accuracy-v1/diagnose.py',
            lab/'experiments/global-pressure-v1/verify_capture.py',Path(__file__).with_name('failure.py'))}
        receipt.update(build_receipt_sha256=qualify.digest(a.binary.parent/'build-receipt.json'),native_source=native_source,active_sources=active_sources)
        import numpy as np
        native=rollout.run(a.binary,scene,a.output/'native-control',control=True,timeout=a.native_timeout_seconds,threads=1)
        qualify.native_metrics(a.output/'native-control',scene,receipt['binary_sha256'])
        captures=[];wanted=sorted({16,scene['steps'],*range(64,scene['steps']+1,64)})
        state_holder=[]
        def observe(request,state,reply,trace):
            state_holder[:]=[state]
            if request['context']['generation'] not in wanted:return
            sample=model_adapter.sample_from_request(request,state)
            pressure=np.asarray(reply['pressure'],dtype='<f8').reshape(sample['base'].shape)
            step=request['context']['generation'];path=a.output/f'state-{step:04d}.npz'
            with path.open('xb') as stream:np.savez(stream,**{k:v.numpy() for k,v in sample.items()},prediction=pressure)
            captures.append({'step':step,'path':str(path),'sha256':qualify.digest(path),'request_sha256':qualify.sha(qualify.canonical(request))})
        learned=rollout.run(a.binary,scene,a.output/'native',adapter_manifest=a.manifest,control=False,timeout=a.native_timeout_seconds,threads=1,observe=observe)
        qualify.require(learned['native_process']['reaped'],'native child must be reaped before oracle')
        qualify.require([r['step'] for r in captures]==wanted,'sparse capture schedule mismatch')
        state=state_holder[0];torch=state.torch;import audit
        sys.path.insert(0,str(lab/'experiments/water-pressure-accuracy-v1'));import diagnose
        audit.TORCH=torch
        spec=importlib.util.spec_from_file_location('candidate_neighbor_oracle',lab/'experiments/global-pressure-v1/verify_capture.py')
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
        samples=[]
        with torch.no_grad():
            for capture in captures:
                with np.load(capture['path'],allow_pickle=False) as z:sample={k:torch.from_numpy(z[k].copy()) for k in z.files}
                pressure=sample.pop('prediction');base=sample['base'];residual=sample['native_input_residual']
                delta=audit.spectral_correction(residual,sample);compatible=residual-residual.mean()
                action=oracle.full_action(delta.flatten().tolist(),list(delta.shape[2:][::-1]),[float(sample['cell_size_m'][0])]*3,float(sample['rho'][0]))
                independent=torch.from_numpy(np.asarray(action,dtype='<f8')).reshape_as(delta)
                def norm(v):return float(torch.linalg.vector_norm(v))
                defect=norm(independent-compatible)/max(norm(compatible),1e-30);exact=base+delta;error=pressure-exact
                relative=norm(error-error.mean())/max(norm(exact-exact.mean()),1e-30)
                rhs=residual+state.model.full_operator(base,sample['cell_size_m'],sample['rho'])
                quality=state.iteration.equation_quality(state.model,pressure,sample,rhs,state.policy['stopping'])
                bound=quality['mean_free_pressure_relative_error_bound'];covered=bound is not None and relative<=bound*(1+1e-6)+1e-11
                qualify.require(defect<1e-9 and covered and quality['equation_passed'],'independent same-RHS operator/pressure bound failed')
                samples.append({**diagnose.error_metrics(pressure,exact),**diagnose.mode_metrics(pressure,exact),'step':capture['step'],'sample_sha256':capture['sha256'],
                                'oracle_independent_relative_defect':defect,'mean_free_pressure_relative_l2':relative,
                                'pressure_bound_relative_l2':bound,'pressure_bound_covers_oracle':covered})
        with (a.output/'diagnostics.json').open('x') as stream:json.dump({'samples':samples},stream,indent=2,allow_nan=False)
        binding={**b['regression']['cases'][a.scene],'expected_ai_steps':scene['steps']-15}
        metrics=qualify.case_metrics(a.output,scene,binding,b['regression']['stopping_values'],receipt['binary_sha256'])
        if a.fixture:
            with np.load(captures[0]['path'],allow_pickle=False) as current,np.load(a.root/b['regression']['first_frame']['capture']['path'],allow_pickle=False) as old:
                for key in old.files:
                    if key=='prediction':continue  # New weights intentionally change this output.
                    qualify.require(current[key].shape==old[key].shape and current[key].dtype==old[key].dtype
                                    and current[key].tobytes()==old[key].tobytes(),'new native first-frame exact replay mismatch '+key)
            receipt['native_first_frame_inputs_exact']=True
        qualify.require(qualify.digest(a.binary)==receipt['binary_sha256'] and qualify.digest(a.manifest)==receipt['manifest_sha256'],'candidate binary/package drift during comparison')
        qualify.require(all(qualify.digest(path)==expected for path,expected in active_sources.items())
                        and verify_native(),'candidate source drift during comparison')
        receipt.update(status='passed',metrics=metrics,sparse=samples,cuda_initialized=torch.cuda.is_initialized(),torch_threads=torch.get_num_threads(),
                       captures=captures,source_sha256s={str(Path(rollout.__file__).relative_to(lab)):qualify.digest(rollout.__file__),
                                                     str(Path(model_adapter.__file__).relative_to(lab)):qualify.digest(model_adapter.__file__)})
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc),**classify(exc,a.output));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        with (a.output/'episode.json').open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','peak_rss_bytes')}))


if __name__=='__main__':main()
