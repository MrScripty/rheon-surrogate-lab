"""Replay exact captured independent checks without advancing a simulation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_paths(case,manifest):
    native_path=case/'native/receipt.json';episode_path=case/'episode.json'
    native=json.loads(native_path.read_text());episode=json.loads(episode_path.read_text())
    if not(native['status']=='completed'and native['native_process']['reaped']and native['native_process']['returncode']==0
        and native['adapter_manifest_sha256']==episode['manifest_sha256']==digest(manifest)):
        raise ValueError('Exact completed/reaped captured case/package required')
    steps=native['scene']['steps']
    if type(steps)is not int or steps<16:raise ValueError('Nonempty post-warmup capture schedule required')
    expected=[case/f'state-{step:04d}.npz'for step in sorted({16,steps,*range(64,steps+1,64)})]
    if sorted(case.glob('state-*.npz'))!=expected:raise ValueError('Exact nonempty captured step inventory required')
    return expected,[native_path,episode_path]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('case','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();captures,case_sources=capture_paths(a.case,a.manifest)
    sys.path[:0]=[str(HERE.parent/'dependency-migration-v1'),str(HERE.parent/'water-residual-learning-v1')]
    import model_adapter,audit
    state=model_adapter.load(a.manifest);torch=state.torch;np=state.np;audit.TORCH=torch
    oracle_path=HERE.parent/'global-pressure-v1/verify_capture.py'
    spec=importlib.util.spec_from_file_location('replayed_neighbor_oracle',oracle_path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    rows=[];sources={str(p):digest(p)for p in(a.manifest,*case_sources,Path(__file__),Path(model_adapter.__file__),Path(audit.__file__),oracle_path,Path(state.iteration.__file__))}
    with torch.no_grad():
        for path in captures:
            sources[str(path)]=digest(path)
            with np.load(path,allow_pickle=False)as z:sample={k:torch.from_numpy(z[k].copy())for k in z.files}
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
            rows.append(dict(step=int(path.stem.split('-')[1]),operator_defect=defect,operator_passed=defect<1e-9,
                mean_free_pressure_relative_l2=relative,pressure_bound=bound,bound_covers_oracle=covered,
                reconstructed_equation_quality=quality,original_gate_passed=defect<1e-9 and covered and quality['equation_passed']))
    result=dict(status='analyzed',rows=rows,all_original_checks_passed=all(r['original_gate_passed']for r in rows),
        elapsed_s=time.perf_counter()-started,cuda_initialized=torch.cuda.is_initialized(),optimizer_updates=0,simulations_advanced=0,
        sources=sources,scope='Exact existing episode oracle functions/formulas replayed on retained candidate NPZ; independent operator, spectral error coverage and reconstructed full-pressure stopping are separate. No thresholds changed. No new model predictions or native trajectories.')
    with a.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
