"""Independent analytical/neighbor operator and same-RHS pressure/gauge witness."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import qualify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','bindings','manifest','frame','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt={'command':sys.argv,'source_sha256':qualify.digest(__file__),'manifest_sha256':qualify.digest(a.manifest),
             'frame_receipt_sha256':qualify.digest(a.frame/'receipt.json'),'optimizer_updates':0}
    try:
        b=qualify.read(a.bindings);source=Path(__file__).resolve().parents[2]
        for relative in ('experiments/water-residual-learning-v1/audit.py','experiments/global-pressure-v1/verify_capture.py'):
            old=subprocess.check_output(['git','-C',str(a.root),'show',b['source_commit']+':'+relative])
            qualify.require((source/relative).read_bytes()==old,'independent oracle source changed')
        import model_adapter
        state=model_adapter.load(a.manifest);np,torch=state.np,state.torch
        sample=model_adapter.sample_from_request(qualify.read(a.frame/'request.json'),state)
        sys.path.insert(0,str(source/'experiments/water-residual-learning-v1'));import audit
        audit.TORCH=torch
        spec=importlib.util.spec_from_file_location('independent_neighbor_oracle',source/'experiments/global-pressure-v1/verify_capture.py')
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
        h=float(sample['cell_size_m'][0]);rho=float(sample['rho'][0])
        def action(v):
            raw=oracle.full_action(v.flatten().tolist(),list(v.shape[2:][::-1]),[h]*3,rho)
            return torch.from_numpy(np.asarray(raw,dtype='<f8')).reshape_as(v)
        def norm(v):return float(torch.linalg.vector_norm(v))
        base=sample['base'];delta=audit.spectral_correction(sample['native_input_residual'],sample)
        compatible=sample['native_input_residual']-sample['native_input_residual'].mean()
        spectral_defect=norm(action(delta)-compatible)/max(norm(compatible),1e-30)
        current_action=state.model.full_operator(base,sample['cell_size_m'],sample['rho'])
        neighbor_defect=norm(current_action-action(base))/max(norm(action(base)),1e-30)
        qualify.require(spectral_defect<1e-9 and neighbor_defect<1e-9 and float(delta.flatten()[0])==0,'independent operator/gauge failed')
        two=torch.tensor([[[[[0.,2.]]]]],dtype=torch.float64)
        acted=state.model.full_operator(two,torch.tensor([.5],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))
        qualify.require(torch.allclose(acted,torch.tensor([[[[[-.001,.001]]]]],dtype=torch.float64),rtol=0,atol=1e-15),'analytical two-cell stencil failed')
        expected=np.load(a.frame/'expected.npy',allow_pickle=False)
        pressure=torch.from_numpy(expected.copy()).reshape_as(base);exact=base+delta
        error=pressure-exact
        relative=norm(error-error.mean())/max(norm(exact-exact.mean()),1e-30)
        rhs=sample['native_input_residual']+current_action
        quality=state.iteration.equation_quality(state.model,pressure,sample,rhs,state.policy['stopping'])
        bound=quality['mean_free_pressure_relative_error_bound']
        covered=bound is not None and relative<=bound*(1+1e-6)+1e-11
        qualify.require(covered and relative<=1e-6 and quality['equation_passed'],'same-RHS independent pressure/bound failed')
        receipt.update(status='passed',two_cell_stencil_passed=True,independent_spectral_operator_relative_defect=spectral_defect,
                       current_operator_vs_neighbor_relative_defect=neighbor_defect,cell_zero_gauge_exact=True,
                       same_rhs_mean_free_pressure_relative_l2=relative,certified_pressure_relative_bound=bound,
                       pressure_bound_covers_oracle=covered,cuda_initialized=torch.cuda.is_initialized(),torch_threads=torch.get_num_threads(),
                       cpu_affinity=sorted(os.sched_getaffinity(0)),scope='same saved request/RHS; independent trajectory pressure drift is a separate metric')
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (a.output/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','same_rhs_mean_free_pressure_relative_l2')}))


if __name__=='__main__':main()
