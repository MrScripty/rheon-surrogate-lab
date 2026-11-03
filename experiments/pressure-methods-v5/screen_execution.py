"""Qualify and profile direct fine stencils and local inference layouts."""
import os,time,statistics
from v5_support import ROOT,HERE,BASE,read,write,sha
from solver import load,Policy
def screen():
    import torch,numerics,multilevel,fast_multilevel
    os.sched_setaffinity(0,{23});torch.set_num_threads(1);rows=[]
    from evaluate import native
    state,wrapper=load(ROOT/'runs/pressure-methods-v5/entries/v5-raw.json')
    samples=[(i,s)for i,s in native(state,'development')if max(i['counts_xyz'])>=64]
    with torch.inference_mode():
        for identity,sample in samples:
            coarse=[];reference=None
            for name,function in [('original',multilevel.correction),('direct-fine',fast_multilevel.correction)]:
                function(sample['native_input_residual'],sample['cell_size_m'],sample['rho'])
                times=[]
                for _ in range(3):
                    tick=time.perf_counter();value=function(sample['native_input_residual'],sample['cell_size_m'],sample['rho']);times.append(time.perf_counter()-tick)
                if reference is None:reference=value
                difference=float(torch.linalg.vector_norm(value-reference))/max(float(torch.linalg.vector_norm(reference)),1e-300)
                if difference>1e-12:raise ValueError('Optimized hierarchy changes the complete correction')
                coarse.append(dict(execution=name,seconds=statistics.median(times),repeat_seconds=times,relative_correction_difference=difference))
            policies=[];baseline=None
            for layout,optimized in [('eager',False),('direct-fine',True),('direct-fine-channels-last',True)]:
                policy=Policy(state,wrapper,optimized=optimized,channels_last=layout.endswith('channels-last'))
                value,work=policy.solve(sample,('warmup',identity['trajectory_id'],identity['frame']));policy.commit(work['qualified'])
                times=[]
                for j in range(3):
                    tick=time.perf_counter();value,work=policy.solve(sample,('independent',j));times.append(time.perf_counter()-tick);policy.commit(work['qualified'])
                if baseline is None:baseline=value
                policies.append(dict(execution=layout,seconds=statistics.median(times),repeat_seconds=times,work=work,
                    relative_pressure_difference=float(torch.linalg.vector_norm(value-baseline))/max(float(torch.linalg.vector_norm(baseline)),1e-300)))
            rows.append(dict(identity=identity,hierarchy=coarse,policy=policies))
    write(BASE/'execution-screen-01.json',dict(status='qualified_exploratory_screen',rows=rows,source_sha256=sha(__file__),
        dependency_sources={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver.py','fast_multilevel.py','recycling.py']},
        cpu=23,threads=1,scope='Actual reserved native requests; exploratory timings can overlap campaign work. Final serial timing is separate.'))
    print('Qualified optimized hierarchy and local-model layouts on real64/rectangular/128 requests',flush=True)
if __name__=='__main__':screen()
