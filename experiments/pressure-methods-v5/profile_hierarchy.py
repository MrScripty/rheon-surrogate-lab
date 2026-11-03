"""Profile the completed hierarchy on exact retained native pressure requests."""
import os,time,cProfile,pstats
from v5_support import ROOT,BASE,read,write,sha
def run():
    import numpy as np,torch,multilevel
    torch.set_num_threads(1);os.sched_setaffinity(0,{23});rows=[]
    for name in ['v4-blind-02-spectral','v4-blind-04-spectral']:
        directory=ROOT/'runs/pressure-methods-v4/native-data'/name;receipt=read(directory/'receipt.json');desc=receipt['snapshots'][0]['arrays']['native_input_residual'];path=directory/desc['path']
        if sha(path)!=desc['sha256']:raise ValueError('Real request changed')
        value=torch.from_numpy(np.fromfile(path,dtype='<f8').reshape(desc['shape']));h=torch.tensor([1/receipt['scene']['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
        multilevel.correction(value,h,rho)
        profiler=cProfile.Profile();profiler.enable()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])as trace:
            start=time.perf_counter();out=multilevel.correction(value,h,rho);seconds=time.perf_counter()-start
        profiler.disable();stats=pstats.Stats(profiler)
        rows.append(dict(counts_xyz=receipt['scene']['counts'],source_receipt_sha256=sha(directory/'receipt.json'),profiled_cycle_seconds=seconds,
            python=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3])for k,v in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:25]],
            torch=[dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total)for e in sorted(trace.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:30]]))
    write(BASE/'hierarchy-profile-01.json',dict(status='profiled',rows=rows,cpu=23,threads=1,source_sha256=sha(__file__),
        hierarchy_source_sha256=sha(ROOT/'experiments/pressure-methods-v4/multilevel.py'),timing_scope='Instrumented exploratory profile; not final quiet timing.'))
    print('Profiled retained hierarchy on actual 64³ and128³ native requests',flush=True)
if __name__=='__main__':run()
