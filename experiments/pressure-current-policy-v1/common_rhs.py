"""Compare frozen models on the same qualified train-only native-amplitude RHS."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sys
import time

import prepare


def initialize(root,config,project,artifacts,bank):
    global BANK,CONFIG,ROOT
    ROOT=Path(root);CONFIG=config;BANK=Path(bank)
    prepare.audit.initialize(ROOT/project,ROOT/artifacts,config['cpus'])
    prepare.CONFIG=config
    spec=importlib.util.spec_from_file_location('common_rhs_independent_oracle',prepare.LAB/'experiments/global-pressure-v1/verify_capture.py')
    prepare.ORACLE=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare.ORACLE)
    vm=int(next(x.split()[1]for x in Path('/proc/self/status').read_text().splitlines()if x.startswith('VmSize:')))*1024
    resource.setrlimit(resource.RLIMIT_AS,(vm+config['worker_growth_bytes'],)*2)


def probe(task):
    import numpy as np
    index,record,amplitude=task;torch=prepare.audit.TORCH
    path=(BANK.parent/record['pair_path']).resolve(strict=True)
    if not path.is_relative_to(BANK.parent)or prepare.digest(path)!=record['pair_sha256']:raise ValueError('common training pair changed')
    rows=[]
    with np.load(path,allow_pickle=False)as arrays,torch.no_grad():
        for stage in ('early','late'):
            rhs=torch.from_numpy(arrays[stage+'_rhs'].copy());reference=torch.from_numpy(arrays[stage+'_reference'].copy())
            sample={'base':torch.zeros_like(rhs),'native_input_residual':rhs,
                **{key:torch.tensor([record[key]],dtype=torch.float64)for key in ('rho','dt','cell_size_m')}}
            measured,compatible,truth=prepare.probe(sample,amplitude)
            error=prepare.norm(prepare.mean_free(truth-reference))/max(prepare.norm(prepare.mean_free(reference)),1e-300)
            if not error<1e-9:raise ValueError('common independent pressure label differs')
            rows.append({'stage':stage,**measured,'label_consistency_relative_l2':error})
    prepare.audit.BASES.clear()
    return {'case':index,'identity':record['identity'],'pair_sha256':record['pair_sha256'],'rows':rows,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_initialized':torch.cuda.is_initialized(),
        'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','preparation','project','artifacts','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);config=json.loads(a.config.read_text());preparation=a.preparation.resolve(strict=True)/'receipt.json';source=json.loads(preparation.read_text())
    if source['status']!='passed'or source['fixture']or len(source['cases'])!=144:raise ValueError('qualified144state preparation required')
    output=prepare.destination(root,a.output);config['cpus']=[cpu for cpu in config['cpus']if cpu in os.sched_getaffinity(0)]
    workers=min(config['workers'],len(config['cpus']));available=prepare.available()
    if available<workers*config['worker_reservation_bytes']+config['minimum_available_bytes']:raise MemoryError('common RHS memory reservation unavailable')
    output.mkdir(parents=True);started=time.perf_counter();project=a.project.resolve(strict=True);artifacts=a.artifacts.resolve(strict=True)
    if not project.is_relative_to(root)or not artifacts.is_relative_to(root):raise ValueError('root-contained frozen model required')
    amplitude=source['cases'][0]['early']['target_rms_pa']*source['cases'][0]['early']['rescale_gain_to_train_rms']
    tasks=[(i,c,amplitude)for i,c in enumerate(source['cases'])]
    sources={str(path):prepare.digest(path)for path in (Path(__file__),Path(prepare.__file__),Path(prepare.audit.__file__),prepare.LAB/'experiments/global-pressure-v1/verify_capture.py',prepare.LAB/'experiments/water-stability-learning-v1/learned_iteration.py')}
    # A fixed finite task set and per-probe reserve checks; executor drains on
    # failure. Concurrent timings are diagnostic, never inference latency claims.
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
        initargs=(str(root),config,str(project),str(artifacts),str(preparation)))
    def interrupted(*_):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise InterruptedError('common RHS owner cancelled; drain finite active work')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        rows=list(pool.map(probe,tasks,buffersize=workers));pool.shutdown(wait=True)
    except BaseException as error:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        pool.shutdown(wait=True,cancel_futures=True)
        prepare.write(output/'failure.json',{'status':'failed','error':str(error),'workers_drained':True});raise
    if any(prepare.digest(Path(path))!=identity for path,identity in sources.items()):raise ValueError('common RHS executed source changed')
    result={'format':'pressure_current_common_rhs_v1','status':'passed','weights_sha256':prepare.digest(artifacts/'weights.npz'),
        'normalization_sha256':prepare.digest(artifacts/'normalization.json'),'source_witness_sha256':prepare.digest(project/'source-witness.json'),
        'preparation_sha256':prepare.digest(preparation),'sources':sources,'cases':rows,'workers':workers,'all_workers_reaped':True,
        'network_calls':len(rows)*8,'elapsed_s':time.perf_counter()-started,'available_at_start_bytes':available,
        'optimizer_updates':0,'simulations_advanced':0,'scope':'Exactly shared training RHS and independently qualified pressure labels, native/rescaled2call probes; concurrent CPU timing is not inference latency. No autonomous drift or causal noise-floor conclusion.'}
    prepare.write(output/'receipt.json',result);print(json.dumps({k:result[k]for k in ('status','workers','network_calls','elapsed_s')}))

if __name__=='__main__':main()
