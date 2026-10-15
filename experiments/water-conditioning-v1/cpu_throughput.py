"""Compare persistent CPU workers on recorded pressure states, not new simulations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from context import data, measured_model, model, policy
import torch


def initialize(barrier):
    global READY
    READY=barrier


def work(arguments):
    rank,workers,threads,cpus,packet,artifacts,repeats=arguments
    os.sched_setaffinity(0,set(cpus[rank*threads:(rank+1)*threads]))
    network,normalization,_receipt,_history=measured_model(Path(artifacts))
    _reservation,parents=data.read_view(Path(packet))
    tasks=[(p,s) for p in parents if p['entry']['split']=='train' for s in range(len(p['frames']))]
    tasks=tasks[rank::workers]
    policy.require(tasks,'benchmark','nonempty distinct worker shard')
    sample=data.sample(*tasks[0],labels=False)
    with torch.no_grad():
        reference=model.predict_pressure(network,sample,normalization)
        torch.set_num_threads(threads)
        for _ in range(3): prediction=model.predict_pressure(network,sample,normalization)
        drift=float((prediction-reference).abs().max())
        clocks=[]
        for _ in range(repeats):
            READY.wait(timeout=60)
            started=time.perf_counter()
            signature=0.
            for parent,slot in tasks:
                sample=data.sample(parent,slot,labels=False)
                prediction=model.predict_pressure(network,sample,normalization)
                signature+=float(prediction.flatten()[1])
            clocks.append({'started':started,'finished':time.perf_counter(),'signature':signature})
    return {'rank':rank,'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
        'threads':torch.get_num_threads(),'cases':len(tasks),'repetitions':clocks,
        'thread_change_pressure_max_abs_difference_pa':drift,'cuda_initialized':torch.cuda.is_initialized()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','artifacts','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--layouts',default='1:8,3:8,6:4,12:2,23:1')
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    policy.require(1<=args.repeats<=5,'benchmark','one to five measured passes')
    cpus=sorted(os.sched_getaffinity(0));layouts=[]
    for item in args.layouts.split(','):
        workers,threads=map(int,item.split(':'))
        policy.require(workers>0 and threads>0 and workers*threads<=len(cpus),'cpu_budget','disjoint core groups')
        layouts.append((workers,threads))
    context=multiprocessing.get_context('spawn');records=[]
    for workers,threads in layouts:
        start=time.perf_counter()
        barrier=context.Barrier(workers)
        tasks=[(i,workers,threads,cpus,str(args.packet),str(args.artifacts),args.repeats) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize,initargs=(barrier,)) as pool:
            results=list(pool.map(work,tasks))
        passes=[]
        count=sum(r['cases'] for r in results)
        for index in range(args.repeats):
            elapsed=max(r['repetitions'][index]['finished'] for r in results)-min(r['repetitions'][index]['started'] for r in results)
            passes.append({'cases':count,'elapsed_s':elapsed,'pressure_states_per_s':count/elapsed})
        record={'workers':workers,'threads_each':threads,'total_elapsed_s_including_startup':time.perf_counter()-start,
                'passes':passes,'workers_evidence':results}
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='workers_evidence'}),flush=True)
    report={'status':'complete','scope':'pressure-inference throughput on fixed recorded training states; not time-evolving simulation throughput',
            'available_cpus':cpus,'records':records,'production_thread_policy_changed':False}
    policy.write(args.output,report)


if __name__=='__main__':main()
