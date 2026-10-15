"""Repeat a captured learned corrector with freshly computed residuals on CPU.

This inference policy adds no native solver/refinement. It preserves the supplied
previous pressure, fixed cubic operator and checkpoint. Count every proposal;
each proposal already contains the checkpoint's configured recurrent passes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]


def correct(model,network,normalization,sample,budget):
    if type(budget) is not int or not 1<=budget<=8:raise ValueError('one to eight learned proposals')
    initial=sample['base'];pressure=initial
    for index in range(budget):
        residual=sample['native_input_residual'] if index==0 else sample['native_input_residual']-model.full_operator(pressure-initial,sample['cell_size_m'],sample['rho'])
        pressure=model.predict_pressure(network,{**sample,'base':pressure,'native_input_residual':residual},normalization)
    return pressure


def predict(task):
    import audit
    parent,slot,case,budget=task
    sample=audit.DATA.sample(parent,slot,labels=False)
    with audit.TORCH.no_grad():
        pressure=correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,budget)
    return case,pressure[0,0].numpy().copy(),os.getpid(),sorted(os.sched_getaffinity(0)),audit.TORCH.cuda.is_initialized()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--population',choices=('train','validation'),required=True)
    parser.add_argument('--proposals',type=int,choices=range(1,9),required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
    import audit
    sys.path.insert(0,str(args.project.absolute()))
    import data
    import numpy as np
    reservation,parents=data.read_view(args.packet.absolute())
    cases=[{'parent':i,'frame':slot,'native_frame':frame['index'],'trajectory_id':parent['entry']['trajectory_id'],
        'split':args.population,'counts_xyz':parent['counts_xyz'],'cell_size_m':parent['cell_size_m']}
        for i,parent in enumerate(parents) if parent['entry']['split']==args.population for slot,frame in enumerate(parent['frames'])]
    if not cases:raise ValueError('nonempty selected population required')
    tasks=[(parents[c['parent']],c['frame'],c,args.proposals) for c in cases]
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus
    args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(tasks)),mp_context=multiprocessing.get_context('spawn'),
        initializer=audit.initialize,initargs=(args.project.absolute(),args.artifacts.absolute(),cpus)) as pool:
        records=list(pool.map(predict,tasks))
    arrays={f'case_{i:03d}':r[1] for i,r in enumerate(records)}
    with (args.output/'predictions.npz').open('xb') as stream:np.savez(stream,**arrays)
    source=json.loads((args.artifacts/'receipt.json').read_text())
    config=json.loads((args.project/'config.json').read_text())
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt={'schema':'rheon-learned-iteration-diagnostic-v1','optimizer_updates':0,
        'scope':'local CPU diagnostic from a managed restore; not a managed inference Run',
        'source_cpu_restore':str(args.artifacts.absolute()),'source_cpu_restore_sha256':digest(args.artifacts/'receipt.json'),
        'source_witness_sha256':digest(args.project/'source-witness.json'),
        'weights_sha256':source['outputs']['weights.npz']['sha256'],
        'normalization_sha256':source['outputs']['normalization.json']['sha256'],
        'packet_experiment_sha256':hashlib.sha256(data.canonical(reservation)).hexdigest(),
        'cases':cases,'prediction_inventory':data.inventory(arrays),'predictions_sha256':digest(args.output/'predictions.npz'),
        'runtime':{'device':'cpu','cuda_initialized':any(r[4] for r in records)},
        'inference_policy':{'name':'fixed-repeated-learned-correction','proposals':args.proposals,
            'nn_passes_per_pressure_solve':args.proposals*config['correction']['steps'],
            'native_pressure_refinement':False,'cpu_physical_arithmetic':'unchanged float64 operator/base accumulation'},
        'used_cpus':sorted({cpu for r in records for cpu in r[3]}),
        'source_sha256':{str(Path(__file__).relative_to(LAB)):digest(Path(__file__)),
            str(Path(audit.__file__).relative_to(LAB)):digest(Path(audit.__file__))}}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(cases),'policy':receipt['inference_policy'],'cuda_initialized':receipt['runtime']['cuda_initialized']}),flush=True)


if __name__=='__main__':main()
