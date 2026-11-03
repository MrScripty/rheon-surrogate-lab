"""Measure whether temporal pressure predictions depend on history or known forcing."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from global_evaluation_support import load_model


def run(args):
    if args.output.exists():
        raise ValueError('Fresh input-dependence diagnosis required')
    os.sched_setaffinity(0,{23})
    torch.set_num_threads(1)
    network,norm,identity=load_model((BASE/'b1-global-pair-01').resolve())
    bank=read(BASE/'temporal-data-01/bank.json')
    rows=[]
    def mean_free(value):
        result=torch.as_tensor(value,dtype=torch.float64).flatten()
        return result-result.mean()
    with torch.inference_mode():
        for row in bank['trajectories']:
            if row['split']not in('train','development'):
                raise ValueError('Reserved scenes must remain unopened')
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:
                raise ValueError('Reference receipt changed')
            receipt=read(path)
            scene=receipt['scene']
            def fields(step):
                record=receipt['states'][step]
                source=path.parent/record['path']
                if sha(source)!=record['sha256']:
                    raise ValueError('Reference state changed')
                with np.load(source,allow_pickle=False)as arrays:
                    return {name:arrays[name].copy()for name in('pressure','vx','vy','vz','tracer')}
            for start in(0,8,63,120):
                current=fields(start)
                truth=mean_free(fields(start+1)['pressure'])
                denominator=max(float(truth.norm()),1e-30)
                normal=mean_free(network.predict(current,scene,start,norm)[0]['pressure'])
                without_history=mean_free(network.predict(dict(current,pressure=np.zeros_like(current['pressure'])),scene,start,norm)[0]['pressure'])
                without_force=mean_free(network.predict(current,dict(scene,acceleration=0.),start,norm)[0]['pressure'])
                rows.append(dict(identifier=row['identifier'],split=row['split'],counts_xyz=scene['counts'],start=start,
                    phase='cold'if start==0 else'warm',normal_relative_pressure_error=float((normal-truth).norm())/denominator,
                    predicted_pressure_amplitude=float(normal.norm())/denominator,
                    zero_history_pressure_amplitude=float(without_history.norm())/denominator,
                    history_sensitivity=float((normal-without_history).norm())/denominator,
                    known_force_sensitivity=float((normal-without_force).norm())/denominator,
                    persistence_relative_pressure_error=float((mean_free(current['pressure'])-truth).norm())/denominator))
    summaries=[]
    for split in('train','development'):
        for phase in('cold','warm'):
            group=[row for row in rows if row['split']==split and row['phase']==phase]
            fields=('normal_relative_pressure_error','predicted_pressure_amplitude','zero_history_pressure_amplitude',
                    'history_sensitivity','known_force_sensitivity','persistence_relative_pressure_error')
            summaries.append(dict(split=split,phase=phase,outputs=len(group),
                medians={field:statistics.median(row[field]for row in group)for field in fields}))
    write(args.output,dict(status='completed',model=identity,rows=rows,summaries=summaries,
        source_sha256=sha(__file__),bank_sha256=sha(BASE/'temporal-data-01/bank.json'),optimizer_updates=0,
        reserved_scenes_consumed=False,scope='Counterfactual response diagnosis; modified inputs have no claimed future reference accuracy'))
    print('Measured pressure-history and known-force dependence on96fixed windows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
