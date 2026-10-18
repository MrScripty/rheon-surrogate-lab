"""Measure each fitted grid/axis using verified restored CPU weights."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write


def evaluate(config_path,material_path,identifier,cpu,output):
    started=time.perf_counter();os.sched_setaffinity(0,{cpu})
    sys.path.insert(0,str(ROOT/'experiments/pressure-pass-sweep-v1'))
    import inference
    state=inference.load(read(material_path),identifier)
    import analytical
    torch=state.torch
    inputs=source_inputs(ROOT/state.entry['project'])
    _,parents=state.data.load(tuple(inputs[k].read_bytes()for k in ['dataset','dataset_extra']),read(inputs['splits']))
    if any(p['entry']['split']!='train'for p in parents):
        raise ValueError('Original train-only scalar calibration required')
    actual=[state.data.sample(p,j)for p in parents for j in range(len(p['frames']))]
    amplitude=analytical.training_correction_rms(actual)
    spec=importlib.util.spec_from_file_location('cpu_fundamental_bank',HERE/'fit_bank.py')
    bank=importlib.util.module_from_spec(spec);spec.loader.exec_module(bank)
    settings=read(config_path)['fitting_bank']
    samples,cases,residency=bank.build(actual[0],settings,amplitude)
    rows=[];defects=[]
    with torch.no_grad():
        for index,(case,sample)in enumerate(zip(cases,samples,strict=True)):
            denominator=torch.linalg.vector_norm(sample['rhs'])
            defect=float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/denominator)
            if defect>1e-10:
                raise ValueError('Independent label disagrees with implemented operator')
            defects.append(defect)
            inference.fixed(state,sample,8)
            measurements=bank.measure(state.network,state.normalization,[sample])
            for value in measurements:
                rows.append({**value,'case':index,**case})
    summaries=[]
    for passes in [1,2,4,8]:
        values=[r['relative_equation_residual']for r in rows if r['passes']==passes]
        summaries.append(dict(passes=passes,cases=len(cases),equation_passes=sum(v<=1e-6 for v in values),
            worst_equation_residual=max(values),median_equation_residual=statistics.median(values)))
    if any(not math.isfinite(r['relative_equation_residual'])for r in rows):
        raise ValueError('Finite fitted measurements required')
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only inference required')
    write(output,dict(status='passed',model=identifier,cases=cases,amplitude_rms_pa=amplitude,
        independent_operator_defects=defects,summary=summaries,measurements=rows,elapsed_s=time.perf_counter()-started,
        resident_tensor_bytes=residency,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu=cpu,threads=torch.get_num_threads(),cuda_initialized=False,material_sha256=sha(material_path),
        config_sha256=sha(config_path),weights_sha256=state.entry['weights_sha256'],
        bank_source_sha256=sha(HERE/'fit_bank.py'),source_sha256=sha(__file__)))
    print(identifier,summaries,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','material','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--cpu',type=int,required=True)
    args=parser.parse_args();evaluate(args.config.resolve(),args.material.resolve(),args.model,args.cpu,args.output.resolve())
