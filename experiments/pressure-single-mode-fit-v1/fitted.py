"""Measure the explicit fitting case using a verified restored CPU checkpoint."""
import argparse
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write


def evaluate(config_path,material_path,identifier,cpu,output):
    started=time.perf_counter()
    os.sched_setaffinity(0,{cpu})
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
    case=read(config_path)['fitting_case']
    shape=tuple(reversed(case['counts_xyz']))
    template={**actual[0],'base':torch.zeros((1,1,*shape),dtype=torch.float64),
        'cell_size_m':torch.tensor([1/case['cells_per_metre']],dtype=torch.float64)}
    sample=analytical.make_sample(template,[(tuple(case['mode_zyx']),1.)],amplitude)
    denominator=torch.linalg.vector_norm(sample['rhs'])
    defect=float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/denominator)
    if defect>1e-10:
        raise ValueError('Independent label disagrees with implemented operator')
    measurements={n:[]for n in [1,2,4,8]}
    with torch.no_grad():
        inference.fixed(state,sample,8)
        for repetition in range(3):
            pressure=sample['base'];elapsed=0.
            for n in range(1,9):
                tick=time.perf_counter();pressure=inference.step(state,sample,pressure);elapsed+=time.perf_counter()-tick
                if n not in measurements:
                    continue
                tick=time.perf_counter()
                residual=sample['rhs']-state.model.full_operator(pressure,sample['cell_size_m'],sample['rho'])
                relative=float(torch.linalg.vector_norm(residual)/denominator)
                quality_s=time.perf_counter()-tick
                measurements[n].append(dict(relative_equation_residual=relative,inference_s=elapsed+quality_s,
                    pressure_rms_error_pa=float((pressure-sample['reference']).square().mean().sqrt())))
    rows=[dict(passes=n,relative_equation_residual=values[-1]['relative_equation_residual'],
        pressure_rms_error_pa=values[-1]['pressure_rms_error_pa'],
        median_inference_ms=1000*statistics.median(v['inference_s']for v in values),
        timing_samples_s=[v['inference_s']for v in values])for n,values in measurements.items()]
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only inference required')
    write(output,dict(status='passed',model=identifier,fitting_case=case,amplitude_rms_pa=amplitude,
        independent_operator_defect=defect,measurements=rows,elapsed_s=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cpu=cpu,threads=torch.get_num_threads(),
        cuda_initialized=False,material_sha256=sha(material_path),config_sha256=sha(config_path),
        weights_sha256=state.entry['weights_sha256'],source_sha256=sha(__file__)))
    print(identifier,rows,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','material','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--cpu',type=int,required=True)
    args=parser.parse_args()
    evaluate(args.config.resolve(),args.material.resolve(),args.model,args.cpu,args.output.resolve())
