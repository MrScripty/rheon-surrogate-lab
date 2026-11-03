"""Post-hoc development diagnosis: global gain geometry and pressure-error bands."""
import argparse
import math
import os
from pathlib import Path
from v4_support import ROOT,read,write,sha,pressure_error

def diagnose(args):
    if args.output.exists():raise ValueError('Fresh geometry diagnosis required')
    args.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{22})
    import inference
    entry=read(args.entry);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    import torch,scalable,numerics
    from evaluate_candidate import new_native
    from frozen_matrix import run_policy
    wrapper,_=scalable.qualify(state)
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    rows=[]
    with torch.inference_mode():
        for identity,sample in new_native(state,'development'):
            if identity['frame']!=0:continue
            pressure,work=run_policy(wrapper,sample,state.normalization,branch,'learned',max_passes=8)
            error=pressure-sample['reference'];error=error-error.mean();energy=float(error.square().sum())
            fractions={}
            for bound in [3,7]:
                axes,_=numerics.geometry(tuple(error.shape[2:]),bound)
                coefficients=numerics.coefficients(error,axes)
                fractions[str(bound)]=float(coefficients.square().sum())/max(energy,1e-300)
            shape=tuple(error.shape[2:]);features=torch.tensor([math.log(n/16)for n in shape],dtype=torch.float32)
            logits=state.network.coarse_logits+state.network.shape_head(features)
            gains=max(shape)**2/2*torch.sigmoid(logits.to(torch.float64))
            _,eigenvalues=numerics.geometry(shape,3);symbol=(gains*eigenvalues.flatten()[1:])
            residual=sample['native_input_residual']-numerics.fine_action(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
            after=pressure+numerics.correction(residual,sample['cell_size_m'],sample['rho'],3)
            remaining=sample['native_input_residual']-numerics.fine_action(after-sample['base'],sample['cell_size_m'],sample['rho'])
            norm=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
            divergence=float(sample['dt'][0]*remaining.abs().max()/sample['cell_size_m'][0]**3)
            qualified=float(torch.linalg.vector_norm(remaining))<=max(1e-12,norm*1e-6)and divergence<=1e-7
            rows.append(dict(**identity,original_work=work,original_pressure_relative_l2=pressure_error(pressure,sample),
                pressure_error_energy_fraction_below_axis_mode=fractions,
                coarse_gain_times_exact_eigenvalue=dict(min=float(symbol.min()),max=float(symbol.max()),
                    mean_absolute_distance_from_one=float((symbol-1).abs().mean()),noncontracting_mode_count=int((symbol>=2).sum())),
                diagnostic_exact_low3_postcorrection=dict(qualified=qualified,pressure_relative_l2=pressure_error(after,sample),
                    relative_residual=float(torch.linalg.vector_norm(remaining))/norm,predicted_divergence=divergence)))
    write(args.output,dict(status='completed',entry=entry,rows=rows,source_sha256=sha(__file__),optimizer_updates=0,
        scope='Post-hoc diagnosis on five development cold starts. Pressure-error energy is distinct from equation-residual energy. An added exact low-mode solve is a diagnostic intervention, not a selected blind policy.'))
    print('Diagnosed pressure bands and geometry gains',entry['id'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['entry','output']:parser.add_argument('--'+name,type=Path,required=True)
    diagnose(parser.parse_args())
