"""Verify the actual spatial-spectrum and cold-start breadth of the added data."""
import os
from v4_support import ROOT,BASE,load,native,write,sha

def run():
    output=BASE/'training-spectrum-01.json'
    if output.exists():raise ValueError('Fresh training-only spectrum diagnosis required')
    os.sched_setaffinity(0,{22});state,wrapper,_=load()
    import torch,numerics
    from evaluate_candidate import new_native
    from frozen_matrix import run_policy
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    rows=[]
    with torch.inference_mode():
        for population,requests in [('prior-native',native(state,'train')),('diverse-native',list(new_native(state,'diverse-train')))]:
            for identity,sample in requests:
                if identity['frame']!=0:continue
                rhs=sample['rhs'];energy=max(float(rhs.square().sum()),1e-300);fractions={}
                for mode in [3,7]:
                    axes,_=numerics.geometry(tuple(rhs.shape[2:]),mode)
                    fractions[str(mode)]=float(numerics.coefficients(rhs,axes).square().sum())/energy
                work=None
                if population=='diverse-native':_,work=run_policy(wrapper,sample,state.normalization,branch,'learned',max_passes=8)
                rows.append(dict(**identity,data_population=population,rhs_energy_fraction_below_axis_mode=fractions,
                    qualified_before_fit=None if work is None else work['qualified'],before_fit_work=work))
    write(output,dict(status='completed',rows=rows,source_sha256=sha(__file__),optimizer_updates=0,
        scope='Training-only cold snapshots. Radius/localization and geometry vary spatial spectra; the scene frequency parameter is temporal, not a spatial-wavenumber input. No new independently swept spatial-mode curriculum is claimed.'))
    print('Verified spatial-spectrum coverage and pre-fit cold-start difficulty',flush=True)

if __name__=='__main__':run()
