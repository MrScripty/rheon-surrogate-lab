"""Capture an explicit native policy adapter with identical physical host gates."""
import argparse
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def prepare(output):
    if output.exists():raise ValueError('Fresh coupled adapter capture required')
    text=(HERE/'capture.py').read_text()
    text=text.replace('"""Bounded native-generated trajectories with a spectral pressure reference."""','"""Bounded coupled surrogate steps, rejected pressures never advance history."""')
    before='    h = torch.tensor('
    addition='''    sys.path[:0]=[str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v3')]
    import inference,scalable
    from frozen_matrix import run_policy,learned_branch
    entry=read(args.entry)
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    wrapper,qualification=scalable.qualify(state)
    inventory=state.data.helpers().state_inventory(state.network)
    if hasattr(state.network,'coarse_logits'):
        branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    else:branch,_=learned_branch(np,torch)
'''
    text=text.replace(before,addition+before,1)
    text=text.replace("                        pressure = base+numerics.correction(residual,h,rho,None)","                        sample=dict(base=base,native_input_residual=residual,rhs=rhs,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))\n                        pressure,work=run_policy(wrapper,sample,state.normalization,branch,args.kind,max_passes=args.passes)")
    text=text.replace('method_seconds=method_seconds,seconds=time.perf_counter()-begin)', 'method_seconds=method_seconds,seconds=time.perf_counter()-begin,work=work)')
    text=text.replace("teacher='Native Rheon advection/forcing/viscosity/projection with exact sealed-grid spectral pressure; original physical gates. Not original-CG trajectory bytes.'", "entry=entry,policy=args.kind,max_passes=args.passes,wrapper_qualification=qualification,teacher='None: actual surrogate pressure; unchanged native physical host gates; no fallback.'")
    text=text.replace("    if not completed: raise ValueError('Native spectral teacher rejected')", "    if inventory!=state.data.helpers().state_inventory(state.network):raise ValueError('Coupled policy changed weights')")
    text=text.replace("    parser.add_argument('--save-fields',action='store_true')", "    parser.add_argument('--save-fields',action='store_true')\n    parser.add_argument('--entry',type=Path,required=True)\n    parser.add_argument('--kind',choices=['numerical','learned'],required=True)\n    parser.add_argument('--passes',type=int,choices=[8,16],default=16)")
    text=text.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path('+repr(str(ROOT))+')')
    output.write_text(text)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
