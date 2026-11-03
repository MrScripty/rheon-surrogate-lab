"""Full 128³ cold/warm forward and fixed deployed-method transfer qualification."""
import argparse
import os
from pathlib import Path
from v4_support import ROOT,read,write,sha

def check(args):
    if args.output.exists():raise ValueError('Fresh full-grid transfer result required')
    args.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{22})
    import inference
    entry=read(args.entry);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    import torch,scalable,numerics
    from preconditioned import gcr
    wrapper=scalable.ScalableFine(state.network).eval();inventory=state.data.helpers().state_inventory(state.network)
    directory=ROOT/'runs/pressure-methods-v3/native-data/v3-train-00-n128-cpm16';receipt=read(directory/'receipt.json')
    rows=[]
    with torch.inference_mode():
        for frame in receipt['snapshots']:
            sample={}
            for field,spec in frame['arrays'].items():
                path=directory/spec['path']
                if sha(path)!=spec['sha256']:raise ValueError('Transfer training field changed')
                sample[field]=torch.from_numpy(state.np.fromfile(path,dtype='<f8').reshape(spec['shape']))
            h=torch.tensor([1/16],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
            sample.update(cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))
            values=(sample['native_input_residual']*16000/state.normalization['scales']['laplacian_rhs']['value']).float()
            if not torch.equal(state.network(values),wrapper(values)):raise ValueError('Full128 fine forward differs')
            captured=state.model.predict_pressure(state.network,sample,state.normalization)
            if 'solver_training'in state.model.ARCHITECTURE:
                zero=torch.zeros_like(sample['base'])
                def action(p):return numerics.fine_action(p,h,rho)
                def proposal(r):
                    local=dict(sample,base=zero,native_input_residual=r,rhs=r)
                    return scalable.predict(wrapper,local,state.normalization,passes=1,kind='learned-coarse')[0]
                delta,work=gcr(sample['native_input_residual'],action,proposal,lambda _:False,max_iterations=2)
                extended=sample['base']+delta;method='Two GCR iterations with fixed one-step neural proposals'
            else:
                extended,_=scalable.predict(wrapper,sample,state.normalization,passes=2,kind='learned-coarse')
                method='Two learned-coarse/fine corrections'
            difference=float(torch.linalg.vector_norm(captured-extended))/max(float(torch.linalg.vector_norm(captured)),1e-300)
            if difference>1e-8:raise ValueError('Captured full128 recurrence differs from independent wrapper')
            rows.append(dict(frame=frame['index'],raw_forward_bitwise_equal=True,pressure_relative_difference=difference,tolerance=1e-8,method=method))
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('Transfer qualification changed model state')
    write(args.output,dict(status='passed',entry=entry,rows=rows,optimizer_updates=0,unchanged_inventory=True,
        source_sha256=sha(__file__),source_receipt_sha256=sha(directory/'receipt.json'),
        scope='Full native training-only cold/warm128-cubed fields, raw CNN bitwise parity and complete fixed two-correction deployment-method agreement.'))
    print('Full128 predictor transfer passed',entry['id'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['entry','output']:parser.add_argument('--'+name,type=Path,required=True)
    check(parser.parse_args())
