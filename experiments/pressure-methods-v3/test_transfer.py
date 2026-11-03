"""Compare actual captured 128³ predictors with the explicit research recurrence."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),str(ROOT/'experiments/pressure-pass-sweep-v1'),
    str(ROOT/'experiments/pressure-methods-v1'),str(ROOT/'experiments/pressure-methods-v2')]
from common import read,write,sha

def test(args):
    if args.output.exists():raise ValueError('Fresh transfer evidence required')
    os.sched_setaffinity(0,{22})
    import inference
    entry=read(args.entry);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    # Load captured modules first: their local coarse implementation must not be
    # shadowed by a helper with the same filename on the research search path.
    state=inference.load(dict(material,models=[entry]),entry['id'])
    import scalable
    torch=state.torch;np=state.np;wrapper=scalable.ScalableFine(state.network).eval()
    inventory=state.data.helpers().state_inventory(state.network)
    directory=ROOT/'runs/pressure-methods-v3/native-data/v3-train-00-n128-cpm16'
    receipt=read(directory/'receipt.json');rows=[]
    with torch.no_grad():
        for frame in receipt['snapshots']:
            sample={}
            for field,descriptor in frame['arrays'].items():
                path=directory/descriptor['path']
                if sha(path)!=descriptor['sha256']:raise ValueError('Training transfer field changed')
                sample[field]=torch.from_numpy(np.fromfile(path,dtype='<f8').reshape(descriptor['shape']))
            sample.update(cell_size_m=torch.tensor([1/16],dtype=torch.float64),rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
            scale=state.normalization['scales']['laplacian_rhs']['value']
            values=(sample['native_input_residual']*16000/scale).to(torch.float32)
            if not torch.equal(state.network(values),wrapper(values)):raise ValueError('Full128³ fine forward changed')
            captured=state.model.predict_pressure(state.network,sample,state.normalization)
            extended,_=scalable.predict(wrapper,sample,state.normalization,passes=2,
                kind='numerical-coarse'if args.kind=='numerical'else'learned-coarse',nodes=6)
            difference=float(torch.linalg.vector_norm(captured-extended))/max(float(torch.linalg.vector_norm(captured)),1e-300)
            if difference>1e-8:raise ValueError('Full captured128³ recurrence differs from explicit policy')
            rows.append(dict(frame=frame['index'],shape_zyx=[128]*3,raw_forward_bitwise_equal=True,
                two_pass_pressure_relative_difference=difference,tolerance=1e-8))
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('Transfer test changed model state')
    write(args.output,dict(status='passed',entry=entry,rows=rows,optimizer_updates=0,unchanged_inventory=True,
        source_sha256=sha(__file__),scalable_source_sha256=sha(ROOT/'experiments/pressure-methods-v2/scalable.py'),
        scope='Full native cold/warm128³ forward and fixed two-correction recurrence; no blind fields consumed.'))
    print('Full128³ cold/warm predictor transfer passed',entry['id'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True);parser.add_argument('--kind',choices=['learned','numerical'],required=True)
    parser.add_argument('--output',type=Path,required=True);test(parser.parse_args())
