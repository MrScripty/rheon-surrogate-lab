"""An explicitly untrained initialization control, never a managed fitted model."""
import argparse,copy
from pathlib import Path
from v5_support import ROOT,BASE,read,write,sha
import evaluate_v2 as evaluate
from solver_v2 import load as restored_load

def initial_load(entry_path):
    state,_=restored_load(entry_path);network=copy.deepcopy(state.network)
    seed=BASE/'tiny-initialization-01';spec=read(seed/'initialization.json')
    if sha(seed/'weights.npz')!=spec['weights_sha256']:raise ValueError('Seed control changed')
    state.data.helpers().load_weights(network,seed/'weights.npz',spec['inventory'])
    result=copy.copy(state);result.network=network
    result.entry=dict(id='v5-initial-control',model_source_project=state.entry['project'],weights_path=str((seed/'weights.npz').relative_to(ROOT)),
        weights_sha256=spec['weights_sha256'],initialization_only=True,optimizer_updates=0,managed_fit=False)
    return result,network

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['gcr','vcycle'],default='gcr');p.add_argument('--cpu',type=int,default=19);a=p.parse_args()
    args=argparse.Namespace(output=a.output,entry=ROOT/'runs/pressure-methods-v5/entries/v5-raw.json',identifier='v5-initial-'+a.method,
        method=a.method,cpu=a.cpu,population='development',old=True,numerical=False,reuse=0,optimized=False,channels_last=False)
    evaluate.load=initial_load;evaluate.run(args)
    result=read(a.output);result.update(initialization_only=True,managed_fit=False,optimizer_updates=0,control_loader_source_sha256=sha(__file__))
    write(a.output,result)
