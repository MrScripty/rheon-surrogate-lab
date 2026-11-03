"""Expose only completed, byte-parity-restored managed checkpoints to evaluation."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def freeze(arm,output,generation=3):
    if output.exists():raise ValueError('Fresh restored entry required')
    base=ROOT/'runs/pressure-methods-v3';project=base/(arm+f'-project-{generation:02d}');study=base/(arm+f'-study-{generation:02d}');restore=study/'restore'
    result=read(study/'receipt.json');receipt=read(restore/'receipt.json')
    if result['status']!='completed' or not all(result['artifact_parity'].values()):raise ValueError('Completed training and fresh restore parity required')
    if receipt['mode']!='restore' or receipt['optimizer_updates']!=0 or receipt['execution']['cuda_initialized']:
        raise ValueError('Fresh zero-update CPU restore required')
    for name in ['weights.npz','normalization.json','history.json']:
        if sha(restore/name)!=receipt['outputs'][name]['sha256']:raise ValueError('Restored output changed')
    identities={str(p.relative_to(ROOT)):sha(p)for p in [project/'source-witness.json',project/'config.json',
        restore/'weights.npz',restore/'normalization.json',restore/'receipt.json',restore/'history.json']}
    output.parent.mkdir(parents=True,exist_ok=True)
    write(output,dict(id=arm,project=str(project.relative_to(ROOT)),restore=str(restore.relative_to(ROOT)),
        weights_sha256=sha(restore/'weights.npz'),identities=identities))
    print('Frozen restored evaluation entry',arm,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',required=True,choices=['v3-replay','v3-native','v3-c37half','v3-c44half'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generation',type=int,choices=[3,4,5,6,7],default=3)
    args=parser.parse_args();freeze(args.arm,args.output,args.generation)
