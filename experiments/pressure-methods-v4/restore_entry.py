"""Expose only published completed fits with a byte-identical fresh CPU restore."""
import argparse
from pathlib import Path
from v4_support import ROOT,read,write,sha

def freeze(args):
    if args.output.exists():raise ValueError('Fresh restored checkpoint entry required')
    project=args.project;study=args.study;restore=study/'restore';result=read(study/'receipt.json');receipt=read(restore/'receipt.json')
    if result['status']!='completed'or not all(result['artifact_parity'].values()):raise ValueError('Successful fit and exact fresh restore required')
    if receipt['mode']!='restore'or receipt['optimizer_updates']!=0 or receipt['execution']['cuda_initialized']:
        raise ValueError('Zero-update CPU restore required')
    if result['assembly_sha256']!=sha(project/'assembly.json'):raise ValueError('Training assembly changed')
    for name in ['weights.npz','normalization.json','history.json']:
        if sha(restore/name)!=receipt['outputs'][name]['sha256']:raise ValueError('Restored artifact changed')
    identities={str(p.relative_to(ROOT)):sha(p)for p in [project/'source-witness.json',project/'config.json',restore/'weights.npz',restore/'normalization.json',restore/'receipt.json',restore/'history.json']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,dict(id=args.identifier,project=str(project.relative_to(ROOT)),restore=str(restore.relative_to(ROOT)),weights_sha256=sha(restore/'weights.npz'),identities=identities))
    print('Frozen completed restored entry',args.identifier,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identifier',required=True)
    for name in ['project','study','output']:parser.add_argument('--'+name,type=lambda p:Path(p).resolve(),required=True)
    freeze(parser.parse_args())
