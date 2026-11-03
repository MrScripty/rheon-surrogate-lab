"""Expose a head-calibrated model only after complete managed fit/restore parity."""
import argparse
from pathlib import Path
from study_support import ROOT,read,write,sha

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--identifier',required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh restored head entry required')
    pair=args.pair.resolve();result=read(pair/'receipt.json');restore=pair/'restore';receipt=read(restore/'receipt.json')
    project=ROOT/result['source_project']
    if result['status']!='completed'or not all(result['artifact_parity'].values())or result['source_assembly_sha256']!=sha(project/'assembly.json'):
        raise ValueError('Successful managed head pair and exact artifact parity required')
    if receipt['mode']!='restore'or receipt['optimizer_updates']!=0 or receipt['execution']['cuda_initialized']or not receipt['fine_tensors_bitwise_unchanged']:
        raise ValueError('Zero-update CPU restore with unchanged fine tensors required')
    files=[project/'source-witness.json',project/'config.json',restore/'weights.npz',restore/'normalization.json',restore/'receipt.json',restore/'history.json']
    write(args.output,dict(id=args.identifier,project=str(project.relative_to(ROOT)),restore=str(restore.relative_to(ROOT)),
        weights_sha256=sha(restore/'weights.npz'),identities={str(p.relative_to(ROOT)):sha(p)for p in files},
        originating_pair_sha256=sha(pair/'receipt.json'),task='pressure_head_calibration'))
    print('Frozen completed restored head entry',args.identifier,flush=True)
