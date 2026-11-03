"""Expose only a completed managed causal component and exact CPU restore."""
import argparse
from pathlib import Path
from study_support import ROOT,read,write,sha


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();pair=args.pair.resolve()
    if args.output.exists():raise ValueError('Fresh causal component entry required')
    result=read(pair/'receipt.json');restore=pair/'restore';receipt=read(restore/'receipt.json')
    project=ROOT/result['source_project']
    if result['status']!='completed'or not result['component_only']or not all(result['artifact_parity'].values()):
        raise ValueError('Complete managed component and exact CPU artifact parity required')
    if receipt['mode']!='restore'or receipt['optimizer_updates']!=0 or receipt['execution']['cuda_initialized']or not receipt['full_cpu_probe_bitwise_verified']:
        raise ValueError('Independent zero-update CPU restore required')
    if sha(project/'assembly.json')!=result['source_assembly_sha256']:raise ValueError('Captured component Project changed')
    paths=[project/'assembly.json',project/'config.json',project/'project.json',restore/'weights.npz',
        restore/'normalization.json',restore/'history.json',restore/'receipt.json',restore/'probe.npy']
    write(args.output,dict(id='b1-factored-request',format='causal-pressure-request-component-v1',
        project=str(project.relative_to(ROOT)),restore=str(restore.relative_to(ROOT)),
        identities={str(path.relative_to(ROOT)):sha(path)for path in paths},
        originating_pair_sha256=sha(pair/'receipt.json'),component_only=True,full_state_temporal_candidate=False))
    print('Frozen completed causal pressure-request component with exact CPU restore',flush=True)
