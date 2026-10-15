"""Evaluate one restored checkpoint under explicit bounded CPU iteration."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

LAB=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','binary','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--population',choices=('train','validation'),required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    prediction=args.output/'inference';evaluation=args.output/'native'
    commands=[[sys.executable,str(LAB/'experiments/water-stability-learning-v1/learned_iteration.py'),
        '--project',str(args.project.absolute()),'--artifacts',str(args.artifacts.absolute()),'--packet',str(args.packet.absolute()),
        '--population',args.population,'--output',str(prediction.absolute()),'--proposals','32','--single-evaluation',
        '--adaptive-profile',str(LAB/'experiments/water-inference-budget-v1/configs/adaptive.json')],
        [sys.executable,str(LAB/'experiments/water-conditioning-v1/native_parallel.py'),'--packet',str(args.packet.absolute()),
        '--inference',str(prediction.absolute()),'--binary',str(args.binary.absolute()),'--output',str(evaluation.absolute()),
        *(['--training-only'] if args.population=='train' else [])]]
    (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    for index,command in enumerate(commands):
        with (args.output/f'stage-{index:02d}.log').open('xb') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    receipt=json.loads((prediction/'receipt.json').read_text());native=json.loads((evaluation/'evaluation.json').read_text())
    counts=[c['network_passes'] for c in receipt['case_iterations']]
    result={'schema':'rheon-multigrid-development-evaluation-v1','weights_sha256':receipt['weights_sha256'],
        'native':native,'network_evaluations':{'mean':statistics.mean(counts),'minimum':min(counts),'maximum':max(counts),'cases':counts},
        'inference_receipt_sha256':hashlib.sha256((prediction/'receipt.json').read_bytes()).hexdigest(),
        'scope':'same per-evaluation stopping and maximum 32; CPU inference only, independent native gates, no refinement'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('weights_sha256','network_evaluations')}),flush=True)
    print(json.dumps(native),flush=True)


if __name__=='__main__':main()
