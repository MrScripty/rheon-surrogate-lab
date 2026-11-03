"""Read-only red/green signal for the actual managed refinement attempt."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/full-ai-next-priorities-v1'),str(ROOT/'runs/dependency-migration-v1/Tuldok'),
              str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
from qualify_typed_pressure import OwnerHTTP
import agent_api


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh read-only observation required')
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    result=owner.call('runs.get',dict(run_id=args.run_id))
    progress=owner.call('runs.progress',dict(run_id=args.run_id))
    args.output.write_text(json.dumps(dict(run=result,progress=progress),indent=2)+'\n')
    passed=result['state']=='completed'and result['custody_phase']=='drained'
    print(json.dumps(dict(signal='GREEN'if passed else'RED',run_id=args.run_id,state=result['state'],custody_phase=result['custody_phase'],
        artifacts=len(result['artifacts']),error=result.get('error'),progress_keys=list(progress))))
    raise SystemExit(0 if passed else 1)
