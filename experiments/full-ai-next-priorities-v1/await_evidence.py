"""Wait for one existing bounded experiment; never submit or retry its work."""
import argparse
from pathlib import Path
import time
from study_support import read

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=3600)
    args=parser.parse_args()
    if not 1<=args.timeout<=7200:raise ValueError('Finite observation deadline required')
    deadline=time.monotonic()+args.timeout
    while True:
        if args.evidence.exists():
            record=read(args.evidence);status=record.get('status')
            if status in ('completed','passed'):break
            if status in ('failed','stopped','cancelled','interrupted','unknown'):
                raise RuntimeError('Existing experiment stopped; inspect its custody without resubmission')
        if time.monotonic()>deadline:raise TimeoutError('Existing evidence observation deadline')
        time.sleep(2)
    print('Existing experiment completed:',args.evidence,flush=True)
