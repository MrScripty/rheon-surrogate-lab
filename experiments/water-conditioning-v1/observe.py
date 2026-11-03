"""One writer combines per-trial observations into the existing Tuldok activity."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

LAB=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('conditioning_activity',LAB/'experiments/live-project-v1/observer.py')
activity=importlib.util.module_from_spec(spec);spec.loader.exec_module(activity)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=LAB/'runs/water-conditioning-v1')
    parser.add_argument('--seconds',type=int,default=3600)
    args=parser.parse_args()
    if not 1<=args.seconds<=3600:parser.error('one to 3,600 seconds')
    observer=activity.Observer(LAB/'runs/live-project');deadline=time.monotonic()+args.seconds
    try:
        while time.monotonic()<deadline:
            rows=[]
            for path in sorted(args.root.glob('*/live/activity.json')):
                if path.stat().st_size>2*1024*1024:raise ValueError('oversized activity')
                rows.append({'trial':path.parents[1].name,**json.loads(path.read_text())})
            active=[r for r in rows if r['state']=='running']
            message='; '.join(f'{r["trial"]}: {r["phase"]}, {r["progress"]["completed"]}/{r["progress"]["total"] or "?"}' for r in active)
            message=message or 'Managed stages currently complete; CPU diagnostics and next experiments remain visible in the project plan.'
            summary={'trials':rows,'active_trials':len(active),'updated_at_unix':time.time()}
            activity.atomic(args.root/'live-summary.json',summary)
            observer.publish('Pressure conditioning · CUDA and CPU work',message,
                sum(r['state']=='completed' for r in rows),len(rows),state='running')
            time.sleep(.5)
    except KeyboardInterrupt:
        print('Observer stopped; managed Run ownership remains with Tuldok.',flush=True)


if __name__=='__main__':main()
