"""Publish bounded external previews of current studies and consumed CPU cases.

Reads public source receipts only. It never opens a catalog or controls Runs,
optimization, simulation, acceptance or promotion. Preview pressure is recorded,
not synthesized; no tracer or velocity animation is claimed.
"""
import argparse
import fcntl
from pathlib import Path
import signal
import sys
from threading import Event

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dependency-migration-v1'))
from live_migration import observer_type, read_json, preview_from_receipt


def once(root, work, observer):
    paths=list(work.glob('*regression-*/case-*/episode.json'))
    if len(paths)>256:raise ValueError('observation exceeds finite development scope')
    records=[]
    for path in paths:
        value,digest=read_json(path);records.append((path,value,digest))
    passed=[row for row in records if row[1]['status']=='passed']
    failed=len(records)-len(passed)
    studies=[]
    for name in ('control','lowband'):
        attempts=[p for p in work.glob(name+'-study-*') if (p/'intent.json').is_file()]
        if len(attempts)>8:raise ValueError('too many finite study attempts')
        study=max(attempts,key=lambda p:(p/'intent.json').stat().st_mtime_ns) if attempts else work/(name+'-study-01')
        if (study/'receipt.json').exists():state='training and CPU restore verified'
        elif (study/'failed-run.json').exists():state='execution attempt failed and retained'
        elif (study/'active.json').exists():state='Run submitted; current state is in Tuldok'
        elif (study/'intent.json').exists():state='preparing admission'
        else:state='not submitted'
        studies.append(name+': '+state)
    preview=None
    if passed:
        path,value,digest=max(passed,key=lambda row:row[0].stat().st_mtime_ns)
        preview=preview_from_receipt(root,path,value,digest)
        preview['source']['qualification']='Completed development case only; model comparison and promotion not asserted'
    observer.publish('Pressure band-feedback experiment',
        '; '.join(studies)+f'. CPU case receipts: {len(passed)} passed, {failed} failed. '
        'Actual Model/Study/Run state belongs to Tuldok; this is external observation. '
        'Preview is a completed sparse pressure frame, not live velocity or tracer flow.',
        completed=len(records),total=None,state='running',preview=preview)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','work','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--once',action='store_true');a=p.parse_args()
    root=a.root.resolve(strict=True);work=a.work.resolve(strict=True);output=a.output.resolve()
    if not work.is_relative_to(root) or not output.is_relative_to(root):raise ValueError('root-contained observation required')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=Event()
        signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
        observer=observer_type()(output)
        while not stop.is_set():
            once(root,work,observer)
            if a.once:break
            stop.wait(2)

if __name__=='__main__':main()
