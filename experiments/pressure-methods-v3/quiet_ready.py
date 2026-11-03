"""Verify terminal campaign workers before serial CPU timing begins."""
from datetime import datetime,timezone
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/pressure-methods-v3'
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def check():
    output=BASE/'quiet-start.json'
    if output.exists():raise ValueError('Quiet timing phase already declared')
    names=['baseline-evaluation-batch-01','native-training-controller-07','native-dependent-evaluation-07',
        'blind-capture-batch-01','blind-evaluation-batch-01','transfer-batch-07','preparation-batch-07']
    evidence={};identifiers=[]
    for name in names:
        p=BASE/name/'execution.json';d=read(p)
        if d['status']!='completed'or d['active']or not d['owned_child_reaped']:raise ValueError('Active campaign controller: '+name)
        evidence[str(p.relative_to(ROOT))]=sha(p)
    for arm in ['v3-replay','v3-native','v3-c37half','v3-c44half']:
        generation=3 if arm=='v3-replay'else 7
        d=read(BASE/f'{arm}-study-{generation:02d}/receipt.json')
        identifiers.extend([d['training_run_id'],d['restore_run_id']])
    identifiers.extend(['ea2b0852c42e41e49b07985307a914de','1e1c146e73084356b6f57687992512d1'])
    live=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit()or int(path.name)==os.getpid():continue
        try:argv=[a.decode(errors='replace')for a in (path/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not argv:continue
        matched=argv[0].startswith(str(BASE))and Path(argv[0]).name=='rheon-large-single-track'
        if Path(argv[0]).name.startswith('python'):
            for arg in argv[1:]:
                if arg.endswith('.py')and'\n'not in arg:
                    p=(ROOT/arg).resolve()
                    if p.is_relative_to(ROOT/'experiments/pressure-methods-v3')or p.is_relative_to(BASE):matched=True
                if'\n'not in arg and any('.external-'+identifier in arg for identifier in identifiers):matched=True
            if any(v.endswith('pressure-methods-v2/run_batch.py')for v in argv)and any(str(BASE)in v or v.startswith('runs/pressure-methods-v3/')for v in argv):matched=True
        if matched:live.append(dict(pid=int(path.name),program=argv[0]))
    if live:raise ValueError('Owned compute worker still live: '+str(live))
    write(output,dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),host_namespace=True,
        no_live_owned_compute_workers=True,controllers=evidence,source_sha256=sha(__file__),
        scope='Campaign compute workers absent. Ordinary desktop or unrelated host activity is not claimed absent. Subsequent timing batches run serially.'))
    print('Quiet CPU phase ready: all campaign fit, restore and evaluation workers terminal',flush=True)

if __name__=='__main__':check()
