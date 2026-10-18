"""Seed exact verified restore bytes before read-only recovery reaches that phase.

The collector already accepts existing whole-SHA-verified artifacts. This helper
never reads private Run storage, replaces files, submits a Run or changes weights.
It requires the recovery owner's exact completed-request proof and leaves at least
two unpublished training artifacts as a conservative phase boundary.
"""
import argparse
import os
from pathlib import Path
import time

from common import read,sha,write


def seed(study,output,timeout):
    started=time.monotonic()
    if output.exists() or not 0<timeout<=300:
        raise ValueError('Fresh receipt and bounded wait required')
    proof=read(study/'recovery/readback.json');journal=read(study/'mutations/13-admit-restore.json')
    original=study/'recovery/original-unresolved.json'
    restored=proof['run']
    if (proof['status']!='exact_completed_request_confirmed' or proof['remote_mutations']!=0
            or proof['original_journal_sha256']!=sha(original) or journal['status']!='acknowledged'
            or journal['result']['id']!=restored['id'] or restored['state']!='completed'
            or restored['custody_phase']!='drained'):
        raise ValueError('Exact completed owner-verified restore required')
    training_id=read(study/'active.json')['training_run_id']
    trained=read(study/('terminal-'+training_id+'.json'))
    if trained['state']!='completed' or trained['custody_phase']!='drained':
        raise ValueError('Completed published training artifacts required')
    pairs=[]
    for target in restored['artifacts']:
        source=next((r for r in trained['artifacts']if (r['sha256'],r['bytes'])==(target['sha256'],target['bytes'])),None)
        if source is not None:pairs.append((source,target))
    rows=[];status='deadline_without_seeding'
    while time.monotonic()-started<timeout:
        remaining=[r for r in trained['artifacts']if not (study/'train'/r['name']).is_file()]
        if len(remaining)<2:
            status='phase_boundary_passed_without_seeding';break
        if all((study/'train'/s['name']).is_file()for s,_ in pairs):
            target_dir=study/'restore';target_dir.mkdir(exist_ok=True)
            for source,target in pairs:
                if any(Path(r['name']).name!=r['name']for r in [source,target]):
                    raise ValueError('Safe artifact basenames required')
                previous=study/'train'/source['name'];destination=target_dir/target['name']
                if previous.is_symlink() or previous.stat().st_size!=target['bytes'] or sha(previous)!=target['sha256']:
                    raise ValueError('Collected source differs from current published restore inventory')
                if destination.exists() or destination.is_symlink():
                    raise ValueError('Collector reached restore phase; preserve existing target')
                os.link(previous,destination,follow_symlinks=False)
                if sha(destination)!=target['sha256'] or destination.stat().st_size!=target['bytes']:
                    raise ValueError('Seeded bytes differ')
                rows.append(dict(name=target['name'],source=source['name'],bytes=target['bytes'],sha256=target['sha256']))
            status='seeded';break
        time.sleep(2)
    write(output,dict(status=status,restore_run_id=restored['id'],training_run_id=training_id,
        remote_mutations=0,retraining_updates=0,rows=rows,elapsed_s=time.monotonic()-started,
        source_sha256=sha(__file__),proof_sha256=sha(study/'recovery/readback.json'),
        contract='Existing collector rechecks every whole SHA and size before reuse.'))
    print(status,len(rows),'exact artifacts',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args();seed(args.study.resolve(),args.output.resolve(),args.timeout)
