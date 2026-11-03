"""Read-only full-input reproduction of native composition construction.

Reads exact checkpoint records; no store publication, payload copy, SQL, runtime
or model inference. Retains a bounded domain error and source-only frame names.
"""
import hashlib
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok-project-composition'
REPORT=ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-composition-candidate-reproduction-01.json'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main(source=SOURCE, report=REPORT):
    global SOURCE, REPORT
    SOURCE, REPORT = source, report
    if REPORT.exists():raise ValueError('Fresh diagnostic report required')
    sys.path.insert(0,str(SOURCE))
    from project_composition import build_candidate
    paths=[ROOT/'runs/model-portfolio-migration-v1/native-composition-primary-01/root/project',
           ROOT/'runs/model-portfolio-migration-v1/full-current-history-cold-01/root/project',
           Path('/tmp/rheon-archive-owner-1eju7eaw/projects/rheon-archived-history-01')]
    pins=['0775213f6a11d93f9d80db80ea76daa6e80e2fcd0194a673dc2e8d093d644570',
          '5f9281c0595bc0d09c81d9291ef40e86f3cd82cafa8957fb009cd89c1982e7ad',
          '6a547cd27a2c6f05d2760a31618de9855c99acb741768468754b192b42a42860']
    code={name:sha((SOURCE/name).read_bytes()) for name in ['project_composition.py','project_fork_composition.py','project_fork.py']}
    snapshots=[];manifests=[];start=time.monotonic()
    for directory,checkpoint in zip(paths,pins):
        raw=(directory/'.tuldok/project.json').read_bytes();manifest=json.loads(raw)
        if manifest['checkpoint']!=checkpoint:raise ValueError('Fixed input checkpoint changed')
        records=[]
        for item in manifest['inventory']:
            relative=Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe canonical record path')
            content=(directory/'.tuldok'/relative).read_bytes()
            if len(content)!=item['bytes'] or sha(content)!=item['sha256']:raise ValueError('Canonical input differs')
            records.append(dict(family=item['family'],id=item['id'],document=json.loads(content)))
        snapshots.append(dict(project_id=manifest['project_id'],name=manifest['name'],checkpoint=checkpoint,records=records))
        manifests.append(sha(raw));print('loaded_'+manifest['project_id']+'_'+str(len(records)),flush=True)
    original=json.loads((ROOT/'runs/model-portfolio-migration-v1/native-composition-authoring-01/compose.request.json').read_bytes())
    result=dict(schema='rheon.native-composition-candidate-reproduction.v1',created_unix=time.time(),source=str(SOURCE.relative_to(ROOT)),
        source_sha256=code,source_checkpoints=pins,source_manifest_sha256=manifests,models_executed=0,SQL_opened=False,
        project_published=False,payload_files_copied=0,runner_sha256=sha(Path(__file__).read_bytes()))
    try:
        candidate=build_candidate(snapshots,'b497d865b6624aed942ec91e5bf2915c',
            created_at='2026-10-24T12:00:00+00:00',id_factory=lambda *_:uuid.uuid4().hex,
            directory=original['arguments']['directory'],name=original['arguments']['name'],request_key=original['arguments']['request_key'])
        result.update(status='candidate_constructed',records=len(candidate['records']),mapping_count=len(candidate['mapping']))
    except Exception as error:
        result.update(status='reproduced_candidate_failure',error_type=type(error).__name__,error_code=getattr(error,'code',None),
            error=str(error)[:1000],frames=[dict(file=Path(frame.filename).name,function=frame.name,line=frame.lineno)
                                          for frame in traceback.extract_tb(error.__traceback__)])
    for directory,expected in zip(paths,manifests):
        if sha((directory/'.tuldok/project.json').read_bytes())!=expected:raise ValueError('Input changed during reproduction')
    if code!={name:sha((SOURCE/name).read_bytes()) for name in code}:raise ValueError('Native code changed during reproduction')
    result.update(elapsed_seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_manifests_unchanged=True,native_source_unchanged=True)
    REPORT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--report',type=Path,default=REPORT)
    args=parser.parse_args()
    if not args.source.resolve().is_relative_to(ROOT/'runs/model-portfolio-migration-v1/sources'):
        parser.error('Explicit task-owned source required')
    if not args.report.resolve().is_relative_to(ROOT/'docs/plans/model-portfolio-migration-v1/reports'):
        parser.error('Fresh task-owned report required')
    main(args.source.resolve(),args.report.resolve())
