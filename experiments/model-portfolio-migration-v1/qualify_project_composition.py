"""Exercise the pure native merge on all three exact retained checkpoints.

This reads canonical project records only. It publishes no project, opens no SQL
or model runtime, and does not stand in for final lifecycle or payload custody.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok-project-composition'
REPORT=ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-composition-actual-inputs-01.json'
INTERSECTION=ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-composition-intersection-02.json'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    if REPORT.exists():raise ValueError('Fresh composition qualification report required')
    sys.path.insert(0,str(SOURCE))
    from project_composition import merge_retained_sources
    from project_history import _observation_document, _approval
    from project_store import _encode
    source_files=['project_composition.py','project_history.py','project_store.py','project_fork.py','runs.py']
    fingerprints={name:sha((SOURCE/name).read_bytes()) for name in source_files}
    expected=json.loads(INTERSECTION.read_bytes())['sources']
    archive=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/archived-history-extraction-03.json').read_bytes())
    if not archive['project_extracted'] or not archive['postgres_stopped']:
        raise ValueError('Terminal native archive extraction required')
    paths={label:Path(row['path']) for label,row in expected.items()}
    paths['archive']=Path(archive['destination'])
    observations={}
    started=time.monotonic()
    def sources():
        for label in ('working','current','archive'):
            path=paths[label];pin=expected[label]
            raw=(path/'.tuldok/project.json').read_bytes()
            if sha(raw)!=pin['manifest_sha256']:raise ValueError('Pinned source manifest changed')
            manifest=json.loads(raw)
            if manifest['project_id']!=pin['project_id'] or manifest['checkpoint']!=pin['checkpoint']:
                raise ValueError('Pinned source identity differs')
            records=[];entries={}
            for entry in manifest['inventory']:
                relative=Path(entry['path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe canonical path')
                member=path/'.tuldok'/relative
                if any(p.is_symlink() for p in (member,*member.parents)):raise ValueError('Symlink in source path')
                content=member.read_bytes()
                if len(content)!=entry['bytes'] or sha(content)!=entry['sha256']:
                    raise ValueError('Canonical source bytes changed')
                key=entry['family'],entry['id']
                if key in entries:raise ValueError('Repeated source record identity')
                records.append(dict(family=key[0],id=key[1],document=json.loads(content)))
                entries[key]=dict(sha256=entry['sha256'],bytes=entry['bytes'])
            if len(records)!=pin['records']:raise ValueError('Source membership count differs')
            observations[label]=dict(project_id=pin['project_id'],checkpoint=pin['checkpoint'],
                path=str(path),manifest_sha256=pin['manifest_sha256'],entries=entries)
            print('merge_source_'+label+'_'+str(len(records)),flush=True)
            yield dict(project_id=manifest['project_id'],name=manifest['name'],checkpoint=manifest['checkpoint'],records=records)
    merged=merge_retained_sources(sources())
    indexed={(row['family'],row['id']):row for row in merged['records']}
    retained=0;preserved={};native_observations=0
    for label,observed in observations.items():
        membership=next(row for row in merged['sources'] if row['project_id']==observed['project_id'])
        count=0
        for key,entry in observed['entries'].items():
            alternate=membership['run_observations'].get(key[1]) if key[0]=='run_history' else None
            if alternate is not None:
                wrapper=indexed[('run_observation',alternate)]
                _observation_document(wrapper)
                document=wrapper['document']['observation']
                _approval(document['admission_proof'],document['run'])
                native_observations+=1
            else:document=indexed[key]['document']
            raw=_encode(document)
            if len(raw)!=entry['bytes'] or sha(raw)!=entry['sha256']:
                raise ValueError('Merged representation does not preserve original canonical file bytes')
            original_envelope=dict(family=key[0],id=key[1],document=document)
            if membership['records'][key[0]][key[1]]!=sha(_encode(original_envelope)):
                raise ValueError('Exact source membership digest differs')
            count+=1
        preserved[label]=dict(project_id=observed['project_id'],checkpoint=observed['checkpoint'],records=count,
                              alternate_Run_observations=len(membership['run_observations']))
        retained+=count
        if sha((paths[label]/'.tuldok/project.json').read_bytes())!=observed['manifest_sha256']:
            raise ValueError('Source checkpoint changed during qualification')
    if fingerprints!={name:sha((SOURCE/name).read_bytes()) for name in source_files}:
        raise ValueError('Native source changed during qualification')
    result=dict(schema='rheon.native-project-composition-actual-inputs.v1',status='qualified_pure_retained_source_merge',
        created_unix=time.time(),source=str(SOURCE.relative_to(ROOT)),source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip()),
        source_sha256=fingerprints,source_stable_through_qualification=True,runner_sha256=sha(Path(__file__).read_bytes()),
        sources=preserved,original_memberships_preserved=retained,merged_records=len(indexed),
        distinct_Run_histories=sum(key[0]=='run_history' for key in indexed),
        distinct_Run_observations=sum(key[0]=='run_observation' for key in indexed),
        native_alternate_proofs_checked=native_observations,
        merged_canonical_document_bytes=sum(len(_encode(row['document'])) for row in merged['records']),
        source_memberships_bytes=len(_encode(merged['sources'])),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_seconds=time.monotonic()-started,SQL_opened=False,models_executed=0,project_published=False,payload_custody_qualified=False,
        scope='All three exact source checkpoint records preserve literal original document bytes; four alternate proof documents pass native observation/admission codecs. Final editable fork, full candidate domain validation, payload copy and native lifecycle publication remain required.')
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
