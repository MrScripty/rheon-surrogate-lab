"""Collect terminal diagnostic logs through the original catalog/custody owner.

Only the single owned Run is read. Internal execution/environment records remain
outside Git; small summaries retain hashes and exact custodian stream identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','tuldok','selection-file','study','output','session-file','assembly','capture-verification','probe-qualification'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('endpoint','catalog-id'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{32}',args.catalog_id):
        raise ValueError('Explicit original catalog identity required')
    sys.path[:0]=[str(args.tuldok.resolve(strict=True)),str(Path(__file__).resolve().parents[1]/'dependency-migration-v1')]
    import agent_api
    import recipe_process
    import psycopg
    from psycopg import sql
    from qualify_typed_pressure import OwnerHTTP
    from run_diagnostic import validate_admission
    start=time.perf_counter()
    limit=validate_admission(args)
    assembly=json.loads(args.assembly.read_text())
    verification=json.loads(args.capture_verification.read_text())
    qualification=json.loads(args.probe_qualification.read_text())
    project_path=args.root/assembly['project']['path']
    project=json.loads(project_path.read_text())
    recipe_row=next(row for row in project['recipes']if row['id']=='train')
    recipe=json.loads((args.root/recipe_row['path']).read_text())
    active=json.loads((args.study/'active.json').read_text())
    terminal=json.loads((args.study/'diagnostic-terminal.json').read_text())
    assert terminal['run_id']==active['training_run_id'] and not terminal['study_open']
    owner=OwnerHTTP(agent_api,args.endpoint,args.catalog_id,args.session_file)
    public=owner.call('runs.get',dict(run_id=terminal['run_id']))
    assert public['state']==terminal['run_state']=='failed' and public['custody_phase']=='drained'
    assert not public['artifacts'] and not terminal['cpu_restore_admitted']
    study=owner.call('studies.get',dict(study_id=active['study_id']))['study']
    assert not study['open']
    selected=json.loads(args.selection_file.read_text())
    with psycopg.connect(selected['dsn'],connect_timeout=5)as db:
        db.execute('SET TRANSACTION READ ONLY')
        db.execute("SET LOCAL statement_timeout='5s'")
        row=db.execute(sql.SQL('SELECT document FROM {}.workspace_runs WHERE id=%s').format(
            sql.Identifier('catalog_'+args.catalog_id)),(public['id'],)).fetchone()[0]
    internal=json.loads(row)if isinstance(row,str)else row
    assert all(internal[key]==public[key]for key in ('id','state','custody_phase','started_at','finished_at'))
    intent=json.loads((args.study/'intent.json').read_text())
    admitted_path=args.study/'mutations/09-admit-training.json'
    admitted=json.loads(admitted_path.read_text())
    opened=json.loads((args.study/'mutations/01-open-project.json').read_text())
    assert intent['assembly_sha256']==digest(args.assembly.read_bytes())==verification['assembly_sha256']
    assert intent['catalog_id']==args.catalog_id
    assert admitted['status']==opened['status']=='acknowledged'
    assert admitted['result']['run']['id']==public['id']
    assert admitted['result']['run']['material_sha256']==public['material_sha256']==internal['material_sha256']
    assert opened['result']['id']==project['id']==assembly['project']['id']
    assert opened['result']['manifest_sha256']==verification['project_manifest_sha256']
    assert internal['material']['recipe']==recipe
    assert internal['material']['project']==dict(project_id=project['id'],reference_id='train',
        path=recipe_row['path'],manifest_sha256=assembly['project']['sha256'],measured_sha256=recipe_row['sha256'])
    assert digest(recipe_process.encoded(internal['spec']))==internal['custody_identity']['spec_sha256']
    assert internal['request_sha256']==internal['custody_identity']['request_sha256']
    output=Path(internal['spec']['output_dir']).resolve(strict=True)
    assert output.name=='output' and output.parent.name=='.external-'+public['id']
    assert Path(internal['spec']['cwd'])==output.parent
    captures={item['path']:item for item in internal['spec']['captured_files']}
    for source in recipe['sources']:
        captured=output.parent/'source'/source['path']
        assert captures[str(captured)]['sha256']==source['sha256']
        assert internal['material']['captures']['source/'+source['path']]['sha256']==source['sha256']
        assert digest(recipe_process.read_regular(captured,4*1024**2))==source['sha256']
    assert str(output.parent/'source'/recipe['entrypoint']) in internal['spec']['argv']
    argv=internal['spec']['argv']
    assert argv.count('--gradient-limit')==1 and argv[argv.index('--gradient-limit')+1]==str(limit)
    reference_source=recipe['arguments'][recipe['arguments'].index({'literal':'--gradient-reference'})+1]['source']
    assert argv.count('--gradient-reference')==1
    assert argv[argv.index('--gradient-reference')+1]==str(output.parent/'source'/reference_source)
    entry_sha=next(item['sha256']for item in recipe['sources']if item['path']==recipe['entrypoint'])
    assert entry_sha==qualification['source_sha256']
    capture_binding=dict(status='passed',assembly_sha256=verification['assembly_sha256'],
        project_manifest_sha256=verification['project_manifest_sha256'],project_id=project['id'],
        recipe_sha256=recipe_row['sha256'],source_witness_sha256=verification['witness_sha256'],
        reference_sha256=verification['reference_sha256'],observer_sha256=entry_sha,
        capture_verifier_sha256=verification['source_sha256'],
        qualification_sha256=digest(args.probe_qualification.read_bytes()),
        capture_verification_sha256=digest(args.capture_verification.read_bytes()),
        admitted_journal_sha256=digest(admitted_path.read_bytes()),actual_material_sha256=internal['material_sha256'],
        actual_spec_sha256=internal['custody_identity']['spec_sha256'],
        all_actual_captured_sources_verified=len(recipe['sources']),diagnostic_limit=limit)
    custody=output.parent/'custody'
    raw_receipt=recipe_process.read_regular(custody/'receipt.json',16384)
    receipt=recipe_process.validate_receipt(raw_receipt,internal['custody_identity'])
    assert receipt['drained']and receipt['echild']and receipt['spawn_closed']
    assert receipt['outcome']=='failed'and receipt['reason']=='nonzero_exit'and receipt['leader_returncode']==1
    args.output.mkdir(parents=True)
    rows={}
    for name,maximum in [('stdout',2*1024**2),('stderr',128*1024)]:
        raw=recipe_process.read_regular(custody/(name+'.log'),maximum)
        assert len(raw)==receipt['streams'][name]['bytes']and digest(raw)==receipt['streams'][name]['sha256']
        with(args.output/(name+'.log')).open('xb')as stream:
            stream.write(raw)
        rows[name]=dict(bytes=len(raw),sha256=digest(raw))
    (args.output/'receipt.json').write_bytes(raw_receipt)
    # Private original record is retained for source/attempt validation, never printed.
    private=args.output/'internal-run.json'
    with private.open('x')as stream:
        private.chmod(0o600)
        json.dump(internal,stream,separators=(',',':'))
        stream.write('\n')
    result=dict(format='pressure-coarse-gradient-custody-v1',status='passed',run_id=public['id'],
        run_state='failed',custody_phase='drained',study_id=study['id'],study_open=False,
        custody_identity=internal['custody_identity'],custody_receipt_sha256=digest(raw_receipt),
        verified_streams=rows,all_children_reaped=True,spawn_closed=True,leader_returncode=1,
        artifacts_published=0,cpu_restore_admitted=False,elapsed_s=time.perf_counter()-start,
        run_started_at=public['started_at'],run_finished_at=public['finished_at'],
        validator_sha256=digest((args.tuldok/'recipe_process.py').read_bytes()),
        capture_binding=capture_binding,
        scope='Original public terminal/closed Study readback bound to one read-only internal catalog record and exact hash-verified custodian stdout/stderr. Diagnostic cause still requires log analysis.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
