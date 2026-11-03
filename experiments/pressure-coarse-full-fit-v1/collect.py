"""Read actual completed fit/restore custody; preserve private Run records outside Git."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tuldok','selection-file','study','output','session-file','assembly','capture-verification','entry-qualification','control-comparison','baseline-custody','baseline-analysis'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('endpoint','catalog-id'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();started=time.perf_counter()
    if not re.fullmatch('[0-9a-f]{32}',args.catalog_id):raise ValueError('Explicit original catalog required')
    if args.output.exists()or not args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('Fresh rooted custody directory required')
    sys.path[:0]=[str(args.tuldok),str(HERE.parent/'dependency-migration-v1'),str(HERE.parent/'pressure-throughput-feedback-v1')]
    import agent_api,recipe_process,psycopg
    from psycopg import sql
    from qualify_typed_pressure import OwnerHTTP
    from compare_training import measured
    from run_fit import validate
    validate(args)
    study,measurement=measured(args.study.resolve(strict=True),args.root.resolve(strict=True))
    assembly=read(args.assembly);verification=read(args.capture_verification)
    assert study['assembly_sha256']==digest(args.assembly)
    project=read(args.root/assembly['project']['path']);active=read(args.study/'active.json')
    owner=OwnerHTTP(agent_api,args.endpoint,args.catalog_id,args.session_file)
    public_study=owner.call('studies.get',dict(study_id=study['study_id']))['study']
    assert not public_study['open']and public_study['id']==active['study_id']
    intent=read(args.study/'intent.json');opened=read(args.study/'mutations/01-open-project.json')
    assert intent['catalog_id']==args.catalog_id and intent['assembly_sha256']==digest(args.assembly)==verification['assembly_sha256']
    assert opened['status']=='acknowledged'and opened['result']['id']==project['id']==study['project_id']
    assert opened['result']['manifest_sha256']==verification['project_manifest_sha256']
    selected=read(args.selection_file);args.output.mkdir(parents=True)
    rows={}
    for label,reference,journal in [('train','train','09-admit-training.json'),('restore','restore','13-admit-restore.json')]:
        run_id=study['training_run_id'if label=='train'else'restore_run_id']
        admitted_path=args.study/'mutations'/journal;admitted=read(admitted_path)
        assert admitted['status']=='acknowledged'
        admitted_run=admitted['result']['run']if label=='train'else admitted['result']
        assert admitted_run['id']==run_id
        public=owner.call('runs.get',dict(run_id=run_id))
        assert public['state']=='completed'and public['custody_phase']=='drained'
        downloads=[v for v in study['downloads']if v['run_id']==run_id]
        authoritative={v['name']:(v['bytes'],v['sha256'])for v in public['artifacts']}
        assert len(authoritative)==len(public['artifacts'])==len(downloads)
        assert {v['name']:(v['bytes'],v['sha256'])for v in downloads}==authoritative
        assert len({v['name']for v in downloads})==len(downloads)
        for artifact in downloads:
            path=Path(artifact['output'])
            if not path.is_absolute():path=args.root/path
            assert path.resolve()==(args.study/label/artifact['name']).resolve()
            assert path.stat().st_size==authoritative[artifact['name']][0]and digest(path)==authoritative[artifact['name']][1]
        assert authoritative['weights.npz'][1]==authoritative['checkpoint-5000.npz'][1]==study['weights_sha256']
        with psycopg.connect(selected['dsn'],connect_timeout=5)as db:
            db.execute('SET TRANSACTION READ ONLY');db.execute("SET LOCAL statement_timeout='5s'")
            document=db.execute(sql.SQL('SELECT document FROM {}.workspace_runs WHERE id=%s').format(sql.Identifier('catalog_'+args.catalog_id)),(run_id,)).fetchone()[0]
        internal=json.loads(document)if isinstance(document,str)else document
        assert all(public[key]==internal[key]for key in ('id','state','custody_phase','started_at','finished_at'))
        assert admitted_run['material_sha256']==public['material_sha256']==internal['material_sha256']
        recipe_row=next(row for row in project['recipes']if row['id']==reference)
        recipe=read(args.root/recipe_row['path'])
        assert internal['material']['recipe']==recipe
        assert internal['material']['project']==dict(project_id=project['id'],reference_id=reference,path=recipe_row['path'],manifest_sha256=assembly['project']['sha256'],measured_sha256=recipe_row['sha256'])
        assert hashlib.sha256(recipe_process.encoded(internal['spec'])).hexdigest()==internal['custody_identity']['spec_sha256']
        assert internal['request_sha256']==internal['custody_identity']['request_sha256']
        output=Path(internal['spec']['output_dir']).resolve(strict=True)
        assert output.name=='output'and output.parent.name=='.external-'+run_id and Path(internal['spec']['cwd'])==output.parent
        captures={item['path']:item for item in internal['spec']['captured_files']}
        for source in recipe['sources']:
            path=output.parent/'source'/source['path']
            assert captures[str(path)]['sha256']==source['sha256']==internal['material']['captures']['source/'+source['path']]['sha256']
            assert hashlib.sha256(recipe_process.read_regular(path,4*1024**2)).hexdigest()==source['sha256']
        argv=internal['spec']['argv'];assert str(output.parent/'source'/recipe['entrypoint'])in argv
        if label=='train':
            assert '--gradient-limit'not in argv and argv.count('--fit-reference')==1
            ref=recipe['arguments'][recipe['arguments'].index({'literal':'--fit-reference'})+1]['source']
            assert argv[argv.index('--fit-reference')+1]==str(output.parent/'source'/ref)
        target=args.output/label;target.mkdir()
        raw=recipe_process.read_regular(output.parent/'custody/receipt.json',16384)
        custody=recipe_process.validate_receipt(raw,internal['custody_identity'])
        assert custody['drained']and custody['echild']and custody['spawn_closed']and custody['leader_returncode']==0
        assert custody['outcome']=='completed'
        (target/'receipt.json').write_bytes(raw)
        streams={}
        for name,maximum in [('stdout',2*1024**2),('stderr',128*1024)]:
            data=recipe_process.read_regular(output.parent/'custody'/(name+'.log'),maximum)
            assert len(data)==custody['streams'][name]['bytes']and hashlib.sha256(data).hexdigest()==custody['streams'][name]['sha256']
            (target/(name+'.log')).write_bytes(data);streams[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        private=target/'internal-run.json'
        with private.open('x')as stream:
            private.chmod(0o600);json.dump(internal,stream,separators=(',',':'));stream.write('\n')
        rows[label]=dict(run_id=run_id,state='completed',custody_phase='drained',echild=True,spawn_closed=True,leader_returncode=0,
            custody_receipt_sha256=hashlib.sha256(raw).hexdigest(),verified_streams=streams,
            capture_binding=dict(recipe_sha256=recipe_row['sha256'],admitted_journal_sha256=digest(admitted_path),actual_material_sha256=internal['material_sha256'],actual_spec_sha256=internal['custody_identity']['spec_sha256'],all_actual_captured_sources_verified=len(recipe['sources'])),
            started_at=public['started_at'],finished_at=public['finished_at'],run_start_through_drainage_s=(datetime.fromisoformat(public['finished_at'])-datetime.fromisoformat(public['started_at'])).total_seconds())
    # Classify only the actual hash-bound training stream, never fit loss as physics accuracy.
    sys.path.insert(0,str(HERE.parent/'pressure-coarse-gradient-v1'))
    from analyze import PREFIX,progress
    events=[json.loads(line[len(PREFIX):])for line in(args.output/'train/stdout.log').read_text().splitlines()if line.startswith(PREFIX)]
    completed=[v for v in events if v['event']=='full_fit_completed'];assert len(completed)==1
    final=completed[0];assert final['completed_updates']==final['successful_cuda_checks']==final['configured_updates']==5000
    assert not any(v['event']in('full_fit_failed','diagnostic_cap_reached','update_failed')for v in events)
    repairs=[v for v in events if v['event']=='clip_repaired'];assert final['clipping_repairs']==len(repairs)
    reference=read(args.root/next(v['path']for v in read(args.root/next(r['path']for r in project['recipes']if r['id']=='train'))['sources']if Path(v['path']).name=='gradient-reference.json'))
    sampling=[v for v in events if v['event']=='sampling_verified'];expected={v['stream']:v for v in verification['events']if v['event']=='sampling_verified'}
    assert len(sampling)==5 and {v['stream']for v in sampling}==set(reference['sampling_order_sha256'])
    assert all(v['full_length']==v['prefix_length']==5000 and v['full_order_sha256']==v['prefix_sha256']==reference['sampling_order_sha256'][v['stream']]==expected[v['stream']]['full_order_sha256']for v in sampling)
    initial=[v for v in events if v['event']=='initial_tensors_verified'];assert len(initial)==1 and initial[0]['device'].startswith('cuda')
    previous_summary=read(args.baseline_custody/'summary.json');previous_receipt=read(args.baseline_custody/'receipt.json')
    assert digest(args.baseline_custody/'receipt.json')==previous_summary['custody_receipt_sha256']
    for name in('stdout','stderr'):
        assert digest(args.baseline_custody/(name+'.log'))==previous_receipt['streams'][name]['sha256']==previous_summary['verified_streams'][name]['sha256']
    baseline=read(args.baseline_analysis)
    assert baseline['classification']=='1500_updates_finite_with_clipping_repair'
    assert baseline['completed_updates']==baseline['successful_cuda_checks']==1500
    assert baseline['run_id']==previous_summary['run_id']==previous_receipt['attempt_id']
    assert baseline['actual_capture_binding']==previous_summary['capture_binding']
    assert baseline['actual_capture_binding']['reference_sha256']==reference['predecessor_reference_sha256']
    for name in ('summary.json','receipt.json','stdout.log','stderr.log'):
        path=args.baseline_custody/name
        assert baseline['sources'][str(path)]['sha256']==digest(path)
    previous=progress(args.baseline_custody/'stdout.log');current=progress(args.output/'train/stdout.log')
    assert set(previous)==set(range(100,1501,100))
    assert set(current)==set(range(100,5001,100))
    comparisons=[dict(update=n,stability_loss=previous[n],full_fit_loss=current[n],exact=previous[n]==current[n])for n in sorted(previous)]
    assert comparisons and all(v['exact']for v in comparisons)
    result=dict(format='pressure-coarse-full-fit-custody-v1',status='passed',study_id=study['study_id'],study_open=False,
        source_sha256=digest(Path(__file__)),measurement=measurement,runs=rows,completed_updates=5000,successful_cuda_checks=5000,
        clipping_repairs=repairs,logged_losses=current,stability_logged_loss_comparison=comparisons,all_predecessor_logged_losses_exact=True,
        full_fit_completed_event=final,all_original_streams_and_initial_tensors_verified=True,analysis_elapsed_s=time.perf_counter()-started,
        pressure_accuracy_asserted=False,holdouts_consumed=0,candidate_promoted=False,
        sources={str(p):digest(p)for p in(args.assembly,args.capture_verification,args.entry_qualification,args.control_comparison,args.baseline_analysis,args.baseline_custody/'summary.json',args.baseline_custody/'receipt.json',args.baseline_custody/'stdout.log',args.study/'receipt.json')},
        scope='Original catalog completed/closed Study, actual admitted materials/specs/staged sources, hash-bound custodian streams and kernel ECHILD for both CUDA fit and fresh CPU restore. Private original Run records remain outside Git. Training stability is separate from CPU pressure accuracy.')
    write(args.output/'summary.json',result)
    print(json.dumps(dict(status='passed',completed_updates=5000,successful_cuda_checks=5000,repairs=len(repairs),optimizer_s=measurement['optimizer_s'],all_children_reaped=True)))


if __name__=='__main__':main()
