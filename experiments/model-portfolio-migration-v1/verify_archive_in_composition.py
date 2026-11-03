"""Verify retained archive observations against original SQL after tmp loss.

No native codecs, SQL connections, model runtimes or payload copies are used.
The absent original portable tree cannot supply a new inode/mode comparison;
this report explicitly separates that limit from literal scientific checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from verify_history_project import original_fact_checks, require, digest, ORIGIN
from verify_native_composition import canonical, INPUTS, independent
from verify_native_portfolio_fork import inventory

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE_ID='1a3adad07d6c4618ba0edb8b0888fd34'
ARCHIVE_CP='6a547cd27a2c6f05d2760a31618de9855c99acb741768468754b192b42a42860'


def member(retained, family, identity, membership):
    alternate=membership['run_observations'].get(identity) if family=='run_history' else None
    doc=(retained[('run_observation',alternate)]['observation'] if alternate is not None
         else retained[(family,identity)])
    envelope=dict(family=family,id=identity,document=doc)
    require(hashlib.sha256(canonical(envelope)).hexdigest()==membership['records'][family][identity],
            'Archive member differs from its retained original envelope binding')
    return doc


def verify(project, report):
    require(not report.exists(),'Fresh archive composition verification report required')
    started=time.monotonic()
    metadata=ROOT/'runs/dependency-migration-v1/original-schema8-backup-02/metadata.json'
    origin=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/catalog-archive-summary-01.json').read_bytes())
    require(str(metadata.relative_to(ROOT))==origin['source_metadata']['path']
            and digest(metadata)==origin['source_metadata']['sha256']
            and digest(metadata.parent/'backup-receipt.json')==origin['backup_receipt']['sha256'],
            'Original archive metadata/receipt binding differs')
    original=json.loads(metadata.read_bytes());require(original['dataset_id']==ORIGIN,'Original catalog differs')
    before=(project/'.tuldok/project.json').read_bytes()
    descriptor, indexed=inventory(project)
    composition=json.loads(indexed[('project_composition',descriptor['project_id'])][1].read_bytes())
    selected=[v for v in composition['sources'] if v['project_id']==ARCHIVE_ID and v['checkpoint']==ARCHIVE_CP]
    require(len(selected)==1,'Original archive source binding differs')
    membership=selected[0]
    keys={(f,i) for f,items in membership['records'].items() for i in items}
    require(len(keys)==46932,'Original archive member population differs')
    needed=set(keys)|{('run_observation',v) for v in membership['run_observations'].values()}
    retained={key:json.loads(indexed[key][1].read_bytes()) for key in needed if key in indexed}
    originals={key:member(retained,*key,membership) for key in sorted(keys)}
    checks,counts,registrations=original_fact_checks(original,originals,all_archive=True)
    table=original['tables']['workspace_projects'];id_column=table['columns'].index('id')
    original_registration_ids={row[id_column] for row in table['rows']}
    prior=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/archive-staged-preservation-01.json').read_bytes())
    require(set(registrations)==set(prior['exact_retained_archived_registration_checks']),
            'Previously retained archive registration population differs')
    # The original extraction retained registrations reachable from history,
    # not every registration in the archived catalog. Keep that scope gap
    # explicit; successful preservation of the selected source is not closure.
    extra_registrations=original_registration_ids-set(registrations)
    extraction=originals[('project_extraction',ARCHIVE_ID)]
    copied=extraction['artifacts']['copied'];missing=extraction['artifacts']['missing_regenerable']
    require(len(copied)==3874 and len(missing)==42510,'Original archive payload disposition differs')
    for sha in copied:
        declaration=originals[('project_artifact',sha)];path=project/'artifacts/sha256'/sha
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==declaration['bytes']
                and digest(path)==sha,'Retained original archive payload differs')
    surviving=[];all_keys=set(keys);payloads=set(copied)
    for source,checkpoint,manifest_sha in INPUTS[:2]:
        source_raw=(source/'.tuldok/project.json').read_bytes()
        require(hashlib.sha256(source_raw).hexdigest()==manifest_sha,'Surviving original manifest changed')
        old,source_records=inventory(source)
        require(old['checkpoint']==checkpoint,'Surviving original source checkpoint changed')
        for key,(_,path) in source_records.items():
            require(key in indexed and path.read_bytes()==indexed[key][1].read_bytes(),
                    'Surviving original canonical record changed')
            independent(path,indexed[key][1]);all_keys.add(key)
            if key[0]=='project_artifact' and (source/'artifacts/sha256'/key[1]).exists():
                before_payload=source/'artifacts/sha256'/key[1];after_payload=project/'artifacts/sha256'/key[1]
                require(digest(before_payload)==key[1] and digest(after_payload)==key[1],
                        'Surviving original payload changed')
                independent(before_payload,after_payload);payloads.add(key[1])
        require((source/'.tuldok/project.json').read_bytes()==source_raw,'Surviving original changed during audit')
        surviving.append(dict(project_id=old['project_id'],checkpoint=checkpoint,original_records_exact=len(source_records)))
    overlay=json.loads(indexed[('project_fork',descriptor['project_id'])][1].read_bytes())
    for item in composition['sources']:
        all_keys.update(('run_observation',v) for v in item['run_observations'].values())
    all_keys.update({('project_composition',descriptor['project_id']),('project_fork',descriptor['project_id'])})
    for item in overlay['mapping']:
        family='workbench_record' if item['family']=='workbench_text' else item['family']
        all_keys.add((family,item['destination_id']))
        if family=='project_file':
            old=json.loads(indexed[(family,item['source_id'])][1].read_bytes())
            new=json.loads(indexed[(family,item['destination_id'])][1].read_bytes())
            require(new['history']==old['history'] and new['current']==old['current'],
                    'Cloned working file lost original revision history')
            if not old['current'].get('deleted',False):
                require((project/old['path']).read_bytes()==(INPUTS[0][0]/old['path']).read_bytes()
                        ==old['current']['text'].encode('utf-8'),'Cloned working file contents changed')
    require(set(indexed)==all_keys,'Composition has missing or unexplained canonical records')
    require({p.name for p in (project/'artifacts/sha256').iterdir()}==payloads
            ==set(overlay['artifacts']['copied']),'Composed payload population differs from retained source custody')
    require((project/'.tuldok/project.json').read_bytes()==before,'Composition checkpoint changed during archive audit')
    result=dict(schema='rheon.archived-source-in-composition.v1',status='all_original_archive_scientific_facts_memberships_and_copied_payloads_exact',
        created_unix=time.time(),project=str(project.relative_to(ROOT)),project_id=descriptor['project_id'],checkpoint=descriptor['checkpoint'],
        composition_manifest_sha256=hashlib.sha256(before).hexdigest(),original_archive_project_id=ARCHIVE_ID,
        original_archive_checkpoint=ARCHIVE_CP,original_metadata_sha256=digest(metadata),
        original_member_envelope_bindings_exact=len(originals),original_scientific_facts_exact=len(checks),
        scientific_family_counts=counts,registered_project_snapshots_exact=len(registrations),
        original_copied_payloads_rehashed=len(copied),original_missing_regenerable_declarations=len(missing),
        archive_registrations_in_original_metadata=len(original_registration_ids),
        archived_registrations_outside_original_extraction_scope=sorted(extra_registrations),
        archive_registration_scope_complete=not extra_registrations,
        surviving_source_preservation=surviving,all_composed_record_membership_exact=len(all_keys),
        all_source_present_payload_population_exact=len(payloads),
        alternate_Run_proofs_checked=len(membership['run_observations']),native_codecs_used=False,SQL_opened=False,
        payload_copies=0,models_executed=0,original_source_inode_or_mode_comparison_available=False,
        limitations='Original temporary archive tree is absent. Archive envelope bindings come from retained native composition provenance; scientific values and registration membership are independently compared to original SQL metadata. This does not recreate or freshly verify the absent original portable manifest or source inodes/modes, nor prove native publication or cold lifecycle.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'experiments/model-portfolio-migration-v1/verify_history_project.py']},
        elapsed_seconds=time.monotonic()-started)
    with report.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','original_member_envelope_bindings_exact','original_scientific_facts_exact','original_copied_payloads_rehashed']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();verify(args.project.resolve(),args.report.resolve())
