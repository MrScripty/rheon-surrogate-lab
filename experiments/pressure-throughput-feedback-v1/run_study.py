"""Run one frozen CUDA fit and fresh CPU restore through the live Tuldok catalog.

Each mutation has a durable dispatch journal. Never replay an interrupted command;
inspect its journal and the catalog first. No model promotion or holdout access.
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dependency-migration-v1'))
from dependency_config import canonical, decode, digest, require
from qualify_typed_pressure import Journal, OwnerHTTP, artifact, await_drained, restore_bindings, sha, write


def execute(args):
    started=time.monotonic()
    require(not args.output.exists(),'fresh command directory required; never replay unknown writes')
    assembly=decode(args.assembly.read_bytes());root=args.root.resolve(strict=True)
    project_file=root/assembly['project']['path'];require(digest(project_file)==assembly['project']['sha256'],'Project identity changed')
    document=decode(project_file.read_bytes());definitions={}
    for row in document['recipes']:
        require(digest(root/row['path'])==row['sha256'],'recipe changed')
        definitions[row['id']]=decode((root/row['path']).read_bytes())
    for row in assembly['sources']+document['data']+document['extensions']:
        require(digest(root/row['path'])==row['sha256'],'captured material changed: '+row['path'])
    require(assembly['optimizer_updates_executed']==0 and 1<=assembly['updates']<=5000,'bounded frozen assembly required')
    sys.path.insert(0,str(args.tuldok.resolve(strict=True)));import agent_api,project_cli
    owner=OwnerHTTP(agent_api,args.endpoint,args.catalog_id,args.session_file)
    args.output.mkdir(parents=True);(args.output/'mutations').mkdir();journal=Journal(args.output/'mutations')
    write(args.output/'intent.json',dict(assembly=str(args.assembly.resolve()),assembly_sha256=digest(args.assembly),
        endpoint=args.endpoint,catalog_id=args.catalog_id,optimizer_updates=assembly['updates'],started_unix=time.time(),automatic_retry=False))
    def mutate(phase,operation,arguments):
        return journal.mutation(phase,operation,arguments,lambda:owner.call(operation,arguments))
    project=mutate('open-project','project_snapshots.open',dict(root=args.project_alias,manifest_path=assembly['project']['path']))
    require(project['id']==document['id'] and project['manifest_sha256']==assembly['project']['sha256'],'remote Project identity differs')
    refs={row['id']:row for row in project['data']}
    require(all(r['status']=='available' and r['integrity']=='verified' for r in refs.values()),'all Project inputs must verify')
    recipe=next(row for row in project['recipes'] if row['id']=='train');recipe_sha=recipe['observed_sha256']
    inspected=owner.call('recipes.inspect',dict(project_id=project['id'],reference_id='train',expected_sha256=recipe_sha))
    require(inspected['definition']==definitions['train'],'remote trainer differs')
    policy=definitions['train']['model_evidence']
    model=mutate('create-model','models.create',dict(name=document['name'],task=inspected['model_task'],kind='captured_source',
        specification=assembly['model_specification'],source_selector=dict(project_id=project['id'],recipe_id='train',expected_sha256=recipe_sha),
        preprocessing_policy=assembly['preprocessing_policy'],output_policy=assembly['output_policy']))
    published=mutate('publish-model','models.publish',dict(model_id=model['id'],revision=model['revision']))
    graph=mutate('train-graph','graphs.create_external',dict(name=document['name']+' CUDA',project_id=project['id'],recipe_id='train',expected_sha256=recipe_sha,
        inputs={slot:dict(reference_id=slot,expected_sha256=refs[slot]['observed_sha256']) for slot in definitions['train']['inputs']}))
    study=mutate('create-study','studies.create',dict(name=document['name'],limits=assembly['limits'],policy=dict(format='selection_policy_v1',
        metric_definition_sha256=sha(policy['metric_definitions'][0]),checkpoint_rule='explicit',replicate_reduce='individual',
        required_capabilities=policy['profile']['required_capabilities'],numerical_acceptance='diagnostic',tie_break='trial_id')))
    key='band-'+sha(dict(catalog=args.catalog_id,output=str(args.output.resolve())))[:24]
    trial=mutate('prepare-trial','studies.prepare_trial',dict(study_id=study['id'],request_key=key+'-prepare',revision=study['revision'],
        graph_id=graph['id'],graph_revision=graph['revision'],model=published['ref'],replicate=0))
    closed=owner.call('studies.get',dict(study_id=study['id']))
    require(not closed['study']['open'] and not closed['reservations'],'closed pre-admission study required')
    grant=mutate('approve-trial','studies.approve',dict(study_id=study['id'],trial_id=trial['id'],expected_material_sha256=trial['material']['sha256'],consumer_actor_id=owner.identity['actor_id'],ttl_seconds=3600))
    opened=mutate('open-admission','studies.continue',dict(study_id=study['id'],request_key=key+'-open',revision=closed['study']['revision'],admission_epoch=closed['study']['admission_epoch']))
    admitted=mutate('admit-training','studies.admit',dict(study_id=study['id'],trial_id=trial['id'],request_key=key+'-fit',revision=opened['revision'],admission_epoch=opened['admission_epoch'],approval_id=grant['id']))
    run_id=admitted['run']['id'];write(args.output/'active.json',dict(project_id=project['id'],study_id=study['id'],training_run_id=run_id,
        url=args.endpoint+'/?workspace=models',optimizer_updates=assembly['updates']))
    print(json.dumps(dict(phase='training',project_id=project['id'],run_id=run_id)),flush=True)
    training=await_drained(owner,run_id,timeout=2100)
    current=owner.call('studies.get',dict(study_id=study['id']))['study']
    stopped=mutate('close-study','studies.stop',dict(study_id=study['id'],request_key=key+'-stop',revision=current['revision']))
    evidence=owner.call('runs.model_evidence',dict(run_id=run_id,split='validation'))
    require(evidence['state']=='verified' and len(evidence['fields'])==len(evidence['checkpoints'])==1,'retained final field/checkpoint must verify')
    checkpoint=f"checkpoint-{assembly['updates']:03d}.npz"
    require(evidence['checkpoints'][0]['weights']['sha256']==artifact(training,checkpoint)['sha256'],'SDK checkpoint differs from actual weights')
    restore_ref=next(row for row in project['recipes'] if row['id']=='restore');restore_sha=restore_ref['observed_sha256']
    bindings=restore_bindings(definitions['restore'],project,training)
    restore_graph=mutate('restore-graph','graphs.create_external',dict(name=document['name']+' CPU restore',project_id=project['id'],recipe_id='restore',expected_sha256=restore_sha,inputs=bindings))
    preflight=owner.call('graphs.preflight_external',dict(graph_id=restore_graph['id'],revision=restore_graph['revision']))
    body=dict(graph_id=restore_graph['id'],graph_revision=restore_graph['revision'],expected_digest=preflight['material_sha256'],consumer_actor_id=owner.identity['actor_id'],ttl_seconds=3600)
    approval=journal.mutation('approve-restore','runs.approve_external',body,lambda:owner.raw('POST','/api/workspace/runs/approve-external',body))
    restored_admit=mutate('admit-restore','runs.submit_external',dict(request_key=key+'-restore',graph_id=restore_graph['id'],graph_revision=restore_graph['revision'],expected_digest=preflight['material_sha256'],approval_id=approval['id']))
    print(json.dumps(dict(phase='CPU restore',run_id=restored_admit['id'])),flush=True)
    restored=await_drained(owner,restored_admit['id'],timeout=2100)
    download_receipts=[]
    for label,run in (('train',training),('restore',restored)):
        target=args.output/label;target.mkdir()
        for row in run['artifacts']:
            # Host already decoded the declared safe inventory; download every
            # published result with its full hash, never a private Run directory.
            download_receipts.append(project_cli.download(owner.transport,run['id'],row['name'],row['sha256'],target/row['name']))
    trained=decode((args.output/'train/receipt.json').read_bytes());cpu=decode((args.output/'restore/receipt.json').read_bytes())
    require(trained['mode']=='train' and trained['optimizer_updates']==assembly['updates'] and trained['execution']['device'].startswith('cuda')
        and cpu['mode']=='restore' and cpu['optimizer_updates']==0 and cpu['execution']['device']=='cpu'
        and not cpu['execution']['cuda_initialized'] and cpu['originating_training_run']['run_id']==run_id
        and trained['closure']==cpu['closure'] and trained['prediction_inventory']==cpu['prediction_inventory'],'actual CUDA/CPU restore custody and parity required')
    parity={}
    for name in ('weights.npz','normalization.json','predictions.npz','history.json',checkpoint):
        parity[name]=digest(args.output/'train'/name)==digest(args.output/'restore'/name)
    require(all(parity.values()),'exact CPU artifact parity required')
    result=dict(format='rheon_pressure_training_study_v1',status='completed',elapsed_s=time.monotonic()-started,
        assembly_sha256=digest(args.assembly),project_id=project['id'],study_id=study['id'],trial_id=trial['id'],model=published['ref'],
        training_run_id=run_id,restore_run_id=restored['id'],optimizer_updates=assembly['updates'],band_feedback=assembly['band_feedback'],
        weights_sha256=artifact(training,'weights.npz')['sha256'],artifact_parity=parity,training_receipt=trained,restore_receipt=cpu,
        downloads=download_receipts,model_evidence=evidence,model_promotion=False,untouched_scenes_consumed=0)
    write(args.output/'receipt.json',result);print(json.dumps({k:result[k] for k in ('status','elapsed_s','weights_sha256','optimizer_updates')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tuldok','assembly','session-file','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--endpoint',required=True);p.add_argument('--catalog-id',required=True);p.add_argument('--project-alias',required=True)
    execute(p.parse_args())

if __name__=='__main__':main()
