"""Verify matched actual CUDA arms while permitting only the selected bank stage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-throughput-feedback-v1'))
from compare_training import measured


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_inputs(early,late,reference):
    expected=set(reference)|{'current_dataset','current_manifest'}
    if set(early)!=expected or set(late)!=expected:raise ValueError('exact original plus current input inventory required')
    if any(early[key]!=late[key]for key in expected-{'current_dataset'}):raise ValueError('unselected input differs between arms')
    if any(early[key]!=value or late[key]!=value for key,value in reference.items()):raise ValueError('original fitting input changed')
    if early['current_dataset']==late['current_dataset']:raise ValueError('distinct early and late current payloads required')


def project(study,root,receipt):
    intent=json.loads((study/'intent.json').read_text());assembly=Path(intent['assembly']).resolve(strict=True)
    if not assembly.is_relative_to(root)or digest(assembly)!=receipt['assembly_sha256']:raise ValueError('actual captured assembly differs')
    declaration=json.loads(assembly.read_text())
    for row in declaration['sources']:
        path=(root/row['path']).resolve(strict=True)
        if not path.is_relative_to(root)or digest(path)!=row['sha256']:raise ValueError('actual assembly source changed')
    witness_path=assembly.parent/'source-witness.json'
    closure=receipt['training_receipt']['closure']
    if digest(witness_path)!=closure['source_witness_sha256']:raise ValueError('source witness differs from actual training closure')
    if digest(assembly.parent/'config.json')!=closure['experiment_sha256']:raise ValueError('configuration differs from actual training closure')
    witness=json.loads(witness_path.read_text())
    for name,identity in witness['sources'].items():
        if Path(name).name!=name or digest(assembly.parent/name)!=identity:raise ValueError('captured source changed')
    return json.loads((assembly.parent/'config.json').read_text()),witness


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','early','late','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True)
    if a.output.exists():raise ValueError('fresh output required')
    a.reference=a.reference.resolve(strict=True)
    reference,reference_row=measured(a.reference,root)
    reference_config,_=project(a.reference,root,reference)
    receipts={};rows={};configs={};witnesses={};histories={}
    for stage in ('early','late'):
        folder=getattr(a,stage).resolve(strict=True);receipt,row=measured(folder,root)
        config,witness=project(folder,root,receipt)
        if config['current_feedback']['stage']!=stage or config['training']['updates']!=5000:raise ValueError('actual selected arm differs')
        receipts[stage]=receipt;rows[stage]=row;configs[stage]=config;witnesses[stage]=witness
        histories[stage]=json.loads((folder/'train/history.json').read_text())
    original=reference['training_receipt']['closure'];left=receipts['early']['training_receipt']['closure'];right=receipts['late']['training_receipt']['closure']
    matched_inputs(left['inputs'],right['inputs'],original['inputs'])
    for key in ('normalization_sha256','runtime_manifest_sha256','sample_order_sha256'):
        if left[key]!=right[key]or left[key]!=original[key]:raise ValueError('actual common material changed: '+key)
    if any(row['initial_checkpoint_sha256']!=reference_row['initial_checkpoint_sha256']for row in rows.values()):raise ValueError('initialization differs from retained0.03')
    selected=[]
    for stage in ('early','late'):
        config=json.loads(json.dumps(configs[stage]));config.pop('id');config['current_feedback'].pop('stage');selected.append(config)
        anchor=json.loads(json.dumps(config));anchor.pop('current_feedback');anchor['schema']=reference_config['schema']
        reference_settings={k:v for k,v in reference_config.items()if k!='id'}
        if anchor!=reference_settings:raise ValueError('original training configuration changed')
    if selected[0]!=selected[1]:raise ValueError('unselected configuration changed')
    sources=[{k:v for k,v in witnesses[stage]['sources'].items()if k!='config.json'}for stage in ('early','late')]
    if sources[0]!=sources[1]:raise ValueError('scientific source differs between arms')
    original_history=json.loads((a.reference/'train/history.json').read_text())
    for stream in ('analytical_feedback','supplemental_feedback','direction_feedback'):
        if any(histories[stage][stream]!=original_history[stream]for stage in ('early','late')):raise ValueError('existing feedback stream changed: '+stream)
    new=[]
    for stage in ('early','late'):
        value=dict(histories[stage]['current_feedback']);value.pop('stage');value.pop('payload_sha256');new.append(value)
        if value['source']['weights_sha256']!=reference_row['weights_sha256']:raise ValueError('current-policy generation checkpoint differs')
        rows[stage]['current_feedback']=histories[stage]['current_feedback'];rows[stage]['optimizer_work']=histories[stage]['optimizer_work']
    if new[0]!=new[1]:raise ValueError('current bank sampling/cell budget changed')
    if histories['early']['optimizer_work']!=histories['late']['optimizer_work']:raise ValueError('actual optimizer network work differs')
    result={'format':'pressure_current_policy_matched_arms_v1','status':'passed','source_sha256':digest(Path(__file__)),
        'reference':reference_row,'arms':rows,'allowed_input_difference':'Selected current_dataset only; current_manifest shared and original five inputs exact.',
        'all_shared_material_exact':True,'actual_optimizer_work_equal':True,'scientific_accuracy_asserted':False,'optimizer_updates_executed_by_report':0,
        'scope':'Actual5000CUDAupdates and five numerical freshCPU artifact pairs; matched early/late budget. Retained0.03 reused for identity/context, not another fit. Complete Study clocks include admission/queue/restore/readback; accuracy requires independent trajectories.'}
    with a.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({stage:{k:row[k]for k in ('optimizer_s','trainer_s','optimizer_work')}for stage,row in rows.items()}))

if __name__=='__main__':main()
