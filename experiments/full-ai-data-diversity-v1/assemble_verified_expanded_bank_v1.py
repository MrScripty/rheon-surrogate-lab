"""Assemble audited train/development labels; never open reserved references."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {'pressure','vx','vy','vz','tracer','rhs','native_input_residual','pressure_base'}


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    if path.exists():raise ValueError('Fresh assembled bank output required')
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def run(args):
    if args.output.exists():raise ValueError('Fresh assembled bank directory required')
    reg=read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen assembly input changed: '+path)
    reservation=read(ROOT/reg['reservation'])
    declared={row['identifier']:row for row in reservation['rows']}
    base_audits=[read(ROOT/path) for path in reg['training_archive_audits']]
    dev_audit=read(ROOT/reg['development_archive_audit'])
    supplements=[read(ROOT/path) for path in reg['supplement_controllers']]
    supplement_audits=[read(ROOT/path) for path in reg['supplement_archive_audits']]
    for audit in [*base_audits,dev_audit]:
        if (audit['status']!='completed_expanded_archive_integrity_audit'
                or not audit['every_field_decoded_typed_finite_and_hashed']
                or not audit['every_registered_archive_hash_rechecked_before_and_after_decoding']
                or audit['reserved_references_opened']!=0):
            raise ValueError('Complete archive decoding evidence required')
    for controller, audit in zip(supplements,supplement_audits):
        if (controller['status']!='completed_registered_training_window_supplements'
                or audit['status']!='completed_multistep_supplement_archive_integrity_audit'
                or audit['decoded_new_states']!=controller['generated_new_states']
                or audit['verified_reused_states']!=controller['reused_existing_states']
                or not audit['every_new_saved_field_decoded_and_hash_verified']
                or not audit['every_reused_archive_hash_and_decoded_baseline_provenance_verified']
                or audit['reserved_references_opened']!=0):
            raise ValueError('Complete native-window archive integrity evidence required')
    train_sources=[row for audit in base_audits for row in audit['scenes']]
    if len(train_sources)!=148 or len({row['identifier'] for row in train_sources})!=148:
        raise ValueError('Exactly148 distinct training scenes required')
    if len(train_sources)!=148:raise ValueError('All148 training scenes required')
    windows={row['identifier']:row for controller in supplements for row in controller['completed']}
    if len(windows)!=148:raise ValueError('All148 scene windows required')
    base_files=collections.defaultdict(list)
    for audit in [*base_audits,dev_audit]:
        for row in audit['files']:base_files[row['identifier']].append(row)
    banks={}
    for split,sources in [('train',train_sources),('development',dev_audit['scenes'])]:
        trajectories=[]
        for selected in sources:
            identifier=selected['identifier'];reserved=declared[identifier]
            if reserved['split']!=split:raise ValueError('Reserved or cross-split scene prohibited')
            path=ROOT/selected['receipt']
            if digest(path)!=selected['receipt_sha256']:raise ValueError('Original native receipt changed')
            receipt=read(path)
            owner=receipt['native_process']
            if (receipt['status']!='completed' or receipt['actual_steps']!=240
                    or len(receipt['frames'])!=240 or not owner['reaped']
                    or owner['returncode']!=0 or not owner['graceful_protocol_complete']
                    or not all(frame['frame']['advanced'] and frame['frame']['quality']['qualified']
                               and not frame['frame']['quality']['hard_stop'] for frame in receipt['frames'])):
                raise ValueError('Complete qualified native trajectory and closed owner required')
            if receipt['split']!=split or receipt['group']!=reserved['group']:
                raise ValueError('Whole forcing-group split identity required')
            states={}
            for value in base_files[identifier]:
                if set(value['fields'])!=FIELDS:raise ValueError('All eight typed physical/request fields required')
                states[value['step']]={k:value[k] for k in ('step','time_s','path','sha256','fields')}
            if len(states)!=22:raise ValueError('All22 original native archives required')
            native_windows=[]
            if split=='train':
                window=windows[identifier]
                if window['source_receipt']!=selected['receipt'] or window['source_receipt_sha256']!=selected['receipt_sha256']:
                    raise ValueError('One consistent native lineage per training scene required')
                for item in window['windows']:
                    if states[item['start']]['sha256']!=item['current_state_sha256']:
                        raise ValueError('Original current-state provenance changed')
                    outputs=[]
                    for value in item['outputs']:
                        fields=value['arrays']
                        if set(fields)!=FIELDS:raise ValueError('All eight native label fields required')
                        entry={k:value[k] for k in ('step','time_s','path','sha256')};entry['fields']=fields
                        if entry['step'] in states:
                            old=states[entry['step']]
                            if old['sha256']!=entry['sha256'] or old['path']!=entry['path']:
                                raise ValueError('Duplicate state has inconsistent native provenance')
                        else:states[entry['step']]=entry
                        outputs.append(dict(lead=value['lead'],step=value['step'],path=value['path'],sha256=value['sha256']))
                    if [row['lead'] for row in outputs]!=[1,2,3,4]:raise ValueError('All intermediate labels required')
                    native_windows.append(dict(start=item['start'],outputs=outputs))
                if len(states)!=31 or [row['start'] for row in native_windows]!=[0,32,96,192]:
                    raise ValueError('All31 unique states and four full K4windows required')
            trajectories.append(dict(identifier=identifier,split=split,group=receipt['group'],
                profile=reserved['profile'],motion=reserved['motion'],counts_xyz=receipt['scene']['counts'],
                cells_per_metre=receipt['scene']['cells_per_metre'],receipt=selected['receipt'],receipt_sha256=selected['receipt_sha256'],
                reference_substeps=receipt.get('reference_substeps',1),forcing_control_substeps=1,
                output_dt_s=1/60,reference_dt_s=1/(60*receipt.get('reference_substeps',1)),
                reference_scheme_id='sampled-controls-native-substeps-'+str(receipt.get('reference_substeps',1)),
                temporal_convergence_qualified=False,states=[states[step] for step in sorted(states)],windows=native_windows))
        expected=148 if split=='train' else 24
        if len(trajectories)!=expected or len({row['identifier'] for row in trajectories})!=expected:
            raise ValueError('Full registered split population required')
        banks[split]=dict(status='assembled_from_actual_archive_integrity_evidence',split=split,
            trajectories=trajectories,scene_count=len(trajectories),
            group_count=len({row['group'] for row in trajectories}),
            unique_state_archives=sum(len(row['states']) for row in trajectories),
            every_field_archive_decoded_typed_finite_and_hash_verified_by_linked_audits=True,
            native_reference_quality_qualified=True,reference_temporal_convergence_qualified=False,
            future_transition_lineages_may_not_be_pooled_as_identical=True,
            reserved_references_opened=0,optimizer_updates=0,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    train_groups={row['group'] for row in banks['train']['trajectories']}
    dev_groups={row['group'] for row in banks['development']['trajectories']}
    reserved_groups={row['group'] for row in reservation['rows'] if row['split']=='reserved'}
    if train_groups&dev_groups or train_groups&reserved_groups or dev_groups&reserved_groups:
        raise ValueError('Whole-group splits must be disjoint')
    args.output.mkdir(parents=True)
    for split, bank in banks.items():write(args.output/(split+'-bank.json'),bank)
    summary=dict(status='completed_full_expanded_train_development_bank_assembly',created_unix=time.time(),
        training_scenes=148,development_scenes=24,reserved_scenes=52,reserved_references_opened=0,
        unique_training_states=banks['train']['unique_state_archives'],
        unique_development_states=banks['development']['unique_state_archives'],
        verified_field_arrays=8*sum(bank['unique_state_archives'] for bank in banks.values()),
        complete_training_K4_windows=592,complete_training_output_label_references=2368,
        densities_cells_per_metre=sorted({r['cells_per_metre'] for r in banks['train']['trajectories']}),
        training_scenes_by_density=dict(collections.Counter(str(r['cells_per_metre']) for r in banks['train']['trajectories'])),
        training_scenes_by_profile=dict(collections.Counter(r['profile'] for r in banks['train']['trajectories'])),
        training_scenes_by_motion=dict(collections.Counter(r['motion'] for r in banks['train']['trajectories'])),
        group_splits_disjoint=True,optimizer_updates=0,model_fitting_executed=False,
        transition_supervision_contract_still_requires_freeze=True,latency_claim=False,
        evidence_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
        banks={split:dict(path=str((args.output/(split+'-bank.json')).relative_to(ROOT)),sha256=digest(args.output/(split+'-bank.json')))
               for split in banks})
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Assembly input changed during execution: '+path)
    write(args.output/'results.json',summary)
    print('Assembled148train/24development scenes;',summary['unique_training_states'],'training states;52reserved untouched.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
