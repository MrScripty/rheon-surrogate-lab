"""Package verified GPU full-state replays for native comparison, without inference.

Every scalar points to its original JSON value. Missing case coverage and timing
remain explicit. One union preserves the residuals' entire population; its shared
anchor filter supplies the exact common population for comparison with the cell
coordinate control. Temporal pilots have their own physical-time scope.
"""
import argparse
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'
DOCS=ROOT/'docs/plans/model-portfolio-migration-v1'
SOURCE=BASE/'sources/Tuldok-composition-source-artifacts'
sys.path.insert(0,str(SOURCE))
from retained_measurements import MeasurementTable, encode, digest, MAX_BYTES

METRICS=(
    ('pressure_relative_rms','Full-state pressure error (mean-free RMS)','1','relative','pressure'),
    ('velocity_relative_rms','Full-state velocity error (concatenated RMS)','1','relative','velocity'),
    ('tracer_relative_rms','Full-state tracer error (RMS)','1','relative','tracer'),
    ('pressure_equation_relative_l2','Full-state pressure equation residual (L2)','1','relative','equation'),
    ('pressure_absolute_rms','Pressure absolute RMS error','captured pressure units','absolute_rms','pressure'),
    ('velocity_absolute_rms','Velocity absolute RMS error','captured velocity units','absolute_rms','velocity'),
    ('tracer_absolute_rms','Tracer absolute RMS error','captured tracer units','absolute_rms','tracer'),
    ('complete_block_ms','Complete K-output GPU block latency','ms',None,None),
    ('per_output_ms','Complete GPU latency per physical output','ms/output',None,None),
)
CASE_COLUMNS=(('grid_x','Domain X cells','integer','cells'),('grid_y','Domain Y cells','integer','cells'),
    ('grid_z','Domain Z cells','integer','cells'),('cells_per_metre','Grid density','number','cells/m'),
    ('trajectory_id','Original trajectory','string',None),('split','Recorded scene split','string',None),
    ('K','Physical outputs per pass','integer','outputs'),('lead','Physical output lead','integer','outputs'),
    ('output_dt_s','Physical duration per output','number','s'),('output_steps_per_s','Output simulation steps/s','number','outputs/s'),
    ('elapsed_s','Physical lead from input','number','s'),('block_duration_s','Physical block duration','number','s'),
    ('reference_substeps','Reference updates per output','integer','updates/output'),
    ('start_index','Original start index (scope defines clock)','integer','index'),
    ('shared_anchor_case','Exact cell/control shared population','boolean',None),
    ('force_profile','Recorded force profile','string',None),('force_motion','Recorded force motion','string',None),
    ('fluid_density_kg_m3','Scoring equation fluid density','number','kg/m3'))
SUBJECT_COLUMNS=(('inference_device','Inference device','string',None),('gpu_model','Recorded GPU','string',None),
    ('execution_variant','Recorded execution variant','string',None),('fit_item_id','Residual fit-card identity','string',None),
    ('parent_fit_item_id','Frozen parent fit-card identity','string',None),('model_identity_sha256','Complete model identity digest','string',None),
    ('pressure_dtype','Pressure output dtype','string',None),('carrier_dtype','Carrier output dtype','string',None),
    ('tf32','TF32 enabled','boolean',None),('autocast','Autocast enabled','boolean',None))
OBS_COLUMNS=(('advances_block','Actual internal advances per block','integer','advances'),
    ('advances_output','Actual internal advances per output','integer','advances/output'),
    ('residual_calls','Residual network calls per output','integer','calls/output'),
    ('diagnostic_pass','Recorded four-field diagnostic pass','boolean',None))


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path,expected=None):
    raw=path.read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest()==expected,'Changed retained source: '+str(path))
    return json.loads(raw),raw


def src(identity,label,raw):
    return dict(id=identity,label=label,kind='json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                data_base64=base64.b64encode(raw).decode('ascii'))


def definitions(normalization,normalization_sha,scorer_sha):
    result=[]
    for identity,label,units,group,field in METRICS:
        value=dict(id=identity,label=label,units=units,kind='unqualified_latency' if group is None else 'retained_accuracy',
                   scorer=dict(path='experiments/full-ai-tight-targets-v1/diagnose_causal_composition.py',sha256=scorer_sha))
        if group=='relative' and field!='equation':
            value.update(formula='RMS(pred-reference) / max(RMS(reference), frozen training-only RMS floor)',
                         pressure_gauge='Mean removed independently from predicted/reference pressure' if field=='pressure' else None,
                         velocity_layout='Concatenated vx/vy/vz' if field=='velocity' else None,
                         denominator_floor=normalization['relative_denominator_rms_floor'][field],normalization_sha256=normalization_sha,
                         actual_per_row_denominator='Not retained by original evaluator; not reconstructed here.')
        elif field=='equation':value.update(formula='L2(fine_action(pressure)-native_rhs) / max(L2(native_rhs),1e-30)',rhs_floor=1e-30)
        elif group=='absolute_rms':value.update(formula='RMS(pred-reference), using same pressure gauge and component layout as relative metric')
        else:value.update(boundary='Complete current native CPU fields to all K owned CPU output states, including packing, H2D, all actual advances, output allocation, D2H and synchronization.',
                          divisor='K physical outputs only for per_output_ms; no division by internal advances.',
                          reason='All accepted replay captures explicitly have latency_claim=false. No historical timing is substituted.')
        result.append(value)
    return result


def key(row,temporal=False):
    return (row['identifier'],row['start_fine_step'] if temporal else row['start'],
            row.get('output_steps_per_s',60),row['K'],row['lead'])


def checked_rows(value,temporal=False):
    result={}
    for index,row in enumerate(value['rows']):
        identity=key(row,temporal)
        # A temporal source contains three executions of the same physical case.
        identity=(row['mode'],*identity) if temporal else identity
        require(identity not in result,'Duplicate physical output/execution in retained replay')
        require(row.get('metric_error') is None and row.get('metrics') is not None
                and not any(row['nonfinite_counts'].values()),'This fixed replay population must have finite scored outputs')
        require(all(math.isfinite(row['metrics'][g][f]) for _,_,_,g,f in METRICS if g is not None),
                'Nonfinite retained scalar cannot be represented as a measurement')
        result[identity]=(index,row)
    return result


def case(row,selection=None,temporal=False):
    dt=row['output_dt_s'] if temporal else selection['output_dt_s']
    start=row['start_fine_step'] if temporal else row['start']
    reference_substeps=240//row['output_steps_per_s'] if temporal else row['teacher_substeps']
    dims=dict(zip(('grid_x','grid_y','grid_z'),row['counts_xyz']))
    dims.update(cells_per_metre=row['cells_per_metre'],trajectory_id=row['identifier'],split=row['split'],K=row['K'],lead=row['lead'],
        output_dt_s=dt,output_steps_per_s=row.get('output_steps_per_s',60),elapsed_s=row.get('elapsed_s',row['lead']*dt),
        block_duration_s=row.get('block_duration_s',row['K']*dt),reference_substeps=reference_substeps,start_index=start,
        shared_anchor_case=None,force_profile=row.get('force_profile',selection.get('profile') if selection else None),
        force_motion=row.get('force_motion',selection.get('motion') if selection else None),fluid_density_kg_m3=1000.)
    binding=dict(trajectory_id=row['identifier'],start_index=start,K=row['K'],lead=row['lead'],output_dt_s=dt,
                 reference_substeps=reference_substeps,start_index_clock='master_240_updates' if temporal else 'physical_output_indices')
    if not temporal:
        require(selection['counts_xyz']==row['counts_xyz'] and selection['cells_per_metre']==row['cells_per_metre']
                and selection['reference_substeps']==row['teacher_substeps'],'Case geometry/density/reference regime differs from original selection')
        states={x['step']:x for x in selection['states']}
        binding.update(receipt=dict(path=selection['receipt'],sha256=selection['receipt_sha256']),
            reference_scheme_id=selection.get('reference_scheme_id'),
            temporal_convergence_qualified=selection.get('temporal_convergence_qualified'),
            native_input={k:states[start].get(k) for k in ('path','sha256','time_s')},
            native_reference={k:states[start+row['lead']].get(k) for k in ('path','sha256','time_s')})
    # Local display key only; the complete source definition/binding governs
    # comparison, and the full population must have unique keys below.
    return dict(id='c-'+digest(binding)[:16],dimensions=dims,binding=binding)


def observation_work(row,temporal=False,residual=False):
    if row is None:return dict.fromkeys(k for k,_,_,_ in OBS_COLUMNS)
    stages=len(row['groups_per_output']) if temporal else row['student_stages']
    total=row['actual_internal_advances'] if temporal else row['actual_internal_advances_per_block']
    require(total==row['K']*stages,'Recorded internal-work count differs from physical block')
    return dict(advances_block=total,advances_output=stages,
        residual_calls=3 if residual and row['teacher_substeps']==8 else 0,
        diagnostic_pass=row['all_accuracy_pass'])


def audited_master_ids(audits,expected):
    found=set()
    for audit in audits:
        require(audit['status']=='passed_completed_reference_labels_windows_and_late_replay',
                'Previously qualified selected temporal masters required')
        found.update(c['identifier'] for c in audit['cases'])
    require(found==set(expected),'Selected temporal master coverage differs from qualified audits')
    return found


def packet(context,subjects,result_sources,metrics):
    # Case dimensions are already retained in the native case table. Their
    # source reference supplies the complete input/reference binding once;
    # it does not duplicate the display columns or producer-only lookup keys.
    source_context={**context,'cases':[c['binding'] for c in context['cases']]}
    sources=[src('context','Frozen full-state comparison definitions and exact source bindings',encode(source_context)),*result_sources]
    dimensions=[dict(id=i,label=l,type=t,units=u,scope=owner) for owner,cols in
                [('case',CASE_COLUMNS),('subject',SUBJECT_COLUMNS),('observation',OBS_COLUMNS)] for i,l,t,u in cols]
    cases=[dict(id=c['id'],dimensions=c['dimensions'],reference=dict(source_id='context',pointer='/cases/'+str(i)))
           for i,c in enumerate(context['cases'])]
    rows=[];native_subjects=[]
    for subject in subjects:
        native_subjects.append(dict(id=subject['id'],label=subject['label'],dimensions=subject['dimensions'],
                                    reference=dict(source_id=subject['source_id'],pointer='/model_identity')))
        for c in context['cases']:
            found=subject['rows'].get(tuple(c['key']))
            index,row=found if found is not None else (None,None)
            work=observation_work(row,context['temporal'],subject.get('residual',False))
            for metric,_,_,group,field in METRICS:
                unavailable=row is None or group is None
                if row is None:
                    pointer='/status';reason='No scored output for this case.'
                elif group is None:
                    pointer='/latency_claim';reason='Capture declares latency_claim=false.'
                else:pointer=f'/rows/{index}/metrics/{group}/{field}';reason=None
                rows.append(dict(id='o'+str(len(rows)).zfill(6),subject_id=subject['id'],case_id=c['id'],metric_id=metric,
                    state='unavailable' if unavailable else 'measured',value=None if unavailable else row['metrics'][group][field],
                    reason=reason,dimensions=work,reference=dict(source_id=subject['source_id'],pointer=pointer)))
    return dict(format='retained_measurements_v1',schema_version=1,title=context['scope']['label'],
        scope={**{k:context['scope'][k] for k in ('id','label','population')},'definition':dict(source_id='context',pointer='/scope',sha256=digest(context['scope']))},
        dimensions=dimensions,subjects=native_subjects,cases=cases,
        metrics=[dict(id=m['id'],label=m['label'],units=m['units'],definition=dict(source_id='context',pointer='/metrics/'+str(i),sha256=digest(m))) for i,m in enumerate(metrics)],
        observations=rows,sources=sources)


def prepare(output):
    require(not output.exists() and output.is_relative_to(BASE),'Fresh task-owned packet output required')
    normal_path=ROOT/'docs/plans/full-ai-tight-targets-v1/sampler-accuracy-normalization-01.json'
    norm,_=read(normal_path);normal_sha=sha(normal_path)
    scorer=ROOT/'experiments/full-ai-tight-targets-v1/diagnose_causal_composition.py';scorer_sha=sha(scorer)
    metrics=definitions(norm,normal_sha,scorer_sha)
    anchor,anchor_raw=read(DOCS/'reports/anchor-accuracy-reproduction-01.json')
    residual,residual_raw=read(DOCS/'reports/full-state-residual-accuracy-reproduction-01.json')
    work,_=read(BASE/'discovery/model-worklist-05.json')
    accepted=[('cell-coordinate',anchor['comparisons'][0]),*[(c['variant'],c) for c in residual['comparisons']],('temporal',anchor['comparisons'][1])]
    loaded={}
    for name,item in accepted:
        require(item['status']=='exact_reproduction' if 'status' in item else item['same_full_accuracy_rows'] and item['same_output_hashes'],
                'Exact accepted reproduction is required')
        value,raw=read(ROOT/item['replay']['path'],item['replay']['sha256'])
        require(value['latency_claim'] is False and value['cpu_model_forward'] is False
                and value['inference_device']=='cuda' and value['learned_parameters_unchanged'], 'Verified unchanged GPU accuracy capture required')
        require(value['source_sha256'][str(scorer.relative_to(ROOT))]==scorer_sha
                and value['source_sha256'][str(normal_path.relative_to(ROOT))]==normal_sha,'Scorer or training-only normalization binding changed')
        loaded[name]=(value,raw,item)
    selection_path=ROOT/'docs/plans/full-ai-data-diversity-v1/macrostep-student-block-cases-01.json'
    selection,_=read(selection_path);selected={c['identifier']:c for c in selection['cases']}
    for name,(value,_,_) in loaded.items():
        if name!='temporal':require(value['source_sha256'][str(selection_path.relative_to(ROOT))]==sha(selection_path),'Original native case selection changed')
    cell_rows=checked_rows(loaded['cell-coordinate'][0]);reference_rows=checked_rows(loaded['residual-s1'][0])
    require(len(cell_rows)==1018 and len(reference_rows)==2120 and set(cell_rows)<=set(reference_rows),'Original shared/full populations differ')
    contexts=[];subjects_by_scope=[]
    for temporal in (False,True):
        baseline=checked_rows(loaded['temporal'][0],True) if temporal else reference_rows
        if temporal:baseline={k[1:]:v for k,v in baseline.items() if k[0]=='fine_reference'}
        cases=[]
        for k,(_,row) in baseline.items():
            c=case(row,None if temporal else selected[row['identifier']],temporal)
            c['key']=list(k);c['dimensions']['shared_anchor_case']=None if temporal else k in cell_rows;cases.append(c)
        require(len({c['id'] for c in cases})==len(cases),'Physical case identities repeat')
        scope=dict(id='old-temporal-full-state-180' if temporal else 'old-diversity-full-state-2120',
            label='Temporal full-state pilots · 30/60/120 outputs/s' if temporal else 'Expanded full-state controls and reduced-pass residuals',population='consumed',
            task='All pressure, velocity and tracer physical outputs; every K lead scored.',
            physical_duration='Each output spans output_dt_s; block spans K*output_dt_s, never output_dt_s/K.',
            comparison='All three modes share180cases.' if temporal else 'Use shared_anchor_case=true to compare all9executions on the exact1018common cases. All2120residual cases remain retained; absent control cases are explicit unavailable.',
            normalization='Frozen training-only RMS floors; mean-free pressure and concatenated velocity components. Actual per-row denominators were not retained.',
            diagnostic_gate='Recorded four relative diagnostics <=1e-5, not application/drift acceptance.',
            source_binding='Original reproduced captures and frozen protocols. No fresh inference, independent holdout or speed qualification.',
            absolute_time='Native time_s is retained when present; absent timestamps remain null, not inferred from frame numbers.',
            case_binding_sha256=digest([c['binding'] for c in cases]))
        context=dict(scope=scope,cases=cases,metrics=metrics,temporal=temporal,
            normalization=norm,normalization_sha256=normal_sha,scorer_sha256=scorer_sha,
            reproduction_receipts=[dict(path=str((DOCS/'reports'/n).relative_to(ROOT)),sha256=sha(DOCS/'reports'/n)) for n in
                                   ['anchor-accuracy-reproduction-01.json','full-state-residual-accuracy-reproduction-01.json']])
        context['capture_inventory']=[dict(source_id='r'+str(i),name=name,**item['replay']) for i,(name,(_,_,item)) in enumerate(loaded.items())]
        if temporal:
            protocol_path=ROOT/'docs/plans/full-ai-data-diversity-v1/temporal-frozen-backbone-qualification-protocol-03.json'
            protocol,_=read(protocol_path,loaded['temporal'][0]['protocol_sha256'])
            scope['audited_native_masters']=[]
            audits=[]
            for audit_path in protocol['audits']:
                original_audit,_=read(ROOT/audit_path,loaded['temporal'][0]['source_sha256'][audit_path])
                audits.append(original_audit)
                scope['audited_native_masters'].append(dict(path=audit_path,sha256=sha(ROOT/audit_path),
                    selected_identifiers=[c['identifier'] for c in original_audit['cases']],
                    all_four_original_pilots_valid=original_audit['all_four_pilots_valid']))
            audited_master_ids(audits,{c['dimensions']['trajectory_id'] for c in cases})
            scope['master_scope']='Two previously audited training masters only; all-four-pilot or held-out confirmation is not claimed.'
        subjects=[]
        for capture_number,(name,(value,raw,item)) in enumerate(loaded.items()):
            if temporal!=(name=='temporal'):continue
            modes=['fine_reference','half_passes','single_pass'] if temporal else [None]
            for mode in modes:
                rows=checked_rows(value,temporal)
                if temporal:rows={k[1:]:v for k,v in rows.items() if k[0]==mode}
                require(set(rows)<=set(baseline),'Execution has outputs outside its declared native case population')
                by_key={tuple(c['key']):c for c in cases}
                for k,(_,row) in rows.items():
                    actual=case(row,None if temporal else selected[row['identifier']],temporal)
                    expected=by_key[k]
                    require(actual['binding']==expected['binding'],'Equal-named cases have different native inputs/references')
                    if not temporal:
                        for field in ('native_input','native_reference'):
                            bound=actual['binding'][field]
                            require(value['source_evidence'][bound['path']]==bound['sha256'],
                                    'Replay did not bind the selected native input/reference bytes')
                model=value['model_identity'];pair=model.get('residual_pair');fit=None
                if pair:
                    found=[f['item_id'] for f in work['fit_cards'] if any(p.startswith(pair+'/') for p in f['receipt_paths'])]
                    require(len(found)==1,'Residual fit provenance is ambiguous');fit=found[0]
                precision=value.get('precision',{})
                parent=[f['item_id'] for f in work['fit_cards'] if any(p.startswith(model['parent_pair']+'/') for p in f['receipt_paths'])]
                require(parent==['fit-718037c1d106335e'],'Frozen parent fit identity differs')
                identity='temporal-'+mode if temporal else name
                dims=dict(inference_device=value['inference_device'],gpu_model=value['gpu_model'],execution_variant=mode or item.get('execution','original_N1_N8'),
                    fit_item_id=fit,parent_fit_item_id='fit-718037c1d106335e',model_identity_sha256=digest(model),
                    pressure_dtype=value.get('pressure_dtype',precision.get('pressure')),carrier_dtype=value.get('carrier_dtype',precision.get('carriers')),
                    tf32=value.get('tf32',precision.get('tf32')),autocast=value.get('autocast',precision.get('autocast')))
                subjects.append(dict(id=identity,label=identity.replace('-',' '),dimensions=dims,source_id='r'+str(capture_number),capture_name=name,rows=rows,residual=pair is not None,
                    receipt=item['replay']))
        contexts.append(context);subjects_by_scope.append(subjects)
    output.mkdir();packets=[]
    for context,subjects in zip(contexts,subjects_by_scope):
        # One or two executions per bounded packet; preserve identical scope,
        # metric and case definitions across partitions. No scientific fallback.
        pending=list(subjects)
        while pending:
            count=min(2,len(pending))
            while True:
                chosen=pending[:count];names={s['source_id']:s['capture_name'] for s in chosen}
                p=packet(context,chosen,[src(i,'Original '+n+' · '+loaded[n][2]['replay']['path'],loaded[n][1]) for i,n in names.items()],metrics)
                raw=encode(p)
                if len(raw)<=MAX_BYTES:break
                require(count>1,'One execution exceeds native packet byte limit');count=1
            table=MeasurementTable(raw);filename=f'full-state-part-{len(packets)+1:02d}.json';(output/filename).write_bytes(raw)
            packets.append(dict(path=filename,sha256=table.sha256,bytes=len(raw),scope=p['scope']['id'],scope_definition_sha256=p['scope']['definition']['sha256'],
                subjects=[s['id'] for s in chosen],observations=len(p['observations']),measured=sum(o['state']=='measured' for o in p['observations']),
                unavailable=sum(o['state']=='unavailable' for o in p['observations'])))
            del pending[:count]
    bundle=dict(schema='rheon.full-state-native-measurement-bundle.v1',status='prepared_not_native_imported',packets=packets,
        source_reproduction_reports=[hashlib.sha256(x).hexdigest() for x in [anchor_raw,residual_raw]],
        execution_subjects=12,scored_physical_outputs=18518,new_neural_forwards=0,new_scoring=False,new_reference_generation=False,
        qualified_latency=False,independent_confirmation=False,shared_anchor_cases=1018,full_residual_cases=2120,temporal_cases_per_mode=180,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str((SOURCE/'retained_measurements.py').relative_to(ROOT)):sha(SOURCE/'retained_measurements.py')})
    require(sum(p['measured'] for p in packets)==129626 and sum(p['observations'] for p in packets)==176580,'Complete full-state scalar denominator differs')
    (output/'bundle.json').write_bytes(encode(bundle))
    from projects import manifest as validate_manifest
    manifest=dict(schema_version=1,id=digest(bundle)[:32],name='Full-state and temporal comparison · retained GPU accuracy',
        data=[dict(id='full-state-part-'+str(i+1).zfill(2),label=p['scope']+' · '+', '.join(p['subjects']),
                   path=str((output/p['path']).relative_to(ROOT)),sha256=p['sha256']) for i,p in enumerate(packets)],
        recipes=[],evidence=[],requirements=[])
    validate_manifest(manifest);(output/'source-project.json').write_bytes(encode(manifest))
    require(not any(n=='torch' or n.startswith('torch.') or n=='numpy' for n in sys.modules),'No numerical runtime belongs in packaging')
    print(json.dumps(dict(status=bundle['status'],packets=len(packets),bytes=sum(p['bytes'] for p in packets),measured=129626,observations=176580)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
