"""Audit original requirements against current artifacts without admitting work.

Missing results remain pending. A failed target remains failed even if every
experiment ran. This tool never updates the thread goal or selects a new seed.
"""
import math
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


class Pending(Exception):pass


def run():
    sources={};checks={};details={}
    def document(path):
        if not path.is_file():raise Pending(str(path.relative_to(ROOT)))
        value=read(path);sources[str(path.relative_to(ROOT))]=sha(path);return value
    def bound(path,digest):
        if not path.is_file():raise Pending(str(path.relative_to(ROOT)))
        if sha(path)!=digest:raise ValueError('Changed evidence: '+str(path.relative_to(ROOT)))
        sources[str(path.relative_to(ROOT))]=digest
    def report(path):
        value=document(path)
        if value['status']!='completed':raise ValueError('Report is not complete: '+str(path.relative_to(ROOT)))
        for relative,digest in value.get('source_evidence',value.get('evidence_sha256',{})).items():bound(ROOT/relative,digest)
        return value
    def owner(name):
        value=document(BASE/name/'execution.json')
        if value['status']=='running':raise Pending('Live owner: '+name)
        if value['status']!='completed'or not value.get('owned_child_reaped',False)or value.get('active')is not None:
            raise ValueError('Owner did not complete/reap: '+name)
        if any(not c['reaped']or c['returncode']!=0 for c in value['completed']):raise ValueError('Owner retains failed numerical work: '+name)
        return value
    original=document(DOCS/'registration.json');base=document(DOCS/'requirements-audit.json')
    definitions={r['id']:r['requirement']for r in original['requirements']}
    for identifier,requirement in definitions.items():checks[identifier]=dict(id=identifier,requirement=requirement,status='pending')
    def group(identifiers,operation):
        try:
            note=operation();status='passed'
        except Pending as error:status='pending';note=str(error)
        except (AssertionError,ValueError,KeyError,TypeError)as error:status='failed';note=type(error).__name__+': '+str(error)
        for identifier in identifiers:checks[identifier].update(status=status,note=note)
    def preserved():
        if sha(DOCS/'plan.md')!=original['plan_sha256']or sha(DOCS/'requested-scope.md')!=original['requested_scope_sha256']:
            raise ValueError('Original plan/requested scope changed')
        for path in ('fixed-viscosity-replication-fit-registration-01.json',
                     'fixed-viscosity-replication-postprocessing-registration-01.json',
                     'temporal-final-confirmation-registration-01.json','application-domain-change-registration-01.json',
                     'application-domain-change-repair-registration-02.json'):
            value=document(DOCS/path)
            for name,digest in value['source_sha256'].items():bound(HERE/name,digest)
        return 'Original plan/scope and all registered training, execution and final-confirmation sources unchanged'
    group(('P-immutable',),preserved)
    # These scopes already have completed qualified studies. Read their actual
    # evidence and verify every current binding; later temporal gates below do
    # not inherit the old failed-model screen or its timing/accuracy grades.
    retained={r['id']:r for r in base['requirements']}
    preserved_ids=[i for i in definitions if i.startswith('A-')and i!='A-target']+['B-contract','B-restart']
    for identifier in preserved_ids:
        def previous(identifier=identifier):
            row=retained[identifier]
            if row['status']!='passed':raise Pending('Original qualified scope remains '+row['status'])
            if not row['evidence']:raise ValueError('No authoritative evidence')
            for entry in row['evidence']:
                bound(ROOT/entry['path'],entry['sha256'])
                if (ROOT/entry['path']).suffix=='.json':document(ROOT/entry['path'])
            return row['note']+'; all current evidence identities verified'
        group((identifier,),previous)
    def pressure_target():
        value=report(DOCS/'pressure-confirmation-v2/results.json')
        if not value['primary_independent_target_pass']or not value['all_three_continuation_seeds_independent_target_pass']:
            raise ValueError('Independent pressure target/continuation seeds did not pass')
        if value['primary_selection_changed']or value['candidate_numerical_pressure_solve']or value['model_promotion']:
            raise ValueError('Pressure selection/inference/promotion policy differs')
        return 'Frozen original pressure primary and both continuation seeds pass the independent prepared-pressure target; native acceptance remains separate'
    group(('A-target',),pressure_target)
    def train_and_restore():
        owner('fixed-viscosity-replication-training-batch-01')
        family=report(DOCS/'fixed-viscosity-family-results.json');replicas=report(DOCS/'fixed-viscosity-replication-results.json')
        if not all(all(r.values())for r in family['control_audit'].values())or not all(all(r.values())for r in replicas['control_audit'].values()):
            raise ValueError('Actual full-state training control audit failed')
        registry=document(BASE/'fixed-viscosity-replication-inputs-01/models.json')
        if len(registry['models'])!=6:raise ValueError('Two additional seeds per selected bundle and B1required')
        for entry in registry['models']:
            pair=BASE/(entry['identifier']+'-pair-01');receipt=document(pair/'receipt.json')
            trained=document(pair/'train/receipt.json');restored=document(pair/'restore/receipt.json')
            history=document(pair/'restore/history.json')
            if receipt['status']!='completed'or trained['mode']!='train'or trained['optimizer_updates']!=entry['optimizer_updates']:
                raise ValueError('Actual registered managed updates required')
            if trained['execution']['device']!='cuda'or not trained['execution']['cuda_initialized']:
                raise ValueError('Actual CUDA training required')
            if restored['mode']!='restore'or restored['optimizer_updates']!=0 or restored['execution']['cuda_initialized']:
                raise ValueError('Independent zero-update CPU restoration required')
            if restored['originating_training_run']['run_id']!=receipt['training_run_id']or trained['closure']!=restored['closure']:
                raise ValueError('Typed CPU artifact origin/closure differs')
            for name in ('weights.npz','initial-weights.npz','history.json','normalization.json','probe.npy'):
                if sha(pair/'train'/name)!=sha(pair/'restore'/name):raise ValueError('Actual artifact bytes differ: '+name)
            if history['optimizer_work']['predicted_physical_states']!=36000:raise ValueError('Registered physical-target work differs')
            for kind,run_id in (('train',receipt['training_run_id']),('restore',receipt['restore_run_id'])):
                terminal=document(pair/('terminal-'+run_id+'.json'))
                if terminal['state']!='completed'or terminal['custody_phase']!='drained':raise ValueError('Every managed Run must complete/drain')
                for artifact in terminal['artifacts']:bound(pair/kind/artifact['name'],artifact['sha256'])
        details['managed_replica_pairs']=6
        return 'Six actual managed CUDA fits, typed exact CPU restores and 36000physical targets each verified; shared frozen component pretraining and every seed retained'
    group(('B-four-models','B-loss','B-continuation','B-seeds','P-custody'),train_and_restore)
    def reserved_bank():
        value=report(DOCS/'temporal-reserved-accuracy-results.json')
        if value['verified_state_archives']!=1928 or not value['original_scene_reservation_preserved']or not value['original_primary_seed_retained']:
            raise ValueError('Complete original reserved population/freeze required')
        reservation=document(DOCS/'temporal-scene-reservation.json')['rows']
        if len(reservation)!=32 or len({r['identifier']for r in reservation})!=32:
            raise ValueError('All32unique scene identities required')
        if {s:sum(r['split']==s for r in reservation)for s in ('train','development','reserved')}!={'train':16,'development':8,'reserved':8}:
            raise ValueError('Original whole-scene split differs')
        if {r['cells_per_metre']for r in reservation}!={8,16}or len({tuple(r['counts'])for r in reservation})!=4:
            raise ValueError('Four domain groups and both original spacings required')
        first=document(BASE/'temporal-data-01/bank.json');reserved=document(BASE/'temporal-reserved-240-01/bank.json')
        for bank in (first,reserved):
            bound(ROOT/'experiments/pressure-methods-v1/numerics.py',bank['numerical_reference_sha256'])
        if first['status']!='completed'or len(first['trajectories'])!=24 or reserved['status']!='completed'or len(reserved['trajectories'])!=8:
            raise ValueError('All original train/development/reserved captures required')
        for rows,expected_states in ((first['trajectories'],129),(reserved['trajectories'],241)):
            for row in rows:
                if row['states']!=expected_states:raise ValueError('Original consecutive frame count differs')
        if {r['identifier']for r in first['trajectories']} & {r['identifier']for r in reserved['trajectories']}:
            raise ValueError('Scene split leakage')
        frozen=DOCS/'temporal-final-selection-01.json';selection=document(frozen)
        if reserved['frozen_selection_sha256']!=sha(frozen)or selection['primary_seed_reselected']:
            raise ValueError('Reserved data must follow actual original-primary freeze')
        details['reserved_scenes']=8;details['reserved_state_archives']=1928
        return 'All32original whole scenes and16/8/8split,129train/dev states,241reserved states and pre-capture selection verified'
    group(('B-bank','B-frames','B-causality'),reserved_bank)
    def raw_outputs():
        report(DOCS/'temporal-reserved-accuracy-results.json')
        registry=document(BASE/'temporal-finalists-01/models.json');norm=document(BASE/'temporal-training-view-02/normalization.json')
        if len(registry['methods'])!=10:raise ValueError('All four primary horizons and six added seeds required')
        total=0;primary_pass={}
        for method in registry['methods']:
            identifier=method['identifier'];k=method['model']['horizons']
            directory=BASE/'temporal-reserved-evaluation-01'/identifier;summary=document(directory/'summary.json')
            if summary['status']!='completed'or len(summary['scenes'])!=8 or summary['candidate_numerical_pressure_solve']or summary['optimizer_updates']:
                raise ValueError('Complete actual every-scene neural evaluation required')
            if summary['model']['weights_sha256']!=method['model']['weights_sha256']or summary['model']['application_policy']!=method['model']['policy']:
                raise ValueError('Evaluated weights/arithmetic differ from pre-capture freeze')
            passed=True
            for scene in summary['scenes']:
                raw=document(directory/(scene['identifier']+'.json'));rows=raw['rows']
                single=[r for r in rows if r['mode']=='single_block'];own=[r for r in rows if r['mode']=='own_history']
                if len(single)!=121*k or {(r['start'],r['horizon'])for r in single}!={(s,h)for s in range(121)for h in range(1,k+1)}:
                    raise ValueError('Missing intermediate prediction')
                limit=240 if raw['first_unavailable_step']is None else raw['first_unavailable_step']-1
                if [r['step']for r in own]!=list(range(1,limit+1)):raise ValueError('Unreported own-history state or failure')
                for row in rows:
                    metric=row['metrics'];expected={'equation':metric['relative']['equation']<=.004}
                    for field in ('pressure','velocity','tracer'):
                        expected[field]=(metric['absolute_rms'][field]<=norm['degenerate_absolute_rms_tolerance'][field]
                            if metric['degenerate_reference'][field]else metric['relative'][field]<=.004)
                    if expected!=metric['criterion_pass']or all(expected.values())!=metric['all_accuracy_criteria_pass']:
                        raise ValueError('Raw accuracy decision differs from original limits')
                    if any(not math.isfinite(v)for v in metric['relative'].values()):raise ValueError('Nonfinite observed error')
                    for key in ('predicted_physics','reference_physics','tracer','absolute_rms'):
                        if key not in metric:raise ValueError('Required physical diagnostic missing')
                    total+=1
                for item in raw['summaries']:
                    selected=[r for r in rows if r['mode']==item['mode']and r['step']<=item['interval_steps']]
                    if item['outputs']!=len(selected)or item['worst']!={f:max(r['metrics']['relative'][f]for r in selected)for f in ('pressure','equation','velocity','tracer')}:
                        raise ValueError('Raw count/extrema differ from declared interval')
                    passed &= item['complete']and item['all_accuracy_criteria_pass']
                for field,levels in raw['diagnostic_drift_thresholds'].items():
                    for level,observed in levels.items():
                        first=next((r for r in own if r['metrics']['relative'][field]>float(level)),None)
                        expected=None if first is None else dict(step=first['step'],time_s=first['step']/60)
                        if observed!=expected:raise ValueError('Diagnostic threshold crossing differs from raw trace')
            if method['replication_ordinal']==1:primary_pass[k]=bool(passed)
        if set(primary_pass)!={1,2,3,4}:raise ValueError('Every original primary horizon required')
        details['raw_reserved_outputs_checked']=total;details['primary_reserved_accuracy_pass']=primary_pass
        return f'Raw coverage, physical diagnostics, original criteria, interval extrema and drift crossings checked for{total}physical outputs; unfavorable outcomes retained'
    group(('B-rollout','B-physical'),raw_outputs)
    def temporal_target():
        value=report(DOCS/'temporal-reserved-cost-results.json')
        registry=document(BASE/'temporal-finalists-01/models.json')
        primary_ids={m['identifier']for m in registry['methods']if m['replication_ordinal']==1 and m['model']['horizons']in(2,3,4)}
        points=[p for p in value['points']if p['identifier']in primary_ids]
        if len(points)!=24:raise ValueError('Every primary horizon, spatial tier and accuracy interval required')
        if not all(p['all_finite_complete']and p['accuracy_pass']for p in points):raise ValueError('Reserved primary field/equation accuracy fails')
        small=[p for p in points if p['regime']=='reserved_small'and p['interval']=='rollout_240']
        if {p['horizons']for p in small}!={2,3,4}or not all(p['median_ms']<2 and p['full_block_ms']<2*p['horizons']and p['target_met']for p in small):
            raise ValueError('A requested horizon misses its complete reserved combined median budget')
        if not all(details.get('primary_reserved_accuracy_pass',{}).get(k,False)for k in (2,3,4)):
            raise ValueError('Independent raw-record primary accuracy audit has not passed')
        details['reserved_primary_target_points']=small
        return 'Frozen K2/K3/K4pass every original reserved field/horizon/own240criterion and fresh complete small-grid median budgets;32cubedlatency/tails remain separate'
    group(('B-target',),temporal_target)
    def application_and_timing():
        owner('temporal-final-confirmation-batch-01')
        report(DOCS/'fixed-viscosity-replication-cost-results.json');report(DOCS/'temporal-reserved-cost-results.json')
        app=report(DOCS/'application-results/results.json')
        if app['timed_episodes']!=560 or app['repeats']!=5 or not app['serial_CPU']:
            raise ValueError('Full seven-method five-repeat application campaign required')
        campaign=BASE/'temporal-final-application-01';execution=document(campaign/'execution.json')
        if execution['status']!='completed'or len(execution['completed'])!=35 or not all(w['reaped']and w['returncode']==0 for w in execution['workers']):
            raise ValueError('Every application response/worker must complete and reap')
        count=0
        for response in execution['completed']:
            path=ROOT/response['path'];bound(path,response['sha256']);value=document(path)
            if len(value['rows'])!=16 or value['threads']!=1 or value['affinity']!=[23]or value['cuda_initialized']:
                raise ValueError('Application CPU/population differs')
            for row in value['rows']:
                path=ROOT/row['receipt'];bound(path,row['receipt_sha256']);episode=document(path);count+=1
                if not episode['offline_references_excluded_from_episode']or episode['warmup']:
                    raise ValueError('Independent ground truth entered application timing')
                if not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in episode['native_processes']):
                    raise ValueError('Native application child did not complete/reap')
                if episode.get('native_state_setter_is_structural_only'):
                    if episode['native_numerical_steps_executed']or episode['fallback']or not episode['native_final_state_bytes_exact']:
                        raise ValueError('Numerical correction/altered native state entered a candidate')
                    if len(episode['diagnostics']['rows'])!=episode['physical_steps']:
                        raise ValueError('Every consumed forecast state must have independent offline diagnostics')
                    if episode['generated_states']!=episode['physical_steps']+episode['event_discarded_states']+episode['endpoint_discarded_states']:
                        raise ValueError('Uncounted discarded prediction work')
                    if episode['scenario']=='forcing'and episode['event_discarded_states']!=episode['horizons']-1:
                        raise ValueError('Stale outputs not fully invalidated')
                elif not episode['native_acceptance_gates_unchanged']or episode['fallback']:
                    raise ValueError('Native control policy differs')
        if count!=560:raise ValueError('Every timed application episode required')
        details['application_episodes_verified']=count
        return 'All560actual steady/event episodes, seven methods, five serial CPU repeats, causal inputs, discarded work and process custody verified; native rejection has no completed throughput claim'
    group(('B-timing','B-controls'),application_and_timing)
    def supported_boundary_events():
        if details.get('application_episodes_verified')!=560:raise Pending('Original forcing/application campaign still incomplete')
        owner('application-domain-change-repair-batch-02')
        repair=document(DOCS/'application-domain-change-repair-registration-02.json')
        failure=repair['original_failure']
        bound(ROOT/failure['execution_path'],failure['execution_sha256'])
        bound(ROOT/failure['log_path'],failure['log_sha256'])
        prior=document(ROOT/failure['execution_path'])
        if prior['status']!='stopped' or not prior['owned_child_reaped']:
            raise ValueError('Original bookkeeping failure must remain retained/reaped')
        retained=repair['retained_case'];bound(ROOT/retained['path'],retained['sha256'])
        value=report(DOCS/'application-domain-change-results.json')
        if value['completed_original_episodes_reused']!=1 or value['new_episodes_executed']!=31 or not value['original_failed_owner_preserved']:
            raise ValueError('Exactly one retained episode plus31new episodes and original failure history required')
        if len(value['rows'])!=32 or value['physical_states_scored']!=1536 or not value['supported_domain_boundary_change_regenerates_predictions']:
            raise ValueError('Actual supported-boundary continuation required for all four horizons and eight scenes')
        methods={m['identifier']:m for m in document(BASE/'temporal-finalists-01/application-models.json')['methods']if m['kind']=='forecast'}
        if {m['model']['horizons']for m in methods.values()}!={1,2,3,4}:raise ValueError('All actual primary horizons required')
        for row in value['rows']:
            path=ROOT/row['receipt'];bound(path,row['receipt_sha256']);episode=document(path)
            method=methods[row['model']];k=method['model']['horizons'];old=row['cells_per_metre_before'];new=row['cells_per_metre_after']
            if new!=24-old or episode['model']['weights_sha256']!=method['model']['weights_sha256']:
                raise ValueError('Boundary geometry or actual checkpoint identity differs')
            if episode['physical_steps']!=48 or len(episode['diagnostics'])!=48 or episode['event_discarded_states']!=k-1:
                raise ValueError('Incomplete physical continuation or stale block discards')
            if episode['native_numerical_steps_executed']or episode['fallback']or not episode['native_final_state_bytes_exact']:
                raise ValueError('Native correction/altered fields entered a boundary candidate')
            if not any(c['start_step']==13 and c['cells_per_metre']==new for c in episode['calls']):
                raise ValueError('Boundary change did not trigger an actual fresh prediction')
            if any(c['cells_per_metre']!=(old if c['start_step']<13 else new)for c in episode['calls']):
                raise ValueError('Future geometry leaked or stale geometry reused')
            if any(d['cells_per_metre']!=(old if d['step']<=13 else new)for d in episode['diagnostics']):
                raise ValueError('Offline errors used the wrong physical geometry')
            if not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in episode['native_processes']):
                raise ValueError('Boundary native child did not complete/reap')
        guard=document(BASE/'forecast-application-qualification-01/receipt.json')
        if guard['status']!='passed':raise ValueError('Unsupported-boundary guard must remain qualified')
        details['supported_boundary_episodes_verified']=32
        return 'Actual forcing events and32supported physical-box boundary changes invalidate stale outputs and trigger fresh predictions; unsupported boundary types retain the explicit stop guard; all errors/discards reported'
    group(('B-events',),supported_boundary_events)
    def published():
        value=document(DOCS/'frontiers.json')
        required={'replica_accuracy','replica_cost','reserved_accuracy','reserved_cost','application','domain_change'}
        if not required<=set(value['temporal_confirmation']):raise Pending('Consolidated report still awaits confirmation reports')
        for relative,digest in value['evidence_sha256'].items():
            path=ROOT/relative
            if path.name=='execution.json'and path.is_relative_to(BASE)and value['batches'].get(path.parent.name,{}).get('status')=='running':
                if path.is_file()and sha(path)!=digest:
                    raise Pending('Report snapshot predates progress of tracked live owner: '+path.parent.name)
            bound(path,digest)
        for name in ('results.md','results.html','frontiers.csv','accuracy.csv',
                     'figures/pressure-precise-confirmation-frontier.svg','figures/temporal-reserved-rollout_240-frontier.svg'):
            path=DOCS/name
            if not path.is_file()or path.stat().st_size==0:raise Pending('Missing report artifact: '+name)
        if value['status']!='complete'or value['goal_complete']is not True:raise Pending('Final report still marked interim; scientific gates must pass before finalization')
        return 'Final separate pressure/temporal frontiers, all seed/reserved/application evidence and successful/failed outcomes published'
    group(('P-report',),published)
    scientific=all(r['status']=='passed'for i,r in checks.items()if i!='P-report')
    complete=scientific and checks['P-report']['status']=='passed'
    value=dict(status='completion_proven'if complete else'incomplete',completion_proven=complete,
        updated_unix=time.time(),
        scientific_gates_passed=scientific,requirements=list(checks.values()),details=details,
        original_registration_sha256=sha(DOCS/'registration.json'),source_evidence=sources,
        auditor_sha256=sha(__file__),goal_status_updated=False,
        pending=[i for i,r in checks.items()if r['status']=='pending'],failed=[i for i,r in checks.items()if r['status']=='failed'])
    write(DOCS/'completion-audit.json',value)
    lines=['# Current-state completion audit','',
        'This audit reads current artifacts, verifies source identities and checks raw final output/application records. '
        'Missing work remains pending; a completed experiment cannot substitute for a failed target. The thread goal is not changed.','',
        '| Requirement | Status | Finding |','| --- | --- | --- |']
    lines.extend(f'| {i} | {r["status"]} | {r.get("note","")} |'for i,r in checks.items())
    lines.extend(['','[Machine-readable audit and evidence](completion-audit.json)',''])
    (DOCS/'completion-audit.md').write_text('\n'.join(lines))
    print('Completion audit:',sum(r['status']=='passed'for r in checks.values()),'passed;',len(value['pending']),'pending;',len(value['failed']),'failed; goal unchanged',flush=True)


if __name__=='__main__':run()
