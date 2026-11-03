"""Read-only audit of the Full AI scatter; no inference, simulation or chart edits.

Run: python3 docs/plans/full-ai-chart-audit-2026-10-19/audit.py
All output goes beside this script. Unknown settings remain unknown.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import ast,csv,hashlib,html,json,math,re,statistics,struct,zipfile

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
LAB=BASE.parent.parent
PAGE=BASE/'cohort-progress.html'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
chart_sha=sha(PAGE)
text=PAGE.read_text()
d=json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>',text,re.S)[1])
full=[p for p in d['cohort_points'] if d['solver_comparison']['classification'][p['id']]['kind']=='full']
sources={};source_checks={};failures=[]

def load(source,expected=None):
    path=BASE/source
    if not path.exists():
        archived=LAB/'runs/dependency-migration-v1/lab'/path.resolve().relative_to(LAB)
        if archived.exists():path=archived
    raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
    recorded=expected or d['history_sources'].get(source)
    sources[source]=actual
    source_checks[source]=dict(sha256=actual,recorded_sha256=recorded,status='verified' if recorded==actual else 'changed' if recorded else 'not_pinned_in_chart',resolved_path=__import__('os').path.relpath(path,BASE))
    if recorded and actual!=recorded:failures.append('Source digest changed: '+source)
    return ([json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix=='.jsonl' else json.loads(raw))

def identity(c):
    spacing=c.get('cell_size_m',c.get('spacing_m'))
    if isinstance(spacing,list)and len(set(spacing))==1:spacing=spacing[0]
    return (c.get('trajectory_id',c.get('case_id')),c.get('native_frame',c.get('frame')),tuple(c.get('counts_xyz',[])),spacing,c.get('split'))

def popkey(cases):return sorted(json.dumps(identity(c))for c in cases)
refpoint=next(p for p in full if p['cases']==128 and p.get('timing_cases')and all(c['split']=='validation'for c in p['timing_cases']))
reference=popkey(refpoint['timing_cases'])
current_builder_sha=sha(BASE/'build-cohort-diagnostics.py')
sources['build-cohort-diagnostics.py']=current_builder_sha
canonical_manifest_path='../../runs/water-early-pressure-v1/control/project/inputs/validation_splits.json'
canonical_manifest=load(canonical_manifest_path)

def field_identity(manifest):
    return sorted(json.dumps([p['entry']['trajectory_id'],f['index'],f.get('sha256'),p['counts_xyz'],p['cell_size_m'],p.get('density_kg_m3'),p['entry'].get('dt_s')],sort_keys=True)for p in manifest['parents']for f in p['frames'])
canonical_fields=field_identity(canonical_manifest)
records=[];raw_by_id={};timing_by_id={}
for p in full:
    problems=[]
    acc=load(p['best_error_source']);cases=acc['cases']if isinstance(acc,dict)else acc
    raw_by_id[p['id']]=cases
    errors=[(c['raw_learned'].get('quality')or c['raw_learned'])['relative_residual']for c in cases]
    checks=dict(accuracy_cases=len(cases)==p['cases'],worst=math.isclose(max(errors),p['worst_error'],rel_tol=1e-12),median=math.isclose(statistics.median(errors),p['median_error'],rel_tol=1e-12),raw_best=math.isclose(min(errors),p['best_error_raw'],rel_tol=1e-12))
    outer=load(p['timing_source']);r=outer.get('recipe_receipt',outer)
    timing_by_id[p['id']]=outer
    if p['id']=='C02':
        times=[c['recorded_model_inference_s']for c in cases]
        timing_method='Retained development-prediction timings'
    else:
        times=r.get('per_case_inference_s',[])
        if len(times)!=p['timed_cases']:
            selected={json.dumps(identity(c))for c in p.get('timing_cases',[])}
            times=[t for c,t in zip(r.get('cases',[]),times)if json.dumps(identity(c))in selected]
        timing_method='Chart single-sweep CPU diagnostic' if r.get('schema')=='rheon-chart-validation-v1' else 'Managed CPU restore/test export' if p['id']!='C01' else 'Early fixed-grid prediction export'
    checks.update(timed_cases=len(times)==p['timed_cases'],timing_median=bool(times)and math.isclose(statistics.median(times)*1000,p['inference_median_ms'],rel_tol=1e-12),timing_min=bool(times)and math.isclose(min(times)*1000,p['inference_best_ms'],rel_tol=1e-12),timing_max=bool(times)and math.isclose(max(times)*1000,p['inference_worst_ms'],rel_tol=1e-12))
    if p.get('timing_cases'):
        checks['accuracy_timing_identities']=popkey(cases)==popkey(p['timing_cases'])
    elif r.get('witness_cases'):
        checks['accuracy_timing_identities']=popkey(cases)==popkey(r['witness_cases'])
    if not all(checks.values()):failures.append(p['id']+' values differ: '+str(checks))
    matched=bool(p.get('timing_cases'))and p['cases']==128 and all(c.get('split')=='validation'for c in p['timing_cases'])and popkey(p['timing_cases'])==reference
    rt=r.get('runtime',r.get('runtime_observations',{}))
    ex=r.get('execution',{})
    project=None;artifacts=None;entry=None
    if p.get('replay_project'):project=BASE/p['replay_project']
    elif r.get('project_source'):project=(BASE/r['project_source']).parent
    elif int(p['id'][1:])>=3:project=(BASE/p['timing_source']).parents[3]/'project'
    architecture=r.get('architecture',{})
    config={};manifest=None;manifest_source=None
    if p.get('replay_artifacts'):artifacts=BASE/p['replay_artifacts']
    if artifacts and (artifacts/'receipt.json').exists():
        rr=load(str((artifacts/'receipt.json').relative_to(BASE))if artifacts.is_relative_to(BASE)else __import__('os').path.relpath(artifacts/'receipt.json',BASE))
        architecture=rr.get('architecture',architecture)
        checks['weights_digest']=sha(artifacts/'weights.npz')==p['weights_sha256']
        if not checks['weights_digest']:failures.append(p['id']+' checkpoint differs')
    witness={}
    if project and (project/'source-witness.json').exists():
        witness=load(__import__('os').path.relpath(project/'source-witness.json',BASE),r.get('project_sha256')or r.get('source_witness_sha256')or r.get('closure',{}).get('source_witness_sha256'))
        if not architecture:architecture=witness.get('architecture',{})
    if project and (project/'config.json').exists():config=load(__import__('os').path.relpath(project/'config.json',BASE),witness.get('experiment_sha256'))
    if r.get('admission_source'):
        admission=load(r['admission_source'],r['admission_sha256'])
        entry=next(e for e in admission['models']if e['name']==p['name'])
        inputs=LAB/entry['validation_inputs']if entry.get('validation_inputs')else project/'inputs'
        manifest_source=__import__('os').path.relpath(inputs/'validation_splits.json',BASE)
    elif project and (project/'inputs/validation_splits.json').exists():manifest_source=__import__('os').path.relpath(project/'inputs/validation_splits.json',BASE)
    if manifest_source:
        manifest_expected=r.get('inputs',{}).get('validation_splits.json')or witness.get('inputs',{}).get('validation_splits',{}).get('sha256')
        manifest=load(manifest_source,manifest_expected)
    # Some retained chart diagnostics predate explicit replay paths. Recover
    # their captured architecture from the hash-bound admission.
    if artifacts is None and entry and entry.get('artifacts'):
        candidate=LAB/entry['artifacts']
        if (candidate/'receipt.json').exists():
            artifacts=candidate
            rr=load(__import__('os').path.relpath(artifacts/'receipt.json',BASE))
            architecture=rr.get('architecture',architecture)
            checks['weights_digest']=sha(artifacts/'weights.npz')==p['weights_sha256']
    weight_path=artifacts/'weights.npz' if artifacts and (artifacts/'weights.npz').exists() else None
    if weight_path is None and project:
        for candidate in sorted(project.parent.glob('managed*/stage-*/artifacts/weights.npz')):
            if sha(candidate)==p['weights_sha256']:
                weight_path=candidate;checks['weights_digest']=True;break
    weight_dtypes=[]
    if weight_path:
        source=__import__('os').path.relpath(weight_path,BASE);actual=sha(weight_path)
        sources[source]=actual;source_checks[source]=dict(sha256=actual,recorded_sha256=p['weights_sha256'],status='verified' if actual==p['weights_sha256']else 'changed',resolved_path=source)
        if actual!=p['weights_sha256']:failures.append('Checkpoint changed: '+p['id'])
        with zipfile.ZipFile(weight_path) as archive:
            for name in archive.namelist():
                if not name.endswith('.npy'):continue
                with archive.open(name)as stream:
                    magic=stream.read(8)
                    if magic[:6]!=b'\x93NUMPY':raise ValueError('Invalid tensor archive')
                    size=struct.unpack('<H'if magic[6]==1 else '<I',stream.read(2 if magic[6]==1 else 4))[0]
                    header=ast.literal_eval(stream.read(size).decode('latin1').strip());weight_dtypes.append(header['descr'])
    exported=r.get('prediction_inventory',r.get('predictions_inventory',{}))
    output_dtypes=sorted({spec.get('dtype')for spec in exported.values()if isinstance(spec,dict)and spec.get('dtype')})if isinstance(exported,dict)else []
    field_match=(field_identity(manifest)==canonical_fields if manifest and 'parents'in manifest and all('entry'in a and 'cell_size_m'in a for a in manifest['parents'])else None)
    proposal_stages=architecture.get('correction',config.get('correction',{})).get('steps')
    if proposal_stages is None and architecture and architecture.get('schema')not in ['rheon-physical-residual-pressure-v1']:
        proposal_stages=1
    if proposal_stages is None and project and (project/'model.py').exists():
        model_source=(project/'model.py').read_text()
        if 'physical-residual'not in str(config)and 'physical_residual'not in model_source:proposal_stages=1
    rows=p.get('timing_cases')or cases
    grids=[c['counts_xyz']for c in rows if c.get('counts_xyz')]
    h=[c['cell_size_m']for c in rows if c.get('cell_size_m')]
    density=sorted({a['density_kg_m3']for a in manifest.get('parents',[])if 'density_kg_m3'in a})if manifest else []
    dt=sorted({a['entry']['dt_s']for a in manifest.get('parents',[])if a.get('entry',{}).get('dt_s')is not None})if manifest else []
    context='torch.no_grad (current builder hash verified)' if r.get('builder_sha256')==current_builder_sha else 'Historical builder hash; exact source not retained in this audit' if r.get('schema')=='rheon-chart-validation-v1'else 'torch.no_grad (retained project prediction-export source)' if project and (project/'train.py').exists()and 'with torch.no_grad():'in(project/'train.py').read_text()else 'Not established'
    thread_source=None
    if project:
        for filename in ['model.py','original_model.py','baseline_model.py']:
            path=project/filename
            if path.exists()and 'torch.set_num_threads(1)'in path.read_text():thread_source=__import__('os').path.relpath(path,BASE);break
    if not matched:problems.append('Different/early input population; excluded from current Pareto recommendations')
    if r.get('schema')=='rheon-chart-validation-v1':problems.extend(['Single measured observation per case; no dedicated benchmark warmup/repetitions','Package versions absent from timing record'])
    if not rt.get('cpu_affinity'):problems.append('Timing CPU affinity absent from timing record')
    if 'threads'not in rt and 'torch_threads'not in rt:problems.append('Observed PyTorch thread count absent from timing record')
    raw_numerical=sum(bool(c['raw_learned'].get('any_pcg'))for c in cases)
    if raw_numerical:failures.append(p['id']+' raw metric includes PCG')
    record=dict(id=p['id'],name=p['name'],family=p['family'],matched_validation128=matched,accuracy_cases=len(cases),timing_cases=len(times),
        worst_error=p['worst_error'],median_error=p['median_error'],main_median_ms=p['inference_median_ms'],
        case_p95_ms=sorted(times)[math.ceil(.95*len(times))-1]*1000 if times else None,
        timing_method=timing_method,timing_schema=r.get('schema'),repetitions_per_case=1,warmup='No standardized dedicated warmup recorded',
        timing_context=context,builder_sha256=r.get('builder_sha256'),builder_source_verified=r.get('builder_sha256')==current_builder_sha,
        recorded_torch=rt.get('torch',rt.get('torch_version')),recorded_numpy=rt.get('numpy',rt.get('numpy_version')),
        recorded_threads=rt.get('threads',rt.get('torch_threads')),source_configures_threads_one=thread_source,
        recorded_interop_threads=rt.get('interop_threads',rt.get('torch_interop_threads')),recorded_affinity=rt.get('cpu_affinity'),
        recorded_device=ex.get('device',rt.get('device')),recorded_cuda_initialized=ex.get('cuda_initialized',rt.get('cuda_initialized')),
        architecture_dtype=architecture.get('dtype'),pressure_dtype=architecture.get('pressure_dtype'),weights_dtypes=sorted(set(weight_dtypes)),exported_pressure_dtypes=output_dtypes,weight_source=__import__('os').path.relpath(weight_path,BASE)if weight_path else None,
        fine_network_stages=proposal_stages,global_layers=dict(learned_coarse=bool(architecture.get('learned_coarse_revision')),shape_head=bool(architecture.get('shape_informed_coarse')),learned_tail=bool(architecture.get('final_learned_global_block'))),
        unique_simulations=len({c.get('trajectory_id')for c in rows if c.get('trajectory_id')}),
        frame_indices=sorted({c.get('native_frame',c.get('frame'))for c in rows if c.get('native_frame',c.get('frame'))is not None}),
        unique_grids=len({tuple(g)for g in grids}),cell_count_min=min(map(math.prod,grids))if grids else None,cell_count_max=max(map(math.prod,grids))if grids else None,
        cell_size_m=sorted(set(h)),density_kg_m3=density,dt_s=dt,manifest_source=manifest_source,
        field_hash_metadata_matches_reference=field_match,accuracy_backend='Python/Torch equation residual only' if outer.get('schema')=='rheon-chart-validation-v1' else 'Native verification of saved pressure predictions',original_proposal=True,closed_loop_steps_measured=0,
        entire_population_meets_chart_equation_target=max(errors)<=d['model_equation_target'],
        raw_cases_with_pcg=raw_numerical,checks=checks,issues=problems,
        accuracy_source=p['best_error_source'],timing_source=p['timing_source'],architecture_source=__import__('os').path.relpath(artifacts/'receipt.json',BASE)if artifacts else p['timing_source'],
        source= p['source'],training_source=p['training_source'])
    records.append(record)

frontier=[p for p in records if p['matched_validation128'] and not any(q['matched_validation128'] and q['worst_error']<=p['worst_error']and q['main_median_ms']<=p['main_median_ms']and(q['worst_error']<p['worst_error']or q['main_median_ms']<p['main_median_ms'])for q in records)]
for p in records:p['on_default_frontier']=p in frontier
repeated_path='../../runs/full-ai-next-priorities-v1/pressure-compiled-timing-01/summary.json'
repeated=load(repeated_path)
contrasts=[]
for identifier,id in [('C107','C107'),('a-head-calibration-s2','C126')]:
    r=next(r for r in repeated['rows']if r['checkpoint']==identifier and r['variant']=='original')
    main=next(p for p in records if p['id']==id)
    contrasts.append(dict(id=id,main_ms=main['main_median_ms'],repeated_ms=r['median_ms'],repeated_min_ms=r['min_repeat_median_ms'],repeated_max_ms=r['max_repeat_median_ms'],main_over_repeated=main['main_median_ms']/r['median_ms'],worst_error=main['worst_error']))
# A common isolated export campaign is an independent check on historical timing drift.
pass_examples=[]
for id in ['C24','C25','C26','C36','C53','C88']:
    main=next(p for p in records if p['id']==id)
    stage=main['fine_network_stages']
    settings=[r for r in d['pass_sweep']['points']if r['model']==id and r['passes']==stage]
    if settings:
        r=settings[0]
        pass_examples.append(dict(id=id,fine_stages=stage,main_ms=main['main_median_ms'],isolated_pass_ms=r['inference_median_ms'],ratio=r['inference_median_ms']/main['main_median_ms'],pass_source=r['timing_source'],worst_error_matches=math.isclose(main['worst_error'],r['worst_error'],rel_tol=1e-12),scope_warning='The isolated fixed-pass recurrence includes final equation-error verification; the main chart excludes that verification. Its wrapper also updates the residual between fine passes. This contrast is not a pure estimate of runtime noise.'))
counts=dict(full_points=len(records),matched_validation_points=sum(r['matched_validation128']for r in records),different_population_ids=[r['id']for r in records if not r['matched_validation128']],default_frontier_ids=[r['id']for r in sorted(frontier,key=lambda r:r['main_median_ms'])],
    timing_methods=dict(Counter(r['timing_method']for r in records)),recorded_torch_versions=dict(Counter(r['recorded_torch']or'not recorded'for r in records)),
    observed_thread_counts=dict(Counter(str(r['recorded_threads'])for r in records)),affinity_recorded=sum(r['recorded_affinity']is not None for r in records),
    fine_stage_counts=dict(Counter(str(r['fine_network_stages'])for r in records)),field_hash_matches=sum(r['field_hash_metadata_matches_reference']is True for r in records),
    all_reconstructed_values_match=all(all(r['checks'].values())for r in records),entire_population_meets_chart_target=sum(r['entire_population_meets_chart_equation_target']for r in records),
    weights_verified=sum(bool(r['weight_source'])for r in records),float32_weights_verified=sum(r['weights_dtypes']==['<f4']for r in records),
    source_digest_status=dict(Counter(r['status']for r in source_checks.values())))
parents=canonical_manifest['parents'];domains=[dict(counts_xyz=p['counts_xyz'],cell_size_m=p['cell_size_m'],cells_per_metre=1/p['cell_size_m'],cell_count=math.prod(p['counts_xyz']),extent_m=[n*p['cell_size_m']for n in p['counts_xyz']],volume_m3=math.prod(p['counts_xyz'])*p['cell_size_m']**3,trajectory=p['entry']['trajectory_id'])for p in parents]
report=dict(schema='rheon-full-ai-chart-audit-v1',audited_utc=datetime.now(timezone.utc).isoformat(),chart_sha256=chart_sha,scope='The 113 original proposals classified Full AI in the frozen chart; does not admit newer fits or change chart data',
    counts=counts,reference_population=dict(cases=128,simulations=64,snapshots_per_simulation=2,frame_indices=[15,63],cells_min=min(r['cell_count']for r in domains),cells_median=statistics.median(r['cell_count']for r in domains),cells_max=max(r['cell_count']for r in domains),grid_axes_min=5,grid_axes_max=16,cells_per_metre=[8,16],cell_size_m=[.125,.0625],density_kg_m3=[1000.0],dt_s=1/60,closed_loop_steps_measured=0,domain_volumes_min_max_m3=[min(r['volume_m3']for r in domains),max(r['volume_m3']for r in domains)],source=canonical_manifest_path),
    contrast=contrasts,contrast_protocol={k:v for k,v in repeated.items()if k!='rows'},historical_isolated_timing_examples=pass_examples,
    records=records,domains=domains,sources=sources,source_checks=source_checks,failures=failures,
    limitations=['No fresh inference, simulation or physical acceptance experiment was run.','Matching per-field manifest hashes checks declared tensor identity, not fresh decompression and re-hashing of every archived NPZ.','Missing historical hardware/software and warmup observations cannot be reconstructed from the current host.','Historical builder hashes differ; the exact old builder bodies are not included in this audit, so current-source execution mode is not asserted as observed for them.','This audit does not establish why single-sweep timing differs from repeated warm timing; warmup, context and run conditions are not independently varied.'])
(OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
fields=['id','name','family','matched_validation128','on_default_frontier','worst_error','main_median_ms','timing_method','recorded_torch','recorded_threads','recorded_affinity','timing_context','fine_network_stages','architecture_dtype','pressure_dtype','accuracy_backend','unique_simulations','frame_indices','cell_count_min','cell_count_max','cell_size_m','density_kg_m3','dt_s','field_hash_metadata_matches_reference','closed_loop_steps_measured','accuracy_source','timing_source','issues']
with(OUT/'cohorts.csv').open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in records:w.writerow({k:json.dumps(r[k])if isinstance(r[k],(list,dict))else r[k]for k in fields})
assert sha(PAGE)==chart_sha,'The audited dashboard changed during the audit'
print(json.dumps(dict(counts=counts,failures=failures,contrast=contrasts,isolated_examples=pass_examples),indent=2))
