"""Reconcile measured campaign evidence into a standalone report; never promote models."""
import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'runs/pressure-methods-v2'
DOCS=ROOT/'docs/plans/pressure-methods-v2'
INITIAL=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape','v2-num4','v2-num6']
REPEATS=['v2-headshape-s25','v2-headshape-s26','v2-num6-s25','v2-num6-s26']
def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def passes(row):return row.get('passes',row.get('policy',{}).get('passes'))
def accuracy(rows):
 return dict(cases=len(rows),qualified=sum(r['qualified'] for r in rows),
  worst_relative_residual=max(r['relative_residual'] for r in rows),
  worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in rows))
def generate(partial=False):
 evidence={};missing=[]
 def load(p,required=True):
  if not p.exists():
   if required:missing.append(str(p.relative_to(ROOT)))
   return None
  evidence[str(p.relative_to(ROOT))]=digest(p);return read(p)
 controllers=['repeat-training-controller-01','repeat-evaluation-controller-01','blind-initial-controller-01','blind-dependent-controller-01','scaling-reconciliation-controller-01','finalist-native-controller-01','finalist-native-long-controller-01','quiet-dependent-controller-03','scaling-warm-start-controller-01']
 custody=[]
 for name in controllers:
  d=load(RUNS/name/'execution.json')
  if not d:continue
  drained=d['status']=='completed' and not d.get('active') and d.get('owned_child_reaped',False)
  custody.append(dict(controller=name,status=d['status'],drained=drained))
  if not drained:missing.append(name+': terminal custody pending')
 training=[];runs=[]
 for arm in INITIAL+REPEATS:
  repeat=arm in REPEATS
  receipt=load(RUNS/(arm+('-study-01' if repeat else '-study-02'))/'receipt.json')
  prior=load(RUNS/('replicate-evaluation-01' if repeat else 'evaluation-01')/arm/'results.json')
  if not receipt or not prior:continue
  if receipt['status']!='completed' or not all(receipt['artifact_parity'].values()):raise ValueError('Training/restore parity failed: '+arm)
  if receipt['training_receipt']['status']!='complete' or receipt['restore_receipt']['status']!='complete':raise ValueError('Incomplete captured training receipt')
  if receipt['model_promotion']:raise ValueError('Unexpected model promotion')
  for path,expected in prior['entry']['identities'].items():
   if digest(ROOT/path)!=expected:raise ValueError('Checkpoint/source identity changed: '+path)
  rows=[r for r in prior['rows'] if passes(r)==8]
  if len(rows)!=156:raise ValueError('Complete development population required')
  populations={p:accuracy([r for r in rows if r['population']==p]) for p in sorted({r['population'] for r in rows})}
  history=load(RUNS/(arm+('-study-01' if repeat else '-study-02'))/'restore/history.json')
  training.append(dict(arm=arm,repeat=repeat,optimizer_updates=receipt['optimizer_updates'],fine_forward_cells=history['optimizer_work']['forward_cells'],network_forwards=history['optimizer_work']['network_forwards'],weights_sha256=receipt['weights_sha256'],**accuracy(rows),populations=populations))
  runs.extend([receipt['training_run_id'],receipt['restore_run_id']])
 if len(runs)!=len(set(runs)):raise ValueError('Managed Run identity reused')
 baseline=load(RUNS/'evaluation-01/C53-extended/results.json')
 if baseline:training.insert(0,dict(arm='C53',repeat=False,optimizer_updates=0,**accuracy([r for r in baseline['rows'] if passes(r)==8])))
 cost_screens=[]
 for state_name in ['small-finetuned','trained-coarse']:
  screened=load(RUNS/'cost-screen-01'/state_name/'results.json')
  confirmed=load(RUNS/'confirmation-01'/state_name/'results.json')
  if screened:
   policies=sorted({json.dumps(r['policy'],sort_keys=True) for r in screened['rows']})
   for key in policies:
    group=[r for r in screened['rows'] if json.dumps(r['policy'],sort_keys=True)==key];policy=group[0]['policy']
    cost_screens.append(dict(state=state_name,policy=policy,kind=policy['kind'],nodes=policy.get('nodes'),placement=policy.get('placement'),adaptive=policy.get('adaptive'),**accuracy(group),median_network_calls=statistics.median(r['network_calls'] for r in group)))
 load(RUNS/'confirmation-physical-01/results.json')
 load(RUNS/'num6-adaptive-confirmation-01/results.json')
 load(DOCS/'application-results.json')
 blind=[]
 for arm in ['C53','v2-small16','v2-headshape','v2-num6']+REPEATS:
  d=load(RUNS/'blind-assessment-01'/arm/'results.json')
  if not d:continue
  if d['blind_scenes_consumed']!=8 or d['optimizer_updates']!=0 or d['measured_cases']!=16:raise ValueError('Blind population/tuning guard failed')
  for n in [4,6,7,8]:
   rows=[r for r in d['rows'] if passes(r)==n]
   blind.append(dict(arm=arm,passes=n,**accuracy(rows)))
 application=[]
 for size in [16,32]:
  for name in ['native','native-loose','spectral','C53','v2-headshape','v2-num6','v2-num6-adaptive','v2-small16-adaptive']:
   group=[]
   for rep in range(3):
    d=load(RUNS/'single-final-quiet-01'/f'n{size}-{name}-rep{rep}'/'results.json')
    if d:group.append(d)
   if len(group)!=3:continue
   complete=all(d['status']=='completed' and d['completed_steps']==d['scene']['steps'] and d['native_process']['reaped'] for d in group)
   frame=[d['post_warmup_median_ms']['frame_seconds'] for d in group if d['post_warmup_median_ms']]
   application.append(dict(size=size,method=name,status='completed' if complete else 'pressure_rejection',
    complete_repetitions=sum(d['status']=='completed' for d in group),repetitions=3,
    median_frame_ms=statistics.median(frame) if frame else None,
    min_frame_ms=min(frame) if frame else None,max_frame_ms=max(frame) if frame else None,
    median_preparation_ms=statistics.median(d['post_warmup_median_ms']['preparation_seconds'] for d in group if d['post_warmup_median_ms']) if frame else None,
    median_pressure_ms=statistics.median(d['post_warmup_median_ms']['pressure_seconds'] for d in group if d['post_warmup_median_ms']) if frame else None,
    median_projection_ms=statistics.median(d['post_warmup_median_ms']['projection_seconds'] for d in group if d['post_warmup_median_ms']) if frame else None,
    median_request_bytes=statistics.median(c['request_bytes'] for d in group for c in d['callbacks']) if any(d['callbacks'] for d in group) else 0,
    median_reply_bytes=statistics.median(c['reply_bytes'] for d in group for c in d['callbacks']) if any(d['callbacks'] for d in group) else 0,
    median_loading_s=statistics.median(d['model_load_seconds'] for d in group),
    median_native_setup_s=statistics.median(d['native_setup']['setup_seconds'] for d in group),
    median_wall_s=statistics.median(d['simulation_wall_seconds'] for d in group),
    peak_worker_mib=max(d['peak_worker_rss_kib'] for d in group)/1024,
    peak_native_mib=max(d['peak_native_rss_kib'] for d in group)/1024,
    median_network_calls=statistics.median(c['network_calls'] for d in group for c in d['callbacks']) if any(d['callbacks'] for d in group) else 0,
    timing_eligible=complete))
 scaling=[]
 for folder in ['scaling-baseline-01','scaling-controls-128-01','scaling-selected-128-01','scaling-256-01','scaling-256-02','scaling-warm-start-01']:
  for p in sorted((RUNS/folder).glob('*/results.json')):
   d=load(p)
   if 'frames' in d:
    f=d['frames'][-1];q=f['quality']
    scaling.append(dict(case=p.parent.name,scope='native cold first step' if d['scene']['warmup_steps']==0 else 'native: one warm-up + one predicted step',status=d['status'],size=d['grid_dimensions'][0],cells_per_metre=d['cells_per_metre'],physical_extent_m=d['physical_extent_m'][0],
     method=d['kind'],qualified=f['advanced'],relative_residual=q['relative_residual'],frame_seconds=f['frame_seconds'],
     pressure_seconds=f['pressure_seconds'],preparation_seconds=f['preparation_seconds'],projection_seconds=f['projection_seconds'],model_loading_seconds=d['model_loading_seconds'],native_setup_seconds=d['native_setup']['setup_seconds'],native_wall_seconds=d['native_wall_seconds'],peak_worker_mib=d['peak_worker_rss_kib']/1024,peak_native_mib=d['peak_native_rss_kib']/1024,
     path=str(p.relative_to(ROOT))))
   else:
    for row in d['rows']:
     scaling.append(dict(case=p.parent.name,scope='manufactured equation',size=d['grid_dimensions'][0],cells_per_metre=d['cells_per_metre'],
      physical_extent_m=d['physical_extent_m'][0],method=row['policy'],qualified=row['qualified'],relative_residual=row['relative_residual'],
      pressure_relative_l2=row.get('mean_free_pressure_relative_l2'),seconds=row.get('seconds',row.get('warm_seconds')),
      peak_worker_mib=d['peak_worker_rss_kib']/1024,path=str(p.relative_to(ROOT))))
 native_accuracy=load(DOCS/'native-scaling-accuracy.json')
 if native_accuracy:
  by_case={row['case']:row for row in native_accuracy['rows']}
  for row in scaling:
   if row['scope'].startswith('native') and row['case'] in by_case:
    row.update({k:v for k,v in by_case[row['case']].items() if k.startswith('spectral_')})
    row['pressure_relative_l2']=row['spectral_mean_free_pressure_relative_l2']
 warm_plan=load(DOCS/'warm-start-followup.json')
 if warm_plan:
  for row in warm_plan['cases']:load(ROOT/row['result'])
 scaling_manifest=load(RUNS/'scaling-reschedule-manifest.json')
 if scaling_manifest:
  for row in scaling_manifest['cases']:load(ROOT/row['result'])
 drift=[]
 for folder in ['harder-drift-03','harder-drift-long-01','cost-native-01','finalist-native-01','finalist-native-long-01']:
  for p in sorted((RUNS/folder).glob('*/results.json')):
   d=load(p)
   if not d['native_process']['reaped'] or d['native_process']['returncode']!=0:raise ValueError('Rollout native custody unresolved')
   points=d['points'];metrics=['pressure_trajectory_relative_l2','velocity_trajectory_relative_l2','tracer_trajectory_relative_l2']
   drift.append(dict(case=p.parent.name,folder=folder,status=d['status'],steps=d['scene']['steps'],last_time_s=points[-1]['time_s'] if points else None,
    pressure_max=max((r[metrics[0]] for r in points),default=None),velocity_max=max((r[metrics[1]] for r in points),default=None),
    tracer_max=max((r[metrics[2]] for r in points),default=None),network_calls=d['network_calls'],external_numerical_corrections=d['numerical_corrections'],
    first_crossings=d['first_crossings'],absolute_first_crossings=d['absolute_first_crossings'],path=str(p.relative_to(ROOT))))
 for arm in ['v2-headshape','v2-num6']:
  load(RUNS/'finalist-native-long-01'/f'{arm}-scene0-7'/'results.json')
 chart=[]
 def points(d,label,ai,hybrid=False):
  rows=d['rows']
  groups=sorted({str(r.get('policy',{})) for r in rows}) if hybrid else sorted({passes(r) for r in rows})
  for group in groups:
   selected=[r for r in rows if (str(r.get('policy',{})) if hybrid else passes(r))==group]
   policy=selected[0].get('policy',{})
   name=({'spectral':'Full spectral','pcg':'Numerical two-level PCG6','hybrid':label+' adaptive'}[policy['kind']] if hybrid else label+f' {group} passes')
   isai=ai and (not hybrid or policy['kind']=='hybrid')
   for pop in ['all-development']+sorted({r['population'] for r in selected}):
    subset=selected if pop=='all-development' else [r for r in selected if r['population']==pop]
    chart.append(dict(label=name,population=pop,ai=isai,**accuracy(subset),median_ms=1000*statistics.median(r['median_policy_seconds'] for r in subset)))
 for arm in ['C53','v2-small16','v2-headshape','v2-num6']:
  d=load(RUNS/'quiet-assessment-01'/arm/'results.json')
  if d:
   if d['concurrency_scope']!='owned-workers-quiet' or d['repetitions']!=3:raise ValueError('Quiet repeated timing contract failed')
   points(d,'C53 extended' if arm=='C53' else arm,True)
 d=load(RUNS/'quiet-confirmation-01/results.json')
 if d:
  if d['concurrency_scope']!='owned-workers-quiet' or d['repetitions']!=3:raise ValueError('Quiet numerical timing contract failed')
  points(d,'v2-num6',True,True)
 small_adaptive=load(RUNS/'quiet-small-adaptive-01/results.json')
 if small_adaptive:
  if small_adaptive['concurrency_scope']!='owned-workers-quiet' or small_adaptive['repetitions']!=3:raise ValueError('Quiet frozen hybrid timing failed')
  points(small_adaptive,'v2-small16',True,True)
 selection=load(DOCS/'selection.json');unlock=load(DOCS/'blind-unlock.json')
 if unlock and unlock['selection_sha256']!=digest(DOCS/'selection.json'):raise ValueError('Frozen selection changed')
 sampling=load(DOCS/'sampling-audit.json')
 if sampling and sampling['status']!='passed':raise ValueError('Sampling audit failed')
 repeat_sampling=load(DOCS/'repeat-sampling-audit.json')
 if repeat_sampling and repeat_sampling['status']!='passed':raise ValueError('Repeat sampling audit failed')
 managed=load(DOCS/'managed-run-audit.json')
 if managed and (managed['status']!='passed' or managed['successful_runs']!=24):raise ValueError('Live managed Run custody audit failed')
 load(DOCS/'drift-acceptance.json')
 load(DOCS/'native-challenge-limit.json')
 load(DOCS/'scaling-reservation.json')
 native_binding=load(RUNS/'harder-drift-build-02/binding.json')
 if native_binding:
  if native_binding['original_selected_rheon_modified']:raise ValueError('Original Rheon was modified')
  for path,expected in native_binding['original_sources'].items():
   if digest(Path(path))!=expected:raise ValueError('Original Rheon source identity changed: '+path)
 for build,binary in [('single-build-01','rheon-single-track'),('harder-drift-build-02','rheon-drift-rollout'),('large-native-build-01','rheon-large-single-track')]:
  receipt=load(RUNS/build/'build-receipt.json')
  if receipt:
   if receipt['status']!='qualified_build' or digest(RUNS/build/binary)!=receipt['binary_sha256']:raise ValueError('Qualified native build identity changed')
   for path,expected in receipt.get('sources',{}).items():
    if digest(ROOT/path)!=expected:raise ValueError('Bound native build source changed')
 # Retain exact material identities for the original model cohorts.
 material=load(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json');baseline_identities=[]
 for entry in material['models']:
  if entry['id'] not in ['C37','C44','C53']:continue
  for path,expected in entry['identities'].items():
   if digest(ROOT/path)!=expected:raise ValueError('Original baseline artifact changed: '+path)
  baseline_identities.append(dict(id=entry['id'],verified_files=len(entry['identities'])))
 missing=sorted(set(missing))
 result=dict(status='incomplete' if missing else 'completed',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  missing=missing,direction_coverage=dict(application='48 final serial runs plus 48 initial bounded single-track runs; setup, transport, pressure, projection, memory and frame costs retained',shape_retention='Eight matched initial fits plus four sampling repeats, actual neural-cell work audit and whole-scene blind evaluation',hybrid_cost='640 initial screen points plus 156-request confirmations, quiet numerical controls and adaptive trained/frozen hybrids',drift='31 comparative rollouts with pressure, velocity, tracer, relative/absolute diagnostic crossings and retained native CFL failures',global_architecture='Matched learned gains, shape-informed learned branch and numerical coarse controls; fixed-initialization sampling repeats',scaling='64/128/256 manufactured and native probes on fixed-spacing/fixed-extent tracks, plus 12 matched warm-start diagnostics; failures retained'),training=training,cost_screens=cost_screens,successful_fits=len(runs)//2,managed_run_ids=runs,blind=blind,application=application,scaling=scaling,drift=drift,chart=chart,
  custody=custody,original_baselines=baseline_identities,execution_adjustments=dict(scaling='One unfinished CPU-only probe was interrupted and drained after four completed cases; ten unfinished cases were explicitly rescheduled across two cores. The reconciliation receipt retains all fourteen reserved measurements.',audit='An idle read-only audit was replaced with the qualified control interpreter before API reconciliation.',quiet_schedulers='Idle dependent controllers were superseded to keep all added diagnostics ahead of final serial timing.'),application_acceptance_claimed=False,promoted=False,
  failed_launch=dict(run_id='b8c8c019dafe483b80615aca3ddc04f7',phase='pre-optimizer wall-deadline validation',status='failed_drained'),
  evidence=evidence,source_sha256=digest(Path(__file__)))
 if missing and not partial:raise ValueError('Campaign incomplete: '+', '.join(missing[:12]))
 write(DOCS/'results.json',result)
 render(result)
 print(json.dumps(dict(status=result['status'],successful_fits=result['successful_fits'],missing=len(missing),blind_methods=len(blind)//4,application_groups=len(application),scaling_rows=len(scaling),drift_episodes=len(drift))))
 return result

def render(d):
 def table(rows,fields):
  def fmt(v):
   if isinstance(v,float):return f'{v:.6g}'
   return str(v) if v is not None else '—'
  return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(name)+'</th>' for _,name in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(key)))+'</td>' for key,_ in fields)+'</tr>' for r in rows)+'</tbody></table></div>'
 sections=[]
 sections.append('<h2>Training and eight-pass development coverage</h2><p>The small16/small24 controls continue training on small domains; mixed16/retain24 expand coverage with fixed updates or retained small-case exposure. Headbase/headshape add learned global corrections, with shape features in headshape. Num4/num6 use four/six numerical coarse nodes per axis before the fine network. S25/S26 change paired sampling order while preserving initialization. Prepared-request qualification means relative equation residual ≤ 10⁻⁶ or absolute residual norm ≤ 10⁻¹². Native frame acceptance separately enforces the unchanged host physical gates.</p>'+table(d['training'],[('arm','Model'),('optimizer_updates','Updates'),('fine_forward_cells','Fine cell visits'),('qualified','Qualified / 156'),('worst_relative_residual','Worst equation residual'),('worst_pressure_relative_l2','Worst pressure relative L2')]))
 sections.append('<h2>Coarse correction cost screen</h2><p>Two fixed-weight states × 20 policies × 16 requests = 640 measured screen points. Coarse grids use 2/3/4/6 nodes per axis, once or before every fine pass, with fixed or residual-based stopping. Promising policies were then confirmed on all 156 development requests; the final adaptive policy additionally enforces predicted divergence. Screen timings were contended; the plotted final comparison uses the quiet measurements.</p><details><summary>Open all coarse screen outcomes</summary>'+table(d['cost_screens'],[('state','Fine state'),('kind','Policy'),('nodes','Coarse nodes / axis'),('placement','Placement'),('adaptive','Adaptive'),('qualified','Qualified / 16'),('worst_relative_residual','Worst equation residual'),('median_network_calls','Median NN calls')])+'</details>')
 sections.append('<h2>Blind whole scenes</h2><p>Eight reserved scenes, two native snapshots each. Selection was frozen before capture. No checkpoint was tuned on these results.</p>'+table([r for r in d['blind'] if r['passes']==8],[('arm','Model'),('qualified','Qualified / 16'),('worst_relative_residual','Worst equation residual'),('worst_pressure_relative_l2','Worst pressure relative L2')]))
 sections.append('<h2>Actual single-track frame cost</h2><p>Three serial repetitions on the same CPU and scenes. All IPC, preparation, projection and physical gates are counted. Rejected prefixes are ineligible for speed claims. Loading and memory are separate. Native-loose uses CG tolerance 10⁻⁶ for its warm-up and measured steps; other policies use native 10⁻⁸ during warm-up. Host acceptance gates are unchanged.</p>'+table(d['application'],[('size','Cells / side'),('method','Policy'),('status','Status'),('median_frame_ms','Frame median ms'),('median_preparation_ms','Prepare ms'),('median_pressure_ms','Pressure + transfer ms'),('median_projection_ms','Project ms'),('min_frame_ms','Repeat min ms'),('max_frame_ms','Repeat max ms'),('median_loading_s','Loading s'),('peak_worker_mib','Python peak MiB'),('peak_native_mib','Native peak MiB'),('median_network_calls','NN calls / request')]))
 sections.append('<h2>Scaling: equation tests versus native first step</h2><p>Fixed cells/metre holds 16 and increases physical extent. Fixed physical extent holds four metres and changes cells/metre from 16 to 32 to 64. Native pressure errors use matched spectral references; manufactured errors use independent analytic labels where recorded. These scaling timings were measured during campaign work; they are exploratory. Only the frozen original-seed finalists are used for this scaling survey; sampling repeats are evaluated on development and blind populations. A single accepted step is not a rollout qualification. Manufactured spacing variants share the same dimensionless equation and are not independent random tests.</p>'+table(d['scaling'],[('case','Case'),('scope','Scope'),('cells_per_metre','Cells / m'),('physical_extent_m','Extent m'),('qualified','Qualified'),('relative_residual','Equation residual'),('pressure_relative_l2','Pressure relative L2'),('frame_seconds','Native frame s'),('peak_worker_mib','Worker peak MiB'),('peak_native_mib','Native peak MiB')]))
 sections.append('<h2>Long rollouts: diagnostic drift</h2><p>Pressure uses cell-zero-gauged L2; velocity includes staggered faces; tracer measures cell concentration. Application acceptance limits are deferred by user instruction. The strongest rectangular forcing exceeded the native CFL limit at step 70; a reduced-force version exceeded it at step 167. Those native failures were retained, and the qualified lower-force scene is explicitly versioned. External numerical counts exclude numerical coarse stages captured inside a trained model; v2-num6 executes one such stage per neural pass.</p>'+table([r for r in d['drift'] if r['steps']>=7200],[('case','Policy'),('status','Status'),('last_time_s','Last simulated s'),('pressure_max','Max pressure drift'),('velocity_max','Max velocity drift'),('tracer_max','Max tracer drift'),('network_calls','NN calls'),('external_numerical_corrections','External numerical calls')]))
 sections.append('<h2>Suggested next series</h2><ol><li>Profile total frame cost with spectral pressure, then target the largest remaining preparation kernel for acceleration.</li><li>Add native initial-projection and warm-start residuals at 64³/128³ while preserving small-domain exposure. Keep cold and warm whole-scene holdouts separate, and diagnose the 256³ gap before larger captures.</li><li>Compare the winning coarse corrections with smaller C44/C37 fine networks, beginning with frozen weights and residual stopping. Evaluate complete application cost alongside the neural-policy frontier.</li></ol><p>Retain the learned shape branch for its 128³ warm-start coverage and the numerical hybrid for its lower small-case inference cost. Keep drift thresholds diagnostic until application acceptance limits are chosen.</p>')
 sections.append('<h2>Scientific figure exports</h2><p><a href="policy-frontiers.svg">Policy frontiers (SVG)</a> · <a href="drift-120s.svg">120-second drift curves (SVG)</a>. PNG exports are available beside the SVG files.</p><h2>Interpretation and limits</h2><p>The comparisons isolate data retention, global learned structure and numerical coarse correction. Extra numerical stages are timed and are not FLOP-matched to the fine network. The repeated seeds change sampling order while preserving initialization. Larger-grid native failures remain measured negatives even where manufactured equations pass.</p><p>The full spectral solver exploits sealed domains with uniform cubic cells and constant fluid mass density. Cells per metre describes spatial sampling density and is separate from fluid mass density. Obstacles, free surfaces and variable coefficients require a different operator contract. Native CG labels have finite error, so apparent pressure improvements near that error floor require the independent equation residual for interpretation.</p><p>No model was promoted. Original C37, C44 and C53 artifact identities are verified. Every successful fit has a separate fresh CPU restore with artifact parity. One rejected launch failed deadline validation before any optimizer update.</p>')
 findings=[]
 if d['successful_fits']==12:findings.append('All twelve fits and fresh CPU restores completed. Both frozen families qualify all 156 development requests and all 16 blind requests at eight passes across three paired sampling orders.')
 warm=[r for r in d['scaling'] if r['method']=='learned-coarse' and r['scope']=='native: one warm-up + one predicted step' and r['size']==128]
 if len(warm)==2 and all(r['qualified'] for r in warm):findings.append('The shape-informed learned branch passes the 128³ native step after one native warm-up on both spacing tracks. Cold-start and warm-start results are separate; the cold 256³ neural probes fail at eight passes.')
 if len(d['application'])==16:
  for size in [16,32]:
   valid=[r for r in d['application'] if r['size']==size and r['timing_eligible']]
   best=min(valid,key=lambda r:r['median_frame_ms'])
   findings.append(f"At {size}³, the fastest completed application policy is {best['method']}: {best['median_frame_ms']:.3f} ms median frame time across three serial repetitions.")
 summary_html='<div class="findings">'+''.join('<p>'+html.escape(x)+'</p>' for x in findings)+'</div>'
 state='Complete measured campaign' if d['status']=='completed' else 'Campaign in progress'
 payload=json.dumps(d['chart'],allow_nan=False).replace('<','\\u003c')
 head='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pressure methods — second campaign</title><style>
 :root{font-family:system-ui,sans-serif;color:#e8edf5;background:#111722}body{margin:0 auto;max-width:1240px;padding:30px}h1{font-size:32px;margin-bottom:8px}h2{margin-top:40px;font-size:22px}p{line-height:1.55;color:#bec9da;max-width:1000px}.badge{display:inline-block;padding:6px 12px;border:1px solid #607693;border-radius:16px}select{background:#202d40;color:#fff;border:1px solid #697d98;padding:8px;margin:4px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #33415a;white-space:nowrap}th{color:#99d8ff}a{color:#83c9ff}svg{width:100%;height:auto;background:#162131;border-radius:8px}#details{min-height:28px}label{display:inline-block;margin:6px}small{color:#abb7c8}</style>
 <h1>Pressure methods: second campaign</h1>'''
 chart='''<h2>Worst error versus prepared pressure-policy time</h2><p>Each point uses the same selected request population: median warm policy time across requests and worst error. Three warm repetitions follow the first call. Model loading, IPC and fluid simulation are excluded here and included separately in the frame table. Lines include only policies meeting the equation residual criterion on every displayed request. C53 uses its qualified extended-shape wrapper here; candidate policies use their exact captured recurrences. The 16³/32³ application comparison uses the captured C53 path. Original historical chart timings use a different scope.</p><label>Population <select id="population"></select></label><label>Error <select id="metric"><option value="worst_pressure_relative_l2">Mean-free pressure relative L2</option><option value="worst_relative_residual">Equation residual</option></select></label><label>Frontier <select id="family"><option value="all">All prepared policies</option><option value="ai">Policies containing a neural stage</option></select></label><svg id="plot" viewBox="0 0 1100 530" role="img" aria-label="Worst error versus pressure policy time"></svg><p><small>C: C53 extended; S: small control; H: learned shape branch; N: trained numerical6; number: passes. NA/SA: adaptive numerical6 with trained/small-control weights; MG6: numerical PCG6; SP: spectral.</small></p><p id="details">Hover over a point to inspect its qualification and error.</p><small>Pressure errors near the finite native reference-solver error floor should be interpreted alongside equation residuals. Incomplete results produce no points until their timing is available.</small>'''
 js='''<script>
 const data=DATA; const pop=document.querySelector('#population');
 [...new Set(data.map(d=>d.population))].sort().forEach(p=>{const o=document.createElement('option');o.textContent=p;o.value=p;pop.append(o)});pop.value='consumed128';
 const ns='http://www.w3.org/2000/svg';const plot=document.querySelector('#plot');
 function el(name,attrs,text){const e=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;plot.append(e);return e}
 function shortLabel(label){if(label==='Full spectral')return 'SP';if(label==='Numerical two-level PCG6')return 'MG6';if(label==='v2-num6 adaptive')return 'NA';if(label==='v2-small16 adaptive')return 'SA';const p=label.startsWith('C53')?'C':label.startsWith('v2-small16')?'S':label.startsWith('v2-headshape')?'H':'N';return p+label.split(' ').at(-2)}
 function draw(){plot.replaceChildren();const metric=document.querySelector('#metric').value;const ai=document.querySelector('#family').value==='ai';const rows=data.filter(d=>d.population===pop.value&&(!ai||d.ai)&&d.median_ms>0&&d[metric]>0);if(!rows.length){el('text',{x:30,y:50,fill:'white'},'Timing pending');return}
 const lx=rows.map(d=>Math.log10(d.median_ms)),ly=rows.map(d=>Math.log10(d[metric]));let x0=Math.floor(Math.min(...lx)),x1=Math.ceil(Math.max(...lx));let y0=Math.floor(Math.min(...ly)),y1=Math.ceil(Math.max(...ly));if(x0===x1)x1++;if(y0===y1)y1++;const X=v=>90+930*(Math.log10(v)-x0)/(x1-x0),Y=v=>450-400*(Math.log10(v)-y0)/(y1-y0);
 for(let i=x0;i<=x1;i++){const x=X(10**i);el('line',{x1:x,x2:x,y1:45,y2:450,stroke:'#30435c'});el('text',{x:x,y:475,fill:'#bec9da','text-anchor':'middle'},(10**i).toPrecision(2))}
 for(let i=y0;i<=y1;i++){const y=Y(10**i);el('line',{x1:90,x2:1020,y1:y,y2:y,stroke:'#30435c'});el('text',{x:80,y:y+5,fill:'#bec9da','text-anchor':'end'},'1e'+i)}
 const eligible=rows.filter(d=>d.qualified===d.cases);const frontier=eligible.filter(d=>!eligible.some(o=>o!==d&&o.median_ms<=d.median_ms&&o[metric]<=d[metric]&&(o.median_ms<d.median_ms||o[metric]<d[metric]))).sort((a,b)=>a.median_ms-b.median_ms);
 if(frontier.length>1)el('polyline',{points:frontier.map(d=>X(d.median_ms)+','+Y(d[metric])).join(' '),stroke:'#7ae7b7','stroke-width':2,fill:'none'});
 rows.forEach((d,i)=>{const color=d.qualified<d.cases?'#f58c85':d.ai?'#8bcaff':'#7ae7b7';const c=el('circle',{cx:X(d.median_ms),cy:Y(d[metric]),r:6,fill:color,stroke:'#111722','stroke-width':2});el('text',{x:X(d.median_ms)+9,y:Y(d[metric])+(i%2?14:-8),fill:color,'font-size':11},shortLabel(d.label));c.addEventListener('mouseenter',()=>document.querySelector('#details').textContent=`${d.label}: ${d.median_ms.toFixed(3)} ms; worst error ${d[metric].toExponential(4)}; ${d.qualified}/${d.cases} meet the equation criterion.`)});
 el('text',{x:555,y:518,fill:'#e8edf5','text-anchor':'middle'},'Median prepared policy time (ms, log scale; lower is faster)');el('text',{x:90,y:26,fill:'#e8edf5'},'Worst error (log scale; lower is better)');}
 document.querySelectorAll('select').forEach(s=>s.addEventListener('change',draw));draw();
 </script></html>'''.replace('DATA',payload)
 content=head+'<span class="badge">'+state+'</span><p>'+str(d['successful_fits'])+' completed fits; '+str(len(d['managed_run_ids']))+' successful managed Runs including CPU restores. No promotion. '+html.escape(d['generated_utc'])+'</p>'+summary_html+chart+''.join(sections)+'<p><a href="results.json">Full results, diagnostic crossings, and SHA-256 evidence ledger</a> · <a href="final-audit.json">Final custody and integrity audit</a></p>'+js
 (DOCS/'results.html').write_text(content)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true');args=parser.parse_args();generate(args.partial)
