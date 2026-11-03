"""Render current retained evidence, keeping pending work explicit."""
from datetime import datetime,timezone
import html
import json
import statistics
from v4_support import ROOT,BASE,DOCS,read,write,sha

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>'for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in row)+'</tr>'for row in rows)+'</tbody></table>'

def build():
    complete=(DOCS/'completion-audit.json').exists() and read(DOCS/'completion-audit.json')['status']=='complete'
    payload=dict(status='complete'if complete else'active',updated_utc=datetime.now(timezone.utc).isoformat(),promotion=False,source_sha256=sha(__file__),evidence={})
    sections=[];fits=[]
    arms=['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control','v4-replay','v4-gcr','v4-diverse','v4-onpolicy']
    def retained(path):
        payload['evidence'][str(path.relative_to(ROOT))]=sha(path);return read(path)
    for arm in arms:
        generation=2 if arm=='v4-student75-distill'else 1
        study=BASE/(arm+f'-study-{generation:02d}');receipt=study/'receipt.json'
        if receipt.exists():
            r=retained(receipt);history=retained(study/'train/history.json');training=retained(study/'train/receipt.json')
            checkpoint=history['checkpoints']['checkpoint-2400.npz']['inventory'];parameters=sum(s['bytes']//4 for s in checkpoint.values())
            fits.append([arm,r['status'],parameters,r['optimizer_updates'],history.get('optimizer_work',{}).get('forward_cells',''),
                training.get('fine_tuning_statistics',{}).get('distillation_teacher_forward_cells',0),all(r['artifact_parity'].values())])
        else:fits.append([arm,'Admitted; outcome pending'if(study/'active.json').exists()else'Prepared; waiting','','','','',''])
    payload['successful_fits']=sum(r[1]=='completed'for r in fits)
    payload['fits']=fits
    sections.append('<h2>Managed fits and fresh CPU restores</h2><p>Students use 75% and 50% of the fine-network channel widths, with the learned global branch retained. Each has a matched non-distillation control. Full-width arms compare replay, GCR-aware training, forcing/shape diversity, and model-generated states. Every fit has 2,400 updates and retains the original 1,600-small-sample stream; actual cell and teacher work are separate. A completed fit is not a claim that its pressure predictions qualify.</p>'+table(['Arm','State','Parameters','Updates','Student optimizer-forward cells','Distillation teacher-forward cells','CPU artifact parity'],fits))
    legacy=[]
    for p in sorted((BASE/'old-evaluation').glob('*.json')):
        d=retained(p)
        for s in d['summary']:
            if s['max_passes']==8:legacy.append([d['entry']['id'],s['population'],f"{s['qualified']}/{s['cases']}",f"{s['worst_relative_residual']:.3g}",f"{s['worst_pressure_relative_l2']:.3g}"])
    if legacy:sections.append('<h2>Retained development screen</h2><p>Consumed historical and v3 native development requests. Both original physical gates apply at eight corrections. These results expose regressions; they are not fresh blind evidence. Cube accuracy does not imply rectangular-domain coverage.</p>'+table(['Model','Population','Gates passed','Worst residual','Worst mean-free pressure rel. L2'],legacy))
    expanded=[]
    for p in sorted((BASE/'evaluation').glob('*.json')):
        d=retained(p);new=next((s for s in d['summary']if s['population']=='v4-development'),None)
        if new is None:continue
        old=[r for r in d['rows']if r['population']not in ['native-development','v4-development']]
        expanded.append([d['entry']['id'],d['policy'],f"{sum(r['qualified']for r in old)}/{len(old)}",
            f"{new['qualified']}/{new['cases']}",new['median_network_calls'],f"{new['worst_pressure_relative_l2']:.3g}"])
    if expanded:sections.append('<h2>Broader v4 development coverage</h2><p>Ten new cold/warm pressure requests from five independently reserved forcing scenes, including odd rectangular grids. This population is separate from the easier eight v3 native requests. All rows use eight-correction ceilings and unchanged physical gates. Large errors on rejected requests remain visible; they do not represent accepted simulation steps.</p>'+table(['Model','Policy','Retained prior gates','New v4 gates','New median CNN calls','New worst pressure rel. L2'],expanded))
    blind=[]
    for p in sorted((BASE/'blind-evaluation').glob('*.json')):
        d=retained(p);group=d['rows']
        blind.append(dict(identifier=p.stem,cases=len(group),qualified=sum(r['qualified']for r in group),
            median_network_calls=statistics.median(r['network_calls']for r in group),worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in group)))
    payload['blind']=blind
    if blind:
        sections.insert(0,'<h2>Fresh blind scenes after frozen selection</h2><p>Five independent forcing scenes produce ten cold/warm requests on 32³, 37×29×19, 64³, 73×49×33 and 128³ grids. Selection was frozen after the complete development matrix and all eight full-128³ deployment-transfer checks, before these scenes were captured. The GCR-trained full model and the frozen multilevel/CNN hybrid each pass all ten; the previous learned policy passes eight and the selected 75%-channel student seven. The hybrid misses one of 156 retained prior requests; the GCR-trained policy retains all 156. The matched replay model also passes all ten when used inside GCR, with the same median four CNN calls as the solver-trained model. Thus the development median-call improvement from solver-aware training did not carry through to this blind median. The spectral label defines the pressure comparison, so its zero relative pressure difference is not a claim of zero equation error.</p>'+table(['Policy','Gates passed','Median CNN calls','Worst pressure rel. L2'],[[r['identifier'],f"{r['qualified']}/{r['cases']}",r['median_network_calls'],f"{r['worst_pressure_relative_l2']:.3g}"]for r in blind]))
    timing_complete=(BASE/'all-timing-batch-02/execution.json').exists() and read(BASE/'all-timing-batch-02/execution.json')['status']=='completed'
    if timing_complete:
        quiet=[retained(p)for p in sorted((BASE/'quiet').glob('*.json'))]
        payload['historical_timing']=[dict(identifier=d['identifier'],parameters=d['parameters'],**d['summary'])for d in quiet]
        sections.append('<h2>Repeated pressure-policy timing on identical historical requests</h2><p>Nineteen policies, 128 identical requests on grids with 5–16 cells per axis, CPU 23 and one thread. These are 128 requests, not 128³ domains. One complete warmup precedes three timed repeats per request. All network, coarse solve, conversion, equation and stopping work counts. Initialization, loading and teacher-label generation are outside policy time. This compares selected matched prior controls, not the complete historical cohort frontier. A point that fails either physical gate is excluded from the qualified frontier.</p>'+table(['Policy','Parameters','Gates passed','Median ms','Worst pressure rel. L2'],[[d['identifier'],d['parameters'],f"{d['summary']['qualified']}/128",f"{d['summary']['median_ms']:.3f}",f"{d['summary']['worst_pressure_relative_l2']:.3g}"]for d in quiet]))
        selection=retained(DOCS/'selection.json');reservation=retained(DOCS/'timing-reservation.json');native=[]
        keys=[r['identifier']for r in selection['selected']]+['v3-native-baseline','v3-native-layout','spectral','numerical-gcr-multilevel']
        for row in reservation['rows']:
            for key in keys:
                episodes=[retained(BASE/'coupled-quiet'/(row['trajectory_id']+'-'+key+f'-r{rep}')/'receipt.json')for rep in range(3)]
                medians=[statistics.median(f['frame_seconds']for f in d['frames'])for d in episodes]
                native.append(dict(counts=row['scene']['counts'],identifier=key,accepted_repeats=sum(d['status']=='completed'for d in episodes),
                    median_seconds=statistics.median(medians),min_seconds=min(medians),max_seconds=max(medians)))
        payload['native_timing']=native
        big={r['identifier']:r for r in native if r['counts']==[128,128,128]}
        previous=big['v3-native-baseline']['median_seconds'];hybrid=big['v3-native-vcycle-neural']['median_seconds'];layout=big['v3-native-layout']['median_seconds']
        payload['matched_128_cube_comparison']=dict(previous_learned_seconds=previous,multilevel_neural_seconds=hybrid,
            multilevel_neural_reduction_fraction=1-hybrid/previous,layout_seconds=layout,layout_reduction_fraction=1-layout/previous,
            spectral_seconds=big['spectral']['median_seconds'],numerical_multilevel_seconds=big['numerical-gcr-multilevel']['median_seconds'])
        sections.append('<h2>What the matched large-grid timing establishes</h2><p>On the reserved 128³ four-step scene, the multilevel/CNN hybrid reduces the previous learned policy from '+f'{previous:.2f} to {hybrid:.2f}'+' seconds per step ('+f'{100*(1-hybrid/previous):.1f}'+'% lower). Channels-last execution reduces that previous policy to '+f'{layout:.2f}'+' seconds ('+f'{100*(1-layout/previous):.1f}'+'% lower). The numerical multilevel/GCR control takes '+f"{big['numerical-gcr-multilevel']['median_seconds']:.2f}"+' seconds and the spectral control '+f"{big['spectral']['median_seconds']:.2f}"+' seconds. The GCR-trained model takes '+f"{big['v4-gcr-gcr-learned']['median_seconds']:.2f}"+' seconds; its broader qualification does not imply a speed gain. These comparisons are scene-specific, headless and single-threaded, with three repeats and all pressure gates retained.</p>')
        sections.append('<h2>Repeated native fluid-step cost</h2><p>Sixty-three serial episodes: three independently reserved blind-scene prefixes, seven methods, three balanced-order repetitions. Each accepted episode advances four steps. Timings include native preparation, pressure exchange, projection and acceptance; label generation, full-state exports, loading and GUI are disabled. Bars and this table use the median of the three per-run frame medians, with the observed run range. A rejected attempt is explicitly marked and cannot be compared as a usable simulation speed.</p>'+table(['Domain cells','Policy','Completed repeats','Median step seconds','Run-median range seconds'],[['×'.join(map(str,r['counts'])),r['identifier'],f"{r['accepted_repeats']}/3",f"{r['median_seconds']:.4f}",f"{r['min_seconds']:.4f}–{r['max_seconds']:.4f}"]for r in native]))
    inference=BASE/'execution-screen-01.json'
    if inference.exists():
        d=retained(inference);rows=[]
        for shape in sorted({tuple(r['counts_xyz'])for r in d['rows']}):
            for name in ['eager','channels_last','traced_frozen']:
                group=[r for r in d['rows']if tuple(r['counts_xyz'])==shape and r['execution']==name]
                rows.append([str(shape),name,len(group),sum(r['qualified']for r in group),f"{1000*statistics.median(r['seconds']for r in group):.2f}",f"{100*statistics.median(r['cnn_seconds']/r['seconds']for r in group):.1f}%"])
        sections.append('<h2>Inference execution screen</h2><p>Three real requests, three balanced repeats each, CPU 23, one thread. Input layout conversion and the whole adaptive pressure policy count; compilation is recorded separately. The nine trials per execution are three distinct requests, not nine independent scenes. This preliminary screen does not establish coupled speed or broad accuracy retention.</p>'+table(['Grid','Execution','Trials','Trials passing gates','Median policy ms','CNN share'],rows))
    for filename,title in [('multilevel-screen-01.json','Genuine variational hierarchy'),('preconditioner-screen-01.json','Frozen GCR preconditioners')]:
        path=BASE/filename
        if not path.exists():continue
        d=retained(path);rows=[]
        for method in sorted({r['method']for r in d['rows']}):
            group=[r for r in d['rows']if r['method']==method]
            rows.append([method,len(group),sum(r['qualified']for r in group),statistics.median(r['network_calls']for r in group),f"{max(r['pressure_relative_l2']for r in group):.3g}"])
        sections.append('<h2>'+title+'</h2><p>Frozen weights, consumed development only. The hierarchy preserves constants and uses exact recursive tensor-product Galerkin operators. GCR does not assume that a neural preconditioner is SPD. Exploratory timings may overlap other campaign work, so this table reports accuracy and work; repeated timing remains separate.</p>'+table(['Method','Requests','Gates passed','Median CNN calls','Worst pressure rel. L2'],rows))
    data=[]
    for name in ['diverse-training-view-01','onpolicy-training-view-01']:
        p=BASE/name/'manifest.json'
        if p.exists():
            d=retained(p);data.append([name,len(d['parents']),d['retained_v3_parents'],d['added_v4_parents'],f"{d['encoded_total_bytes']/1024**2:.2f}",d['exact_full_decoded_population_verified']])
    if data:sections.append('<h2>Separate data expansions</h2><p>The diversified and on-policy views both retain every field of the prior 34-parent bank. Added on-policy labels solve each actual model-generated pressure equation with the spectral teacher. Neither development nor blind scenes enter training. Grid dimensions, cells per metre, fluid density and timestep remain distinct metadata.</p>'+table(['View','Parents','Retained v3','Added v4','Encoded MiB','All decoded field hashes verified'],data))
    drift=[]
    for p in sorted(BASE.glob('drift-development-*-01.json')):
        d=retained(p)
        if not d['rows']:continue
        r=d['rows'][-1];drift.append([d['trajectory'],d['matched_accepted_steps'],r['time_s'],*[f"{r[k]['relative_l2']:.3g}"if r[k]['relative_l2']is not None else'undefined'for k in ['pressure','velocity','tracer']]])
    if drift:sections.append('<h2>Independent trajectory drift</h2><p>Matched initial conditions and forcing; independently advanced spectral and v3-native tracks. Full pressure, velocity and tracer arrays are retained and hash-verified. Rejected steps are excluded from matched times. These are scene-specific diagnostic curves, not application accuracy limits or a universal safe step count.</p>'+table(['Scene','Accepted steps','Simulated seconds','Final pressure rel. L2','Final velocity rel. L2','Final tracer rel. L2'],drift))
    matched=[]
    for label,name in [('v3 baseline','drift-development-64-01.json'),('Replay continuation','drift-v4-replay-64-01.json'),('On-policy continuation','drift-v4-onpolicy-64-01.json')]:
        p=BASE/name
        if p.exists():
            d=retained(p);r=d['rows'][-1]
            matched.append([label,d['matched_accepted_steps'],r['time_s'],*[f"{r[k]['relative_l2']:.4g}"for k in ['pressure','velocity','tracer']]])
    if matched:sections.append('<h2>Matched post-training drift comparison</h2><p>One independent 64³ scene, 120 accepted steps, two simulated seconds, identical starting conditions and forcing. On-policy training reduces the final pressure difference versus matched replay, but velocity remains around 1.1×10⁻⁷ and tracer around 2.7×10⁻⁸. This does not demonstrate a material improvement in transported motion or a longer acceptable application horizon. No end-to-end differentiable fluid rollout was trained.</p>'+table(['Model','Accepted steps','Simulated seconds','Final pressure rel. L2','Final velocity rel. L2','Final tracer rel. L2'],matched))
    early=[]
    for policy in ['learned','spectral']:
        p=BASE/'native-data'/('v4-early256-development-'+policy)/'receipt.json'
        if p.exists():
            d=retained(p);early.append([policy,d['status'],sum(f['advanced']for f in d['frames']),d['callbacks'][0]['work'].get('network_calls',0),f"{d['callbacks'][0]['relative_residual']:.3g}"])
    if early:sections.append('<h2>Bounded 256³ cold-start diagnostic</h2><p>One separately reserved development forcing scene at 32 cells per metre. Passing one pressure request does not qualify general 256³ deployment or long trajectories.</p>'+table(['Policy','Result','Advanced steps','CNN calls','Equation residual'],early))
    geometry=BASE/'geometry-diagnosis-v3-01.json'
    if geometry.exists():
        d=retained(geometry);r=next(r for r in d['rows']if r['counts_xyz']==[73,49,33])
        payload['global_error_diagnosis']=r
        sections.append('<h2>Where further work is most justified</h2><p>The failed 73×49×33 development cold start places '+f"{100*r['pressure_error_energy_fraction_below_axis_mode']['3']:.2f}"+'% of mean-free pressure-error energy in the lowest three modes per axis. This is pressure-error energy, not equation-residual energy. A post-hoc exact low-mode correction improves pressure error but still fails the unchanged gates, so it is a diagnosis rather than a selected solution.</p><ol><li>Reduce global-correction cost: learn scale- and geometry-aware transfer or correction within the validated hierarchy, using the numerical multilevel and spectral methods as controls.</li><li>Train a small fine-detail corrector on residuals produced by that hierarchy. The current students imitate the entire fine branch and did not retain coverage; those failures do not test this narrower role.</li><li>Reuse useful correction directions across consecutive timesteps. The current GCR solve starts a fresh direction space for each pressure request. A bounded recycled space could reduce repeated global work; compare it against the existing pressure warm start, with cold starts and abrupt forcing changes included.</li><li>Optimize repeated operator work, allocations and native/runtime exchange, guided by the complete fluid-step profile and preserving gates.</li><li>Broaden spatial and geometric coverage deliberately, then assess 256³ on multiple scenes. Added forcing frequency in this campaign is temporal; spatial spectra vary through radius, localization and shape. An independently controlled spatial-wavenumber curriculum remains untested.</li><li>Use short differentiable rollouts only if a longer independent drift study identifies a consequential failure. Current on-policy results do not establish a velocity/tracer drift benefit.</li></ol><p>These are ranked hypotheses from this campaign, not measured gains. Learning multigrid transfer and neural preconditioning have independent research precedents: <a href="https://proceedings.mlr.press/v97/greenfeld19a.html">Learning to Optimize Multigrid PDE Solvers</a> and <a href="https://proceedings.mlr.press/v235/lan24a.html">A Neural-Preconditioned Poisson Solver</a>. Reusing solver directions also has a direct pressure-Poisson precedent in <a href="https://arxiv.org/abs/1501.03358">Recycling Krylov subspaces for CFD applications</a>. These published results do not establish a speedup for this implementation. Finer teacher timesteps and sparse pressure anchors remain lower priority: neither automatically supplies a better label for the existing pressure equation.</p>')
    figures=DOCS/'figure-evidence.json'
    if figures.exists():
        retained(figures);visuals=[]
        for name,title in [('historical-frontier','Historical accuracy versus pressure-policy cost'),('native-coverage','Coverage on retained and new populations'),('native-fluid-time','Native fluid-step costs'),('trajectory-drift','Matched trajectory drift')]:
            visuals.append('<figure><img loading="lazy" src="figures/'+name+'.svg" alt="'+title+'"><figcaption>'+title+' · <a href="figures/'+name+'.pdf">PDF</a> · <a href="figures/'+name+'.png">PNG</a></figcaption></figure>')
        sections.insert(0,'<h2>Measured comparisons</h2>'+''.join(visuals))
    boundary='All six requested explorations are complete and audited. Eight CUDA fits (19,200 updates), eight fresh CPU restores, the complete development/blind matrices and final serial timing are retained.'if complete else'The exploration remains active until final serial timing, figure generation and the evidence audit finish.'
    sections.append('<h2>Completion boundary</h2><p>'+boundary+' No model has been promoted. Application-specific drift acceptance, independently swept spatial-wavenumber training, end-to-end differentiable fluid rollouts and production 512³/1024³ domains remain outside this qualification. The bounded 256³ check is one cold request.</p>')
    if complete:
        retained(DOCS/'completion-audit.json');retained(DOCS/'final-audit.json')
        sections.append('<p><a href="completion-audit.json">Completion audit</a> · <a href="final-audit.json">Retained evidence and fresh Run custody audit</a></p>')
    lead='Completed controlled exploration. Solver integration and global correction improve blind coverage; the tested distillation and added-data recipes do not establish a general improvement.'if complete else'Work in progress. Solver integration improves fresh blind coverage; final repeated timing and audit are being completed.'
    body='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Pressure methods v4</title><style>body{font:16px system-ui;max-width:1300px;margin:36px auto;padding:0 24px;color:#18313c;background:#f8fafb}p,li{line-height:1.55;max-width:1050px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0 36px}th,td{padding:9px;border-bottom:1px solid #ccd6dc;text-align:left}th{background:#e4edf2}h1,h2{color:#123e54}figure{margin:22px 0 36px}img{width:100%;height:auto}figcaption{font-size:13px;color:#426171}</style><h1>Pressure methods v4: six-track exploration</h1><p>'+lead+' Updated '+html.escape(payload['updated_utc'])+'.</p>'+''.join(sections)+'<p><a href="results.json">Machine-readable report</a> · <a href="campaign.json">Requested tracks and evidence requirements</a> · <a href="selection-rule.json">Frozen selection rule</a> · <a href="../pressure-methods-v3/results.html">Prior campaign</a></p></html>'
    (DOCS/'results.html').write_text(body);write(DOCS/'results.json',payload)
    print('Rendered current six-track evidence report;',payload['successful_fits'],'completed fits',flush=True)

if __name__=='__main__':build()
