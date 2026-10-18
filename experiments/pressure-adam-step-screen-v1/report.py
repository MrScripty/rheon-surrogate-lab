"""Retain strict guards, empirical value calibration and an offline step viewer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def summarize(config_path,execution_path,directory):
    if (directory/'results.json').exists():raise ValueError('Fresh summary required')
    config=read(config_path);execution=read(execution_path)
    assert execution['status']=='completed' and execution['all_owned_children_reaped']
    assert execution['maximum_active_workers']<=config['workers']
    assert execution['config_sha256']==sha(config_path)
    qualification=read(directory/'reports/qualification.json')
    assert qualification['status']=='passed' and qualification['scalar_adam_expected']!=0
    snapshots=[];sources=[config_path,execution_path,HERE/'report.py']
    for update in config['updates']:
        path=directory/'reports'/('update-%03d.json'%update);measurement=read(path);sources.append(path)
        assert measurement['status']=='completed' and measurement['update']==update
        assert not measurement['cuda_initialized'] and measurement['optimizer_updates']==0 and measurement['teacher_forwards']==0
        assert measurement['adam_reconstruction_max_absolute_error']<=config['adam_parameter_absolute_tolerance']
        if update==1:assert measurement['initialized_reset_control_exact']
        for name,identity in measurement['sources'].items():assert sha(ROOT/name)==identity,name
        def metrics(policy,fraction):
            candidate=next(r for r in measurement['candidates'] if r['policy']==policy and r['fraction']==fraction)
            return measurement['evaluations'][candidate['parameter_sha256']]['metrics']
        def pressures(m):return {(r['case'],r['passes']):r['relative_equation_residual'] for r in m['pressure']}
        baseline=metrics('observed',0);after=metrics('observed',1)
        bp=pressures(baseline);ap=pressures(after)
        gb,ga=measurement['cuda_observed_before'],measurement['cuda_observed_after']
        gbp,gap=pressures(gb),pressures(ga)
        loss_gaps=[max(abs(baseline['losses'][i]-gb['losses'][i]),abs(after['losses'][i]-ga['losses'][i])) for i in range(8)]
        pressure_gaps={key:max(abs(bp[key]-gbp[key]),abs(ap[key]-gap[key])) for key in bp}
        weak=config['weak_case'];target=config['target_relative_residual'];rows=[]
        for candidate in measurement['candidates']:
            m=measurement['evaluations'][candidate['parameter_sha256']]['metrics'];p=pressures(m)
            weak_loss_improvement=baseline['losses'][weak]-m['losses'][weak]
            weak_pressure_improvement=bp[weak,4]-p[weak,4]
            short=[dict(case=i,passes=n,before=bp[i,n],after=p[i,n]) for i in range(8) for n in (1,2) if p[i,n]>bp[i,n]]
            lost=[dict(case=i,passes=n,before=bp[i,n],after=p[i,n]) for i in range(8) for n in (4,8) if bp[i,n]<=target<p[i,n]]
            robust=(weak_loss_improvement>config['calibration_multiplier']*loss_gaps[weak]
                and weak_pressure_improvement>config['calibration_multiplier']*pressure_gaps[weak,4])
            worst=max(p[i,4] for i in range(8));worst_before=max(bp[i,4] for i in range(8))
            rows.append(dict(**candidate,weak_loss=m['losses'][weak],weak_four_pass_residual=p[weak,4],
                weak_loss_improvement=weak_loss_improvement,weak_pressure_improvement=weak_pressure_improvement,
                mean_loss=sum(m['losses'])/8,worst_four_pass_residual=worst,
                four_pass_qualified=sum(p[i,4]<=target for i in range(8)),eight_pass_qualified=sum(p[i,8]<=target for i in range(8)),
                short_pass_regressions=short,lost_previously_fitted=lost,
                weak_improvement_exceeds_empirical_gap=robust,
                guard_passed=candidate['fraction']>0 and robust and not short and not lost and worst<worst_before))
        zero={r['parameter_sha256'] for r in measurement['candidates'] if r['fraction']==0};assert len(zero)==1
        assert len(rows)==15
        snapshots.append(dict(update=update,learning_rate=measurement['learning_rate'],cases=measurement['cases'],rows=rows,
            calibration=dict(loss_absolute_gaps=loss_gaps,pressure_absolute_gaps=[dict(case=i,passes=n,gap=v) for (i,n),v in sorted(pressure_gaps.items())],
                multiplier=config['calibration_multiplier'],scope='Empirical pre/post value gaps only; not a bound on unmeasured states or a relaxed physical acceptance target.'),
            baseline=baseline,observed_after=after,measurements=measurement['evaluations'],
            unique_states=len(measurement['evaluations']),elapsed_s=measurement['elapsed_s'],evaluation_s=measurement['evaluation_s'],
            peak_rss_bytes=measurement['peak_rss_bytes'],cpu=measurement['cpu'],
            adam_reconstruction_max_absolute_error=measurement['adam_reconstruction_max_absolute_error']))
    passing=[dict(update=s['update'],policy=r['policy'],fraction=r['fraction']) for s in snapshots for r in s['rows'] if r['guard_passed']]
    decision=dict(status='cpu_screen_completed',model_promoted=False,baseline_retained='C53',new_cuda_training_started=False,
        guard_passing_candidates=passing,matched_training_policy_qualified=False,
        reason='One-step CPU improvements do not qualify a CUDA training trajectory. Inspect strict guard failures and snapshot coverage before choosing one policy.')
    result=dict(schema='rheon-adam-step-screen-results-v1',status='completed',config=config,snapshots=snapshots,decision=decision,
        execution=execution,qualification=qualification,optimizer_updates=0,cuda_initialized=False,
        unique_evaluations=sum(s['unique_states'] for s in snapshots),candidate_aliases=sum(len(s['rows']) for s in snapshots),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        limits=['Five consumed training snapshots; no simulated rollout or retrained history.',
            'Same captured CUDA gradient used for every Adam endpoint; no CPU derivative substitution.',
            'Fractions interpolate rounded float32 parameter endpoints; not exact LR changes.',
            'CPU/GPU value gaps are empirical cautions, not rigorous numerical bounds.',
            'Strict short-pass and previously-fitted guards are never relaxed by the empirical gap.'])
    write(directory/'results.json',result)
    write(directory/'decision.json',decision)
    print('Guard-passing candidates:',passing,flush=True)
    for snapshot in snapshots:
        best=min((r for r in snapshot['rows'] if r['fraction']>0),key=lambda r:r['weak_four_pass_residual'])
        print(snapshot['update'],'best weak',best['policy'],best['fraction'],best['weak_four_pass_residual'],
            'short regressions',len(best['short_pass_regressions']),'lost fitted',len(best['lost_previously_fitted']),flush=True)
    render(result,directory)


def render(result,directory):
    data=json.dumps(result,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>C69 CPU Adam step screen</title>
<style>body{font:16px system-ui;background:#101923;color:#e3edf5;max-width:1150px;margin:2rem auto;padding:0 1rem}a{color:#94cfff}p{line-height:1.5}.note{color:#bccbd9}select{background:#203345;color:inherit;padding:.4rem}svg{width:100%;height:350px;background:#152534}table{display:block;overflow-x:auto;border-collapse:collapse;max-width:100%;font-size:14px}td,th{padding:.55rem;border-bottom:1px solid #405366;text-align:right}td:first-child,th:first-child{text-align:left}.yes{color:#7be0be}.no{color:#ffd18a}</style>
<h1>C69 CPU Adam state and step-size screen</h1><p><a href="../cohort-progress.html">Cohort progress</a> · <a href="../pressure-adam-replay-v1/index.html">Captured replay</a> · <a href="plan.md">Method</a> · <a href="results.json">Full records</a> · <a href="execution-ledger.md">Results and timing</a> · <a href="next-steps.md">Next proposal</a></p>
<p id="summary"></p><p class="note">C53 remains deployed. No CUDA training or pressure-network optimizer update. Each proposal is an independent one-step counterfactual using the saved CUDA gradient. All network parameters and endpoints are float32; physical calculations use the captured float64 owner.</p>
<label>Snapshot <select id="update"></select></label> <label>Case <select id="case"></select></label> <label>Measurement <select id="metric"><option value="pressure">Four-pass relative equation residual</option><option value="loss">Full captured loss</option></select></label>
<label>Scale <select id="scale"><option value="relative">Change from before (%)</option><option value="absolute">Absolute logarithmic</option></select></label>
<p class="note">Fractions interpolate each rounded endpoint: 0 is the saved pre-state, 1 is the full proposed step. Lines connect sampled fractions only. Lower values are better. Observed and preserved/reset curves may overlap exactly.</p><svg id="chart" viewBox="0 0 1050 350" role="img" aria-label="CPU pressure or loss versus step fraction"></svg><div id="legend"></div>
<h2>Weak-case result and all-case guards</h2><table><thead><tr><th>Endpoint</th><th>Fraction</th><th>Delta L2</th><th>Weak residual · 4</th><th>Worst residual · 4</th><th>Passes · 4/8</th><th>Short regressions</th><th>Lost fits</th><th>Guard</th></tr></thead><tbody id="rows"></tbody></table>
<h2>Runtime and value calibration</h2><table><thead><tr><th>Update</th><th>CPU</th><th>Worker seconds</th><th>Unique states</th><th>Peak MiB</th><th>CPU Adam max |error|</th><th>Max CPU/GPU loss gap</th></tr></thead><tbody id="runtime"></tbody></table><p class="note" id="limits"></p>
<script id="data" type="application/json">DATA</script><script>
const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),f=x=>Number(x).toExponential(4),policies=[['observed','#82bdff'],['preserved','#f3c66b'],['reset','#75d6b7']],ns='http://www.w3.org/2000/svg';
$('summary').textContent=d.candidate_aliases+' policy/fraction entries, '+d.unique_evaluations+' distinct parameter states, all eight cases. CPU pool '+d.execution.elapsed_s.toFixed(2)+' s, peak concurrency '+d.execution.maximum_active_workers+'. '+d.decision.guard_passing_candidates.length+' entries pass all predeclared guards.';
$('update').innerHTML=d.snapshots.map(s=>'<option>'+s.update+'</option>').join('');$('update').value='100';$('case').innerHTML=d.snapshots[0].cases.map((c,i)=>'<option value="'+i+'">'+i+' · '+c.counts_xyz.join(' × ')+'</option>').join('');$('case').value='7';
$('runtime').innerHTML=d.snapshots.map(s=>'<tr>'+[s.update,s.cpu,s.elapsed_s.toFixed(3),s.unique_states,(s.peak_rss_bytes/1048576).toFixed(1),f(s.adam_reconstruction_max_absolute_error),f(Math.max(...s.calibration.loss_absolute_gaps))].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');
$('limits').textContent=d.limits.join(' ')+' A candidate needs weak loss and four-pass residual gains exceeding twice its observed CPU/GPU pre/post gaps, no case worse at 1/2 passes, no previously qualifying 4/8-pass case lost, and lower worst four-pass residual. These are strict before-state guards; a local pass does not qualify a training policy.';
function render(){const s=d.snapshots.find(s=>s.update===Number($('update').value)),index=Number($('case').value),metric=$('metric').value,relative=$('scale').value==='relative';
const value=r=>{const m=s.measurements[r.parameter_sha256].metrics;return metric==='loss'?m.losses[index]:m.pressure.find(p=>p.case===index&&p.passes===4).relative_equation_residual};
$('rows').innerHTML=s.rows.map(r=>'<tr>'+[r.policy,r.fraction,f(r.delta_l2),f(r.weak_four_pass_residual),f(r.worst_four_pass_residual),r.four_pass_qualified+'/8 · '+r.eight_pass_qualified+'/8',r.short_pass_regressions.length,r.lost_previously_fitted.length,r.guard_passed?'pass':'fail'].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');
const baseline=value(s.rows.find(r=>r.policy==='observed'&&r.fraction===0)),transform=v=>relative?100*(v/Math.max(baseline,1e-30)-1):Math.log10(Math.max(v,1e-16));let vals=s.rows.map(value);if(!relative&&metric==='pressure')vals.push(1e-6);let lo=Math.min(...vals.map(transform)),hi=Math.max(...vals.map(transform));if(lo===hi)hi=lo+1;const pad=(hi-lo)*.08;lo-=pad;hi+=pad;const x=v=>90+v*885,y=v=>292-(transform(v)-lo)/(hi-lo)*240;
const svg=$('chart');svg.replaceChildren();function add(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}
for(let i=0;i<5;i++){const Y=292-i*60;add('line',{x1:90,x2:975,y1:Y,y2:Y,stroke:'#354b61'});add('text',{x:83,y:Y+4,fill:'#c8d8e8','text-anchor':'end','font-size':12},relative?(lo+(hi-lo)*i/4).toFixed(2)+'%':f(10**(lo+(hi-lo)*i/4)))}
if(relative){add('line',{x1:90,x2:975,y1:y(baseline),y2:y(baseline),stroke:'#ed9db4','stroke-dasharray':'5 4'});add('text',{x:970,y:y(baseline)-6,fill:'#ed9db4','text-anchor':'end','font-size':12},'before · 0%')}else if(metric==='pressure'){add('line',{x1:90,x2:975,y1:y(1e-6),y2:y(1e-6),stroke:'#ed9db4','stroke-dasharray':'5 4'});add('text',{x:970,y:y(1e-6)-6,fill:'#ed9db4','text-anchor':'end','font-size':12},'target 1e-6')}
policies.forEach(([policy,color])=>{const rows=s.rows.filter(r=>r.policy===policy);add('polyline',{points:rows.map(r=>x(r.fraction)+','+y(value(r))).join(' '),fill:'none',stroke:color,'stroke-width':2});rows.forEach(r=>{const label=policy+' fraction '+r.fraction+': '+f(value(r)),e=add('circle',{cx:x(r.fraction),cy:y(value(r)),r:5,fill:color,tabindex:0,'aria-label':label});const t=document.createElementNS(ns,'title');t.textContent=label;e.append(t)})});d.config.fractions.forEach(v=>add('text',{x:x(v),y:322,fill:'#d8e4ef','text-anchor':'middle','font-size':12},v));$('legend').innerHTML=policies.map(([p,c])=>'<span style="color:'+c+'">● '+p+'</span>').join(' · ')+' · fraction of rounded endpoint · '+(relative?'change from before (%)':'absolute logarithmic scale');}
['update','case','metric','scale'].forEach(id=>$(id).onchange=render);render();
</script></html>'''.replace('DATA',data)
    (directory/'index.html').write_text(page)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','execution','directory'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.config.resolve(),a.execution.resolve(),a.directory.resolve())
