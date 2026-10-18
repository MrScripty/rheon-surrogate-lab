"""Retain small scientific/lifecycle receipts and an offline optimizer viewer."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def report(base, output):
    analysis=read(base/'analysis.json')
    study=base/'adam-replay-study-01'
    receipt=read(study/'receipt.json')
    controller=read(base/'adam-replay-controller-01/execution.json')
    qualification=read(base/'qualification.json')
    history=read(study/'train/history.json')
    original=read(ROOT/'runs/pressure-balanced-shapes-v1/balanced-shapes-study-01/train/history.json')
    original_receipt=read(ROOT/'runs/pressure-balanced-shapes-v1/balanced-shapes-study-01/receipt.json')
    assert controller['status']=='completed' and controller['owned_controller_reaped']
    assert qualification['status']=='passed' and qualification['adam_observer']['status']=='passed'
    assert controller['qualification_sha256']==sha(base/'qualification.json')
    assert read(base/'analysis-execution.json')['reaped']
    work=history['optimizer_work']
    for key in ('network_forwards','forward_cells','scope'):
        assert work[key]==original['optimizer_work'][key]
    assert work['network_forwards']==6400 and work['forward_cells']==7424000
    training=receipt['training_receipt'];restore=receipt['restore_receipt']
    bank=training['fine_tuning_statistics']['fundamental_bank']
    control=original_receipt['training_receipt']['fine_tuning_statistics']['fundamental_bank']
    for key in ('cases','sampling_order_sha256','amplitude_rms_pa','expected_exposures','optimizer_exposures','optimizer_examples','optimizer_updates','aggregation'):
        assert bank[key]==control[key]
    assert bank['optimizer_examples']==1600 and bank['optimizer_updates']==200
    assert training['execution']['parameters_gradients_adam_moments_cuda']
    assert not training['execution']['amp'] and not training['execution']['tf32']
    assert restore['optimizer_updates']==0 and not restore['execution']['cuda_initialized']
    readback=read(study/'readback-profile.json')
    files=[base/'qualification.json',base/'qualification-execution.json',base/'analysis-execution.json',
        base/'analysis.json',base/'adam-replay-controller-01/execution.json',study/'receipt.json',study/'readback-profile.json',
        base/'adam-replay-project-01/assembly.json',*sorted(HERE.glob('*.py')),HERE/'config.json']
    result=dict(analysis, managed=dict(project_id=receipt['project_id'],training_run_id=receipt['training_run_id'],
        restore_run_id=receipt['restore_run_id'],trainer_elapsed_s=training['elapsed_s'],
        training_loop_s=history['training_finished_unix']-history['training_started_unix'],
        observer_elapsed_s=work['observer_elapsed_s'],optimizer_work=work,
        managed_workflow_s=receipt['elapsed_s'],controller_s=controller['elapsed_s'],
        peak_cuda_allocated_bytes=training['execution']['peak_allocated_bytes'],
        peak_host_rss_bytes=training['runtime']['peak_rss_bytes'],runtime=training['runtime'],
        execution=training['execution'],restore_runtime=restore['runtime'],
        artifact_count=len(receipt['downloads']),retained_training_bytes=sum(r['bytes'] for r in receipt['downloads'] if r['run_id']==receipt['training_run_id']),
        reference_trainer_elapsed_s=original_receipt['training_receipt']['elapsed_s'],
        reference_training_loop_s=original['training_finished_unix']-original['training_started_unix'],
        api_call_s=sum(r['elapsed_s'] for r in readback['calls']),
        verified_artifact_reuses=readback['reused_artifacts']),
        source_receipts={str(p.relative_to(ROOT)):sha(p) for p in files},
        standards=dict(snapshot='snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711',unresolved_fact_categories=0),
        limits=['One seed, eight consumed train-only fundamental cases; no untouched holdouts.',
            'Instrumentation timing is not an uninstrumented speed benchmark.',
            'CPU component projections are approximate when gradient precision checks fail.',
            'Local capped direction is not a test of capped Adam training.',
            'No new loss, schedule, deployment or physical threshold change.'])
    if (output/'results.json').exists():
        raise ValueError('Fresh retained result required')
    output.mkdir(parents=True,exist_ok=True)
    write(output/'results.json',result)
    for name in ('qualification.json','qualification-execution.json','analysis-execution.json'):
        shutil.copyfile(base/name,output/'reports'/name)
    shutil.copyfile(base/'adam-replay-controller-01/execution.json',output/'reports/controller.json')
    status='exactly reproduced C69 weights and CPU pressure results' if result['final_weights_exact'] and result['final_quality_exact'] else 'showed a reproducibility discrepancy; see retained comparisons'
    encoded=json.dumps(result,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>C69 Adam observation replay</title>
<style>body{font:16px system-ui;background:#101923;color:#e6edf5;max-width:1120px;margin:2rem auto;padding:0 1rem}a{color:#8ecbff}svg{background:#142330;width:100%;height:330px}table{display:block;overflow-x:auto;max-width:100%;border-collapse:collapse;width:100%;font-size:14px}td,th{padding:.6rem;border-bottom:1px solid #455464;text-align:right}th:first-child,td:first-child{text-align:left}select{padding:.4rem;background:#203245;color:inherit}p{line-height:1.5}.note{color:#bdcbd7}.cards{display:flex;gap:2rem;flex-wrap:wrap}code{color:#a6ddff}</style>
<h1>C69 Adam observation replay</h1><p><a href="../cohort-progress.html">Cohort progress</a> · <a href="plan.md">Method</a> · <a href="results.json">Full scalar records</a> · <a href="execution-ledger.md">Timing and failures</a> · <a href="next-steps.md">Next proposal</a></p>
<p id="summary"></p><p class="note">C53 remains deployed. CUDA float32 training; fresh CPU float32 network restore with float64 physical calculations. Same 200 updates, 1,600 examples, loss, samples/order and learning-rate schedule as C69. Target ≤1e-6 relative equation residual.</p>
<div class="cards" id="times"></div>
<h2>Selected-update evidence</h2><label>Case <select id="case"></select></label> <label>Measurement <select id="metric"><option value="loss">Loss change: measured vs first-order prediction</option><option value="residual">Four-pass relative pressure residual, before vs after</option><option value="cosine">Full gradient vs actual Adam delta</option></select></label>
<p class="note">Negative loss change or gradient–delta cosine indicates local descent. Samples are sparse observations at five updates, not a complete learning curve. Loss is dimensionless; residual is an equation norm ratio. CPU predictions are qualified separately below.</p><svg id="chart" viewBox="0 0 1050 330" role="img" aria-label="Selected optimizer observations"></svg><div id="legend"></div>
<h2>Actual Adam steps and reconstruction</h2><table><thead><tr><th>Update</th><th>LR</th><th>Delta L2</th><th>Unchanged parameters</th><th>CPU Adam max |error|</th><th>CPU/CUDA gradient rel L2</th></tr></thead><tbody id="steps"></tbody></table>
<h2>Selected case</h2><table><thead><tr><th>Update</th><th>Loss before</th><th>Loss after</th><th>Actual change</th><th>Predicted change</th><th>Full gradient · delta cosine</th></tr></thead><tbody id="cases"></tbody></table>
<p class="note" id="limits"></p><script id="data" type="application/json">DATA</script><script>
const d=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id),f=x=>x===null?'n/a':Number(x).toExponential(4),ns='http://www.w3.org/2000/svg';
$('summary').textContent='This replay STATUS.';
$('times').innerHTML=[['CUDA trainer',d.managed.trainer_elapsed_s],['Observer included in trainer',d.managed.observer_elapsed_s],['Managed workflow',d.managed.managed_workflow_s],['CPU analysis',d.elapsed_s]].map(([k,v])=>'<p>'+k+'<br><strong>'+v.toFixed(2)+' s</strong></p>').join('');
$('limits').textContent='CPU Adam reconstruction: '+(d.all_adam_reconstructions_passed?'all passed':'limited')+' (absolute tolerance '+d.adam_absolute_tolerance+'). CPU/CUDA gradient precision: '+(d.all_gradient_precision_checks_passed?'all passed':'one or more checks limited')+' (relative tolerance '+d.gradient_relative_tolerance+'). '+d.limits.join(' ');
$('case').innerHTML=d.observations[0].cases.map(r=>'<option value="'+r.case+'">'+r.case+' · '+r.counts_xyz.join(' × ')+'</option>').join('');$('case').value='7';
$('steps').innerHTML=d.observations.map(r=>'<tr>'+[r.update,f(r.learning_rate),f(r.delta_l2),(100*r.fraction_parameters_unchanged).toFixed(2)+'%',f(r.adam_reconstruction_max_abs),f(r.cpu_mean_vs_captured_cuda_gradient_relative_l2)].map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('');
function render(){const index=Number($('case').value),mode=$('metric').value,rows=d.observations.map(r=>({...r,c:r.cases[index]}));
$('cases').innerHTML=rows.map(r=>'<tr>'+[r.update,f(r.c.gpu_loss_before),f(r.c.gpu_loss_after),f(r.c.actual_loss_change),f(r.c.predicted_first_order_loss_change),f(r.c.full_gradient_vs_delta.cosine)].map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('');
let series=mode==='loss'?[['Measured','#69d3bd',r=>r.c.actual_loss_change],['First-order CPU prediction','#ffca6b',r=>r.c.predicted_first_order_loss_change]]:mode==='residual'?[['Before','#82bdff',r=>r.observed_before.pressure.find(x=>x.case===index&&x.passes===4).relative_equation_residual],['After','#69d3bd',r=>r.observed_after.pressure.find(x=>x.case===index&&x.passes===4).relative_equation_residual]]:[['Cosine','#b8a1ff',r=>r.c.full_gradient_vs_delta.cosine]];
const transform=x=>mode==='residual'?Math.log10(Math.max(x,1e-16)):mode==='loss'?Math.sign(x)*Math.log10(1+Math.abs(x)/1e-6):x;
let values=series.flatMap(s=>rows.map(r=>transform(s[2](r))));if(mode==='residual')values.push(transform(1e-6));let lo=Math.min(...values),hi=Math.max(...values);if(mode!=='residual'){lo=Math.min(lo,0);hi=Math.max(hi,0)}if(lo===hi)hi=lo+1;let margin=(hi-lo)*.08;lo-=margin;hi+=margin;
const svg=$('chart');svg.replaceChildren();function add(tag,attrs,text){let e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}
const x=u=>85+u/200*900,y=v=>280-(transform(v)-lo)/(hi-lo)*240;
for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4,Y=280-i*60;add('line',{x1:85,y1:Y,x2:985,y2:Y,stroke:'#35485c'});let native=mode==='residual'?10**v:mode==='loss'?Math.sign(v)*1e-6*(10**Math.abs(v)-1):v;add('text',{x:78,y:Y+4,fill:'#c8d6e4','text-anchor':'end','font-size':12},f(native))}
if(mode==='residual'){add('line',{x1:85,x2:985,y1:y(1e-6),y2:y(1e-6),stroke:'#ffca6b','stroke-dasharray':'5 4'});add('text',{x:980,y:y(1e-6)-6,fill:'#ffca6b','text-anchor':'end','font-size':12},'target 1e-6')}
series.forEach(([name,color,value])=>{add('polyline',{points:rows.map(r=>x(r.update)+','+y(value(r))).join(' '),fill:'none',stroke:color,'stroke-width':2});rows.forEach(r=>{let e=add('circle',{cx:x(r.update),cy:y(value(r)),r:5,fill:color,tabindex:0,'aria-label':'Update '+r.update+' '+name+' '+f(value(r))});let t=document.createElementNS(ns,'title');t.textContent='Update '+r.update+': '+name+' '+f(value(r));e.append(t)})});
rows.forEach(r=>add('text',{x:x(r.update),y:307,fill:'#ccd9e5','text-anchor':'middle','font-size':12},r.update));$('legend').textContent=series.map(s=>s[0]).join(' · ')+' · update number · '+(mode==='residual'?'log scale':mode==='loss'?'symmetric log scale (1e-6 linear scale)':'linear scale');}
$('case').onchange=render;$('metric').onchange=render;render();
</script></html>'''.replace('DATA',encoded).replace('STATUS',html.escape(status))
    (output/'index.html').write_text(page)
    print('Retained results and offline report',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.base.resolve(),a.output.resolve())
