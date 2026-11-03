const passVisibleCounts=new Set(data.pass_sweep?.passes||[1,2,4,8,12,16,24,32]);
const passColors=['#60dfc3','#8dafff','#f4ba71','#e4a5ee','#f28c95','#b8da75','#7dd7ee','#d1b9a3'];
function passColor(n){return n===6?'#9cce88':n===7?'#ef94ba':passColors[[1,2,4,8,12,16,24,32].indexOf(n)%passColors.length]||color.teal}
function passModelColor(id){return `hsl(${(Number(id.slice(1))*137.508)%360} 72% 70%)`}
function passModelPool(){const scope=$('pass-scope').value;return(data.pass_sweep?.models||[]).filter(m=>cohortMatchesSelection(driftCohorts.get(m.weights_sha256),'passsweep')&&(scope==='all'||(scope==='frontier'?frontierWeights().has(m.weights_sha256):m.tags.includes(scope))))}
function passChosenModels(){const selected=$('pass-model').value;return passModelPool().filter(m=>selected==='all'||m.id===selected)}
function setupPassSweep(){
 const previous=$('pass-model').value,pool=passModelPool();$('pass-model').replaceChildren(new Option('All models in this group','all'));
 for(const model of pool)$('pass-model').append(new Option(model.id+' · '+model.name,model.id));
 $('pass-model').value=pool.some(m=>m.id===previous)||previous==='all'?previous:pool.find(m=>m.id==='C44')?.id||pool[0]?.id||'all';
 const host=$('pass-count-options');host.replaceChildren();
 for(const n of data.pass_sweep?.passes||[]){const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('i');input.type='checkbox';input.checked=passVisibleCounts.has(n);input.setAttribute('aria-label','Show '+n+' neural passes');input.dataset.passCount=n;input.addEventListener('change',()=>{input.checked?passVisibleCounts.add(n):passVisibleCounts.delete(n);renderPassSweep()});swatch.style.borderColor=passColor(n);label.append(input,swatch,document.createTextNode(n+(n===1?' pass':' passes')));host.append(label)}
 renderPassSweep();
}
function passBounds(values,log){
 const finite=values.filter(v=>Number.isFinite(v)&&(!log||v>0));if(!finite.length)return log?[-9,0]:[0,1];
 let lo=log?Math.floor(Math.log10(arrayMin(finite))):0,hi=log?Math.ceil(Math.log10(arrayMax(finite))):arrayMax(finite)*1.08;
 if(lo===hi)hi=lo+1;return[lo,hi];
}
function passAxes(svg,xBounds,yBounds,xlog,ylog,xLabel){
 const plot={left:95,right:1125,top:30,bottom:418};
 const project=(v,bounds,log,start,end)=>start+((log?Math.log10(v):v)-bounds[0])/(bounds[1]-bounds[0])*(end-start);
 const x=v=>project(v,xBounds,xlog,plot.left,plot.right),y=v=>project(v,yBounds,ylog,plot.bottom,plot.top);
 const ticks=(bounds,log)=>log?Array.from({length:Math.ceil((bounds[1]-bounds[0])/Math.max(1,Math.ceil((bounds[1]-bounds[0])/8)))+1},(_,i)=>10**Math.min(bounds[1],bounds[0]+i*Math.max(1,Math.ceil((bounds[1]-bounds[0])/8)))):Array.from({length:6},(_,i)=>bounds[0]+(bounds[1]-bounds[0])*i/5);
 for(const value of ticks(xBounds,xlog)){line(svg,x(value),plot.top,x(value),plot.bottom);text(svg,x(value),445,xlog?(value>=.01&&value<1e4?Number(value.toPrecision(3)):sci(value)):Math.round(value),{'text-anchor':'middle'})}
 for(const value of ticks(yBounds,ylog)){line(svg,plot.left,y(value),plot.right,y(value));text(svg,plot.left-10,y(value)+4,sci(value),{'text-anchor':'end'})}
 text(svg,(plot.left+plot.right)/2,480,xLabel,{'text-anchor':'middle','font-size':12});return{...plot,x,y};
}
function passHighlight(model,passes){
 for(const node of document.querySelectorAll('#pass-drift path[data-pass-drift]')){const match=!model||node.dataset.passDrift===model&&Number(node.dataset.passes)===passes;node.setAttribute('opacity',match?1:.13);node.setAttribute('stroke-width',match&&model?3.2:1.8)}
 for(const node of document.querySelectorAll('#pass-drift [data-pass-rejection],#pass-drift [data-pass-drift-end]')){const match=!model||(node.dataset.passRejection||node.dataset.passDriftEnd)===model&&Number(node.dataset.passes)===passes;node.setAttribute('opacity',match?1:.13)}
 for(const node of document.querySelectorAll('#pass-scatter circle[data-pass-model]')){const match=!model||node.dataset.passModel===model&&Number(node.dataset.passes)===passes;node.setAttribute('opacity',match?1:.3)}
}
function passPointDetail(point){
 const study=data.pass_sweep,trace=study.trajectories.find(t=>t.id===point.model&&t.passes===point.passes),outcome=!trace?'not recorded on this setup':trace.completed?`${trace.recorded_ai_steps} steps completed`:`${trace.recorded_ai_steps} advanced; rejected at AI step ${trace.rejected[0].step-study.scene.warmup_steps}`;
 return `${point.model} · ${point.passes} neural passes · worst equation residual ${sci(point.worst_error)} · median ${sci(point.median_error)} · ${point.inference_median_ms.toFixed(3)} ms on 128 common states · equation checks ${point.equation_passes}/128 · mean actual calls ${point.mean_actual_calls.toFixed(2)} · independent scene: ${outcome}`;
}
function renderPassSweep(){
 const study=data.pass_sweep,models=passChosenModels(),ids=new Set(models.map(m=>m.id));
 const points=(study?.points||[]).filter(p=>ids.has(p.model)&&passVisibleCounts.has(p.passes));
 const traces=(study?.trajectories||[]).filter(t=>ids.has(t.id)&&passVisibleCounts.has(t.passes));
 $('pass-coverage').textContent=study?`${study.status==='completed'?'Completed':'In progress'} · ${study.models.length} unique selected checkpoints · ${study.points.length} measured snapshot settings · ${study.trajectories.length} recorded trajectory settings · ${models.length} model(s) and ${passVisibleCounts.size} pass count(s) selected.`:'This inference-pass campaign has no retained measurements yet.';
 $('pass-method').textContent=study?study.timing_scope:'';
 renderPassScatter(points,models);renderPassDrift(traces,models);
 const body=$('pass-values');body.replaceChildren();
 for(const point of points){const trace=traces.find(t=>t.id===point.model&&t.passes===point.passes),row=document.createElement('tr');
  for(const value of [point.model+' · '+point.passes+' passes',sci(point.worst_error),point.inference_median_ms.toFixed(3)+' ms',point.equation_passes+'/128',trace?trace.completed?(study.scene.steps-study.scene.warmup_steps)+' AI steps completed':trace.recorded_ai_steps+' advanced; stopped at '+(trace.rejected[0]?.step??'?'):'Not recorded on this setup']){const cell=document.createElement('td');cell.textContent=value;row.append(cell)}
  const cell=document.createElement('td');for(const[label,href]of[['Snapshots',point.source],['Timing',point.timing_source],['Trajectory',trace?.source]])if(href){const a=document.createElement('a');a.textContent=label;a.href=href;cell.append(a,' ')}row.append(cell);body.append(row)
 }
}
function renderPassScatter(points,models){
 const svg=$('pass-scatter'),key=$('pass-model-key');svg.replaceChildren();key.replaceChildren();
 const metric=$('pass-error').value,valid=points.filter(p=>p.inference_median_ms>0&&p[metric]>0);
 $('pass-scatter-title').textContent=(metric==='worst_error'?'Worst':'Median')+' pressure equation error versus CPU inference time';
 if(!valid.length){text(svg,580,230,'No measured settings match these filters.',{'text-anchor':'middle'});return}
 const axes=passAxes(svg,passBounds(valid.map(p=>p.inference_median_ms),true),passBounds(valid.map(p=>p[metric]),true),true,true,'Median CPU inference milliseconds per fixed-pass correction');
 const frontier=valid.filter(p=>!valid.some(q=>q[metric]<=p[metric]&&q.inference_median_ms<=p.inference_median_ms&&(q[metric]<p[metric]||q.inference_median_ms<p.inference_median_ms))).sort((a,b)=>a.inference_median_ms-b.inference_median_ms);
 svg.append(el('path',{d:frontier.map((p,i)=>(i?'L':'M')+axes.x(p.inference_median_ms)+','+axes.y(p[metric])).join(' '),fill:'none',stroke:'#f2f5f7','stroke-width':2,'stroke-dasharray':'5 5','data-pass-frontier':'true'}));
 for(const model of models){const rows=valid.filter(p=>p.model===model.id).sort((a,b)=>a.passes-b.passes);if(!rows.length)continue;
  const color=passModelColor(model.id);svg.append(el('path',{d:rows.map((p,i)=>(i?'L':'M')+axes.x(p.inference_median_ms)+','+axes.y(p[metric])).join(' '),fill:'none',stroke:color,'stroke-width':1.8,'data-pass-model':model.id}));
  const label=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=color;label.append(swatch,model.id+' · '+model.name);key.append(label);
  for(const point of rows){const x=axes.x(point.inference_median_ms),y=axes.y(point[metric]),detail=passPointDetail(point),dot=historyDot(svg,x,y,color,detail);dot.dataset.passModel=point.model;dot.dataset.passes=point.passes;dot.addEventListener('focus',()=>{$('pass-detail').textContent=detail;passHighlight(point.model,point.passes)});dot.addEventListener('pointerenter',()=>{$('pass-detail').textContent=detail;passHighlight(point.model,point.passes)});dot.addEventListener('blur',()=>passHighlight(null,null));dot.addEventListener('pointerleave',()=>passHighlight(null,null));text(svg,x+8,y-6,String(point.passes),{fill:color,'font-size':10})}
 }
}
function renderPassDrift(traces,models){
 const svg=$('pass-drift'),key=$('pass-drift-key');svg.replaceChildren();key.replaceChildren();
 const study=data.pass_sweep,metric=$('pass-drift-metric').value,log=$('pass-drift-scale').value==='log',column={relative:2,absolute:3,reference:4}[metric];
 $('pass-drift-note').textContent=`X: AI simulation step after ${study?.scene.warmup_steps||15} native warm-up steps. Y: ${metric==='relative'?'relative pressure-field error':metric==='reference'?'reference-pressure RMS magnitude in Pa':'absolute pressure-field RMS error in Pa (bounded estimate)'}. Colors identify neural pass counts. Each model/pass setting evolves independently.`;
 const total=study?study.scene.steps-study.scene.warmup_steps:497;
 const values=traces.flatMap(t=>t.points.map(p=>p[column])).filter(v=>Number.isFinite(v)&&(!log||v>0));
 const axes=passAxes(svg,[1,total],passBounds(values,log),false,log,'AI simulation step (native warm-up excluded)');
 const rejected=traces.filter(t=>!t.completed);
 $('pass-rejections').textContent=rejected.length?rejected.map(t=>`${t.id}/${t.passes} passes: ${t.recorded_ai_steps} advanced; rejected at AI step ${(t.rejected[0]?.step??15)-study.scene.warmup_steps}`).join(' · ')+'. Rejection ticks mark an attempted step without assigning it an unmeasured pressure error.':traces.length?'All displayed recorded trajectories completed their planned windows.':'No trajectories were recorded on this setup for the selected models/pass counts.';
 if(!traces.length){text(svg,580,220,'No trajectory measurements match these filters.',{'text-anchor':'middle'});return}
 for(const trace of traces){const color=passColor(trace.passes),describe=p=>`${trace.id} · ${trace.passes} neural passes · AI step ${p[0]-study.scene.warmup_steps}/${total} · ${p[1].toFixed(3)} s · relative pressure error ${sci(p[2])} · absolute RMS ${Number.isFinite(p[3])?sci(p[3])+' Pa':'unavailable'} · reference RMS ${Number.isFinite(p[4])?sci(p[4])+' Pa':'unavailable'}`;
  let path='',connected=false;const valid=[];
  for(const point of trace.points){const value=point[column];if(!Number.isFinite(value)||(log&&value<=0)){connected=false;continue}valid.push(point);path+=(connected?'L':'M')+axes.x(point[0]-study.scene.warmup_steps)+','+axes.y(value)+' ';connected=true}
  if(path){svg.append(el('path',{d:path,fill:'none',stroke:color,'stroke-width':1.8,opacity:.85,'data-pass-drift':trace.id,'data-passes':trace.passes}));
   const hit=el('path',{d:path,fill:'none',stroke:'transparent','stroke-width':10,'pointer-events':'stroke'});
   hit.addEventListener('pointermove',event=>{const pos=svg.createSVGPoint();pos.x=event.clientX;pos.y=event.clientY;const local=pos.matrixTransform(svg.getScreenCTM().inverse()),step=1+(local.x-axes.left)/(axes.right-axes.left)*(total-1);let nearest=valid[0];for(const p of valid)if(Math.abs(p[0]-study.scene.warmup_steps-step)<Math.abs(nearest[0]-study.scene.warmup_steps-step))nearest=p;const detail=describe(nearest);$('pass-detail').textContent=detail;showTip(event,detail,hit)});hit.addEventListener('pointerleave',()=>$('tooltip').hidden=true);svg.append(hit);
   const last=valid.at(-1),dot=historyDot(svg,axes.x(last[0]-study.scene.warmup_steps),axes.y(last[column]),color,describe(last));dot.dataset.passDriftEnd=trace.id;dot.dataset.passes=trace.passes;dot.addEventListener('focus',()=>$('pass-detail').textContent=describe(last))}
  if(trace.rejected.length){const step=trace.rejected[0].step-study.scene.warmup_steps,x=axes.x(step);line(svg,x,axes.bottom+2,x,axes.bottom+14,color);const tick=el('circle',{cx:x,cy:axes.bottom+9,r:4,fill:color,'data-pass-rejection':trace.id,'data-passes':trace.passes});tick.append(el('title',{},`${trace.id} · ${trace.passes} passes: rejected at AI step ${step}; no accepted frame for that step`));svg.append(tick)}
  const label=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=color;label.append(swatch,trace.id+' · '+trace.passes+' passes · '+(trace.completed?'complete':trace.recorded_ai_steps+' advanced / stopped'));key.append(label)
 }
 if(!values.length)text(svg,580,210,'No accepted drift samples at these settings. Rejected attempts are marked below the axis.',{'text-anchor':'middle'});
}
$('pass-scope').addEventListener('change',setupPassSweep);
for(const id of ['pass-model','pass-error','pass-drift-metric','pass-drift-scale'])$(id).addEventListener('change',renderPassSweep);

