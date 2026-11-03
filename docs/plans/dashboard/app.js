
'use strict';
const data=JSON.parse(document.getElementById('chart-data').textContent);
const $=id=>document.getElementById(id);
const color={teal:'#60dfc3',amber:'#f4ba71',native:'#c2ced7',grid:'#263744',muted:'#a2b3c0',blue:'#8dafff'};
const names=['Single-grid','Multi-grid · 1 pass','Multi-grid · 2 pass','Cohort · 16 sims'];
const ns='http://www.w3.org/2000/svg';
function el(tag,attrs={},text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
function text(svg,x,y,label,attrs={}){svg.append(el('text',{x,y,fill:color.muted,'font-size':11,...attrs},label))}
function line(svg,x1,y1,x2,y2,stroke=color.grid,dash){svg.append(el('line',{x1,y1,x2,y2,stroke,'stroke-width':1,...(dash?{'stroke-dasharray':dash}:{})}))}
// Stream extrema so growing frequency and long-rollout populations cannot exceed browser argument limits.
function arrayMin(values){let result=Infinity;for(const value of values)result=Math.min(result,value);return result}
function arrayMax(values){let result=-Infinity;for(const value of values)result=Math.max(result,value);return result}
function sci(v){return v.toExponential(2).replace('e-','e−')}
function setup(id,labels,min,max,log,ticks){const svg=$(id);svg.replaceChildren();const p={left:75,right:578,top:42,bottom:269};const x=i=>p.left+30+i*(p.right-p.left-60)/(labels.length-1);const transform=v=>log?Math.log10(v):v;const y=v=>p.bottom-(transform(v)-transform(min))/(transform(max)-transform(min))*(p.bottom-p.top);for(const t of ticks){line(svg,p.left,y(t),p.right,y(t));text(svg,p.left-10,y(t)+4,log?'1e'+Math.round(Math.log10(t)):(t===1?'1×':t.toFixed(2)+'×'),{'text-anchor':'end'})}labels.forEach((label,i)=>{const parts=label.split('|');parts.forEach((part,j)=>text(svg,x(i),293+j*16,part,{'text-anchor':'middle','font-size':11}))});return{svg,x,y,p}}
function target(c,value,label,stroke=color.native){const yy=c.y(value);line(c.svg,c.p.left,yy,c.p.right,yy,stroke,'5 5');text(c.svg,c.p.right,yy-7,label,{'text-anchor':'end',fill:stroke,'font-size':10})}
function showTip(event,label,node){const tip=$('tooltip');tip.textContent=label;tip.hidden=false;const r=node.getBoundingClientRect();const x=event.clientX||r.x+r.width/2;const y=event.clientY||r.y;tip.style.left=Math.max(8,Math.min(x+12,innerWidth-tip.offsetWidth-12))+'px';tip.style.top=Math.max(8,Math.min(y+12,innerHeight-tip.offsetHeight-12))+'px'}
function series(c,values,stroke,labels,{connect=true,offset=0}={}){if(connect){let path='';values.forEach((v,i)=>{if(v===null){path+=' ';return}path+=(i===0||values[i-1]===null?'M':'L')+(c.x(i)+offset)+','+c.y(v)+' '});c.svg.append(el('path',{d:path,fill:'none',stroke,'stroke-width':2.5}))}values.forEach((v,i)=>{if(v===null)return;const dot=el('circle',{cx:c.x(i)+offset,cy:c.y(v),r:5.5,fill:stroke,stroke:'#111e29','stroke-width':2,tabindex:0,class:'point','aria-label':labels[i]});dot.append(el('title',{},labels[i]));dot.addEventListener('pointerenter',e=>showTip(e,labels[i],dot));dot.addEventListener('pointermove',e=>showTip(e,labels[i],dot));dot.addEventListener('focus',e=>showTip(e,labels[i],dot));for(const event of ['pointerleave','blur'])dot.addEventListener(event,()=>$('tooltip').hidden=true);c.svg.append(dot)})}
function render(){const scene=Number($('scene').value),threads=Number($('threads').value),fixed=$('policy').value==='fixed';const timing=data.timings.find(s=>s.threads===threads).aggregate.filter(r=>r.scene===scene);const rows=data.models.map(m=>timing.find(r=>r.model===m.key));const previous=rows[0],latest=rows[3];const speed=rows.map(r=>r.native_median_seconds/r.ai_median_seconds);const gain=previous.ai_median_seconds/latest.ai_median_seconds;const milliseconds=v=>(v*1000).toFixed(2)+' ms';
 $('speed-gain').textContent=gain.toFixed(2)+'× faster';$('speed-caption').textContent=milliseconds(previous.ai_median_seconds)+' → '+milliseconds(latest.ai_median_seconds)+' · '+data.scenes[scene]+' / '+threads+' threads';$('accuracy-gain').textContent=data.fixed.median_relative_residual_improvement.toFixed(2)+'× lower';$('native-speed').textContent=speed[3].toFixed(2)+'× Rheon';$('native-caption').textContent='Model takes '+latest.ai_to_native_ratio.toFixed(2)+'× the paired native time';
 const labels=['Single-grid|1 simulation','Multi-grid 1 pass|1 simulation','Multi-grid 2 pass|1 simulation','Cohort 2 pass|16 simulations'];const c=setup('speed-chart',labels,0,1.14,false,[0,.25,.5,.75,1]);target(c,1,'Rheon baseline · 1×');series(c,speed,color.teal,rows.map((r,i)=>names[i]+': '+speed[i].toFixed(3)+'× Rheon; model '+milliseconds(r.ai_median_seconds)+', paired native '+milliseconds(r.native_median_seconds)));speed.forEach((v,i)=>text(c.svg,c.x(i),c.y(v)-13,v.toFixed(2)+'×',{'text-anchor':'middle',fill:color.teal,'font-size':13,'font-weight':600}));
 $('timing-note').textContent='Three repetitions · '+data.scenes[scene]+' · '+threads+'-thread CPU budget · paired model/native medians';
 const accuracy=data.models.map(m=>fixed?m.fixed:m.adaptive);const a=setup('accuracy-chart',labels,1e-8,fixed?1e-2:1e-5,true,fixed?[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:[1e-8,1e-7,1e-6,1e-5]);target(a,1e-8,'Rheon target ≤1e−8');target(a,1e-6,'Model target ≤1e−6',color.blue);for(const [key,col,offset] of [['median',color.teal,-3],['maximum',color.amber,3]])series(a,accuracy.map(r=>r?r.relative_residual[key]:null),col,accuracy.map((r,i)=>r?names[i]+': '+key+' error '+sci(r.relative_residual[key])+'; '+r.qualified+'/128 physical passes':''),{offset});
 if(fixed){[1,2].forEach(i=>text(a.svg,a.x(i),a.y(1e-4),'No fixed-8 data',{'text-anchor':'middle','font-size':10}));$('accuracy-note').textContent='Equal work: exactly 8 network evaluations. Only the first and last checkpoints were measured. Both still fail some physical checks at this budget.'}else $('accuracy-note').textContent='Adaptive policy: all 128 states pass for all four checkpoints. Evaluation counts vary, so similar final residuals do not imply similar work.';
 $('finding').textContent='On '+data.scenes[scene]+' with '+threads+' CPU threads, the 16-simulation cohort reduces model runtime by '+((1-latest.ai_median_seconds/previous.ai_median_seconds)*100).toFixed(1)+'% versus the previous checkpoint. Rheon remains '+latest.ai_to_native_ratio.toFixed(2)+'× faster. At equal 8-evaluation work, the cohort’s worst equation error improves '+data.fixed.worst_relative_residual_improvement.toFixed(2)+'×.';
 $('table-caption').textContent='Speed: '+data.scenes[scene]+' / '+threads+' threads. Accuracy: '+(fixed?'fixed 8 evaluations':'adaptive, maximum 32 evaluations')+' on the separate 128-state population.';
 $('measurements').innerHTML=rows.map((r,i)=>{const m=data.models[i],acc=accuracy[i];return '<tr class="'+(i===3?'latest':'')+'"><th scope="row">'+m.name+'</th><td>'+m.simulations+' / '+m.states+'</td><td>'+milliseconds(r.ai_median_seconds)+' / '+milliseconds(r.native_median_seconds)+'</td><td>'+speed[i].toFixed(3)+'×</td><td>'+(acc?sci(acc.relative_residual.median):'Not measured')+'</td><td>'+(acc?sci(acc.relative_residual.maximum):'Not measured')+'</td><td>'+(acc?acc.qualified+'/128':'Not measured')+'</td></tr>'}).join('')+'<tr class="native-row"><th scope="row">Rheon native reference</th><td>—</td><td>Paired per checkpoint</td><td>1.000×</td><td colspan="2">≤1e−8 acceptance target; observed values not plotted</td><td>128/128</td></tr>';
}
function renderHistory(){const labels=['One-case fit|1 sim / 1 state','Longer cohort|256 sims / 512 states','Matched small|256 sims / 256 states','Matched large|1,024 sims / 1,024 states'];const c=setup('cohort-chart',labels,1e-6,1,true,[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]);target(c,1e-6,'Model target ≤1e−6');for(const [key,col] of [['median',color.teal],['maximum',color.amber]])series(c,data.early.map(r=>r.relative_residual[key]),col,data.early.map(r=>r.name+': '+key+' residual '+sci(r.relative_residual[key])));$('early-gain').textContent=((1-data.early[3].relative_residual.median/data.early[2].relative_residual.median)*100).toFixed(1)+'%';$('work-bars').innerHTML=data.models.map(m=>'<div class="work-row"><span>'+m.short+'</span><div class="track"><div class="fill" style="width:'+(m.mean_evaluations/15*100)+'%"></div></div><strong>'+m.mean_evaluations.toFixed(2)+'</strong></div>').join('');$('work-summary').textContent='The 16-simulation cohort uses '+((1-data.models[3].mean_evaluations/data.models[0].mean_evaluations)*100).toFixed(2)+'% fewer evaluations on average. Worst-case work falls from '+data.models[0].max_evaluations+' to '+data.models[3].max_evaluations+' evaluations.';$('earlier-table').innerHTML=data.early.map(r=>'<tr><td>'+r.name+'</td><td>'+r.simulations+' / '+r.states+'</td><td>'+r.training_seconds.toFixed(2)+' s</td></tr>').join('')}
for(const id of ['scene','threads','policy'])$(id).addEventListener('change',render);
$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='rheon-cohort-progress-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
function historySpeed(row){
 if(row.speed_baseline)return row.speed?{value:row.speed,note:row.speed_scope+'; baseline '+row.speed_baseline}:null;
 if(row.timing_key){const scene=Number($('history-scene').value),threads=Number($('history-threads').value);const r=data.timings.find(s=>s.threads===threads).aggregate.find(r=>r.scene===scene&&r.model===row.timing_key);return{value:r.native_median_seconds/r.ai_median_seconds,note:data.scenes[scene]+' / '+threads+' threads; model '+(r.ai_median_seconds*1000).toFixed(2)+' ms; native '+(r.native_median_seconds*1000).toFixed(2)+' ms'}}
 const value=$('history-baseline').value==='sgs'?(row.speed_sgs??row.speed):row.speed;
 return value?{value,note:(row.speed_scope||row.policy)+'; baseline '+($('history-baseline').value==='sgs'?'SGS PCG':'Jacobi PCG')}:null;
}
function historyDot(svg,x,y,fill,label){const dot=el('circle',{cx:x,cy:y,r:5,fill,tabindex:0,class:'point','aria-label':label});dot.append(el('title',{},label));dot.addEventListener('pointerenter',e=>showTip(e,label,dot));dot.addEventListener('focus',e=>showTip(e,label,dot));for(const event of ['pointerleave','blur'])dot.addEventListener(event,()=>$('tooltip').hidden=true);svg.append(dot);return dot}
@@FILTERS_JS@@
@@HISTORY_JS@@
@@PRESSURE_JS@@
@@POLICIES_JS@@
@@FREQUENCY_JS@@
@@DOMAINS_JS@@
@@DRIFT_JS@@
// BEGIN PRECISION CHART
@@PRECISION_JS@@
// END PRECISION CHART
@@BUDGETS_JS@@
@@STUDIES_JS@@
$('partial-measurement').addEventListener('change',updatePartialComparison);
$('partial-cohort-clear-emphasis').addEventListener('click',clearPartialCohortEmphasis);
$('partial-cohort-isolate-group').addEventListener('click',()=>{if(partialEmphasizedCohortFamily){$('cohorts-partial-group').value=partialEmphasizedCohortFamily;$('cohorts-partial-group').dispatchEvent(new Event('change',{bubbles:true}))}});
$('partial-cohort-scatter').addEventListener('keydown',e=>{if(e.key==='Escape')clearPartialCohortEmphasis()});
for(const id of ['partial-cohort-filter','partial-cohort-labels'])$(id).addEventListener('change',renderPartialCohortPlots);
$('partial-tradeoff-weight').addEventListener('input',renderPartialCohortPlots);
function setupPartialComparison(){
 const body=$('partial-components');body.replaceChildren();for(const p of solverOriginalPoints('component')){const tr=document.createElement('tr');for(const value of [p.id+' · '+p.name,sci(p.worst_error),p.inference_median_ms.toFixed(3)+' ms']){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=p.source;a.textContent='Network-only evidence';td.append(a);tr.append(td);body.append(tr)}updatePartialComparison();
}

// FULL AI STUDY START
@@FULL_AI_JS@@
// FULL AI STUDY END
setupCohortSelectors();setupEvaluation();setupPartialComparison();setupMethodsStudy();setupMethodsV2();setupMethodsV3();setupMethodsV4();setupMethodsV5();setupMethodsV6();setupFullAIStudy();setupPassSweep();setupPrecision();
render();renderHistory();renderAllHistory();selectLatestStudy();renderCohortPlots();renderCheckpointQualification();selectDriftBuild();setupResolution();setupShapes();setupFrequency();setupTemporalSpectrogram();setupStoppingPolicies();renderTrainingProgress();setupAnalysisTabs();setupFilterMemory();

// Tab changes toggle existing panels, preserving dependent options, model toggles,
// comparison choices, and chart detail state without resetting another analysis.
@@NAVIGATION_JS@@
