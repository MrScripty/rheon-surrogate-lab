const driftHidden=new Set();
const driftCohorts=new Map(data.cohort_points.map(p=>[p.weights_sha256,p]));
function driftMedian(values){const a=[...values].sort((a,b)=>a-b),m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2}
function driftStats(trace,window=Number($('drift-baseline').value)){
 const baseline=driftMedian(trace.points.slice(0,window).map(p=>p[2]));
 return{baseline,baselineSteps:Math.min(window,trace.points.length),final:trace.points.at(-1)[2],finalGrowth:baseline>0?trace.points.at(-1)[2]/baseline:null,worstGrowth:baseline>0?Math.max(...trace.points.map(p=>p[2]))/baseline:null};
}
function driftScopeAllowed(){
 const scope=$('drift-scope').value,frontier=frontierWeights();
 return(data.drift_traces||[]).filter(t=>cohortMatchesScope(driftCohorts.get(t.weights_sha256),scope,'drift'));
}
function driftAllowed(){const build=$('drift-build').value;return driftScopeAllowed().filter(t=>build==='all'||t.native_binary_sha256===build)}
function driftBuildGroups(){
 const groups=new Map();for(const t of data.drift_traces||[]){const build=t.native_binary_sha256;if(!groups.has(build))groups.set(build,[]);groups.get(build).push(t)}
 return [...groups].map(([build,traces])=>({build,traces,weights:new Set(traces.map(t=>t.weights_sha256)),latest:Math.max(...traces.map(t=>driftCohorts.get(t.weights_sha256)?.order||0))})).sort((a,b)=>b.latest-a.latest||a.build.localeCompare(b.build));
}
function driftCohortIDs(weights){
 const orders=[...weights].map(w=>Number(driftCohorts.get(w)?.id.slice(1))).filter(Number.isFinite).sort((a,b)=>a-b),ranges=[];
 for(let i=0;i<orders.length;i++){const first=orders[i];let last=first;while(orders[i+1]===last+1)last=orders[++i];const id=n=>'C'+String(n).padStart(2,'0');ranges.push(id(first)+(last!==first?'–'+id(last):''))}return ranges.join(', ');
}
function driftSceneKey(t){return $('drift-build').value==='all'?driftSceneIdentity(t.scene):t.scene_key}
function driftDisplayLabel(t){return driftLabel(t)+($('drift-build').value==='all'?' · build '+t.native_binary_sha256.slice(0,8):'')}
function driftSceneIdentity(scene){return JSON.stringify(Object.keys(scene).sort().map(key=>[key,scene[key]]))}
function selectDriftBuild(){
 const previous=$('drift-build').value,allowed=driftScopeAllowed(),groups=driftBuildGroups(),select=$('drift-build');select.replaceChildren();
 for(const [i,g] of groups.entries()){const count=new Set(allowed.filter(t=>t.native_binary_sha256===g.build).map(t=>t.weights_sha256)).size,option=new Option((i===0?'Newest recordings':'Earlier recordings')+' · '+g.build.slice(0,8)+' · '+count+'/'+g.weights.size+' cohorts in scope',g.build);option.disabled=count===0;select.append(option)}
 select.append(new Option('All builds together · '+new Set(allowed.map(t=>t.weights_sha256)).size+' cohorts in scope','all'));
 const enabled=[...select.options].filter(o=>!o.disabled);select.value=enabled.some(o=>o.value===previous)?previous:enabled.find(o=>o.value!=='all')?.value||'all';selectDriftScene();
}
function driftLabel(t){const c=driftCohorts.get(t.weights_sha256);return(c?c.id+' · ':'')+(c?.name||t.model)+(t.policy_label&&t.policy_label!==c?.name?' · '+t.policy_label:'')}
function driftPolicy(t){const p=t.policy;return(p.adaptive_stopping?'Adaptive, up to ':'Fixed ')+(p.learned_proposals_per_pressure_solve??'?')+' proposals; '+(p.proposal_network_evaluations??'?')+' network evaluation(s) each'+(p.minimum_network_evaluations?' · minimum '+p.minimum_network_evaluations+' evaluations':'')+(p.adaptive_stopping?.mean_free_pressure_relative_error?' · pressure-aware stop':'')+(p.reuse_capacity!==undefined?' · reuse '+p.reuse_capacity+' directions':'')+(p.method?' · '+p.method+(p.numerical_interval?' every '+p.numerical_interval+' AI steps':''):'')}
function driftColor(t){const c=driftCohorts.get(t.weights_sha256);return `hsl(${((c?.order||parseInt(t.weights_sha256.slice(0,4),16))*137.508)%360} 72% 70%)`}
function selectDriftScene(){

 const prior=$('drift-scene').value,priorScene=(data.drift_traces||[]).find(t=>t.scene_key===prior||driftSceneIdentity(t.scene)===prior)?.scene,groups=new Map();
 for(const t of driftAllowed()){const key=driftSceneKey(t);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(t)}
 const longCube=ts=>ts[0].scene.steps>=256&&new Set(ts[0].scene.counts).size===1;
 const sorted=[...groups].sort((a,b)=>Number(longCube(b[1]))-Number(longCube(a[1]))||b[1][0].scene.steps-a[1][0].scene.steps||b[1].length-a[1].length||a[0].localeCompare(b[0]));
 $('drift-scene').replaceChildren();
 for(const[key,traces]of sorted){const s=traces[0].scene,option=document.createElement('option');option.value=key;option.textContent=`${s.counts.join(' × ')} · ${s.steps-s.warmup_steps} AI steps · ${traces.length} runs${$('drift-build').value==='all'?' · '+new Set(traces.map(t=>t.native_binary_sha256)).size+' builds':''} · setup ${traces[0].scene_key.slice(0,5)}`;$('drift-scene').append(option)}
 if(groups.has(prior))$('drift-scene').value=prior;else if(priorScene){const equivalent=sorted.find(([,ts])=>driftSceneIdentity(ts[0].scene)===driftSceneIdentity(priorScene));if(equivalent)$('drift-scene').value=equivalent[0]}
 driftHidden.clear();renderDrift();
}
function driftMagnitudeDescription(trace,index){
 const row=trace.pressure_magnitudes?.[index];
 if(!row)return ' · absolute/reference pressure unavailable in this recording';
 if(['independent-full-field-v4','independent-full-field-v5','independent-full-field-v6'].includes(trace.recording_format))return ' · exact recorded absolute RMS error '+sci(row[0])+' Pa; spectral reference RMS '+sci(row[1])+' Pa; spatial pressure mean removed';
 const error=trace.points[index][2];
 return ` · estimated absolute RMS error ${sci(row[0])} Pa [${(error*row[2]).toExponential(8)}, ${(error*row[3]).toExponential(8)}] · estimated reference RMS ${sci(row[1])} Pa [${row[2].toExponential(8)}, ${row[3].toExponential(8)}]`;
}
function renderDriftMagnitudes(traces,buildDash,describe){
 const count=traces.reduce((n,t)=>n+t.points.length,0);
 const available=traces.reduce((n,t)=>n+(t.pressure_magnitudes||[]).filter(Boolean).length,0);
 const widths=traces.flatMap(t=>(t.pressure_magnitudes||[]).filter(Boolean).map(row=>(row[3]-row[2])/row[1]));
 const maxWidth=widths.length?arrayMax(widths):null;
 const exact=traces.filter(t=>['independent-full-field-v4','independent-full-field-v5','independent-full-field-v6'].includes(t.recording_format)).reduce((n,t)=>n+(t.pressure_magnitudes||[]).filter(Boolean).length,0);
 $('drift-magnitude-coverage').textContent=`Pa measurements: ${available.toLocaleString()}/${count.toLocaleString()} visible steps; ${exact.toLocaleString()} exact mean-free full-field norms, ${(available-exact).toLocaleString()} bounded estimates. `+(available>exact?`Shading shows estimate bounds; largest reference-bound width is ${(maxWidth*100).toPrecision(3)}% of the estimate. `:'')+(available<count?'Some recordings lack the retained norms needed for Pa values; their relative-error curves remain available. ':'')+'All three plots share simulation steps, run toggles, build line styles, and axis scale. Recorded norms and estimates retain each recording’s pressure gauge and qualification.';
 for(const kind of ['absolute','reference'])renderDriftMagnitudePanel(kind,traces,buildDash,describe);
}
function renderDriftMagnitudePanel(kind,traces,buildDash,describe){
 const svg=$('drift-'+kind+'-chart');svg.replaceChildren();
 const log=$('drift-scale').value==='log';
 const valueAt=(trace,index)=>{
  const row=trace.pressure_magnitudes?.[index];if(!row)return null;
  const factor=kind==='absolute'?trace.points[index][2]:1;
  return {value:row[kind==='absolute'?0:1],lower:row[2]*factor,upper:row[3]*factor};
 };
 const plottable=v=>Number.isFinite(v)&&(!log||v>0);
 const values=[];let measured=0,zero=0;
 for(const trace of traces)for(let i=0;i<trace.points.length;i++){
  const metric=valueAt(trace,i);if(!metric)continue;measured++;
  if(!plottable(metric.value)){zero++;continue}
  values.push(...[metric.value,metric.lower,metric.upper].filter(plottable));
 }
 if(!values.length){
  const message=!traces.length?'Select a run in the legend to display its curve.':measured&&log?'All retained values are zero; choose a linear axis.':'Pa measurements unavailable for these recordings; relative error remains below.';
  text(svg,580,160,message,{'text-anchor':'middle'});return;
 }
 let lo=log?Math.floor(Math.log10(arrayMin(values))):0;
 let hi=log?Math.ceil(Math.log10(arrayMax(values))):arrayMax(values)*1.08;
 if(hi===lo)hi=lo+1;
 const scene=traces[0].scene,total=scene.steps-scene.warmup_steps;
 const plot={left:90,right:1125,top:25,bottom:276};
 const x=step=>plot.left+(step-1)/Math.max(1,total-1)*(plot.right-plot.left);
 const y=v=>plot.bottom-((log?Math.log10(v):v)-lo)/(hi-lo)*(plot.bottom-plot.top);
 const ticks=log?Array.from({length:hi-lo+1},(_,i)=>10**(lo+i)):Array.from({length:6},(_,i)=>hi*i/5);
 for(const tick of ticks){line(svg,plot.left,y(tick),plot.right,y(tick));text(svg,plot.left-12,y(tick)+4,sci(tick),{'text-anchor':'end'})}
 const xTicks=[1,...Array.from({length:5},(_,i)=>Math.round(total*(i+1)/5))].filter((s,i,a)=>a.indexOf(s)===i);
 for(const step of xTicks){line(svg,x(step),plot.top,x(step),plot.bottom);text(svg,x(step),302,String(step),{'text-anchor':'middle'})}
 text(svg,(plot.left+plot.right)/2,335,'AI simulation step (native warm-up excluded)',{'text-anchor':'middle','font-size':12});
 text(svg,plot.left,15,'RMS Pa · recorded norms / bounded estimates',{'font-size':11});
 for(const trace of traces){
  let path='',connected=false,segment=[],bands='';const valid=[];
  const closeBand=()=>{
   if(segment.length){bands+='M'+segment.map(p=>p.x+','+y(p.lower)).join(' L')+' L'+[...segment].reverse().map(p=>p.x+','+y(p.upper)).join(' L')+' Z ';segment=[]}
  };
  for(let i=0;i<trace.points.length;i++){
   const point=trace.points[i],metric=valueAt(trace,i);
   if(!metric||!plottable(metric.value)){connected=false;closeBand();continue}
   const px=x(point[0]-scene.warmup_steps);valid.push(i);
   path+=(connected?'L':'M')+px+','+y(metric.value)+' ';connected=true;
   if(plottable(metric.lower)&&plottable(metric.upper))segment.push({x:px,...metric});else closeBand();
  }
  closeBand();if(!valid.length)continue;
  const stroke=driftColor(trace);
  svg.append(el('path',{d:bands,fill:stroke,opacity:.12,'pointer-events':'none','data-pressure-bounds':trace.id}));
  svg.append(el('path',{d:path,fill:'none',stroke,'stroke-width':1.8,opacity:.85,'stroke-dasharray':buildDash(trace)||'none','data-pressure-trace':trace.id,'data-drift-build':trace.native_binary_sha256}));
  const hit=el('path',{d:path,fill:'none',stroke:'transparent','stroke-width':12,'pointer-events':'stroke',cursor:'crosshair'});
  hit.addEventListener('pointermove',event=>{
   const coord=svg.createSVGPoint();coord.x=event.clientX;coord.y=event.clientY;
   const local=coord.matrixTransform(svg.getScreenCTM().inverse());
   const step=1+(local.x-plot.left)/(plot.right-plot.left)*(total-1);
   let nearest=valid[0];for(const index of valid)if(Math.abs(trace.points[index][0]-scene.warmup_steps-step)<Math.abs(trace.points[nearest][0]-scene.warmup_steps-step))nearest=index;
   const detail=describe(trace,trace.points[nearest]);$('drift-detail').textContent=detail;showTip(event,detail,hit);
  });
  hit.addEventListener('pointerleave',()=>$('tooltip').hidden=true);svg.append(hit);
  const index=valid.at(-1),point=trace.points[index],detail=describe(trace,point);
  const dot=historyDot(svg,x(point[0]-scene.warmup_steps),y(valueAt(trace,index).value),stroke,detail);
  dot.setAttribute('data-pressure-end',trace.id);dot.addEventListener('focus',()=>$('drift-detail').textContent=detail);
  if(!trace.completed||trace.qualified_steps<trace.points.length||trace.independent_certificate_failed){dot.setAttribute('stroke',color.amber);dot.setAttribute('stroke-width',3)}
 }
 if(zero)text(svg,plot.right,15,zero+' zero-valued point(s) hidden on log axis',{'text-anchor':'end','font-size':11});
}
function renderDriftSingle(){
 const all=data.drift_traces||[],allowed=driftAllowed(),rankedIDs=new Set(driftRankedEntries().filter(g=>g.score!==null).slice(0,5).map(g=>g.id)),traces=allowed.filter(t=>($('drift-summary').value==='all'||$('drift-top').value!=='5'||rankedIDs.has(t.id))&&driftSceneKey(t)===$('drift-scene').value).sort((a,b)=>driftLabel(a).localeCompare(driftLabel(b))||a.id.localeCompare(b.id));
 const svg=$('drift-chart'),key=$('drift-key'),body=$('drift-values');svg.replaceChildren();key.replaceChildren();body.replaceChildren();
 const buildGroups=driftBuildGroups(),build=buildGroups.find(g=>g.build===$('drift-build').value),scopeWeights=new Set(driftScopeAllowed().map(t=>t.weights_sha256)),shownWeights=new Set(allowed.map(t=>t.weights_sha256)),elsewhere=[...scopeWeights].filter(w=>!shownWeights.has(w)).length;
 $('drift-build-note').textContent=$('drift-build').value==='all'?'Combined view: identical simulation inputs across '+new Set(traces.map(t=>t.native_binary_sha256)).size+' recorded build(s). Solid lines mark the newest build; dashed lines mark earlier builds. Each curve retains its own native reference; the optional growth view uses each run’s own starting relative error.':build?'Build '+build.build.slice(0,8)+': recorded cohorts '+driftCohortIDs(build.weights)+'. '+shownWeights.size+' of '+scopeWeights.size+' checkpoints in this tab’s cohort scope have records on this build.'+(elsewhere?' '+elsewhere+' others have recordings on a different build; switch builds to inspect them.':'')+($('drift-scope').value==='frontier'?' Choose All cohorts with step records to include models outside the accuracy–speed line.':''): 'No builds have recordings under this cohort scope.';
 if($('drift-build').value==='all'&&$('drift-scope').value==='frontier')$('drift-build-note').textContent+=' The cohort scope still shows only accuracy–speed line models; choose All cohorts with step records to include newer models.';
 if($('drift-build').value==='all'&&new Set(traces.map(t=>t.native_binary_sha256)).size===1)$('drift-build-note').textContent+=' This simulation is recorded on only one build; choose a simulation labelled 2 builds to compare both.';
 $('drift-detail').textContent='Hover over a line or focus its endpoint to inspect an individual step.';
 $('drift-coverage').textContent=`${new Set(all.map(t=>t.weights_sha256)).size} checkpoints with verified step records · ${all.length} distinct traces · ${all.reduce((n,t)=>n+t.points.length,0).toLocaleString()} measured AI steps · ${traces.length} runs on this setup under the cohort filter. Repeated identical traces are shown once.`;
 if(!traces.length){$('drift-build-key').replaceChildren();text(svg,580,220,$('drift-top').value==='5'?'No run has a complete score at '+$('drift-horizon').value+' steps. Select fewer AI steps.':'No recorded trajectories match this cohort filter.',{'text-anchor':'middle'});$('drift-scene-details').textContent='No matching simulation.';renderDriftMagnitudes([],()=>'',()=> '');return}
 const normalized=$('drift-metric').value==='normalized',log=$('drift-scale').value==='log',scene=traces[0].scene,total=scene.steps-scene.warmup_steps;
 $('drift-plot-title').textContent=normalized?'Relative error ÷ starting relative error':'Relative pressure error · fraction of reference magnitude';
 $('drift-axis-note').textContent=`X: AI step after ${scene.warmup_steps} native warm-up steps. Y: ${normalized?'pressure error ÷ median error in the first '+$('drift-baseline').value+' AI step(s)':'pressure-field relative L2 error versus Rheon'}. ${log?'Logarithmic':'Linear'} scale. Lower means ${normalized?'less error relative to the starting window':'closer agreement with Rheon'}.`;
 $('drift-baseline').disabled=!normalized;
 $('drift-scene-details').textContent=JSON.stringify($('drift-build').value==='all'?{setup_ids:[...new Set(traces.map(t=>t.scene_key))],native_binary_sha256s:[...new Set(traces.map(t=>t.native_binary_sha256))],...scene}:{setup_id:traces[0].scene_key,native_binary_sha256:traces[0].native_binary_sha256,...scene},null,2)+'\n\nCombined view groups every physical scene field exactly; native builds remain identified separately. Each run’s receipt links its executable and inference policy.';
 const buildDash=t=>t.policy.adaptive_reuse?'2 4':t.policy.reuse_capacity>0?'3 3':$('drift-build').value==='all'?(['','7 4','2 3','10 3 2 3'][buildGroups.findIndex(g=>g.build===t.native_binary_sha256)%4]||''):'';
 const buildKey=$('drift-build-key');buildKey.replaceChildren();if($('drift-build').value==='all')for(const g of buildGroups.filter(g=>traces.some(t=>t.native_binary_sha256===g.build))){const span=document.createElement('span'),swatch=document.createElement('i');swatch.style.cssText='width:20px;height:0;border-radius:0;background:none;border-top:2px '+(g===buildGroups[0]?'solid':'dashed')+' var(--muted)';span.append(swatch,document.createTextNode((g===buildGroups[0]?'Newest build · solid · ':'Earlier build · dashed · ')+g.build.slice(0,8)));buildKey.append(span)}
 const stats=new Map(traces.map(t=>[t.id,driftStats(t)])),visible=traces.filter(t=>!driftHidden.has(t.id)),value=(t,p)=>normalized?(stats.get(t.id).baseline>0?p[2]/stats.get(t.id).baseline:null):p[2];
 for(const t of traces){const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('i');input.type='checkbox';input.checked=!driftHidden.has(t.id);input.setAttribute('aria-label','Show '+driftDisplayLabel(t)+' '+t.id.slice(0,5));input.addEventListener('change',()=>{input.checked?driftHidden.delete(t.id):driftHidden.add(t.id);renderDrift()});swatch.style.borderColor=driftColor(t);swatch.style.borderTopStyle=buildDash(t)?'dashed':'solid';label.append(input,swatch,document.createTextNode(driftDisplayLabel(t)+(traces.filter(a=>a.weights_sha256===t.weights_sha256).length>1?' / '+t.id.slice(0,5):'')));key.append(label)}
 let values=visible.flatMap(t=>t.points.map(p=>value(t,p))).filter(v=>Number.isFinite(v)&&(!log||v>0));
 if(normalized)values.push(1);
 if(!values.length)values=[log?0.1:0,1];
 let lo=log?Math.floor(Math.log10(arrayMin(values))):0,hi=log?Math.ceil(Math.log10(arrayMax(values))):arrayMax(values)*1.08;
 if(hi===lo)hi=lo+1;if(!Number.isFinite(hi)||(!log&&hi<=0))hi=1;
 const plot={left:90,right:1125,top:35,bottom:406},x=s=>plot.left+(s-1)/Math.max(1,total-1)*(plot.right-plot.left),y=v=>plot.bottom-((log?Math.log10(v):v)-lo)/(hi-lo)*(plot.bottom-plot.top);
 const ticks=log?Array.from({length:hi-lo+1},(_,i)=>10**(lo+i)):Array.from({length:6},(_,i)=>hi*i/5);
 for(const tick of ticks){line(svg,plot.left,y(tick),plot.right,y(tick));text(svg,plot.left-12,y(tick)+4,normalized?(tick>=0.01&&tick<1000?Number(tick.toPrecision(3)):sci(tick))+'×':sci(tick),{'text-anchor':'end'})}
 const xTicks=[1,...Array.from({length:5},(_,i)=>Math.round(total*(i+1)/5))].filter((s,i,a)=>a.indexOf(s)===i);
 for(const step of xTicks){line(svg,x(step),plot.top,x(step),plot.bottom);text(svg,x(step),432,String(step),{'text-anchor':'middle'})}
 text(svg,(plot.left+plot.right)/2,462,'AI simulation step (native warm-up excluded)',{'text-anchor':'middle','font-size':12});
 if(normalized){line(svg,plot.left,y(1),plot.right,y(1),color.native,'6 5');text(svg,plot.right,y(1)-8,'1× starting error',{'text-anchor':'end',fill:color.native})}
 const describe=(t,p)=>{const s=stats.get(t.id),ratio=s.baseline>0?(p[2]/s.baseline).toPrecision(4)+'×':'undefined (zero starting error)';return `${driftDisplayLabel(t)} · AI step ${p[0]-scene.warmup_steps}/${total} (physical step ${p[0]}, ${p[1].toFixed(3)} s) · error growth ${ratio} · relative error ${sci(p[2])} (${(100*p[2]).toPrecision(4)}%)${driftMagnitudeDescription(t,t.points.indexOf(p))} · starting relative error ${sci(s.baseline)} · physical checks ${p[3]?'passed':'FAILED'}${t.campaign_disposition?' · campaign: '+t.campaign_disposition.replaceAll('_',' '):''}`};
 let zeroPoints=0,zeroBaselines=0;
 for(const t of visible){const s=stats.get(t.id);if(normalized&&s.baseline===0){zeroBaselines++;continue}
  let path='',connected=false;for(const p of t.points){const v=value(t,p);if(!Number.isFinite(v)||(log&&v<=0)){connected=false;zeroPoints++;continue}path+=(connected?'L':'M')+x(p[0]-scene.warmup_steps)+','+y(v)+' ';connected=true}
  const stroke=driftColor(t),curve=el('path',{d:path,fill:'none',stroke,'stroke-width':1.8,opacity:.85,'stroke-dasharray':buildDash(t)||'none','data-drift-build':t.native_binary_sha256,'data-drift-trace':t.id});svg.append(curve);
  const hit=el('path',{d:path,fill:'none',stroke:'transparent','stroke-width':12,'pointer-events':'stroke',cursor:'crosshair'});
  hit.addEventListener('pointermove',event=>{const coord=svg.createSVGPoint();coord.x=event.clientX;coord.y=event.clientY;const local=coord.matrixTransform(svg.getScreenCTM().inverse()),step=1+(local.x-plot.left)/(plot.right-plot.left)*(total-1);let nearest=t.points[0];for(const p of t.points)if(Math.abs(p[0]-scene.warmup_steps-step)<Math.abs(nearest[0]-scene.warmup_steps-step))nearest=p;const detail=describe(t,nearest);$('drift-detail').textContent=detail;showTip(event,detail,hit)});
  hit.addEventListener('pointerleave',()=>$('tooltip').hidden=true);svg.append(hit);
  const p=t.points.at(-1),v=value(t,p);if(Number.isFinite(v)&&(!log||v>0)){const dot=historyDot(svg,x(p[0]-scene.warmup_steps),y(v),stroke,describe(t,p));dot.setAttribute('data-drift-end',t.id);dot.addEventListener('focus',()=>$('drift-detail').textContent=describe(t,p));if(!t.completed||t.qualified_steps<t.points.length||t.independent_certificate_failed){dot.setAttribute('stroke',color.amber);dot.setAttribute('stroke-width',3)} }
 }
 if(traces.some(t=>['independent-full-field-v4','independent-full-field-v5','independent-full-field-v6'].includes(t.recording_format)))$('drift-axis-note').textContent+=' Pressure Methods IV–VI uses independently advanced spectral references and spatial-mean-removed pressure; Pa norms are recorded exactly. Other recordings retain their original reference and gauge.';
 if(zeroPoints||zeroBaselines)$('drift-axis-note').textContent+=` ${zeroBaselines} zero-baseline trace(s) cannot be normalized; ${zeroPoints} zero-valued point(s) are omitted on the logarithmic axis. Use relative error or the linear axis to inspect them.`;
 if(!visible.length)text(svg,580,220,'Select a run in the legend to display its curve.',{'text-anchor':'middle'});
 renderDriftMagnitudes(visible,buildDash,describe);
 const sorted=[...traces].sort((a,b)=>(stats.get(a.id).worstGrowth??Infinity)-(stats.get(b.id).worstGrowth??Infinity)||driftLabel(a).localeCompare(driftLabel(b)));
 for(const t of sorted){const s=stats.get(t.id),tr=document.createElement('tr'),cells=[driftDisplayLabel(t),sci(s.baseline),sci(s.final),s.finalGrowth===null?'Undefined':s.finalGrowth.toPrecision(4)+'×',s.worstGrowth===null?'Undefined':s.worstGrowth.toPrecision(4)+'×',`${t.points.length} / ${total} · ${t.completed?'full window':t.hard_stop?'hard stop':'shortened'}`,`${t.qualified_steps} / ${t.points.length}`];tr.dataset.driftRun=t.id;tr.dataset.worstGrowth=s.worstGrowth??'';
  for(const value of cells){const td=document.createElement('td');td.textContent=value;tr.append(td)}
  const policy=document.createElement('small');policy.textContent=driftPolicy(t)+' · trace '+t.id.slice(0,5)+(t.campaign_disposition?' · '+t.campaign_disposition.replaceAll('_',' '):'');tr.firstChild.append(policy);tr.firstChild.style.color=driftColor(t);
  const baselineNote=document.createElement('small');baselineNote.textContent=s.baselineSteps+' initial AI step(s)';tr.children[1].append(baselineNote);
  if(!t.completed||t.qualified_steps<t.points.length||t.independent_certificate_failed)tr.children[5].className='drift-short';
  const evidence=document.createElement('td');for(const[label,href]of[['Frames',t.frames_source],['Receipt',t.receipt_source],['Seal',t.source]]){if(evidence.childNodes.length)evidence.append(' · ');const link=document.createElement('a');link.textContent=label;link.href=href;evidence.append(link)}tr.append(evidence);body.append(tr)
 }
}
// Scores normalize each trace before combining equally weighted simulation cases.
const driftAggregateHidden=new Set(),driftWindowSteps=new Set([4,8,32,64,128]);
let driftSeriesCache=new Map(),driftSeriesMode='';
function driftCanonical(value){
 if(Array.isArray(value))return value.map(driftCanonical);
 if(value&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().filter(k=>value[k]!==null&&!(k==='minimum_network_evaluations'&&value[k]===0)).map(k=>[k,driftCanonical(value[k])]));
 return value;
}
function driftGroupID(t){return JSON.stringify([t.weights_sha256,t.native_binary_sha256,driftCanonical(t.policy)])}
function driftScoreSeries(t,kind=$('drift-score').value){
 const mode=$('drift-baseline').value;if(driftSeriesMode!==mode){driftSeriesMode=mode;driftSeriesCache.clear()}
 const key=t.id+'|'+kind;if(driftSeriesCache.has(key))return driftSeriesCache.get(key);
 const baseline=driftStats(t).baseline,values=[];let sum=0,worst=0,complete=true;
 for(const p of t.points){const step=p[0]-t.scene.warmup_steps,v=baseline>0?p[2]/baseline:null;if(step!==values.length+1||!Number.isFinite(v)){complete=false;break}sum+=v;worst=Math.max(worst,v);values.push(kind==='worst'?worst:kind==='mean'?sum/step:v)}
 const result={values,baseline,complete};driftSeriesCache.set(key,result);return result;
}
function driftGroups(traces){
 const groups=new Map();for(const t of traces){const id=driftGroupID(t);if(!groups.has(id))groups.set(id,{id,groupId:id,trace:t,traces:[],cases:new Map()});const g=groups.get(id),scene=driftSceneIdentity(t.scene);g.traces.push(t);if(!g.cases.has(scene))g.cases.set(scene,[]);g.cases.get(scene).push(t)}return [...groups.values()];
}
function driftReduce(values,kind){return kind==='worst'?arrayMax(values):kind==='median'?driftMedian(values):values.reduce((a,b)=>a+b,0)/values.length}
function driftSummarize(g,horizon,kind=$('drift-score').value,reducer=$('drift-reducer').value,populationHorizon=horizon){
 const cases=[...g.cases.values()].filter(ts=>ts[0].scene.steps-ts[0].scene.warmup_steps>=populationHorizon),curve=[],series=cases.map(ts=>ts.map(t=>driftScoreSeries(t,kind).values));
 let missing=0,failed=0;
 for(let i=0;i<cases.length;i++){if(series[i].some(v=>v.length<horizon))missing++;if(cases[i].some(t=>t.points.slice(0,horizon).some(p=>!p[3])))failed++}
 for(let step=1;step<=horizon;step++){
  const values=series.map(repeats=>repeats.every(v=>Number.isFinite(v[step-1]))?Math.max(...repeats.map(v=>v[step-1])):null);
  curve.push(values.length&&values.every(v=>v!==null)?driftReduce(values,reducer):null);
 }
 const campaign=(data.rollout_campaign_status||[]).find(c=>c.weights_sha256===g.trace.weights_sha256&&!c.campaign_complete),aggregate=g.id===g.groupId,unavailableCampaign=aggregate&&campaign;
 return {...g,score:unavailableCampaign?null:curve.at(-1)??null,curve,count:cases.length,total:g.cases.size,short:g.cases.size-cases.length,missing,failed,selectedCases:cases,campaign:unavailableCampaign?campaign:null,certificateFailed:cases.filter(ts=>ts.some(t=>t.independent_certificate_failed)).length};
}
function driftScoreEntries(){
 const horizon=Number($('drift-horizon').value),all=$('drift-summary').value==='all';
 if(all)return driftGroups(driftAllowed()).map(g=>driftSummarize(g,horizon));
 return driftAllowed().filter(t=>driftSceneKey(t)===$('drift-scene').value).map(t=>{const g={id:t.id,groupId:driftGroupID(t),trace:t,traces:[t],cases:new Map([[driftSceneIdentity(t.scene),[t]]])};return driftSummarize(g,horizon,$('drift-score').value,'mean')});
}
function driftRankedEntries(){return driftScoreEntries().sort((a,b)=>(a.score??Infinity)-(b.score??Infinity)||driftDisplayLabel(a.trace).localeCompare(driftDisplayLabel(b.trace))||a.id.localeCompare(b.id))}
function driftShownEntries(){const entries=driftRankedEntries();return $('drift-top').value==='5'?entries.filter(g=>g.score!==null).slice(0,5):entries}
function driftScoreFormat(value){return value===null?'Unranked':Number(value.toPrecision(5))+'×'}
function driftSourceLinks(cell,g){
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent=g.traces.length+' trace record(s)';details.append(summary);
 for(const t of g.traces){const row=document.createElement('div');row.style.marginBottom='6px';row.append(document.createTextNode(t.scene.counts.join('×')+' · '+(t.scene.steps-t.scene.warmup_steps)+' steps · '+t.scene_key.slice(0,5)+' · '));for(const [label,href] of [['Receipt',t.receipt_source],['Frames',t.frames_source]]){const a=document.createElement('a');a.textContent=label;a.href=href;row.append(a,' ')}details.append(row)}cell.append(details);
}
function renderDriftScores(){
 const entries=driftRankedEntries(),shown=driftShownEntries(),horizon=Number($('drift-horizon').value),all=$('drift-summary').value==='all',body=$('drift-score-values');body.replaceChildren();
 $('drift-reducer-control').hidden=!all;$('drift-single-card').hidden=all;$('drift-aggregate-card').hidden=!all;$('drift-raw-records').hidden=all;$('drift-setup-record').hidden=all;
 $('drift-scene').disabled=all;$('drift-metric').disabled=all;$('drift-baseline').disabled=!all&&$('drift-metric').value!=='normalized';
 const rankable=entries.filter(g=>g.score!==null).length,counts=entries.filter(g=>g.count).map(g=>g.count),sameSuites=new Set(entries.filter(g=>g.count).map(g=>JSON.stringify(g.selectedCases.map(ts=>driftSceneIdentity(ts[0].scene)).sort()))).size;
 $('drift-score-note').textContent=`Ranking: ${$('drift-score').selectedOptions[0].textContent.toLowerCase()} at ${horizon} AI steps, normalized to each simulation’s own starting error. ${all?$('drift-reducer').selectedOptions[0].textContent+'. Checkpoint, inference policy and native build stay separate; repeat records of the same setup use their worst score, then every distinct simulation has equal weight. ':'Only the selected simulation contributes. '}${rankable}/${entries.length} scores are rankable; ${shown.length} shown${$('drift-top').value==='5'?' (best five, before manual curve toggles)':''}. Shorter planned windows are excluded; any missing required steps or zero starting error leave the score unranked. Physical-check failures are reported separately.${all&&counts.length?' Contributing simulations per score: '+arrayMin(counts)+'–'+arrayMax(counts)+'.':''}${all&&sameSuites>1?' These models have different recorded simulation sets: the ranking describes their available evidence, so scores are not a controlled comparison on identical inputs.':''}`;
 for(const g of shown){const tr=document.createElement('tr'),rank=g.score===null?'—':entries.filter(a=>a.score!==null).indexOf(g)+1;tr.dataset.driftScore=g.id;tr.dataset.score=g.score??'';const cells=[rank+'. '+driftDisplayLabel(g.trace),driftScoreFormat(g.score),g.count+' / '+g.total+' distinct simulations',g.short+' shorter windows · '+g.missing+' missing/undefined · '+g.failed+' with failed physical checks'+(g.certificateFailed?' · '+g.certificateFailed+' independent certificate failure(s)':'')+(g.campaign?' · incomplete '+g.campaign.scheduled_scenes+'-scene campaign; aggregate unranked':'')];for(const v of cells){const td=document.createElement('td');td.textContent=v;tr.append(td)}const policy=document.createElement('small');policy.textContent=driftPolicy(g.trace);tr.firstChild.append(policy);tr.firstChild.style.color=driftColor(g.trace);const evidence=document.createElement('td');driftSourceLinks(evidence,g);tr.append(evidence);body.append(tr)}
 if(all){
  $('drift-build-note').textContent=$('drift-build').value==='all'?'All recorded native builds are included. Each checkpoint / policy / build has its own score. Solid curves identify the newest build; dashed curves identify earlier builds.':'Native build '+$('drift-build').value.slice(0,8)+' · all simulations in this tab’s cohort scope. Choose All builds together to include recordings from other builds.';
  $('drift-coverage').textContent=driftAllowed().length+' traces in scope · '+entries.length+' checkpoint / policy / build groups · '+entries.reduce((n,g)=>n+g.count,0)+' contributing simulation cases across groups at '+horizon+' steps.';
  renderDriftAggregate(shown);
 }
 renderDriftWindows(shown);
}
function driftScoreAxes(svg,values,total){
 const log=$('drift-scale').value==='log',valid=values.filter(v=>Number.isFinite(v)&&(!log||v>0));valid.push(1);
 let lo=log?Math.floor(Math.log10(arrayMin(valid))):0,hi=log?Math.ceil(Math.log10(arrayMax(valid))):arrayMax(valid)*1.08;if(hi<=lo)hi=lo+1;
 const plot={left:90,right:1125,top:35,bottom:386},x=step=>plot.left+(step-1)/Math.max(1,total-1)*(plot.right-plot.left),y=v=>plot.bottom-((log?Math.log10(v):v)-lo)/(hi-lo)*(plot.bottom-plot.top);
 const ticks=log?Array.from({length:hi-lo+1},(_,i)=>10**(lo+i)):Array.from({length:6},(_,i)=>hi*i/5);
 for(const v of ticks){line(svg,plot.left,y(v),plot.right,y(v));text(svg,plot.left-12,y(v)+4,Number(v.toPrecision(3))+'×',{'text-anchor':'end'})}
 line(svg,plot.left,y(1),plot.right,y(1),color.native,'6 5');return{...plot,x,y,log};
}
function renderDriftAggregate(entries){
 const svg=$('drift-aggregate-chart'),key=$('drift-aggregate-key'),horizon=Number($('drift-horizon').value),visible=entries.filter(g=>!driftAggregateHidden.has(g.id));svg.replaceChildren();key.replaceChildren();
 $('drift-aggregate-note').textContent=`X: AI steps, through ${horizon}. Y: ${$('drift-reducer').selectedOptions[0].textContent.toLowerCase()} of each simulation’s ${$('drift-score').selectedOptions[0].textContent.toLowerCase()}. Each model uses the same simulation set at every point of its curve: only setups planned to reach ${horizon} steps. Missing steps break the curve. Build IDs and exact policies are listed in the score table. The raw-error measure applies to the single-simulation chart; these scores always use relative growth.`;
 const p=driftScoreAxes(svg,visible.flatMap(g=>g.curve),horizon),builds=driftBuildGroups();
 for(const step of [...new Set([1,...[4,8,16,32,64,128,256].filter(n=>n<=horizon),horizon])]){text(svg,p.x(step),p.bottom+26,String(step),{'text-anchor':'middle'});line(svg,p.x(step),p.top,p.x(step),p.bottom)}
 text(svg,600,458,'AI simulation steps · fixed simulation population within each curve',{'text-anchor':'middle'});
 for(const g of entries){const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('i');input.type='checkbox';input.checked=!driftAggregateHidden.has(g.id);input.setAttribute('aria-label','Show aggregate '+driftDisplayLabel(g.trace)+' '+g.trace.id.slice(0,5));input.addEventListener('change',()=>{input.checked?driftAggregateHidden.delete(g.id):driftAggregateHidden.add(g.id);renderDrift()});swatch.style.borderColor=driftColor(g.trace);label.append(input,swatch,document.createTextNode(driftDisplayLabel(g.trace)+' · '+g.count+' simulations'));key.append(label)}
 for(const g of visible){let path='',connected=false;g.curve.forEach((v,i)=>{if(v===null||(p.log&&v<=0)){connected=false;return}path+=(connected?'L':'M')+p.x(i+1)+','+p.y(v)+' ';connected=true});const stroke=driftColor(g.trace),dash=g.trace.policy.adaptive_reuse?'2 4':g.trace.policy.reuse_capacity>0?'3 3':$('drift-build').value==='all'&&builds.findIndex(b=>b.build===g.trace.native_binary_sha256)>0?'7 4':'none';svg.append(el('path',{d:path,fill:'none',stroke,'stroke-width':2,'stroke-dasharray':dash,'data-drift-aggregate':g.id}));for(const step of [...new Set([4,8,16,32,64,128,256,horizon])].filter(n=>n<=horizon)){const v=g.curve[step-1];if(v===null||(p.log&&v<=0))continue;historyDot(svg,p.x(step),p.y(v),stroke,driftDisplayLabel(g.trace)+' · '+step+' AI steps · '+driftScoreFormat(v)+' · '+g.count+' simulations'+(g.campaign?' · observed subset of incomplete campaign; unranked':'')+' · '+driftPolicy(g.trace))}}
 if(!visible.some(g=>g.curve.some(v=>v!==null)))text(svg,600,220,'No score has complete evidence at this window. Select fewer AI steps.',{'text-anchor':'middle'});
}
function renderDriftWindows(shown){
 const svg=$('drift-window-chart'),key=$('drift-window-key'),head=$('drift-window-head'),body=$('drift-window-values'),steps=[...driftWindowSteps].sort((a,b)=>a-b),matched=$('drift-window-coverage').value==='matched',top=$('drift-top').value==='5',selected=new Set(shown.map(g=>g.groupId)),groups=driftGroups(driftAllowed()).filter(g=>!top||selected.has(g.id)).sort((a,b)=>(driftCohorts.get(a.trace.weights_sha256)?.order||0)-(driftCohorts.get(b.trace.weights_sha256)?.order||0)||a.id.localeCompare(b.id));
 svg.replaceChildren();key.replaceChildren();head.replaceChildren();body.replaceChildren();
 const records=groups.map(g=>({g,scores:steps.map(n=>driftSummarize(g,n,'mean','mean',matched?arrayMax(steps):n))}));
 $('drift-window-note').textContent=`Each point averages the first N AI steps, then averages distinct simulations; duplicate setup records use the worst average. ${matched?'For each model/policy, every selected window uses the same simulations planned to reach the longest selected window ('+(steps.length?arrayMax(steps):'—')+' steps).':'Each window uses all simulations planned to reach it; counts and simulation mix can change between windows.'} ${top?'Models follow the best-five ranking above.':'Models run left to right in cohort training order; separate policies/builds occupy separate positions.'} Missing required steps or incomplete campaigns produce a gap, never an extrapolation. Models may have different recorded simulation sets; hover/focus a point or open the table for counts.`;
 if(!steps.length||!groups.length){text(svg,600,220,!steps.length?'Select at least one AI-step window.':'No models match the score filter.',{'text-anchor':'middle'});return}
 const colors=['#60dfc3','#f4ba71','#8dafff','#eaa0d9','#ff827d','#b4de79','#bba0ff'],allSteps=[4,8,16,32,64,128,256],p=driftScoreAxes(svg,records.flatMap(r=>r.scores.map(s=>s.score)),Math.max(2,groups.length)),x=i=>groups.length===1?(p.left+p.right)/2:p.x(i+1);
 svg.style.minWidth=Math.max(760,groups.length*42)+'px';
 const row=document.createElement('tr');for(const title of ['Model / policy / native build',...steps.map(n=>n+' AI steps · mean growth / simulations')]){const th=document.createElement('th');th.textContent=title;row.append(th)}head.append(row);
 records.forEach(({g,scores},i)=>{const id=driftCohorts.get(g.trace.weights_sha256)?.id||g.trace.model;const label=el('text',{x:x(i),y:p.bottom+22,fill:color.muted,'font-size':10,'text-anchor':'end',transform:'rotate(-45 '+x(i)+' '+(p.bottom+22)+')'},id+' · '+(i+1));label.append(el('title',{},driftDisplayLabel(g.trace)+' · '+driftPolicy(g.trace)));svg.append(label);const tr=document.createElement('tr'),name=document.createElement('td');name.textContent=(i+1)+'. '+driftDisplayLabel(g.trace)+' · '+driftPolicy(g.trace)+' · build '+g.trace.native_binary_sha256.slice(0,8);tr.append(name);scores.forEach((s,j)=>{const td=document.createElement('td');td.textContent=driftScoreFormat(s.score)+' · '+s.count+'/'+s.total+' simulations'+(s.missing?' · '+s.missing+' missing':'');td.dataset.window=steps[j];td.dataset.windowScore=s.score??'';td.dataset.simulations=s.count;tr.append(td)});body.append(tr)});
 steps.forEach((n,j)=>{const stroke=colors[allSteps.indexOf(n)],span=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=stroke;span.append(swatch,document.createTextNode(n+' AI steps'));key.append(span);let path='',connected=false;records.forEach(({g,scores},i)=>{const s=scores[j],v=s.score;if(v===null||(p.log&&v<=0)){connected=false;return}path+=(connected?'L':'M')+x(i)+','+p.y(v)+' ';connected=true;const dot=historyDot(svg,x(i),p.y(v),stroke,n+' AI steps · '+driftDisplayLabel(g.trace)+' · mean '+driftScoreFormat(v)+' · '+s.count+'/'+s.total+' simulations · '+s.failed+' with failed physical checks · '+driftPolicy(g.trace));dot.dataset.driftWindow=n;dot.dataset.simulations=s.count;dot.dataset.score=v});svg.prepend(el('path',{d:path,fill:'none',stroke,'stroke-width':2,'data-drift-window-line':n}))});
 text(svg,600,466,'Models / inference policies in cohort training order · numbered positions match the table',{'text-anchor':'middle'});
}
function renderCheckpointQualification(){
 const box=$('cohort-qualification-note');box.replaceChildren();const campaigns=(data.rollout_campaign_status||[]).filter(c=>pressureBenchmarkPoints($('pressure-benchmark').value||data.evaluation.default_benchmark,$('pressure-execution').value).some(p=>p.weights_sha256===c.weights_sha256)&&cohortMatchesScope(driftCohorts.get(c.weights_sha256),$('cohort-filter').value,'overview'));box.hidden=!campaigns.length;
 for(const c of campaigns){const row=document.createElement('p');row.style.color='inherit';row.append(document.createTextNode((driftCohorts.get(c.weights_sha256)?.id||c.model)+' · '+c.model+': training completed, rollout qualification '+c.qualification+'. Its plotted speed and accuracy are CPU diagnostics on the same saved validation states as other checkpoints. The rollout campaign is incomplete; these measurements do not establish a qualified solver. '));const link=document.createElement('a');link.href=c.source;link.textContent='Recorded outcome';row.append(link);box.append(row)}
}
function renderRolloutCampaignStatus(){
 const box=$('drift-campaign-status'),allowed=new Set(driftAllowed().map(t=>t.weights_sha256));box.replaceChildren();const campaigns=(data.rollout_campaign_status||[]).filter(c=>allowed.has(c.weights_sha256));box.hidden=!campaigns.length;
 for(const c of campaigns){const row=document.createElement('p'),d=c.dispositions;row.style.color='inherit';row.append(document.createTextNode((driftCohorts.get(c.weights_sha256)?.id||c.model)+' · rollout qualification '+c.qualification+': '+(d.passed_complete_window||0)+' complete passing windows, '+(d.native_candidate_rejection||0)+' native rejections, '+(d.independent_pressure_bound_failure||0)+' independent pressure-certificate failure, '+(d.cancelled_after_pool_stop||0)+' cancelled, '+(d.not_started||0)+' unstarted. '+c.observed_traces+' simulations have saved AI-step curves. Cancelled/unstarted cases are not model failures. Aggregate scores remain unranked because this '+c.scheduled_scenes+'-scene campaign is incomplete; curves describe recorded cases only. '));const a=document.createElement('a');a.href=c.source;a.textContent='Qualification record';row.append(a);box.append(row)}
}
function renderFourPassRolloutStatus(){
 const box=$('drift-four-pass-status'),summaries=data.four_pass_rollout_summary||[];box.replaceChildren();box.hidden=!summaries.length;if(!summaries.length)return;
 const note=document.createElement('p');note.style.color='inherit';note.textContent=summaries.map(r=>r.model_id+' · fixed '+r.passes+' passes: '+r.complete_windows+'/'+r.scheduled_scenes+' complete windows, '+r.rejected_windows+' rejected windows').join(' · ')+'. '+data.cohort_points.filter(p=>p.initialized_from&&!(data.drift_traces||[]).some(t=>t.weights_sha256===p.weights_sha256)).map(p=>p.id).sort().join(', ')+' have no recorded rollout curves. These fresh inputs and fixed-pass policies differ from the older adaptive campaigns; they remain separate score groups. The two eight-pass models can be overlaid on the same fresh setup.';
 const button=document.createElement('button');button.type='button';button.textContent='Show C53 / C59 fresh simulation';button.addEventListener('click',()=>{const t=(data.drift_traces||[]).find(t=>t.chart_study==='pressure-four-pass-distillation-v1');if(!t)return;$('cohorts-drift-reset').click();$('drift-scope').value='all';$('drift-scope').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();$('drift-summary').value='single';$('drift-top').value='all';$('drift-scene').value=driftSceneKey(t);driftHidden.clear();renderDrift()});
 const link=document.createElement('a');link.href='pressure-four-pass-distillation-v1/index.html';link.textContent='Full fine-tuning study';box.append(note,button,document.createTextNode(' '),link);
}
function renderDrift(){renderDriftSingle();renderDriftScores();renderRolloutCampaignStatus();renderFourPassRolloutStatus()}
for(const n of [4,8,16,32,64,128,256]){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=n;input.checked=driftWindowSteps.has(n);input.setAttribute('aria-label','Compare '+n+' AI-step averages');input.addEventListener('change',()=>{input.checked?driftWindowSteps.add(n):driftWindowSteps.delete(n);renderDrift()});label.append(input,document.createTextNode(String(n)));$('drift-window-options').append(label)}
for(const id of ['drift-summary','drift-top','drift-horizon','drift-score','drift-reducer','drift-window-coverage'])$(id).addEventListener('change',renderDrift);

$('drift-scene').addEventListener('change',()=>{driftHidden.clear();renderDrift()});
for(const id of ['drift-metric','drift-baseline','drift-scale'])$(id).addEventListener('change',renderDrift);
$('drift-scope').addEventListener('change',selectDriftBuild);
$('drift-build').addEventListener('change',selectDriftScene);
const c98DriftPoint=data.cohort_points.find(p=>p.id==='C98');
const c98DriftTraces=(data.drift_traces||[]).filter(t=>t.weights_sha256===c98DriftPoint?.weights_sha256);
$('drift-c98-shortcut').hidden=!c98DriftTraces.length;
if(c98DriftTraces.length){
 const longest=Math.max(...c98DriftTraces.map(t=>t.points.at(-1)?.[1]||0)),hasLong=c98DriftTraces.some(t=>t.scene.steps===7200);
 $('drift-show-c98').textContent='Compare C98 and C95 · '+(hasLong?'120-second':'8.53-second')+' drift';
 $('drift-c98-note').textContent='C98: '+c98DriftTraces.length+' completed recordings, through '+longest.toFixed(2)+' seconds. Seven pressure-correction passes per timestep, matching C95. Select another simulation above to inspect each scene. '+(hasLong?'The aggregate view includes four recorded setups: three scenes, one also recorded for 120 seconds.':'The three short scenes are available; the 120-second follow-up is not included until its measurement completes.');
}
$('drift-show-c98').addEventListener('click',()=>{
 const t=c98DriftTraces.find(t=>t.scene.steps===7200)||c98DriftTraces[0];if(!t)return;
 $('cohorts-drift-reset').click();$('cohorts-drift-from').value='C95';$('cohorts-drift-to').value='C98';$('cohorts-drift-from').dispatchEvent(new Event('change',{bubbles:true}));
 $('drift-scope').value='all';$('drift-scope').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();
 $('drift-summary').value='single';$('drift-top').value='all';$('drift-scene').value=driftSceneKey(t);driftHidden.clear();renderDrift();
});
$('cohort-clear-emphasis').addEventListener('click',clearCohortEmphasis);
$('cohort-isolate-group').addEventListener('click',()=>{if(emphasizedCohortFamily){$('cohorts-overview-group').value=emphasizedCohortFamily;$('cohorts-overview-group').dispatchEvent(new Event('change',{bubbles:true}))}});
$('cohort-scatter').addEventListener('keydown',e=>{if(e.key==='Escape')clearCohortEmphasis()});
$('cohort-filter').addEventListener('change',renderCohortPlots);$('cohort-labels').addEventListener('change',renderCohortPlots);
$('tradeoff-weight').addEventListener('input',renderCohortPlots);
