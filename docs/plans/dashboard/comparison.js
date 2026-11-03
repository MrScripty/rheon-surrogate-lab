function pressureBenchmarkPoints(id,execution='original'){return Evaluation.pressurePoints(data,id,execution)}
function renderCohortPlots(){
 renderCheckpointQualification();
 const id=$('pressure-benchmark').value,benchmark=data.evaluation.benchmarks.find(b=>b.id===id),historical=id==='historical',gpu=$('pressure-device').value==='gpu';
 const pool=gpu?[]:pressureBenchmarkPoints(id,$('pressure-execution').value).filter(p=>cohortMatchesSelection(p,'overview'));
 const analysis=Evaluation.analyze(pool,Number($('tradeoff-weight').value)/100),frontier=new Set(analysis.frontier);
 const scope=$('cohort-filter').value;
 const points=pool.filter(p=>scope==='frontier'?frontier.has(p):scope==='matched'?(p.historical_population||cohortPopulationKey(p)===matchedPopulationKey):true);
 const final=Evaluation.analyze(points,analysis.weight),winner=final.ranked[0];
 $('cohort-filter').querySelector('[value="frontier"]').disabled=historical||!benchmark?.ranking_eligible;
 $('tradeoff-weight').disabled=historical||gpu||!benchmark?.ranking_eligible;
 $('pressure-execution').disabled=historical;
 $('tradeoff-weight-value').textContent=Math.round(analysis.weight*100)+'%';
 $('tradeoff-count').textContent=final.ranked.length;
 $('pressure-method').replaceChildren();
 const method=$('pressure-method');$('pressure-summary').textContent=gpu?'GPU timing unavailable':historical?'Historical timing procedures differ · exploratory only':`CPU · ${benchmark?.threads??'?'} thread · batch ${benchmark?.batch_size??'?'} · ${benchmark?.population.replaceAll('_',' ')||''} · benchmark conditions & evidence`;
 method.textContent=gpu?'No measured GPU inference timing is recorded. CUDA training does not establish GPU inference performance. Select CPU to view captured timings.':historical?'Historical exploration: original CPU proposal captures across different timing procedures. A common 128-state error population does not make these latency measurements comparable. No recommendation or accuracy–speed line is calculated. Choose a repeated benchmark to rank.':`${benchmark?.label||'No benchmark'} · CPU · ${benchmark?.threads??'unrecorded'} Torch thread · batch ${benchmark?.batch_size??'unrecorded'} · ${benchmark?.cpu===null?'CPU affinity not declared in this summary':'CPU '+benchmark?.cpu}. ${benchmark?.timing_scope||''}. Warmup: ${benchmark?.warmup||''}. Accuracy and timing coverage are detailed in the evidence.`;
 if(benchmark&&!gpu){const link=document.createElement('a');link.href=benchmark.source;link.textContent=' Benchmark evidence';method.append(link)}
 $('tradeoff-finding').textContent=winner?`Preferred proposal at this weighting: ${winner.id} · ${winner.measurement_label} — worst equation error ${sci(winner.worst_error)}, ${winner.inference_median_ms.toFixed(3)} ms. ${final.frontier.length} nondominated executions. This ranks proposals; it does not establish accepted solver speed. All captured full-AI proposals remain above the 1e−6 equation target.`:historical?'Historical observations have no comparable latency ranking. Use Training progress for all matched accuracy results.':'No comparable measurements in this selection. Broaden the cohort range, execution filter or benchmark; GPU data cannot be substituted with CPU timing.';
 $('cohort-coverage').textContent=`${new Set(points.map(p=>p.id)).size} selected checkpoints · ${points.length} executions · ${data.cohort_points.filter(p=>data.solver_comparison?.classification[p.id]?.kind==='full').length} full-AI cohorts in the historical view. Neural correction stages are not generated simulation steps.`;
 const svg=$('cohort-scatter');svg.setAttribute('viewBox','0 0 1200 510');svg.replaceChildren();
 if(points.length){
  const a=evaluationAxes(svg,points.flatMap(p=>[p.inference_median_ms,p.repeat_min_ms||p.inference_median_ms,p.repeat_max_ms||p.inference_median_ms]),points.map(p=>p.worst_error),'Median CPU time per pressure proposal (ms) →');
  text(svg,20,220,'Worst relative equation residual →',{transform:'rotate(-90 20 220)','text-anchor':'middle','font-size':12});
  svg.append(el('path',{d:final.frontier.map((p,i)=>(i?'L':'M')+a.x(p.inference_median_ms)+','+a.y(p.worst_error)).join(' '),fill:'none',stroke:color.blue,'stroke-width':2,'stroke-dasharray':'6 5','data-cohort-frontier':''}));
  for(const p of points){if(p.repeat_min_ms>0&&p.repeat_max_ms>0){const y=a.y(p.worst_error);line(svg,a.x(p.repeat_min_ms),y,a.x(p.repeat_max_ms),y,cohortColor(p));for(const t of [p.repeat_min_ms,p.repeat_max_ms])line(svg,a.x(t),y-4,a.x(t),y+4,cohortColor(p))}
   cohortDot(svg,p,a.x(p.inference_median_ms),a.y(p.worst_error),cohortColor(p),`${p.measurement_label}; ${p.inference_median_ms.toFixed(3)} ms; worst equation error ${sci(p.worst_error)}`);
   if($('cohort-labels').checked)text(svg,a.x(p.inference_median_ms)+10,a.y(p.worst_error)-10,p.id,{'font-size':11,fill:cohortColor(p),'data-cohort-group':p.family});
  }
  if(winner){const ring=el('circle',{cx:a.x(winner.inference_median_ms),cy:a.y(winner.worst_error),r:11,fill:'none',stroke:'#fff','stroke-width':2,'aria-label':'Preferred proposal '+winner.id,'data-cohort-group':winner.family});svg.append(ring)}
 }else text(svg,600,230,gpu?'GPU inference timing has not been recorded.':'No measurements match these filters.',{'text-anchor':'middle','font-size':17});
 $('cohort-key').replaceChildren();for(const family of [...new Set(points.map(p=>p.family))]){const button=document.createElement('button');button.type='button';button.dataset.cohortGroup=family;const swatch=document.createElement('i');swatch.style.background=cohortColor(points.find(p=>p.family===family));button.append(swatch,document.createTextNode(family));button.addEventListener('click',()=>{selectedCohort=null;emphasizedCohortFamily=emphasizedCohortFamily===family?null:family;applyCohortEmphasis()});$('cohort-key').append(button)}
 if(!points.some(p=>p.id===selectedCohort))selectedCohort=null;
 if(!points.some(p=>p.family===emphasizedCohortFamily))emphasizedCohortFamily=null;
 applyCohortEmphasis();
 $('cohort-detail').textContent='Select a point for its execution, learned correction stages, initialization and source records. Bars show repeat-median ranges where available.';
 const body=$('cohort-values');body.replaceChildren();
 for(const p of [...points].sort((a,b)=>a.worst_error-b.worst_error)){
  const tr=document.createElement('tr');for(const value of [p.id+' · '+p.measurement_label,p.trained_at.replace('T',' ').slice(0,19),p.best_error>0?sci(p.best_error):'Not recorded for this population',p.best_error_raw>0?sci(p.best_error_raw):'Not recorded',sci(p.worst_error),p.inference_median_ms.toFixed(3)+' ms',p.cases??'See evidence',p.best_error_eligible_cases==null?'See evidence':p.best_error_eligible_cases+' / '+p.best_error_excluded_cases]){const td=document.createElement('td');td.textContent=value;tr.append(td)}
  const td=document.createElement('td');for(const [label,href]of [['Accuracy',p.source],['Timing',p.timing_source],['Training',p.training_source]]){if(!href)continue;const a=document.createElement('a');a.href=href;a.textContent=label;td.append(a,document.createTextNode(' '))}tr.append(td);body.append(tr);
 }
}
function renderTrainingProgress(){
 const type=$('progress-kind').value,scope=$('progress-scope').value;
 const points=data.cohort_points.filter(p=>cohortMatchesSelection(p,'progress')&&(type==='all'||data.solver_comparison?.classification[p.id]?.kind===type)&&(scope==='all'||cohortPopulationKey(p)===matchedPopulationKey)).sort((a,b)=>a.order-b.order);
 $('progress-coverage').textContent=`${points.length} checkpoints in training order. Each model’s error uses its original captured population; matched-only selects the common 128 requests. Colors show best and worst error. Filter a training group to follow its method.`;
 const svg=$('cohort-chronology');svg.setAttribute('viewBox','0 0 1200 510');svg.replaceChildren();
 if(!points.length){text(svg,600,220,'No accuracy records match this selection.',{'text-anchor':'middle'});return}
 const a=passAxes(svg,[0,Math.max(1,points.length-1)],passBounds(points.flatMap(p=>[p.best_error,p.worst_error]).filter(v=>v>0),true),false,true,'Chronological training order →');
 for(const n of [...svg.children])if(n.tagName==='text'&&n.getAttribute('y')==='445')n.remove();
 text(svg,20,220,'Relative equation residual →',{transform:'rotate(-90 20 220)','text-anchor':'middle','font-size':12});
 for(const [key,col]of [['best_error',color.teal],['worst_error',color.amber]]){
  const kinds=key==='best_error'?['filtered','historical_unfiltered']:['all'];
  for(const kind of kinds){let d='',connected=false;points.forEach((p,i)=>{const valid=p[key]>0&&(key!=='best_error'||(p.best_error_status==='historical_unfiltered')===(kind==='historical_unfiltered'));if(!valid){connected=false;return}d+=(connected?'L':'M')+a.x(i)+','+a.y(p[key])+' ';connected=true});svg.append(el('path',{d,fill:'none',stroke:col,'stroke-width':2,'stroke-dasharray':kind==='historical_unfiltered'?'5 4':'none','data-best-population':kind}))}
  points.forEach((p,i)=>{if(!(p[key]>0))return;const label=`${p.id} · ${p.family} · ${p.name} · ${key==='best_error'?cohortBestSummary(p):'Worst error '+sci(p.worst_error)} · trained ${p.trained_at}`;const dot=historyDot(svg,a.x(i),a.y(p[key]),col,label);dot.dataset.progressCohort=p.id;for(const ev of ['focus','pointerenter','click'])dot.addEventListener(ev,()=>{$('progress-detail').textContent=label})});
 }
 const stride=Math.max(1,Math.ceil(points.length/20));points.forEach((p,i)=>{if(i%stride===0||i===points.length-1)text(svg,a.x(i),447,p.id,{'text-anchor':'middle','font-size':10})});
}
function setupEvaluation(){
 const select=$('pressure-benchmark');for(const b of data.evaluation.benchmarks)select.append(new Option(b.label+' · '+b.measurements.length+' executions',b.id));select.append(new Option('Historical · all captured cohorts · exploratory only','historical'));select.value=data.evaluation.default_benchmark;
 for(const id of ['pressure-benchmark','pressure-device','pressure-execution'])$(id).addEventListener('change',()=>{if(id==='pressure-benchmark'&&select.value==='historical'&&$('cohort-filter').value==='frontier')$('cohort-filter').value='all';renderCohortPlots()});
 for(const id of ['progress-kind','progress-scope'])$(id).addEventListener('change',renderTrainingProgress);
 setupMultiStep();setupExpandedModels();setupTemporalPilots();renderMethodology();
}
// Fit log axes to observed data, including repeat ranges. Narrow common timing
// captures should not be squeezed into a full decade with indistinguishable dots.
function evaluationAxes(svg,xValues,yValues,xLabel){
 const bounds=values=>{const logs=values.filter(v=>v>0).map(v=>Math.log10(v));const lo=arrayMin(logs),hi=arrayMax(logs),pad=Math.max((hi-lo)*.12,.015);return[lo-pad,hi+pad]};
 const xb=bounds(xValues),yb=bounds(yValues),left=95,right=1125,top=30,bottom=418;
 const x=v=>left+(Math.log10(v)-xb[0])/(xb[1]-xb[0])*(right-left),y=v=>bottom-(Math.log10(v)-yb[0])/(yb[1]-yb[0])*(bottom-top);
 for(let i=0;i<=5;i++){const xv=10**(xb[0]+i*(xb[1]-xb[0])/5),yv=10**(yb[0]+i*(yb[1]-yb[0])/5);line(svg,x(xv),top,x(xv),bottom);text(svg,x(xv),445,Number(xv.toPrecision(3)),{'text-anchor':'middle'});line(svg,left,y(yv),right,y(yv));text(svg,left-10,y(yv)+4,sci(yv),{'text-anchor':'end'})}
 text(svg,(left+right)/2,480,xLabel,{'text-anchor':'middle','font-size':12});return{x,y,left,right,top,bottom};
}
