const cohortPalette=['#c2ced7','#7dd3fc','#c4a4ff','#f4ba71','#f78fa5','#60dfc3','#90c978','#9cb4ff','#e49af1','#ff9f78','#59c9d5','#e3d78a'];
const cohortFamilies=[...new Set(data.cohort_points.map(p=>p.family))];
const cohortColor=p=>cohortPalette[cohortFamilies.indexOf(p.family)%cohortPalette.length];
let selectedCohort=null;
let emphasizedCohortFamily=null;
function applyCohortEmphasis(){
 const family=emphasizedCohortFamily;
 for(const node of document.querySelectorAll('#cohort-scatter [data-cohort-group],#cohort-key [data-cohort-group]')){
  const dimmed=!!family&&node.dataset.cohortGroup!==family;node.style.opacity=dimmed?'.12':'1';node.classList.toggle('cohort-dimmed',dimmed);
  if(node.tagName==='BUTTON')node.setAttribute('aria-pressed',String(node.dataset.cohortGroup===family));
 }
 const frontier=$('cohort-scatter').querySelector('[data-cohort-frontier]');if(frontier)frontier.style.opacity=family?'.25':'1';
 const selected=data.cohort_points.find(p=>p.id===selectedCohort);
 $('cohort-highlight-note').textContent=family?(selected?selected.id+' selected · ':'')+family+' highlighted. Other training groups are dimmed; the recommendation still compares every displayed cohort.':'Select a point or color to highlight its training group.';
 $('cohort-isolate-group').disabled=!family||cohortFilterState('overview').group===family;
 $('cohort-clear-emphasis').disabled=!family;
}
function clearCohortEmphasis(){emphasizedCohortFamily=null;selectedCohort=null;document.querySelectorAll('#panel-overview [data-cohort]').forEach(e=>{e.setAttribute('stroke','#111e29');e.setAttribute('stroke-width','1.5')});applyCohortEmphasis()}

function cohortBestSummary(p){
 if(p.best_error_status==='historical_unfiltered')return 'Historical best error '+sci(p.best_error_raw)+' (unfiltered; grid dimensions unavailable)';
 if(!(p.best_error>0))return 'Best outside tiny grids unavailable: no eligible cases';
 return 'Best error outside tiny grids '+sci(p.best_error)+' over '+p.best_error_eligible_cases+' cases; '+p.best_error_excluded_cases+' tiny-grid cases excluded; original best over all tests '+sci(p.best_error_raw);
}
function cohortSelection(p,{select=true}={}){if(select){selectedCohort=p.id;emphasizedCohortFamily=p.family;applyCohortEmphasis()}document.querySelectorAll('#panel-overview [data-cohort]').forEach(e=>{e.setAttribute('stroke',e.dataset.cohort===p.id?'#fff':'#111e29');e.setAttribute('stroke-width',e.dataset.cohort===p.id?'3':'1.5')});const box=$('cohort-detail');box.replaceChildren();const strong=document.createElement('strong');strong.textContent=p.id+' · '+p.name;box.append(strong,document.createTextNode(' — trained '+p.trained_at.replace('T',' ').slice(0,19)+' UTC. '+cohortBestSummary(p)+'; worst error '+sci(p.worst_error)+'; median CPU inference '+p.inference_median_ms.toFixed(3)+' ms. Timing population: '+(p.comparison_population||'historical original capture')+'. '+p.policy+'. '+(p.proposal_network_evaluations??'Unrecorded')+' fine CNN correction stages per pressure invocation; no simulation steps generated. '+(p.initialization_description||'')+'. '+(p.measurement_scope==='complete-hybrid-policy'?'Numerical pressure stages and learned correction; complete policy measured':data.solver_comparison?.classification[p.id]?.reason||'')+'. '+(p.diagnostic_scope?p.diagnostic_scope+' ':'')));for(const [title,path] of [['Accuracy evidence',p.source],['Timing evidence',p.timing_source],['Training timestamp',p.training_source]]){if(!path)continue;const a=document.createElement('a');a.textContent=title;a.href=path;box.append(a,document.createTextNode(' · '))}}
function cohortDot(svg,p,x,y,fill,metric){const label=p.id+' · '+p.name+'; '+metric+'; trained '+p.trained_at;const dot=el('circle',{cx:x,cy:y,r:6,fill,stroke:p.id===selectedCohort?'#fff':'#111e29','stroke-width':p.id===selectedCohort?3:1.5,tabindex:0,class:'comparison-point','data-cohort':p.id,'data-cohort-measurement':p.comparison_id||p.id,'data-cohort-group':p.family,'aria-label':label});dot.append(el('title',{},label));dot.addEventListener('pointerenter',e=>{showTip(e,label,dot);cohortSelection(p,{select:false})});dot.addEventListener('focus',e=>{showTip(e,label,dot);cohortSelection(p)});dot.addEventListener('click',()=>cohortSelection(p));dot.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();cohortSelection(p)}});for(const event of ['pointerleave','blur'])dot.addEventListener(event,()=>$('tooltip').hidden=true);svg.append(dot)}
function powerTicks(min,max){const ticks=[];for(let e=Math.ceil(min);e<=Math.floor(max);e++)ticks.push(10**e);return ticks}
function cohortPopulationKey(p){return p.cases===128&&p.timing_cases?.every(c=>c.split==='validation')?p.timing_cases.map(c=>JSON.stringify([c.trajectory_id,c.native_frame??c.frame,c.counts_xyz,c.cell_size_m])).sort().join('|'):null}
const matchedPopulationKey=cohortPopulationKey(data.cohort_points.find(p=>p.cases===128&&p.timing_cases?.every(c=>c.split==='validation')));
function tradeoffAnalysis(points,weight=Number($('tradeoff-weight').value)/100){return Evaluation.analyze(points,weight)}
@@COMPARISON_JS@@
function solverOriginalPoints(kind){return data.cohort_points.filter(p=>data.solver_comparison?.classification[p.id]?.kind===kind)}
function partialComparisonPoints(){const mode=$('partial-measurement').value;return mode==='complete-v6'?(data.solver_comparison?.hybrid_v6_policies||[]):mode==='complete'?(data.solver_comparison?.hybrid_policies||[]):solverOriginalPoints('hybrid')}
function updatePartialComparison(){
 const historical=$('partial-measurement').value==='proposal';$('partial-cohort-filter').querySelector('[value="frontier"]').disabled=historical;$('partial-tradeoff-weight').disabled=historical;if(historical&&$('partial-cohort-filter').value==='frontier')$('partial-cohort-filter').value='all';
 $('partial-cohort-filter').querySelector('[value="matched"]').textContent=$('partial-measurement').value==='complete-v6'?'Matched 20 native requests':'Matched 128-state validation';
 $('partial-measurement-note').textContent=$('partial-measurement').value==='complete-v6'?'Complete V6 pressure policies on 20 consumed native requests, 32³–128³ and rectangular grids; CPU 23, one thread, warmup and three repetitions. All numerical, neural, stopping and common returned-field gate costs count. Numerical-only and inactive-CNN controls remain on Pressure methods VI. This is a separate population from V5 historical128.':$('partial-measurement').value==='complete'?'Complete pressure-policy comparison: the same 128 saved requests, CPU 23, one thread, warmup and three repetitions. All numerical, neural, conversion, stopping and common returned-pressure gate costs count. Fluid integration and GUI are excluded. Multiple policies can share a cohort.':'Original captured hybrid proposals: numerical pressure stages and learned corrections are included. Timings exclude independent verification and later adaptive repetition. This protocol is separate from the complete-policy measurements.';
 renderPartialCohortPlots();
}

let partialSelectedCohort=null;
let partialEmphasizedCohortFamily=null;
function applyPartialCohortEmphasis(){
 const family=partialEmphasizedCohortFamily;
 for(const node of document.querySelectorAll('#partial-cohort-scatter [data-cohort-group],#partial-cohort-key [data-cohort-group]')){
  const dimmed=!!family&&node.dataset.cohortGroup!==family;node.style.opacity=dimmed?'.12':'1';node.classList.toggle('partial-cohort-dimmed',dimmed);
  if(node.tagName==='BUTTON')node.setAttribute('aria-pressed',String(node.dataset.cohortGroup===family));
 }
 const frontier=$('partial-cohort-scatter').querySelector('[data-cohort-frontier]');if(frontier)frontier.style.opacity=family?'.25':'1';
 const selected=data.cohort_points.find(p=>p.id===partialSelectedCohort);
 $('partial-cohort-highlight-note').textContent=family?(selected?selected.id+' selected · ':'')+family+' highlighted. Other training groups are dimmed; the recommendation still compares every displayed cohort.':'Select a point or color to highlight its training group.';
 $('partial-cohort-isolate-group').disabled=!family||cohortFilterState('partial').group===family;
 $('partial-cohort-clear-emphasis').disabled=!family;
}
function clearPartialCohortEmphasis(){partialEmphasizedCohortFamily=null;partialSelectedCohort=null;document.querySelectorAll('#panel-partial [data-cohort]').forEach(e=>{e.setAttribute('stroke','#111e29');e.setAttribute('stroke-width','1.5')});applyPartialCohortEmphasis()}

function partialCohortSelection(p,{select=true}={}){if(select){partialSelectedCohort=p.id;partialEmphasizedCohortFamily=p.family;applyPartialCohortEmphasis()}document.querySelectorAll('#panel-partial [data-cohort]').forEach(e=>{e.setAttribute('stroke',e.dataset.cohort===p.id?'#fff':'#111e29');e.setAttribute('stroke-width',e.dataset.cohort===p.id?'3':'1.5')});const box=$('partial-cohort-detail');box.replaceChildren();const strong=document.createElement('strong');strong.textContent=p.id+' · '+p.name;box.append(strong,document.createTextNode(' — trained '+p.trained_at.replace('T',' ').slice(0,19)+' UTC. '+cohortBestSummary(p)+'; worst error '+sci(p.worst_error)+'; median CPU inference '+p.inference_median_ms.toFixed(3)+' ms. Timing population: '+(p.comparison_population||'historical original capture')+'. '+p.policy+'. '+(p.proposal_network_evaluations??'Unrecorded')+' fine CNN correction stages per pressure invocation; no simulation steps generated. '+(p.initialization_description||'')+'. '+(p.measurement_scope==='complete-hybrid-policy'?'Numerical pressure stages and learned correction; complete policy measured':data.solver_comparison?.classification[p.id]?.reason||'')+'. '+(p.diagnostic_scope?p.diagnostic_scope+' ':'')));for(const [title,path] of [['Accuracy evidence',p.source],['Timing evidence',p.timing_source],['Training timestamp',p.training_source]]){if(!path)continue;const a=document.createElement('a');a.textContent=title;a.href=path;box.append(a,document.createTextNode(' · '))}}
function partialCohortDot(svg,p,x,y,fill,metric){const label=p.id+' · '+p.name+'; '+metric+'; trained '+p.trained_at;const dot=el('circle',{cx:x,cy:y,r:6,fill,stroke:p.id===partialSelectedCohort?'#fff':'#111e29','stroke-width':p.id===partialSelectedCohort?3:1.5,tabindex:0,class:'comparison-point','data-cohort':p.id,'data-cohort-measurement':p.comparison_id||p.id,'data-cohort-group':p.family,'aria-label':label});dot.append(el('title',{},label));dot.addEventListener('pointerenter',e=>{showTip(e,label,dot);partialCohortSelection(p,{select:false})});dot.addEventListener('focus',e=>{showTip(e,label,dot);partialCohortSelection(p)});dot.addEventListener('click',()=>partialCohortSelection(p));dot.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();partialCohortSelection(p)}});for(const event of ['pointerleave','blur'])dot.addEventListener(event,()=>$('tooltip').hidden=true);svg.append(dot)}
function renderPartialCohortPlots(){
 const pool=partialComparisonPoints().filter(p=>cohortMatchesSelection(p,'partial'));
 const frontier=new Set(tradeoffAnalysis(pool.filter(p=>p.qualified===null||p.qualified===undefined||p.qualified===p.cases),Number($('partial-tradeoff-weight').value)/100).frontier);
 const points=pool.filter(p=>$('partial-cohort-filter').value==='frontier'?frontier.has(p):$('partial-cohort-filter').value==='matched'?(!!p.comparison_population||cohortPopulationKey(p)===matchedPopulationKey):true);
 if(partialSelectedCohort&&!points.some(p=>p.id===partialSelectedCohort))partialSelectedCohort=null;
 if(partialEmphasizedCohortFamily&&!points.some(p=>p.family===partialEmphasizedCohortFamily))partialEmphasizedCohortFamily=null;
 const timed=points.filter(p=>p.inference_median_ms>0&&p.worst_error>0),ordered=points.filter(p=>p.trained_at&&p.worst_error>0).sort((a,b)=>a.order-b.order||(a.comparison_id||a.id).localeCompare(b.comparison_id||b.id));
 if(!points.length){
  for(const id of ['partial-cohort-scatter','partial-cohort-chronology','partial-cohort-key','partial-cohort-values'])$(id).replaceChildren();
  text($('partial-cohort-scatter'),580,240,'No measurements match. Change Measurements or broaden the cohort selection.',{'text-anchor':'middle','font-size':16});
  text($('partial-cohort-chronology'),580,200,'No cohorts match this selection.',{'text-anchor':'middle','font-size':16});
  $('partial-cohort-coverage').textContent='0 trained cohorts match the range, training group and chart subset.';
  $('partial-tradeoff-count').textContent='0';$('partial-tradeoff-weight-value').textContent=$('partial-tradeoff-weight').value+'%';$('partial-tradeoff-finding').textContent='No matched checkpoints available for a recommendation.';
  $('partial-cohort-detail').textContent='Broaden the cohort selection to view measurements.';applyPartialCohortEmphasis();return;
 }
 $('partial-cohort-detail').textContent='Click or focus a cohort to highlight its group; hover to preview measurements.';
 $('partial-cohort-coverage').textContent=new Set(points.map(p=>p.weights_sha256)).size+' trained checkpoints · '+timed.length+' timed measurements · '+ordered.length+' training-order markers. Only the selected partial-AI measurement protocol is ranked here.';
 const s=$('partial-cohort-scatter');s.replaceChildren();const left=104,right=1100,top=40,bottom=480;
 const minX=Math.floor(Math.log10(Math.min(...timed.map(p=>p.inference_median_ms))))-.15,maxX=Math.ceil(Math.log10(Math.max(...timed.map(p=>p.inference_median_ms))))+.1;
 const minY=Math.floor(Math.log10(Math.min(...timed.map(p=>p.worst_error))))-.3,maxY=Math.ceil(Math.log10(Math.max(...timed.map(p=>p.worst_error))))+.25;
 const x=v=>left+(Math.log10(v)-minX)/(maxX-minX)*(right-left),y=v=>bottom-(Math.log10(v)-minY)/(maxY-minY)*(bottom-top);
 const tradeoff=tradeoffAnalysis(timed.filter(p=>p.qualified===null||p.qualified===undefined||p.qualified===p.cases),Number($('partial-tradeoff-weight').value)/100),winner=tradeoff.ranked[0],mostAccurate=[...tradeoff.ranked].sort((a,b)=>a.worst_error-b.worst_error)[0];
 $('partial-tradeoff-weight-value').textContent=Math.round(tradeoff.weight*100)+'%';
 $('partial-tradeoff-count').textContent=tradeoff.ranked.length;
 $('partial-tradeoff-finding').textContent=winner?(tradeoff.weight===.5?'Best balanced choice: ':'Best choice at this weighting: ')+winner.id+' · '+winner.name+' — worst error '+sci(winner.worst_error)+' at '+winner.inference_median_ms.toFixed(3)+' ms. '+(winner===mostAccurate?'This is also the lowest-error checkpoint.':mostAccurate.name+' has the lowest error ('+sci(mostAccurate.worst_error)+'), but takes '+(mostAccurate.inference_median_ms/winner.inference_median_ms).toFixed(2)+'× as long ('+mostAccurate.inference_median_ms.toFixed(3)+' ms).')+' '+tradeoff.frontier.length+' choices lie on the speed/accuracy frontier.':'No complete matched policy measurements in this selection qualify for a recommendation.';
 const xTicks=[];for(let exponent=Math.floor(minX);exponent<=Math.ceil(maxX);exponent++)for(const multiplier of [1,2,5]){const value=multiplier*10**exponent;if(Math.log10(value)>=minX&&Math.log10(value)<=maxX)xTicks.push(value)}
 for(const t of xTicks){line(s,x(t),top,x(t),bottom);text(s,x(t),bottom+24,t>=1?t.toFixed(0):t.toPrecision(1),{'text-anchor':'middle','font-size':12})}
 for(const t of powerTicks(minY,maxY)){line(s,left,y(t),right,y(t));text(s,left-14,y(t)+4,'1e'+Math.round(Math.log10(t)),{'text-anchor':'end','font-size':12})}
 if(minY<=-6){line(s,left,y(1e-6),right,y(1e-6),color.blue,'5 5');text(s,right,y(1e-6)-8,'Model equation target · 1e−6',{'text-anchor':'end',fill:color.blue,'font-size':11})}else text(s,right,top-12,'Model target 1e−6 lies below the displayed range',{'text-anchor':'end',fill:color.blue,'font-size':11});
 text(s,(left+right)/2,540,s.id==='partial-cohort-scatter'?'Median CPU pressure-policy time (ms) →':'Median CPU inference time per proposal (ms) →',{'text-anchor':'middle','font-size':14});text(s,25,(top+bottom)/2,'Worst relative equation error →',{transform:'rotate(-90 25 '+((top+bottom)/2)+')','text-anchor':'middle','font-size':14});
 s.append(el('path',{d:tradeoff.frontier.map((p,i)=>(i?'L':'M')+x(p.inference_median_ms)+','+y(p.worst_error)).join(' '),fill:'none',stroke:color.blue,'stroke-width':2,'stroke-dasharray':'6 5','aria-label':'Speed and accuracy tradeoff frontier','data-cohort-frontier':''}));
 for(const p of timed){partialCohortDot(s,p,x(p.inference_median_ms),y(p.worst_error),cohortColor(p),'worst error '+sci(p.worst_error)+'; inference '+p.inference_median_ms.toFixed(3)+' ms');if($('partial-cohort-labels').checked)text(s,x(p.inference_median_ms)+9,y(p.worst_error)-7,p.id,{fill:cohortColor(p),'font-size':10,'data-cohort-group':p.family})}
 if(winner){const wx=x(winner.inference_median_ms),wy=y(winner.worst_error);s.append(el('path',{d:'M'+wx+','+(wy-12)+' L'+(wx+12)+','+wy+' L'+wx+','+(wy+12)+' L'+(wx-12)+','+wy+' Z',fill:'none',stroke:'#fff','stroke-width':2,'aria-label':'Recommended checkpoint '+winner.id,'data-cohort-group':winner.family}));text(s,wx+17,wy+22,winner.id+' · '+(tradeoff.weight===.5?'best balance':'selected weighting'),{fill:'#fff','font-size':12,'data-cohort-group':winner.family})};
 const c=$('partial-cohort-chronology');c.replaceChildren();const cb=350,ct=35;
 const cmin=Math.floor(Math.log10(Math.min(1e-6,...ordered.filter(p=>p.best_error>0).map(p=>p.best_error))))-.2,cmax=Math.ceil(Math.log10(Math.max(...ordered.map(p=>p.worst_error))))+.2;
 const cx=i=>ordered.length===1?(left+right)/2:left+i*(right-left)/Math.max(1,ordered.length-1),cy=v=>cb-(Math.log10(v)-cmin)/(cmax-cmin)*(cb-ct);
 for(const t of powerTicks(cmin,cmax)){line(c,left,cy(t),right,cy(t));text(c,left-14,cy(t)+4,'1e'+Math.round(Math.log10(t)),{'text-anchor':'end','font-size':12})}
 line(c,left,cy(1e-6),right,cy(1e-6),color.blue,'5 5');text(c,right,cy(1e-6)-8,'Model target · 1e−6',{'text-anchor':'end',fill:color.blue,'font-size':10});
 for(const [key,col] of [['best_error',color.teal],['worst_error',color.amber]]){
  const kinds=key==='best_error'?['filtered','historical_unfiltered']:['all'];
  for(const kind of kinds){let path='',connected=false;ordered.forEach((p,i)=>{const match=p[key]>0&&(key!=='best_error'||(p.best_error_status==='historical_unfiltered')===(kind==='historical_unfiltered'));if(!match){connected=false;return}path+=(connected?'L':'M')+cx(i)+','+cy(p[key])+' ';connected=true});c.append(el('path',{d:path,fill:'none',stroke:col,'stroke-width':1.8,'stroke-dasharray':kind==='historical_unfiltered'?'5 4':'none','data-best-population':key==='best_error'?kind:'all-original-tests'}))}
  ordered.forEach((p,i)=>{if(p[key]>0)partialCohortDot(c,p,cx(i),cy(p[key]),col,key==='best_error'?cohortBestSummary(p):'worst error over all original tests '+sci(p[key]))});
 }
 ordered.forEach((p,i)=>text(c,cx(i),cb+25,p.id,{'text-anchor':'middle','font-size':9,transform:'rotate(-50 '+cx(i)+' '+(cb+25)+')'}));
 text(c,(left+right)/2,425,'Training order → earliest on the left, latest on the right',{'text-anchor':'middle','font-size':14});text(c,25,(ct+cb)/2,'Relative equation error →',{transform:'rotate(-90 25 '+((ct+cb)/2)+')','text-anchor':'middle','font-size':14});
 $('partial-cohort-key').replaceChildren();for(const family of [...new Set(points.map(p=>p.family))]){const button=document.createElement('button'),i=document.createElement('i');button.type='button';button.dataset.cohortGroup=family;button.setAttribute('aria-label','Highlight '+family);i.style.background=cohortPalette[cohortFamilies.indexOf(family)%cohortPalette.length];button.append(i,document.createTextNode(family));button.addEventListener('click',()=>{partialSelectedCohort=null;partialEmphasizedCohortFamily=partialEmphasizedCohortFamily===family?null:family;document.querySelectorAll('#panel-partial [data-cohort]').forEach(e=>{e.setAttribute('stroke','#111e29');e.setAttribute('stroke-width','1.5')});applyPartialCohortEmphasis()});$('partial-cohort-key').append(button)}
 applyPartialCohortEmphasis();
 $('partial-cohort-values').replaceChildren();for(const p of points){const tr=document.createElement('tr');for(const value of [p.id+' · '+p.name,p.trained_at.replace('T',' ').slice(0,19),p.best_error_status==='historical_unfiltered'?'Historical, unfiltered':p.best_error===null?'No eligible cases':sci(p.best_error),sci(p.best_error_raw),sci(p.worst_error),p.inference_median_ms.toFixed(3)+' ms',p.cases,p.best_error_eligible_cases===null?'Grid dimensions unavailable':p.best_error_eligible_cases+' / '+p.best_error_excluded_cases]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td');for(const [label,href] of [['Accuracy',p.source],['Time',p.timing_source],['Training',p.training_source]]){if(!href)continue;const a=document.createElement('a');a.href=href;a.textContent=label;td.append(a,document.createTextNode(' '))}tr.append(td);$('partial-cohort-values').append(tr)}
}

