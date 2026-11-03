// Saved method-study policies use a separate timing scope from original proposals.
function methodsModelAllowed(id){return cohortMatchesScope(data.cohort_points.find(p=>p.id===id),$('methods-scope').value,'methods')&&($('methods-model').value==='all'||$('methods-model').value===id)}
function setupMethodsStudy(){
 const study=data.methods_study,previous=$('methods-model').value,ids=new Set([...(study?.models||[]),'C37','C44','C53']);
 const pool=data.cohort_points.filter(p=>ids.has(p.id)&&cohortMatchesScope(p,$('methods-scope').value,'methods'));
 $('methods-model').replaceChildren(new Option('All models in this selection','all'));
 for(const p of pool)$('methods-model').append(new Option(p.id+' · '+p.name,p.id));
 $('methods-model').value=previous==='all'||pool.some(p=>p.id===previous)?previous:'all';
 renderMethodsStudy();
}
function methodsDetail(row){
 const label=`${row.model} · ${row.arm} · ${row.ordering||row.policy||'neural'} · ${row.passes??$('methods-passes').value} neural passes · ${row.population} · worst equation residual ${sci(row.worst_relative_equation_residual)} · ${row.qualified}/${row.cases} equation checks · median policy time ${row.median_policy_ms.toFixed(3)} ms`;
 return label;
}
function methodsDot(svg,x,y,row){
 const label=methodsDetail(row),dot=historyDot(svg,x,y,passModelColor(row.model),label);dot.dataset.methodsModel=row.model;
 for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>{const box=$('methods-detail'),link=document.createElement('a');link.href=row.source;link.textContent='Source evidence';box.replaceChildren(document.createTextNode(label+' · '),link)});
 return dot;
}
function renderMethodsCategories(id,rows,categories,axisLabel){
 const svg=$(id);svg.replaceChildren();const valid=rows.filter(r=>r.worst_relative_equation_residual>0);
 if(!valid.length){text(svg,580,220,'No saved results match this selection.',{'text-anchor':'middle'});return}
 const axes=passAxes(svg,[0,Math.max(1,categories.length-1)],passBounds(valid.map(r=>r.worst_relative_equation_residual),true),false,true,axisLabel);
 for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
 categories.forEach(([population,label],i)=>{line(svg,axes.x(i),axes.top,axes.x(i),axes.bottom);text(svg,axes.x(i),445,label,{'text-anchor':'middle'})});
 for(const model of [...new Set(valid.map(r=>r.model))]){
  const selected=categories.map(([population],i)=>({row:valid.find(r=>r.model===model&&r.population===population),i})).filter(r=>r.row);
  let d='',previous=-2;for(const {row,i}of selected){d+=(i===previous+1?'L':'M')+axes.x(i)+','+axes.y(row.worst_relative_equation_residual)+' ';previous=i}
  svg.append(el('path',{d,fill:'none',stroke:passModelColor(model),'stroke-width':2,'data-methods-line':model}));
  for(const {row,i}of selected)methodsDot(svg,axes.x(i),axes.y(row.worst_relative_equation_residual),row);
 }
}
function renderMethodsStudy(){
 const study=data.methods_study,svg=$('methods-quiet-chart');svg.replaceChildren();
 if(!study){$('methods-coverage').textContent='No completed pressure-method campaign is available.';return}
 const passes=Number($('methods-passes').value),quietPass=$('methods-quiet-passes').value;
 const quiet=study.quiet.filter(r=>methodsModelAllowed(r.model)&&(quietPass==='all'||r.passes===Number(quietPass)));
 const domain=study.summaries.filter(r=>methodsModelAllowed(r.model)&&r.arm!=='teacher-coarse'&&r.policy==='neural-'+passes).map(r=>({...r,passes}));
 const spacing=study.spacing.filter(r=>methodsModelAllowed(r.model)&&r.arm!=='teacher-coarse'&&r.passes===passes);
 $('methods-coverage').textContent=`${study.models.length} distinct checkpoints · 12 completed fits · ${quiet.length} quiet policy settings · ${new Set(domain.map(r=>r.model)).size} models with domain results at ${passes} passes. Teacher-coarse is shown once, as C78.`;
 $('methods-timing-note').textContent=study.timing_scope+' Quiet measurements are plotted only against quiet measurements. Domain/spacing tables retain initial campaign timings; compare their errors within each named population.';
 const valid=quiet.filter(r=>r.median_policy_ms>0&&r.worst_relative_equation_residual>0);
 if(valid.length){
  const axes=passAxes(svg,passBounds(valid.map(r=>r.median_policy_ms),true),passBounds(valid.map(r=>r.worst_relative_equation_residual),true),true,true,'Median complete pressure-policy time · milliseconds');
  const frontier=valid.filter(p=>!valid.some(q=>q.median_policy_ms<=p.median_policy_ms&&q.worst_relative_equation_residual<=p.worst_relative_equation_residual&&(q.median_policy_ms<p.median_policy_ms||q.worst_relative_equation_residual<p.worst_relative_equation_residual))).sort((a,b)=>a.median_policy_ms-b.median_policy_ms);
  svg.append(el('path',{d:frontier.map((r,i)=>(i?'L':'M')+axes.x(r.median_policy_ms)+','+axes.y(r.worst_relative_equation_residual)).join(' '),fill:'none',stroke:'#f2f5f7','stroke-width':2,'stroke-dasharray':'5 5'}));
  for(const group of [...new Set(valid.map(r=>r.model+'|'+r.ordering))]){
   const rows=valid.filter(r=>r.model+'|'+r.ordering===group).sort((a,b)=>a.passes-b.passes);
   svg.append(el('path',{d:rows.map((r,i)=>(i?'L':'M')+axes.x(r.median_policy_ms)+','+axes.y(r.worst_relative_equation_residual)).join(' '),fill:'none',stroke:passModelColor(rows[0].model),'stroke-width':1.5,...(rows[0].ordering==='after'?{'stroke-dasharray':'3 4'}:rows[0].ordering==='before'?{'stroke-dasharray':'8 4'}:{})}));
  }
  for(const row of valid){const x=axes.x(row.median_policy_ms),y=axes.y(row.worst_relative_equation_residual);methodsDot(svg,x,y,row);text(svg,x+7,y-7,String(row.passes),{'font-size':10,fill:passModelColor(row.model)})}
 }else text(svg,580,220,'No quiet policy measurements match this selection.',{'text-anchor':'middle'});
 const fastest=[...valid].filter(r=>r.qualified===r.cases).sort((a,b)=>a.median_policy_ms-b.median_policy_ms)[0];
 $('methods-best').textContent=fastest?`Fastest displayed policy meeting the equation target on all 128 requests: ${fastest.model} / ${fastest.ordering} / ${fastest.passes} neural passes, ${fastest.median_policy_ms.toFixed(3)} ms; worst residual ${sci(fastest.worst_relative_equation_residual)}. This is a prepared-request result on the tested uniform grids.`:'No displayed quiet policy meets the equation target on every request.';
 renderMethodsCategories('methods-domain-chart',domain,[['consumed128','Standard · 128'],['domain32','32³ · 4'],['domain64','64³ · 4'],['domain-odd','Rectangular · 4']],'Evaluation population · counts show saved pressure requests');
 renderMethodsCategories('methods-spacing-chart',spacing,[['cpm4','4'],['cpm8','8'],['cpm16','16'],['cpm32','32']],'Cells per metre · 6 held-out development requests per spacing');
 $('methods-key').replaceChildren();for(const model of [...new Set([...quiet,...domain,...spacing].map(r=>r.model))]){const label=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=passModelColor(model);label.append(swatch,model+' · '+data.cohort_points.find(p=>p.id===model)?.name);$('methods-key').append(label)}
 const body=$('methods-values');body.replaceChildren();for(const row of [...quiet.map(r=>({...r,population:'Quiet · '+r.population})),...domain,...spacing]){
  const tr=document.createElement('tr');for(const value of [row.model+' · '+row.arm+' / '+(row.ordering||row.policy||'neural')+' / '+(row.passes??passes)+' passes',row.population,sci(row.worst_relative_equation_residual),row.qualified+'/'+row.cases,row.median_policy_ms.toFixed(3)+' ms']){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=row.source;a.textContent='Saved results';td.append(a);tr.append(td);body.append(tr);
 }
 const native=study.native.filter(r=>methodsModelAllowed(r.model)),rejected=native.filter(r=>r.status==='pressure_rejection');
 $('methods-native-note').textContent=`${native.length} native settings in this selection; ${native.filter(r=>r.accepted_steps>0).length} have measured drift curves in Drift over time. ${rejected.length} stopped on pressure rejection; unadvanced attempts receive no invented drift value. Policies, native builds and scene setups remain separate when averaging or ranking.`;
 $('methods-limits').replaceChildren();for(const limit of study.limits){const li=document.createElement('li');li.textContent=limit;$('methods-limits').append(li)}
 $('methods-outcomes').replaceChildren();for(const r of study.cohorts){const tr=document.createElement('tr');for(const v of [r.id+' · '+r.name,r.outcome]){const td=document.createElement('td');td.style.whiteSpace='normal';td.textContent=v;tr.append(td)}$('methods-outcomes').append(tr)}
}
$('methods-scope').addEventListener('change',setupMethodsStudy);
for(const id of ['methods-model','methods-passes','methods-quiet-passes'])$(id).addEventListener('change',renderMethodsStudy);
$('methods-show-drift').addEventListener('click',()=>{
 const t=(data.drift_traces||[]).find(t=>t.chart_study==='pressure-methods-v1'&&t.scene.seed===20261103&&t.points.length===497&&Number(driftCohorts.get(t.weights_sha256)?.id.slice(1))>=78);if(!t)return;
 $('cohorts-drift-reset').click();$('cohorts-drift-from').value='C78';$('cohorts-drift-from').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();$('drift-summary').value='single';$('drift-top').value='all';$('drift-scene').value=driftSceneKey(t);driftHidden.clear();renderDrift();$('tab-drift').click();
});

// Second-campaign measurements retain distinct validation and runtime scopes.
function methodsV2Allowed(row){
 if(!row.model)return $('methodsv2-controls').checked;
 const point=driftCohorts.get(data.cohort_points.find(p=>p.id===row.model)?.weights_sha256);
 return cohortMatchesScope(point,$('methodsv2-scope').value,'methodsv2')&&($('methodsv2-model').value==='all'||$('methodsv2-model').value===row.model);
}
function methodsV2Color(row){return row.model?passModelColor(row.model):row.label?.includes('spectral')||row.method?.includes('spectral')?color.native:color.blue}
function setupMethodsV2(){
 const study=data.methods_v2,prior=$('methodsv2-model').value,ids=new Set([...(study?.models||[]),...(study?.controls||[])]);
 const pool=data.cohort_points.filter(p=>ids.has(p.id)&&cohortMatchesScope(p,$('methodsv2-scope').value,'methodsv2'));
 $('methodsv2-model').replaceChildren(new Option('All models in this selection','all'));
 for(const p of pool)$('methodsv2-model').append(new Option(p.id+' · '+p.name,p.id));
 $('methodsv2-model').value=prior==='all'||pool.some(p=>p.id===prior)?prior:'all';
 if(!$('methodsv2-population').options.length)for(const pop of ['all-development','consumed128','new-rectangular','new-small','old-cube32','old-cube64','old-rectangular'])$('methodsv2-population').append(new Option(pop,pop));
 renderMethodsV2();
}
function methodsV2Dot(svg,x,y,row,label){
 const dot=historyDot(svg,x,y,methodsV2Color(row),label);dot.dataset.methodsv2Model=row.model||'numerical-control';
 for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>{const a=document.createElement('a');a.href=row.source;a.textContent='Source evidence';$('methodsv2-detail').replaceChildren(document.createTextNode(label+' · '),a)});
 return dot;
}
function methodsV2Categories(id,rows,categories){
 const svg=$(id),metric=$('methodsv2-metric').value;svg.replaceChildren();const valid=rows.filter(r=>Number.isFinite(r[metric])&&r[metric]>0);
 if(!valid.length){text(svg,580,220,'No saved measurements match this selection.',{'text-anchor':'middle'});return}
 const axes=passAxes(svg,[0,categories.length-1],passBounds(valid.map(r=>r[metric]),true),false,true,'Evaluation population · distinct saved request sets');
 for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
 categories.forEach(([pop,label],i)=>{line(svg,axes.x(i),axes.top,axes.x(i),axes.bottom);text(svg,axes.x(i),445,label,{'text-anchor':'middle','font-size':10})});
 for(const model of [...new Set(valid.map(r=>r.model))]){
  const selected=categories.map(([pop],i)=>({row:valid.find(r=>r.model===model&&r.population===pop),i})).filter(r=>r.row);
  let d='',last=-2;for(const {row,i}of selected){d+=(i===last+1?'L':'M')+axes.x(i)+','+axes.y(row[metric])+' ';last=i}
  svg.append(el('path',{d,fill:'none',stroke:passModelColor(model),'stroke-width':2,'data-methodsv2-line':model}));
  for(const {row,i}of selected)methodsV2Dot(svg,axes.x(i),axes.y(row[metric]),row,`${model} · ${row.arm} · ${row.population} · worst ${metric==='worst_relative_residual'?'equation residual':'mean-free pressure error'} ${sci(row[metric])} · ${row.qualified}/${row.cases} qualified · ${row.passes} passes`);
 }
}
function renderMethodsV2Application(rows){
 const svg=$('methodsv2-application-chart');svg.replaceChildren();const size=Number($('methodsv2-grid').value),all=data.methods_v2.application.filter(r=>r.size===size),baseline=all.find(r=>r.method==='native');
 const valid=rows.filter(r=>r.timing_eligible&&r.median_frame_ms>0);
 if(!baseline||!valid.length){text(svg,580,220,'No complete application measurements match this selection.',{'text-anchor':'middle'});$('methodsv2-application-note').textContent='Rejected settings are excluded from the latency ranking.';return}
 const left=295,right=1110,top=32,bottom=410,max=Math.max(1,...valid.map(r=>r.median_frame_ms/baseline.median_frame_ms))*1.1,x=v=>left+v/max*(right-left),height=Math.min(40,(bottom-top)/valid.length);
 for(let i=0;i<=5;i++){const value=max*i/5;line(svg,x(value),top,x(value),bottom);text(svg,x(value),445,value.toFixed(2)+'×',{'text-anchor':'middle'})}
 line(svg,x(1),top,x(1),bottom,color.native,'5 5');
 valid.forEach((r,i)=>{const y=top+(i+.5)*(bottom-top)/valid.length,ratio=r.median_frame_ms/baseline.median_frame_ms,label=`${r.model?r.model+' · ':''}${r.method} · ${size}³ · ${r.median_frame_ms.toFixed(3)} ms/frame · ${ratio.toFixed(3)}× Rheon time · ${r.complete_repetitions}/${r.repetitions} complete repetitions · median ${r.median_network_calls} network calls`;
  text(svg,left-12,y+4,(r.model?r.model+' · ':'')+r.method,{'text-anchor':'end','font-size':11});
  svg.append(el('rect',{x:left,y:y-height*.32,width:x(ratio)-left,height:height*.64,fill:methodsV2Color(r),'fill-opacity':.65,'data-methodsv2-application':r.method}));methodsV2Dot(svg,x(ratio),y,r,label);
 });
 text(svg,(left+right)/2,480,'Simulation time ÷ matched Rheon time · lower is faster',{'text-anchor':'middle','font-size':12});
 $('methodsv2-application-note').textContent=`Matched Rheon baseline: ${baseline.median_frame_ms.toFixed(3)} ms/frame on ${size}³. ${valid.length} complete policy settings plotted; ${rows.filter(r=>!r.timing_eligible).length} rejected settings excluded. Native uses 1e−8 solver tolerance; native-loose uses 1e−6. Candidate host gates remain the recorded physical checks. This is the measured CPU/Python integration.`;
}
function methodsV2ScalingRows(){
 const mode=$('methodsv2-scaling-scope').value,cpm=Number($('methodsv2-spacing').value);
 return(data.methods_v2?.scaling||[]).filter(r=>methodsV2Allowed(r)&&r.cells_per_metre===cpm&&(mode==='manufactured'?r.scope==='manufactured equation':mode==='cold'?r.scope==='native cold first step':r.scope==='native: one warm-up + one predicted step'));
}
function renderMethodsV2Scaling(){
 const rows=methodsV2ScalingRows(),valid=rows.filter(r=>Number.isFinite(r.relative_residual)&&r.relative_residual>0),svg=$('methodsv2-scaling-chart');svg.replaceChildren();
 if(!valid.length)text(svg,580,220,'No recorded scaling probes match this selection.',{'text-anchor':'middle'});
 else{const axes=passAxes(svg,[64,256],passBounds(valid.map(r=>r.relative_residual),true),false,true,'Cubic grid side · cells');
  for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
  for(const size of [64,128,256]){line(svg,axes.x(size),axes.top,axes.x(size),axes.bottom);text(svg,axes.x(size),445,String(size),{'text-anchor':'middle'})}
  for(const r of valid){const label=`${r.model?r.model+' · ':''}${r.case} · ${r.method} · ${r.size}³ at ${r.cells_per_metre} cells/metre · ${r.scope} · relative residual ${sci(r.relative_residual)} · ${r.qualified?'qualified':'failed qualification'}`,dot=methodsV2Dot(svg,axes.x(r.size),axes.y(r.relative_residual),r,label);if(!r.qualified){dot.setAttribute('fill','none');dot.setAttribute('stroke',methodsV2Color(r));dot.setAttribute('stroke-width','2')}dot.dataset.methodsv2Qualified=String(r.qualified)}
 }
 $('methodsv2-scaling-note').textContent=`${rows.length} probes selected; ${rows.filter(r=>r.qualified).length} qualified. Exact-zero or unavailable residuals remain in the table rather than being assigned a logarithmic coordinate. Native probes cover the recorded one/two-step windows, not a long-run scaling claim.`;
 const body=$('methodsv2-scaling-values');body.replaceChildren();for(const r of rows){const tr=document.createElement('tr');for(const value of [r.case+' / '+r.method,r.size+'³ / '+r.cells_per_metre+' cells/m',r.scope,Number.isFinite(r.relative_residual)?sci(r.relative_residual):'Unavailable',r.qualified?'Yes':'No',Number.isFinite(r.frame_seconds??r.seconds)?((r.frame_seconds??r.seconds)*1000).toFixed(3)+' ms':'Unavailable']){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=r.source;a.textContent='Probe';td.append(a);tr.append(td);body.append(tr)}
}
function renderMethodsV2(){
 const study=data.methods_v2,svg=$('methodsv2-quiet-chart');svg.replaceChildren();if(!study){$('methodsv2-coverage').textContent='No completed second campaign is available.';return}
 const passes=Number($('methodsv2-passes').value),metric=$('methodsv2-metric').value,pop=$('methodsv2-population').value;
 const quiet=study.quiet.filter(r=>methodsV2Allowed(r)&&r.population===pop),valid=quiet.filter(r=>r.median_ms>0&&r[metric]>0);
 const domain=study.summaries.filter(r=>methodsV2Allowed(r)&&r.passes===passes).map(r=>({...r,passes}));
 const blind=study.blind.filter(r=>methodsV2Allowed(r)&&r.passes===passes).map(r=>({...r,population:'blind16'}));
 const development=[...new Set(domain.map(r=>r.model))].map(model=>{const rows=domain.filter(r=>r.model===model);return{model,arm:rows[0].arm,population:'development156',passes,source:rows[0].source,cases:rows.reduce((n,r)=>n+r.cases,0),qualified:rows.reduce((n,r)=>n+r.qualified,0),worst_relative_residual:Math.max(...rows.map(r=>r.worst_relative_residual)),worst_pressure_relative_l2:Math.max(...rows.map(r=>r.worst_pressure_relative_l2))}}).filter(r=>r.cases===156);
 const compared=[...development,...blind];
 const application=study.application.filter(r=>methodsV2Allowed(r)&&r.size===Number($('methodsv2-grid').value));
 $('methodsv2-coverage').textContent=`${study.models.length} distinct new checkpoints from 12 fits · control small16 = C78 · ${new Set(domain.map(r=>r.model)).size} models with development data at ${passes} passes · ${blind.length} blind results · ${quiet.length} quiet policy settings on ${pop}.`;
 if(valid.length){const axes=passAxes(svg,passBounds(valid.map(r=>r.median_ms),true),passBounds(valid.map(r=>r[metric]),true),true,true,'Median complete prepared pressure-policy time · milliseconds');
  const frontier=valid.filter(p=>!valid.some(q=>q.median_ms<=p.median_ms&&q[metric]<=p[metric]&&(q.median_ms<p.median_ms||q[metric]<p[metric]))).sort((a,b)=>a.median_ms-b.median_ms);
  svg.append(el('path',{d:frontier.map((r,i)=>(i?'L':'M')+axes.x(r.median_ms)+','+axes.y(r[metric])).join(' '),fill:'none',stroke:'#f2f5f7','stroke-width':2,'stroke-dasharray':'5 5'}));
  for(const r of valid)methodsV2Dot(svg,axes.x(r.median_ms),axes.y(r[metric]),r,`${r.model?r.model+' · ':''}${r.label} · ${r.population} · ${r.cases} cases · worst ${sci(r[metric])} · ${r.median_ms.toFixed(3)} ms · ${r.qualified} qualified`);
 }else text(svg,580,220,'No quiet measurements match this selection.',{'text-anchor':'middle'});
 $('methodsv2-quiet-note').textContent='Quiet prepared-request timing includes complete declared numerical/neural policy and final checks. These times have a different scope from original-proposal inference and application frame time. Adaptive policies have variable network work.';
 methodsV2Categories('methodsv2-domain-chart',domain,[['consumed128','Original · 128'],['new-small','New small · 4'],['old-cube32','32³ · 4'],['old-cube64','64³ · 4'],['old-rectangular','Old rect. · 4'],['new-rectangular','New rect. · 12']]);
 methodsV2Categories('methodsv2-blind-chart',compared,[['development156','Development · 156'],['blind16','Blind · 16']]);
 renderMethodsV2Application(application);renderMethodsV2Scaling();
 $('methodsv2-key').replaceChildren();for(const model of [...new Set([...quiet,...domain,...blind].map(r=>r.model))]){const span=document.createElement('span'),i=document.createElement('i');i.style.background=model?passModelColor(model):color.native;span.append(i,model?model+' · '+data.cohort_points.find(p=>p.id===model)?.name:'Numerical controls');$('methodsv2-key').append(span)}
 const body=$('methodsv2-values');body.replaceChildren();const rows=[...quiet.map(r=>({...r,display:r.label,time:r.median_ms})),...domain.map(r=>({...r,display:r.arm+' / '+r.passes+' passes',time:r.median_policy_ms})),...blind.map(r=>({...r,display:r.arm+' / '+r.passes+' passes'}))];
 for(const r of rows){const tr=document.createElement('tr');for(const value of [(r.model?r.model+' · ':'')+r.display,r.population,Number.isFinite(r[metric])?sci(r[metric]):'Unavailable',r.qualified+'/'+r.cases,Number.isFinite(r.time)?r.time.toFixed(3)+' ms':'Not a timing comparison']){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=r.source;a.textContent='Saved results';td.append(a);tr.append(td);body.append(tr)}
 const native=study.native.filter(methodsV2Allowed);$('methodsv2-native-note').textContent=`${native.length} recorded dual-track simulation settings in scope, including ${(native.filter(r=>r.steps===7200)).length} two-minute settings. Available curves are in Drift over time; missing checkpoint rollouts stay unmeasured. Policies and native builds remain separate in aggregate scores.`;
}
$('methodsv2-scope').addEventListener('change',setupMethodsV2);
for(const id of ['methodsv2-model','methodsv2-passes','methodsv2-population','methodsv2-metric','methodsv2-grid','methodsv2-controls','methodsv2-scaling-scope','methodsv2-spacing'])$(id).addEventListener('change',renderMethodsV2);
$('methodsv2-show-drift').addEventListener('click',()=>{
 const t=(data.drift_traces||[]).find(t=>t.chart_study==='pressure-methods-v2'&&driftCohorts.get(t.weights_sha256)?.id==='C93'&&t.scene.steps===512);if(!t)return;
 $('cohorts-drift-reset').click();$('cohorts-drift-from').value='C53';$('cohorts-drift-from').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();$('drift-summary').value='single';$('drift-top').value='all';$('drift-scene').value=driftSceneKey(t);driftHidden.clear();renderDrift();$('tab-drift').click();
});

function methodsV3Allowed(row){
 if(!row.model)return $('methodsv3-controls').checked;
 const point=data.cohort_points.find(p=>p.id===row.model);
 return point&&cohortMatchesScope(point,$('methodsv3-scope').value,'methodsv3')&&($('methodsv3-model').value==='all'||$('methodsv3-model').value===row.model);
}
function setupMethodsV3(){
 const study=data.methods_v3,prior=$('methodsv3-model').value,ids=new Set([...(study?.models||[]),...(study?.controls||[])]);
 const pool=data.cohort_points.filter(p=>ids.has(p.id)&&cohortMatchesScope(p,$('methodsv3-scope').value,'methodsv3'));
 $('methodsv3-model').replaceChildren(new Option('All models in this selection','all'));
 for(const p of pool)$('methodsv3-model').append(new Option(p.id+' · '+p.name,p.id));
 $('methodsv3-model').value=pool.some(p=>p.id===prior)?prior:'all';renderMethodsV3();
}
function methodsV3Dot(svg,x,y,row,label){
 const stroke=row.model?passModelColor(row.model):color.native,dot=historyDot(svg,x,y,stroke,label);dot.dataset.methodsv3Model=row.model||'spectral';
 const failed=row.qualified!==undefined&&row.qualified!==(row.cases??true);
 if(failed){dot.setAttribute('fill','none');dot.setAttribute('stroke',stroke);dot.setAttribute('stroke-width','2')}
 dot.dataset.methodsv3Qualified=String(!failed);
 for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>{const a=document.createElement('a');a.href=row.source;a.textContent='Source evidence';$('methodsv3-detail').replaceChildren(document.createTextNode(label+' · '),a)});
 return dot;
}
function methodsV3NativeRows(){return(data.methods_v3?.rows||[]).filter(r=>r.stage===$('methodsv3-stage').value&&r.population==='native-'+r.stage&&r.max_passes===Number($('methodsv3-passes').value)&&methodsV3Allowed(r))}
function renderMethodsV3(){
 const study=data.methods_v3;if(!study)return;
 const blind=$('methodsv3-stage').value==='blind';$('methodsv3-passes').options[0].disabled=blind;if(blind)$('methodsv3-passes').value='16';
 const metric=$('methodsv3-metric').value,quiet=study.quiet.filter(methodsV3Allowed),native=methodsV3NativeRows(),value=r=>r[metric];
 $('methodsv3-coverage').textContent=quiet.length+' matched historical settings · '+native.length+' selected native requests. Cohort range and training group apply only to this tab. Hollow markers retain failed checks. No drift curves are inferred from the four-step host tests.';
 let svg=$('methodsv3-quiet-chart');svg.replaceChildren();
 if(quiet.length){const a=passAxes(svg,passBounds(quiet.map(r=>r.median_ms),true),passBounds(quiet.map(value),true),true,true,'Median prepared pressure policy time (ms)');
  for(const r of quiet)methodsV3Dot(svg,a.x(r.median_ms),a.y(value(r)),r,`${r.model||'Spectral'} · ${r.identifier} · historical128 · ${metric} ${sci(value(r))} · ${r.qualified}/${r.cases} qualified · ${r.median_ms.toFixed(3)} ms`);
 }else text(svg,580,220,'No recorded settings match this selection.',{'text-anchor':'middle'});
 svg=$('methodsv3-native-chart');svg.replaceChildren();
 const category=r=>r.counts_xyz[0]+'³ · '+r.cells_per_metre+' cells/m · '+r.context;
 const all=study.rows.filter(r=>r.stage===$('methodsv3-stage').value&&r.population==='native-'+r.stage),categories=[...new Set(all.map(category))];
 const rawMetric=metric==='worst_relative_residual'?'relative_residual':'pressure_relative_l2';
 if(native.length){const a=passAxes(svg,[0,Math.max(1,categories.length-1)],passBounds(native.map(r=>r[rawMetric]),true),false,true,'Native request · grid / spacing / cold or warm');
  for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
  categories.forEach((label,i)=>{line(svg,a.x(i),a.top,a.x(i),a.bottom);const parts=label.split(' · ');text(svg,a.x(i),442,parts[0]+' · '+parts[1],{'text-anchor':'middle','font-size':9});text(svg,a.x(i),456,parts[2],{'text-anchor':'middle','font-size':10})});
  for(const model of [...new Set(native.map(r=>r.model))]){let path='',connected=false;for(let i=0;i<categories.length;i++){const r=native.find(r=>r.model===model&&category(r)===categories[i]);if(!r){connected=false;continue}path+=(connected?'L':'M')+a.x(i)+','+a.y(r[rawMetric])+' ';connected=true;methodsV3Dot(svg,a.x(i),a.y(r[rawMetric]),r,`${r.model} · ${categories[i]} · ${r.stage} · ${rawMetric} ${sci(r[rawMetric])} · ${r.network_calls}/${r.max_passes} CNN calls · ${r.qualified?'qualified':'failed physical checks'}`)}const line=el('path',{d:path,fill:'none',stroke:passModelColor(model),'stroke-width':1.5,'data-methodsv3-line':model});svg.insertBefore(line,svg.querySelector('circle'))}
 }else text(svg,580,220,'No saved native requests match this selection.',{'text-anchor':'middle'});
 svg=$('methodsv3-coupled-chart');svg.replaceChildren();
 const size=Number($('methodsv3-grid').value),allCoupled=study.coupled.filter(r=>r.counts_xyz[0]===size),baseline=allCoupled.find(r=>r.model===null),chosen=allCoupled.filter(methodsV3Allowed),complete=chosen.filter(r=>r.complete);
 if(baseline&&complete.length){const left=350,right=1090,top=65,bottom=400,max=Math.max(1,...complete.map(r=>r.median_frame_ms/baseline.median_frame_ms))*1.15,x=v=>left+v/max*(right-left),height=(bottom-top)/complete.length;
  for(const tick of [0,1,2,4,8].filter(t=>t<=max)){line(svg,x(tick),top-15,x(tick),bottom);text(svg,x(tick),bottom+28,tick+'×',{'text-anchor':'middle'})}
  complete.forEach((r,i)=>{const y=top+(i+.5)*height,ratio=r.median_frame_ms/baseline.median_frame_ms;svg.append(el('rect',{x:left,y:y-height*.3,width:x(ratio)-left,height:height*.6,fill:r.model?passModelColor(r.model):color.native,'fill-opacity':.6,'data-methodsv3-coupled':r.identifier}));text(svg,left-15,y+4,(r.model||'Spectral')+' · '+r.kind,{'text-anchor':'end','font-size':12});methodsV3Dot(svg,x(ratio),y,r,`${r.identifier} · ${r.median_frame_ms.toFixed(1)} ms/frame · ${ratio.toFixed(2)}× spectral · ${r.advanced}/${r.requested} frames · ${r.median_network_calls} median CNN calls`);text(svg,x(ratio)+10,y+4,ratio.toFixed(2)+'×',{'font-size':12})});
 }else text(svg,580,220,'No complete coupled checks match this selection.',{'text-anchor':'middle'});
 $('methodsv3-coupled-note').textContent=baseline?`Spectral reference: ${baseline.median_frame_ms.toFixed(1)} ms per fluid step on ${size}³. ${complete.length} complete settings plotted; ${chosen.filter(r=>!r.complete).length} rejected settings excluded from latency ranking. Four-step medians remain short diagnostic measurements.`:'No matched spectral reference available.';
 const body=$('methodsv3-values');body.replaceChildren();
 function evidence(cells,source){const tr=document.createElement('tr');for(const v of cells){const td=document.createElement('td');td.textContent=v;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=source;a.textContent='Source';td.append(a);tr.append(td);body.append(tr)}
 for(const r of quiet)evidence([(r.model||'Spectral')+' · '+r.kind,'Historical128 · adaptive ceiling 8',sci(value(r))+' · '+r.median_ms.toFixed(3)+' ms',r.qualified+'/'+r.cases],r.source);
 for(const r of native)evidence([r.model+' · '+r.kind,category(r)+' · '+r.stage,sci(r[rawMetric]),r.qualified?'Qualified':'Failed physical checks'],r.source);
 for(const r of chosen)evidence([(r.model||'Spectral')+' · '+r.kind,size+'³ coupled · '+r.status,r.median_frame_ms.toFixed(1)+(r.complete?' ms/frame':' ms/attempt · rejected, excluded'),r.advanced+'/'+r.requested+' advanced'],r.source);
 const training=$('methodsv3-training');training.replaceChildren();for(const r of study.training.filter(methodsV3Allowed)){const tr=document.createElement('tr');for(const v of [r.model+' · '+r.name,r.initialization_description,r.parameters.toLocaleString(),r.optimizer_updates.toLocaleString(),r.optimizer_work.forward_cells.toLocaleString()]){const td=document.createElement('td');td.textContent=v;tr.append(td)}training.append(tr)}
}
$('methodsv3-scope').addEventListener('change',setupMethodsV3);
for(const id of ['methodsv3-model','methodsv3-metric','methodsv3-stage','methodsv3-passes','methodsv3-controls','methodsv3-grid'])$(id).addEventListener('change',renderMethodsV3);

function methodsV4Allowed(row){
 if($('methodsv4-policy').value!=='all'&&row.policy!==$('methodsv4-policy').value)return false;
 if(!row.model)return $('methodsv4-controls').checked;
 const p=data.cohort_points.find(p=>p.id===row.model);
 return p&&cohortMatchesScope(p,$('methodsv4-scope').value,'methodsv4')&&($('methodsv4-model').value==='all'||$('methodsv4-model').value===row.model)&&($('methodsv4-policy').value==='all'||row.policy===$('methodsv4-policy').value);
}
function setupMethodsV4(){
 const d=data.methods_v4,prior=$('methodsv4-model').value,ids=new Set([...(d?.models||[]),...(d?.controls||[])]);
 $('methodsv4-model').replaceChildren(new Option('All selected cohorts','all'));
 for(const p of data.cohort_points.filter(p=>ids.has(p.id)&&cohortMatchesScope(p,$('methodsv4-scope').value,'methodsv4')))$('methodsv4-model').append(new Option(p.id+' · '+p.name,p.id));
 $('methodsv4-model').value=[...$('methodsv4-model').options].some(o=>o.value===prior)?prior:'all';
 if(!$('methodsv4-policy').options.length){$('methodsv4-policy').append(new Option('All measured policies','all'));for(const policy of [...new Set((d?.quiet||[]).map(r=>r.policy))])$('methodsv4-policy').append(new Option(policy,policy))}
 if(!$('methodsv4-grid').options.length)for(const counts of [...new Set((d?.timing||[]).map(r=>r.counts.join('×')))])$('methodsv4-grid').append(new Option(counts,counts));
 renderMethodsV4();
}
function methodsV4Dot(svg,x,y,r,label){
 const stroke=r.model?passModelColor(r.model):color.native,dot=historyDot(svg,x,y,stroke,label);dot.dataset.methodsv4Identifier=r.identifier;dot.dataset.methodsv4Model=r.model||'numerical';
 const ok=r.qualified===(r.cases??true);dot.dataset.methodsv4Qualified=String(ok);if(!ok){dot.setAttribute('fill','none');dot.setAttribute('stroke',stroke);dot.setAttribute('stroke-width','2')}
 for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>{const a=document.createElement('a');a.href=r.source;a.textContent='Source evidence';$('methodsv4-detail').replaceChildren(document.createTextNode(label+' · '),a)});return dot;
}
function methodsV4NativeRows(){const stage=$('methodsv4-stage').value;return(data.methods_v4?.rows||[]).filter(r=>r.stage===stage&&r.population==='v4-'+stage&&methodsV4Allowed(r))}
function renderMethodsV4(){
 const d=data.methods_v4;if(!d)return;
 const metric=$('methodsv4-metric').value,quiet=d.quiet.filter(methodsV4Allowed),native=methodsV4NativeRows(),rawMetric=metric==='worst_relative_residual'?'relative_residual':'pressure_relative_l2';
 let svg=$('methodsv4-quiet-chart');svg.replaceChildren();const valid=quiet.filter(r=>r[metric]>0);
 if(valid.length){const a=passAxes(svg,passBounds(valid.map(r=>r.median_ms),true),passBounds(valid.map(r=>r[metric]),true),true,true,'Median prepared pressure-policy time (ms)');for(const r of valid)methodsV4Dot(svg,a.x(r.median_ms),a.y(r[metric]),r,`${r.model||'Numerical control'} · ${r.identifier} · ${r.qualified}/${r.cases} qualified · ${metric} ${sci(r[metric])} · ${r.median_ms.toFixed(3)} ms`)}else text(svg,580,220,'No positive recorded errors match this logarithmic plot.',{'text-anchor':'middle'});
 svg=$('methodsv4-native-chart');svg.replaceChildren();const validNative=native.filter(r=>r[rawMetric]>0),category=r=>r.counts_xyz.join('×')+' · '+r.cells_per_metre+'/m · '+(r.frame===0?'cold':'warm');
 const categories=[...new Set(d.rows.filter(r=>r.stage===$('methodsv4-stage').value&&r.population==='v4-'+r.stage).map(category))];
 if(validNative.length){const a=passAxes(svg,[0,Math.max(1,categories.length-1)],passBounds(validNative.map(r=>r[rawMetric]),true),false,true,'Matched grid / spacing / starting state');for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
  categories.forEach((label,i)=>{line(svg,a.x(i),a.top,a.x(i),a.bottom);const p=label.split(' · ');text(svg,a.x(i),441,p[0],{'text-anchor':'middle','font-size':9});text(svg,a.x(i),456,p[1]+' · '+p[2],{'text-anchor':'middle','font-size':9})});
  for(const identifier of [...new Set(validNative.map(r=>r.identifier))]){const selected=validNative.filter(r=>r.identifier===identifier);let path='',connected=false;for(let i=0;i<categories.length;i++){const r=selected.find(r=>category(r)===categories[i]);if(!r){connected=false;continue}path+=(connected?'L':'M')+a.x(i)+','+a.y(r[rawMetric])+' ';connected=true;methodsV4Dot(svg,a.x(i),a.y(r[rawMetric]),r,`${r.model||'Numerical'} · ${identifier} · ${categories[i]} · ${r.qualified?'qualified':'failed physical checks'} · ${rawMetric} ${sci(r[rawMetric])} · ${r.network_calls} CNN calls`)}svg.insertBefore(el('path',{d:path,fill:'none',stroke:selected[0].model?passModelColor(selected[0].model):color.native,'stroke-width':1.5,'stroke-dasharray':selected[0].policy.startsWith('gcr')?'6 4':'none','data-methodsv4-line':identifier}),svg.querySelector('circle'))}
 }else text(svg,580,220,'No positive native errors match this selection.',{'text-anchor':'middle'});
 const all=d.timing.filter(r=>r.counts.join('×')===$('methodsv4-grid').value),baseline=all.find(r=>r.identifier==='spectral'),chosen=all.filter(methodsV4Allowed),complete=chosen.filter(r=>r.complete);svg=$('methodsv4-timing-chart');svg.replaceChildren();
 if(baseline&&complete.length){const left=400,right=1090,top=45,bottom=405,maximum=Math.max(1,...complete.map(r=>r.median_seconds/baseline.median_seconds))*1.15,x=v=>left+v/maximum*(right-left),height=(bottom-top)/complete.length;
  for(const tick of [0,1,2,4,8].filter(t=>t<=maximum)){line(svg,x(tick),top,x(tick),bottom);text(svg,x(tick),435,tick+'×',{'text-anchor':'middle'})}
  complete.forEach((r,i)=>{const y=top+(i+.5)*height,ratio=r.median_seconds/baseline.median_seconds;svg.append(el('rect',{x:left,y:y-height*.3,width:x(ratio)-left,height:height*.6,fill:r.model?passModelColor(r.model):color.native,'fill-opacity':.65,'data-methodsv4-timing':r.identifier}));text(svg,left-10,y+3,(r.model||'Numerical')+' · '+r.identifier,{'text-anchor':'end','font-size':10});const row={...r,qualified:true};methodsV4Dot(svg,x(ratio),y,row,`${r.identifier} · ${ratio.toFixed(2)}× spectral · ${(1000*r.median_seconds).toFixed(1)} ms/step · 3/3 complete repetitions`);text(svg,x(ratio)+8,y+3,ratio.toFixed(2)+'×',{'font-size':11})});
 }else text(svg,580,220,'No complete fluid-step timing matches this selection.',{'text-anchor':'middle'});
 $('methodsv4-coverage').textContent=`${quiet.length} historical settings · ${native.length} selected native requests · ${chosen.filter(r=>!r.complete).length} rejected timing settings excluded from ranking. Pure numerical controls follow their own checkbox. Policies using the same cohort remain separate. Zero pressure differences define the spectral label; the table retains them while logarithmic plots omit zeros.`;
 const body=$('methodsv4-values');body.replaceChildren();function row(cells,source){const tr=document.createElement('tr');for(const v of cells){const td=document.createElement('td');td.textContent=v;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=source;a.textContent='Source';td.append(a);tr.append(td);body.append(tr)}
 for(const r of quiet)row([(r.model||'Numerical')+' · '+r.identifier,'Historical128 · adaptive policy',sci(r[metric])+' · '+r.median_ms.toFixed(3)+' ms',r.qualified+'/'+r.cases],r.source);
 for(const r of native)row([(r.model||'Numerical')+' · '+r.identifier,category(r)+' · '+r.stage,sci(r[rawMetric]),r.qualified?'Qualified':'Failed physical checks'],r.source);
 for(const r of chosen)row([(r.model||'Numerical')+' · '+r.identifier,$('methodsv4-grid').value+' · serial fluid timing',(r.median_seconds*1000).toFixed(1)+(r.complete?' ms/step':' ms/attempt · rejected, excluded'),r.accepted_repeats+'/3 complete repetitions'],r.source);
 const training=$('methodsv4-training');training.replaceChildren();for(const r of d.training.filter(r=>methodsV4Allowed({...r,policy:$('methodsv4-policy').value}))){const tr=document.createElement('tr');for(const v of [r.model+' · '+r.name,r.parameters.toLocaleString(),r.optimizer_updates.toLocaleString(),r.optimizer_work.forward_cells.toLocaleString(),(r.teacher_work.distillation_teacher_forward_cells||0).toLocaleString()]){const td=document.createElement('td');td.textContent=v;tr.append(td)}training.append(tr)}
}
$('methodsv4-scope').addEventListener('change',setupMethodsV4);
for(const id of ['methodsv4-model','methodsv4-policy','methodsv4-stage','methodsv4-metric','methodsv4-grid','methodsv4-controls'])$(id).addEventListener('change',renderMethodsV4);
$('methodsv4-show-drift').addEventListener('click',()=>{
 const t=(data.drift_traces||[]).find(t=>t.chart_study==='pressure-methods-v4'&&t.scene.counts[0]===64);if(!t)return;
 $('cohorts-drift-reset').click();$('cohorts-drift-from').value='C101';$('cohorts-drift-from').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();$('drift-scene').value=t.scene_key;$('drift-summary').value='single';$('drift-top').value='all';$('drift-horizon').value='64';driftHidden.clear();renderDrift();$('tab-drift').click();
});

function methodsV5Allowed(row){
 if($('methodsv5-policy').value!=='all'&&row.policy!==$('methodsv5-policy').value)return false;
 if(!row.model)return $('methodsv5-controls').checked;
 const p=data.cohort_points.find(p=>p.id===row.model);
 return p&&cohortMatchesScope(p,$('methodsv5-scope').value,'methodsv5')&&($('methodsv5-model').value==='all'||$('methodsv5-model').value===row.model)&&($('methodsv5-policy').value==='all'||row.policy===$('methodsv5-policy').value);
}
function setupMethodsV5(){
 const d=data.methods_v5,prior=$('methodsv5-model').value,ids=new Set([...(d?.models||[]),...(d?.controls||[])]);
 $('methodsv5-model').replaceChildren(new Option('All selected cohorts','all'));
 for(const p of data.cohort_points.filter(p=>ids.has(p.id)&&cohortMatchesScope(p,$('methodsv5-scope').value,'methodsv5')))$('methodsv5-model').append(new Option(p.id+' · '+p.name,p.id));
 $('methodsv5-model').value=[...$('methodsv5-model').options].some(o=>o.value===prior)?prior:'all';
 if(!$('methodsv5-policy').options.length){$('methodsv5-policy').append(new Option('All measured policies','all'));for(const policy of [...new Set([...(d?.quiet||[]),...(d?.rows||[]),...(d?.timing||[])].map(r=>r.policy))])$('methodsv5-policy').append(new Option(policy,policy))}
 if(!$('methodsv5-grid').options.length)for(const counts of [...new Set((d?.timing||[]).map(r=>r.counts.join('×')))])$('methodsv5-grid').append(new Option(counts,counts));
 renderMethodsV5();
}
function methodsV5Dot(svg,x,y,r,label){
 const stroke=r.model?passModelColor(r.model):color.native,dot=historyDot(svg,x,y,stroke,label);dot.dataset.methodsv5Identifier=r.identifier;dot.dataset.methodsv5Model=r.model||'numerical';
 const ok=r.qualified===(r.cases??true);dot.dataset.methodsv5Qualified=String(ok);if(!ok){dot.setAttribute('fill','none');dot.setAttribute('stroke',stroke);dot.setAttribute('stroke-width','2')}
 for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>{const a=document.createElement('a');a.href=r.source;a.textContent='Source evidence';$('methodsv5-detail').replaceChildren(document.createTextNode(label+' · '),a)});return dot;
}
function methodsV5NativeRows(){const stage=$('methodsv5-stage').value;return(data.methods_v5?.rows||[]).filter(r=>r.stage===stage&&r.population==='v5-'+$('methodsv5-cases').value+'-'+stage&&methodsV5Allowed(r))}
function renderMethodsV5(){
 const d=data.methods_v5;if(!d)return;
 const metric=$('methodsv5-metric').value,quiet=d.quiet.filter(r=>r.population===$('methodsv5-population').value&&methodsV5Allowed(r)),native=methodsV5NativeRows(),rawMetric=metric==='worst_relative_residual'?'relative_residual':'pressure_relative_l2';
 let svg=$('methodsv5-quiet-chart');svg.replaceChildren();const valid=quiet.filter(r=>r[metric]>0);
 if(valid.length){const a=passAxes(svg,passBounds(valid.map(r=>r.median_ms),true),passBounds(valid.map(r=>r[metric]),true),true,true,'Median prepared pressure-policy time (ms)');for(const r of valid)methodsV5Dot(svg,a.x(r.median_ms),a.y(r[metric]),r,`${r.model||r.control_kind||'Numerical control'} · ${r.identifier} · ${r.qualified}/${r.cases} qualified · ${metric} ${sci(r[metric])} · ${r.median_ms.toFixed(3)} ms`)}else text(svg,580,220,'No positive recorded errors match this logarithmic plot.',{'text-anchor':'middle'});
 svg=$('methodsv5-native-chart');svg.replaceChildren();const validNative=native.filter(r=>r[rawMetric]>0),category=r=>r.counts_xyz.join('×')+' · '+r.cells_per_metre+'/m · '+($('methodsv5-cases').value==='native'?(r.frame===0?'cold':'warm'):r.trajectory_id.slice(-5)+' #'+r.frame);
 const categories=[...new Set(d.rows.filter(r=>r.stage===$('methodsv5-stage').value&&r.population==='v5-'+$('methodsv5-cases').value+'-'+r.stage).map(category))];
 if(validNative.length){const a=passAxes(svg,[0,Math.max(1,categories.length-1)],passBounds(validNative.map(r=>r[rawMetric]),true),false,true,'Matched grid / spacing / starting state');for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
  categories.forEach((label,i)=>{line(svg,a.x(i),a.top,a.x(i),a.bottom);const p=label.split(' · ');text(svg,a.x(i),441,p[0],{'text-anchor':'middle','font-size':9});text(svg,a.x(i),456,p[1]+' · '+p[2],{'text-anchor':'middle','font-size':9})});
  for(const identifier of [...new Set(validNative.map(r=>r.identifier))]){const selected=validNative.filter(r=>r.identifier===identifier);let path='',connected=false;for(let i=0;i<categories.length;i++){const r=selected.find(r=>category(r)===categories[i]);if(!r){connected=false;continue}path+=(connected?'L':'M')+a.x(i)+','+a.y(r[rawMetric])+' ';connected=true;methodsV5Dot(svg,a.x(i),a.y(r[rawMetric]),r,`${r.model||r.control_kind||'Numerical'} · ${identifier} · ${categories[i]} · ${r.qualified?'qualified':'failed physical checks'} · ${rawMetric} ${sci(r[rawMetric])} · ${r.network_calls} CNN calls`)}svg.insertBefore(el('path',{d:path,fill:'none',stroke:selected[0].model?passModelColor(selected[0].model):color.native,'stroke-width':1.5,'stroke-dasharray':selected[0].policy.startsWith('gcr')?'6 4':'none','data-methodsv5-line':identifier}),svg.querySelector('circle'))}
 }else text(svg,580,220,'No positive native errors match this selection.',{'text-anchor':'middle'});
 const all=d.timing.filter(r=>r.counts.join('×')===$('methodsv5-grid').value),baseline=all.find(r=>r.identifier==='spectral'),chosen=all.filter(methodsV5Allowed),complete=chosen.filter(r=>r.complete);svg=$('methodsv5-timing-chart');svg.replaceChildren();
 if(baseline&&complete.length){const left=400,right=1090,top=45,bottom=405,maximum=Math.max(1,...complete.map(r=>r.median_seconds/baseline.median_seconds))*1.15,x=v=>left+v/maximum*(right-left),height=(bottom-top)/complete.length;
  for(const tick of [0,1,2,4,8].filter(t=>t<=maximum)){line(svg,x(tick),top,x(tick),bottom);text(svg,x(tick),435,tick+'×',{'text-anchor':'middle'})}
  complete.forEach((r,i)=>{const y=top+(i+.5)*height,ratio=r.median_seconds/baseline.median_seconds;svg.append(el('rect',{x:left,y:y-height*.3,width:x(ratio)-left,height:height*.6,fill:r.model?passModelColor(r.model):color.native,'fill-opacity':.65,'data-methodsv5-timing':r.identifier}));text(svg,left-10,y+3,(r.model||r.control_kind||'Numerical')+' · '+r.identifier,{'text-anchor':'end','font-size':10});const row={...r,qualified:true};methodsV5Dot(svg,x(ratio),y,row,`${r.identifier} · ${ratio.toFixed(2)}× spectral · ${(1000*r.median_seconds).toFixed(1)} ms/step · 3/3 complete repetitions`);text(svg,x(ratio)+8,y+3,ratio.toFixed(2)+'×',{'font-size':11})});
 }else text(svg,580,220,'No complete fluid-step timing matches this selection.',{'text-anchor':'middle'});
 $('methodsv5-coverage').textContent=`${quiet.length} timed settings · ${native.length} selected case measurements · ${chosen.filter(r=>!r.complete).length} rejected timing settings excluded from ranking. Numerical and untrained controls follow their own checkbox. Policies using the same cohort remain separate. Zero pressure differences define the spectral label; the table retains them while logarithmic plots omit zeros.`;
 const body=$('methodsv5-values');body.replaceChildren();function row(cells,source){const tr=document.createElement('tr');for(const v of cells){const td=document.createElement('td');td.textContent=v;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=source;a.textContent='Source';td.append(a);tr.append(td);body.append(tr)}
 for(const r of quiet)row([(r.model||r.control_kind||'Numerical')+' · '+r.identifier,r.population+' · adaptive policy',sci(r[metric])+' · '+r.median_ms.toFixed(3)+' ms',r.qualified+'/'+r.cases],r.source);
 for(const r of native)row([(r.model||r.control_kind||'Numerical')+' · '+r.identifier,category(r)+' · '+r.stage,sci(r[rawMetric]),r.qualified?'Qualified':'Failed physical checks'],r.source);
 for(const r of chosen)row([(r.model||r.control_kind||'Numerical')+' · '+r.identifier,$('methodsv5-grid').value+' · serial fluid timing',(r.median_seconds*1000).toFixed(1)+(r.complete?' ms/step':' ms/attempt · rejected, excluded'),r.accepted_repeats+'/3 complete repetitions'],r.source);
 const training=$('methodsv5-training');training.replaceChildren();for(const r of d.training.filter(r=>methodsV5Allowed({...r,policy:$('methodsv5-policy').value}))){const tr=document.createElement('tr');for(const v of [r.model+' · '+r.name,r.parameters.toLocaleString(),r.optimizer_updates.toLocaleString(),r.optimizer_work.forward_cells.toLocaleString(),r.initialization_description]){const td=document.createElement('td');td.textContent=v;tr.append(td)}training.append(tr)}
}
$('methodsv5-scope').addEventListener('change',setupMethodsV5);
for(const id of ['methodsv5-model','methodsv5-policy','methodsv5-stage','methodsv5-metric','methodsv5-grid','methodsv5-controls','methodsv5-population','methodsv5-cases'])$(id).addEventListener('change',renderMethodsV5);
$('methodsv5-show-drift').addEventListener('click',()=>{
 const t=(data.drift_traces||[]).find(t=>t.chart_study==='pressure-methods-v5'&&t.scene.counts[0]===64);if(!t)return;
 $('cohorts-drift-reset').click();$('cohorts-drift-from').value='C114';$('cohorts-drift-from').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();$('drift-scene').value=t.scene_key;$('drift-summary').value='single';$('drift-top').value='all';$('drift-horizon').value='32';driftHidden.clear();renderDrift();$('tab-drift').click();
});


function methodsV6Allowed(row){
 if($('methodsv6-policy').value!=='all'&&row.policy!==$('methodsv6-policy').value)return false;
 if(!row.model)return $('methodsv6-controls').checked;
 const p=data.cohort_points.find(p=>p.id===row.model);
 return p&&cohortMatchesScope(p,$('methodsv6-scope').value,'methodsv6')&&($('methodsv6-model').value==='all'||$('methodsv6-model').value===row.model)&&($('methodsv6-policy').value==='all'||row.policy===$('methodsv6-policy').value);
}
function setupMethodsV6(){
 const d=data.methods_v6,prior=$('methodsv6-model').value,ids=new Set([...(d?.models||[]),...(d?.controls||[])]);
 $('methodsv6-model').replaceChildren(new Option('All selected cohorts','all'));
 for(const p of data.cohort_points.filter(p=>ids.has(p.id)&&cohortMatchesScope(p,$('methodsv6-scope').value,'methodsv6')))$('methodsv6-model').append(new Option(p.id+' · '+p.name,p.id));
 $('methodsv6-model').value=[...$('methodsv6-model').options].some(o=>o.value===prior)?prior:'all';
 if(!$('methodsv6-policy').options.length){$('methodsv6-policy').append(new Option('All measured policies','all'));for(const policy of [...new Set([...(d?.quiet||[]),...(d?.rows||[]),...(d?.timing||[])].map(r=>r.policy))])$('methodsv6-policy').append(new Option(policy,policy))}
 if(!$('methodsv6-grid').options.length)for(const counts of [...new Set((d?.timing||[]).map(r=>r.counts.join('×')))])$('methodsv6-grid').append(new Option(counts,counts));
 renderMethodsV6();renderMethodsV6DriftTable();
}
function methodsV6Dot(svg,x,y,r,label){
 const stroke=r.model?passModelColor(r.model):color.native,dot=historyDot(svg,x,y,stroke,label);dot.dataset.methodsv6Identifier=r.identifier;dot.dataset.methodsv6Model=r.model||'numerical';
 const ok=r.qualified===(r.cases??true);dot.dataset.methodsv6Qualified=String(ok);if(!ok){dot.setAttribute('fill','none');dot.setAttribute('stroke',stroke);dot.setAttribute('stroke-width','2')}
 for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>{const a=document.createElement('a');a.href=r.source;a.textContent='Source evidence';$('methodsv6-detail').replaceChildren(document.createTextNode(label+' · '),a)});return dot;
}
function methodsV6CaseAllowed(r){const stage=$('methodsv6-stage').value,spatial=$('methodsv6-cases').value==='spectrum';return r.stage===stage&&(stage==='blind'?(!spatial&&['blind','large-blind'].includes(r.population)):spatial?r.population.startsWith('v5-spectrum-'):r.population.startsWith('v5-native-'))}
function methodsV6NativeRows(){return(data.methods_v6?.rows||[]).filter(r=>methodsV6CaseAllowed(r)&&methodsV6Allowed(r))}
function renderMethodsV6(){
 const d=data.methods_v6;if(!d)return;
 const metric=$('methodsv6-metric').value,quiet=d.quiet.filter(r=>r.population===$('methodsv6-population').value&&methodsV6Allowed(r)),native=methodsV6NativeRows(),rawMetric=metric==='worst_relative_residual'?'relative_residual':'pressure_relative_l2';
 let svg=$('methodsv6-quiet-chart');svg.replaceChildren();const valid=quiet.filter(r=>r[metric]>0);
 if(valid.length){const a=passAxes(svg,passBounds(valid.map(r=>r.median_ms),true),passBounds(valid.map(r=>r[metric]),true),true,true,'Median prepared pressure-policy time (ms)');for(const r of valid)methodsV6Dot(svg,a.x(r.median_ms),a.y(r[metric]),r,`${r.model||r.control_kind||'Numerical control'} · ${r.identifier} · ${r.qualified}/${r.cases} qualified · ${metric} ${sci(r[metric])} · ${r.median_ms.toFixed(3)} ms`)}else text(svg,580,220,'No positive recorded errors match this logarithmic plot.',{'text-anchor':'middle'});
 svg=$('methodsv6-native-chart');svg.replaceChildren();const validNative=native.filter(r=>r[rawMetric]>0),category=r=>r.counts_xyz.join('×')+' · '+(r.population.includes('development')?'D':'B')+r.trajectory_id.split('-').slice($('methodsv6-cases').value==='native'?-1:-2).join('-')+' #'+r.frame+' · '+r.cells_per_metre+'/m';
 const categories=[...new Set(d.rows.filter(r=>methodsV6CaseAllowed(r)).map(category))];
 if(validNative.length){const a=passAxes(svg,[0,Math.max(1,categories.length-1)],passBounds(validNative.map(r=>r[rawMetric]),true),false,true,'Matched grid / spacing / starting state');for(const node of [...svg.children])if(node.tagName==='text'&&node.getAttribute('y')==='445'||node.tagName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'))node.remove();
  categories.forEach((label,i)=>{line(svg,a.x(i),a.top,a.x(i),a.bottom);const p=label.split(' · ');text(svg,a.x(i),441,p[0],{'text-anchor':'middle','font-size':9});text(svg,a.x(i),456,p[1]+' · '+p[2],{'text-anchor':'middle','font-size':9})});
  for(const identifier of [...new Set(validNative.map(r=>r.identifier))]){const selected=validNative.filter(r=>r.identifier===identifier);let path='',connected=false;for(let i=0;i<categories.length;i++){const r=selected.find(r=>category(r)===categories[i]);if(!r){connected=false;continue}path+=(connected?'L':'M')+a.x(i)+','+a.y(r[rawMetric])+' ';connected=true;methodsV6Dot(svg,a.x(i),a.y(r[rawMetric]),r,`${r.model||r.control_kind||'Numerical'} · ${identifier} · ${categories[i]} · ${r.qualified?'qualified':'failed physical checks'} · ${rawMetric} ${sci(r[rawMetric])} · ${r.network_calls} CNN calls`)}svg.insertBefore(el('path',{d:path,fill:'none',stroke:selected[0].model?passModelColor(selected[0].model):color.native,'stroke-width':1.5,'stroke-dasharray':selected[0].policy.startsWith('gcr')?'6 4':'none','data-methodsv6-line':identifier}),svg.querySelector('circle'))}
 }else text(svg,580,220,'No positive native errors match this selection.',{'text-anchor':'middle'});
 const all=d.timing.filter(r=>r.counts.join('×')===$('methodsv6-grid').value),baseline=all.find(r=>r.identifier==='spectral'),chosen=all.filter(methodsV6Allowed),complete=chosen.filter(r=>r.complete);svg=$('methodsv6-timing-chart');svg.replaceChildren();
 if(baseline&&complete.length){const left=400,right=1090,top=45,bottom=405,maximum=Math.max(1,...complete.map(r=>r.median_seconds/baseline.median_seconds))*1.15,x=v=>left+v/maximum*(right-left),height=(bottom-top)/complete.length;
  for(const tick of [0,1,2,4,8].filter(t=>t<=maximum)){line(svg,x(tick),top,x(tick),bottom);text(svg,x(tick),435,tick+'×',{'text-anchor':'middle'})}
  complete.forEach((r,i)=>{const y=top+(i+.5)*height,ratio=r.median_seconds/baseline.median_seconds;svg.append(el('rect',{x:left,y:y-height*.3,width:x(ratio)-left,height:height*.6,fill:r.model?passModelColor(r.model):color.native,'fill-opacity':.65,'data-methodsv6-timing':r.identifier}));text(svg,left-10,y+3,(r.model||r.control_kind||'Numerical')+' · '+r.identifier,{'text-anchor':'end','font-size':10});const row={...r,qualified:true};methodsV6Dot(svg,x(ratio),y,row,`${r.identifier} · ${ratio.toFixed(2)}× spectral · ${(1000*r.median_seconds).toFixed(1)} ms/step · 3/3 complete repetitions`);text(svg,x(ratio)+8,y+3,ratio.toFixed(2)+'×',{'font-size':11})});
 }else text(svg,580,220,'No complete fluid-step timing matches this selection.',{'text-anchor':'middle'});
 $('methodsv6-coverage').textContent=`${quiet.length} timed settings · ${native.length} selected case measurements · ${chosen.filter(r=>!r.complete).length} rejected timing settings excluded from ranking. Numerical and inactive-CNN controls follow their own checkbox. Policies using the same cohort remain separate. Zero pressure differences define the spectral label; the table retains them while logarithmic plots omit zeros.`;
 const body=$('methodsv6-values');body.replaceChildren();function row(cells,source){const tr=document.createElement('tr');for(const v of cells){const td=document.createElement('td');td.textContent=v;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=source;a.textContent='Source';td.append(a);tr.append(td);body.append(tr)}
 for(const r of quiet)row([(r.model||r.control_kind||'Numerical')+' · '+r.identifier,r.population+' · adaptive policy',sci(r[metric])+' · '+r.median_ms.toFixed(3)+' ms',r.qualified+'/'+r.cases],r.source);
 for(const r of native)row([(r.model||r.control_kind||'Numerical')+' · '+r.identifier,category(r)+' · '+r.stage,sci(r[rawMetric]),r.qualified?'Qualified':'Failed physical checks'],r.source);
 for(const r of chosen)row([(r.model||r.control_kind||'Numerical')+' · '+r.identifier,$('methodsv6-grid').value+' · serial fluid timing',(r.median_seconds*1000).toFixed(1)+(r.complete?' ms/step':' ms/attempt · rejected, excluded'),r.accepted_repeats+'/3 complete repetitions'],r.source);
 const training=$('methodsv6-training');training.replaceChildren();for(const r of d.training.filter(r=>methodsV6Allowed({...r,policy:$('methodsv6-policy').value}))){const tr=document.createElement('tr');for(const v of [r.model+' · '+r.name,r.parameters.toLocaleString(),r.solver_training_iterations,r.optimizer_updates.toLocaleString(),r.optimizer_work.forward_cells.toLocaleString(),r.initialization_description]){const td=document.createElement('td');td.textContent=v;tr.append(td)}training.append(tr)}
}
function renderMethodsV6DriftTable(){const body=$('methodsv6-drift-values');body.replaceChildren();for(const r of [...(data.methods_v6?.drift||[]),...(data.methods_v6?.drift_controls||[])]){const trace=r.points?r:data.drift_traces.find(t=>t.source===r.source),points=trace.points,scene=trace.scene,tr=document.createElement('tr');for(const value of [(r.model||'Numerical control')+' · '+r.policy,scene.counts.join('×'),points.length,sci(arrayMax(points.map(p=>p[2])))]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href=r.source;a.textContent='Source';td.append(a);tr.append(td);body.append(tr)}}
$('methodsv6-scope').addEventListener('change',setupMethodsV6);
for(const id of ['methodsv6-model','methodsv6-policy','methodsv6-stage','methodsv6-metric','methodsv6-grid','methodsv6-controls','methodsv6-population','methodsv6-cases'])$(id).addEventListener('change',renderMethodsV6);
$('methodsv6-show-drift').addEventListener('click',()=>{
 const t=(data.drift_traces||[]).find(t=>t.chart_study==='pressure-methods-v6'&&t.scene.counts[0]===64);if(!t)return;
 $('cohorts-drift-reset').click();$('cohorts-drift-from').value='C117';$('cohorts-drift-from').dispatchEvent(new Event('change',{bubbles:true}));$('drift-build').value=t.native_binary_sha256;selectDriftScene();$('drift-scene').value=t.scene_key;$('drift-summary').value='single';$('drift-top').value='all';$('drift-horizon').value='256';driftHidden.clear();renderDrift();$('tab-drift').click();
});


