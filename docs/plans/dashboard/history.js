function renderAllHistory(){
 const family=$('history-family').value,status=$('history-status').value;
 const scope=$('history-scope').value;
 const all=data.history.filter(r=>cohortMatchesScope(cohortForRun(r),scope,'history')&&(family==='all'||r.group===family)&&(status==='all'||(status==='measured'&&(r.median!==null||r.worst!==null))||(status==='timed'&&historySpeed(r))));
 const pending=all.filter(r=>r.status==='pending').length,timed=all.filter(r=>historySpeed(r)).length;
 $('history-coverage').textContent=all.length+' entries shown · '+new Set(all.map(r=>r.group)).size+' study families · '+pending+' pending · '+timed+' with paired speed measurements'+(scope==='frontier'?' · accuracy–speed line only':scope==='matched'?' · matched validation cohorts only':'')+' · lowest worst error first';
 const host=$('history-charts');host.replaceChildren();
 for(const group of [...new Set(all.map(r=>r.group))].sort()){
  const rows=all.filter(r=>r.group===group).sort((a,b)=>worstFirst(a,b)||a.name.localeCompare(b.name)),field=rows[0].metric==='field_l2';
  const card=document.createElement('section');card.className='card history-card';
  const heading=document.createElement('h3');heading.textContent=group;card.append(heading);
  const note=document.createElement('p');note.textContent=(field?'This experiment reports pressure-field relative L2, a different quantity from equation residual. Amber = worst; teal = median; blue = mean.':'Equation residual and raw physical passes. Distinct policies for the same checkpoint are labelled separately.')+' Runs are sorted by worst error, lowest first.';card.append(note);
  const wrap=document.createElement('div');wrap.className='history-scroll';wrap.tabIndex=0;wrap.setAttribute('aria-label',group+' chart; scroll horizontally on narrow screens');card.append(wrap);
  const h=112+rows.length*64,svg=el('svg',{viewBox:'0 0 1240 '+h,role:'img','aria-label':group+': '+rows.length+' cohort and policy results'});wrap.append(svg);
  const errLeft=350,errRight=770,speedLeft=920,speedRight=1170,rowStart=108,last=h-24;
  const fieldValues=rows.flatMap(r=>[r.median,r.worst,r.mean]).filter(v=>Number.isFinite(v)&&v>0);
  const lo=field?Math.floor(Math.log10(arrayMin(fieldValues)))-.5:-13,hi=field?Math.ceil(Math.log10(arrayMax(fieldValues)))+.5:2,ex=v=>errLeft+(Math.log10(v)-lo)/(hi-lo)*(errRight-errLeft),sx=v=>speedLeft+(Math.log10(v)-Math.log10(.05))/(Math.log10(12)-Math.log10(.05))*(speedRight-speedLeft);
  text(svg,16,19,'COHORT / MODEL / INFERENCE POLICY',{'font-size':11,'font-weight':650});
  text(svg,(errLeft+errRight)/2,19,field?'PRESSURE-FIELD ERROR · LOWER IS BETTER':'EQUATION ERROR · LOWER IS BETTER',{'text-anchor':'middle','font-weight':650});
  text(svg,837,19,'PASSES',{'text-anchor':'middle','font-weight':650});text(svg,1045,19,'SPEED VS RHEON · HIGHER IS BETTER',{'text-anchor':'middle','font-weight':650});
  const ticks=field?powerTicks(lo,hi):[1e-12,1e-9,1e-6,1e-3,1,100];
  for(const t of ticks){line(svg,ex(t),62,ex(t),last);text(svg,ex(t),48,'1e'+Math.round(Math.log10(t)),{'text-anchor':'middle'})}
  for(const t of [.1,.5,1,5,10]){line(svg,sx(t),62,sx(t),last,t===1?color.native:color.grid,t===1?'4 4':null);text(svg,sx(t),48,t+'×',{'text-anchor':'middle'})}
  if(!field&&(/0[2-8] ·/.test(group)||Number(group.slice(0,2))>=11)){line(svg,ex(1e-6),62,ex(1e-6),last,color.blue,'4 4');text(svg,ex(1e-6),76,'model target',{'text-anchor':'middle',fill:color.blue,'font-size':9})}
  rows.forEach((r,i)=>{
   const y=rowStart+i*64;line(svg,16,y+39,1220,y+39);
   const a=el('a',{href:r.source,'data-run':r.name,'data-worst':r.worst??'','data-checkpoint':cohortForRun(r)?.weights_sha256??''});const label=r.name.length>44?r.name.slice(0,42)+'…':r.name;a.append(el('text',{x:16,y:y-5,fill:r.status==='pending'?color.amber:color.native,'font-size':12.5},label));a.append(el('title',{},r.name+' · '+r.source));svg.append(a);
   text(svg,16,y+12,r.population,{'font-size':10});text(svg,16,y+27,r.policy.length>54?r.policy.slice(0,52)+'…':r.policy,{'font-size':9.5});
   const tip=r.name+'; '+r.population+'; '+r.policy;
   if(r.median!==null&&r.worst!==null)line(svg,ex(r.median),y,ex(r.worst),y,color.muted);
   if(r.median!==null)historyDot(svg,ex(r.median),y,color.teal,tip+'; median '+(field?'pressure-field error ':'residual ')+sci(r.median));
   if(r.worst!==null)historyDot(svg,ex(r.worst),y,color.amber,tip+'; worst '+(field?'pressure-field error ':'equation residual ')+sci(r.worst));
   if(r.mean)historyDot(svg,ex(r.mean),y,color.blue,tip+'; mean pressure-field error '+sci(r.mean));
   if(r.median===null&&r.worst===null)text(svg,(errLeft+errRight)/2,y+4,r.status==='pending'?'Results pending':'Residual not reported',{'text-anchor':'middle',fill:r.status==='pending'?color.amber:color.muted});
   else text(svg,(errLeft+errRight)/2,y+24,(r.median!==null?'median '+sci(r.median)+' / ':'')+'worst '+sci(r.worst),{'text-anchor':'middle','font-size':10});
   text(svg,837,y+4,r.qualified!==null?r.qualified+'/'+r.cases:'—',{'text-anchor':'middle',fill:r.qualified===r.cases&&r.cases!==null?color.teal:color.muted});
   const speed=historySpeed(r);if(speed){historyDot(svg,sx(speed.value),y,color.teal,tip+'; '+speed.value.toFixed(3)+'× Rheon; '+speed.note);text(svg,sx(speed.value),y+24,speed.value.toFixed(2)+'×',{'text-anchor':'middle',fill:color.teal,'font-size':11})}else text(svg,1045,y+4,r.status==='pending'?'Pending':'No paired timing',{'text-anchor':'middle','font-size':11});
  });
  host.append(card);
 }
}
for(const family of [...new Set(data.history.map(r=>r.group))].sort()){const option=document.createElement('option');option.value=family;option.textContent=family;$('history-family').append(option)}
for(const id of ['history-scope','history-family','history-status','history-baseline','history-scene','history-threads'])$(id).addEventListener('change',renderAllHistory);
$('history-download').addEventListener('click',()=>$('download').click());
function renderLatest(){
 const study=data.trajectory_studies.find(s=>s.id===$('latest-study').value),population=$('latest-population').value,selection={...study,reference:study.populations[population].reference||study.reference,candidate:study.populations[population].candidate||study.candidate},models=study.populations[population].models;
 const scope=$('latest-scope').value,rows=Object.entries(models).filter(([name])=>cohortMatchesScope(driftCohorts.get(study.model_weights?.[name])||cohortForRun(name),scope,'outcomes')).sort((a,b)=>worstFirst(a[1],b[1],'worst_pressure_relative_l2')||a[0].localeCompare(b[0]));
 const labels={'cohort-two-reference':'Previous reference','cohort-two-shape-feedback':'Shape feedback','cohort-four-control':'Four-pass control','cohort-one-control':'One-pass control','cohort-two-seed2':'Two-pass · seed 2','cohort-two-seed3':'Two-pass · seed 3','shape-two-reference':'Shape reference','shape-two-seed2':'Two-pass · seed 2','shape-two-seed3':'Two-pass · seed 3','shape-one-control':'One-pass · 5k','shape-one-longer':'One-pass · 10k','two-pass-10000':'Two-pass · 10k','coverage-only':'Coverage only','feedback-only':'Feedback only'};
 $('latest-heading').textContent=study.title+' · trajectory results';$('latest-source').href=study.source;$('latest-selection-source').href=study.selection_source;
 for(const [id,key,log] of [['latest-passes','qualified_scenes',false],['latest-work','mean_network_evaluations',false],['latest-pressure','worst_pressure_relative_l2',true]]){
  const svg=$(id),height=70+rows.length*55;svg.replaceChildren();svg.setAttribute('viewBox','0 0 410 '+height);
  if(!rows.length){text(svg,205,42,'No cohorts match this tab’s filters',{'text-anchor':'middle','font-size':12});continue}
  const left=145,right=345,values=rows.map(([,r])=>r[key]);
  const lo=log?Math.floor(Math.log10(arrayMin(values)))-1:0,hi=log?Math.ceil(Math.log10(arrayMax(values)))+1:id==='latest-passes'?Math.max(...rows.map(([,r])=>r.scenes)):arrayMax(values)*1.18;
  const x=v=>left+((log?Math.log10(v):v)-lo)/(hi-lo)*(right-left);
  const ticks=log?[lo,(lo+hi)/2,hi]:[0,hi/2,hi];
  for(const tick of ticks){const px=left+(tick-lo)/(hi-lo)*(right-left);line(svg,px,28,px,height-25);text(svg,px,18,log?'1e'+Number(tick.toFixed(1)):tick.toFixed(id==='latest-passes'?0:1),{'text-anchor':'middle','font-size':10})}
  rows.forEach(([name,r],i)=>{const y=55+i*55,candidate=name===selection.candidate,col=candidate?(study.decision?.candidate_promoted===false?color.amber:color.teal):color.muted,value=r[key],label=key==='qualified_scenes'?value+'/'+r.scenes:key==='mean_network_evaluations'?value.toFixed(3):sci(value);
   text(svg,8,y+4,labels[name]||name,{fill:col,'font-size':12});
   if(log){const dot=historyDot(svg,x(value),y,col,name+': worst pressure-field deviation '+sci(value));dot.setAttribute('data-run',name);dot.setAttribute('data-worst',r.worst_pressure_relative_l2)}
   else {const bar=el('rect',{x:left,y:y-8,width:x(value)-left,height:16,rx:3,fill:col,'data-run':name,'data-worst':r.worst_pressure_relative_l2});bar.append(el('title',{},name+': '+label));svg.append(bar)}
   text(svg,log?x(value):x(value)+6,y+(log?21:4),label,{'text-anchor':log?'middle':'start',fill:col,'font-size':11});
  });
 }
 const previous=models[selection.reference],latest=models[selection.candidate];
 const workChange=(latest.mean_network_evaluations/previous.mean_network_evaluations-1)*100,pressureRatio=previous.worst_pressure_relative_l2/latest.worst_pressure_relative_l2;
 $('latest-finding').textContent=(labels[selection.candidate]||selection.candidate)+': '+latest.qualified_scenes+'/'+latest.scenes+' simulations and '+latest.qualified_ai_steps.toLocaleString()+'/'+latest.attempted_ai_steps.toLocaleString()+' AI steps pass. Previous reference: '+previous.qualified_scenes+'/'+previous.scenes+' simulations. Mean evaluations '+(workChange<=0?'fall ':'rise ')+Math.abs(workChange).toFixed(1)+'%; worst pressure-field deviation is '+(pressureRatio>=1?pressureRatio:1/pressureRatio).toFixed(2)+'× '+(pressureRatio>=1?'lower':'higher')+'. '+(study.decision?.candidate_promoted===false?'Candidate not promoted; previous laboratory control retained. ':'')+(study.kind==='stopping-policy'?'Serial timing populations are available in the Policy comparisons tab.':'No new paired runtime benchmark was reported.');
 if(population==='final_holdout'&&study.unexecuted_final_models?.length)$('latest-finding').textContent+=' No final-holdout result for: '+study.unexecuted_final_models.map(name=>labels[name]||name).join(', ')+'.';
 if(scope!=='all')$('latest-finding').textContent=rows.length+' of '+Object.keys(models).length+' runs shown: '+(scope==='frontier'?'accuracy–speed line cohorts only. ':'matched validation cohorts only. ')+rows.map(([name,r])=>(labels[name]||name)+': '+r.qualified_scenes+'/'+r.scenes+' simulations; worst pressure-field deviation '+sci(r.worst_pressure_relative_l2)).join('. ')+'. Frontier membership uses the Accuracy & speed tab; these rows show the selected trajectory population.';
 if(population==='final_holdout'&&study.decision?.original_final_protocol_satisfied===false)$('latest-finding').textContent+=' Final results use the amended protocol; the original final protocol was not satisfied.';
 if(study.outcome_note)$('latest-finding').textContent+=' '+study.outcome_note;
}
function selectLatestStudy(){const study=data.trajectory_studies.find(s=>s.id===$('latest-study').value);$('latest-population').replaceChildren();for(const key of ['final_holdout',...Object.keys(study.populations).filter(k=>k!=='final_holdout')]){if(!study.populations[key])continue;const option=document.createElement('option');option.value=key;const first=Object.values(study.populations[key].models)[0];option.textContent=(key==='final_holdout'&&study.decision?.original_final_protocol_satisfied===false?'Amended final (original protocol not satisfied)':(study.population_labels?.[key]||key.replaceAll('_',' ')))+' · '+first.scenes+' simulations';$('latest-population').append(option)}if(study.default_population)$('latest-population').value=study.default_population;renderLatest()}
for(const study of [...data.trajectory_studies].reverse()){const option=document.createElement('option');option.value=study.id;option.textContent=study.title;$('latest-study').append(option)}
$('latest-study').addEventListener('change',selectLatestStudy);
$('latest-population').addEventListener('change',renderLatest);
$('latest-scope').addEventListener('change',renderLatest);
