const factorExposureByWeight=new Map((data.training_exposures||[]).map(e=>[e.weights_sha256,e]));
const factorCountNames={unique_real_snapshots:'Unique real training snapshots',real_simulations:'Real training simulations',manufactured_presentations:'Manufactured training presentations',unique_synthetic_snapshots:'Unique synthetic training patterns'};
function factorTrainingCounts(exposure,domain,matching=$('factor-training-match').value){
 if(!exposure)return null;const totals={unique_real_snapshots:0,real_simulations:0,manufactured_presentations:0,unique_synthetic_snapshots:0};
 const counts=matching==='permuted'?[...domain.counts_xyz].sort((a,b)=>a-b):domain.counts_xyz;
 for(const d of exposure.domains){const candidate=matching==='permuted'?[...d.counts_xyz].sort((a,b)=>a-b):d.counts_xyz;if(JSON.stringify(candidate)!==JSON.stringify(counts)||(matching!=='counts'&&d.cell_size_m!==domain.cell_size_m))continue;for(const key of Object.keys(totals))totals[key]+=d[key]||0}
 return totals;
}
function factorQuantile(values,q){const sorted=[...values].sort((a,b)=>a-b),position=(sorted.length-1)*q,lo=Math.floor(position),fraction=position-lo;return sorted[lo]+fraction*(sorted[Math.min(lo+1,sorted.length-1)]-sorted[lo])}
function factorCorrelation(points,xKey='x',yKey='value'){return shapeCorrelation(points.filter(p=>Number.isFinite(p[xKey])&&Number.isFinite(p[yKey])).map(p=>({aspect:p[xKey],score:p[yKey]})))}
function factorAnalysis(){
 const{models,domains}=shapeAnalysis(),relative=$('factor-error').value==='relative',metric=$('factor-training-count').value,baselines=new Map(models.map(e=>[e.weights_sha256,driftMedian(e.domains.map(d=>d.worst_error))]));
 const pairs=[],cells=[];
 for(const d of domains){const values=[];for(const m of d.measurements){const model=m.evaluation,raw=m.domain.worst_error,baseline=baselines.get(model.weights_sha256),value=relative?(baseline>0?raw/baseline:null):raw,exposure=factorExposureByWeight.get(model.weights_sha256),counts=factorTrainingCounts(exposure,d);pairs.push({domain:d,measurement:m.domain,model,raw,baseline,value,exposure,counts,x:counts?counts[metric]:null});if(Number.isFinite(value))values.push(value)}if(values.length)cells.push({...d,x:d.cells,value:$('shape-score').value==='maximum'?arrayMax(values):driftMedian(values)})}
 const spread=models.map(model=>{const rows=pairs.filter(p=>p.model.weights_sha256===model.weights_sha256),raw=rows.map(p=>p.raw),positive=raw.filter(v=>v>0),logs=positive.map(Math.log10),seen=rows.filter(p=>p.x>0),unseen=rows.filter(p=>p.x===0);return{model,rows,median:raw.length?driftMedian(raw):null,fold:positive.length===raw.length&&raw.length?10**(factorQuantile(logs,.75)-factorQuantile(logs,.25)):null,range:raw.length&&arrayMin(raw)>0?arrayMax(raw)/arrayMin(raw):null,seen,unseen,rho:factorCorrelation(rows)}}).sort((a,b)=>(a.fold??Infinity)-(b.fold??Infinity)||a.model.cohort.id.localeCompare(b.model.cohort.id));
 return{models,domains,pairs,cells,spread,relative,metric};
}
function renderFactors(){
 const a=factorAnalysis(),{models,domains,pairs,cells,spread,relative,metric}=a,cellSvg=$('factor-cells-chart'),trainSvg=$('factor-training-chart'),body=$('factor-values'),spreadBody=$('factor-spread-values'),key=$('factor-model-key');for(const e of [cellSvg,trainSvg,body,spreadBody,key])e.replaceChildren();
 const log=$('shape-scale').value==='log',countLog=$('factor-count-scale').value==='log',eligible=pairs.filter(p=>p.x!==null&&Number.isFinite(p.value)),label=relative?'Error ÷ model median':'Worst relative equation error';
 $('factor-cells-note').textContent=`X: X × Y × Z total cells${countLog?' on log(1 + count) scale':''}. Y: ${relative?'each model’s error normalized to its own median across all evaluation resolutions':'worst equation error'}${models.length>1?', aggregated using '+$('shape-score').selectedOptions[0].textContent.toLowerCase():''}. Color identifies cell spacing.`;
 $('factor-training-note').textContent=`X: ${factorCountNames[metric].toLowerCase()} matching this resolution${countLog?' on log(1 + count) scale, including zero':''}. Y: ${relative?'error ÷ this model’s median error':'this model’s worst equation error'}. One point per model–resolution pair. Color identifies the model.`;
 $('factor-coverage').textContent=`${cells.length} resolution points in the cell-count graph · ${eligible.length} model–resolution points with training evidence (${eligible.filter(p=>p.x>0).length} with matching training samples; ${eligible.filter(p=>p.x===0).length} with zero for the selected measure) · ${pairs.length-eligible.length} pairs omitted for unavailable counts or undefined normalization. Relative error 1× means a model’s typical error across its complete resolution set.`;
 $('factor-detail').textContent='Hover, focus, or click a point to inspect its cell count, training coverage, error, and source records.';
 const rho=factorCorrelation(cells),within=spread.map(s=>s.rho).filter(v=>v!==null);
 $('factor-finding').textContent=`Cell count versus difficulty: rank correlation ${rho===null?'unavailable':rho.toFixed(2)}. Training count versus difficulty: median within-model correlation ${within.length?driftMedian(within).toFixed(2):'unavailable'} across ${within.length} models with variation. Compare error level and proportional spread in the table below; a flat high-error line can still be inaccurate.`;
 if(!domains.length||!models.length){for(const svg of [cellSvg,trainSvg])text(svg,380,240,'No resolution results match these model filters.',{'text-anchor':'middle'});return}
 const values=[...cells,...eligible].map(p=>p.value).filter(v=>Number.isFinite(v)&&(!log||v>0));let lo=log&&values.length?Math.floor(Math.log10(arrayMin(values))):0,hi=log&&values.length?Math.ceil(Math.log10(arrayMax(values))):values.length?arrayMax(values)*1.08:1;if(hi===lo)hi=lo+1;
 const p={left:85,right:720,top:40,bottom:405},y=v=>p.bottom-((log?Math.log10(v):v)-lo)/(hi-lo)*(p.bottom-p.top),transform=v=>countLog?Math.log10(1+v):v;
 function axes(svg,points,xTitle){const max=Math.max(1,...points.map(d=>d.x)),x=v=>p.left+14+transform(v)/transform(max)*(p.right-p.left-28),ticks=countLog?[...new Set([0,...Array.from({length:Math.floor(Math.log10(max))+1},(_,i)=>10**i),max])].sort((a,b)=>a-b):max<=10?Array.from({length:Math.floor(max)+1},(_,i)=>i):Array.from({length:6},(_,i)=>Math.round(max*i/5));for(const v of ticks){line(svg,x(v),p.top,x(v),p.bottom);text(svg,x(v),431,v.toLocaleString(),{'text-anchor':'middle'})}const errors=log?Array.from({length:hi-lo+1},(_,i)=>10**(lo+i)):Array.from({length:6},(_,i)=>hi*i/5);for(const v of errors){line(svg,p.left,y(v),p.right,y(v));text(svg,p.left-10,y(v)+4,relative?(v>=.01&&v<1000?Number(v.toPrecision(3)):sci(v))+'×':sci(v),{'text-anchor':'end'})}text(svg,400,475,xTitle,{'text-anchor':'middle','font-size':12});text(svg,20,225,label+' · '+(log?'log':'linear'),{transform:'rotate(-90 20 225)','text-anchor':'middle','font-size':12});if(relative&&lo<=(log?0:1)&&hi>=(log?0:1)){line(svg,p.left,y(1),p.right,y(1),color.native,'5 5');text(svg,p.right,y(1)-7,'1× model’s typical error',{'text-anchor':'end','font-size':10})}return x}
 const cellX=axes(cellSvg,cells,'Total cells in domain'+(countLog?' · log(1 + cells)':'')),trainX=axes(trainSvg,eligible,factorCountNames[metric]+(countLog?' · log(1 + count)':''));
 const spacings=[...new Set((data.resolution_evaluations||[]).flatMap(e=>e.domains.map(d=>d.cell_size_m)))].sort((a,b)=>b-a),spacingColor=s=>[color.teal,color.blue,color.amber,'#ec9cda'][spacings.indexOf(s)%4];
 const detail=(textValue,pair)=>{const links=[];for(const[label,href]of [['Evaluation cases',pair?.model.source],['Training evidence',pair?.exposure?.source],['Training history',pair?.exposure?.history_source]]){if(!href)continue;const link=document.createElement('a');link.href=href;link.textContent=label;links.push(link)}$('factor-detail').replaceChildren(document.createTextNode(textValue+' '));for(const link of links){if($('factor-detail').querySelector('a'))$('factor-detail').append(' · ');$('factor-detail').append(link)}};
 function point(svg,cx,value,fill,textValue,attributes,pair){if(log&&value<=0)return;const dot=historyDot(svg,cx,y(value),fill,textValue);dot.setAttribute('r',4);dot.setAttribute('fill-opacity',.75);dot.setAttribute('stroke','#0b141c');dot.setAttribute('stroke-width',.8);for(const[k,v]of Object.entries(attributes))dot.setAttribute(k,v);for(const event of ['pointerenter','focus','click'])dot.addEventListener(event,()=>detail(textValue,pair))}
 for(const d of cells){const evidence=pairs.filter(p=>p.domain.key===d.key).sort((a,b)=>b.raw-a.raw)[0];point(cellSvg,cellX(d.x),d.value,spacingColor(d.cell_size_m),`${d.counts_xyz.join(' × ')} at ${d.cell_size_m} m · ${d.cells.toLocaleString()} cells · ${label} ${relative?d.value.toPrecision(4)+'×':sci(d.value)} · ${d.measurements.length} models`,{'data-factor-kind':'cells','data-factor-key':d.key,'data-factor-x':d.x,'data-factor-value':d.value},evidence)}
 for(const row of eligible){const d=row.domain,description=`${row.model.cohort.id} ${row.model.cohort.name} · ${d.counts_xyz.join(' × ')} at ${d.cell_size_m} m · ${d.cells.toLocaleString()} cells · ${factorCountNames[metric]} ${row.x.toLocaleString()} · raw worst error ${sci(row.raw)} · error/model median ${row.baseline>0?(row.raw/row.baseline).toPrecision(4)+'×':'undefined'}`;point(trainSvg,trainX(row.x),row.value,driftColor(row.model),description,{'data-factor-kind':'training','data-factor-model':row.model.cohort.id,'data-factor-key':d.key,'data-factor-x':row.x,'data-factor-value':row.value},row)}
 if(!eligible.length)text(trainSvg,400,230,'No complete training evidence for these filters.',{'text-anchor':'middle'});
 const physicalStages=models.filter(m=>m.cohort.training_stage);if(physicalStages.length)$('factor-coverage').textContent+=' '+physicalStages.map(m=>m.cohort.id).join(', ')+': counts cover the recorded physical-request training stage only; inherited C53 training is excluded. Unique snapshots are available training-set membership, not optimizer draw frequencies.';
 const stageOnly=models.filter(m=>factorExposureByWeight.get(m.weights_sha256)?.scope?.startsWith('Additional fine-tuning stage only:'));if(stageOnly.length)$('factor-coverage').textContent+=' '+stageOnly.map(m=>m.cohort.id).join(', ')+': counts cover the manufactured fine-tuning stage; inherited training from '+[...new Set(stageOnly.map(m=>m.cohort.initialized_from).filter(Boolean))].sort().join(', ')+' is not counted again.';
 const replayDraws=models.map(m=>({model:m,draws:factorExposureByWeight.get(m.weights_sha256)?.current_policy_presentations||0})).filter(m=>m.draws);if(replayDraws.length)$('factor-coverage').textContent+=' Additional current-policy residual presentations: '+replayDraws.map(m=>m.model.cohort.id+' '+m.draws.toLocaleString()).join(', ')+'. These replays are separate from the original snapshot and analytical-draw counts on these axes; they are not new simulations.';
 const zeroValues=[...cells,...eligible].filter(d=>d.value===0).length;if(log&&zeroValues)$('factor-coverage').textContent+=` ${zeroValues} zero-error points are omitted on the log Y-axis; choose linear error scale to show them.`;
 const spacingNote=document.createElement('span');spacingNote.textContent='Cell-count plot:';key.append(spacingNote);for(const spacing of [...new Set(domains.map(d=>d.cell_size_m))].sort((a,b)=>b-a)){const span=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=spacingColor(spacing);span.append(swatch,document.createTextNode(spacing+' m spacing'));key.append(span)}const modelNote=document.createElement('span');modelNote.textContent='Training plot:';key.append(modelNote);for(const e of models){const span=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=driftColor(e);span.append(swatch,document.createTextNode(e.cohort.id+' · '+e.cohort.name));key.append(span)}
 for(const row of pairs){const d=row.domain,tr=document.createElement('tr');tr.dataset.factorModel=row.model.cohort.id;tr.dataset.factorKey=d.key;for(const v of [row.model.cohort.id+' '+row.model.cohort.name+' · '+d.counts_xyz.join(' × ')+' · '+d.cell_size_m+' m',d.cells.toLocaleString(),...['unique_real_snapshots','real_simulations','manufactured_presentations'].map(k=>row.counts?row.counts[k].toLocaleString():'Unavailable'),sci(row.raw),row.baseline>0?(row.raw/row.baseline).toPrecision(4)+'×':'Undefined']){const td=document.createElement('td');td.textContent=v;tr.append(td)}const td=document.createElement('td');for(const[label,href]of[['Error',row.model.source],['Training',row.exposure?.source]]){if(!href)continue;if(td.childNodes.length)td.append(' · ');const link=document.createElement('a');link.href=href;link.textContent=label;td.append(link)}tr.append(td);body.append(tr)}
 for(const s of spread){const tr=document.createElement('tr');tr.dataset.factorSpreadModel=s.model.cohort.id;tr.dataset.factorSpread=s.fold??'';const typical=rows=>rows.length?sci(driftMedian(rows.map(p=>p.raw))):'—';for(const v of [s.model.cohort.id+' · '+s.model.cohort.name,s.median===null?'—':sci(s.median),s.fold===null?'Zero error present':s.fold.toFixed(2)+'×',s.range===null?'Zero error present':s.range.toFixed(1)+'×',s.seen.length+' / '+s.unseen.length+(s.rows.some(p=>p.x===null)?' · counts unavailable':''),typical(s.seen)+' / '+typical(s.unseen)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.firstChild.style.color=driftColor(s.model);const td=document.createElement('td'),exposure=factorExposureByWeight.get(s.model.weights_sha256);if(exposure){const link=document.createElement('a');link.href=exposure.source;link.textContent='Complete training split';td.append(link)}else td.textContent='Unavailable';tr.append(td);spreadBody.append(tr)}
}
for(const id of ['factor-error','factor-training-count','factor-training-match','factor-count-scale'])$(id).addEventListener('change',renderFactors);
function shapeModelPool(){
 const scope=$('shape-scope').value;
 return(data.resolution_evaluations||[]).map(e=>({...e,cohort:driftCohorts.get(e.weights_sha256)})).filter(e=>cohortMatchesScope(e.cohort,scope,'factors')&&($('shape-family').value==='all'||e.cohort.family===$('shape-family').value));
}
function shapeGeometry(counts){const sides=[...counts].sort((a,b)=>a-b);return{aspect:sides[2]/sides[0],middle:sides[1]/sides[0],cells:counts.reduce((a,b)=>a*b,1)}}
function shapeAnalysis(){
 const pool=shapeModelPool(),selection=$('shape-model').value,models=pool.filter(e=>selection==='aggregate'||e.weights_sha256===selection),grouped=new Map();
 for(const e of models)for(const d of e.domains){if($('shape-spacing').value!=='all'&&String(d.cell_size_m)!==$('shape-spacing').value)continue;const key=resolutionDomainKey(d);if(!grouped.has(key))grouped.set(key,{key,counts_xyz:d.counts_xyz,cell_size_m:d.cell_size_m,...shapeGeometry(d.counts_xyz),measurements:[]});grouped.get(key).measurements.push({evaluation:e,domain:d})}
 const domains=[...grouped.values()].map(d=>({...d,score:$('shape-score').value==='maximum'?Math.max(...d.measurements.map(m=>m.domain.worst_error)):driftMedian(d.measurements.map(m=>m.domain.worst_error))})).sort((a,b)=>b.score-a.score||a.key.localeCompare(b.key));
 return{pool,models,domains};
}
function shapeRanks(values){
 const sorted=values.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v),ranks=[];let i=0;
 while(i<sorted.length){let end=i+1;while(end<sorted.length&&sorted[end].v===sorted[i].v)end++;for(let j=i;j<end;j++)ranks[sorted[j].i]=(i+end-1)/2;i=end}return ranks;
}
function shapeCorrelation(domains){
 if(domains.length<3)return null;const a=shapeRanks(domains.map(d=>d.aspect)),b=shapeRanks(domains.map(d=>d.score)),mean=(a.length-1)/2;let covariance=0,varianceA=0,varianceB=0;
 for(let i=0;i<a.length;i++){covariance+=(a[i]-mean)*(b[i]-mean);varianceA+=(a[i]-mean)**2;varianceB+=(b[i]-mean)**2}return varianceA&&varianceB?covariance/Math.sqrt(varianceA*varianceB):null;
}
function selectShapeModels(){
 const prior=$('shape-model').value;$('shape-model').replaceChildren();
 const aggregate=document.createElement('option');aggregate.value='aggregate';aggregate.textContent='Aggregate of filtered models';$('shape-model').append(aggregate);
 for(const e of shapeModelPool()){const o=document.createElement('option');o.value=e.weights_sha256;o.textContent=e.cohort.id+' · '+e.cohort.name;$('shape-model').append(o)}
 if([...$('shape-model').options].some(o=>o.value===prior))$('shape-model').value=prior;renderShapes();
}
function setupShapes(){
 const evaluations=data.resolution_evaluations||[];
 for(const family of [...new Set(evaluations.map(e=>driftCohorts.get(e.weights_sha256)?.family).filter(Boolean))].sort()){const o=document.createElement('option');o.value=family;o.textContent=family;$('shape-family').append(o)}
 for(const spacing of [...new Set(evaluations.flatMap(e=>e.domains.map(d=>d.cell_size_m)))].sort((a,b)=>b-a)){const o=document.createElement('option');o.value=spacing;o.textContent=spacing+' m';$('shape-spacing').append(o)}
 selectShapeModels();
}
function renderShapes(){
 const{pool,models,domains}=shapeAnalysis(),aspectSvg=$('shape-aspect-chart'),mapSvg=$('shape-map-chart'),body=$('shape-values'),key=$('shape-spacing-key');for(const e of [aspectSvg,mapSvg,body,key])e.replaceChildren();
 const aggregate=$('shape-model').value==='aggregate',log=$('shape-scale').value==='log',stat=$('shape-score').value==='maximum'?'maximum':'median';$('shape-score').disabled=!aggregate;
 $('shape-detail').textContent='Hover, focus, or click a point in either graph to inspect the resolution and error.';
 $('shape-coverage').textContent=`${domains.length} resolutions · ${models.length} model(s) contributing / ${pool.length} under the cohort and family filters · ${aggregate?stat+' of each model’s worst error':models[0]?.cohort.id+' '+models[0]?.cohort.name+' · worst error per resolution'}. Both graphs use all matching resolutions, independent of the ranked chart’s top-10/20 limit.`;
 const rho=shapeCorrelation(domains);
 $('shape-finding').textContent=rho===null?'Rank correlation is unavailable: select at least three resolutions with variation in both shape and difficulty.':`Aspect ratio versus difficulty: rank correlation ${rho.toFixed(2)} across ${domains.length} resolutions. Near zero means little overall monotonic relationship; positive means more elongated shapes tend to have higher error, and negative means lower error. This is an association in these recorded cases.`;
 if(!domains.length){for(const svg of [aspectSvg,mapSvg])text(svg,380,240,'No per-case results match these model filters.',{'text-anchor':'middle'});renderFactors();return}
 const spacings=[...new Set((data.resolution_evaluations||[]).flatMap(e=>e.domains.map(d=>d.cell_size_m)))].sort((a,b)=>b-a),spacingColor=s=>[color.teal,color.blue,color.amber,'#ec9cda'][spacings.indexOf(s)%4];
 for(const spacing of [...new Set(domains.map(d=>d.cell_size_m))].sort((a,b)=>b-a)){const label=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=spacingColor(spacing);label.append(swatch,document.createTextNode(spacing+' m spacing'));key.append(label)}
 const maxCells=Math.max(...domains.map(d=>d.cells)),radius=d=>18*Math.sqrt(d.cells/maxCells),maxRatio=Math.max(1.5,Math.ceil(Math.max(...domains.map(d=>d.aspect))*2)/2),ratioTicks=Array.from({length:6},(_,i)=>1+(maxRatio-1)*i/5);
 const plot={left:85,right:720,top:38,bottom:412},x=v=>plot.left+(v-1)/(maxRatio-1)*(plot.right-plot.left);
 const positives=domains.map(d=>d.score).filter(v=>v>0);let lo=log&&positives.length?Math.floor(Math.log10(arrayMin(positives))):0,hi=log&&positives.length?Math.ceil(Math.log10(arrayMax(positives))):Math.max(...domains.map(d=>d.score))*1.08;if(hi===lo)hi=lo+1;
 const y=v=>plot.bottom-((log?Math.log10(v):v)-lo)/(hi-lo)*(plot.bottom-plot.top),errorTicks=log?Array.from({length:hi-lo+1},(_,i)=>10**(lo+i)):Array.from({length:6},(_,i)=>hi*i/5);
 for(const tick of errorTicks){line(aspectSvg,plot.left,y(tick),plot.right,y(tick));text(aspectSvg,plot.left-10,y(tick)+4,sci(tick),{'text-anchor':'end'})}
 for(const tick of ratioTicks){line(aspectSvg,x(tick),plot.top,x(tick),plot.bottom);text(aspectSvg,x(tick),437,tick.toFixed(1),{'text-anchor':'middle'})}
 text(aspectSvg,400,470,'Longest side ÷ shortest side (1 = cube)',{'text-anchor':'middle','font-size':12});text(aspectSvg,20,225,'Worst equation error · '+(log?'logarithmic':'linear'),{transform:'rotate(-90 20 225)','text-anchor':'middle','font-size':12});
 const map={left:85,right:485,top:32,bottom:432},mx=v=>map.left+(v-1)/(maxRatio-1)*400,my=v=>map.bottom-(v-1)/(maxRatio-1)*400;
 mapSvg.append(el('path',{d:`M${mx(1)},${my(1)} L${mx(1)},${my(maxRatio)} L${mx(maxRatio)},${my(maxRatio)} Z`,fill:'#172936'}));
 for(const tick of ratioTicks){line(mapSvg,map.left,my(tick),map.right,my(tick));line(mapSvg,mx(tick),map.top,mx(tick),map.bottom);text(mapSvg,map.left-10,my(tick)+4,tick.toFixed(1),{'text-anchor':'end'});text(mapSvg,mx(tick),457,tick.toFixed(1),{'text-anchor':'middle'})}
 line(mapSvg,mx(1),my(1),mx(maxRatio),my(maxRatio),color.native,'5 5');
 text(mapSvg,285,490,'Middle side ÷ shortest side',{'text-anchor':'middle','font-size':12});text(mapSvg,22,232,'Longest side ÷ shortest side',{transform:'rotate(-90 22 232)','text-anchor':'middle','font-size':12});
 const transformed=v=>log?(v>0?Math.log10(v):null):v,colorValues=domains.map(d=>transformed(d.score)).filter(v=>v!==null),colorMin=colorValues.length?arrayMin(colorValues):0,colorMax=colorValues.length?arrayMax(colorValues):1;
 const errorColor=t=>{t=Math.max(0,Math.min(1,t));const stops=[[96,223,195],[244,186,113],[239,109,157]],a=t<.5?stops[0]:stops[1],b=t<.5?stops[1]:stops[2],u=t<.5?t*2:(t-.5)*2;return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*u)).join(',')+')'};
 const difficultyColor=d=>errorColor(colorMax===colorMin?.5:((transformed(d.score)??colorMin)-colorMin)/(colorMax-colorMin));
 const defs=el('defs'),gradient=el('linearGradient',{id:'shape-error-gradient',x1:'0%',y1:'100%',x2:'0%',y2:'0%'});for(const offset of [0,.5,1])gradient.append(el('stop',{offset:offset*100+'%','stop-color':errorColor(offset)}));defs.append(gradient);mapSvg.append(defs);
 mapSvg.append(el('rect',{x:555,y:72,width:16,height:290,rx:4,fill:'url(#shape-error-gradient)'}));text(mapSvg,553,46,'Difficulty',{'font-size':12});
 for(const t of [0,.5,1]){const exponent=colorMin+t*(colorMax-colorMin),v=log?10**exponent:exponent;text(mapSvg,581,366-t*290,sci(v),{'font-size':11})}text(mapSvg,550,395,log?'Log color scale':'Linear color scale');text(mapSvg,550,414,'Teal: lower error');text(mapSvg,550,433,'Pink: higher error');
 const describe=d=>`${d.counts_xyz.join(' × ')} cells · spacing ${d.cell_size_m} m · ${d.cells.toLocaleString()} total cells · longest/shortest ${d.aspect.toFixed(4)} · middle/shortest ${d.middle.toFixed(4)} · difficulty ${sci(d.score)} (${aggregate?stat+' of model worst errors':'model worst error'}, ${d.measurements.length} model(s))`;
 const highest=d=>d.measurements.reduce((a,b)=>a.domain.worst_error>=b.domain.worst_error?a:b);
 function inspect(d){const worst=highest(d),link=document.createElement('a');link.href=worst.evaluation.source;link.textContent='Source cases';$('shape-detail').replaceChildren(document.createTextNode(describe(d)+' · highest-error model '+worst.evaluation.cohort.id+' '+worst.evaluation.cohort.name+' ('+sci(worst.domain.worst_error)+'). '),link);for(const svg of [aspectSvg,mapSvg])for(const p of svg.querySelectorAll('[data-shape-key]')){const selected=p.dataset.shapeKey===d.key;p.setAttribute('stroke',selected?'#ffffff':'#0b141c');p.setAttribute('stroke-width',selected?2.5:1)}}
 function dot(svg,cx,cy,fill,d,kind){const point=historyDot(svg,cx,cy,fill,describe(d));point.setAttribute('r',radius(d));point.setAttribute('fill-opacity',.85);point.setAttribute('stroke','#0b141c');point.setAttribute('stroke-width',1);point.dataset.shapeKey=d.key;point.dataset.shapeKind=kind;point.dataset.shapeScore=d.score;point.dataset.shapeAspect=d.aspect;point.dataset.shapeMiddle=d.middle;point.dataset.shapeCells=d.cells;for(const event of ['pointerenter','focus','click'])point.addEventListener(event,()=>inspect(d))}
 let zeros=0;for(const d of [...domains].sort((a,b)=>b.cells-a.cells||a.score-b.score)){if(!log||d.score>0)dot(aspectSvg,x(d.aspect),y(d.score),spacingColor(d.cell_size_m),d,'aspect');else zeros++;dot(mapSvg,mx(d.middle),my(d.aspect),difficultyColor(d),d,'map')}
 text(aspectSvg,plot.left,505,'Bubble area = total cells',{'font-size':11});
 const cellExamples=[Math.min(...domains.map(d=>d.cells)),Math.round(driftMedian(domains.map(d=>d.cells))),maxCells];cellExamples.forEach((cells,i)=>{const cx=plot.left+25+i*210;aspectSvg.append(el('circle',{cx,cy:537,r:18*Math.sqrt(cells/maxCells),fill:color.native,'fill-opacity':.5}));text(aspectSvg,cx+26,541,cells.toLocaleString()+' cells')});
 if(zeros)$('shape-coverage').textContent+=` ${zeros} exact-zero errors are omitted from the logarithmic error axis; use linear to show them. Zero errors use the lowest difficulty color.`;
 for(const d of domains){const worst=highest(d),tr=document.createElement('tr');tr.dataset.shapeKey=d.key;for(const value of [d.counts_xyz.join(' × ')+' · '+d.cell_size_m+' m',d.cells.toLocaleString(),d.aspect.toFixed(4),d.middle.toFixed(4),sci(d.score),d.measurements.length]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),link=document.createElement('a');link.href=worst.evaluation.source;link.textContent=worst.evaluation.cohort.id+' · '+worst.evaluation.cohort.name;td.append(link);tr.append(td);body.append(tr)}
 renderFactors();
}
$('shape-family').addEventListener('change',selectShapeModels);
for(const id of ['shape-model','shape-spacing','shape-score','shape-scale'])$(id).addEventListener('change',renderShapes);
$('shape-scope').addEventListener('change',selectShapeModels);
const resolutionHidden=new Set();
function resolutionDomainKey(d){return JSON.stringify([d.counts_xyz,d.cell_size_m])}
function resolutionModelPool(){
 const scope=$('resolution-scope').value;
 return(data.resolution_evaluations||[]).map(e=>({...e,cohort:driftCohorts.get(e.weights_sha256)})).filter(e=>cohortMatchesScope(e.cohort,scope,'domains')&&($('resolution-family').value==='all'||e.cohort.family===$('resolution-family').value));
}
function resolutionAnalysis(){
 const pool=resolutionModelPool(),models=pool.filter(e=>!resolutionHidden.has(e.weights_sha256)),ranking=$('resolution-reference').value==='all'?(data.resolution_evaluations||[]):models;
 const grouped=new Map();
 for(const e of ranking)for(const d of e.domains){const key=resolutionDomainKey(d);if(!grouped.has(key))grouped.set(key,{key,counts_xyz:d.counts_xyz,cell_size_m:d.cell_size_m,errors:[]});grouped.get(key).errors.push(d.worst_error)}
 let domains=[...grouped.values()].filter(d=>$('resolution-spacing').value==='all'||String(d.cell_size_m)===$('resolution-spacing').value).map(d=>({...d,score:$('resolution-score').value==='maximum'?Math.max(...d.errors):driftMedian(d.errors)})).sort((a,b)=>b.score-a.score||a.key.localeCompare(b.key));
 const total=domains.length;domains=domains.map((d,i)=>({...d,rank:i+1}));if($('resolution-limit').value!=='all')domains=domains.slice(0,Number($('resolution-limit').value));
 return{pool,models,ranking,domains,total};
}
function setupResolution(){
 const families=[...new Set((data.resolution_evaluations||[]).map(e=>driftCohorts.get(e.weights_sha256)?.family).filter(Boolean))].sort();
 for(const family of families){const o=document.createElement('option');o.value=family;o.textContent=family;$('resolution-family').append(o)}
 const spacings=[...new Set((data.resolution_evaluations||[]).flatMap(e=>e.domains.map(d=>d.cell_size_m)))].sort((a,b)=>b-a);
 for(const spacing of spacings){const o=document.createElement('option');o.value=spacing;o.textContent=spacing+' m';$('resolution-spacing').append(o)}
 renderResolution();
}
function renderResolution(){

 const{pool,models,ranking,domains,total}=resolutionAnalysis(),svg=$('resolution-chart'),key=$('resolution-key'),body=$('resolution-values');svg.replaceChildren();key.replaceChildren();body.replaceChildren();
 $('resolution-detail').textContent='Hover or focus a point to inspect its worst saved-state result.';
 const available=(data.resolution_evaluations||[]).length,excluded=data.cohort_points.length-available;
 $('resolution-coverage').textContent=`${models.length} models shown / ${pool.length} under current filters · ${domains.length} / ${total} resolutions · ${available} checkpoints with matching per-case evidence; ${excluded} historical checkpoints use other populations. Difficulty ranks use ${ranking.length} ${$('resolution-reference').value==='all'?'comparable models and remain fixed when filtering models':'shown models and update with model filters'}.`;
 const scoreLabel=$('resolution-score').value==='maximum'?'maximum':'median';
 $('resolution-axis-note').textContent=`X: resolutions ordered by descending ${scoreLabel} of each reference model’s worst error (hardest → easiest). Y: each model’s maximum relative equation residual at that resolution; lower is more accurate. ${$('resolution-scale').value==='log'?'Logarithmic':'Linear'} error scale.`;
 $('resolution-finding').textContent=domains.length?`Hardest observed resolutions: ${domains.slice(0,3).map(d=>'#'+d.rank+' '+d.counts_xyz.join(' × ')+' at '+d.cell_size_m+' m ('+sci(d.score)+')').join('; ')}. Scores are ${scoreLabel} model worst errors across ${ranking.length} reference models.`:'No resolutions match these filters. Select a model or broaden the filters.';
 for(const e of pool){const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('i');input.type='checkbox';input.checked=!resolutionHidden.has(e.weights_sha256);input.setAttribute('aria-label','Show '+e.cohort.id+' '+e.cohort.name);input.addEventListener('change',()=>{input.checked?resolutionHidden.delete(e.weights_sha256):resolutionHidden.add(e.weights_sha256);renderResolution()});swatch.style.borderColor=driftColor(e);label.append(input,swatch,document.createTextNode(e.cohort.id+' · '+e.cohort.name));key.append(label)}
 const width=Math.max(1160,domains.length*74+150);svg.setAttribute('viewBox',`0 0 ${width} 620`);svg.style.minWidth=width+'px';
 if(!domains.length||!models.length){text(svg,580,230,domains.length?'Select a model in the legend to display its results.':'No resolutions match these filters.',{'text-anchor':'middle'});return}
 const lookups=new Map(models.map(e=>[e.weights_sha256,new Map(e.domains.map(d=>[resolutionDomainKey(d),d]))]));
 const values=models.flatMap(e=>domains.map(d=>lookups.get(e.weights_sha256).get(d.key)?.worst_error)).filter(Number.isFinite),log=$('resolution-scale').value==='log',positive=values.filter(v=>v>0);
 let lo=log?Math.floor(Math.log10(arrayMin(positive))):0,hi=log?Math.ceil(Math.log10(arrayMax(positive))):arrayMax(values)*1.08;
 if(!Number.isFinite(lo))lo=-1;if(!Number.isFinite(hi))hi=1;if(hi===lo)hi=lo+1;
 const plot={left:90,right:width-45,top:38,bottom:402},x=i=>domains.length===1?(plot.left+plot.right)/2:plot.left+i/(domains.length-1)*(plot.right-plot.left),y=v=>plot.bottom-((log?Math.log10(v):v)-lo)/(hi-lo)*(plot.bottom-plot.top);
 const ticks=log?Array.from({length:hi-lo+1},(_,i)=>10**(lo+i)):Array.from({length:6},(_,i)=>hi*i/5);
 for(const tick of ticks){line(svg,plot.left,y(tick),plot.right,y(tick));text(svg,plot.left-12,y(tick)+4,sci(tick),{'text-anchor':'end'})}
 domains.forEach((d,i)=>{line(svg,x(i),plot.top,x(i),plot.bottom);text(svg,x(i),plot.top-15,'#'+d.rank,{'text-anchor':'middle','data-resolution-rank':d.rank});text(svg,x(i),426,d.counts_xyz.join('×')+' · '+d.cell_size_m+' m',{'text-anchor':'end',transform:`rotate(-55 ${x(i)} 426)`,'font-size':11})});
 text(svg,(plot.left+plot.right)/2,598,'Hardest resolution → easiest resolution (ranked by observed error, not cell count)',{'text-anchor':'middle','font-size':12});
 let zeroCount=0;
 for(const e of models){const lookup=lookups.get(e.weights_sha256),stroke=driftColor(e);let path='',connected=false;
  for(let i=0;i<domains.length;i++){const measurement=lookup.get(domains[i].key),v=measurement?.worst_error;if(!Number.isFinite(v)||(log&&v<=0)){connected=false;if(v===0)zeroCount++;continue}path+=(connected?'L':'M')+x(i)+','+y(v)+' ';connected=true}
  const curve=el('path',{d:path,fill:'none',stroke,'stroke-width':1.6,opacity:.8,'data-resolution-model':e.cohort.id});svg.append(curve);
  domains.forEach((d,i)=>{const m=lookup.get(d.key);if(!m)return;
   const worst=m.worst_case,description=`#${d.rank} · ${d.counts_xyz.join(' × ')} · spacing ${d.cell_size_m} m · ${e.cohort.id} ${e.cohort.name} · worst error ${sci(m.worst_error)} across ${m.cases} cases · worst snapshot ${worst.trajectory_id}, physical frame ${worst.native_frame} · difficulty score ${sci(d.score)} (${scoreLabel}, ${ranking.length} models)`;
   if(!log||m.worst_error>0){const dot=historyDot(svg,x(i),y(m.worst_error),stroke,description);dot.setAttribute('r',3.5);dot.dataset.resolutionKey=d.key;dot.dataset.resolutionModel=e.cohort.id;dot.dataset.worstError=m.worst_error;for(const event of ['pointerenter','focus'])dot.addEventListener(event,()=>$('resolution-detail').textContent=description)}
   const tr=document.createElement('tr');tr.dataset.resolutionRank=d.rank;tr.dataset.resolutionModel=e.cohort.id;
   for(const v of [`#${d.rank} · ${d.counts_xyz.join(' × ')} · ${d.cell_size_m} m`,sci(d.score),e.cohort.id+' · '+e.cohort.name,sci(m.worst_error),m.cases,worst.trajectory_id+' · frame '+worst.native_frame]){const td=document.createElement('td');td.textContent=v;tr.append(td)}
   tr.children[2].style.color=stroke;const evidence=document.createElement('td'),link=document.createElement('a');link.href=e.source;link.textContent='Per-case results';evidence.append(link);tr.append(evidence);body.append(tr);
  });
 }
 // Keep the evidence table in the same hardest-to-easiest order as the axis.
 body.append(...[...body.children].sort((a,b)=>Number(a.dataset.resolutionRank)-Number(b.dataset.resolutionRank)||a.dataset.resolutionModel.localeCompare(b.dataset.resolutionModel)));
 if(zeroCount)$('resolution-axis-note').textContent+=` ${zeroCount} exact-zero result(s) are listed in the table; choose the linear axis to plot them.`;
}
for(const id of ['resolution-family','resolution-spacing','resolution-limit','resolution-reference','resolution-score','resolution-scale'])$(id).addEventListener('change',renderResolution);
$('resolution-show-all').addEventListener('click',()=>{resolutionHidden.clear();renderResolution()});
$('resolution-hide-all').addEventListener('click',()=>{for(const e of resolutionModelPool())resolutionHidden.add(e.weights_sha256);renderResolution()});
$('resolution-scope').addEventListener('change',renderResolution);
