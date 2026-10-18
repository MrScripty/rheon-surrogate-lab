const precisionColumns=Object.fromEntries((data.precision?.point_columns||[]).map((name,i)=>[name,i]));
function precisionValue(point,name){return point[precisionColumns[name]]}
function precisionTrace(){return data.precision?.traces.find(t=>t.id===$('precision-scene').value)}
function precisionNumber(value){return Number.isFinite(value)?(value===0?'0':value.toExponential(4)):'undefined (zero reference norm)'}
function precisionPlot(id,points,series,unit){
 const svg=$(id),key=$(id+'-key'),log=$('precision-scale').value==='log';svg.replaceChildren();key.replaceChildren();
 const values=series.flatMap(s=>points.map(p=>s.value(p))).filter(Number.isFinite),positive=values.some(v=>v>0);
 for(const s of series){const label=document.createElement('span');label.style.color=s.color;label.textContent=s.label+(log&&points.every(p=>s.value(p)===0)?' · exact zero (not on log axis)':'');key.append(label)}
 if(log&&!positive){text(svg,580,220,'All measured values are exactly zero.',{'text-anchor':'middle','font-size':18});text(svg,580,250,'Select a linear scale to display zero curves.',{'text-anchor':'middle'});return}
 const axes=passAxes(svg,[0,points.at(-1)[precisionColumns.step]],passBounds(values,log),false,log,'Simulation step · Δt = 1/60 s');
 text(svg,axes.left,17,unit,{'font-size':12});
 for(const s of series){let d='';for(const p of points){const v=s.value(p);if(!Number.isFinite(v)||(log&&v<=0)){d+=' ';continue}const prefix=d.trim()&&!d.endsWith(' ')?'L':'M';d+=prefix+axes.x(precisionValue(p,'step'))+','+axes.y(v)}
  svg.append(el('path',{d:d.trim(),fill:'none',stroke:s.color,'stroke-width':1.8,'stroke-dasharray':s.dash||''}));
  const selected=Number($('precision-step').value);
  for(const p of points){const step=precisionValue(p,'step'),v=s.value(p);if(!Number.isFinite(v)||(log&&v<=0)||!(step===1||step%16===0||step===selected||step===points.length))continue;
   historyDot(svg,axes.x(step),axes.y(v),s.color,`${s.label} · step ${step} · ${precisionValue(p,'time_s').toFixed(4)} s · ${precisionNumber(v)} ${unit}`)}
 }
 const step=Number($('precision-step').value);svg.append(el('line',{x1:axes.x(step),x2:axes.x(step),y1:axes.top,y2:axes.bottom,stroke:color.muted,'stroke-dasharray':'3 5','stroke-width':1}));
}
function renderPrecision(){
 const trace=precisionTrace();if(!trace)return;
 const field=$('precision-field').value,relative=$('precision-norm').value==='relative',unit=relative?'relative L2 (dimensionless)':({pressure:'Pa',velocity:'m/s',tracer:'tracer fraction'})[field];
 const column=field+(relative?'_relative_l2':({pressure:'_rms_pa',velocity:'_rms_m_s',tracer:'_rms'})[field]);
 $('precision-accum-title').textContent=`Accumulated ${field} difference · ${relative?'relative L2':'absolute RMS'} (${unit})`;
 precisionPlot('precision-accum',trace.points,[{label:'Float32 − float64',color:color.teal,value:p=>precisionValue(p,column)},{label:'Float64 − repeated float64',color:color.amber,value:()=>0,dash:'5 4'}],unit);
 const stages=field==='tracer'?[['tracer_cast_rms','Tracer advection']]:[['advection_cast_rms','Velocity advection'],['force_cast_rms','Force'],['viscosity_cast_rms','Viscosity'],['projection_cast_rms','Pressure projection']];
 const localUnit=field==='tracer'?'tracer fraction':'m/s';
 $('precision-local-title').textContent=`Local ${field==='tracer'?'tracer':'velocity'} storage rounding · absolute RMS (${localUnit})`;
 precisionPlot('precision-local',trace.points,stages.map(([column,label],i)=>({label,color:[color.teal,color.amber,color.blue,'#e4a5ee'][i],value:p=>precisionValue(p,column)})),localUnit);
 precisionPlot('precision-div',trace.points,[['projection_div_before','Before float32 cast',color.blue],['projection_div_after','After float32 storage',color.amber],['projection_div_cast','D(casting perturbation)',color.teal]].map(([column,label,color])=>({label,color,value:p=>precisionValue(p,column)})),'s⁻¹ · maximum absolute divergence');
 const p=trace.points[Number($('precision-step').value)-1],step=precisionValue(p,'step');
 $('precision-detail').textContent=`Step ${step} · ${precisionValue(p,'time_s').toFixed(4)} s · ${field} difference ${precisionNumber(precisionValue(p,column))} ${unit} · projection cast RMS ${precisionNumber(precisionValue(p,'projection_cast_rms'))} m/s · D(cast) ${precisionNumber(precisionValue(p,'projection_div_cast'))} s⁻¹`;
 const s=trace.scene,h=1/s.cells_per_metre,extent=s.counts.map(n=>+(n*h).toFixed(4));
 $('precision-physics').textContent=`${s.counts.join(' × ')} cells · ${extent.join(' × ')} m · cubic h = ${h} m · water 1000 kg/m³ · μ = 0.001 Pa·s · Δt = 1/60 s · origin (0,0,0) · closed free-slip walls. Force base center (${s.center.join(', ')}) as fractions of domain extents (oscillating), radius scale ${s.radius} × shortest extent, acceleration ${s.acceleration} m/s², swirl ${s.swirl}, fluctuation ${s.fluctuation}, frequency ${s.frequency} Hz.`;
 $('precision-status').textContent=`${trace.points.length} retained steps · float32 canonical fields: ${trace.summary.canonical_f32_exact_steps}/${trace.points.length} bit-exact · float64 repeat: ${trace.summary.f64_repeat_exact_steps}/${trace.points.length} exact zero. ${$('precision-scale').value==='log'?'Zero values are omitted from logarithmic axes; no artificial error floor is added.':'Zero values are shown on linear axes.'}`;
 $('precision-receipt').textContent=`${data.precision.verified_cast_values.toLocaleString()} casts checked across four scenes. Study elapsed ${data.precision.execution.elapsed_s.toFixed(2)} s with up to three single-core workers. This includes control replays and is not an inference-speed benchmark.`;
 const tab=$('tab-state-precision');if(tab)tab.textContent=trace.label;
}
function setupPrecision(){
 const traces=data.precision?.traces||[];
 if(!traces.length){$('precision-status').textContent='Precision study has no qualified records yet.';return}
 for(const trace of traces)$('precision-scene').append(new Option(trace.label,trace.id));
 function sceneChanged(){const trace=precisionTrace();$('precision-step').max=trace.points.length;$('precision-step').value=trace.points.length;renderPrecision()}
 $('precision-scene').addEventListener('change',sceneChanged);
 for(const id of ['precision-field','precision-norm','precision-scale'])$(id).addEventListener('change',renderPrecision);
 $('precision-step').addEventListener('input',renderPrecision);sceneChanged();
}
