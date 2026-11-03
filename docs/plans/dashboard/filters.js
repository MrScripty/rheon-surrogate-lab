function frontierMode(){return $('cohort-filter').value==='frontier'}
let frontierWeightCache;
function frontierWeights(){return frontierWeightCache??=new Set(['full','hybrid'].flatMap(kind=>(kind==='full'?tradeoffAnalysis(pressureBenchmarkPoints(data.evaluation.default_benchmark,'original')).frontier:[])).concat(tradeoffAnalysis(data.solver_comparison?.hybrid_policies||[]).frontier,tradeoffAnalysis(data.solver_comparison?.hybrid_v6_policies||[]).frontier).map(p=>p.weights_sha256))}
// Cohort ranges and color-coded training groups are independent for each analysis.
const cohortFilterStates=new Map();
const cohortNumber=p=>Number(p.id.slice(1));
const cohortIDsByNumber=[...data.cohort_points].sort((a,b)=>cohortNumber(a)-cohortNumber(b));
function defaultCohortFilter(){return{from:cohortIDsByNumber[0].id,to:cohortIDsByNumber.at(-1).id,group:'all'}}
function cohortFilterState(tab){return cohortFilterStates.get(tab)||defaultCohortFilter()}
function cohortMatchesSelection(cohort,tab){
 const state=cohortFilterState(tab),defaults=defaultCohortFilter();
 if(!cohort)return state.from===defaults.from&&state.to===defaults.to&&state.group==='all';
 const number=cohortNumber(cohort);
 return number>=Number(state.from.slice(1))&&number<=Number(state.to.slice(1))&&(state.group==='all'||cohort.family===state.group);
}
function cohortFilterSummary(tab){
 const s=cohortFilterState(tab),d=defaultCohortFilter(),parts=[];
 if(s.from!==d.from||s.to!==d.to)parts.push(s.from+'–'+s.to);
 if(s.group!=='all')parts.push(s.group);
 return parts.length?' · '+parts.join(' · '):'';
}
function updateCohortFilterNote(tab){
 const state=cohortFilterState(tab),count=data.cohort_points.filter(p=>cohortMatchesSelection(p,tab)).length;
 $('cohorts-'+tab+'-summary').textContent='Cohort range & training group · '+state.from+'–'+state.to+' · '+(state.group==='all'?'all groups':state.group);
 $('cohorts-'+tab+'-status').textContent=count+' / '+data.cohort_points.length+' trained cohorts selected · '+state.from+'–'+state.to+(state.group==='all'?' · all training groups':' · '+state.group)+'. Charts use the saved measurements available for these cohorts.';
}
function setupCohortSelectors(){
 const refresh={overview:renderCohortPlots,progress:renderTrainingProgress,partial:updatePartialComparison,policies:renderStoppingPolicies,outcomes:renderLatest,domains:renderResolution,factors:selectShapeModels,passsweep:setupPassSweep,drift:selectDriftBuild,spatial:selectFrequencyModels,spectrum:selectTemporalModels,history:renderAllHistory,methods:setupMethodsStudy,methodsv2:setupMethodsV2,methodsv3:setupMethodsV3,methodsv4:setupMethodsV4,methodsv5:setupMethodsV5,methodsv6:setupMethodsV6,fullai:setupFullAIStudy};
 for(const panel of document.querySelectorAll('.analysis-panel[data-scope]')){
  const tab=panel.id.slice(6);if(!refresh[tab])continue;
  const state=defaultCohortFilter();cohortFilterStates.set(tab,state);
  const fieldset=document.createElement('fieldset');fieldset.className='cohort-selectors';fieldset.dataset.cohortSelectors=tab;
  const legend=document.createElement('legend');legend.textContent='Cohorts to view · this tab';fieldset.append(legend);
  const controls=document.createElement('div');controls.className='cohort-selector-fields';fieldset.append(controls);
  for(const [key,title]of [['from','From cohort'],['to','To cohort'],['group','Training group']]){
   const label=document.createElement('label');label.className='control';label.append(document.createTextNode(title));
   const select=document.createElement('select');select.id='cohorts-'+tab+'-'+key;
   if(key==='group'){
    select.append(new Option('All color-coded groups','all'));
    for(const group of [...new Set(data.cohort_points.map(p=>p.family))]){
     const ids=data.cohort_points.filter(p=>p.family===group).map(p=>p.id);
     select.append(new Option(group+' · '+ids.length+' cohort'+(ids.length===1?'':'s'),group));
    }
   }else for(const p of cohortIDsByNumber)select.append(new Option(p.id+' · '+p.name,p.id));
   select.value=state[key];label.append(select);controls.append(label);
   select.addEventListener('change',()=>{
    const from=$('cohorts-'+tab+'-from'),to=$('cohorts-'+tab+'-to');
    if(Number(from.value.slice(1))>Number(to.value.slice(1))){if(key==='from')to.value=from.value;else from.value=to.value}
    cohortFilterStates.set(tab,{from:from.value,to:to.value,group:$('cohorts-'+tab+'-group').value});
    // A cohort selection shows its full history; chart-specific subsets can be reapplied below.
    updateCohortFilterNote(tab);refresh[tab]();
   });
  }
  const reset=document.createElement('button');reset.type='button';reset.textContent='Reset selection';reset.id='cohorts-'+tab+'-reset';controls.append(reset);
  reset.addEventListener('click',()=>{
   const defaults=defaultCohortFilter();for(const key of ['from','to','group'])$('cohorts-'+tab+'-'+key).value=defaults[key];if(tab==='overview')clearCohortEmphasis();if(tab==='partial')clearPartialCohortEmphasis();
   $('cohorts-'+tab+'-from').dispatchEvent(new Event('change',{bubbles:true}));
  });
  const note=document.createElement('p');note.id='cohorts-'+tab+'-status';note.className='note';note.setAttribute('aria-live','polite');fieldset.append(note);
  const help=document.createElement('p');help.className='note';help.textContent='Inclusive cohort-ID range; skipped IDs add no results. Groups match the existing colors. Range, training group and chart subset intersect. Change or reset each control explicitly. Accuracy–speed filters use the common repeated pressure benchmark and separate complete hybrid protocols.';fieldset.append(help);
  const disclosure=document.createElement('details');disclosure.className='cohort-filter-disclosure';const summary=document.createElement('summary');summary.id='cohorts-'+tab+'-summary';disclosure.append(summary,fieldset);const anchor=panel.querySelector('.tab-filter-note');if(anchor)anchor.after(disclosure);else panel.prepend(disclosure);
  updateCohortFilterNote(tab);
 }
}

function cohortMatchesScope(cohort,scope,tab){return cohortMatchesSelection(cohort,tab)&&(scope==='all'||!!cohort&&(scope==='frontier'?frontierWeights().has(cohort.weights_sha256):cohortPopulationKey(cohort)===matchedPopulationKey))}
function cohortForRun(run){
 if(typeof run==='object'&&run.weights_sha256)return data.cohort_points.find(p=>p.weights_sha256===run.weights_sha256)||null;
 const name=typeof run==='string'?run:run.name;
 const aliases={'Previous single-grid checkpoint':'stability-modes-single-pass','single-pass-trained weights':'stability-modes-single-pass','Multi-grid · one-pass training':'multigrid-modes-single-pass','Multi-grid · two-pass training':'multigrid-modes-two-pass','16-simulation cohort · two-pass':'multigrid-cohort16-2pass','cohort-two-reference':'multigrid-cohort16-2pass','shape-two-reference':'cohort-two-shape-feedback','One-case fit':'fit-one','Longer original cohort':'long-256','Matched 256 cohort':'scale-256','Matched 1,024 cohort':'scale-1024'};
 const exact=data.cohort_points.filter(p=>name===p.name||name.startsWith(p.name+' ·'));
 if(exact.length===1)return exact[0];
 for(const [label,target] of Object.entries(aliases))if(name===label||name.startsWith(label+' ·'))return data.cohort_points.find(p=>p.name===target);
 if(typeof run==='object'){const bySource=data.cohort_points.filter(p=>p.source===run.source);if(bySource.length===1)return bySource[0]}
 return null;
}
function worstFirst(a,b,key='worst'){return (Number.isFinite(a[key])?a[key]:Infinity)-(Number.isFinite(b[key])?b[key]:Infinity)}
