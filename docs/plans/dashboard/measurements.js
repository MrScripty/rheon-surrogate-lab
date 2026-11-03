/* Pure measurement rules shared by the dashboard and Node verification. */
const Evaluation = (() => {
 function analyze(points, weight=.5) {
  const valid=points.filter(p=>Number.isFinite(p.inference_median_ms)&&p.inference_median_ms>0&&Number.isFinite(p.worst_error)&&p.worst_error>0);
  const keys=new Set(valid.map(p=>p.comparison_population));
  if(!valid.length||keys.size!==1||keys.has(undefined)||valid.some(p=>p.ranking_eligible===false||(p.ranking_eligible!==true&&p.measurement_scope!=='complete-hybrid-policy')))return {ranked:[],frontier:[],weight};
  const score=p=>weight*Math.log(p.worst_error)+(1-weight)*Math.log(p.inference_median_ms);
  const ranked=[...valid].sort((a,b)=>score(a)-score(b)||a.inference_median_ms-b.inference_median_ms);
  const frontier=valid.filter(p=>!valid.some(q=>q.worst_error<=p.worst_error&&q.inference_median_ms<=p.inference_median_ms&&(q.worst_error<p.worst_error||q.inference_median_ms<p.inference_median_ms))).sort((a,b)=>a.inference_median_ms-b.inference_median_ms);
  return {ranked,frontier,weight};
 }
 function pressurePoints(data,benchmark,execution='original') {
  if(benchmark==='historical')return data.cohort_points.filter(p=>data.solver_comparison?.classification[p.id]?.kind==='full').map(p=>({...p,ranking_eligible:false,measurement_label:'Historical captured proposal'}));
  const group=data.evaluation.benchmarks.find(b=>b.id===benchmark);
  return (group?.measurements||[]).filter(r=>execution==='all'||(execution==='original'?(!r.execution.includes('/')||r.execution.endsWith('/original')):r.execution.includes('/')&&!r.execution.endsWith('/original'))).map(r=>{
   const p=data.cohort_points.find(p=>p.id===r.cohort);
   // Fresh populations must not inherit historical best scores or case counts.
   const accuracy=r.historical_population?{}:{best_error:null,best_error_raw:null,best_error_status:'unavailable',best_error_eligible_cases:null,best_error_excluded_cases:null,cases:null,timed_cases:null,timing_cases:[]};
   return {...p,...accuracy,...r,measurement_label:r.execution,policy:'Recorded execution: '+r.execution};
  });
 }
 return {analyze,pressurePoints};
})();
if(typeof module!=='undefined')module.exports=Evaluation;
