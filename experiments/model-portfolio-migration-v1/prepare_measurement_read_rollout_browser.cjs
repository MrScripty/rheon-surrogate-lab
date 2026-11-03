'use strict';
// Reuse the original three real chart selections, adding exact scalar checks.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1');
const job=path.join(run,'native-measurement-source-read-rollout-02');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const header=read(path.join(root,'projects/rheon-model-portfolio-unified-04/.tuldok/project.json'));
const custody=read(path.join(job,'before-custody.json'));
assert.equal(header.checkpoint,custody.facts.checkpoint);
const original=path.join(run,'project-read-performance-live-integration-05/timing-browser-actions.json');
const output=path.join(job,'browser-actions.json'),expectationPath=path.join(job,'browser-expectations.json');
assert.ok(!fs.existsSync(output)&&!fs.existsSync(expectationPath),'Preserve original UI preparation');
const originalActions=read(original),actions=[],expectations=[];
let table=null,filters={},metric=null,sourceIdentity=null,numericChecks=0;
const sourcePage=read(path.join(run,'measurement-source-reads-native-baseline-01/sources.response.json')).result.value;
assert.equal(sourcePage.total,31);assert.equal(sourcePage.items.length,31);
for(const originalAction of originalActions){
  const action=structuredClone(originalAction);
  if(action.kind==='assert_text')action.text=header.checkpoint;
  if(action.selector==='#project-measurements-source-page')action.text='1–31 of 31 sources';
  if(action.kind==='select_values'&&action.selector==='#project-measurements-source'){
    assert.equal(action.values.length,1);sourceIdentity=action.values[0];
    const source=sourcePage.items.find(item=>item.origin_catalog_id+'.'+item.id===sourceIdentity);assert.ok(source);
    const raw=fs.readFileSync(path.join(root,'projects/rheon-model-portfolio-unified-04/artifacts/sha256',source.blob.sha256));
    assert.equal(raw.length,source.blob.bytes);assert.equal(hash(raw),source.blob.sha256);
    table=JSON.parse(raw);filters={};metric=null;
  }
  if(action.kind==='fill'&&action.selector==='#project-measurements-metric')metric=action.value;
  if(action.kind==='fill'&&action.selector.startsWith('#project-measurements-filter-value-')){
    const index=Number(action.selector.split('-').at(-1));
    const dimension=table.dimensions[index];assert.ok(dimension);
    filters[dimension.id]=action.value;
  }
  actions.push(action);
  if(action.kind==='wait_text'&&action.selector==='#project-measurements-status'&&action.text==='Exact retained comparison.'){
    assert.ok(table&&metric);
    const cases=new Set(table.cases.filter(item=>Object.entries(filters).every(([key,value])=>String(item.dimensions[key])===value)).map(item=>item.id));
    assert.ok(cases.size);
    const maxima=[];
    for(const [index,subject]of table.subjects.entries()){
      const rows=table.observations.filter(row=>row.subject_id===subject.id&&row.metric_id===metric&&cases.has(row.case_id));
      assert.equal(rows.length,cases.size);assert.ok(rows.every(row=>row.state==='measured'&&Number.isFinite(row.value)));
      const maximum=Math.max(...rows.map(row=>row.value));
      actions.push({kind:'wait_text',selector:`#project-measurements-maxima tbody tr:nth-child(${index+1}) td:nth-child(2) button`,
        text:String(maximum)+' · inspect maximum for '+subject.label});numericChecks++;
      maxima.push({subject_id:subject.id,label:subject.label,maximum});
    }
    expectations.push({source:sourceIdentity,metric_id:metric,filters:structuredClone(filters),cases_per_subject:cases.size,maxima});
  }
}
assert.equal(expectations.length,3);assert.equal(numericChecks,14);
fs.writeFileSync(output,JSON.stringify(actions,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(expectationPath,JSON.stringify({status:'three_original_chart_selections_prepared_with14_exact_scalar_assertions',
  checkpoint:header.checkpoint,original_actions:{path:path.relative(root,original),sha256:hash(fs.readFileSync(original))},
  actions_sha256:hash(fs.readFileSync(output)),native_sources:31,chart_selections:3,numeric_presence_assertions:numericChecks,
  screenshots:actions.filter(action=>action.kind==='screenshot').map(action=>action.name),expectations,
  source_or_scientific_payload_copies:0,model_forwards:0,actual_UI_qualified:false,producer_sha256:hash(fs.readFileSync(__filename))},null,2)+'\n',{flag:'wx'});
console.log(actions.length+' actions, 3 original charts, 14 literal numerical assertions prepared.');
