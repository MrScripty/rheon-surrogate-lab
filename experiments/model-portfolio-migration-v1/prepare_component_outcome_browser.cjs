'use strict';
// Observe the published notes through the real model guide and captured cards.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {modelQuery}=require('./matched_model_query.cjs');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04'),job=path.join(run,'native-component-outcome-adoption-01');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8')),hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const binding=p=>({path:path.relative(root,p),sha256:hash(fs.readFileSync(p))});
const destination=path.join(job,'browser-actions.json'),expectedPath=path.join(job,'browser-expectations.json');
assert.ok(!fs.existsSync(destination)&&!fs.existsSync(expectedPath),'Preserve first browser preparation');
const auditPath=path.join(root,'docs/plans/model-portfolio-migration-v1/reports/native-component-outcome-adoption-01.json');
const audit=read(auditPath),header=read(path.join(project,'.tuldok/project.json'));
assert.equal(audit.status,'native23component_outcome_notes_exact_and_scientific_records_unchanged');assert.equal(header.checkpoint,audit.checkpoint);
const prepPath=path.join(run,'native-component-outcome-notes-01/preparation.json'),prep=read(prepPath);
const guide=read(path.join(project,'notes/00-models.portfolio.json'));
const targets=new Set(prep.notes.filter(n=>n.fit_item_id).map(n=>n.fit_item_id));assert.equal(targets.size,20);
const actions=[],add=(kind,data={})=>actions.push({kind,...data});
add('assert_text',{text:audit.checkpoint});add('assert_text',{text:'Experiment progression'});
add('click',{selector:'#project-portfolio-notes'});
add('wait_text',{selector:'#project-portfolio-note-count',text:'1 organization notes found'});
add('wait_enabled',{selector:'#project-portfolio-note'});
add('fill',{selector:'#project-portfolio-note',value:prep.notes.find(n=>n.path==='notes/00-models.portfolio.json').file_id});
add('click',{selector:'#project-portfolio-load'});
add('wait_text',{selector:'#project-portfolio-status',text:'Loaded retained authored model organization.'});
add('wait_text',{selector:'#project-portfolio-count',text:'205 models in this guide; 205 source reviewed.'});
add('wait_text',{selector:'#project-model-portfolio',text:'revision '+prep.guide_revision});
add('wait_text',{selector:'#project-portfolio-description',text:guide.description});
add('scroll',{selector:'#project-portfolio-title'});add('screenshot',{name:'component-205-model-guide.png'});
let summaryChecks=0,leadChecks=0;
const ambiguous=new Set(['fit-0b2b6b6a8c42a47a','fit-4ca6c279be34e629']);
for(const model of guide.models.filter(m=>targets.has(m.id))){
  // These two imported names genuinely collide. Select their explicit graph IDs;
  // do not pretend that a shared historical label uniquely identifies a fit.
  const query=ambiguous.has(model.id)?'execution':modelQuery(guide,model);
  const matching=guide.models.filter(m=>[m.label,...m.aliases].some(v=>v.toLowerCase().includes(query.toLowerCase())));
  assert.ok(matching.some(m=>m.id===model.id));assert.equal(matching.length,ambiguous.has(model.id)?2:1);
  for(const[name,value]of[['group',''],['search',query],['review','source_reviewed']])add('fill',{selector:'#project-portfolio-'+name,value});
  add('click',{selector:'#project-portfolio-filter-form button'});
  add('wait_text',{selector:'#project-portfolio-count',text:matching.length+' match the applied filters.'});
  add('click',{selector:'[data-lineage-node="model/'+model.id+'"]'});
  add('wait_text',{selector:'#project-portfolio-status',text:'Exact retained model summary and captured evidence.'});
  add('wait_text',{selector:'#project-portfolio-model-heading',text:model.label});
  for(const field of ['method','purpose','experiment','outcome','lesson','reproduction']){
    add('wait_text',{selector:'#project-portfolio-detail',text:model.summary[field]});summaryChecks++;
  }
  if(['fit-10d3a512b41ab2a6','fit-d18b434d52b73791'].includes(model.id)){
    add('scroll',{selector:'#project-portfolio-detail h4:nth-of-type(4)'});
    add('screenshot',{name:model.id+'-outcome.png'});
    add('click',{selector:'#project-portfolio-detail details > summary',text:'Exact captured card and original receipts'});
    for(const text of ['Accuracy at every physical output','Exact original receipt bytes','native portable scalar/source closure remains pending'])
      add('wait_text',{selector:'#project-portfolio-detail',text});
    const text=fs.readFileSync(path.join(project,model.card.path),'utf8');
    for(const line of text.split('\n').filter(line=>/^\| (30|60|120) \|/.test(line)&&line.includes(' | pressure_relative_l2 | '))){
      add('wait_text',{selector:'#project-portfolio-detail',text:line});leadChecks++;
    }
    add('scroll',{selector:'#project-portfolio-detail details[open] > summary'});
    add('screenshot',{name:model.id+'-physical-lead-card.png'});
  }
}
assert.equal(summaryChecks,120);assert.equal(leadChecks,60);
const owned=[];
for(const entry of header.inventory)if(entry.family==='project_file'&&entry.namespace===header.project_id){
  const doc=read(path.join(project,'.tuldok',entry.path));if(doc.project_id===header.project_id)owned.push(doc.path);
}
owned.sort();const overviewIndex=owned.indexOf('notes/portfolio-overview.md');assert.ok(overviewIndex>=0);
add('click',{selector:'#project-working-load'});
for(let start=0;start<=overviewIndex;start+=40){
  if(start){add('wait_enabled',{selector:'#project-working-more'});add('click',{selector:'#project-working-more'});}
  add('wait_text',{selector:'#project-working-list',text:owned[Math.min(start+39,owned.length-1)]});
}
add('click',{selector:'#project-working-list button',text:'notes/portfolio-overview.md'});
for(const text of ['205 accuracy outcomes, 195 qualified timings','Every physical output retains its own accuracy',
  'Native chart publication','Prediction deletion remains paused']){
  add('wait_value',{selector:'#project-working-text',text});
}
add('scroll',{selector:'#project-working-heading'});add('screenshot',{name:'component-updated-overview.png'});
fs.writeFileSync(destination,JSON.stringify(actions,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(expectedPath,JSON.stringify({updated_model_summaries:20,summary_paragraph_assertions:summaryChecks,
  pressure_lead_rows_in_actual_card:leadChecks,guide_revision:prep.guide_revision,guide_models:205,ancestry_unchanged:true,
  known_ambiguous_legacy_labels:[...ambiguous],native_chart_publication_pending:true,portable_scalar_source_closure_complete:false,
  preparation:binding(prepPath),custody:binding(auditPath),producer:binding(__filename)},null,2)+'\n',{flag:'wx'});
console.log(actions.length+' real UI actions prepared:20summaries and60physical pressure-lead rows.');
