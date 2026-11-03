'use strict';
// Check only changed notes;prior13chart checks remain under exact scientific custody.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {modelQuery}=require('./matched_model_query.cjs');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const job=path.join(run,'native-pressure-reproduction-status-adoption-01');
const notes=path.join(run,'native-pressure-reproduction-status-notes-01');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const output=path.join(job,'browser-actions.json'),expectationsPath=path.join(job,'browser-expectations.json');
assert.ok(!fs.existsSync(output)&&!fs.existsSync(expectationsPath),'Preserve original corrected-note browser actions');
const custody=read(path.join(root,'docs/plans/model-portfolio-migration-v1/reports/native-pressure-reproduction-status-adoption-01.json'));
assert.equal(custody.status,'native26pressure_outcome_notes_exact_and_scientific_records_unchanged');
assert.ok(custody.current_reproduction_status_corrected&&custody.measurement_sources_and_payloads_unchanged);
const header=read(path.join(project,'.tuldok/project.json'));assert.equal(header.checkpoint,custody.checkpoint);
const prep=read(path.join(notes,'preparation.json')),guide=read(path.join(project,'notes/00-models.portfolio.json'));
assert.equal(prep.guide_revision,9);assert.equal(guide.models.length,205);assert.equal(guide.relationships.length,26);
assert.equal(hash(fs.readFileSync(path.join(project,'notes/00-models.portfolio.json'))),hash(fs.readFileSync(path.join(notes,'notes/00-models.portfolio.json'))));
const actions=[],add=(kind,body={})=>actions.push({kind,...body});
add('assert_text',{text:custody.checkpoint});
add('click',{selector:'#project-portfolio-notes'});
add('wait_text',{selector:'#project-portfolio-note-count',text:'1 organization notes found'});
add('wait_enabled',{selector:'#project-portfolio-note'});
const guideNote=prep.notes.find(note=>note.path==='notes/00-models.portfolio.json');
add('fill',{selector:'#project-portfolio-note',value:guideNote.file_id});add('click',{selector:'#project-portfolio-load'});
add('wait_text',{selector:'#project-portfolio-status',text:'Loaded retained authored model organization.'});
add('wait_text',{selector:'#project-portfolio-count',text:'205 models in this guide; 205 source reviewed.'});
add('wait_text',{selector:'#project-model-portfolio',text:'revision 9'});
add('scroll',{selector:'#project-portfolio-title'});add('screenshot',{name:'pressure-status-guide9.png'});
const targets=new Set(prep.notes.filter(note=>note.fit_item_id).map(note=>note.fit_item_id));assert.equal(targets.size,23);
for(const model of guide.models.filter(item=>targets.has(item.id))){
  const query=modelQuery(guide,model);
  assert.ok(!model.summary.reproduction.includes('fresh qualified latency'));
  assert.ok(!model.summary.reproduction.includes('a current common benchmark for this unnamed fit remain unqualified'));
  for(const[name,value]of[['group',''],['search',query],['review','source_reviewed']])add('fill',{selector:'#project-portfolio-'+name,value});
  add('click',{selector:'#project-portfolio-filter-form button'});
  add('wait_text',{selector:'#project-portfolio-count',text:'1 match the applied filters.'});
  add('click',{selector:'[data-lineage-node="model/'+model.id+'"]'});
  add('wait_text',{selector:'#project-portfolio-status',text:'Exact retained model summary and captured evidence.'});
  add('wait_text',{selector:'#project-portfolio-model-heading',text:model.label});
  for(const key of['method','purpose','experiment','outcome','lesson','reproduction'])add('wait_text',{selector:'#project-portfolio-detail',text:model.summary[key]});
  add('scroll',{selector:'#project-portfolio-detail h4:nth-of-type(6)'});
  add('screenshot',{name:model.id+'-current-reproduction.png'});
}
const representative=guide.models.find(model=>model.id==='fit-8992e19d389ea016');
for(const[name,value]of[['group',''],['search',modelQuery(guide,representative)],['review','source_reviewed']])add('fill',{selector:'#project-portfolio-'+name,value});
add('click',{selector:'#project-portfolio-filter-form button'});add('click',{selector:'[data-lineage-node="model/'+representative.id+'"]'});
add('wait_text',{selector:'#project-portfolio-detail',text:representative.summary.reproduction});
add('click',{selector:'#project-portfolio-detail details > summary',text:'Exact captured card and original receipts'});
for(const text of ['Common GPU pressure benchmark','Superseded benchmark-stage reproduction statement','Exact original receipt bytes'])
  add('wait_text',{selector:'#project-portfolio-detail',text});
add('scroll',{selector:'#project-portfolio-detail details[open] > summary'});add('screenshot',{name:'pressure-status-card-original-receipt.png'});
const owned=[];
for(const entry of header.inventory)if(entry.family==='project_file'){
  const doc=read(path.join(project,'.tuldok',entry.path));if(doc.project_id===header.project_id)owned.push(doc.path);
}
owned.sort();const index=owned.indexOf('notes/portfolio-overview.md');assert.ok(index>=0);
add('click',{selector:'#project-working-load'});
for(let start=0;start<=index;start+=40){
  if(start){add('wait_enabled',{selector:'#project-working-more'});add('click',{selector:'#project-working-more'});}
  add('wait_text',{selector:'#project-working-list',text:owned[Math.min(start+39,owned.length-1)]});
}
add('click',{selector:'#project-working-list button',text:'notes/portfolio-overview.md'});
const overview=fs.readFileSync(path.join(notes,'notes/portfolio-overview.md'),'utf8');
// Exact full captured source,including newlines;no guessed cross-line substrings.
add('wait_value',{selector:'#project-working-text',text:overview});
add('scroll',{selector:'#project-working-heading'});add('screenshot',{name:'pressure-status-readable-overview.png'});
add('assert_text',{text:'Experiment progression'});
fs.writeFileSync(output,JSON.stringify(actions,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(expectationsPath,JSON.stringify({current_checkpoint:custody.checkpoint,updated_model_summaries:23,
  summary_paragraph_assertions:138,guide_revision:9,source_card_and_original_receipts_checked:true,overview_exact_full_text_checked:true,
  no_GPU_retiming:true,prior_chart_actions_unchanged:true,producer_sha256:hash(fs.readFileSync(__filename))},null,2)+'\n',{flag:'wx'});
console.log(actions.length+' current-guide9/status-note actions prepared;23summaries/138paragraphs/sourcecard/exact overview;no chart reexecution.');
