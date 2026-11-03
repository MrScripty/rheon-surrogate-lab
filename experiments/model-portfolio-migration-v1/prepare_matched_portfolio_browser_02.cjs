'use strict';
const {modelQuery}=require('./matched_model_query.cjs');
// Prepare actual UI interactions only after both native custody gates pass.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const run=path.join(root,'runs/model-portfolio-migration-v1');
const docs=path.join(root,'docs/plans/model-portfolio-migration-v1');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const job=path.join(run,'native-matched-generator-outcome-adoption-01');
const packets=path.join(run,'matched-generator-GPU-measurements-02');
const notes=path.join(run,'native-matched-generator-outcome-notes-01');
const destination=path.join(job,'browser-actions.json');
const expectationsPath=path.join(job,'browser-expectations.json');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
assert.ok(!fs.existsSync(destination)&&!fs.existsSync(expectationsPath),'Preserve original browser actions');
const importAudit=read(path.join(docs,'reports/native-matched-generator-GPU-import-custody-01.json'));
const ownerPath=path.resolve(root,importAudit.original_owner.path);
assert.ok(['native-matched-generator-GPU-import-01','native-matched-generator-GPU-import-reconciliation-01']
  .some(name=>ownerPath===path.join(run,name,'status.json')));
assert.equal(hash(fs.readFileSync(ownerPath)),importAudit.original_owner.sha256);
const imported=read(ownerPath);
const noteAudit=read(path.join(docs,'reports/native-matched-generator-outcome-adoption-01.json'));
assert.equal(imported.status,'four_readable_matched_GPU_imports_native_metadata_and_maxima_passed');
assert.equal(importAudit.status,'four_readable_matched_GPU_tables_native_custody_exact');
assert.equal(imported.last_acknowledged_checkpoint,importAudit.checkpoint);
assert.equal(noteAudit.status,'native38matched_outcome_notes_exact_and_scientific_records_unchanged');
assert.equal(noteAudit.changed_note_records,38);assert.equal(noteAudit.matched_fit_cards,35);
const header=read(path.join(project,'.tuldok/project.json'));
assert.equal(header.checkpoint,noteAudit.checkpoint);
const prep=read(path.join(packets,'preparation.json')),notePrep=read(path.join(notes,'preparation.json'));
const guide=read(path.join(project,'notes/00-models.portfolio.json'));
assert.equal(hash(fs.readFileSync(path.join(project,'notes/00-models.portfolio.json'))),
  hash(fs.readFileSync(path.join(notes,'notes/00-models.portfolio.json'))));
const actions=[],expectations=[];
const add=(kind,values={})=>actions.push({kind,...values});
add('assert_text',{text:noteAudit.checkpoint});
add('click',{selector:'#project-measurements-sources'});
add('wait_text',{selector:'#project-measurements-source-page',text:'1–29 of 29 sources'});
add('wait_enabled',{selector:'#project-measurements-source'});
let numericPresenceAssertions=0;
for(const part of prep.packets){
  const admitted=imported.imports.find(item=>item.K===part.K);
  assert.ok(admitted);assert.equal(admitted.id,part.sha256);
  const raw=fs.readFileSync(path.join(packets,part.path));assert.equal(hash(raw),part.sha256);
  const packet=JSON.parse(raw),k=part.K,count=packet.subjects.length;
  add('select_values',{selector:'#project-measurements-source',values:[admitted.origin_catalog_id+'.'+admitted.id]});
  add('click',{selector:'#project-measurements-load'});
  add('wait_text',{selector:'#project-measurements-subject-page',text:count+' selected across all pages'});
  add('wait_enabled',{selector:'#project-measurements-compare'});
  add('click',{selector:'#project-measurements-form > details > summary'});
  const queries=['pressure_relative_l2','velocity_relative_l2','tracer_relative_l2','equation_relative_l2',
    'block_median_ms','per_output_median_ms'].map(metric=>({name:`K${k}-all-${metric}`,metric,filters:{}}));
  for(const [density,size]of[[8,16],[16,32]])for(const[suffix,metric]of[['pressure','pressure_relative_l2'],['time','per_output_median_ms']])
    queries.push({name:`K${k}-cpm${density}-grid${size}-${suffix}`,metric,filters:{cells_per_metre:density,grid_x:size,grid_y:size,grid_z:size}});
  let previous={};
  for(const q of queries){
    for(const [index,dimension]of packet.dimensions.entries()){
      const key=dimension.id;
      if(Object.hasOwn(q.filters,key)){
        add('fill',{selector:`#project-measurements-filter-mode-${index}`,value:'value'});
        add('fill',{selector:`#project-measurements-filter-value-${index}`,value:String(q.filters[key])});
      }else if(Object.hasOwn(previous,key))add('fill',{selector:`#project-measurements-filter-mode-${index}`,value:'any'});
    }
    previous=q.filters;
    const selected=new Set(packet.cases.filter(c=>Object.entries(q.filters).every(([key,value])=>c.dimensions[key]===value)).map(c=>c.id));
    const rows=packet.observations.filter(o=>selected.has(o.case_id)&&o.metric_id===q.metric);
    assert.ok(selected.size);assert.equal(rows.length,count*selected.size);assert.ok(rows.every(o=>o.state==='measured'));
    const maxima=Object.fromEntries(packet.subjects.map(s=>[s.id,Math.max(...rows.filter(o=>o.subject_id===s.id).map(o=>o.value))]));
    expectations.push({name:q.name,K:k,packet_sha256:part.sha256,metric_id:q.metric,filters:q.filters,
      cases:selected.size,observations:rows.length,maxima,device:'CUDA',
      statistic:'Maximum of per-case statistic over every matching case; not median across cases.'});
    add('fill',{selector:'#project-measurements-metric',value:q.metric});
    add('click',{selector:'#project-measurements-compare'});
    add('wait_enabled',{selector:'#project-measurements-compare'});
    add('wait_text',{selector:'#project-measurements-counts',text:`${count} selected subjects of ${count}; ${rows.length} matching observations.`});
    add('wait_text',{selector:'#project-measurements-status',text:'Exact retained comparison.'});
    // Subject order is retained by native metadata and by comparison summaries.
    // These assert numeric text presence in the appropriate row, not a captured
    // HTTP-response-body comparison or independent physical reexecution.
    for(const [index,s]of packet.subjects.entries()){
      assert.ok(s.label.includes('GPU'));assert.ok(!s.id.startsWith('fit-'));
      add('wait_text',{selector:`#project-measurements-maxima tbody tr:nth-child(${index+1}) td:nth-child(2) button`,
        text:String(maxima[s.id])+' · inspect maximum for '+s.label});
      numericPresenceAssertions++;
    }
    add('scroll',{selector:'#project-measurements-chart'});
    add('screenshot',{name:q.name+'.png'});
  }
  const baseline=packet.subjects.find(s=>s.id===`F64-N1-K${k}`);assert.ok(baseline);
  add('click',{selector:'#project-measurements-maxima button',text:'inspect maximum for '+baseline.label});
  add('wait_text',{selector:'#project-measurements-inspection',text:baseline.label});
  add('click',{selector:'#project-measurements-inspection > details:last-child summary'});
  add('wait_text',{selector:'#project-measurements-inspection',text:admitted.id});
  add('scroll',{selector:'#project-measurements-inspection'});
  add('screenshot',{name:`K${k}-F64-source-inspection.png`});
}
assert.equal(expectations.length,40);assert.equal(numericPresenceAssertions,390);
const guideNote=notePrep.notes.find(n=>n.path==='notes/00-models.portfolio.json');
add('click',{selector:'#project-portfolio-notes'});
add('wait_text',{selector:'#project-portfolio-note-count',text:'1 organization notes found'});
add('wait_enabled',{selector:'#project-portfolio-note'});
add('fill',{selector:'#project-portfolio-note',value:guideNote.file_id});
add('click',{selector:'#project-portfolio-load'});
add('wait_text',{selector:'#project-portfolio-status',text:'Loaded retained authored model organization.'});
add('wait_text',{selector:'#project-portfolio-count',text:'205 models in this guide; 205 source reviewed.'});
add('wait_text',{selector:'#project-model-portfolio',text:'revision 7'});
add('scroll',{selector:'#project-portfolio-title'});
add('screenshot',{name:'matched-205-model-guide.png'});
const targets=new Set(notePrep.notes.filter(n=>n.fit_item_id).map(n=>n.fit_item_id));assert.equal(targets.size,35);
for(const model of guide.models.filter(m=>targets.has(m.id))){
  const query=modelQuery(guide,model);
  const matching=guide.models.filter(m=>[m.label,...m.aliases].some(a=>a.toLowerCase().includes(query.toLowerCase())));
  assert.equal(matching.length,1);assert.equal(matching[0].id,model.id);
  for(const[name,value]of[['group',''],['search',query],['review','source_reviewed']])
    add('fill',{selector:'#project-portfolio-'+name,value});
  add('click',{selector:'#project-portfolio-filter-form button'});
  add('wait_text',{selector:'#project-portfolio-count',text:'1 match the applied filters.'});
  add('click',{selector:'[data-lineage-node="model/'+model.id+'"]'});
  add('wait_text',{selector:'#project-portfolio-status',text:'Exact retained model summary and captured evidence.'});
  add('wait_text',{selector:'#project-portfolio-model-heading',text:model.label});
  for(const field of['method','purpose','experiment','outcome','lesson','reproduction'])
    add('wait_text',{selector:'#project-portfolio-detail',text:model.summary[field]});
  add('scroll',{selector:'#project-portfolio-detail h4:nth-of-type(4)'});
  add('screenshot',{name:model.id+'-matched-outcome.png'});
}
const representative=guide.models.find(m=>m.id==='fit-718037c1d106335e');
for(const[name,value]of[['group',''],['search',modelQuery(guide,representative)],['review','source_reviewed']])
  add('fill',{selector:'#project-portfolio-'+name,value});
add('click',{selector:'#project-portfolio-filter-form button'});
add('click',{selector:'[data-lineage-node="model/'+representative.id+'"]'});
add('wait_text',{selector:'#project-portfolio-detail',text:representative.summary.outcome});
add('click',{selector:'#project-portfolio-detail details > summary',text:'Exact captured card and original receipts'});
add('wait_text',{selector:'#project-portfolio-detail',text:'Matched GPU benchmark'});
add('wait_text',{selector:'#project-portfolio-detail',text:'| Cells/m | Grid cells | Worst pressure relative L2 |'});
add('wait_text',{selector:'#project-portfolio-detail',text:'Previous source-review statements'});
add('wait_text',{selector:'#project-portfolio-detail',text:'Exact original receipt bytes'});
add('scroll',{selector:'#project-portfolio-detail details[open] > summary'});
add('screenshot',{name:'matched-card-original-receipt-and-grid-table.png'});
// Inspect the updated overview through the real native working-file reader.
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
for(const text of['Historical timing correction','forward-validation hook inside the timing interval','7.645–31.085',
  'The table above reports medians of per-case medians.'])add('wait_value',{selector:'#project-working-text',text});
add('scroll',{selector:'#project-working-heading'});
add('screenshot',{name:'matched-overview-timing-scope-correction.png'});
add('assert_text',{text:'Experiment progression'});
fs.writeFileSync(destination,JSON.stringify(actions,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(expectationsPath,JSON.stringify({queries:expectations,numeric_presence_assertions:390,
  matched_model_summaries:35,summary_paragraph_assertions:210,generated_by:{path:path.relative(root,__filename),sha256:hash(fs.readFileSync(__filename))}},null,2)+'\n',{flag:'wx'});
console.log(actions.length+' native actions prepared:40 charts/390 numeric-text checks/35 outcomes/overview; no browser started.');
