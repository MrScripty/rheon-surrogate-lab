'use strict';
// Real UI checks of combined150pressure sources and23updated model summaries.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {modelQuery}=require('./matched_model_query.cjs');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1');
const docs=path.join(root,'docs/plans/model-portfolio-migration-v1');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const job=path.join(run,'native-pressure-outcome-adoption-01');
const packets=path.join(run,'all150-pressure-GPU-measurements-01');
const notes=path.join(run,'native-pressure-outcome-notes-01');
const destination=path.join(job,'browser-actions.json'),expectedPath=path.join(job,'browser-expectations.json');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
assert.ok(!fs.existsSync(destination)&&!fs.existsSync(expectedPath),'Preserve original browser preparation');
const importAudit=read(path.join(docs,'reports/native-all150-pressure-import-custody-01.json'));
const imported=read(path.join(root,importAudit.original_owner.path));
assert.equal(importAudit.original_owner.sha256,hash(fs.readFileSync(path.join(root,importAudit.original_owner.path))));
assert.equal(imported.status,'all150_pressure_imports_native_metadata_and_maxima_passed');
assert.equal(importAudit.status,'all150_pressure_tables_native_custody_exact');
assert.equal(imported.last_acknowledged_checkpoint,importAudit.checkpoint);
const noteAudit=read(path.join(docs,'reports/native-pressure-outcome-adoption-01.json'));
assert.equal(noteAudit.status,'native26pressure_outcome_notes_exact_and_scientific_records_unchanged');
const header=read(path.join(project,'.tuldok/project.json'));assert.equal(header.checkpoint,noteAudit.checkpoint);
const prep=read(path.join(packets,'preparation.json')),notePrep=read(path.join(notes,'preparation.json'));
const guide=read(path.join(project,'notes/00-models.portfolio.json'));
assert.equal(hash(fs.readFileSync(path.join(project,'notes/00-models.portfolio.json'))),hash(fs.readFileSync(path.join(notes,'notes/00-models.portfolio.json'))));
const tables=prep.packets.map(part=>{
  const file=path.join(packets,part.path),raw=fs.readFileSync(file);assert.equal(hash(raw),part.sha256);return JSON.parse(raw);
});
const subjects=tables.flatMap(table=>table.subjects),cases=tables[0].cases;
assert.equal(subjects.length,150);assert.equal(new Set(subjects.map(s=>s.id)).size,150);
assert.deepEqual(tables[0].cases,tables[1].cases);assert.deepEqual(tables[0].dimensions,tables[1].dimensions);
const values=new Map();
for(const table of tables)for(const row of table.observations){
  const key=row.subject_id+'|'+row.metric_id;if(!values.has(key))values.set(key,[]);
  values.get(key).push({case_id:row.case_id,value:row.value});assert.equal(row.state,'measured');
}
const actions=[],expectations=[];let numericChecks=0,previousFilters={};
const add=(kind,data={})=>actions.push({kind,...data});
add('assert_text',{text:noteAudit.checkpoint});
add('click',{selector:'#project-measurements-sources'});
add('wait_text',{selector:'#project-measurements-source-page',text:'1–31 of 31 sources'});
add('wait_enabled',{selector:'#project-measurements-source'});
add('select_values',{selector:'#project-measurements-source',values:imported.imports.map(item=>item.origin_catalog_id+'.'+item.id)});
add('click',{selector:'#project-measurements-load'});
add('wait_text',{selector:'#project-measurements-subject-page',text:'150 selected across all pages'});
add('wait_enabled',{selector:'#project-measurements-compare'});
add('click',{selector:'#project-measurements-form > details > summary'});
function compare(name,metric,filters,selectedSubjects){
  for(const[index,dimension]of tables[0].dimensions.entries()){
    const key=dimension.id;
    if(Object.hasOwn(filters,key)){
      add('fill',{selector:`#project-measurements-filter-mode-${index}`,value:'value'});
      add('fill',{selector:`#project-measurements-filter-value-${index}`,value:String(filters[key])});
    }else if(Object.hasOwn(previousFilters,key))add('fill',{selector:`#project-measurements-filter-mode-${index}`,value:'any'});
  }
  previousFilters=filters;
  const selectedCases=new Set(cases.filter(c=>Object.entries(filters).every(([key,value])=>c.dimensions[key]===value)).map(c=>c.id));
  assert.ok(selectedCases.size);
  const maxima={};
  for(const subject of selectedSubjects){
    const matching=values.get(subject.id+'|'+metric).filter(row=>selectedCases.has(row.case_id));
    assert.equal(matching.length,selectedCases.size);maxima[subject.id]=Math.max(...matching.map(row=>row.value));
  }
  const count=selectedSubjects.length,total=count*selectedCases.size;
  expectations.push({name,metric_id:metric,filters,subjects:selectedSubjects.map(s=>s.id),cases_per_subject:selectedCases.size,
    observations:total,maxima,device:'CUDA',statistic:'Maximum of the selected per-case statistic;not median across cases.'});
  add('fill',{selector:'#project-measurements-metric',value:metric});
  add('click',{selector:'#project-measurements-compare'});
  add('wait_enabled',{selector:'#project-measurements-compare'});
  add('wait_text',{selector:'#project-measurements-counts',text:`${count} selected subjects of 150; ${total} matching observations.`});
  add('wait_text',{selector:'#project-measurements-status',text:'Exact retained comparison.'});
  for(const[index,subject]of selectedSubjects.entries()){
    assert.ok(subject.label.includes('GPU'));assert.ok(!subject.id.startsWith('fit-'));
    add('wait_text',{selector:`#project-measurements-maxima tbody tr:nth-child(${index+1}) td:nth-child(2) button`,
      text:String(maxima[subject.id])+' · inspect maximum for '+subject.label});numericChecks++;
  }
  add('scroll',{selector:'#project-measurements-chart'});add('screenshot',{name:name+'.png'});
}
for(const[metric,suffix]of[['pressure_mean_free_relative_l2','pressure'],['equation_relative_l2','equation'],['pressure_call_ms','GPU-time']])
  compare('all150-'+suffix,metric,{},subjects);
for(const density of[8,16])for(const[metric,suffix]of[['pressure_mean_free_relative_l2','pressure'],['pressure_call_ms','GPU-time']])
  compare('all150-cpm'+density+'-'+suffix,metric,{cells_per_metre:density},subjects);
// Keep a readable7subject comparison that spans both partitions and4pages.
const wanted=new Set(['C53','C78','C106','C107','c53-u','teacher-coarse','coarse-control']);
const selected=subjects.filter(subject=>wanted.has(subject.id));assert.equal(selected.length,7);
add('click',{selector:'#project-measurements-subjects-clear'});
for(let offset=0;offset<150;offset+=40){
  if(offset){add('click',{selector:'#project-measurements-subjects-next'});add('wait_text',{selector:'#project-measurements-subject-page',text:`Subjects ${offset+1}–${Math.min(offset+40,150)} of 150.`});}
  const own=subjects.slice(offset,offset+40).filter(subject=>wanted.has(subject.id)).map(subject=>subject.id);
  if(own.length)add('select_values',{selector:'#project-measurements-subjects',values:own});
}
add('wait_text',{selector:'#project-measurements-subject-page',text:'7 selected across all pages'});
for(const[metric,suffix]of[['pressure_mean_free_relative_l2','pressure'],['equation_relative_l2','equation'],['pressure_call_ms','GPU-time']]){
  compare('selected7-'+suffix,metric,{},selected);
  compare('selected7-cpm8-grid16-'+suffix,metric,{cells_per_metre:8,grid_x:16,grid_y:16,grid_z:16},selected);
}
assert.equal(expectations.length,13);assert.equal(numericChecks,1092);
const representativeSubject=subjects.find(subject=>subject.id==='c53-u');assert.ok(representativeSubject);
add('click',{selector:'#project-measurements-maxima button',text:'inspect maximum for '+representativeSubject.label});
add('wait_text',{selector:'#project-measurements-inspection',text:representativeSubject.label});
add('click',{selector:'#project-measurements-inspection > details:last-child summary'});
add('wait_text',{selector:'#project-measurements-inspection',text:'/models/c53-u/rows/'});
add('scroll',{selector:'#project-measurements-inspection'});add('screenshot',{name:'pressure-original-source.png'});
const guideNote=notePrep.notes.find(note=>note.path==='notes/00-models.portfolio.json');
add('click',{selector:'#project-portfolio-notes'});
add('wait_text',{selector:'#project-portfolio-note-count',text:'1 organization notes found'});
add('wait_enabled',{selector:'#project-portfolio-note'});
add('fill',{selector:'#project-portfolio-note',value:guideNote.file_id});add('click',{selector:'#project-portfolio-load'});
add('wait_text',{selector:'#project-portfolio-status',text:'Loaded retained authored model organization.'});
add('wait_text',{selector:'#project-portfolio-count',text:'205 models in this guide; 205 source reviewed.'});
add('wait_text',{selector:'#project-model-portfolio',text:'revision '+notePrep.guide_revision});
add('scroll',{selector:'#project-portfolio-title'});add('screenshot',{name:'pressure-205-model-guide.png'});
const targets=new Set(notePrep.notes.filter(note=>note.fit_item_id).map(note=>note.fit_item_id));assert.equal(targets.size,23);
for(const model of guide.models.filter(item=>targets.has(item.id))){
  const query=modelQuery(guide,model),matching=guide.models.filter(item=>[item.label,...item.aliases].some(alias=>alias.toLowerCase().includes(query.toLowerCase())));
  assert.equal(matching.length,1);assert.equal(matching[0].id,model.id);
  for(const[name,value]of[['group',''],['search',query],['review','source_reviewed']])add('fill',{selector:'#project-portfolio-'+name,value});
  add('click',{selector:'#project-portfolio-filter-form button'});
  add('wait_text',{selector:'#project-portfolio-count',text:'1 match the applied filters.'});
  add('click',{selector:'[data-lineage-node="model/'+model.id+'"]'});
  add('wait_text',{selector:'#project-portfolio-status',text:'Exact retained model summary and captured evidence.'});
  add('wait_text',{selector:'#project-portfolio-model-heading',text:model.label});
  for(const field of['method','purpose','experiment','outcome','lesson','reproduction'])add('wait_text',{selector:'#project-portfolio-detail',text:model.summary[field]});
  add('scroll',{selector:'#project-portfolio-detail h4:nth-of-type(4)'});add('screenshot',{name:model.id+'-pressure-outcome.png'});
}
const representative=guide.models.find(model=>model.id===representativeSubject.dimensions.fit_item_id);
for(const[name,value]of[['group',''],['search',modelQuery(guide,representative)],['review','source_reviewed']])add('fill',{selector:'#project-portfolio-'+name,value});
add('click',{selector:'#project-portfolio-filter-form button'});add('click',{selector:'[data-lineage-node="model/'+representative.id+'"]'});
add('wait_text',{selector:'#project-portfolio-detail',text:representative.summary.outcome});
add('click',{selector:'#project-portfolio-detail details > summary',text:'Exact captured card and original receipts'});
for(const text of['Common GPU pressure benchmark','| Cells/m | Grid cells | Input dt (s) |','Previous source-review statements','Exact original receipt bytes'])
  add('wait_text',{selector:'#project-portfolio-detail',text});
add('scroll',{selector:'#project-portfolio-detail details[open] > summary'});add('screenshot',{name:'pressure-card-original-receipt.png'});
const owned=[];
for(const entry of header.inventory)if(entry.family==='project_file'){
  const doc=read(path.join(project,'.tuldok',entry.path));if(doc.project_id===header.project_id)owned.push(doc.path);
}
owned.sort();const overviewIndex=owned.indexOf('notes/portfolio-overview.md');assert.ok(overviewIndex>=0);
add('click',{selector:'#project-working-load'});
for(let start=0;start<=overviewIndex;start+=40){
  if(start){add('wait_enabled',{selector:'#project-working-more'});add('click',{selector:'#project-working-more'});}
  add('wait_text',{selector:'#project-working-list',text:owned[Math.min(start+39,owned.length-1)]});
}
add('click',{selector:'#project-working-list button',text:'notes/portfolio-overview.md'});
for(const text of['Pressure benchmarks now cover 150 fits','185 measured default','Twenty other components','Historical timing correction'])
  add('wait_value',{selector:'#project-working-text',text});
add('scroll',{selector:'#project-working-heading'});add('screenshot',{name:'pressure-updated-overview.png'});
add('assert_text',{text:'Experiment progression'});
fs.writeFileSync(destination,JSON.stringify(actions,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(expectedPath,JSON.stringify({queries:expectations,numeric_presence_assertions:numericChecks,updated_model_summaries:23,
  summary_paragraph_assertions:138,generated_by:{path:path.relative(root,__filename),sha256:hash(fs.readFileSync(__filename))}},null,2)+'\n',{flag:'wx'});
console.log(actions.length+' actions prepared:13charts/1092numeric-text checks/23summaries/source/card/overview;no browser started.');
