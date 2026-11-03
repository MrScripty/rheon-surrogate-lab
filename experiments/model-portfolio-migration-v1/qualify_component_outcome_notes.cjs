'use strict';
// Exercise the actual native document decoder and exact retained-note boundaries.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04'),folder=path.join(run,'native-component-outcome-notes-01');
const source=path.join(run,'sources/Tuldok-composition-source-artifacts');
const output=path.join(root,'docs/plans/model-portfolio-migration-v1/reports/component-outcome-notes-qualification-01.json');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const binding=p=>({path:path.relative(root,p),sha256:hash(fs.readFileSync(p))});
const equal=(a,b)=>assert.deepEqual(JSON.parse(JSON.stringify(a)),JSON.parse(JSON.stringify(b)));
assert.ok(!fs.existsSync(output),'Preserve original qualification');
const prepPath=path.join(folder,'preparation.json'),prep=read(prepPath);
for(const spec of [prep.producer,...prep.evidence])assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
const nativeFiles=fs.readFileSync(path.join(source,'project_files.py'),'utf8');
const maxItems=Number(nativeFiles.match(/^MAX_ADOPT_ITEMS=(\d+)$/m)[1]);
const maxBytes=Number(nativeFiles.match(/^MAX_ADOPT_BYTES=(\d+)\*1024\*1024$/m)[1])*1024*1024;
assert.equal(prep.notes.length,23);assert.equal(prep.component_fits,20);assert.ok(prep.notes.length<=maxItems);
assert.ok(prep.notes.reduce((n,note)=>n+note.bytes,0)<=maxBytes);
const header=read(path.join(project,'.tuldok/project.json')),captured=new Map();
assert.equal(header.checkpoint,prep.source_checkpoint_at_read);assert.equal(header.project_id,prep.source_project_id);
for(const entry of header.inventory)if(entry.family==='project_file'&&entry.namespace===header.project_id){
  const raw=fs.readFileSync(path.join(project,'.tuldok',entry.path));
  assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);
  const doc=JSON.parse(raw);
  if(doc.project_id===header.project_id){assert.ok(!captured.has(doc.path));captured.set(doc.path,{entry,doc});}
}
const notes=new Map(prep.notes.map(n=>[n.path,n]));assert.equal(notes.size,23);
for(const note of prep.notes){
  const original=captured.get(note.path),raw=fs.readFileSync(path.join(root,note.prepared_path));
  assert.ok(original);assert.equal(hash(raw),note.sha256);assert.equal(raw.length,note.bytes);assert.ok(raw.length<=1048576);
  assert.equal(original.entry.id,note.file_id);assert.equal(original.doc.current.sha256,note.previous_sha256);
  assert.equal(hash(fs.readFileSync(path.join(project,note.path))),note.previous_sha256);
  assert.equal(note.expected_revision,original.doc.current.revision);assert.equal(note.revision,note.expected_revision+1);
}
const newText=p=>fs.readFileSync(path.join(root,notes.get(p).prepared_path),'utf8');
const oldGuide=read(path.join(project,'notes/00-models.portfolio.json')),oldIndex=read(path.join(project,'notes/portfolio-index.json'));
const index=JSON.parse(newText('notes/portfolio-index.json')),window={},codecPath=path.join(source,'static/project-model-portfolio.js');
vm.runInNewContext(fs.readFileSync(codecPath,'utf8'),{window,TextEncoder});
const guide=window.ProjectModelPortfolioDocument.decode(newText('notes/00-models.portfolio.json'));
const coverage=read(path.join(root,'docs/plans/model-portfolio-migration-v1/reports/common-fit-benchmark-coverage-16.json'));
const selected=new Map(coverage.rows.filter(r=>r.accuracy_benchmark&&r.task!=='pressure_proposal').map(r=>[r.fit_item_id,r]));
const leadSource=read(path.join(root,'docs/plans/model-portfolio-migration-v1/reports/multistep-lead-statistics-01.json'));
assert.equal(selected.size,20);assert.equal(guide.models.length,205);assert.equal(index.fits.length,205);
for(const key of Object.keys(oldGuide))if(!['description','models'].includes(key))equal(guide[key],oldGuide[key]);
const oldModels=new Map(oldGuide.models.map(m=>[m.id,m])),oldFits=new Map(oldIndex.fits.map(f=>[f.item_id,f]));
const fits=new Map(index.fits.map(f=>[f.item_id,f]));
const labels={method:'Method',purpose:'Purpose',experiment:'Experiment',outcome:'Observed outcome',lesson:'Lesson',reproduction:'Reproduction'};
const fields=Object.keys(labels),tailMarker='## Original retained fit facts before this review\n\n';
let changed=0,paragraphs=0,physicalLeadMetricRows=0,qualifiedTimingFits=0;
for(const model of guide.models){
  const before=oldModels.get(model.id),fit=fits.get(model.id),oldFit=oldFits.get(model.id);
  const text=selected.has(model.id)?newText(fit.card_path):fs.readFileSync(path.join(project,fit.card_path),'utf8');
  if(selected.has(model.id)){
    changed++;
    const row=selected.get(model.id),note=notes.get(fit.card_path),oldText=fs.readFileSync(path.join(project,fit.card_path),'utf8');
    assert.equal(hash(Buffer.from(text)),note.sha256);assert.equal(hash(Buffer.from(text.split(tailMarker)[1])),note.original_tail_sha256);
    assert.equal(text.split(tailMarker)[1],oldText.split(tailMarker)[1]);
    const receipt=fs.readFileSync(path.join(root,note.original_receipt.path));assert.equal(hash(receipt),note.original_receipt.sha256);
    assert.equal(text.split(receipt.toString()).length,2);assert.equal(oldText.split(receipt.toString()).length,2);
    equal(model,{...before,summary:model.summary,card:{...before.card,revision:note.revision,sha256:note.sha256}});
    for(const key of ['method','purpose','experiment'])assert.equal(model.summary[key],before.summary[key]);
    assert.ok(model.summary.lesson.startsWith(before.summary.lesson+' '));
    assert.ok(model.summary.reproduction.startsWith(before.summary.reproduction+' '));
    equal(fit,{...oldFit,card_sha256:note.sha256,scoped_GPU_component_evaluation:fit.scoped_GPU_component_evaluation,
      retained_scalar_statistics:row.retained_field_statistics});
    equal(fit.scoped_GPU_component_evaluation.accuracy,row.accuracy_benchmark);
    equal(fit.scoped_GPU_component_evaluation.qualified_timing,row.default_benchmark);
    assert.equal(fit.scoped_GPU_component_evaluation.native_chart_publication_pending,true);
    if(row.default_benchmark){qualifiedTimingFits++;assert.ok(model.summary.outcome.includes(row.default_benchmark.timing_units));}
    else assert.ok(model.summary.outcome.includes('Qualified GPU latency remains unavailable'));
    assert.ok(text.includes('native portable scalar/source closure remains pending'));
    if(row.task.startsWith('guided_')){
      const expected=leadSource.rows.filter(r=>r.subject_id===model.id);
      assert.equal(expected.length,270);
      equal([...new Set(expected.map(r=>r.metric))].sort(),[
        'equation_absolute_rms','equation_relative_l2','pressure_absolute_rms','pressure_cell_zero_absolute_error',
        'pressure_relative_l2','tracer_absolute_rms','tracer_relative_l2','velocity_absolute_rms','velocity_relative_l2']);
      assert.equal(new Set(expected.map(r=>JSON.stringify([r.K,r.output_dt_s,r.lead]))).size,30);
      const tableRows=text.split('\n').filter(line=>/^\| (30|60|120) \|/.test(line));
      assert.equal(tableRows.length,expected.length);
      for(const r of expected){
        const prefix='| '+(1/r.output_dt_s)+' | '+r.K+' | '+r.lead+' | '+r.output_role+' | ';
        const candidates=tableRows.filter(line=>line.startsWith(prefix)&&line.includes(' | '+r.metric+' | '));
        assert.equal(candidates.length,1);
        assert.ok(candidates[0].includes(' | '+r.statistics.samples+' | '));
        physicalLeadMetricRows++;
      }
    }else assert.ok(!text.includes('## Accuracy at every physical output'));
  }else{equal(model,before);equal(fit,oldFit);}
  for(let i=0;i<fields.length;i++){
    const prefix='**'+labels[fields[i]]+'.** ',start=text.indexOf(prefix)+prefix.length;
    const endMarker=i+1<fields.length?'\n\n**'+labels[fields[i+1]]+'.** ':'\n\n## Exact source bindings for this review\n';
    const end=text.indexOf(endMarker,start);assert.ok(end>start);assert.equal(model.summary[fields[i]],text.slice(start,end).trim());paragraphs++;
  }
}
assert.equal(changed,20);assert.equal(qualifiedTimingFits,10);assert.equal(paragraphs,1230);assert.equal(physicalLeadMetricRows,540);
for(const key of Object.keys(oldIndex))if(!['fits','sources'].includes(key))equal(index[key],oldIndex[key]);
for(const key of Object.keys(oldIndex.sources))equal(index.sources[key],oldIndex.sources[key]);
assert.equal(notes.get('notes/00-models.portfolio.json').revision,prep.guide_revision);
const overview=newText('notes/portfolio-overview.md'),oldOverview=fs.readFileSync(path.join(project,'notes/portfolio-overview.md'),'utf8');
assert.ok(overview.endsWith(oldOverview.slice('# Rheon model portfolio\n\n'.length)));
for(const text of ['205 accuracy outcomes','195 qualified timings','portable scalar/source closure remain pending','Prediction deletion remains paused'])assert.ok(overview.includes(text));
const result={status:'component23note_proposal_native_codec_original_tails_and_physical_leads_qualified',created_unix:Date.now()/1000,
  notes:23,component_fits:20,models:205,literal_summary_paragraphs:1230,authored_comparisons:26,native_batch_items_bound_passed:true,
  native_max_items:maxItems,native_max_bytes:maxBytes,original_tails_preserved:20,unchanged_other_models:185,
  physical_lead_metric_rows:540,qualified_component_timing_fits:10,unavailable_component_timing_fits:10,
  initial_oracle_correction:'The initial check assumed12metrics perlead. The accepted guided metric contract has9explicitly checked metrics across30cadence/K/lead combinations perfit;no metric orlead was removed.',
  Trial_lineage_modified:false,native_publication_pending:true,actual_UI_pending:true,portable_scalar_source_closure_complete:false,
  producer:binding(__filename),preparation:binding(prepPath),native_UI_codec:binding(codecPath),
  evidence:prep.notes.map(n=>binding(path.join(root,n.prepared_path))),full_portfolio_complete:false};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log('23notes qualified:20component outcomes,205summaries,20exact original tails,540physical-lead metric rows.');
