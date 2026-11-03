'use strict';
// Verify all205cards through the real immutable portfolio codec before staging.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const folder=path.join(run,'native-pressure-outcome-notes-01');
const source=path.join(run,'sources/Tuldok-composition-source-artifacts');
const output=path.join(root,'docs/plans/model-portfolio-migration-v1/reports/pressure-outcome-notes-qualification-01.json');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const binding=p=>({path:path.relative(root,p),sha256:hash(fs.readFileSync(p))});
const equal=(a,b)=>assert.deepEqual(JSON.parse(JSON.stringify(a)),JSON.parse(JSON.stringify(b)));
assert.ok(!fs.existsSync(output),'Preserve original26note qualification');
const prepPath=path.join(folder,'preparation.json'),prep=read(prepPath);
for(const spec of [prep.producer,...prep.evidence])assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
const nativeFiles=fs.readFileSync(path.join(source,'project_files.py'),'utf8');
const maxItems=Number(nativeFiles.match(/^MAX_ADOPT_ITEMS=(\d+)$/m)[1]);
const maxBytes=Number(nativeFiles.match(/^MAX_ADOPT_BYTES=(\d+)\*1024\*1024$/m)[1])*1024*1024;
assert.equal(prep.notes.length,26);assert.ok(prep.notes.length<=maxItems);
assert.ok(prep.notes.reduce((n,note)=>n+note.bytes,0)<=maxBytes);
const header=read(path.join(project,'.tuldok/project.json')),captured=new Map();
for(const entry of header.inventory)if(entry.family==='project_file'){
  const raw=fs.readFileSync(path.join(project,'.tuldok',entry.path));
  assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);
  const doc=JSON.parse(raw);
  if(doc.project_id===header.project_id){assert.ok(!captured.has(doc.path));captured.set(doc.path,{entry,doc});}
}
const notes=new Map(prep.notes.map(n=>[n.path,n]));assert.equal(notes.size,26);
for(const note of prep.notes){
  const original=captured.get(note.path),raw=fs.readFileSync(path.join(root,note.prepared_path));
  assert.ok(original);assert.equal(hash(raw),note.sha256);assert.equal(raw.length,note.bytes);assert.ok(raw.length<=1048576);
  assert.equal(original.entry.id,note.file_id);assert.equal(original.doc.current.sha256,note.previous_sha256);
  assert.equal(hash(fs.readFileSync(path.join(project,note.path))),note.previous_sha256);
  assert.equal(note.expected_revision,original.doc.current.revision);assert.equal(note.revision,note.expected_revision+1);
}
const newText=p=>fs.readFileSync(path.join(root,notes.get(p).prepared_path),'utf8');
const oldGuide=read(path.join(project,'notes/00-models.portfolio.json'));
const oldIndex=read(path.join(project,'notes/portfolio-index.json'));
const index=JSON.parse(newText('notes/portfolio-index.json'));
const window={},codecPath=path.join(source,'static/project-model-portfolio.js');
vm.runInNewContext(fs.readFileSync(codecPath,'utf8'),{window,TextEncoder});
const codec=window.ProjectModelPortfolioDocument,guide=codec.decode(newText('notes/00-models.portfolio.json'));
assert.equal(guide.models.length,205);equal(guide.groups,oldGuide.groups);equal(guide.relationships,oldGuide.relationships);
assert.equal(guide.relationships.length,26);equal(index.aliases,oldIndex.aliases);equal(index.definitions,oldIndex.definitions);
const benchmark=read(path.join(root,'docs/plans/model-portfolio-migration-v1/reports/original-pressure-GPU-timing-batch-01.json'));
const measured=new Map(benchmark.rows.map(row=>[row.fit_item_id,row]));assert.equal(measured.size,23);
const chartPrep=read(path.join(run,'all150-pressure-GPU-measurements-01/preparation.json'));
const oldModels=new Map(oldGuide.models.map(m=>[m.id,m])),oldFits=new Map(oldIndex.fits.map(f=>[f.item_id,f]));
const fits=new Map(index.fits.map(f=>[f.item_id,f]));
const fields=['method','purpose','experiment','outcome','lesson','reproduction'];
const labels={method:'Method',purpose:'Purpose',experiment:'Experiment',outcome:'Observed outcome',lesson:'Lesson',reproduction:'Reproduction'};
const marker='## Original retained fit facts before this review\n\n';
let changed=0,paragraphs=0;const tails=[];
for(const model of guide.models){
  const before=oldModels.get(model.id),fit=fits.get(model.id),oldFit=oldFits.get(model.id);
  const note=notes.get(fit.card_path),current=captured.get(fit.card_path),text=note?newText(fit.card_path):current.doc.current.text;
  const version=note?{...current.doc.current,revision:note.revision,sha256:note.sha256,text}:current.doc.current;
  codec.checkCard(guide,model,{id:current.entry.id,project_id:header.project_id,kind:'note',path:fit.card_path,version});
  assert.equal(model.card.sha256,fit.card_sha256);assert.equal(model.review,'source_reviewed');
  if(measured.has(model.id)){
    changed++;assert.ok(note);const b=measured.get(model.id);
    for(const name of ['method','purpose','experiment'])assert.equal(model.summary[name],before.summary[name]);
    const numeric=model.summary.outcome.match(/worst mean-free pressure relative L2 is (\S+) and worst equation residual relative L2 is (\S+)\. Median case GPU latency is (\S+) ms per pressure request; the largest case median is (\S+) ms\/request\./);
    assert.ok(numeric,'Pressure outcome must state ordered field/equation/time values');
    for(const [i,value]of[b.worst.pressure_mean_free_relative_l2,b.worst.common_equation_relative_l2].entries())
      assert.ok(Math.abs(Number(numeric[i+1])-value)<=Math.max(Math.abs(value)*5e-6,1e-300));
    assert.equal(numeric[3],b.case_median_median_ms.toFixed(5));assert.equal(numeric[4],b.worst_case_median_ms.toFixed(5));
    assert.ok(model.summary.outcome.includes('not complete simulation steps or independent holdout'));
    assert.ok(model.summary.lesson.startsWith(before.summary.lesson));assert.ok(model.summary.reproduction.startsWith(before.summary.reproduction));
    const oldText=current.doc.current.text;assert.equal(oldText.split(marker).length,2);
    const suffix=oldText.slice(oldText.indexOf(marker)+marker.length);
    assert.ok(text.endsWith(suffix));assert.equal(hash(Buffer.from(suffix)),note.original_tail_sha256);
    const receipt=fs.readFileSync(path.join(root,note.original_receipt.path),'utf8');assert.equal(hash(Buffer.from(receipt)),note.original_receipt.sha256);
    assert.equal(oldText.split(receipt).length,2);assert.equal(text.split(receipt).length,2);
    for(const name of ['outcome','lesson','reproduction'])assert.ok(text.includes(JSON.stringify(before.summary[name])));
    const rows=text.split('\n').filter(row=>/^\| (8|16) \|/.test(row));assert.equal(rows.length,b.density_grid_timestep_strata.length);
    for(const group of b.density_grid_timestep_strata)assert.ok(rows.some(row=>row.startsWith('| '+group.cells_per_metre+' | '+group.counts_xyz.join('x')+' | ')));
    equal(model,{...before,summary:model.summary,aliases:model.aliases,card:{...before.card,revision:note.revision,sha256:note.sha256}});
    assert.equal(model.aliases.length,before.aliases.length+1);assert.equal(model.aliases.at(-1),'GPU pressure: '+chartPrep.fit_to_subject[model.id]);
    equal(fit,{...oldFit,card_sha256:note.sha256,common_GPU_pressure_benchmark:fit.common_GPU_pressure_benchmark});
    assert.equal(fit.common_GPU_pressure_benchmark.cases,128);assert.equal(fit.common_GPU_pressure_benchmark.inference_device,'cuda');
    assert.equal(fit.common_GPU_pressure_benchmark.timing_units,'ms/pressure_request');
    assert.equal(fit.common_GPU_pressure_benchmark.subject_id,chartPrep.fit_to_subject[model.id]);
    tails.push({fit_item_id:model.id,original_tail_sha256:note.original_tail_sha256});
  }else{equal(model,before);equal(fit,oldFit);}
  for(let i=0;i<fields.length;i++){
    const prefix='**'+labels[fields[i]]+'.** ',start=text.indexOf(prefix)+prefix.length;
    const endMarker=i+1<fields.length?'\n\n**'+labels[fields[i+1]]+'.** ':'\n\n## Exact source bindings for this review\n';
    const end=text.indexOf(endMarker,start);assert.ok(end>start);assert.equal(model.summary[fields[i]],text.slice(start,end).trim());paragraphs++;
  }
}
assert.equal(changed,23);assert.equal(paragraphs,1230);
for(const key of Object.keys(oldIndex))if(!['fits','sources'].includes(key))equal(index[key],oldIndex[key]);
for(const key of Object.keys(oldIndex.sources))equal(index.sources[key],oldIndex.sources[key]);
assert.equal(notes.get('notes/00-models.portfolio.json').revision,prep.guide_revision);
const overview=newText('notes/portfolio-overview.md'),oldOverview=fs.readFileSync(path.join(project,'notes/portfolio-overview.md'),'utf8');
assert.ok(overview.endsWith(oldOverview.slice('# Rheon model portfolio\n\n'.length)));
for(const text of ['150 pressure fits','185 measured default','Twenty other components','not advance a full simulation step','Historical timing correction'])assert.ok(overview.includes(text));
const result={status:'pressure26outcome_note_proposal_actual_UI_codec_and_original_tails_qualified',created_unix:Date.now()/1000,
  notes:26,pressure_fits:23,models:205,literal_summary_paragraphs:1230,authored_comparisons:26,native_batch_items_bound_passed:true,
  native_max_items:maxItems,native_max_bytes:maxBytes,original_tails_preserved:tails,unchanged_other_model_summaries:182,
  Trial_lineage_modified:false,native_publication_pending:true,actual_UI_pending:true,full_portfolio_complete:false,
  producer:binding(__filename),preparation:binding(prepPath),native_UI_codec:binding(codecPath),evidence:prep.notes.map(n=>binding(path.join(root,n.prepared_path)))};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log('26pressure outcome notes qualified through native UI codec;23exact original tails;205summaries;26relationships unchanged.');
