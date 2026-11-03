'use strict';
// Check the proposed native notes through the actual immutable UI codec.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const folder=path.join(root,'runs/model-portfolio-migration-v1/native-matched-generator-outcome-notes-01');
const source=path.join(root,'runs/model-portfolio-migration-v1/sources/Tuldok-composition-source-artifacts');
const output=path.join(root,'docs/plans/model-portfolio-migration-v1/reports/matched-generator-outcome-notes-qualification-01.json');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const binding=p=>({path:path.relative(root,p),sha256:hash(fs.readFileSync(p))});
const equal=(a,b)=>assert.deepEqual(JSON.parse(JSON.stringify(a)),JSON.parse(JSON.stringify(b)));
assert.ok(!fs.existsSync(output),'Preserve original note qualification');
const prepPath=path.join(folder,'preparation.json'),prep=read(prepPath);
for(const spec of [prep.producer,...prep.evidence])assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
const nativeFiles=fs.readFileSync(path.join(source,'project_files.py'),'utf8');
const maxItems=Number(nativeFiles.match(/^MAX_ADOPT_ITEMS=(\d+)$/m)[1]);
const maxBytes=Number(nativeFiles.match(/^MAX_ADOPT_BYTES=(\d+)\*1024\*1024$/m)[1])*1024*1024;
assert.equal(prep.notes.length,38);assert.ok(prep.notes.length<=maxItems);
assert.ok(prep.notes.reduce((n,note)=>n+note.bytes,0)<=maxBytes);
assert.equal(new Set(prep.notes.map(n=>n.path)).size,38);
const header=read(path.join(project,'.tuldok/project.json'));
const captured=new Map();
for(const entry of header.inventory)if(entry.family==='project_file'){
  const raw=fs.readFileSync(path.join(project,'.tuldok',entry.path));
  assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);
  const doc=JSON.parse(raw);if(doc.project_id===header.project_id)captured.set(doc.path,{entry,doc});
}
const notes=new Map(prep.notes.map(n=>[n.path,n]));
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
const window={};const codecPath=path.join(source,'static/project-model-portfolio.js');
vm.runInNewContext(fs.readFileSync(codecPath,'utf8'),{window,TextEncoder});
const codec=window.ProjectModelPortfolioDocument;
const guide=codec.decode(newText('notes/00-models.portfolio.json'));
assert.equal(guide.models.length,205);equal(guide.groups,oldGuide.groups);equal(guide.relationships,oldGuide.relationships);
assert.equal(guide.relationships.length,26);equal(index.aliases,oldIndex.aliases);equal(index.definitions,oldIndex.definitions);
const benchmark=read(path.join(root,'docs/plans/model-portfolio-migration-v1/reports/complete-generator-GPU-benchmark-01.json'));
const measured=new Map(benchmark.models.map(m=>[m.fit_item_id,m]));
const oldModels=new Map(oldGuide.models.map(m=>[m.id,m]));
const oldFits=new Map(oldIndex.fits.map(f=>[f.item_id,f]));
const fits=new Map(index.fits.map(f=>[f.item_id,f]));
const fields=['method','purpose','experiment','outcome','lesson','reproduction'];
const labels={method:'Method',purpose:'Purpose',experiment:'Experiment',outcome:'Observed outcome',lesson:'Lesson',reproduction:'Reproduction'};
const tailMarker='## Original retained fit facts before this review\n\n';
let changed=0,paragraphs=0;
const suffixes=[];
for(const model of guide.models){
  const before=oldModels.get(model.id),fit=fits.get(model.id),oldFit=oldFits.get(model.id);
  const note=notes.get(fit.card_path),current=captured.get(fit.card_path),text=note?newText(fit.card_path):current.doc.current.text;
  const version=note?{...current.doc.current,revision:note.revision,sha256:note.sha256,text}:current.doc.current;
  codec.checkCard(guide,model,{id:current.entry.id,project_id:header.project_id,kind:'note',path:fit.card_path,version});
  assert.equal(model.card.sha256,fit.card_sha256);assert.equal(model.review,'source_reviewed');
  if(measured.has(model.id)){
    changed++;assert.ok(note);const b=measured.get(model.id),s=b.summary,w=s.worst_metrics;
    for(const name of ['method','purpose','experiment'])assert.equal(model.summary[name],before.summary[name]);
    // Compare the recorded rounded numeric tokens without imposing JavaScript's
    // different fixed/scientific formatting threshold on the Python prose.
    const trio=model.summary.outcome.match(/relative L2 are (\S+), (\S+) and (\S+)\. Median time is /);
    assert.ok(trio,'Outcome must name ordered pressure/velocity/tracer values');
    for(const [index,value]of[w.pressure_relative_l2,w.velocity_relative_l2,w.tracer_relative_l2].entries())
      assert.ok(Math.abs(Number(trio[index+1])-value)<=Math.max(Math.abs(value)*5e-6,1e-300));
    assert.ok(model.summary.outcome.includes(s.median_case_median_per_output_ms.toFixed(5)+' ms per physical output'));
    assert.ok(model.summary.outcome.includes(s.median_case_median_block_ms.toFixed(5)+' ms per K'+b.K+' block'));
    assert.ok(model.summary.outcome.includes('not an independent holdout or long rollout'));
    const oldText=current.doc.current.text,suffix=oldText.slice(oldText.indexOf(tailMarker)+tailMarker.length);
    assert.equal(oldText.split(tailMarker).length,2);assert.ok(text.endsWith(suffix));assert.equal(hash(Buffer.from(suffix)),note.original_tail_sha256);
    const receipt=fs.readFileSync(path.join(root,note.original_receipt.path),'utf8');assert.equal(hash(Buffer.from(receipt)),note.original_receipt.sha256);
    assert.equal(oldText.split(receipt).length,2);assert.equal(text.split(receipt).length,2);
    for(const name of ['outcome','lesson','reproduction'])assert.ok(text.includes(JSON.stringify(before.summary[name])),'Previous review statement must remain exact');
    const rows=text.split('\n').filter(line=>/^\| (8|16) \|/.test(line));assert.equal(rows.length,8);
    const groups=Object.entries(b.by_density_and_grid).map(([key,value])=>[key,value]);
    for(const [key,group]of groups){
      const [density,grid]=key.split('cpm:');assert.ok(rows.some(row=>row.startsWith('| '+density+' | '+grid+' | ')));
    }
    const expected={...before,summary:model.summary,aliases:model.aliases,card:{...before.card,revision:note.revision,sha256:note.sha256}};equal(model,expected);
    assert.equal(model.aliases.length,before.aliases.length+1);assert.ok(model.aliases.at(-1).startsWith('GPU benchmark: '));
    equal(fit,{...oldFit,card_sha256:note.sha256,matched_GPU_benchmark:fit.matched_GPU_benchmark});
    assert.equal(fit.matched_GPU_benchmark.cases,64);assert.equal(fit.matched_GPU_benchmark.K,b.K);
    suffixes.push({fit_item_id:model.id,original_tail_sha256:note.original_tail_sha256});
  }else{equal(model,before);equal(fit,oldFit);}
  for(let i=0;i<fields.length;i++){
    const prefix='**'+labels[fields[i]]+'.** ',start=text.indexOf(prefix)+prefix.length;
    const endMarker=i+1<fields.length?'\n\n**'+labels[fields[i+1]]+'.** ':'\n\n## Exact source bindings for this review\n';
    const end=text.indexOf(endMarker,start);assert.ok(end>start);assert.equal(model.summary[fields[i]],text.slice(start,end).trim());paragraphs++;
  }
}
assert.equal(changed,35);assert.equal(paragraphs,1230);
for(const key of Object.keys(oldIndex))if(!['fits','sources'].includes(key))equal(index[key],oldIndex[key]);
for(const key of Object.keys(oldIndex.sources))equal(index.sources[key],oldIndex.sources[key]);
assert.equal(notes.get('notes/00-models.portfolio.json').revision,prep.guide_revision);
const overview=newText('notes/portfolio-overview.md');
assert.ok(overview.includes('forward-validation hook inside the timing interval'));assert.ok(overview.includes('Original Trial ancestry')||overview.includes('unknown ancestry'));
const historical=read(path.join(root,'runs/model-portfolio-migration-v1/common-generator-GPU-measurements-01/preparation.json'));
for(const p of historical.packets)assert.ok(overview.includes(p.sha256));
const result={status:'matched_outcome38note_proposal_actual_UI_codec_and_original_tails_qualified',created_unix:Date.now()/1000,
  notes:38,matched_fits:35,models:205,literal_summary_paragraphs:1230,authored_comparisons:26,
  native_batch_items_bound_passed:true,native_max_items:maxItems,native_max_bytes:maxBytes,
  original_tails_preserved:suffixes,unchanged_other_model_summaries:170,Trial_lineage_modified:false,
  native_publication_pending:true,actual_UI_pending:true,full_portfolio_complete:false,
  producer:binding(__filename),preparation:binding(prepPath),native_UI_codec:binding(codecPath),
  evidence:prep.notes.map(n=>binding(path.join(root,n.prepared_path)))};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
process.stdout.write('38 proposed notes qualified through native UI codec; 35 exact original tails; 205 summaries; 26 relationships unchanged.\n');
