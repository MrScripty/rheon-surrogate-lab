'use strict';
// Independent note/revision/evidence validation through the actual UI codec.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const packet=path.join(root,'runs/model-portfolio-migration-v1/native-complete-source-review-publication-01');
const output=path.join(root,'docs/plans/model-portfolio-migration-v1/reports/complete-source-review-native-preparation-01.json');
const source=path.join(root,'runs/model-portfolio-migration-v1/sources/Tuldok-composition-source-artifacts');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8')),hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const binding=p=>({path:path.relative(root,p),sha256:hash(fs.readFileSync(p))});
const equal=(a,b)=>assert.deepEqual(JSON.parse(JSON.stringify(a)),JSON.parse(JSON.stringify(b)));
assert.ok(!fs.existsSync(output),'Preserve original qualification');
const prepPath=path.join(packet,'preparation.json'),prep=read(prepPath);
const headerPath=path.join(project,'.tuldok/project.json'),headerBytes=fs.readFileSync(headerPath),header=JSON.parse(headerBytes);
assert.equal(hash(headerBytes),prep.source_manifest_sha256);assert.equal(header.checkpoint,prep.source_checkpoint);
assert.ok(prep.notes.reduce((n,r)=>n+r.bytes,0)<=8*1048576);assert.equal(prep.notes.length,63);assert.equal(new Set(prep.notes.map(n=>n.path)).size,63);
for(const spec of prep.evidence)assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
const reviewPath=path.join(root,prep.review_path),review=read(reviewPath);
assert.equal(hash(fs.readFileSync(reviewPath)),prep.review_sha256);assert.equal(review.rows.length,60);
for(const spec of [review.producer,...review.helpers])assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
const captured=new Map();
for(const entry of header.inventory)if(entry.family==='project_file'){
  const raw=fs.readFileSync(path.join(project,'.tuldok',entry.path));assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);
  const value=JSON.parse(raw);if(value.project_id===header.project_id){assert.ok(!captured.has(value.path));captured.set(value.path,{entry,value});}
}
const notes=new Map(prep.notes.map(n=>[n.path,n]));
for(const note of prep.notes){
  const raw=fs.readFileSync(path.join(root,note.prepared_path)),old=captured.get(note.path).value.current;
  assert.equal(hash(raw),note.sha256);assert.equal(raw.length,note.bytes);assert.ok(raw.length<=1048576);
  assert.equal(hash(fs.readFileSync(path.join(project,note.path))),old.sha256);
  assert.equal(old.sha256,note.previous_sha256);assert.equal(old.revision,note.expected_revision);assert.equal(note.revision,old.revision+1);
}
const rows=new Map(review.rows.map(r=>[r.fit_item_id,r]));assert.equal(rows.size,60);
const labels={method:'Method',purpose:'Purpose',experiment:'Experiment',outcome:'Observed outcome',lesson:'Lesson',reproduction:'Reproduction'};
const fields=Object.keys(labels),suffixMarker='## Retained facts and comparison evidence\n\n';
const suffixes=[];
for(const row of rows.values()){
  const note=notes.get(row.note_path),old=captured.get(row.note_path).value.current.text;
  const proposed=fs.readFileSync(path.join(root,note.prepared_path),'utf8');
  assert.equal(note.fit_item_id,row.fit_item_id);
  const receipt=fs.readFileSync(path.join(root,row.original_receipt.path),'utf8');assert.equal(hash(Buffer.from(receipt)),row.original_receipt.sha256);
  assert.equal(old.split(receipt).length,2);assert.equal(proposed.split(receipt).length,2);
  assert.equal(old.split(suffixMarker).length,2);const suffix=old.slice(old.indexOf(suffixMarker)+suffixMarker.length);
  assert.ok(proposed.endsWith(suffix));assert.equal(hash(Buffer.from(suffix)),note.original_suffix_sha256);
  for(let i=0;i<fields.length;i++){
    const prefix='**'+labels[fields[i]]+'.** ',start=proposed.indexOf(prefix);assert.ok(start>=0);
    const endMarker=i+1<fields.length?'\n\n**'+labels[fields[i+1]]+'.** ':'\n\n## Exact source bindings for this review\n';
    const end=proposed.indexOf(endMarker,start+prefix.length);assert.ok(end>start);
    assert.equal(proposed.slice(start+prefix.length,end).trim(),row[fields[i]].trim());
  }
  suffixes.push({fit_item_id:row.fit_item_id,suffix_sha256:hash(Buffer.from(suffix))});
}
const guidePath='notes/00-models.portfolio.json',indexPath='notes/portfolio-index.json';
const index=read(path.join(root,notes.get(indexPath).prepared_path)),oldIndex=JSON.parse(captured.get(indexPath).value.current.text);
assert.equal(index.fits.length,205);equal(index.aliases,oldIndex.aliases);equal(index.definitions,oldIndex.definitions);
const fits=new Map(index.fits.map(f=>[f.item_id,f]));
for(const old of oldIndex.fits){
  const expected={...old};if(rows.has(old.item_id))Object.assign(expected,{card_sha256:notes.get(old.card_path).sha256,method_review_status:'captured_source_and_receipt_reviewed'});
  equal(fits.get(old.item_id),expected);
}
for(const key of Object.keys(oldIndex))if(!['fits','sources'].includes(key))equal(index[key],oldIndex[key]);
for(const key of Object.keys(oldIndex.sources))if(!['method_reviews','method_review_batches'].includes(key))equal(index.sources[key],oldIndex.sources[key]);
equal(index.sources.method_reviews,binding(reviewPath));
for(const old of oldIndex.sources.method_review_batches)assert.ok(index.sources.method_review_batches.some(x=>JSON.stringify(x)===JSON.stringify(old)));
const window={};const codecPath=path.join(source,'static/project-model-portfolio.js');
vm.runInNewContext(fs.readFileSync(codecPath,'utf8'),{window,TextEncoder});const codec=window.ProjectModelPortfolioDocument;
const guide=codec.decode(fs.readFileSync(path.join(root,notes.get(guidePath).prepared_path),'utf8'));
const oldGuide=codec.decode(captured.get(guidePath).value.current.text);
assert.equal(guide.models.length,205);equal(guide.groups,oldGuide.groups);equal(guide.relationships,oldGuide.relationships);assert.equal(guide.relationships.length,26);
equal([...guide.models.map(m=>m.id)].sort(),[...fits.keys()].sort());
const oldModels=new Map(oldGuide.models.map(m=>[m.id,m]));let reviewed=0,paragraphs=0,changed=0,full=0;
for(const model of guide.models){
  const fit=fits.get(model.id),retained=captured.get(fit.card_path),note=notes.get(fit.card_path);
  const text=note?fs.readFileSync(path.join(root,note.prepared_path),'utf8'):retained.value.current.text;
  const version=note?{...retained.value.current,text,revision:note.revision,sha256:note.sha256}:retained.value.current;
  assert.equal(hash(Buffer.from(text)),version.sha256);assert.equal(model.card.sha256,fit.card_sha256);assert.equal(model.card.revision,version.revision);
  codec.checkCard(guide,model,{id:retained.entry.id,project_id:header.project_id,kind:'note',path:fit.card_path,version});
  const expected=JSON.parse(JSON.stringify(oldModels.get(model.id)));
  if(rows.has(model.id)){
    changed++;expected.review='source_reviewed';expected.summary=model.summary;expected.card.revision=version.revision;expected.card.sha256=version.sha256;
  }
  equal(model,expected);
  assert.equal(model.review,fit.method_review_status==='captured_source_and_receipt_reviewed'?'source_reviewed':'review_pending');
  if(model.review==='source_reviewed'){
    reviewed++;if(fit.task==='full_state_generator')full++;
    for(let i=0;i<fields.length;i++){
      const prefix='**'+labels[fields[i]]+'.** ',start=text.indexOf(prefix);assert.ok(start>=0);
      const endMarker=i+1<fields.length?'\n\n**'+labels[fields[i+1]]+'.** ':'\n\n## Exact source bindings for this review\n';
      const end=text.indexOf(endMarker,start+prefix.length);assert.ok(end>start);assert.equal(model.summary[fields[i]],text.slice(start+prefix.length,end).trim());paragraphs++;
    }
  }else assert.equal(model.summary,null);
}
assert.equal(changed,60);assert.equal(reviewed,205);assert.equal(paragraphs,1230);assert.equal(full,35);
assert.equal(prep.guide_revision,5);assert.equal(notes.get(guidePath).revision,5);
assert.deepEqual(fs.readFileSync(headerPath),headerBytes);
const result={status:'sixtythree_note_packet_205_reviews_all35_generators_independently_qualified',created_unix:Date.now()/1000,
  checkpoint:header.checkpoint,notes:63,new_reviewed_fits:60,reviewed_total:205,models:205,pending_total:0,
  full_state_generators_reviewed:35,literal_summary_paragraphs:1230,authored_comparisons:26,guide_revision:5,
  original_receipt_suffixes_preserved:suffixes,old_reviewed_models_groups_relationships_and_index_fields_preserved:true,
  actual_UI_codec_and_every_card_binding_passed:true,native_adoption_complete:false,full_portfolio_complete:false,
  paragraph_boundary_whitespace_normalized:true,new_model_forwards:0,payload_copies:0,producer:binding(__filename),
  evidence:[prepPath,reviewPath,codecPath,...prep.notes.map(n=>path.join(root,n.prepared_path))].map(binding)};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
process.stdout.write('63-note packet qualified:205 reviews/all35 generators/1230 paragraphs;60 original suffixes exact. No native mutation.\n');
