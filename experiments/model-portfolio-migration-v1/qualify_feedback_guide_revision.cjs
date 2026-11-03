'use strict';
// Validate the next guide against exact native or proposed card bytes and the
// independent reviewed index, before any current project file is replaced.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),doc=path.join(root,'docs/plans/model-portfolio-migration-v1/reports');
const packetRoot=path.join(root,'runs/model-portfolio-migration-v1/native-feedback-review-publication-01');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04');
const source=path.join(root,'runs/model-portfolio-migration-v1/sources/Tuldok-composition-source-artifacts');
const output=path.join(doc,'feedback-guide-revision-preparation-01.json');
const read=file=>JSON.parse(fs.readFileSync(file,'utf8')),hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const bind=file=>({path:path.relative(root,file),sha256:hash(fs.readFileSync(file))});
assert.ok(!fs.existsSync(output),'Preserve original guide qualification');
const prepPath=path.join(packetRoot,'preparation.json'),prep=read(prepPath);
const headerPath=path.join(project,'.tuldok/project.json'),headerRaw=fs.readFileSync(headerPath),header=JSON.parse(headerRaw);
assert.equal(hash(headerRaw),prep.source_manifest_sha256);assert.equal(header.checkpoint,prep.source_checkpoint);
assert.equal(prep.notes.length,13);assert.equal(new Set(prep.notes.map(n=>n.path)).size,13);
const qualificationPath=path.join(doc,'feedback-method-reviews-native-preparation-01.json'),qualification=read(qualificationPath);
assert.equal(qualification.status,'guarded_review_batch_prepared_with_exact_original_evidence');
assert.equal(qualification.new_reviewed_fits,10);assert.equal(qualification.reviewed_fit_cards_after_adoption,79);
for(const spec of qualification.evidence)assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
for(const spec of prep.evidence)assert.equal(hash(fs.readFileSync(path.join(root,spec.path))),spec.sha256);
const captured=new Map();
for(const entry of header.inventory)if(entry.family==='project_file'){
  const raw=fs.readFileSync(path.join(project,'.tuldok',entry.path));assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);
  const value=JSON.parse(raw);if(value.project_id===header.project_id){assert.ok(!captured.has(value.path));captured.set(value.path,{entry,value});}
}
const changes=new Map(prep.notes.map(n=>[n.path,n]));
for(const note of prep.notes){
  const raw=fs.readFileSync(path.join(root,note.prepared_path)),old=captured.get(note.path).value.current;
  assert.equal(hash(raw),note.sha256);assert.equal(raw.length,note.bytes);assert.ok(raw.length<=1048576);
  assert.equal(old.sha256,note.previous_sha256);assert.equal(old.revision,note.expected_revision);assert.equal(note.revision,old.revision+1);
  assert.equal(hash(fs.readFileSync(path.join(project,note.path))),old.sha256);
}
const window={};vm.runInNewContext(fs.readFileSync(path.join(source,'static/project-model-portfolio.js'),'utf8'),{window,TextEncoder});
const codec=window.ProjectModelPortfolioDocument;
const guideNote=changes.get('notes/00-models.portfolio.json');
const guide=codec.decode(fs.readFileSync(path.join(root,guideNote.prepared_path),'utf8'));
const oldGuide=codec.decode(captured.get(guideNote.path).value.current.text);
const index=read(path.join(root,changes.get('notes/portfolio-index.json').prepared_path));
const fits=new Map(index.fits.map(f=>[f.item_id,f]));
assert.equal(guide.models.length,205);assert.equal(fits.size,205);assert.equal(guide.project_id,header.project_id);
assert.deepEqual([...guide.models.map(m=>m.id)].sort(),[...fits.keys()].sort());
assert.deepEqual(JSON.parse(JSON.stringify(guide.groups)),JSON.parse(JSON.stringify(oldGuide.groups)));
assert.deepEqual(JSON.parse(JSON.stringify(guide.relationships.slice(0,16))),JSON.parse(JSON.stringify(oldGuide.relationships)));
const oldModels=new Map(oldGuide.models.map(m=>[m.id,m]));let reviewed=0,paragraphs=0,changed=0;
const labels={method:'Method',purpose:'Purpose',experiment:'Experiment',outcome:'Observed outcome',lesson:'Lesson',reproduction:'Reproduction'};
const texts=new Map();
for(const model of guide.models){
  const fit=fits.get(model.id),retained=captured.get(fit.card_path),note=changes.get(fit.card_path);
  const text=note?fs.readFileSync(path.join(root,note.prepared_path),'utf8'):retained.value.current.text;
  const version=note?{...retained.value.current,text,revision:note.revision,sha256:note.sha256}:retained.value.current;
  assert.equal(hash(Buffer.from(text)),version.sha256);assert.equal(version.sha256,fit.card_sha256);
  assert.equal(model.card.file_id,retained.entry.id);assert.equal(model.card.path,fit.card_path);
  assert.equal(model.card.revision,version.revision);assert.equal(model.card.sha256,version.sha256);
  codec.checkCard(guide,model,{id:retained.entry.id,project_id:header.project_id,kind:'note',path:fit.card_path,version});
  assert.equal(model.review,fit.method_review_status==='captured_source_and_receipt_reviewed'?'source_reviewed':'review_pending');
  if(note){
    changed++;assert.equal(oldModels.get(model.id).review,'review_pending');
    const expected=JSON.parse(JSON.stringify(oldModels.get(model.id)));expected.review=model.review;expected.summary=JSON.parse(JSON.stringify(model.summary));expected.card.revision=version.revision;expected.card.sha256=version.sha256;
    assert.deepEqual(JSON.parse(JSON.stringify(model)),expected);
  }else assert.deepEqual(JSON.parse(JSON.stringify(model)),JSON.parse(JSON.stringify(oldModels.get(model.id))));
  if(model.review==='source_reviewed'){
    reviewed++;const fields=Object.keys(labels);
    for(let i=0;i<fields.length;i++){
      const field=fields[i],prefix='**'+labels[field]+'.** ',start=text.indexOf(prefix);assert.ok(start>=0);
      const suffix=i+1<fields.length?'\n\n**'+labels[fields[i+1]]+'.** ':'\n\n## Exact source bindings for this review\n';
      const end=text.indexOf(suffix,start+prefix.length);assert.ok(end>start);
      assert.equal(model.summary[field],text.slice(start+prefix.length,end).trim());paragraphs++;
    }
  }else assert.equal(model.summary,null);
  texts.set(model.id,text);
}
assert.equal(changed,10);assert.equal(reviewed,79);assert.equal(paragraphs,474);assert.equal(guide.relationships.length,26);
const models=new Map(guide.models.map(m=>[m.id,m]));
for(const relation of guide.relationships){
  assert.equal(relation.kind,'comparison');assert.ok(texts.get(relation.evidence_model).includes(relation.excerpt));
  assert.ok(models.get(relation.to).aliases.some(a=>/^C\d+$/.test(a)&&new RegExp('(?:^|\\W)'+a+'(?!\\d)').test(relation.excerpt)));
}
assert.deepEqual(fs.readFileSync(headerPath),headerRaw);
const result={status:'thirteen_note_packet_79_reviewed_205_fit_guide_exact',created_unix:Date.now()/1000,
  checkpoint:header.checkpoint,notes:13,new_reviewed_fits:10,reviewed_total:79,pending_total:126,
  models:205,literal_summary_paragraphs:474,authored_comparisons:26,guide_revision:2,
  all_original_models_groups_and_comparisons_preserved:true,all_new_card_bindings_and_literal_summaries_exact:true,
  native_adoption_complete:false,full_portfolio_complete:false,new_model_forwards:0,payload_copies:0,
  evidence:[prepPath,qualificationPath,path.join(source,'static/project-model-portfolio.js'),...prep.notes.map(n=>path.join(root,n.prepared_path))].map(bind),
  producer:bind(__filename)};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
process.stdout.write('13-note packet independently qualified;205fits/79reviews/474literal paragraphs/26comparisons. No native mutation.\n');
