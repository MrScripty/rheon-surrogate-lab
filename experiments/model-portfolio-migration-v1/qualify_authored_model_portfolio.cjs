'use strict';
// Independent check against authoritative native envelopes/index, not the
// preparer's in-memory objects. Does not open SQL, a Store, weights or models.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'runs/model-portfolio-migration-v1'),doc=path.join(root,'docs/plans/model-portfolio-migration-v1/reports');
const project=path.join(root,'projects/rheon-model-portfolio-unified-04'),prepared=path.join(run,'authored-model-portfolio-note-01');
const source=path.join(run,'sources/Tuldok-authored-model-portfolio'),output=path.join(doc,'authored-model-portfolio-note-qualification-01.json');
const read=file=>JSON.parse(fs.readFileSync(file,'utf8')),hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const binding=file=>({path:path.relative(root,file),sha256:hash(fs.readFileSync(file))});
if(fs.existsSync(output))throw Error('Preserve original guide qualification');
const prep=read(path.join(prepared,'preparation.json')),headerPath=path.join(project,'.tuldok/project.json'),headerRaw=fs.readFileSync(headerPath),header=JSON.parse(headerRaw);
assert.equal(hash(headerRaw),prep.source_manifest_sha256);assert.equal(header.checkpoint,prep.source_checkpoint);
const bodies=new Map();
for(const entry of header.inventory)if(entry.family==='project_file'){
  const raw=fs.readFileSync(path.join(project,'.tuldok',entry.path));assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);
  const value=JSON.parse(raw);if(value.project_id===header.project_id){assert.ok(!bodies.has(value.path));bodies.set(value.path,{entry,value});}
}
const index=JSON.parse(bodies.get('notes/portfolio-index.json').value.current.text),raw=fs.readFileSync(path.join(prepared,prep.guide_path));
assert.equal(raw.length,prep.guide_bytes);assert.equal(hash(raw),prep.guide_sha256);assert.ok(raw.length<=1048576);
const window={};vm.runInNewContext(fs.readFileSync(path.join(source,'static/project-model-portfolio.js'),'utf8'),{window,TextEncoder});
const guide=window.ProjectModelPortfolioDocument.decode(raw.toString('utf8'));
assert.equal(guide.project_id,header.project_id);assert.equal(guide.models.length,205);
assert.equal(index.fits.length,205);assert.equal(index.aliases.length,379);
const fits=new Map(index.fits.map(fit=>[fit.item_id,fit])),models=new Map(guide.models.map(model=>[model.id,model]));
assert.deepEqual([...models.keys()].sort(),[...fits.keys()].sort());
const groups=new Map(guide.groups.map(group=>[group.id,group]));
let reviewed=0,paragraphs=0,aliases=0;
const labels={method:'Method',purpose:'Purpose',experiment:'Experiment',outcome:'Observed outcome',lesson:'Lesson',reproduction:'Reproduction'};
const order=Object.entries(labels);
for(const model of guide.models){
  const fit=fits.get(model.id),captured=bodies.get(fit.card_path);assert.ok(captured);
  const {entry,value}=captured,version=value.current;
  assert.equal(model.card.file_id,entry.id);assert.equal(model.card.path,fit.card_path);assert.equal(model.card.revision,version.revision);
  assert.equal(model.card.sha256,version.sha256);assert.equal(hash(Buffer.from(version.text)),fit.card_sha256);assert.equal(version.deleted,undefined);
  let group=model.group_id;while(groups.get(group).parent!==null)group=groups.get(group).parent;
  assert.equal(group,fit.task);assert.ok(model.label.includes(fit.name));
  const expectedAliases=[...new Set(index.aliases.filter(alias=>alias.fit_item_id===model.id).map(alias=>String(alias.label)))].sort();
  assert.deepEqual([...model.aliases],expectedAliases);aliases+=model.aliases.length;
  const isReviewed=fit.method_review_status==='captured_source_and_receipt_reviewed';
  assert.equal(model.review,isReviewed?'source_reviewed':'review_pending');
  window.ProjectModelPortfolioDocument.checkCard(guide,model,{id:entry.id,project_id:value.project_id,kind:value.kind,path:value.path,version});
  if(isReviewed){reviewed++;
    // A distinct delimiter walk checks the complete native paragraphs; a
    // merely matching substring cannot substitute for the reviewed summary.
    for(const [index,[field,label]] of order.entries()){
      const prefix='**'+label+'.** ',start=version.text.indexOf(prefix);assert.ok(start>=0);
      const next=index+1<order.length?'\n\n**'+order[index+1][1]+'.** ':'\n\n## Exact source bindings for this review\n';
      const end=version.text.indexOf(next,start+prefix.length);assert.ok(end>start);
      // Boundary padding belongs to the Markdown layout, not the summary.
      // checkCard separately requires this complete string to occur literally.
      assert.equal(model.summary[field],version.text.slice(start+prefix.length,end).trim());paragraphs++;
    }
  }else assert.equal(model.summary,null);
}
assert.equal(reviewed,69);assert.equal(paragraphs,414);assert.equal(guide.relationships.length,16);
for(const relation of guide.relationships){
  assert.equal(relation.kind,'comparison');assert.ok(models.has(relation.from)&&models.has(relation.to));
  const evidence=models.get(relation.evidence_model),card=bodies.get(evidence.card.path).value.current;
  assert.ok(card.text.includes(relation.excerpt));assert.equal(evidence.review,'source_reviewed');
  const target=models.get(relation.to),named=target.aliases.filter(alias=>/^C\d{2,3}$/.test(alias));
  assert.ok(named.some(alias=>new RegExp('(?:^|\\W)'+alias+'(?!\\d)').test(relation.excerpt)));
}
assert.deepEqual(fs.readFileSync(headerPath),headerRaw);
const value={status:'all205_fit_bindings_414_reviewed_paragraphs_and16_authored_comparisons_exact',created_unix:Date.now()/1000,
  checkpoint:header.checkpoint,guide_bytes:raw.length,guide_sha256:hash(raw),fits:205,source_reviewed:69,review_pending:136,
  exact_review_paragraphs:414,exact_alias_links:aliases,authored_comparisons:16,parameter_derivation_or_Trial_lineage_inferred:false,
  all_captured_card_bodies_and_note_bindings_checked:true,source_native_checkpoint_unchanged:true,
  native_adoption_complete:false,actual_native_UI_complete:false,new_neural_forwards:0,SQL_opened:false,payload_copies:0,full_portfolio_complete:false,
  evidence:[binding(path.join(prepared,'preparation.json')),binding(path.join(prepared,prep.guide_path)),binding(path.join(source,'static/project-model-portfolio.js'))],
  producer:binding(__filename)};
fs.writeFileSync(output,JSON.stringify(value,null,2)+'\n',{flag:'wx'});console.log(value.status);
