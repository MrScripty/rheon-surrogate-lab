/* Explicit live readback only after root's exact cutover/launch/machine gates. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const ENDPOINT='http://127.0.0.1:8091',CATALOG='a7da4a8de53541a2a0e2aad08005bb71';
const sha=raw=>crypto.createHash('sha256').update(raw).digest('hex');
function persist(file,value){const descriptor=fs.openSync(file,'wx',0o600);try{fs.writeFileSync(descriptor,JSON.stringify(value,null,2)+'\n');fs.fsyncSync(descriptor);}finally{fs.closeSync(descriptor);}const directory=fs.openSync(path.dirname(file),'r');try{fs.fsyncSync(directory);}finally{fs.closeSync(directory);}}
function validateEnvelope(config){
  assert.equal(config.format,'rheon_live_browser_readback_config_v1');assert.equal(config.root_trigger,true);
  assert.equal(config.endpoint,ENDPOINT);assert.equal(config.catalog_id,CATALOG);
  assert.equal(config.machine.status,'machine_qualified_browser_pending');assert.equal(config.machine.catalog_id,CATALOG);
  assert.equal(config.machine.endpoint,ENDPOINT);assert.equal(config.machine.optimizer_updates_executed,1);
  assert.equal(config.machine.restore_optimizer_updates,0);assert.equal(config.machine.probe.status,'exact');
  assert.equal(config.machine.probe.field_api_cli_exact,true);assert.equal(config.machine.probe.fresh_process,true);assert.equal(config.machine.probe.optimizer_updates,0);assert.equal(config.machine.ordinary_restore_bound_profile,false);assert.equal(config.machine.probe.cuda_initialized,false);
  assert.equal(config.machine.scientific_acceptance,'not_asserted');assert.equal(config.machine.model_promotion,false);
  assert.equal(config.launch.endpoint_owned_by_selected_pid,true);assert.equal(config.launch.deployment_manifest_sha256,config.deployment_manifest_sha256);
  assert.equal(Boolean(config.allow_project_import),!Boolean(config.existing_record));
  assert.equal(config.gates.length,4);
  assert.deepEqual(config.required_evidence,['dependency-migration-plan','dependency-migration-results','dependency-migration-scientific-parity']);
}
function allowImport(request,config,armed){
  const url=new URL(request.url),expected='/api/workspace/projects/'+config.provenance.project_id+'/import-data/'+config.import_reference;
  if(!armed||!config.allow_project_import||request.method!=='POST'||url.origin!==ENDPOINT||url.pathname!==expected||url.search)return false;
  try{const body=JSON.parse(request.body);return body.expected_sha256===config.import_sha256&&Object.keys(body).sort().join(',')==='expected_sha256,groups,name,parents,rights'&&Array.isArray(body.parents)&&body.parents.length===0&&body.rights==='unknown'&&typeof body.name==='string'&&body.name.startsWith('Current migration readback · ')&&JSON.stringify(body.groups)==='["dependency-migration-v1 readback"]';}catch{return false;}
}
function allowEvidenceRead(request,config){
  const selected=config.requests.field,url=new URL(request.url);
  const training=url.pathname===selected.path,restore=url.pathname===config.requests.restore.path+'/model-evidence';
  if(request.method!=='POST'||url.origin!==ENDPOINT||(!training&&!restore)||url.search||request.pin!=='runs.model_evidence/2/'+selected.sha256)return false;
  try{
    const body=JSON.parse(request.body),keys=Object.keys(body).sort();
    if(keys.length===0)return true;
    if(keys.length===1&&body.split==='validation')return true;
    return training&&keys.join(',')===Object.keys(selected.body).sort().join(',')&&keys.every(key=>body[key]===selected.body[key]);
  }catch{return false;}
}
function validateFiles(config){
  validateEnvelope(config);
  const documents=config.gates.map(gate=>{const info=fs.lstatSync(gate.path);assert(info.isFile()&&!info.isSymbolicLink()&&info.size<=8*1024*1024);const raw=fs.readFileSync(gate.path);assert.equal(sha(raw),gate.sha256);return JSON.parse(raw);});
  const [cutover,launch,machine,provenance]=documents;
  assert.equal(cutover.schema,'rheon-tuldok-migration-cutover-v1');assert.equal(cutover.m3_service_switched,true);
  assert.equal(cutover.ordinary_admissions_open,false);assert.equal(cutover.m3_complete,false);assert.deepEqual(cutover.smoke_authorized,{kind:'typed_pressure_one_update_cpu_restore',max_optimizer_updates:1});assert.equal(cutover.endpoint,ENDPOINT);assert.equal(cutover.catalog_id,CATALOG);
  assert.deepEqual(launch,config.launch);assert.deepEqual(machine,config.machine);assert.deepEqual(provenance,config.provenance);
  const raw=fs.readFileSync(path.join(config.application,'deployment-manifest.json'));assert.equal(sha(raw),config.deployment_manifest_sha256);
  assert.equal(sha(fs.readFileSync(path.join(config.application,'app.py'))),config.app_sha256);
}
async function openExistingField(editor,record,digest,waitForResponse){
  assert.equal(record.kind,'field_sequence');assert.equal(record.content_hash,digest);assert(/^[a-f0-9]{32}$/.test(record.id));
  const disclosure=editor.getByText('Filter collection',{exact:true});await disclosure.waitFor({state:'visible'});
  if(!await editor.locator('#kind').isVisible())await disclosure.click();
  await editor.locator('#kind').selectOption('field_sequence');
  await editor.locator('#query').fill('');
  await editor.locator('#group-filter').fill('field-sequence-source:'+digest);
  const pending=waitForResponse(response=>{
    const url=new URL(response.url());return response.request().method()==='GET'&&url.origin===ENDPOINT&&url.pathname==='/api/workbench/records'&&url.searchParams.get('kind')==='field_sequence'&&url.searchParams.get('group')==='field-sequence-source:'+digest&&url.searchParams.get('offset')==='0';
  });
  await editor.locator('#filters button').click();
  const response=await pending;assert(response.ok());const raw=await response.body();assert(raw.length<=8*1024*1024);const value=JSON.parse(raw);
  assert.equal(value.total,1);assert.equal(value.items.length,1);assert.equal(value.items[0].id,record.id);assert.equal(value.items[0].content_hash,digest);
  await editor.locator('[data-record-id="'+record.id+'"]').click();
  return {method:'GET',path:new URL(response.url()).pathname+new URL(response.url()).search,sha256:sha(raw),record_id:record.id,total:1,collection_writes:0};
}
function activityHealth(activity){
  assert.equal(activity?.available,true);assert.equal(activity.authority,'external_observation');assert.equal(activity.stale,false);
  assert(['running','completed'].includes(activity.state));assert.equal(activity.phase,'Dependency migration observation');
  assert.equal(activity.progress.completed,74);assert.equal(activity.progress.total,74);
  assert(activity.message.includes('74/74 current regression case receipts passed'));
  assert(activity.message.includes('0 failed, 0 unreadable'));
  assert(activity.message.includes('typed-pressure-live-01: status=machine_qualified_browser_pending'));
  assert(!activity.message.includes('Observation unavailable'));
  return {authority:activity.authority,healthy:true,state:activity.state,phase:activity.phase,progress:activity.progress,machine_status:'machine_qualified_browser_pending',message:activity.message,message_sha256:sha(Buffer.from(activity.message))};
}
async function main(file){
  const config=JSON.parse(fs.readFileSync(file,'utf8'));validateFiles(config);
  const output=config.output;assert.equal(path.resolve(path.dirname(file)),path.resolve(output));assert(!fs.existsSync(path.join(output,'browser-receipt.json')));
  const authority=fs.lstatSync(config.owner_session_file);assert(authority.isFile()&&!authority.isSymbolicLink()&&authority.uid===process.getuid()&&(authority.mode&0o077)===0&&authority.size<=4096);
  let token=fs.readFileSync(config.owner_session_file,'ascii').trim();assert(token.length>=32&&!/[\x00-\x20\x7f;]/.test(token));
  const {chromium}=require('/home/jeremy/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');
  const started=performance.now(),requests=[],readPosts=[],mutations=[],violations=[],errors=[],screenshots=[],apiReceipts=[];
  let browser,context,phase='browser launch',armed=false,importDispatched=false;
  const write=(name,value)=>persist(path.join(output,name),value);
  try{
    browser=await chromium.launch({executablePath:config.chromium,headless:true,args:['--disable-gpu']});
    context=await browser.newContext({viewport:{width:1440,height:1000},serviceWorkers:'block'});
    await context.addCookies([{name:'tuldok_session',value:token,url:ENDPOINT,httpOnly:true,sameSite:'Strict'}]);token=null;
    await context.route('**/*',async route=>{
      const request=route.request(),url=new URL(request.url()),method=request.method();
      const entry={method,path:url.pathname+url.search};requests.push(entry);
      if(url.origin!==ENDPOINT){violations.push({kind:'other_origin',method});await route.abort();return;}
      if(method==='GET'||method==='HEAD'){await route.continue();return;}
      if(allowEvidenceRead({url:request.url(),method,body:request.postData(),pin:request.headers()['tuldok-operation-contract']},config)){readPosts.push({...entry,effects:'read',operation:'runs.model_evidence/2'});await route.continue();return;}
      if(allowImport({url:request.url(),method,body:request.postData()},config,armed)&&!importDispatched){
        armed=false;importDispatched=true;mutations.push({...entry,expected_sha256:config.import_sha256,effect:'new collection item; source unchanged'});await route.continue();return;
      }
      violations.push({kind:'forbidden_mutation',...entry});await route.abort();
    });
    async function get(route,contract){
      assert(route.startsWith('/api/')&&!route.includes('://'));const headers=contract?{'Content-Type':'application/json','Tuldok-Operation-Contract':contract.operation+'/'+contract.version+'/'+contract.sha256}:{};
      const method=contract?.method||'GET';assert(method==='GET'||contract.operation==='runs.model_evidence'&&method==='POST');
      const response=await context.request.fetch(ENDPOINT+route,{method,headers,...(method==='POST'?{data:contract.body}:{}),maxRedirects:0,timeout:45000});
      if(!response.ok()){const error=new Error('Readback read operation failed');error.status=response.status();throw error;}
      const raw=await response.body();assert(raw.length<=8*1024*1024);apiReceipts.push({method,effects:'read',path:route,sha256:sha(raw),bytes:raw.length});return JSON.parse(raw);
    }
    const call=key=>get(config.requests[key].path,config.requests[key]);
    phase='existing owner and current catalog identity';
    const identity=await get('/api/access/session'),catalog=await get('/api/workspace/context');
    assert.equal(identity.kind,'owner');assert.equal(identity.catalog_id,CATALOG);assert.equal(catalog.dataset_id,CATALOG);
    const capabilities=await get('/api/agent/capabilities');assert.equal(capabilities.catalog_id,CATALOG);
    for(const contract of Object.values(config.requests))assert(capabilities.operations.some(op=>op.id===contract.operation&&op.contracts.some(v=>v.version===contract.version&&v.sha256===contract.sha256)));
    const runsBefore=await get('/api/workspace/runs');const runIDsBefore=runsBefore.map(row=>row.id).sort();
    phase='registered selected current Project';
    const project=await get('/api/workspace/project-snapshots/'+config.provenance.project_id);
    assert.equal(project.id,config.provenance.project_id);assert.equal(project.manifest_sha256,config.provenance.current_manifest_sha256);assert.equal(project.manifest_status,'unchanged');assert.equal(project.schema_version,2);
    assert.equal(project.activity.authority,'external_observation');assert.equal(project.activity.available,true);
    activityHealth(project.activity);
    const selected=project.data.find(row=>row.id===config.import_reference);assert(selected);assert.equal(selected.status,'available');assert.notEqual(selected.integrity,'mismatch');assert.equal(selected.observed_sha256,config.import_sha256);
    const page=await context.newPage();page.on('pageerror',()=>errors.push({kind:'pageerror',phase}));
    const capture=async name=>{const file=name+'.png';await page.screenshot({path:path.join(output,file),fullPage:true});screenshots.push(file);};
    await page.goto(ENDPOINT+'/workspaces?project='+project.id+'#data');await page.locator('#project-data-root .project-header').waitFor();
    await capture('current-project-data');
    phase='external evidence and actual activity refresh';
    await page.locator('#nav-runs').click();await page.locator('[data-project-evidence="activity"]').click();await page.locator('#project-runs-root .project-evidence-output pre').waitFor();
    const evidence=await get('/api/workspace/projects/'+project.id+'/evidence/activity');assert.equal(evidence.external,true);assert.equal(evidence.scientific_verification,'not_performed');
    const first=project.activity.sequence,firstPanel=await page.locator('#project-runs-root .project-live-activity').innerText();let refreshed;
    for(let i=0;i<12;i++){await page.waitForTimeout(1000);refreshed=await get('/api/workspace/project-snapshots/'+project.id);if(refreshed.activity.available&&refreshed.activity.sequence>first)break;}
    assert(refreshed.activity.sequence>first,'Actual external observation did not advance');
    const healthyActivity=activityHealth(refreshed.activity);
    await page.waitForFunction(previous=>{const panel=document.querySelector('#project-runs-root .project-live-activity');return panel&&panel.innerText!==previous&&panel.innerText.includes('External project activity.');},firstPanel);
    const activityPanel=await page.locator('#project-runs-root .project-live-activity').innerText();
    assert(activityPanel.includes('74 / 74')&&activityPanel.includes('typed-pressure-live-01: status=machine_qualified_browser_pending')&&!activityPanel.includes('Observation unavailable'));
    assert(requests.filter(row=>row.path==='/api/workspace/project-snapshots/'+project.id).length>=2,'Native page did not refresh Project observations');
    await capture('current-project-activity');
    const inspectedEvidence=[];
    for(const id of config.required_evidence){
      const reference=project.evidence.find(row=>row.id===id);assert(reference&&reference.status==='available');
      await page.locator('[data-project-evidence="'+id+'"]').click();
      const value=await get('/api/workspace/projects/'+project.id+'/evidence/'+id);
      assert.equal(value.external,true);assert.equal(value.scientific_verification,'not_performed');assert.equal(value.id,id);assert.equal(typeof value.text,'string');assert(Buffer.byteLength(value.text)<=65536);
      await page.waitForFunction(digest=>document.querySelector('#project-runs-root .project-evidence-output')?.textContent.includes(digest),value.sha256);
      assert.equal(await page.locator('#project-runs-root .project-evidence-output pre').innerText(),value.text);
      inspectedEvidence.push({id,sha256:value.sha256,bytes:Buffer.byteLength(value.text),authority:'external_evidence',scientific_verification:'not_performed'});
      await capture('current-project-'+id);
    }
    phase='A5 collection field import or exact existing record';let record;
    if(config.allow_project_import){
      await page.locator('#nav-data').click();await page.locator('[data-project-import="'+config.import_reference+'"]').click();await page.locator('#project-import-dialog').waitFor();
      assert((await page.locator('#project-import-identity').innerText()).includes(config.import_sha256));
      await page.locator('#project-import-name').fill('Current migration readback · '+selected.label);await page.locator('#project-import-rights').fill('unknown');await page.locator('#project-import-group').fill('dependency-migration-v1 readback');
      const claim=path.join(path.dirname(config.gates[0].path),'.live-browser-import-'+config.gates[0].sha256+'-'+project.id+'-'+config.import_reference+'.json');
      persist(claim,{phase:'import_dispatch_intent',output,project_id:project.id,reference_id:config.import_reference,expected_sha256:config.import_sha256,automatic_retry:false});
      write('import-intent.json',{claim,project_id:project.id,reference_id:config.import_reference,expected_sha256:config.import_sha256,effect:'Creates one new draft collection item; not a GET; no review/freeze/acceptance'});
      armed=true;const replyPromise=page.waitForResponse(response=>response.request().method()==='POST'&&new URL(response.url()).pathname==='/api/workspace/projects/'+project.id+'/import-data/'+config.import_reference);
      await page.locator('#project-import-submit').click();const reply=await replyPromise;assert(reply.ok());record=await reply.json();write('import-acknowledged.json',{record_id:record.id,kind:record.kind,content_hash:record.content_hash,review:record.review});
    }else record=await get('/api/workbench/records/'+config.existing_record);
    assert.equal(record.kind,'field_sequence');assert.equal(record.content_hash,config.import_sha256);
    let existingLookup=null;
    if(!config.allow_project_import){await page.locator('#dataset').selectOption('catalog');await page.locator('#nav-data').click();existingLookup=await openExistingField(page.frameLocator('#catalog-frame'),record,config.import_sha256,predicate=>page.waitForResponse(predicate));}
    const editor=page.frameLocator('#catalog-frame');await editor.locator('#scientific-inspection').waitFor();await editor.locator('#scientific-canvas').waitFor({state:'visible'});assert((await editor.locator('#scientific-status').innerText()).startsWith('Frame '));await capture('current-project-collection-field');const collectionStatus=await editor.locator('#scientific-status').innerText();
    phase='actual Model Study Trial API and native experiment view';
    const model=await call('model'),study=await call('study'),trial=await call('trial'),train=await call('train'),restore=await call('restore');
    assert.equal(model.id,config.machine.model.id);assert.equal(model.revision,config.machine.model.revision);assert.equal(trial.id,config.machine.trial.id);assert.deepEqual(trial.material,config.machine.trial.material);assert.equal(study.study.open,false);assert(study.study.trial_ids.includes(trial.id));
    assert.equal(train.state,'completed');assert.equal(restore.state,'completed');assert.equal(train.custody_phase,'drained');assert.equal(restore.custody_phase,'drained');assert.equal(train.study_membership.trial_id,trial.id);
    const exactArtifact=(run,name)=>{const values=run.artifacts.filter(row=>row.name===name);assert.equal(values.length,1);return values[0];};
    const checkpoint=exactArtifact(train,'checkpoint-001.npz');assert.equal(checkpoint.sha256,config.machine.checkpoint_sha256);assert.equal(checkpoint.sha256,exactArtifact(restore,'checkpoint-001.npz').sha256);assert.equal(checkpoint.sha256,config.machine.probe.weights_sha256);
    const runReceipts=[];
    for(const [kind,run] of [['train',train],['restore',restore]]){
      const item=exactArtifact(run,'receipt.json');const response=await context.request.get(ENDPOINT+'/api/workspace/runs/'+run.id+'/artifacts/receipt.json',{maxRedirects:0,timeout:45000});assert(response.ok());const raw=await response.body();assert(raw.length<=4*1024*1024);assert.equal(sha(raw),item.sha256);const value=JSON.parse(raw);assert.equal(value.mode,kind==='train'?'train':'restore');assert.equal(value.optimizer_updates,kind==='train'?1:0);if(kind==='restore'){assert.equal(value.execution.device,'cpu');assert.equal(value.execution.cuda_initialized,false);assert.equal(value.originating_training_run.run_id,train.id);}runReceipts.push({kind,run_id:run.id,artifact:'receipt.json',sha256:item.sha256});
    }
    await page.locator('#dataset').selectOption('catalog');await page.locator('#nav-pipeline').click();await page.locator('#lineage-node-'+trial.id).waitFor();await page.locator('#lineage-node-'+trial.id).click();assert((await page.locator('#lineage-detail').innerText()).includes(trial.material.sha256));await capture('actual-study-trial');
    phase='actual pressure field pair and scalar checkpoint table';
    await page.locator('#nav-runs').click();await page.locator('#real-run-'+train.id).click();await page.locator('#field-prediction').waitFor();
    await page.locator('#field-checkpoint').selectOption(config.machine.browser.checkpoint_sha256);await page.locator('#field-member').selectOption(String(config.machine.browser.member_ordinal));await page.locator('#field-id').selectOption(config.machine.browser.field_id);await page.locator('#field-axis').selectOption('z');await page.locator('#field-index').fill('0');await page.locator('#field-load').click();
    const field=await call('field'),catalogEvidence=await call('evidence');assert.equal(field.field.units,'Pa');assert.equal(field.field.dtype,'<f8');assert.equal(field.field.grid.units,'m');assert.equal(catalogEvidence.manifest.model_sha256,config.machine.model.sha256);assert.equal(catalogEvidence.manifest.producer.material_sha256,trial.material.sha256);
    const retained=JSON.parse(fs.readFileSync(path.join(path.dirname(config.gates[2].path),'field-api.json'),'utf8'));assert.deepEqual(field,retained);
    await page.locator('#field-probe-column').fill('1');await page.locator('#field-probe-column').dispatchEvent('change');await page.waitForFunction(()=>document.querySelector('#field-probe-value')?.textContent.includes('column 1:'));
    assert((await page.locator('#field-probe-value').innerText()).includes('Pa'));assert((await page.locator('.field-evidence table').innerText()).includes('mean_free'));
    await capture('cpu-matching-pressure-and-scalar-table');
    const fieldView={run_id:train.id,checkpoint_sha256:config.machine.browser.checkpoint_sha256,member_ordinal:config.machine.browser.member_ordinal,field_id:config.machine.browser.field_id,units:field.field.units,dtype:field.field.dtype,grid:field.field.grid,gauge_policy_sha256:field.field.gauge_policy_sha256,time_units:field.field.time_units,probe_row:0,probe_column:1,prediction:field.field.prediction[0][1],reference:field.field.reference[0][1],visible_probe:await page.locator('#field-probe-value').innerText(),view:'Actual orthogonal pressure pair and scalar checkpoint table; no physical timeline',cpu_restore_relation:'Machine fresh CPU restore probe exactly equals retained SDK pressure; ordinary restore remains outside bound field profile',cpu_pressure_sha256:config.machine.probe.pressure_sha256};
    phase='ordinary CPU restore artifacts';await page.locator('#real-run-'+restore.id).click();await page.waitForFunction(id=>document.querySelector('[data-real-run="'+id+'"]')?.getAttribute('aria-current')==='true'&&!document.getElementById('field-prediction'),restore.id);assert.equal(await page.locator('#real-run-state').innerText(),'completed');assert.equal(await page.locator('#field-prediction').count(),0);await capture('ordinary-cpu-restore-artifacts');
    const runsAfter=await get('/api/workspace/runs');assert.deepEqual(runsAfter.map(row=>row.id).sort(),runIDsBefore);assert.deepEqual(violations,[]);assert.deepEqual(errors,[]);assert.equal(mutations.length,config.allow_project_import?1:0);
    write('browser-receipt.json',{schema:'rheon-root-gated-live-browser-readback-v1',status:'passed',endpoint:ENDPOINT,catalog_id:CATALOG,gates:config.gates,application:config.application,deployment_manifest_sha256:config.deployment_manifest_sha256,project_id:project.id,project_manifest_sha256:project.manifest_sha256,activity:{...healthyActivity,first_sequence:first,last_sequence:refreshed.activity.sequence,visible_panel_sha256:sha(Buffer.from(activityPanel))},external_evidence:inspectedEvidence,collection:{record_id:record.id,content_hash:record.content_hash,created_new_item:config.allow_project_import,read_only_reuse:!config.allow_project_import,visible_status:collectionStatus,existing_record_lookup:existingLookup},model:config.machine.model,trial,study_id:trial.study_id,composition_sha256:trial.material.sha256,training_run_id:train.id,restore_run_id:restore.id,checkpoint_sha256:checkpoint.sha256,run_receipts:runReceipts,field_view:fieldView,requests,read_only_evidence_posts:readPosts,api_receipts:apiReceipts,mutations,extra_runs:0,approval_or_admission_or_promotion:false,scientific_acceptance:'not_asserted',screenshots,elapsed_s:(performance.now()-started)/1000});
    process.stdout.write('Root-gated live readback passed; no new Run or acceptance claim.\n');
  }catch(error){
    const site=String(error.stack||'').match(/live_readback\.cjs:(\d+):(\d+)/);
    write('failure.json',{status:'stopped_inspect_before_retry',phase,error_kind:error.name,assertion_site:site?{file:'live_readback.cjs',line:Number(site[1]),column:Number(site[2])}:null,http_status:error.status||null,mutations,import_dispatch_may_have_committed:importDispatched,automatic_retry:false,authority_logged:false});
    process.stderr.write('Live readback stopped during '+phase+'; inspect bounded failure receipt.\n');process.exitCode=1;
  }finally{token=null;await context?.close();await browser?.close();}
}
module.exports={validateEnvelope,allowImport,allowEvidenceRead,validateFiles,openExistingField,activityHealth};
if(require.main===module)main(process.argv[2]).catch(()=>{process.stderr.write('Live readback gates or private authority validation failed; no retry.\n');process.exitCode=1;});
