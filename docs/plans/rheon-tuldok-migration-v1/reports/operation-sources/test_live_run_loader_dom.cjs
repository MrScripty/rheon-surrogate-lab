/* Actual deployed Run loader + lineage click owners, delayed metadata-only reads.
   Orthogonal model/package views are inert; no scientific output is fabricated. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {openTrainingOutcome}=require('./live_readback.cjs');const app=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);
(async()=>{
 const {chromium}=require('/home/jeremy/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');let browser,context;const requests=[],errors=[];let baseline,green,kind;
 try{
  browser=await chromium.launch({executablePath:'/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--disable-gpu']});context=await browser.newContext({serviceWorkers:'block'});
  const train='7'.repeat(32),restore='8'.repeat(32),trial='2'.repeat(32),catalog='1'.repeat(32);
  const run=id=>({id,state:'completed',graph:{recipe:'external_recipe',name:'Metadata-only loader fixture'},graph_revision:1,attempt:1,created_at:'2026-01-01',events:[],artifacts:[]});
  const node={trial:{id:trial,material:{catalog_id:catalog,sha256:'a'.repeat(64)}},study:{name:'Loader fixture'},recipe:{id:'3'.repeat(32),revision:1},runs:[{id:train,state:'completed',validation:{}}],attempt_count:1,lineage:null};
  await context.route('**/*',async route=>{const r=route.request(),u=new URL(r.url());requests.push({method:r.method(),path:u.pathname,fulfilled_locally:true});assert.equal(r.method(),'GET');
   if(u.pathname==='/fixture')return route.fulfill({contentType:'text/html',body:`<div id="pipeline"></div><div id="runs"></div><div id="lineage"></div><script>
   window.WorkspaceModelContracts={};window.workspaceEscape=x=>String(x);window.TuldokAuth={identity:()=>({kind:'owner',actor_id:'fixture',credential_id:'fixture',scopes:[]})};
   const inert=extra=>new Proxy(extra||{},{get:(o,k)=>k in o?o[k]:()=>{}});
   window.WorkspaceExternal=()=>inert({isGraph:g=>g?.recipe==='external_recipe',runHTML:()=>''});
   window.WorkspaceModel=()=>inert({load:async()=>{},html:()=>''});window.WorkspaceFieldEvidence=()=>inert({load:async()=>new Promise(r=>setTimeout(r,200)),html:()=>''});window.WorkspaceCandidatePackageEvidence=()=>inert({load:async()=>{},html:()=>''});
   </script><script src="/experiment-lineage.js"></script><script>
   window.NativeLineage=window.WorkspaceExperimentLineage;window.WorkspaceExperimentLineage=options=>{window.actualRunCallback=options.openRun;return inert();};
   </script><script src="/workspace-real.js"></script><script>
   window.fixtureView='runs';window.owner=WorkspaceReal({pipelineRoot:document.getElementById('pipeline'),runsRoot:document.getElementById('runs'),announce:()=>{},openRecord:()=>{},openImportedRecord:()=>{},openRun:()=>{fixtureView='runs';window.pending=owner.show('runs');}});
   window.startDefault=()=>{fixtureView='runs';window.pending=owner.show('runs');};
   window.mountTrial=()=>{fixtureView='pipeline';const lineage=NativeLineage({api:async()=>(${JSON.stringify({nodes:[node],related:[],revision:'revision',next_cursor:null})}),announce:()=>{},getIntent:()=>0,getView:()=>fixtureView,getGraph:()=>null,isDirty:()=>false,openRecipe:()=>{},openRun:actualRunCallback,compare:()=>{}});lineage.mount(document.getElementById('lineage'),{dataset_id:'${catalog}'});};
   </script>`});
   if(['/workspace-real.js','/experiment-lineage.js'].includes(u.pathname))return route.fulfill({path:path.join(app,'static',u.pathname.slice(1))});
   if(u.pathname==='/api/workspace/runs'){await new Promise(r=>setTimeout(r,300));return route.fulfill({json:[run(restore),run(train)]});}
   if(u.pathname.endsWith('/progress'))return route.fulfill({json:{state:'completed'}});
   if(u.pathname==='/api/workspace/runs/'+train||u.pathname==='/api/workspace/runs/'+restore)return route.fulfill({json:run(u.pathname.split('/').pop())});
   return route.fulfill({status:404,body:'Fixture resource absent'});
  });
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.name));await page.goto('http://127.0.0.1:8091/fixture');await page.evaluate(()=>startDefault());
  try{await page.locator('#real-run-'+train).click({timeout:50});baseline='unexpected_pass';}catch(error){baseline=error.name;}assert.equal(baseline,'TimeoutError');assert((await page.locator('#runs').innerText()).includes('Loading durable run receipts'));
  await page.evaluate(()=>window.pending);await page.evaluate(()=>mountTrial());await page.locator('#lineage-node-'+trial).click();green=await openTrainingOutcome(page,train,1500);assert.equal(await page.locator('#real-run-'+train).getAttribute('aria-current'),'true');assert.deepEqual(errors,[]);
 }catch(error){kind=error.name;process.exitCode=1;}finally{await context?.close();await browser?.close();fs.writeFileSync(path.join(out,'run-loader-dom-regression.json'),JSON.stringify({schema:'rheon-actual-run-loader-dom-regression-v1',status:kind?'failed':'passed',error_kind:kind,baseline,green,requests,errors,real_deployed_owners:['workspace-real.js','experiment-lineage.js'],metadata_only:true,scientific_results:false,all_requests_fulfilled_locally:true,original_endpoint_contacted:false,source_hashes:Object.fromEntries(['workspace-real.js','experiment-lineage.js'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(app,'static',name))).digest('hex')]))},null,2)+'\n',{flag:'wx'});}
})().catch(()=>{process.exitCode=1;});
