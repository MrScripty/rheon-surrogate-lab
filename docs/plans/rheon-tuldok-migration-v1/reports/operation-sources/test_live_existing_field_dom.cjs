/* Real deployed Workbench + embedded adapter DOM; all requests fulfilled locally.
   No sockets, owner cookie, original endpoint request, DB or scientific results. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {openExistingField}=require('./live_readback.cjs');
const app=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);
const digest='d'.repeat(64),id='7'.repeat(32),origin='http://127.0.0.1:8091';
const record={id,kind:'field_sequence',name:'Layout identity fixture',content_hash:digest,task:'field_sequence_fields',review:'draft',revision:1,source_revision:1,groups:['field-sequence-source:'+digest],rights_note:'unknown',annotation:{},provenance:{method:'fixture'}};
(async()=>{
 const {chromium}=require('/home/jeremy/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');let browser,context;const requests=[],errors=[];let verdict='failed',kind=null;
 try{
  browser=await chromium.launch({executablePath:'/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--disable-gpu']});context=await browser.newContext({serviceWorkers:'block'});
  await context.route('**/*',async route=>{
   const request=route.request(),url=new URL(request.url());requests.push({method:request.method(),path:url.pathname+url.search,fulfilled_locally:true});assert.equal(request.method(),'GET');
   if(url.pathname==='/fixture')return route.fulfill({contentType:'text/html',body:'<iframe id="catalog-frame" style="width:1400px;height:900px" src="/workbench?workspace=1"></iframe>'});
   if(url.pathname==='/workbench'){let html=fs.readFileSync(path.join(app,'static/workbench.html'),'utf8').replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'');html=html.replace('</body>','<script src="/workbench.js"></script><script src="/workspaces-data.js"></script></body>');return route.fulfill({contentType:'text/html',body:html});}
   if(url.pathname==='/api/workbench/records')return route.fulfill({json:{items:[record],total:1,offset:0,limit:40,analysis:{records:1,unlabeled:1,protected_groups:1,duplicate_content_records:0,unknown_rights:1,labels:{}}}});
   if(url.pathname==='/api/workbench/records/'+id)return route.fulfill({json:record});
   const file=path.join(app,'static',path.basename(url.pathname));if(fs.existsSync(file)&&['.js','.css'].includes(path.extname(file)))return route.fulfill({path:file});
   return route.fulfill({status:404,body:'Fixture resource absent'});
  });
  const page=await context.newPage();page.setDefaultTimeout(1000);page.on('pageerror',error=>errors.push(error.name));await page.goto(origin+'/fixture');const editor=page.frameLocator('#catalog-frame');
  await editor.getByText('Filter collection',{exact:true}).waitFor();assert.equal(await editor.locator('#kind').isVisible(),false);
  await openExistingField(editor,record,digest,predicate=>page.waitForResponse(predicate));
  assert.equal(await editor.locator('#kind').isVisible(),true);assert.equal(await editor.locator('#group-filter').isVisible(),true);assert(requests.some(row=>row.path.includes('group=field-sequence-source%3A'+digest)));assert.deepEqual(errors,[]);verdict='passed';
 }catch(error){kind=error.name;process.exitCode=1;}finally{await context?.close();await browser?.close();fs.writeFileSync(path.join(out,'dom-regression.json'),JSON.stringify({schema:'rheon-deployed-embedded-filter-dom-regression-v1',status:verdict,error_kind:kind,requests,errors,all_requests_fulfilled_locally:true,owner_authority_used:false,original_endpoint_contacted:false,fixture_scientific_results:false,source_hashes:Object.fromEntries(['workbench.js','workspaces-data.js','workbench.html'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(app,'static',name))).digest('hex')]))},null,2)+'\n',{flag:'wx'});}
 process.stdout.write('Embedded real DOM regression '+verdict+' '+(kind||'')+'\n');
})().catch(()=>{process.exitCode=1;});
