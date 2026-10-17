/* Owned read-only diagnostic, never acceptance or a write retry. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {validateFiles,openExistingField}=require('./live_readback.cjs');
(async()=>{
 const config=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));validateFiles(config);assert(!config.allow_project_import);
 const out=config.output,steps=[],requests=[];let browser,context,phase='launch';
 const receipt=()=>fs.writeFileSync(path.join(out,'diagnostic-progress.json'),JSON.stringify({schema:'rheon-readonly-field-selector-diagnosis-v1',phase,steps,requests},null,2)+'\n',{mode:0o600});
 const mark=(name,details={})=>{phase=name;steps.push({phase:name,...details});receipt();};
 try{
  const {chromium}=require('/home/jeremy/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');browser=await chromium.launch({executablePath:config.chromium,headless:true,args:['--disable-gpu']});context=await browser.newContext({viewport:{width:1440,height:1000},serviceWorkers:'block'});
  const token=fs.readFileSync(config.owner_session_file,'ascii').trim();await context.addCookies([{name:'tuldok_session',value:token,url:config.endpoint,httpOnly:true,sameSite:'Strict'}]);
  await context.route('**/*',async route=>{const req=route.request(),url=new URL(req.url());if(url.origin!==config.endpoint||!['GET','HEAD'].includes(req.method())){mark('forbidden request blocked',{method:req.method(),path:url.pathname});await route.abort();return;}await route.continue();});
  const page=await context.newPage();page.setDefaultTimeout(2000);page.on('response',response=>{const url=new URL(response.url());if(url.pathname==='/api/workbench/records')requests.push({method:response.request().method(),path:url.pathname+url.search,status:response.status()});});
  mark('navigate catalog');await page.goto(config.endpoint+'/workspaces#data');await page.locator('#dataset').selectOption('catalog');await page.locator('#nav-data').click();const editor=page.frameLocator('#catalog-frame');await editor.locator('#filters').waitFor({state:'attached',timeout:5000});
  const inspect=async()=>editor.locator('#filters').evaluate(form=>({frame_path:location.pathname+location.search,controls:['kind','query','group-filter'].map(id=>{const node=document.getElementById(id),details=[];for(let p=node?.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')details.push({summary:p.querySelector(':scope > summary')?.textContent,open:p.open});return {id,count:document.querySelectorAll('#'+id).length,visible:!!node?.getClientRects().length,details};})}));
  mark('actual DOM before baseline',await inspect());await page.screenshot({path:path.join(out,'filters-closed.png'),fullPage:true});
  const record={kind:'field_sequence',content_hash:config.import_sha256,id:config.existing_record};let baseline;
  try{await openExistingField(editor,record,config.import_sha256,predicate=>page.waitForResponse(predicate));baseline='unexpected_pass';}catch(error){baseline=error.name;}
  mark('baseline existing helper',{verdict:baseline});assert.equal(baseline,'TimeoutError');
  mark('single visible disclosure click');await editor.getByText('Filter collection',{exact:true}).click();mark('actual DOM after disclosure',await inspect());
  const lookup=await openExistingField(editor,record,config.import_sha256,predicate=>page.waitForResponse(predicate));mark('same helper after one disclosure change',{verdict:'passed',lookup});await editor.locator('#scientific-canvas').waitFor({state:'visible',timeout:5000});await page.screenshot({path:path.join(out,'existing-field-visible.png'),fullPage:true});
  fs.writeFileSync(path.join(out,'browser-receipt.json'),JSON.stringify({schema:'rheon-readonly-field-selector-diagnosis-v1',status:'cause_confirmed',cause:'Embedded Filter collection disclosure starts closed; existing helper omitted normal visible summary click',baseline,after_single_disclosure:'passed',lookup,steps,requests,collection_writes:0,new_runs:0,scientific_acceptance:'not_asserted'},null,2)+'\n',{flag:'wx',mode:0o600});
 }catch(error){mark('diagnostic stopped',{error_kind:error.name});process.exitCode=1;}finally{await context?.close();await browser?.close();}
})().catch(()=>{process.stderr.write('Read-only diagnostic gate failure\n');process.exitCode=1;});
