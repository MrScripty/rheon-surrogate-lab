'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]||'runs/pressure-fundamental-bank-fit-v1/browser');
if(!binary)throw Error('Pass Chromium executable');fs.mkdirSync(out,{recursive:true});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'fundamental-bank-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-fundamental-bank-fit-v1/index.html'),checks=[],errors=[],pending=new Map();
let ws,sequence=0,receipt={status:'failed'},log='';child.stderr.on('data',b=>log+=b);
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn){for(let i=0;i<200;i++){const v=await fn();if(v)return v;await pause(100)}throw Error('Browser deadline')}
(async()=>{
 const port=await until(()=>{try{return fs.readFileSync(path.join(temporary,'DevToolsActivePort'),'utf8').split('\n')[0]}catch{return null}});
 const target=(await(await fetch('http://127.0.0.1:'+port+'/json')).json()).find(t=>t.type==='page');
 ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP deadline'))},30000);pending.set(id,{resolve:v=>{clearTimeout(timer);resolve(v)},reject});ws.send(JSON.stringify({id,method,params}))});
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails)};
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 const change=async(id,value)=>evaluate(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));return e.value})()`);
 await send('Runtime.enable');await send('Page.enable');await send('Page.navigate',{url:require('node:url').pathToFileURL(path.resolve('docs/plans/cohort-progress.html')).href});
 await until(()=>evaluate(`document.readyState==="complete"&&!!document.querySelector('a[href="pressure-fundamental-bank-fit-v1/index.html"]')`));
 await evaluate(`document.querySelector('a[href="pressure-fundamental-bank-fit-v1/index.html"]').click()`);
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#accuracy circle").length>=8'));
 checks.push('historical_dashboard_link_opens_capacity_report');
 const dimensions=await evaluate(`(()=>{const d=JSON.parse(document.getElementById('study-data').textContent);return {arms:Object.keys(d.arms),training:Object.keys(d.arms).length}})()`);
 for(const arm of dimensions.arms){
  await change('arm',arm);
  const cases=await evaluate('selected().training[0].fundamental_bank.cases.length');
  assert.equal(await evaluate('document.querySelectorAll("#case option").length'),cases);
  for(const passes of ['1','2','4','8']){
   await change('passes',passes);
   assert.equal(await evaluate('document.querySelectorAll("#learning circle").length'),21);
   assert.equal(await evaluate('document.querySelectorAll("#heatmap rect").length'),cases);
   assert.equal(await evaluate(`(()=>{const a=selected(),id=a.training[0].id,n=+document.getElementById('passes').value;return Array.from(document.querySelectorAll('#heatmap rect title')).every((t,i)=>t.textContent.includes(a.fitted[id].measurements.find(r=>r.case===i&&r.passes===n).relative_equation_residual.toExponential(3)))})()`),true);
  }
  await change('case',String(cases-1));
  assert.equal(await evaluate('document.querySelectorAll("#learning circle").length'),21);
  assert.equal(await evaluate('document.querySelectorAll("#fitted tbody tr").length'),8);
  assert.equal(await evaluate('document.querySelectorAll("#regression tbody tr").length'),8);
  assert.equal(await evaluate('document.querySelectorAll("#accuracy circle").length'),8);
  assert.equal(await evaluate('document.querySelectorAll("#accuracy polyline").length'),2);
  await evaluate('document.querySelector("#heatmap rect").focus()');
  assert.match(await evaluate('document.getElementById("detail").textContent'),/C53/);
 }
 checks.push('all_arms_all_passes_all_case_counts_and_retained_heatmap_scalars');
 checks.push('learning_curves_include_baseline_and_twenty_observations');
 checks.push('fitted_real_manufactured_and_connected_accuracy_time_results');
 checks.push('keyboard_accessible_case_comparison_details');
 assert.equal(await evaluate('document.querySelectorAll("#training tbody tr").length'),dimensions.training);
 checks.push('execution_time_and_memory_for_each_arm');
 await change('arm',dimensions.arms[0]);await change('passes','4');
 const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'screen.png'),Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:false});assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth+1'),false);checks.push('mobile_document_has_no_overflow');assert.deepEqual(errors,[]);
 receipt={status:'passed',checks,console_errors:errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex'),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),command:['node',__filename,...process.argv.slice(2)]};
})().catch(e=>{console.error(e);receipt.error=String(e);fs.writeFileSync(path.join(out,'failure.log'),String(e)+'\n'+log);process.exitCode=1}).finally(async()=>{
 if(ws)ws.close();if(child.exitCode===null){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{if(child.exitCode===null)child.kill('SIGKILL')},3000).unref()})}
 receipt.browser_reaped=child.exitCode!==null||child.signalCode!==null;fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');fs.rmSync(temporary,{recursive:true,force:true});console.log(JSON.stringify(receipt));
});
