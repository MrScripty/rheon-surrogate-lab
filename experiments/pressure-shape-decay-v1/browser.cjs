'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]||'runs/pressure-shape-decay-v1/browser');
if(!binary)throw Error('Pass Chromium executable');fs.mkdirSync(out,{recursive:true});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'shape-decay-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-shape-decay-v1/index.html'),checks=[],errors=[],pending=new Map();
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
 await until(()=>evaluate(`document.readyState==="complete"&&!!document.querySelector('a[href="pressure-shape-decay-v1/index.html"]')`));
 await evaluate(`document.querySelector('a[href="pressure-shape-decay-v1/index.html"]').click()`);
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#gradient-map rect").length===64'));
 checks.push('historical_dashboard_link_opens_schedule_report');
 const dimensions=await evaluate(`(()=>{const d=JSON.parse(document.getElementById('study-data').textContent);return {arms:Object.keys(d.arms),training:Object.keys(d.arms).length}})()`);
 for(const checkpoint of ['C53','C66']){
  await change('gradient',checkpoint);
  assert.equal(await evaluate('document.querySelectorAll("#gradient-map rect").length'),64);
  assert.equal(await evaluate(`(()=>{const g=data.gradients[document.getElementById('gradient').value];return Array.from(document.querySelectorAll('#gradient-map rect title')).every((t,k)=>t.textContent.endsWith(g.cosine_matrix[Math.floor(k/8)][k%8].toFixed(6)))})()`),true);
  assert.equal(await evaluate('document.querySelectorAll("#norms tbody tr").length'),8);
  await evaluate('document.querySelector("#gradient-map rect").focus()');
  assert.match(await evaluate('document.getElementById("gradient-detail").textContent'),new RegExp(checkpoint));
 }
 checks.push('both_64_cell_gradient_matrices_match_retained_scalars_and_keyboard_details');
 for(const arm of dimensions.arms){
  await change('arm',arm);
  const cases=await evaluate('selected().training.fundamental_bank.cases.length');
  assert.equal(await evaluate('document.querySelectorAll("#case option").length'),cases);
  for(const passes of ['1','2','4','8']){
   await change('passes',passes);
   assert.equal(await evaluate('document.querySelectorAll("#learning circle").length'),40);
   assert.equal(await evaluate('document.querySelectorAll("#learning polyline").length'),2);
   assert.equal(await evaluate('document.querySelectorAll("#heatmap rect").length'),cases);
   assert.equal(await evaluate(`(()=>{const a=selected(),id=a.training.id,n=+document.getElementById('passes').value;return Array.from(document.querySelectorAll('#heatmap rect title')).every((t,i)=>t.textContent.includes(a.fitted[id].measurements.find(r=>r.case===i&&r.passes===n).relative_equation_residual.toExponential(3)))})()`),true);
  }
  await change('case',String(cases-1));
  assert.equal(await evaluate('document.querySelectorAll("#learning circle").length'),40);
  assert.equal(await evaluate('document.querySelectorAll("#fitted tbody tr").length'),12);
  assert.equal(await evaluate('document.querySelectorAll("#regression tbody tr").length'),12);
  assert.equal(await evaluate('document.querySelectorAll("#accuracy circle").length'),8);
  assert.equal(await evaluate('document.querySelectorAll("#accuracy polyline").length'),2);
 }
 checks.push('all_completed_arms_pass_counts_learning_curves_and_regression_summaries');
 assert.equal(await evaluate('document.querySelectorAll("#training tbody tr").length'),dimensions.training);
 if(dimensions.arms.length){await change('arm',dimensions.arms[0]);await change('passes','4')}
 const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'screen.png'),Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:false});assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth+1'),false);checks.push('mobile_document_has_no_overflow');assert.deepEqual(errors,[]);
 receipt={status:'passed',completed_arms:dimensions.arms,checks,console_errors:errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex'),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),command:['node',__filename,...process.argv.slice(2)]};
})().catch(e=>{console.error(e);receipt.error=String(e);fs.writeFileSync(path.join(out,'failure.log'),String(e)+'\n'+log);process.exitCode=1}).finally(async()=>{
 if(ws)ws.close();if(child.exitCode===null){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{if(child.exitCode===null)child.kill('SIGKILL')},3000).unref()})}
 receipt.browser_reaped=child.exitCode!==null||child.signalCode!==null;fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');fs.rmSync(temporary,{recursive:true,force:true});console.log(JSON.stringify(receipt));
});
