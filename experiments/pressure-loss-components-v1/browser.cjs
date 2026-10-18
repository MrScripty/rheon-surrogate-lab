'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]||'runs/pressure-loss-components-v1/browser');
if(!binary)throw Error('Pass Chromium executable');fs.mkdirSync(out,{recursive:true});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'loss-components-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-loss-components-v1/index.html'),checks=[],errors=[],pending=new Map();
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
 await until(()=>evaluate(`document.readyState==="complete"&&!!document.querySelector('a[href="pressure-loss-components-v1/index.html"]')`));
 await evaluate(`document.querySelector('a[href="pressure-loss-components-v1/index.html"]').click()`);
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#matrix rect").length===64'));
 checks.push('historical_dashboard_link_opens_component_report');
 for(const model of ['C53','C66','C67']){
  await change('model',model);
  for(const term of ['canonical','final','early','full','without_early']){
   await change('term',term);
   assert.equal(await evaluate('document.querySelectorAll("#matrix rect").length'),64);
   assert.equal(await evaluate(`(()=>{const g=selected().across[document.getElementById('term').value];return Array.from(document.querySelectorAll('#matrix rect title')).every((t,k)=>t.textContent.endsWith(g.cosine_matrix[Math.floor(k/8)][k%8].toFixed(6)))})()`),true);
   assert.equal(await evaluate('document.querySelectorAll("#values tbody tr").length'),8);
   assert.equal(await evaluate('document.querySelectorAll("#norms tbody tr").length'),8);
   await evaluate('document.querySelector("#matrix rect").focus()');
   assert.match(await evaluate('document.getElementById("detail").textContent'),new RegExp(model+' '+term));
  }
  assert.equal(await evaluate(`(()=>{const rows=Array.from(document.querySelectorAll('#norms tbody tr'));return rows.every((r,i)=>r.lastElementChild.textContent===(selected().cases[i].separate_gradient_precision_check_passed?'pass':'limited'))})()`),true);
  assert.equal(await evaluate(`(()=>{const rows=Array.from(document.querySelectorAll('#values tbody tr'));return rows.every((r,i)=>r.children[4].textContent===selected().cases[i].full_value.toExponential(3)&&r.lastElementChild.textContent==='yes')})()`),true);
 }
 checks.push('all_15_matrices_match_retained_cosines_and_keyboard_details');
 checks.push('loss_values_joint_equivalence_and_per_case_precision_limits_match_receipts');
 assert.equal(await evaluate('document.querySelectorAll("#summary tbody tr").length'),3);
 assert.equal(await evaluate('document.querySelectorAll("#components tbody tr").length'),15);
 assert.equal(await evaluate('document.querySelectorAll("#execution tbody tr").length'),3);
 assert.match(await evaluate('document.getElementById("lifecycle").textContent'),/14\/24/);
 await change('model','C66');await change('term','full');
 const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'screen.png'),Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:false});assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth+1'),false);checks.push('mobile_document_has_no_overflow');assert.deepEqual(errors,[]);
 receipt={status:'passed',models:['C53','C66','C67'],objectives:5,checks,console_errors:errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex'),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),command:['node',__filename,...process.argv.slice(2)]};
})().catch(e=>{console.error(e);receipt.error=String(e);fs.writeFileSync(path.join(out,'failure.log'),String(e)+'\n'+log);process.exitCode=1}).finally(async()=>{
 if(ws)ws.close();if(child.exitCode===null){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{if(child.exitCode===null)child.kill('SIGKILL')},3000).unref()})}
 receipt.browser_reaped=child.exitCode!==null||child.signalCode!==null;fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');fs.rmSync(temporary,{recursive:true,force:true});console.log(JSON.stringify(receipt));
});
