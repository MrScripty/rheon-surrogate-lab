'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]);
if(!binary)throw Error('Pass Chromium executable and fresh output directory');
fs.mkdirSync(out,{recursive:false});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'balanced-shapes-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run',
 '--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-balanced-shapes-v1/index.html');
const checks=[],errors=[],pending=new Map();let ws,sequence=0,receipt={status:'failed'},log='';
child.stderr.on('data',buffer=>log+=buffer);
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(fn){for(let i=0;i<150;i++){if(await fn())return;await pause(100)}throw Error('Browser deadline')}
(async()=>{
 let port;await until(()=>{try{port=fs.readFileSync(path.join(temporary,'DevToolsActivePort'),'utf8').split('\n')[0];return port}catch{return false}});
 const target=(await(await fetch('http://127.0.0.1:'+port+'/json')).json()).find(t=>t.type==='page');
 ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 const send=(method,params={})=>new Promise((resolve,reject)=>{
  const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP deadline'))},15000);
  pending.set(id,{resolve:value=>{clearTimeout(timer);resolve(value)},reject});ws.send(JSON.stringify({id,method,params}));
 });
 ws.onmessage=event=>{const message=JSON.parse(event.data);if(message.id){const request=pending.get(message.id);if(request){pending.delete(message.id);message.error?request.reject(Error(JSON.stringify(message.error))):request.resolve(message.result)}}if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails)};
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 const change=async(id,value)=>evaluate(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change'));return e.value})()`);
 await send('Runtime.enable');await send('Page.enable');
 await send('Page.navigate',{url:require('node:url').pathToFileURL(html).href});
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#learning circle").length===40'));
 assert.equal(await evaluate('document.querySelectorAll("#case option").length'),8);
 for(const passes of ['1','2','4','8']){
  await change('passes',passes);
  assert.equal(await evaluate(`(()=>{const n=+document.getElementById('passes').value;return Array.from(document.querySelectorAll('#cases tbody tr')).every((tr,i)=>['C53','C66','C69'].every((model,j)=>tr.cells[j+2].textContent===fitted[model].measurements.find(r=>r.case===i&&r.passes===n).relative_equation_residual.toExponential(3)))})()`),true);
  for(const chosen of ['0','7']){
   await change('case',chosen);
   assert.equal(await evaluate('document.querySelectorAll("#learning circle").length'),40);
   assert.equal(await evaluate(`(()=>{const n=+document.getElementById('passes').value,c=+document.getElementById('case').value;const expected=[data.control.training,training].flatMap(t=>t.fundamental_bank.trace.map(row=>t.id+': '+row.exposures_per_case+' exposures, '+row.measurements.find(v=>v.case===c&&v.passes===n).relative_equation_residual.toExponential(3)));return Array.from(document.querySelectorAll('#learning circle title')).every((node,i)=>node.textContent===expected[i])})()`),true);
  }
 }
 checks.push('all_pass_counts_case_tables_and_both_learning_curves_match_retained_scalars');
 assert.equal(await evaluate('document.querySelectorAll("#fitted tbody tr").length'),12);
 assert.equal(await evaluate('document.querySelectorAll("#regression tbody tr").length'),12);
 assert.equal(await evaluate('document.querySelectorAll("#accuracy circle").length'),8);
 assert.equal(await evaluate('Array.from(document.querySelectorAll("#training tbody tr")).map(r=>r.cells[2].textContent).join(",")'),'1600,200');
 assert.equal(await evaluate('document.getElementById("decision").textContent.includes(data.decision.capacity_gate_passed?"passes":"fails")'),true);
 await evaluate('document.querySelector("#learning circle").focus()');
 assert.match(await evaluate('document.getElementById("detail").textContent'),/exposures/);
 checks.push('timing_and_regression_tables_distinguish_example_and_update_work_and_keyboard_details');
 await change('case','0');await change('passes','4');
 const screenshot=await send('Page.captureScreenshot',{format:'png'});
 fs.writeFileSync(path.join(out,'screen.png'),Buffer.from(screenshot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:false});
 assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth+1'),false);
 assert.deepEqual(errors,[]);checks.push('narrow_layout_no_document_overflow_and_no_console_exceptions');
 receipt={status:'passed',checks,console_errors:errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex'),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex')};
})().catch(error=>{console.error(error);receipt.error=String(error);process.exitCode=1;fs.writeFileSync(path.join(out,'failure.log'),String(error)+'\n'+log)}).finally(async()=>{
 if(ws)ws.close();
 if(child.exitCode===null){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{if(child.exitCode===null)child.kill('SIGKILL')},3000).unref()})}
 receipt.browser_reaped=child.exitCode!==null||child.signalCode!==null;
 fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');
 fs.rmSync(temporary,{recursive:true,force:true});console.log(JSON.stringify(receipt));
});
