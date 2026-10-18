'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]);
if(!binary)throw Error('Pass Chromium executable and fresh output directory');
fs.mkdirSync(out,{recursive:false});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'adam-momentum-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run',
 '--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-adam-momentum-damping-v1/index.html');
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
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#chart circle").length===5'));
 assert.equal(await evaluate('document.querySelectorAll("#case option").length'),8);
 assert.equal(await evaluate('document.querySelectorAll("#update option").length'),5);
 for(const update of ['1','10','100','150','200']){
  await change('update',update);
  for(const metric of ['pressure','loss']){
   await change('metric',metric);
   for(let index=0;index<8;index++){
    await change('case',String(index));
    assert.equal(await evaluate('document.querySelectorAll("#chart circle").length'),5);
    assert.equal(await evaluate(`(()=>{const s=d.snapshots.find(s=>s.update===Number(document.getElementById('update').value)),index=Number(document.getElementById('case').value),metric=document.getElementById('metric').value;const expected=s.rows.filter(r=>r.policy==='damped').map(r=>{const m=s.measurements[r.parameter_sha256].metrics,v=metric==='loss'?m.losses[index]:m.pressure.find(p=>p.case===index&&p.passes===4).relative_equation_residual;return 'retained '+r.factor+': '+v.toExponential(4)});return Array.from(document.querySelectorAll('#chart circle title')).every((node,j)=>node.textContent===expected[j])})()`),true);
   }
  }
  assert.equal(await evaluate(`(()=>{const s=d.snapshots.find(s=>s.update===Number(document.getElementById('update').value));return Array.from(document.querySelectorAll('#rows tr')).every((tr,j)=>tr.cells[5].textContent===s.rows[j].weak_four_pass_residual.toExponential(4)&&tr.cells[7].textContent===String(s.rows[j].short_pass_regressions.length)&&tr.cells[9].textContent===(s.rows[j].guard_passed?'pass':'fail'))})()`),true);
 }
 checks.push('all_five_snapshots_eight_cases_and_both_metrics_match_retained_values_and_guards');
 assert.equal(await evaluate('document.querySelectorAll("#runtime tr").length'),5);
 assert.match(await evaluate('document.getElementById("limits").textContent'),/not rigorous numerical bounds/);
 assert.equal(await evaluate('document.getElementById("summary").textContent.includes(d.decision.guard_passing_candidates.length+" damping entries pass")'),true);
 await change('update','100');await change('case','7');await change('metric','pressure');
 await evaluate('document.querySelector("#chart circle").focus()');
 assert.equal(await evaluate('document.activeElement.hasAttribute("aria-label")'),true);
 checks.push('failed_guards_empirical_value_limits_runtime_and_keyboard_labels_visible');
 await change('scale','absolute');assert.equal(await evaluate('document.getElementById("chart").textContent.includes("target 1e-6")'),true);
 await change('scale','relative');checks.push('percentage_and_absolute_axes_preserve_measured_values_and_target');
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
