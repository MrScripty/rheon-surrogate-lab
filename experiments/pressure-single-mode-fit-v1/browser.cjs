'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]||'runs/pressure-single-mode-fit-v1/browser');
if(!binary)throw Error('Pass Chromium executable');fs.mkdirSync(out,{recursive:true});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'single-mode-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-single-mode-fit-v1/index.html'),checks=[],errors=[],pending=new Map();
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
 await until(()=>evaluate(`document.readyState==="complete"&&!!document.querySelector('a[href="pressure-single-mode-fit-v1/index.html"]')`));
 await evaluate(`document.querySelector('a[href="pressure-single-mode-fit-v1/index.html"]').click()`);
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#accuracy circle").length>=8'));
 checks.push('historical_dashboard_link_opens_capacity_report');
 const dimensions=await evaluate(`(()=>{const d=JSON.parse(document.getElementById('study-data').textContent);return {snapshots:Object.keys(d.snapshots).length,fitted:Object.keys(d.fitted).length,diagnostics:Object.keys(d.diagnostics).length,training:d.training.length}})()`);
 assert.equal(await evaluate('document.querySelectorAll("#measurements tbody tr").length'),dimensions.fitted*4);
 checks.push('retained_checkpoints_four_cpu_pass_settings');
 for(const passes of ['1','2','4','8']){
  await change('passes',passes);
  assert.equal(await evaluate('document.querySelectorAll("#learning circle").length'),21);
 }
 checks.push('all_four_learning_curves_include_baseline_and_twenty_observations');
 for(const population of ['fitted','regression']){
  await change('population',population);
  assert.equal(await evaluate('document.querySelectorAll("#accuracy circle").length'),dimensions[population==='fitted'?'fitted':'snapshots']*4);
  await evaluate('document.querySelector("#accuracy circle").focus()');
  assert.match(await evaluate('document.getElementById("accuracy-detail").textContent'),/passes.*ms.*residual/);
  assert.equal(await evaluate(`(()=>{const d=JSON.parse(document.getElementById('study-data').textContent);return Array.from(document.querySelectorAll('#measurements tbody tr')).every(tr=>{const v=Array.from(tr.cells).map(c=>c.textContent),m=population==='fitted'?d.fitted[v[0]].measurements:d.snapshots[v[0]].summary;const r=m.find(r=>r.passes===Number(v[1]));return v[2]===(population==='fitted'?r.relative_equation_residual:r.worst_equation_residual).toExponential(4)})})()` .replaceAll('population===',JSON.stringify(population)+'===')),true);
 }
 checks.push('both_populations_bound_to_retained_scalars_and_keyboard_details');
 assert.equal(await evaluate('document.querySelectorAll("#regression tbody tr").length'),dimensions.diagnostics*2);
 assert.equal(await evaluate('document.querySelectorAll("#training tbody tr").length'),dimensions.training);
 checks.push('manufactured_guards_timing_and_memory_present');
 if(dimensions.training===2){assert.equal(await evaluate('document.querySelectorAll("#modes tbody tr").length'),12);checks.push('matched_control_per_mode_comparisons_present')}
 assert.equal(await evaluate('document.querySelectorAll("#accuracy polyline").length'),dimensions.snapshots);checks.push('fixed_pass_performance_connected_per_checkpoint');
 await change('passes','4');await change('population','regression');
 const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'screen.png'),Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:false});assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth+1'),false);checks.push('mobile_document_has_no_overflow');assert.deepEqual(errors,[]);
 receipt={status:'passed',checks,console_errors:errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex'),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),command:['node',__filename,...process.argv.slice(2)]};
})().catch(e=>{console.error(e);receipt.error=String(e);fs.writeFileSync(path.join(out,'failure.log'),String(e)+'\n'+log);process.exitCode=1}).finally(async()=>{
 if(ws)ws.close();if(child.exitCode===null){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{if(child.exitCode===null)child.kill('SIGKILL')},3000).unref()})}
 receipt.browser_reaped=child.exitCode!==null||child.signalCode!==null;fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');fs.rmSync(temporary,{recursive:true,force:true});console.log(JSON.stringify(receipt));
});
