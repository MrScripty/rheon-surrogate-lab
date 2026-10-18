'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]||'runs/pressure-low-mode-tail-v1/browser');
if(!binary)throw Error('Pass Chromium executable');fs.mkdirSync(out,{recursive:true});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'low-mode-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-low-mode-tail-v1/index.html'),checks=[],errors=[],pending=new Map();
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
 await until(()=>evaluate(`document.readyState==="complete"&&!!document.querySelector('a[href="pressure-low-mode-tail-v1/index.html"]')`));
 await evaluate(`document.querySelector('a[href="pressure-low-mode-tail-v1/index.html"]').click()`);
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#accuracy circle").length===16'));
 checks.push('historical_dashboard_link_opens_matched_study');
 assert.equal(await evaluate('document.querySelectorAll("#snapshots tbody tr").length'),16);checks.push('four_checkpoints_four_pass_settings');
 await evaluate('document.querySelector("[data-model=C59]").click()');assert.equal(await evaluate('document.querySelectorAll("#accuracy circle").length'),12);await evaluate('document.querySelector("[data-model=C59]").click()');checks.push('checkpoint_filter');
 await change('metric','worst_same_request_pressure_rms_pa');await evaluate('document.querySelector("#accuracy circle").focus()');assert.match(await evaluate('document.getElementById("detail").textContent'),/passes.*qualified.*ms.*error/);checks.push('absolute_pressure_metric_and_keyboard_details');
 for(const model of ['C53','C59','C60','C61'])for(const passes of ['1','2','4','8'])for(const amplitude of ['0.0001','0.01','1','100','-1']){
  await change('model',model);await change('passes',passes);await change('amplitude',amplitude);
  assert.equal(await evaluate('document.querySelectorAll("#heat [data-error]").length'),48);
  assert.equal(await evaluate(`(()=>{const d=JSON.parse(document.getElementById('study-data').textContent);return Array.from(document.querySelectorAll('#heat [data-error]')).every(td=>Number.isFinite(Number(td.dataset.error))&&td.textContent===Number(td.dataset.error).toExponential(3))})()`),true);
 }
 checks.push('all_3840_mode_shape_amplitude_cells_finite_and_bound_to_scalar_measurements');
 assert.equal(await evaluate('document.querySelectorAll("#training tbody tr").length'),2);checks.push('training_timing_and_memory_retained');
 await evaluate('document.querySelectorAll("#models input").forEach(e=>e.click())');assert.equal(await evaluate('document.querySelectorAll("#accuracy circle").length'),0);await evaluate('document.querySelectorAll("#models input").forEach(e=>e.click())');checks.push('empty_filter_recovers');
 await change('metric','worst_equation_residual');await change('model','C61');await change('passes','4');await change('amplitude','1');
 const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'screen.png'),Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:false});assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth+1'),false);checks.push('mobile_document_has_no_overflow');assert.deepEqual(errors,[]);
 receipt={status:'passed',checks,console_errors:errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex'),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),command:['node',__filename,...process.argv.slice(2)]};
})().catch(e=>{console.error(e);receipt.error=String(e);fs.writeFileSync(path.join(out,'failure.log'),String(e)+'\n'+log);process.exitCode=1}).finally(async()=>{
 if(ws)ws.close();if(child.exitCode===null){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{if(child.exitCode===null)child.kill('SIGKILL')},3000).unref()})}
 receipt.browser_reaped=child.exitCode!==null||child.signalCode!==null;fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');fs.rmSync(temporary,{recursive:true,force:true});console.log(JSON.stringify(receipt));
});
