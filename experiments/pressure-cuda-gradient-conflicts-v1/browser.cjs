'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawn}=require('node:child_process'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const binary=process.argv[2],out=path.resolve(process.argv[3]);
if(!binary)throw Error('Pass Chromium executable and fresh output directory');
fs.mkdirSync(out,{recursive:false});
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'cuda-gradient-view-'));
const child=spawn(binary,['--headless','--no-sandbox','--disable-gpu','--no-first-run',
 '--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+temporary,'about:blank'],{stdio:['ignore','ignore','pipe']});
const html=path.resolve('docs/plans/pressure-cuda-gradient-conflicts-v1/index.html');
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
 await until(()=>evaluate('document.readyState==="complete"&&document.querySelectorAll("#chart rect").length===40'));
 assert.equal(await evaluate('document.querySelectorAll("#update option").length'),3);
 for(const update of ['100','150','200']){
  await change('update',update);
  for(const anchor of ['vs_weak_final','vs_weak_full']){
   await change('anchor',anchor);
   assert.equal(await evaluate(`(()=>{const s=d.snapshots.find(s=>s.update===Number(document.getElementById('update').value)),anchor=document.getElementById('anchor').value,terms=['1','2','3','4','full'],expected=s.cases.flatMap((c,i)=>terms.map(term=>{const r=s.rows.find(r=>r.case===i&&r.term===term);return 'case '+i+' term '+term+' cosine '+Number(r[anchor].cosine).toFixed(4)+' gradient norm '+f(r.gradient_l2)}));return Array.from(document.querySelectorAll('#chart rect title')).every((t,i)=>t.textContent===expected[i])})()`),true);
  }
  assert.equal(await evaluate('document.querySelectorAll("#short tr").length'),6);
  for(const index of ['0','1','4']){
   await change('layercase',index);
   for(const passes of ['1','2']){
    await change('layerpass',passes);
    assert.equal(await evaluate(`(()=>{const s=d.snapshots.find(s=>s.update===Number(document.getElementById('update').value)),pair=s.layer_comparisons.find(p=>p.case===Number(document.getElementById('layercase').value)&&p.passes===Number(document.getElementById('layerpass').value));return Array.from(document.querySelectorAll('#layers tr')).every((tr,i)=>tr.cells[0].textContent===pair.layers[i].name&&tr.cells[1].textContent===f(pair.layers[i].dot))})()`),true);
   }
  }
 }
 checks.push('all_three_snapshots_eight_cases_five_terms_two_anchors_and_all_layer_pairs_match_scalar_records');
 assert.equal(await evaluate('document.getElementById("summary").textContent.includes("18/18")'),true);
 assert.equal(await evaluate('document.querySelectorAll("#replay tr").length'),3);
 assert.match(await evaluate('document.getElementById("limits").textContent'),/first-order descent conflict/);
 await change('update','150');await change('anchor','vs_weak_final');await change('layercase','1');await change('layerpass','2');
 await evaluate('document.querySelector("#chart rect").focus()');
 assert.equal(await evaluate('document.activeElement.hasAttribute("aria-label")'),true);
 checks.push('opposing_gradient_summary_replay_precision_limits_and_keyboard_labels_visible');
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
