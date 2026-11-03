'use strict';
// Actual native graph DOM/lifecycle against controlled transport. No service,
// training, real history or scientific qualification is represented here.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),crypto=require('node:crypto');
const {spawn,execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),source=path.join(root,'runs/model-portfolio-migration-v1/sources/Tuldok-progression');
const output=path.resolve(process.argv[2]);
if(fs.existsSync(output))throw Error('Fresh DOM fixture output required');
fs.mkdirSync(output,{recursive:true,mode:0o700});
const {waitForDebugger}=require(path.join(source,'tests/browser_startup.cjs'));
const {cdpSession}=require(path.join(source,'tests/browser_cdp_session.cjs'));
const engine='/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell';
const report={status:'running',scope:'Actual Chromium DOM/CSS and native controller; synthetic transport only',neural_forwards:0,live_catalog_accessed:false,checks:[],errors:[]};
let server,browser,session;
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const check=(value,label)=>{if(!value)throw Error(label);report.checks.push(label);};
async function main(){
  try{
    const files=['static/experiment-graph.js','static/project-progression.js','static/experiment-lineage.css'];
    report.source_sha256=Object.fromEntries(files.map(n=>[n,crypto.createHash('sha256').update(fs.readFileSync(path.join(source,n))).digest('hex')]));
    report.runner_sha256=crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
    server=http.createServer((req,res)=>{
      if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<main><section id="graph" class="lineage-panel"></section></main><script src="/experiment-graph.js"></script><script src="/project-progression.js"></script><link rel="stylesheet" href="/experiment-lineage.css">');return;}
      const name=files.find(n=>'/'+path.basename(n)===req.url);
      if(!name){res.statusCode=404;res.end();return;}
      res.setHeader('Content-Type',name.endsWith('.css')?'text/css':'text/javascript');res.end(fs.readFileSync(path.join(source,name)));
    });
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    const log=fs.openSync(path.join(output,'browser.log'),'w');
    browser=spawn(engine,['--headless','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+path.join(output,'profile'),'about:blank'],{stdio:['ignore',log,log]});fs.closeSync(log);
    const port=await waitForDebugger(browser,path.join(output,'profile/DevToolsActivePort'));
    const tabs=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json(),socket=new WebSocket(tabs.find(v=>v.type==='page').webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
    session=cdpSession(socket,{onEvent:event=>{if(event.method==='Runtime.exceptionThrown')report.errors.push(event.params.exceptionDetails.text);}});
    const send=(...args)=>session.send(...args);
    async function ev(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description??r.exceptionDetails.text);return r.result.value;}
    async function until(expression){const end=Date.now()+15000;while(Date.now()<end){if(await ev(expression))return;await pause(30);}throw Error('DOM fixture wait expired: '+expression);}
    async function press(key,code=key,number=key==='Enter'?13:9){await send('Input.dispatchKeyEvent',{type:'keyDown',key,code,windowsVirtualKeyCode:number,...(key==='Enter'?{text:'\r'}:{})});await send('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:number});}
    async function activate(selector){await ev(`document.querySelector(${JSON.stringify(selector)}).focus()`);await press('Enter');}
    await send('Runtime.enable');await send('Page.enable');await send('Page.navigate',{url:'http://127.0.0.1:'+server.address().port+'/'});
    await until("typeof window.ProjectExperimentProgression==='function'");
    await ev(`
      window.original='a'.repeat(32);window.project='b'.repeat(32);window.calls=[];window.opened=[];window.mode='normal';
      window.node=i=>({trial:{format:'trial_v1',id:i.toString(16).padStart(32,'0'),study_id:'3'.repeat(32),material:{kind:'recipe_material',catalog_id:original,id:'c'.repeat(64),sha256:'c'.repeat(64)},replicate:0},study:{id:'3'.repeat(32),name:'Original Study <img src=x onerror=bad>'},lineage:null,attempt_count:1,runs:[{id:'4'.repeat(32),state:'completed',parent_run_id:null,created_at:'2026-10-23',finished_at:'2026-10-23'}],runs_truncated:false});
      window.response=args=>{const count=args.cohort?21:2;return {origin_catalog_id:original,evidence:'saved_metadata_only',historical:true,execution_available:false,nodes:Array.from({length:Math.min(20,Math.max(0,count-args.offset))},(_,i)=>node(args.offset+i+1)),related:[],offset:args.offset,limit:20,total:count,cohort:args.cohort??null};};
      window.ui=ProjectExperimentProgression({root:document.querySelector('#graph'),read:async(action,args,signal)=>{calls.push({action,args:structuredClone(args)});if(mode==='forbidden')throw Error('Current inspection was revoked');if(action==='progression_origins')return {items:[{origin_catalog_id:original,study_count:1,trial_count:2}],offset:args.offset,limit:40,total:1};if(mode==='held')return new Promise(resolve=>window.release=()=>resolve(response(args)));if(mode==='wrong')return {...response(args),origin_catalog_id:project};return response(args);},inspectStudy:(origin,id)=>opened.push({kind:'study',origin,id}),inspectTrial:(origin,id)=>opened.push({kind:'trial',origin,id}),inspectRun:(origin,id)=>opened.push({kind:'run',origin,id})});
      ui.setContext({key:'one',project_id:project,enabled:true});
    `);
    async function sources(){await activate('#project-progression-origins');await until("!document.querySelector('#project-progression-origin').disabled");await ev("document.querySelector('#project-progression-origin').value=original;document.querySelector('#project-progression-origin').dispatchEvent(new Event('change',{bubbles:true}))");}
    async function load(){await activate('#project-progression-load');await until("document.querySelector('#graph').getAttribute('aria-busy')==='false'");}
    await sources();await load();
    check(await ev("document.querySelectorAll('[data-lineage-node]').length===2 && !document.querySelector('[data-lineage-node]').disabled && document.querySelectorAll('path').length===0 && document.querySelectorAll('img').length===0"),'Original unknown-ancestry nodes remain separate, enabled and escaped');
    await ev("document.querySelector('#project-progression-zoom-reset').focus()");await press('Tab');await press('Tab');await press('Tab');
    check(await ev("document.activeElement.dataset.lineageNode===original+'.'+node(2).trial.id && document.activeElement.matches(':focus-visible') && parseFloat(getComputedStyle(document.activeElement).outlineWidth)>=3"),'Native graph is reachable with visible keyboard focus');
    await press('Enter');
    check(await ev("document.activeElement.getAttribute('aria-pressed')==='true' && document.querySelector('#project-progression-detail').innerText.includes('Historical ancestry and intent are unknown')"),'Native selection retains focus and reports unknown ancestry');
    await activate('#project-progression-detail button:last-of-type');
    check(await ev("opened.at(-1).kind==='run' && opened.at(-1).origin===original && opened.at(-1).id==='4'.repeat(32)"),'Run navigation retains the original namespace and identity');
    const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(output,'native-progression.png'),Buffer.from(shot.data,'base64'));
    await ev("document.querySelector('#project-progression-cohort').value='scope-a'");await load();
    await ev("document.querySelector('#project-progression-cohort').value='draft-b'");await activate('#project-progression-more');await until("document.querySelector('#graph').getAttribute('aria-busy')==='false'");
    check(await ev("document.querySelectorAll('[data-lineage-node]').length===21 && calls.at(-1).args.cohort==='scope-a' && calls.at(-1).args.offset===20"),'Pagination keeps the applied cohort rather than an unsubmitted draft');
    await ev("mode='held'");await activate('#project-progression-load');await until("typeof release==='function'");
    await ev("ui.setContext({key:'two',project_id:project,enabled:true});release()");await pause(50);
    check(await ev("document.querySelectorAll('[data-lineage-node]').length===0 && document.querySelector('#project-progression-origin').value===''"),'A late read after a host authority change cannot repopulate history');
    await ev("mode='normal'");await sources();await load();await ev("mode='wrong'");await load();
    check(await ev("document.querySelectorAll('[data-lineage-node]').length===0 && document.querySelector('#project-progression-status').innerText.includes('selected source')"),'A mismatched origin clears old graph state and reports refusal');
    await ev("mode='normal'");await sources();await load();await ev("mode='forbidden'");await load();
    check(await ev("document.querySelectorAll('[data-lineage-node]').length===0 && document.querySelector('#project-progression-status').innerText.includes('revoked') && document.activeElement.id==='project-progression-origins'"),'Failed inspection clears the graph and restores a useful focus target');
    check(report.errors.length===0,'No JavaScript exceptions');report.status='passed_native_progression_controlled_DOM';
    report.browser_version=execFileSync(engine,['--version'],{encoding:'utf8'}).trim();
  }catch(error){report.status='failed_native_progression_controlled_DOM';report.error=String(error);throw error;}
  finally{
    session?.close();if(browser&&browser.exitCode===null){browser.kill('SIGTERM');for(let i=0;i<50&&browser.exitCode===null;i++)await pause(100);if(browser.exitCode===null)browser.kill('SIGKILL');}
    if(server)await new Promise(resolve=>server.close(resolve));report.browser_stopped=!browser||browser.exitCode!==null||browser.signalCode!==null;
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  }
  console.log(JSON.stringify({status:report.status,checks:report.checks.length,browser_stopped:report.browser_stopped}));
}
main().catch(error=>{console.error(String(error));process.exitCode=1;});
