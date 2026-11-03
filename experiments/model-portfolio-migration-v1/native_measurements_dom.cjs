'use strict';
// Real Chromium, renderer, CSS and controller with codec-authored controlled
// replies. This fixture does not qualify a native service or scientific result.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),crypto=require('node:crypto');
const {spawn,execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),source=path.join(root,'runs/model-portfolio-migration-v1/sources/Tuldok-measurements');
const input=path.resolve(process.argv[2]),output=path.resolve(process.argv[3]);
if(fs.existsSync(output))throw Error('Fresh controlled DOM output required');
fs.mkdirSync(output,{recursive:true,mode:0o700});
const {waitForDebugger}=require(path.join(source,'tests/browser_startup.cjs'));
const {cdpSession}=require(path.join(source,'tests/browser_cdp_session.cjs'));
const engine='/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell';
const report={status:'running',scope:'Actual Chromium DOM/CSS/renderer; codec-authored synthetic scalar transport',checks:[],errors:[],external:[],new_neural_forwards:0};
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const check=(value,label)=>{if(!value)throw Error(label);report.checks.push(label);};
let browser,server,session;
async function main(){
  try{
    const files=['static/project-measurements.js','static/project-measurements.css','static/project-files.css','static/retained-chart.js','static/retained-measurement-maxima.vl.json',
      ...['vega','vega-lite','vega-embed','vega-interpreter'].map(n=>'static/vendor/retained-analysis/'+n+'.min.js')];
    report.source_sha256=Object.fromEntries(files.map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(source,name))).digest('hex')]));
    report.runner_sha256=crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
    report.input_sha256=crypto.createHash('sha256').update(fs.readFileSync(input)).digest('hex');
    const assets=new Map(files.map(name=>['/'+name,name]));
    server=http.createServer((req,res)=>{
      res.setHeader('Content-Security-Policy',"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'");
      if(req.url==='/'){
        res.setHeader('Content-Type','text/html');
        const scripts=[...files.filter(n=>n.includes('/vendor/')), 'static/retained-chart.js', 'static/project-measurements.js'];
        res.end('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/static/project-files.css"><link rel="stylesheet" href="/static/project-measurements.css"><main><p id="outside" tabindex="-1">Current project</p><section id="project-measurements"></section></main>'+scripts.map(n=>'<script src="/'+n+'"></script>').join(''));
        return;
      }
      if(req.url==='/fixture.json'){res.setHeader('Content-Type','application/json');res.end(fs.readFileSync(input));return;}
      const name=assets.get(req.url);if(!name){res.statusCode=404;res.end();return;}
      res.setHeader('Content-Type',name.endsWith('.css')?'text/css':name.endsWith('.json')?'application/json':'text/javascript');res.end(fs.readFileSync(path.join(source,name)));
    });
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    const base='http://127.0.0.1:'+server.address().port;
    const log=fs.openSync(path.join(output,'browser.log'),'w');
    browser=spawn(engine,['--headless','--disable-gpu','--no-first-run','--remote-debugging-port=0','--window-size=1440,1100','--user-data-dir='+path.join(output,'profile'),'about:blank'],{stdio:['ignore',log,log]});fs.closeSync(log);
    const port=await waitForDebugger(browser,path.join(output,'profile/DevToolsActivePort'));
    const tabs=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json(),socket=new WebSocket(tabs.find(v=>v.type==='page').webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
    session=cdpSession(socket,{onEvent:event=>{
      if(event.method==='Runtime.exceptionThrown')report.errors.push(event.params.exceptionDetails.exception?.description??event.params.exceptionDetails.text);
      if(event.method==='Network.requestWillBeSent'){const url=event.params.request.url;if(!url.startsWith(base)&&!url.startsWith('data:'))report.external.push(url);}
    }});
    const send=(...args)=>session.send(...args);
    async function ev(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description??r.exceptionDetails.text);return r.result.value;}
    async function until(expression){const end=Date.now()+15000;while(Date.now()<end){if(await ev(expression))return;await pause(30);}throw Error('DOM wait expired: '+expression+'; '+await ev('document.body.innerText.slice(-3500)'));}
    async function press(key='Enter',number=key==='Enter'?13:9){await send('Input.dispatchKeyEvent',{type:'keyDown',key,code:key,windowsVirtualKeyCode:number,...(key==='Enter'?{text:'\r'}:{})});await send('Input.dispatchKeyEvent',{type:'keyUp',key,code:key,windowsVirtualKeyCode:number});}
    async function activate(id){await ev(`document.getElementById(${JSON.stringify('project-measurements-'+id)}).focus()`);await press();}
    const idle="document.querySelector('#project-measurements').getAttribute('aria-busy')==='false'";
    await send('Runtime.enable');await send('Network.enable');await send('Page.enable');await send('Page.navigate',{url:base});
    await until("typeof window.ProjectMeasurements==='function' && typeof window.RetainedChart==='object' && typeof window.vegaEmbed==='function'");
    await ev(`(async()=>{
      window.fixture=await(await fetch('/fixture.json')).json();window.mode='normal';window.dataset='base';window.calls=[];window.csp=[];
      document.addEventListener('securitypolicyviolation',e=>csp.push(e.violatedDirective));
      const canonical=v=>Array.isArray(v)?v.map(canonical):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])])):v;
      window.answer=(action,args)=>{const data=fixture[dataset],wrap=value=>({source:data.source,value});
        if(action==='measurement_sources')return {items:[data.source_item].slice(args.offset,args.offset+args.limit),offset:args.offset,limit:args.limit,total:1};
        if(action==='measurement_metadata')return wrap(data.metadata);
        if(action==='measurement_subjects')return wrap({items:data.subjects.slice(args.offset,args.offset+args.limit),offset:args.offset,limit:args.limit,total:data.subjects.length});
        if(action==='measurement_inspect')return wrap(data.inspections[args.observation_id]);
        if(action==='measurement_query'){const {origin_catalog_id,packet_sha256,...query}=args;const found=data.queries.find(row=>JSON.stringify(canonical(row.request))===JSON.stringify(canonical(query)));if(!found)throw Error('Undeclared controlled query '+JSON.stringify(query));return wrap(found.value);}
        throw Error('Undeclared fixture action '+action);
      };
      window.ui=ProjectMeasurements({root:document.querySelector('#project-measurements'),onContextFocusLost:()=>document.querySelector('#outside').focus(),
        read:async(action,args,signal)=>{calls.push({action,args:structuredClone(args)});if(mode==='revoked')throw Error('Current scalar inspection was revoked');const value=structuredClone(answer(action,args));if(action==='measurement_query'&&mode==='held')return new Promise(resolve=>window.release=()=>resolve(value));if(action==='measurement_query'&&mode==='wrong')value.value.packet_sha256='f'.repeat(64);if(action==='measurement_subjects'&&mode==='wrongtotal')value.value.total+=1;return value;}});
      ui.setContext({key:'one',enabled:true});
    })()`);
    async function load(){await activate('sources');await until(idle);await activate('load');await until(idle);check(await ev("!document.querySelector('#project-measurements-form').hidden"),'Source and paged subjects loaded');}
    async function compare(){await activate('compare');await until(idle);}
    async function filter(index,match,value=''){await ev(`(()=>{const mode=document.querySelector('#project-measurements-filter-mode-${index}'),input=document.querySelector('#project-measurements-filter-value-${index}');mode.value=${JSON.stringify(match)};mode.dispatchEvent(new Event('change',{bubbles:true}));input.value=${JSON.stringify(value)};input.dispatchEvent(new Event('input',{bubbles:true}));})()`);}
    await load();check(await ev("document.activeElement.id==='project-measurements-metric'"),'Loading restores focus after enabling metric controls');
    await compare();await until("document.querySelectorAll('#project-measurements-chart g.mark-symbol path').length===2");
    await ev("document.querySelector('#project-measurements-results').scrollIntoView({block:'start'})");
    const overview=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(output,'comparison-overview.png'),Buffer.from(overview.data,'base64'));
    check(await ev("document.querySelector('#project-measurements-maxima').innerText.includes('2e-15') && document.querySelector('#project-measurements-maxima').innerText.includes('0.02') && !document.querySelector('#project-measurements img')"),'Exact tiny maxima, complete status counts and escaped labels are rendered');
    await ev("document.querySelector('#project-measurements-maxima button').focus()");
    check(await ev("document.activeElement.matches(':focus-visible') && parseFloat(getComputedStyle(document.activeElement).outlineWidth)>=3"),'Maximum inspection has visible keyboard focus');
    await press();await until(idle);
    check(await ev("calls.at(-1).action==='measurement_inspect' && calls.at(-1).args.observation_id==='a-case2-error' && document.querySelector('#project-measurements-inspection').innerText.includes('2e-15') && document.activeElement.tagName==='H3'"),'Keyboard selection reads exact worst case/source and focuses its details');
    const screenshot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(output,'scalar-comparison.png'),Buffer.from(screenshot.data,'base64'));
    await ev("document.querySelector('#project-measurements-chart').scrollIntoView()");
    const point=await ev("(()=>{const r=document.querySelectorAll('#project-measurements-chart g.mark-symbol path')[1].getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()");
    await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});await until("calls.at(-1).args.observation_id==='b-case2-error' && "+idle);
    check(await ev("document.querySelector('#project-measurements-inspection').innerText.includes('0.02')"),'Actual chart pointer selection opens the matching observation');
    await filter(0,'value','16');await compare();
    check(await ev("calls.at(-1).args.filters.cpm===16 && document.querySelector('#project-measurements-counts').innerText.includes('2 matching') && document.querySelector('#project-measurements-maxima').innerText.includes('1e-17')"),'Grid density filters retain their own exact cases and maxima');
    await filter(0,'any');await filter(1,'value','16');await compare();
    check(await ev("Object.hasOwn(calls.at(-1).args.filters,'grid_x') && !Object.hasOwn(calls.at(-1).args.filters,'cpm') && document.querySelector('#project-measurements-maxima').innerText.includes('2e-15')"),'Domain cell count selection is separate from grid density');
    await filter(0,'value','16');await compare();
    check(await ev("document.querySelector('#project-measurements-counts').innerText.includes('0 matching') && !document.querySelector('#project-measurements-chart svg') && document.querySelector('#project-measurements-maxima').innerText.includes('No measured value')"),'An empty exact intersection has no invented values or chart points');
    await filter(0,'any');await filter(1,'any');
    await ev("document.querySelector('#project-measurements-metric').value='latency';document.querySelector('#project-measurements-metric').dispatchEvent(new Event('change',{bubbles:true}))");await compare();
    check(await ev("!document.querySelector('#project-measurements-chart svg') && document.querySelector('#project-measurements-chart').innerText.includes('No measured values') && document.querySelector('#project-measurements-rows').innerText.includes('failed: Retained failure')"),'Missing and failed timings remain counted and produce no zero-latency points');
    await ev("document.querySelector('#project-measurements-metric').value='error';document.querySelector('#project-measurements-metric').dispatchEvent(new Event('change',{bubbles:true}));mode='held'");
    await activate('compare');await until("typeof release==='function'");
    await ev("ui.setContext({key:'two',enabled:true});release();mode='normal'");await pause(50);
    check(await ev("!document.querySelector('#project-measurements-chart svg') && document.querySelector('#project-measurements-source').options.length===0 && document.activeElement.id==='outside'"),'Late data after a host epoch change cannot restore results or selection');
    await load();await ev("mode='wrong'");await compare();
    check(await ev("document.querySelector('#project-measurements-status').innerText.includes('requested selection') && document.querySelector('#project-measurements-results').hidden && document.activeElement.id==='project-measurements-sources'"),'Wrong packet responses clear results and restore useful focus');
    await ev("mode='normal'");await load();await compare();await ev("mode='revoked'");await compare();
    check(await ev("document.querySelector('#project-measurements-status').innerText.includes('revoked') && !document.querySelector('#project-measurements svg') && document.querySelector('#project-measurements-inspection').hidden"),'Revoked reads clear chart and source inspection');
    await ev("mode='wrongtotal';dataset='large';ui.setContext({key:'wrong-population',enabled:true})");await activate('sources');await until(idle);await activate('load');await until(idle);
    check(await ev("document.querySelector('#project-measurements-status').innerText.includes('retained packet population') && document.querySelector('#project-measurements-form').hidden"),'A page cannot silently change the packet subject population');
    await ev("mode='normal';dataset='large';ui.setContext({key:'three',enabled:true})");await load();await compare();
    check(await ev("calls.at(-1).args.subject_ids.length===40 && document.querySelector('#project-measurements-subject-page').innerText.includes('1–40 of 41') && document.querySelector('#project-measurements-counts').innerText.includes('40 selected subjects of 41')"),'A large subject table exposes its page boundary and compares only an explicit subset');
    await activate('next');await until(idle);
    check(await ev("calls.at(-1).args.offset===50 && calls.at(-1).args.subject_ids.length===40 && document.querySelector('#project-measurements-counts').innerText.includes('51–80')"),'Observation paging preserves applied subjects and complete maxima');
    await activate('subjects-next');await until(idle);await compare();
    check(await ev("calls.at(-1).args.subject_ids.length===1 && calls.at(-1).args.subject_ids[0]==='s040' && document.querySelector('#project-measurements-counts').innerText.includes('1 selected subjects of 41')"),'The final subject page is accessible without silent omission');
    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});await pause(100);
    check(await ev('document.documentElement.scrollWidth<=391'),'Scalar controls and bounded tables fit a narrow viewport');
    await ev("document.querySelector('#project-measurements-maxima button').focus();ui.setContext({key:'three',enabled:false})");
    check(await ev("document.querySelector('#project-measurements').hidden && !document.querySelector('#project-measurements svg') && document.activeElement.id==='outside'"),'Loss of current access clears retained data even without a new context key');
    check(await ev('csp.length===0')&&report.errors.length===0&&report.external.length===0,'Pinned interpreter rendering has no CSP violations, JavaScript exceptions or external requests');
    report.status='passed_native_measurements_controlled_DOM';report.browser_version=execFileSync(engine,['--version'],{encoding:'utf8'}).trim();
  }catch(error){report.status='failed_native_measurements_controlled_DOM';report.error=String(error);if(session){try{const r=await session.send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(output,'failure.png'),Buffer.from(r.data,'base64'));}catch{}}throw error;}
  finally{
    session?.close();if(browser&&browser.exitCode===null){browser.kill('SIGTERM');for(let i=0;i<50&&browser.exitCode===null;i++)await pause(100);if(browser.exitCode===null)browser.kill('SIGKILL');}
    if(server)await new Promise(resolve=>server.close(resolve));report.browser_stopped=!browser||browser.exitCode!==null||browser.signalCode!==null;
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  }
  console.log(JSON.stringify({status:report.status,checks:report.checks.length,browser_stopped:report.browser_stopped}));
}
main().catch(error=>{console.error(String(error));process.exitCode=1;});
