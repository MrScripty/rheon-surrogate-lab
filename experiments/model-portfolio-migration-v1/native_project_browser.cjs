'use strict';
// Real native UI smoke through an installed Chromium headless engine and CDP.
// Own profile only; no existing browser/session or live Tuldok credential reads.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {spawn,execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..');
const controller=path.resolve(process.argv[2]),output=path.resolve(process.argv[3]);
const actions=process.argv[4]?JSON.parse(fs.readFileSync(process.argv[4],'utf8')):[];
const actionWaitMs=Number(process.argv[5]||15000);
if(![15000,300000].includes(actionWaitMs))throw Error('Supported action waits are 15000 or 300000 milliseconds');
if(fs.existsSync(output))throw Error('Fresh isolated browser evidence directory required');
fs.mkdirSync(output,{recursive:true,mode:0o700});
const browser='/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell';
const state=JSON.parse(fs.readFileSync(path.join(controller,'controller.json'),'utf8'));
const source=path.resolve(state.source);
if(!source.startsWith(path.join(root,'runs/model-portfolio-migration-v1/sources')+path.sep))throw Error('Task-owned native application source required');
const {waitForDebugger}=require(path.join(source,'tests/browser_startup.cjs'));
const {cdpSession}=require(path.join(source,'tests/browser_cdp_session.cjs'));
if(state.status!=='ready')throw Error('Observe the owned native controller live before browser use');
const flags=['--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+path.join(output,'profile'),'about:blank'];
if(process.env.RHEON_UI_VERIFY_NO_SANDBOX==='1')flags.unshift('--no-sandbox');
const report={status:'starting',origin:state.origin,controller_catalog_id:state.catalog_id,
  runner_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),
  engine:browser,flags,verification_only:true,normal_application_startup_changed:false,
  renderer_scope:'Headless UI functionality with GPU disabled; no model or rendering performance claim.',
  action_wait_ms:actionWaitMs,steps:[],errors:[],api_requests:[],browser_stopped:false,
  api_timing_scope:'Browser API request through headers/body completion, including server/auth/waits/transport; not model inference.'};
const apiRequests=new Map();
let child,session;
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(fn,label){const deadline=Date.now()+actionWaitMs;while(Date.now()<deadline){if(await fn())return;await pause(100);}throw Error('UI wait expired: '+label);}
async function main(){
  try{
    const log=fs.openSync(path.join(output,'browser.log'),'w');
    child=spawn(browser,flags,{stdio:['ignore',log,log],env:{...process.env,
      XDG_CONFIG_HOME:path.join(output,'config'),XDG_CACHE_HOME:path.join(output,'cache')}});
    fs.closeSync(log);
    const port=await waitForDebugger(child,path.join(output,'profile/DevToolsActivePort'));
    const targets=await (await fetch('http://127.0.0.1:'+port+'/json/list')).json();
    const target=targets.find(t=>t.type==='page');
    const socket=new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
    session=cdpSession(socket,{onEvent:event=>{
      if(event.method==='Runtime.exceptionThrown')report.errors.push(event.params.exceptionDetails.text);
      if(event.method==='Network.requestWillBeSent'){
        const p=event.params,url=new URL(p.request.url);
        if(url.origin===state.origin&&url.pathname.startsWith('/api/')){
          const row={method:p.request.method,path:url.pathname,started_timestamp:p.timestamp};
          apiRequests.set(p.requestId,row);report.api_requests.push(row);
        }
      }else if(event.method==='Network.responseReceived'){
        const p=event.params,row=apiRequests.get(p.requestId);
        if(row){row.status=p.response.status;row.headers_wall_ms=(p.timestamp-row.started_timestamp)*1000;}
      }else if(event.method==='Network.loadingFinished'||event.method==='Network.loadingFailed'){
        const p=event.params,row=apiRequests.get(p.requestId);
        if(row){row.complete=event.method==='Network.loadingFinished';row.total_wall_ms=(p.timestamp-row.started_timestamp)*1000;}
      }
    }});
    const send=session.send;
    await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
    const tokenPath=path.join(controller,'private/browser-login-'+crypto.randomUUID()+'.txt');
    const db=JSON.parse(fs.readFileSync(path.join(controller,'private/database.json'),'utf8'));
    execFileSync(path.join(root,'runs/dependency-migration-v1/app-env/bin/python'),
      ['-B',path.join(source,'access.py'),'login','--data',state.control,'--out',tokenPath],
      {cwd:source,env:{...process.env,TULDOK_DATABASE_URL:db.dsn},stdio:'pipe',timeout:30000});
    const login=await fetch(state.origin+'/api/access/session',{method:'POST',
      headers:{'Content-Type':'application/json','Origin':state.origin},
      body:JSON.stringify({token:fs.readFileSync(tokenPath,'utf8').trim()})});
    const identity=await login.json();
    if(login.status!==200||identity.kind!=='owner'||identity.catalog_id!==state.catalog_id)throw Error('Isolated browser owner login failed');
    const cookie=login.headers.get('set-cookie').split(';')[0];
    const equal=cookie.indexOf('=');
    await send('Network.setCookie',{name:cookie.slice(0,equal),value:cookie.slice(equal+1),
      url:state.origin+'/',path:'/',httpOnly:true,sameSite:'Strict'});
    async function evaluate(expression){const v=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
      if(v.exceptionDetails)throw Error(v.exceptionDetails.exception?.description??v.exceptionDetails.text);return v.result.value;}
    await send('Page.navigate',{url:state.origin+'/project-files'});
    await until(()=>evaluate("document.readyState==='complete' && !!document.querySelector('#project-files-summary') && !document.querySelector('#project-files-summary').textContent.includes('No project status')"),'native project state');
    await until(()=>evaluate("!document.querySelector('#project-files-refresh').matches(':disabled')"),'native controls ready');
    for(const action of actions){
      if(action.kind==='fill')await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(action.selector)});if(!e||e.matches(':disabled'))throw Error('Unavailable input control');e.focus();e.value=${JSON.stringify(action.value)};e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      else if(action.kind==='select_values'){
        if(!Array.isArray(action.values)||!action.values.length||new Set(action.values).size!==action.values.length
            ||action.values.some(value=>typeof value!=='string'))throw Error('Explicit unique selection values required');
        await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(action.selector)}),wanted=${JSON.stringify(action.values)};
          if(!e||e.tagName!=='SELECT'||!e.multiple||e.matches(':disabled'))throw Error('Available multiple selection required');
          if(wanted.some(value=>![...e.options].some(option=>option.value===value)))throw Error('Selection value is unavailable');
          for(const option of e.options)option.selected=wanted.includes(option.value);
          e.focus();e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      }
      else if(action.kind==='click'){
        await evaluate(`(()=>{const all=[...document.querySelectorAll(${JSON.stringify(action.selector)})];const selected=${action.text?`all.filter(e=>e.textContent.includes(${JSON.stringify(action.text)}))`:'all.slice(0,1)'};if(selected.length!==1||selected[0].matches(':disabled'))throw Error('Unavailable or ambiguous control');selected[0].scrollIntoView({block:'center'});selected[0].focus();})()`);
        await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
        await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
      }else if(action.kind==='wait_text')await until(()=>evaluate(`document.querySelector(${JSON.stringify(action.selector||'body')})?.innerText.includes(${JSON.stringify(action.text)})`),action.text);
      else if(action.kind==='wait_value')await until(()=>evaluate(`document.querySelector(${JSON.stringify(action.selector)})?.value.includes(${JSON.stringify(action.text)})`),action.text);
      else if(action.kind==='wait_enabled')await until(()=>evaluate(`!!document.querySelector(${JSON.stringify(action.selector)}) && !document.querySelector(${JSON.stringify(action.selector)}).matches(':disabled')`),action.selector+' enabled');
      else if(action.kind==='assert_text'){
        if(!await evaluate(`document.body.innerText.includes(${JSON.stringify(action.text)})`))throw Error('Expected native UI text missing: '+action.text);
      }else if(action.kind==='scroll')await evaluate(`document.querySelector(${JSON.stringify(action.selector)}).scrollIntoView({block:'center'})`);
      else if(action.kind==='screenshot'){
        if(!/^[A-Za-z0-9_-]{1,80}\.png$/.test(action.name))throw Error('Bounded screenshot filename required');
        const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
        fs.writeFileSync(path.join(output,action.name),Buffer.from(shot.data,'base64'),{flag:'wx'});
      }
      else throw Error('Unknown bounded browser action');
      report.steps.push(action);
    }
    report.page=await evaluate("({title:document.title,body:document.body.innerText,error:document.querySelector('#project-files-error')?.hidden?null:document.querySelector('#project-files-error')?.textContent})");
    if(report.page.error)throw Error('Native UI reports: '+report.page.error);
    const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(output,'native-project.png'),Buffer.from(shot.data,'base64'));
    report.status='passed_native_project_UI_actions';
  }catch(error){
    report.status='failed_native_project_UI_actions';report.error=String(error);
    // Preserve the failed page before closing this owned browser. A timeout
    // alone cannot distinguish pending work, an API refusal and a fixture bug.
    if(session){
      try{
        const state=await session.send('Runtime.evaluate',{returnByValue:true,
          expression:"({title:document.title,body:document.body?.innerText.slice(0,20000),error:document.querySelector('#project-files-error')?.hidden?null:document.querySelector('#project-files-error')?.textContent})"});
        report.failure_page=state.result?.value;
      }catch(captureError){report.failure_page_error=String(captureError);}
      try{
        const shot=await session.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
        fs.writeFileSync(path.join(output,'failed-native-project.png'),Buffer.from(shot.data,'base64'));
      }catch(captureError){report.failure_screenshot_error=String(captureError);}
    }
    throw error;
  }
  finally{
    session?.close();
    if(child&&child.exitCode===null&&child.signalCode===null){child.kill('SIGTERM');
      for(let i=0;i<100&&child.exitCode===null&&child.signalCode===null;i++)await pause(100);
      if(child.exitCode===null&&child.signalCode===null)child.kill('SIGKILL');}
    report.browser_stopped=!child||child.exitCode!==null||child.signalCode!==null;
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  }
  console.log(JSON.stringify({status:report.status,steps:report.steps.length,errors:report.errors,output}));
}
main().catch(error=>{console.error(String(error));process.exitCode=1;});
