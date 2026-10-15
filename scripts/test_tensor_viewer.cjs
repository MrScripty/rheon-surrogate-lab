'use strict';
// Real Chromium interaction evidence; no model inference or scientific execution.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const {spawn} = require('node:child_process');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '..');
const report = path.join(root, 'runs/tensor-viewer');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'rheon-tensor-browser-'));
const checks = [], errors = [], pending = new Map();
let browser, socket, serial = 0;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn) {
  for (let i = 0; i < 200; i++) { const result = await fn(); if (result) return result; await pause(50); }
  throw Error('Timed out waiting for browser state');
}
function command(method, params = {}) {
  const id = ++serial;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(Error('CDP timed out: ' + method)); }, 10000);
    pending.set(id, {resolve: result => { clearTimeout(timer); resolve(result); }, reject: error => { clearTimeout(timer); reject(error); }});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function navigate(url) {
  await command('Page.navigate', {url});
  await until(() => evaluate("document.readyState==='complete' && !!document.getElementById('values')?.children.length"));
}
const server = http.createServer((req, res) => {
  const filename = path.resolve(root, '.' + decodeURIComponent(new URL(req.url, 'http://local').pathname));
  if (!filename.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  fs.readFile(filename, (error, raw) => {
    if (error) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', filename.endsWith('.html') ? 'text/html' : 'application/json'); res.end(raw);
  });
});
(async () => {
  fs.mkdirSync(report, {recursive: true});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = 'http://127.0.0.1:' + server.address().port;
  browser = spawn(process.env.BROWSER || '/usr/bin/chromium', ['--headless', '--no-sandbox', '--disable-gpu', '--remote-debugging-port=0', '--user-data-dir=' + temporary, 'about:blank'], {stdio: 'ignore'});
  const portFile = path.join(temporary, 'DevToolsActivePort');
  await until(() => fs.existsSync(portFile));
  const port = fs.readFileSync(portFile, 'utf8').split('\n')[0];
  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  socket = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) { const item = pending.get(message.id); pending.delete(message.id); message.error ? item.reject(Error(JSON.stringify(message.error))) : item.resolve(message.result); }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  };
  await command('Runtime.enable'); await command('Page.enable');
  await command('Emulation.setDeviceMetricsOverride', {width:1440, height:1100, deviceScaleFactor:1, mobile:false});
  await navigate(origin + '/viewer/tensor.html');
  assert.deepEqual(await evaluate('tensor.shape'), [2,7,16,16,16]);
  assert.match(await evaluate("$('source').textContent"), /Illustrative/);
  await evaluate("$('preset').value='24,16,12';$('preset').dispatchEvent(new Event('change'))");
  assert.deepEqual(await evaluate('tensor.shape'), [2,7,12,16,24]);
  assert.match(await evaluate("$('geometry').textContent"), /1.5 × 1 × 0.75 m/);
  checks.push('Rectangular domain changes counts while preserving 0.0625 m cubic cells');
  await evaluate("document.querySelector('input[name=channel][value=\"2\"]').click();$('batch').value='1';$('batch').dispatchEvent(new Event('change'));$('x').value='3';$('x').dispatchEvent(new Event('input'))");
  assert.equal(await evaluate('value(2)'), .21875);
  assert.match(await evaluate("$('address').textContent"), /input\[1, 2, 6, 8, 3\] = 0.21875/);
  // Native keyboard increment reaches the same authoritative selection state.
  await evaluate("$('x').focus()");
  await command('Input.dispatchKeyEvent', {type:'keyDown',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});
  await command('Input.dispatchKeyEvent', {type:'keyUp',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});
  assert.equal(await evaluate('state.x'), 4);
  const point = await evaluate("(()=>{const r=$('slice').getBoundingClientRect(),g=sliceBounds;return {x:r.left+g.left+2.5*g.step,y:r.top+g.top+(g.ny-1-.5)*g.step};})()");
  await command('Input.dispatchMouseEvent', {type:'mousePressed',x:point.x,y:point.y,button:'left',clickCount:1});
  await command('Input.dispatchMouseEvent', {type:'mouseReleased',x:point.x,y:point.y,button:'left',clickCount:1});
  assert.deepEqual(await evaluate('[state.x,state.y]'), [2,1]);
  checks.push('Sample/channel selection, keyboard indices and real pointer heatmap selection agree');
  await navigate(origin + '/viewer/tensor.html?data=../runs/tensor-viewer/seven-channel-inputs.json');
  await until(() => evaluate("$('status').textContent.includes('Recorded tensor loaded')"));
  const recorded = JSON.parse(fs.readFileSync(path.join(report, 'seven-channel-inputs.json')));
  assert.deepEqual(await evaluate('tensor.shape'), recorded.shape);
  await evaluate("$('batch').value='1';$('batch').dispatchEvent(new Event('change'));document.querySelector('input[name=channel][value=\"0\"]').click();$('z').value='4';$('z').dispatchEvent(new Event('input'));$('y').value='6';$('y').dispatchEvent(new Event('input'));$('x').value='8';$('x').dispatchEvent(new Event('input'))");
  const offset = ((((1*7)*5+4)*7+6)*9+8);
  assert.equal(await evaluate('value(0)'), recorded.values[offset]);
  assert.equal(await evaluate('index(state.b,state.c,state.z,state.y,state.x)'), offset);
  assert.deepEqual(await evaluate('[value(5),value(6)]'), [recorded.values[(7+5)*315],recorded.values[(7+6)*315]]);
  assert.equal(await evaluate("$('channels').querySelectorAll('input').length"),7);
  assert.equal(await evaluate("$('channels').textContent.includes('Voxel size')"),false);
  checks.push('Seven actual input channels; density/time occupy indices 5/6 and no voxel-size maps are displayed');
  checks.push('Recorded float32 tensor values and flattened offsets match the actual export exactly');
  const invalid = await evaluate("(async()=>{const previous=tensor;try{await loadText(JSON.stringify({...tensor,values:[],spacing_m:[.1,.2,.1]}),loadGeneration)}catch(error){loadError(error,loadGeneration)}return [tensor===previous,$('status').textContent]})()");
  assert.equal(invalid[0],true); assert.match(invalid[1], /must be finite, positive and equal/);
  await evaluate("$('status').className='status';$('status').textContent='Recorded tensor loaded. Values are float32 network inputs.'");
  checks.push('Malformed anisotropic input reports an error and preserves the current view');
  await command('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:false});
  await pause(100);
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'), true);
  assert.equal(await evaluate("$('x').max"), '8');
  checks.push('390 px layout has no horizontal overflow; controls retain actual grid bounds');
  await command('Emulation.setDeviceMetricsOverride', {width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await pause(100);
  const screenshot = await command('Page.captureScreenshot', {format:'png',captureBeyondViewport:true});
  fs.writeFileSync(path.join(report, 'tensor-seven-channel.png'), Buffer.from(screenshot.data,'base64'));
  await navigate(pathToFileURL(path.join(root,'viewer/tensor.html')).href);
  assert.deepEqual(await evaluate('tensor.shape'), [2,7,16,16,16]);
  checks.push('Standalone file opens offline with complete illustrative tensor');
  const document = await command('DOM.getDocument');
  const input = await command('DOM.querySelector', {nodeId:document.root.nodeId,selector:'#file'});
  await command('DOM.setFileInputFiles', {nodeId:input.nodeId,files:[path.join(report,'seven-channel-inputs.json')]});
  await until(() => evaluate("$('status').textContent.includes('Recorded tensor loaded')"));
  assert.deepEqual(await evaluate('tensor.shape'),recorded.shape);
  assert.equal(await evaluate("$('preset').value"),'');
  checks.push('Native file input loads recorded tensors offline without a server');
  const historical=path.join(report,'recorded-inputs.json');
  if(fs.existsSync(historical)){
    await command('DOM.setFileInputFiles', {nodeId:input.nodeId,files:[historical]});
    await until(() => evaluate('tensor.shape[1]===10'));
    assert.equal(await evaluate("$('channels').querySelectorAll('input').length"),10);
    assert.equal(await evaluate("$('channels').textContent.includes('Voxel size')"),true);
    checks.push('Preserved historical ten-channel exports retain their actual channel labels and shapes');
  }
  assert.deepEqual(errors, []);
  const evidence = {status:'passed',browser:process.env.BROWSER || '/usr/bin/chromium',checks,consoleExceptions:errors,screenshot:'runs/tensor-viewer/tensor-seven-channel.png',recordedTensorSha256:recorded.provenance.tensor_f32_sha256};
  fs.writeFileSync(path.join(report,'seven-channel-browser-checks.json'),JSON.stringify(evidence,null,2)+'\n');
  console.log(JSON.stringify(evidence));
})().catch(error => { console.error(error); process.exitCode=1; }).finally(async () => {
  if(socket)socket.close();
  if(browser){const exited=new Promise(resolve=>browser.once('exit',resolve));browser.kill();await exited;}
  server.close();fs.rmSync(temporary,{recursive:true,force:true});
});
