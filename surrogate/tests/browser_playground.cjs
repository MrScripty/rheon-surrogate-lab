'use strict';
// The playground page in a headless browser against the fake backend (no GPU):
//   1. the page loads, draws a slice and shows stats that keep changing;
//   2. a press-drag adds a force while held (the server has it), the dye moves, release removes it;
//   3. with "keep" on, two drags leave two forces listed and on the server; Remove takes one away;
//   4. light and dark at 1440 and 390 wide: canvas drawn, no sideways overflow, no script error, no failed request.
//   node surrogate/tests/browser_playground.cjs [--shots <dir>]
// Needs Node 22+ (WebSocket), the lab's .venv and a Chromium headless shell (Playwright's cache, or $BROWSER).
const assert = require('node:assert/strict'), fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { spawn } = require('node:child_process'), { once } = require('node:events');

const root = path.resolve(__dirname, '..', '..');
const argv = process.argv.slice(2), option = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : null; };
const shots = option('--shots') ? path.resolve(option('--shots')) : null; if (shots) fs.mkdirSync(shots, { recursive: true });
const python = process.env.PLAYGROUND_TEST_PYTHON || path.join(root, '.venv', 'bin', 'python');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'playground-browser-'));
const children = [], errors = [], failures = [];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, label, tries = 200) { for (let i = 0; i < tries; i++) { const v = await fn(); if (v) return v; await pause(100); } throw Error('Timed out: ' + label); }
const pass = text => console.log('PASS ' + text);

function findBrowser() {
  const candidates = [process.env.BROWSER];
  const cache = path.join(os.homedir(), '.cache', 'ms-playwright');
  if (fs.existsSync(cache)) for (const name of fs.readdirSync(cache).filter(n => n.startsWith('chromium_headless_shell-')).sort().reverse()) candidates.push(path.join(cache, name, 'chrome-headless-shell-linux64', 'chrome-headless-shell'));
  candidates.push('/usr/bin/chromium', '/usr/bin/google-chrome');
  const found = candidates.find(c => c && fs.existsSync(c)); assert.ok(found, 'No headless browser: set BROWSER'); return found;
}
function launch(command, args, options) { const child = spawn(command, args, options); children.push(child); return child; }

async function main() {
  // ---- the server, fake backend, a free port ----
  const server = launch(python, ['-m', 'surrogate.playground', '--backend', 'fake', '--grid', '32', '--port', '0'], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
  let out = ''; server.stdout.on('data', d => { out += d; }); server.stderr.on('data', d => { out += d; });
  const origin = await until(() => (out.match(/Playground at (http:\/\/127\.0\.0\.1:\d+)\//) || [])[1], 'the server: ' + out, 300);
  const api = async (p, body) => (await fetch(origin + p, body ? { method: 'POST', body: JSON.stringify(body) } : {})).json();

  // ---- the browser ----
  const browser = launch(findBrowser(), ['--headless', '--no-sandbox', '--disable-gpu', '--no-first-run', '--hide-scrollbars', '--remote-debugging-port=0', '--window-size=1440,900', '--user-data-dir=' + path.join(temporary, 'browser'), 'about:blank'], { stdio: 'ignore', env: { ...process.env, XDG_CONFIG_HOME: temporary, XDG_CACHE_HOME: temporary } });
  const activePort = path.join(temporary, 'browser', 'DevToolsActivePort');
  const port = await until(() => fs.existsSync(activePort) && fs.readFileSync(activePort, 'utf8').split('\n')[0], 'the browser debugger');
  const target = (await (await fetch('http://127.0.0.1:' + port + '/json')).json()).find(t => t.type === 'page');
  const socket = new WebSocket(target.webSocketDebuggerUrl); await once(socket, 'open');
  let sequence = 0; const pending = new Map();
  socket.addEventListener('message', event => {
    const m = JSON.parse(event.data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(Error(m.error.message)) : p.resolve(m.result); return; }
    if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map(a => a.value ?? a.description).join(' '));
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push(m.params.entry.text + ' ' + (m.params.entry.url || ''));
    if (m.method === 'Network.loadingFailed' && !m.params.canceled) failures.push(m.params.errorText);
    if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) failures.push(m.params.response.status + ' ' + m.params.response.url);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => { const id = ++sequence; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
  const evaluate = async expression => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails)); return r.result.value; };
  await send('Page.enable'); await send('Runtime.enable'); await send('Network.enable'); await send('Log.enable');
  const mouse = (type, x, y, buttons = 0) => send('Input.dispatchMouseEvent', { type, x, y, button: type === 'mouseMoved' && !buttons ? 'none' : 'left', buttons, clickCount: type === 'mouseMoved' ? 0 : 1 });
  const size = (width, height) => send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: width < 600 });
  const theme = name => send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: name }] });
  const frames = () => evaluate('new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))');
  const shot = async name => { if (!shots) return; await frames(); fs.writeFileSync(path.join(shots, name + '.png'), Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).data, 'base64')); };
  // What the main canvas shows: a checksum over its pixels and how many differ from its first pixel.
  const picture = () => evaluate(`(() => { const c = document.getElementById('main-view'); if (!c.width) return null; const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let sum = 0, varied = 0; for (let i = 0; i < d.length; i += 4) { sum = (sum + (i / 4 + 1) * (d[i] * 3 + d[i + 1] * 5 + d[i + 2] * 7)) % 2147483647; if (d[i] !== d[0] || d[i + 1] !== d[1] || d[i + 2] !== d[2]) varied++; } return { sum, varied, drawn: Number(c.dataset.drawn || 0), w: c.width, h: c.height }; })()`);
  const text = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).textContent`);
  const canvasBox = () => evaluate(`(() => { const r = document.getElementById('main-view').getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; })()`);
  async function load(name, width, height) {
    await size(width, height); await theme(name);
    await send('Page.navigate', { url: origin + '/?theme=' + name });
    await until(() => evaluate(`document.readyState === 'complete' && document.getElementById('status').dataset.state === 'running' && Number(document.getElementById('main-view').dataset.drawn || 0) > 2`), 'page running ' + name + ' ' + width);
  }
  /** Drag on the canvas from fractions (fx0, fy0) to (fx1, fy1) of its box, holding for `holdMs`. */
  async function drag(fx0, fy0, fx1, fy1, holdMs, during) {
    await evaluate(`document.getElementById('main-view').scrollIntoView({ block: 'center' })`); await frames();
    const b = await canvasBox(), x0 = b.x + fx0 * b.w, y0 = b.y + fy0 * b.h, x1 = b.x + fx1 * b.w, y1 = b.y + fy1 * b.h;
    await mouse('mouseMoved', x0, y0); await mouse('mousePressed', x0, y0, 1);
    for (let k = 1; k <= 8; k++) { await mouse('mouseMoved', x0 + (x1 - x0) * k / 8, y0 + (y1 - y0) * k / 8, 1); await pause(30); }
    await pause(holdMs); if (during) await during();
    await mouse('mouseReleased', x1, y1);
  }

  // 1. load, draw, stats
  await load('light', 1440, 900);
  await evaluate(`(() => { const s = document.getElementById('field'); s.value = 'tracer'; s.dispatchEvent(new Event('change')); return true; })()`);
  await pause(400);
  const before = await picture();
  assert.ok(before.varied > 1000, 'the slice is drawn: ' + JSON.stringify(before));
  const stats0 = await text('#stats'), steps0 = (await api('/api/frame?views=z::tracer')).steps;
  await pause(700);
  const stats1 = await text('#stats'), steps1 = (await api('/api/frame?views=z::tracer')).steps;
  assert.ok(/\d ms\/step/.test(stats1) && /× real time/.test(stats1), 'stats line: ' + stats1);
  assert.ok(steps1 > steps0 + 20 && stats1 !== stats0, 'the simulation steps and the stats update');
  pass('the page draws the slice and the stats update: ' + stats1);

  // 2. a held drag is a force on the server; the dye moves; release removes it
  let held = null;
  await drag(0.5, 0.75, 0.5, 0.45, 1200, async () => { held = (await api('/api/frame?views=z::tracer')).forces; });
  assert.equal(held.length, 1, 'one force while the button is held');
  assert.ok(held[0].direction[1] > 0.99 && held[0].magnitude > 1, 'the force pushes up (+y) with the drag: ' + JSON.stringify(held[0]));
  assert.ok(Math.abs(held[0].centre[0] - 0.5) < 0.05 && Math.abs(held[0].centre[1] - 0.25) < 0.06, 'placed at the press point: ' + JSON.stringify(held[0].centre));
  await until(async () => (await api('/api/frame?views=z::tracer')).forces.length === 0, 'the force removed on release');
  await pause(300);
  const after = await picture();
  assert.notEqual(after.sum, before.sum, 'the dye moved'); assert.ok(after.drawn > before.drawn, 'drawn again');
  pass('a held drag pushes along the drag at the press point, the frame changes, release removes it');
  await shot('playground-light-1440-drag');

  // 3. kept forces: listed, on the server, removable
  await evaluate(`(() => { const k = document.getElementById('keep'); if (!k.checked) k.click(); return k.checked; })()`);
  await drag(0.3, 0.5, 0.45, 0.5, 200);
  await evaluate(`(() => { document.querySelector('input[name=kind][value=swirl]').click(); return true; })()`);
  await drag(0.7, 0.5, 0.8, 0.5, 200);
  await until(async () => (await api('/api/frame?views=z::speed')).forces.length === 2, 'two kept forces on the server');
  const listed = await evaluate(`[...document.querySelectorAll('#forces li span')].map(s => s.textContent)`);
  assert.equal(listed.length, 2); assert.ok(listed[0].startsWith('push') && listed[1].startsWith('swirl'), JSON.stringify(listed));
  const kept = (await api('/api/frame?views=z::speed')).forces;
  assert.ok(kept[1].swirl > 0 && Math.abs(kept[1].direction[2]) > 0.99, 'the swirl turns about the screen normal: ' + JSON.stringify(kept[1]));
  await shot('playground-light-1440-kept');
  await evaluate(`document.querySelector('#forces li button').click()`);
  await until(async () => (await api('/api/frame?views=z::speed')).forces.length === 1, 'one force left on the server');
  assert.deepEqual(await evaluate(`[...document.querySelectorAll('#forces li span')].map(s => s.textContent.split(' ')[0])`), ['swirl']);
  await evaluate(`document.getElementById('clear-forces').click()`);
  await until(async () => (await api('/api/frame?views=z::speed')).forces.length === 0, 'all removed');
  pass('kept forces are listed, sent, and removed one by one or all');

  // 4. themes and widths
  for (const [name, width, height] of [['light', 1440, 900], ['dark', 1440, 900], ['light', 390, 844], ['dark', 390, 844]]) {
    await load(name, width, height);
    const p = await picture();
    const layout = await evaluate(`({ overflow: document.documentElement.scrollWidth - window.innerWidth, background: getComputedStyle(document.body).backgroundColor, dark: document.documentElement.dataset.theme })`);
    assert.ok(p && p.varied > 500, 'canvas drawn at ' + width + ' ' + name);
    assert.ok(layout.overflow <= 0, 'no sideways overflow at ' + width + ' ' + name + ': ' + layout.overflow);
    assert.equal(layout.dark, name);
    const bg = layout.background.match(/\d+/g).map(Number), light = bg[0] + bg[1] + bg[2] > 384;
    assert.equal(light, name === 'light', 'background follows the theme: ' + layout.background);
    if (width < 600) await drag(0.5, 0.6, 0.6, 0.4, 300);
    await shot(`playground-${name}-${width}`);
    pass(`${name} at ${width}: drawn, no overflow`);
  }

  const flipped = await evaluate(`(() => { const before = document.documentElement.dataset.theme; document.getElementById('theme').click(); return [before, document.documentElement.dataset.theme, document.getElementById('theme').textContent]; })()`);
  assert.deepEqual(flipped, ['dark', 'light', 'Dark'], 'the theme button switches dark to light');
  pass('the theme button switches the theme');

  assert.deepEqual(errors, [], 'no script errors'); assert.deepEqual(failures, [], 'no failed requests');
  pass('no script error and no failed request');
}

main().then(() => { console.log('OK'); }, error => { console.error('FAIL', error.stack || error); process.exitCode = 1; })
  .finally(async () => {
    await Promise.all(children.filter(c => c.exitCode === null && c.signalCode === null).map(c => { const gone = once(c, 'exit'); c.kill('SIGTERM'); return Promise.race([gone, pause(5000)]); }));
    try { fs.rmSync(temporary, { recursive: true, force: true }); } catch (_) { /* the browser's last writes; the OS cleans /tmp */ }
    process.exit();
  });
