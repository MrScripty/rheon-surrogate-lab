// The playground page: controls, the mouse forces and the frame loop. Drawing is view.js, the server is transport.js.
import { PlaneView, gradientCss, figure } from './view.js';
import { httpTransport } from './transport.js';

const transport = httpTransport('');
const $ = id => document.getElementById(id);
const AXES = ['x', 'y', 'z'], INDEX = { x: 0, y: 1, z: 2 };
const PLANES = { z: ['x', 'y'], y: ['x', 'z'], x: ['z', 'y'] };   // [horizontal, vertical], as the server draws them
const FRAME_MS = 33, SWIRL = 4;
const FIELD_NAMES = { tracer: 'dye', speed: 'speed', vx: 'velocity x', vy: 'velocity y', vz: 'velocity z', pressure: 'pressure', vorticity: 'vorticity', divergence: 'divergence' };

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const main = new PlaneView($('main-view'), { arrowColour: () => css('--arrow'), markColour: () => css('--mark') });
const side = new PlaneView($('side-view'), { markColour: () => css('--mark') });

const state = {
  caps: null, info: null, scene: null, kept: [], active: null, nextId: 1,
  sending: false, resend: false, lastSent: 0, frameTimer: null, busy: false, lastVersion: -1
};

// ---- theme ------------------------------------------------------------------------------------------------------
function storedTheme() { try { return localStorage.getItem('playground-theme'); } catch (_) { return null; } }
const isDark = () => document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
function applyTheme(name) {
  if (name === 'light' || name === 'dark') document.documentElement.dataset.theme = name; else delete document.documentElement.dataset.theme;
  const button = $('theme'), next = isDark() ? 'light' : 'dark';
  button.textContent = next === 'light' ? 'Light' : 'Dark'; button.setAttribute('aria-label', 'Switch to the ' + next + ' theme');
}
applyTheme(new URLSearchParams(location.search).get('theme') || storedTheme());
$('theme').addEventListener('click', () => {
  const next = isDark() ? 'light' : 'dark';
  applyTheme(next); try { localStorage.setItem('playground-theme', next); } catch (_) { /* private window */ }
  main.draw(); side.draw();
});

// ---- messages ---------------------------------------------------------------------------------------------------
function say(text, kind = 'error') { const m = $('message'); m.textContent = text || ''; m.dataset.kind = kind; }

// ---- the box ----------------------------------------------------------------------------------------------------
const counts = () => state.scene ? state.scene.counts : [32, 32, 32];
const extents = () => state.scene ? state.scene.extents_m : [2, 2, 2];
const spacing = () => state.scene ? 1 / state.scene.cells_per_metre : 1 / 16;
const cutAxis = () => $('axis').value;
const sideAxis = () => cutAxis() === 'z' ? 'x' : 'z';
const sliceIndex = () => Number($('index').value);

function setScene(scene) {
  const changed = !state.scene || JSON.stringify(state.scene.counts) !== JSON.stringify(scene.counts) || state.scene.cells_per_metre !== scene.cells_per_metre;
  state.scene = scene;
  if (!changed) return;
  updateIndexRange(true);
  const h = 1 / scene.cells_per_metre, smallest = Math.min(...scene.extents_m), r = $('radius');
  r.min = String(+(0.5 * h).toFixed(4)); r.max = String(+(0.25 * smallest).toFixed(4)); r.step = String(+(h / 10).toFixed(5));
  r.value = String(Math.max(1.5 * h, 0.07 * smallest)); showRadius();
}
function updateIndexRange(middle) {
  const n = counts()[INDEX[cutAxis()]], slider = $('index');
  slider.max = String(n - 1);
  if (middle || Number(slider.value) > n - 1) slider.value = String(Math.floor(n / 2));
  $('index-out').textContent = slider.value;
}
const showRadius = () => { $('radius-out').textContent = figure(Number($('radius').value), 3) + ' m'; };
const showStrength = () => { $('strength-out').textContent = $('strength').value + ' m/s²'; };

// ---- forces -----------------------------------------------------------------------------------------------------
const kind = () => document.querySelector('input[name=kind]:checked').value;
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
const unit = axis => AXES.map(a => (a === axis ? 1 : 0));

/** The blob a drag makes: placed at the press point in the current slice, pushing along the drag. */
function blobOf(drag) {
  const axis = drag.axis, [ha, va] = PLANES[axis], e = extents(), n = counts();
  const centre = [0.5, 0.5, 0.5];
  centre[INDEX[ha]] = drag.start.h; centre[INDEX[va]] = drag.start.v; centre[INDEX[axis]] = (drag.index + 0.5) / n[INDEX[axis]];
  const dh = (drag.now.h - drag.start.h) * e[INDEX[ha]], dv = (drag.now.v - drag.start.v) * e[INDEX[va]], length = Math.hypot(dh, dv);
  const magnitude = drag.strength * length / Math.min(...e);
  const blob = { centre, radius: drag.radius, direction: [0, 0, 0], magnitude: 0, swirl: 0, shake: 0 };
  if (drag.kind === 'push' && length > 0) { blob.direction[INDEX[ha]] = dh / length; blob.direction[INDEX[va]] = dv / length; blob.magnitude = magnitude; }
  if (drag.kind === 'swirl') {   // turning anticlockwise on screen when dragged right: the axis points out of the screen
    const sign = dh >= 0 ? 1 : -1, out = cross(unit(ha), unit(va));
    blob.direction = out.map(c => c * sign); blob.swirl = SWIRL; blob.magnitude = magnitude / SWIRL;
  }
  if (drag.kind === 'shake') { blob.shake = 1; blob.magnitude = magnitude; }
  return blob;
}
const forces = () => state.kept.map(f => f.blob).concat(state.active ? [blobOf(state.active)] : []);

async function sendForces() {
  if (state.sending) { state.resend = true; return; }
  state.sending = true; state.resend = false; state.lastSent = performance.now();
  try { await transport.setForces(forces()); say(''); }
  catch (error) { say(error.message); }
  finally {
    state.sending = false;
    if (state.resend) setTimeout(sendForces, Math.max(0, 40 - (performance.now() - state.lastSent)));
  }
}

function describe(f) {
  const b = f.blob, c = b.centre.map(v => v.toFixed(2)).join(', ');
  const strength = f.kind === 'swirl' ? figure(b.magnitude * b.swirl) : figure(b.magnitude);
  return `${f.kind} ${strength} m/s² at (${c}), r ${figure(b.radius)} m`;
}
function renderForceList() {
  const list = $('forces'), wanted = state.kept.map(f => String(f.id));
  for (const li of [...list.children]) if (!wanted.includes(li.dataset.id)) li.remove();
  state.kept.forEach((f, i) => {
    let li = list.querySelector(`li[data-id="${f.id}"]`);
    if (!li) {
      li = document.createElement('li'); li.dataset.id = String(f.id);
      const text = document.createElement('span'); text.textContent = describe(f); text.title = text.textContent;
      const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove'; remove.setAttribute('aria-label', 'Remove force ' + (i + 1));
      remove.addEventListener('click', () => { state.kept = state.kept.filter(k => k.id !== f.id); renderForceList(); sendForces(); });
      li.append(text, remove);
    }
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
  });
}

function overlays(view, axis) {
  const [ha, va] = PLANES[axis], e = extents();
  const items = state.kept.map(f => ({ f, active: false })).concat(state.active ? [{ f: { kind: state.active.kind, blob: blobOf(state.active) }, active: true }] : []);
  return items.map(({ f, active }) => {
    const b = f.blob, o = { h: b.centre[INDEX[ha]], v: b.centre[INDEX[va]], radius: b.radius / e[INDEX[ha]], kind: f.kind, active, dh: 0, dv: 0, sign: 1 };
    if (active && state.active.axis === axis) { o.dh = state.active.now.h - state.active.start.h; o.dv = state.active.now.v - state.active.start.v; }
    else if (f.kind === 'push') { const s = 0.12 * Math.min(1, b.magnitude / Number($('strength').value)) + 0.04; o.dh = b.direction[INDEX[ha]] * s; o.dv = b.direction[INDEX[va]] * s * e[INDEX[ha]] / e[INDEX[va]]; }
    if (f.kind === 'swirl') { const out = cross(unit(ha), unit(va)); o.sign = Math.sign(out[0] * b.direction[0] + out[1] * b.direction[1] + out[2] * b.direction[2]) || 1; }
    return o;
  });
}

const canvas = $('main-view');
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  const p = main.toPlane(event.clientX, event.clientY);
  if (!p) return;
  const max = state.caps && state.caps.current ? state.caps.current.max_blobs : 8;
  if (state.kept.length >= max) { say(`At most ${max} forces at once: remove one first.`); return; }
  event.preventDefault(); canvas.setPointerCapture(event.pointerId);
  state.active = { pointer: event.pointerId, start: p, now: p, axis: cutAxis(), index: sliceIndex(), kind: kind(), radius: Number($('radius').value), strength: Number($('strength').value) };
  $('hint').hidden = true; main.overlays = overlays(main, cutAxis()); main.draw(); sendForces();
});
canvas.addEventListener('pointermove', event => {
  const a = state.active; if (!a || a.pointer !== event.pointerId) return;
  a.now = main.toPlane(event.clientX, event.clientY, true) || a.now;
  main.overlays = overlays(main, cutAxis()); main.draw();
  if (performance.now() - state.lastSent > 40) sendForces(); else state.resend = true, setTimeout(() => { if (!state.sending && state.resend) sendForces(); }, 45);
});
function release(event) {
  const a = state.active; if (!a || a.pointer !== event.pointerId) return;
  if ($('keep').checked) { state.kept.push({ id: state.nextId++, kind: a.kind, blob: blobOf(a) }); renderForceList(); }
  state.active = null; main.overlays = overlays(main, cutAxis()); main.draw(); sendForces();
}
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);
$('clear-forces').addEventListener('click', () => { state.kept = []; state.active = null; renderForceList(); sendForces(); });

// ---- controls ---------------------------------------------------------------------------------------------------
async function act(promise) { try { const v = await promise; say(''); return v; } catch (error) { say(error.message); return null; } }
$('pause').addEventListener('click', async () => { const paused = $('pause').getAttribute('aria-pressed') === 'true'; await act(transport.control(paused ? 'resume' : 'pause')); });
$('step').addEventListener('click', () => act(transport.control('step', { steps: 1 })));
$('reset').addEventListener('click', () => act(transport.reset({ tracer: $('tracer').value })));
$('reset-tracer').addEventListener('click', () => act(transport.resetTracer($('tracer').value)));
for (const radio of document.querySelectorAll('input[name=mode]')) radio.addEventListener('change', () => act(transport.control('mode', { mode: radio.value })));
$('axis').addEventListener('change', () => { updateIndexRange(true); main.overlays = overlays(main, cutAxis()); });
$('index').addEventListener('input', () => { $('index-out').textContent = $('index').value; });
$('radius').addEventListener('input', showRadius);
$('strength').addEventListener('input', showStrength);
$('side').addEventListener('change', () => { $('side-view').hidden = $('side').value === 'off'; });
$('backend').addEventListener('change', fillBoxChoices);
$('apply').addEventListener('click', async () => {
  const [nx, ny, nz] = $('grid').value.split('x').map(Number);
  const scene = { counts: [nx, ny, nz], cells_per_metre: Number($('cpm').value), tracer: $('tracer').value };
  const backend = $('backend').value, current = state.caps && state.caps.current ? state.caps.current.id : null;
  $('apply').disabled = true; say('Preparing ' + $('grid').value + '…', 'info');
  const result = await act(backend === current ? transport.reset(scene) : transport.switchBackend(backend, scene));
  $('apply').disabled = false;
  if (result) { state.caps = result.capabilities; fillBackends(); }
});

function fillBackends() {
  const select = $('backend'), caps = state.caps, current = caps.current ? caps.current.id : null;
  select.replaceChildren(...caps.backends.map(b => { const o = new Option(b.label + (b.available ? '' : ' — unavailable'), b.id); o.disabled = !b.available; o.title = b.reason || ''; return o; }));
  if (current) select.value = current;
  fillBoxChoices();
}
function fillBoxChoices() {
  const caps = state.caps; if (!caps) return;
  const b = caps.backends.find(x => x.id === $('backend').value) || caps.backends[0];
  const scene = state.scene, now = scene ? scene.counts.join('x') : null;
  const grids = b.grids.map(g => g.join('x'));
  $('grid').replaceChildren(...grids.map(g => new Option(g.replace(/x/g, ' × '), g)));
  if (now && grids.includes(now)) $('grid').value = now;
  $('cpm').replaceChildren(...b.cells_per_metre.map(c => new Option(String(c), String(c))));
  if (scene && b.cells_per_metre.includes(scene.cells_per_metre)) $('cpm').value = String(scene.cells_per_metre);
  const current = caps.current && caps.current.id === b.id ? caps.current : null;
  $('backend-note').textContent = b.available ? (current ? 'Forces: ' + current.forcing + '.' : '') + (b.kind === 'learned' ? ' A learned model, trained on one gentle blob (0.08–0.16 m/s², about a fifth of the box wide): other forces are outside what it learned.' : '') : b.reason;
}

// ---- frames -----------------------------------------------------------------------------------------------------
function showStats(frame) {
  const info = frame.info, s = frame.stats, set = (k, t) => { const e = document.querySelector(`[data-stat="${k}"]`); if (e.textContent !== t) e.textContent = t; };
  if (info) {
    set('backend', info.label);
    if (info.scene) set('grid', info.scene.counts.join(' × ') + ' · ' + info.scene.cells_per_metre + ' cells/m');
  }
  set('ms', s.median_ms === null ? '– ms/step' : `${figure(s.median_ms)} ms/step (p95 ${figure(s.p95_ms)})`);
  set('rate', `${s.steps_per_s.toFixed(0)} steps/s`);
  set('rt', `${s.realtime_factor.toFixed(2)} × real time` + (s.headroom ? ` · could run ${figure(s.headroom, 2)} ×` : ''));
  const st = $('status');
  let label, key;
  if (frame.status === 'error') { label = 'Error'; key = 'error'; }
  else if (frame.status === 'loading') { label = 'Preparing'; key = 'loading'; }
  else if (frame.paused) { label = 'Paused · ' + figure(frame.time_s, 4) + ' s'; key = 'paused'; }
  else if (s.behind) { label = 'Behind real time'; key = 'behind'; }
  else { label = (frame.mode === 'realtime' ? 'Real time' : 'Fastest') + ' · ' + frame.time_s.toFixed(1) + ' s'; key = 'running'; }
  if (st.textContent !== label) st.textContent = label;
  st.dataset.state = key;
  $('pause').setAttribute('aria-pressed', String(!!frame.paused)); $('pause').textContent = frame.paused ? 'Resume' : 'Pause';
  const radio = document.querySelector(`input[name=mode][value="${frame.mode}"]`); if (radio && !radio.checked) radio.checked = true;
  if (frame.status === 'error' && frame.message) say(frame.message);
}

function showLegend(plane) {
  $('legend-name').textContent = FIELD_NAMES[plane.field] + (plane.unit ? ` (${plane.unit})` : '');
  $('legend-min').textContent = figure(plane.min); $('legend-max').textContent = figure(plane.max);
  $('legend-bar').style.background = gradientCss(plane.signed);
}

function lineFor(view, viewAxis, otherAxis, otherIndex) {
  const [ha, va] = PLANES[viewAxis], n = counts()[INDEX[otherAxis]], at = (otherIndex + 0.5) / n;
  if (ha === otherAxis) return { along: 'horizontal', at };
  if (va === otherAxis) return { along: 'vertical', at };
  return null;
}

async function tick() {
  state.frameTimer = null;
  if (document.hidden) { state.frameTimer = setTimeout(tick, 250); return; }
  const started = performance.now();
  const axis = cutAxis(), index = sliceIndex(), field = $('field').value, showSide = $('side').value !== 'off';
  const views = [{ axis, index, field }];
  if (showSide) views.push({ axis: sideAxis(), index: Math.floor(counts()[INDEX[sideAxis()]] / 2), field });
  try {
    const frame = await transport.frame(views, $('arrows').checked);
    if (frame.info && frame.info.scene) setScene(frame.info.scene);
    showStats(frame);
    if (frame.views.length) {
      const [first, second] = frame.views;
      if (first.axis === cutAxis()) {
        main.fit();
        main.overlays = overlays(main, first.axis);
        main.line = second ? lineFor(main, first.axis, second.axis, second.index) : null;
        main.draw(first); showLegend(first);
        canvas.setAttribute('aria-label', `${FIELD_NAMES[first.field]} on the ${first.axis} cut at cell ${first.index}`);
      }
      if (second) { side.fit(); side.line = lineFor(side, second.axis, first.axis, first.index); side.draw(second); }
    }
    if (frame.status !== 'error' && $('message').dataset.kind === 'error' && /frame|fetch|Failed/.test($('message').textContent)) say('');
  } catch (error) {
    say('No frame: ' + error.message);
  }
  state.frameTimer = setTimeout(tick, Math.max(0, FRAME_MS - (performance.now() - started)));
}

new ResizeObserver(() => { if (main.fit()) main.draw(); if (side.fit()) side.draw(); }).observe(canvas);

async function start() {
  showStrength(); showRadius();
  try {
    state.caps = await transport.capabilities();
    if (state.caps.info && state.caps.info.scene) setScene(state.caps.info.scene);
    if (state.caps.info && state.caps.info.scene) $('tracer').value = state.caps.info.scene.tracer;
    fillBackends();
  } catch (error) { say('The server did not answer: ' + error.message); }
  tick();
}
start();
