// The playground server's HTTP API, decoded into what view.js draws. A future in-page backend replaces this module
// with one offering the same methods.

function bytes(base64, Type) {
  const text = atob(base64), out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
  return Type === Int8Array ? new Int8Array(out.buffer) : out;
}

async function call(method, path, body) {
  const response = await fetch(path, method === 'GET' ? { cache: 'no-store' } : {
    method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {})
  });
  let value = null;
  try { value = await response.json(); } catch (_) { value = null; }
  if (!response.ok) throw Object.assign(Error((value && value.error) || response.statusText), { status: response.status });
  return value;
}

export function httpTransport(base = '') {
  return {
    capabilities: () => call('GET', base + '/api/capabilities'),
    switchBackend: (backend, scene) => call('POST', base + '/api/backend', Object.assign({ backend }, scene)),
    reset: scene => call('POST', base + '/api/reset', scene),
    setForces: forces => call('POST', base + '/api/forces', { forces }),
    resetTracer: pattern => call('POST', base + '/api/tracer', { pattern }),
    control: (action, extra) => call('POST', base + '/api/control', Object.assign({ action }, extra || {})),
    /** views: [{ axis, index, field }] (one or two). Resolves to the frame with each plane's values decoded. */
    async frame(views, arrows) {
      const spec = views.map(v => `${v.axis}:${v.index ?? ''}:${v.field}`).join(',');
      const frame = await call('GET', `${base}/api/frame?views=${encodeURIComponent(spec)}&arrows=${arrows ? 1 : 0}`);
      for (const plane of frame.views) {
        plane.values = bytes(plane.data, Uint8Array); delete plane.data;
        if (plane.arrows) { plane.arrows.h = bytes(plane.arrows.h, Int8Array); plane.arrows.v = bytes(plane.arrows.v, Int8Array); }
      }
      return frame;
    }
  };
}
