// Drawing a decoded frame on a canvas. Nothing here knows how frames arrive (see transport.js), so the same code can
// draw frames a model computes in the page itself.
//
// A plane is { width, height, values: Uint8Array (row 0 at the bottom), min, max, signed, horizontal, vertical,
// field, unit, arrows?: { block, width, height, h: Int8Array, v: Int8Array, max_speed } }.

const SEQUENTIAL = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'];
const DIVERGING = ['#3b4cc0', '#5977e3', '#7b9ff9', '#9ebeff', '#c0d4f5', '#dddcdc', '#f2cbb7', '#f7ac8e', '#ee8468', '#d65244', '#b40426'];

function lut(stops) {
  const rgb = stops.map(s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16)));
  const table = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) table[i * 3 + c] = Math.round(rgb[k][c] * (1 - f) + rgb[k + 1][c] * f);
  }
  return table;
}
export const COLOURS = { sequential: lut(SEQUENTIAL), diverging: lut(DIVERGING) };
export const gradientCss = signed => 'linear-gradient(to right, ' + (signed ? DIVERGING : SEQUENTIAL).join(', ') + ')';

/** A view of one plane on one canvas: draws frames, maps pointer positions to the plane, overlays forces. */
export class PlaneView {
  constructor(canvas, { arrowColour = () => '#fff', markColour = () => '#fff' } = {}) {
    this.canvas = canvas; this.context = canvas.getContext('2d');
    this.image = document.createElement('canvas'); this.imageContext = this.image.getContext('2d');
    this.plane = null; this.box = null; this.overlays = []; this.line = null;
    this.arrowColour = arrowColour; this.markColour = markColour;
    this.draws = 0;
  }

  /** Fit the drawing buffer to the canvas's CSS size (device pixels). Returns true when it changed. */
  fit() {
    const r = this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width === w && this.canvas.height === h) return false;
    this.canvas.width = w; this.canvas.height = h; return true;
  }

  /** Where the plane sits in the canvas: square cells, centred. Physical aspect = cells (cubic cells). */
  layout(plane) {
    const W = this.canvas.width, H = this.canvas.height, s = Math.min(W / plane.width, H / plane.height);
    return { x: (W - plane.width * s) / 2, y: (H - plane.height * s) / 2, w: plane.width * s, h: plane.height * s, s };
  }

  draw(plane) {
    if (plane) this.plane = plane;
    plane = this.plane;
    const g = this.context;
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!plane) return;
    const { width, height, values } = plane, table = plane.signed ? COLOURS.diverging : COLOURS.sequential;
    if (this.image.width !== width || this.image.height !== height) { this.image.width = width; this.image.height = height; }
    const pixels = this.imageContext.createImageData(width, height), d = pixels.data;
    for (let row = 0; row < height; row++) {
      const source = (height - 1 - row) * width, target = row * width;   // row 0 of the plane is at the bottom
      for (let col = 0; col < width; col++) {
        const v = values[source + col], o = (target + col) * 4;
        d[o] = table[v * 3]; d[o + 1] = table[v * 3 + 1]; d[o + 2] = table[v * 3 + 2]; d[o + 3] = 255;
      }
    }
    this.imageContext.putImageData(pixels, 0, 0);
    const box = this.box = this.layout(plane);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.drawImage(this.image, box.x, box.y, box.w, box.h);
    if (plane.arrows) this.drawArrows(plane.arrows, box);
    if (this.line) this.drawLine(box);
    for (const o of this.overlays) this.drawForce(o, box);
    this.draws++; this.canvas.dataset.drawn = String(this.draws);
  }

  drawArrows(a, box) {
    const g = this.context, cell = box.s * a.block, longest = cell * 0.9;
    g.save(); g.strokeStyle = this.arrowColour(); g.fillStyle = g.strokeStyle; g.lineWidth = Math.max(1, cell / 14); g.globalAlpha = 0.85;
    for (let r = 0; r < a.height; r++) for (let c = 0; c < a.width; c++) {
      const i = r * a.width + c, u = a.h[i] / 127, v = a.v[i] / 127, m = Math.hypot(u, v);
      if (m < 0.04) continue;
      const cx = box.x + (c + 0.5) * cell, cy = box.y + box.h - (r + 0.5) * cell, dx = u * longest, dy = -v * longest;
      const x0 = cx - dx / 2, y0 = cy - dy / 2, x1 = cx + dx / 2, y1 = cy + dy / 2, head = Math.min(cell * 0.3, Math.hypot(dx, dy) * 0.45), angle = Math.atan2(dy, dx);
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - head * Math.cos(angle - 0.45), y1 - head * Math.sin(angle - 0.45)); g.lineTo(x1 - head * Math.cos(angle + 0.45), y1 - head * Math.sin(angle + 0.45)); g.closePath(); g.fill();
    }
    g.restore();
  }

  /** A line across the plane at a fraction of its horizontal or vertical axis (where another view cuts it). */
  drawLine(box) {
    const g = this.context, { along, at } = this.line;
    g.save(); g.strokeStyle = this.markColour(); g.globalAlpha = 0.7; g.setLineDash([6, 5]); g.lineWidth = Math.max(1, (window.devicePixelRatio || 1));
    g.beginPath();
    if (along === 'horizontal') { const x = box.x + at * box.w; g.moveTo(x, box.y); g.lineTo(x, box.y + box.h); }
    else { const y = box.y + box.h - at * box.h; g.moveTo(box.x, y); g.lineTo(box.x + box.w, y); }
    g.stroke(); g.restore();
  }

  /** A force drawn where it acts: a circle of its radius and its drag. o = { h, v, radius (fraction of width), dh, dv, active, kind }. */
  drawForce(o, box) {
    const g = this.context, x = box.x + o.h * box.w, y = box.y + box.h - o.v * box.h, r = Math.max(4, o.radius * box.w);
    g.save(); g.strokeStyle = this.markColour(); g.lineWidth = Math.max(1.5, (window.devicePixelRatio || 1) * 1.5); g.globalAlpha = o.active ? 1 : 0.65;
    if (!o.active) g.setLineDash([4, 4]);
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
    if (o.kind === 'push' && (o.dh || o.dv)) { g.beginPath(); g.moveTo(x, y); g.lineTo(x + o.dh * box.w, y - o.dv * box.h); g.stroke(); }
    if (o.kind === 'swirl') { g.beginPath(); g.arc(x, y, r * 0.55, o.sign > 0 ? 0.3 : -0.3, o.sign > 0 ? Math.PI * 1.6 : -Math.PI * 1.6, o.sign < 0); g.stroke(); }
    if (o.kind === 'shake') { g.beginPath(); for (let k = -3; k <= 3; k++) { const px = x + k * r / 4, py = y + (k % 2 ? -1 : 1) * r * 0.25; k === -3 ? g.moveTo(px, py) : g.lineTo(px, py); } g.stroke(); }
    g.restore();
  }

  /** A pointer position (client pixels) as fractions of the plane (h to the right, v up), or null outside it. */
  toPlane(clientX, clientY, clamp = false) {
    if (!this.box) return null;
    const r = this.canvas.getBoundingClientRect(), sx = this.canvas.width / r.width, sy = this.canvas.height / r.height;
    let h = ((clientX - r.left) * sx - this.box.x) / this.box.w, v = 1 - ((clientY - r.top) * sy - this.box.y) / this.box.h;
    if (clamp) { h = Math.min(1, Math.max(0, h)); v = Math.min(1, Math.max(0, v)); }
    return h >= 0 && h <= 1 && v >= 0 && v <= 1 ? { h, v } : null;
  }
}

/** Format a number for the legend and the stats line. */
export function figure(x, digits = 3) {
  if (x === null || x === undefined || !isFinite(x)) return '–';
  if (x === 0) return '0';
  const a = Math.abs(x);
  return a >= 1e4 || a < 1e-3 ? x.toExponential(digits - 1).replace('e+', 'e') : String(Number(x.toPrecision(digits)));
}
