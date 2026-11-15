import { STYLE, colHeight } from './net-style.js';

// Static 2D fallback (reduced motion / no WebGL): the same real weights, drawn flat once.
export function drawNet2D(canvas, data, style = {}) {
  const S = { ...STYLE, ...style };
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  const grd = g.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0, S.bgTop); grd.addColorStop(1, S.bgBottom);
  g.fillStyle = grd; g.fillRect(0, 0, w, h);
  const L = data.layers.length, pad = 18;
  const maxH = Math.max(...data.layers.map(l => colHeight(l.size)));
  const P = data.layers.map((ly, k) => {
    const hh = colHeight(ly.size) / maxH * (h - 50);
    return Array.from({ length: ly.n }, (_, i) => [pad + k * (w - 2 * pad) / (L - 1), h / 2 + 10 + (ly.n === 1 ? 0 : (i / (ly.n - 1) - 0.5) * hh)]);
  });
  g.globalCompositeOperation = 'lighter';
  g.lineWidth = 0.6;
  for (const e of data.edges) {
    for (let m = 0; m < e.w.length; m++) {
      const [x0, y0] = P[e.from][e.src[m]], [x1, y1] = P[e.to][e.dst[m]], aw = Math.abs(e.w[m]);
      const c = e.w[m] >= 0 ? S.pos : S.neg;
      g.strokeStyle = `rgba(${c.map(v => Math.round(v * 255)).join(',')},${(0.03 + 0.12 * aw).toFixed(3)})`;
      const dx = (x1 - x0) / 2;
      g.beginPath(); g.moveTo(x0, y0); g.bezierCurveTo(x0 + dx, y0, x1 - dx, y1, x1, y1); g.stroke();
    }
  }
  g.fillStyle = 'rgba(240,238,230,0.9)';
  for (const col of P) for (const [x, y] of col) { g.beginPath(); g.arc(x, y, 1.3, 0, 7); g.fill(); }
}
