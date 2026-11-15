// Training curves from assets/data/training_curves.json (tools/export_training_curves.py; real logs only).
// Small SVG charts whose lines draw in once they enter the view (IntersectionObserver, no scroll listener).
const NS = 'http://www.w3.org/2000/svg';
const el = (t, a = {}, p) => { const e = document.createElementNS(NS, t); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (p) p.appendChild(e); return e; };
const f = (v, d) => Number(v).toLocaleString('en', { minimumFractionDigits: d, maximumFractionDigits: d });

function lineChart(c) {
  const W = 320, H = 150, P = { l: 34, r: 8, t: 10, b: 22 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'cv-svg', role: 'img', 'aria-label': `${c.title}: ${c.y} vs ${c.x}` });
  const all = c.series.flatMap(s => s.points);
  const x0 = Math.min(...all.map(p => p[0])), x1 = Math.max(...all.map(p => p[0]));
  let y0 = Math.min(...all.map(p => p[1])), y1 = Math.max(...all.map(p => p[1]));
  const pad = (y1 - y0) * 0.15 || 0.01; y0 -= pad; y1 += pad;
  const X = x => P.l + (x - x0) / (x1 - x0 || 1) * (W - P.l - P.r), Y = y => H - P.b - (y - y0) / (y1 - y0 || 1) * (H - P.t - P.b);
  for (let k = 0; k <= 2; k++) {
    const v = y0 + (y1 - y0) * k / 2, y = Y(v);
    el('line', { x1: P.l, x2: W - P.r, y1: y, y2: y, class: 'cv-grid' }, svg);
    const t = el('text', { x: P.l - 4, y: y + 3, class: 'cv-ax', 'text-anchor': 'end' }, svg); t.textContent = f(v, v < 1 ? 2 : 1);
  }
  const tx = el('text', { x: W - P.r, y: H - 6, class: 'cv-ax', 'text-anchor': 'end' }, svg); tx.textContent = `${c.x} ${f(x1, 0)}`;
  const t0 = el('text', { x: P.l, y: H - 6, class: 'cv-ax' }, svg); t0.textContent = f(x0, 0);
  c.series.forEach(s => {
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join('');
    el('path', { d, class: s.baseline ? 'cv-line base' : 'cv-line main', pathLength: 1 }, svg);
    if (!s.baseline) { const q = s.points[s.points.length - 1]; el('circle', { cx: X(q[0]), cy: Y(q[1]), r: 3, class: 'cv-dot' }, svg); }
  });
  return svg;
}

function dreamSteps(c) {
  const box = document.createElement('div'); box.className = 'cv-steps';
  c.steps.forEach(s => {
    const d = document.createElement('div'); d.className = 'cv-step ' + (s.accepted ? 'ok' : 'no');
    d.innerHTML = `<b>${s.generation}</b><span>${s.accepted ? 'accepted' : 'rolled back'}</span><code>${f(s.before, 4)} → ${f(s.after, 4)}</code>`;
    box.appendChild(d);
  });
  return box;
}

export function mountCurves(root, { keys = null, compact = false } = {}) {
  if (!root) return;
  fetch('assets/data/training_curves.json').then(r => r.json()).then(J => {
    const ks = keys || Object.keys(J.curves);
    for (const k of ks) {
      const c = J.curves[k]; if (!c) continue;
      const card = document.createElement('figure'); card.className = 'cv-card' + (c.pending ? ' pending' : '');
      const h = document.createElement('figcaption');
      if (c.pending) { h.innerHTML = `<b>${c.title}</b><span>${c.note}</span>`; card.appendChild(h); root.appendChild(card); continue; }
      const legend = c.series ? c.series.map(s => `<i class="${s.baseline ? 'base' : 'main'}"></i>${s.name}`).join(' ') : '<i class="main"></i>accepted <i class="base"></i>rolled back';
      h.innerHTML = `<b>${c.title}</b><span>${c.y}</span><em>${legend}</em>`;
      card.appendChild(h);
      card.appendChild(c.series ? lineChart(c) : dreamSteps(c));
      if (!compact && c.note) { const n = document.createElement('p'); n.className = 'cv-note'; n.textContent = c.note; card.appendChild(n); }
      root.appendChild(card);
    }
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const cards = [...root.querySelectorAll('.cv-card')];
    if (reduce || !('IntersectionObserver' in window) || document.documentElement.classList.contains('shot')) { cards.forEach(c => c.classList.add('in')); return; }
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.35 });
    cards.forEach(c => io.observe(c));
  }).catch(e => console.warn('curves', e));
}
