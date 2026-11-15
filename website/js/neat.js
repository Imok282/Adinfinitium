// "How the policy is trained": the TensorNEAT swarm (swarm/run_swarm.py) evolving the autonomy policy — a residual on classic guidance.
// mountNeat(el, { mode: 'beat' | 'card' | 'teaser' })
//   beat   : el is a tall section; a sticky stage is scrubbed by scroll (update(p) is also returned for an external loop)
//   card   : normal-height block; plays once when it scrolls into view (IntersectionObserver), one rAF only while playing
//   teaser : like card, smaller (no fitness axis labels)
// Reduced motion (or ?static) = final state, drawn once.
//
// DATA: assets/data/neat_training.json
// {
//   "placeholder": true,                    // true = sample values; shows "Training run results coming"
//   "meta": { provenance, source, frame, units, note },
//   "run": "overnight1",
//   "generations": [                         // one per runs/<run>/generation_XXX.json, in order
//     { "generation": 0, "best_fitness": 0.1, "mean_fitness": 0.05,
//       "species": 4,                        // number of species (null if unknown)
//       "best_nodes": 11, "best_conns": 10,  // size of that generation's champion genome
//       "population": 1000 }
//   ],
//   "champion": {                            // the final champion genome, drawn large
//     "generation": 39,
//     "nodes": [ { "id": 0, "type": "input" | "hidden" | "output", "label": "in 1" } ],
//     "conns": [ { "in": 0, "out": 11, "weight": 0.4 } ]          // weight sign -> colour, |weight| -> thickness
//   }
// }
// From a real run: python3 website/tools/export_neat.py runs/<run>   (reads generation_*.json: generation, best_score, mean_score,
// champion_genome {nodes, conns}; writes this file with "placeholder": false). Fitness is plotted as given (no rescaling).

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = t => t * t * (3 - 2 * t);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || '#ccc';
const rnd = i => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

export function mountNeat(el, opts = {}) {
  const mode = opts.mode || 'card';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches || new URLSearchParams(location.search).has('static');
  el.classList.add('neat', `neat-${mode}`);
  if (reduced) el.classList.add('neat-static');
  const cv = el.querySelector('canvas.neat-cv') || el.appendChild(Object.assign(document.createElement('canvas'), { className: 'neat-cv' }));
  cv.setAttribute('role', 'img');
  cv.setAttribute('aria-label', 'A population of small neural networks growing over generations, a rising fitness curve, and the best network drawn large.');
  const g = cv.getContext('2d');
  const note = el.querySelector('[data-neat-note]');
  const genLab = el.querySelector('[data-neat-gen]');
  let D = null, W = 1, H = 1, dpr = 1, last = -1, prog = (reduced || mode !== 'beat') ? 1 : 0, raf = 0, C = {};   // card/teaser show the final state until they play

  function colors() { C = { fg: css('--fg'), fg2: css('--fg-2'), fg3: css('--fg-3'), line: 'rgba(240,238,230,.14)', acc: css('--accent'),
    sage: css('--sage'), peach: css('--peach'), lav: css('--lavender'), sand: css('--sand') }; }
  function resize() {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = cv.clientWidth || 600; H = cv.clientHeight || 400;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); last = -1; draw(prog);
  }

  // champion layout: inputs left, hidden middle, outputs right
  function champLayout(x0, y0, w, h) {
    const N = D.champion.nodes, pos = {};
    const cols = { input: [], hidden: [], output: [] };
    N.forEach(n => (cols[n.type] || cols.hidden).push(n));
    const cx = { input: 0.06, hidden: 0.5, output: 0.94 };
    for (const t of ['input', 'hidden', 'output']) cols[t].forEach((n, i) => {
      const k = cols[t].length;
      const jitter = t === 'hidden' ? (rnd(n.id) - 0.5) * 0.3 : 0;
      pos[n.id] = [x0 + w * (cx[t] + jitter), y0 + h * (k === 1 ? 0.5 : 0.08 + 0.84 * i / (k - 1))];
    });
    return pos;
  }

  function miniNet(x, y, s, nodes, conns, seed, hi) {      // one population member: random small genome of the given size
    const pts = [];
    const nIn = 3, nOut = 2, nHid = Math.max(0, nodes - 11);
    for (let i = 0; i < nIn; i++) pts.push([x + s * 0.12, y + s * (0.2 + 0.6 * i / (nIn - 1))]);
    for (let i = 0; i < nOut; i++) pts.push([x + s * 0.88, y + s * (0.32 + 0.36 * i)]);
    for (let i = 0; i < Math.min(5, nHid); i++) pts.push([x + s * (0.35 + 0.3 * rnd(seed + i)), y + s * (0.15 + 0.7 * rnd(seed + 50 + i))]);
    const nc = Math.max(2, Math.round(conns / 4));
    g.lineWidth = hi ? 1.4 : 1;
    for (let c = 0; c < nc; c++) {
      const a = pts[Math.floor(rnd(seed + c * 3) * pts.length)], b = pts[Math.floor(rnd(seed + c * 3 + 1) * pts.length)];
      if (a === b) continue;
      g.strokeStyle = rnd(seed + c * 7) > 0.5 ? (hi ? C.acc : 'rgba(183,205,190,.55)') : (hi ? C.peach : 'rgba(232,185,160,.5)');
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    }
    g.fillStyle = hi ? C.fg : 'rgba(240,238,230,.75)';
    for (const p of pts) { g.beginPath(); g.arc(p[0], p[1], hi ? 2.4 : 1.7, 0, 7); g.fill(); }
  }

  function draw(p) {
    if (!D) return;
    const G = D.generations.length, gf = p * (G - 1), gi = Math.round(gf), gen = D.generations[gi];
    const key = Math.round(p * 1000);
    if (key === last) return;
    last = key;
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const narrow = W < 640, pad = 8;
    // regions
    const curveH = mode === 'teaser' ? H * 0.28 : narrow ? H * 0.24 : H * 0.3;
    const top = 0, bodyH = H - curveH - 18;
    const popW = narrow ? W : W * 0.5, champX = narrow ? 0 : W * 0.54, champW = narrow ? W : W * 0.46;
    const popH = narrow ? bodyH * 0.5 : bodyH, champY = narrow ? top + popH + 8 : top, champH = narrow ? bodyH * 0.5 - 8 : bodyH;

    // ---- population grid ----
    const cols = narrow ? 6 : 6, rows = narrow ? 3 : 4, cw = popW / cols, ch = popH / rows, s = Math.min(cw, ch) - 10;
    const bestCell = Math.floor(rnd(gi + 3) * cols * rows);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const k = r * cols + c, x = c * cw + (cw - s) / 2, y = top + r * ch + (ch - s) / 2, hi = k === bestCell;
      g.fillStyle = hi ? 'rgba(217,119,87,.14)' : 'rgba(240,238,230,.04)';
      g.beginPath(); g.roundRect ? g.roundRect(x - 3, y - 3, s + 6, s + 6, 10) : g.rect(x - 3, y - 3, s + 6, s + 6); g.fill();
      if (hi) { g.strokeStyle = C.acc; g.lineWidth = 1.5; g.stroke(); }
      const scale = hi ? 1 : 0.55 + 0.4 * rnd(k * 13 + gi);
      miniNet(x, y, s, Math.round(gen.best_nodes * (hi ? 1 : 0.85 + 0.15 * rnd(k))), gen.best_conns * scale, k * 101 + gi * 7, hi);
    }

    // ---- champion, drawn large; connections appear as the run progresses ----
    const pos = champLayout(champX + 20, champY + 18, champW - 40, champH - 36);
    const conns = D.champion.conns, nShow = Math.round(conns.length * clamp(gen.best_conns / Math.max(1, conns.length)));
    const used = new Set();
    for (let i = 0; i < nShow; i++) {
      const c = conns[i], a = pos[c.in], b = pos[c.out];
      if (!a || !b) continue;
      used.add(c.in); used.add(c.out);
      g.strokeStyle = c.weight >= 0 ? C.sage : C.peach;
      g.globalAlpha = 0.85;
      g.lineWidth = 0.8 + 2.6 * Math.min(1, Math.abs(c.weight));
      const mx = (a[0] + b[0]) / 2;
      g.beginPath(); g.moveTo(a[0], a[1]); g.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]); g.stroke();
    }
    g.globalAlpha = 1;
    g.font = `500 ${narrow ? 10 : 11.5}px Inter, sans-serif`;
    for (const n of D.champion.nodes) {
      const q = pos[n.id]; if (!q) continue;
      const on = n.type !== 'hidden' || used.has(n.id);
      if (!on) continue;
      g.fillStyle = n.type === 'output' ? C.acc : n.type === 'input' ? C.lav : C.fg;
      g.beginPath(); g.arc(q[0], q[1], n.type === 'hidden' ? 5 : 6, 0, 7); g.fill();
      if (n.type === 'output' && n.label) { g.fillStyle = C.fg2; g.textAlign = 'right'; g.fillText(n.label, q[0] - 12, q[1] - 10); }
    }
    g.textAlign = 'left';

    // ---- fitness curve ----
    const fy0 = H - curveH, fx0 = narrow ? 30 : 40, fx1 = W - pad, fh = curveH - 26;
    const fmax = Math.max(...D.generations.map(x => x.best_fitness)), fmin = Math.min(0, ...D.generations.map(x => x.mean_fitness));
    const X = i => fx0 + (fx1 - fx0) * i / Math.max(1, G - 1), Y = v => fy0 + 6 + fh * (1 - (v - fmin) / Math.max(1e-9, fmax - fmin));
    g.strokeStyle = C.line; g.lineWidth = 1;
    g.beginPath(); g.moveTo(fx0, fy0 + 6 + fh); g.lineTo(fx1, fy0 + 6 + fh); g.stroke();
    const line = (k, col, w) => {
      g.strokeStyle = col; g.lineWidth = w; g.beginPath();
      for (let i = 0; i <= gi; i++) { const v = D.generations[i][k]; i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); }
      g.stroke();
    };
    line('mean_fitness', 'rgba(200,195,224,.8)', 1.5);
    line('best_fitness', C.acc, 2.5);
    g.fillStyle = C.acc; g.beginPath(); g.arc(X(gi), Y(gen.best_fitness), 4, 0, 7); g.fill();
    g.fillStyle = C.fg3; g.font = '500 11px Inter, sans-serif';
    g.fillText('best', X(gi) + 8, Y(gen.best_fitness) + 4);
    g.fillText('mean', X(gi) + 8, Y(gen.mean_fitness) + 4);
    if (mode !== 'teaser') { g.fillText('fitness', 0, fy0 + 14); g.textAlign = 'right'; g.fillText(`generation ${gen.generation}`, fx1, fy0 + fh + 24 > H ? H - 2 : fy0 + fh + 22); g.textAlign = 'left'; }

    if (genLab) genLab.textContent = `Generation ${gen.generation} · ${gen.species ?? '—'} species · best genome ${gen.best_nodes} nodes, ${gen.best_conns} connections`;
  }

  function play(dur = 4200) {   // card / teaser: one playthrough, one rAF only while running
    if (reduced) { prog = 1; draw(1); return; }
    cancelAnimationFrame(raf);
    const t0 = performance.now();
    const step = now => { prog = ease(clamp((now - t0) / dur)); draw(prog); if (prog < 1 && visible) raf = requestAnimationFrame(step); else if (prog < 1) played = false; };
    raf = requestAnimationFrame(step);
  }
  let visible = false, played = false;

  fetch('assets/data/neat_training.json').then(r => r.json()).then(d => {
    D = d; colors();
    if (note) note.hidden = !d.placeholder;
    resize();
    if (mode === 'beat') dispatchEvent(new Event('scroll'));
    if (mode !== 'beat' && !reduced) {
      new IntersectionObserver(es => es.forEach(e => {
        visible = e.isIntersecting;
        if (visible && !played) { played = true; play(); }
      }), { threshold: 0.35 }).observe(el);
    } else if (reduced) draw(1);
  }).catch(e => console.warn('neat', e));
  addEventListener('resize', () => { colors(); resize(); });

  // beat mode: scrubbed by scroll (own throttled listener so it also works on pages without main.js)
  function update(p) { if (reduced) return; prog = clamp((p - 0.05) / 0.85); draw(prog); }
  if (mode === 'beat' && !reduced && opts.selfDrive !== false) {
    let tick = false;
    const onScroll = () => {
      if (tick) return; tick = true;
      requestAnimationFrame(() => {
        tick = false;
        const r = el.getBoundingClientRect();
        if (r.bottom < 0 || r.top > innerHeight) return;
        update(clamp(-r.top / Math.max(1, r.height - innerHeight)));
      });
    };
    addEventListener('scroll', onScroll, { passive: true });
    document.body.addEventListener('scroll', onScroll, { passive: true });   // ?scroll= screenshot mode scrolls <body>
    onScroll();
  }
  return { update, replay: () => { played = true; play(); } };
}
