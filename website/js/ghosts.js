// Touchdown beat (#landing): every simulated descent in assets/data/ghosts_lander_site06.* (MODELLED, made by
// python -m worldmodel.ghosts, packed by tools/export_ghosts_web.py) as thin additive lines in an oblique view.
// The tracks are drawn INCREMENTALLY into one accumulating canvas as the simulated clock advances with scroll
// (nothing is redrawn per frame; scrubbing back clears and redraws up to the new time once).
import { loadGhosts, fillCounts } from './ghostdata.js';

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (s, a, b) => clamp((s - a) / (b - a));
const ease = t => 1 - Math.pow(1 - t, 3);
const easeIO = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const r2 = x => Math.round(x * 100) / 100;
function set(el, prop, v) { const c = el.__g || (el.__g = {}); if (c[prop] === v) return; c[prop] = v; if (prop.startsWith('--')) el.style.setProperty(prop, v); else el.style[prop] = v; }

// legacy API (old main.js block): kept as a harmless no-op so an old caller never draws over the new beat
export function mountGhosts() { return { setProgress() {} }; }

const AMBER = '#ffb547';
const STAGE = { open_loop: 0, classic: 1, pilot: 2 };   // the order the policies fly in the beat
const AZ = -0.62, EL = 0.36;                 // view azimuth (rad, about the vertical) and elevation of the eye
const CA = Math.cos(AZ), SA = Math.sin(AZ), CE = Math.cos(EL), SE = Math.sin(EL);
// tile ENU (relative to the target) -> unscaled screen (u right, w up)
const proj = (dx, dy, dz) => [dx * CA - dy * SA, dz * CE + (dx * SA + dy * CA) * SE];

export function mount(section, ctx = {}) {
  const live = !ctx.reducedMotion && !document.body.classList.contains('no-webgl');
  section.classList.add(live ? 'td-live' : 'td-static');
  const fig = section.querySelector('.td-fig'), head = section.querySelector('.td-head');
  const cBg = section.querySelector('.td-bg'), cTr = section.querySelector('.td-tr'), cFg = section.querySelector('.td-fg');   // #ghosts = .td-fg (dots + chosen path, never faded)
  const dimEl = section.querySelector('.td-dim'), pills = section.querySelector('.td-legend');
  const nOk = section.querySelector('[data-td="ok"]'), nBad = section.querySelector('[data-td="bad"]');
  const ws = head ? splitWords(head.querySelector('h2')) : [];
  let D = null, sv = live ? 0 : 1, T = -1, W = 0, H = 0, dpr = 1, S = 1, OX = 0, OY = 0, XY = null, tMax = 1;
  let lastOk = '', lastBad = '', totOk = 0, totBad = 0, tSpan = 1, lastStage = '';
  const polTot = {}, lastPol = {};
  const stageEl = section.querySelector('[data-td="stage"]');

  loadGhosts('lander', 'site06').then(d => {
    // stages by policy (no invented checkpoints): no guidance first, then classic guidance, then the AI pilot.
    // Each policy's runs start when the previous stage's clock ends (times offset on a COPY of the shared data).
    let span = 1;
    for (const r of d.runs) span = Math.max(span, d.pts[(r.start + r.n - 1) * 4 + 3]);
    tSpan = span;
    const pts = d.pts.slice();
    for (const r of d.runs) { const o = (STAGE[r.policy] ?? 0) * span; for (let i = r.start; i < r.start + r.n; i++) pts[i * 4 + 3] += o; }
    D = { ...d, pts };
    for (const r of d.runs) { const q = polTot[r.policy] || (polTot[r.policy] = { n: 0, ok: 0 }); q.n++; if (r.ok) q.ok++; }
    fillCounts(section, d.head);
    for (const r of d.runs) r.ok ? totOk++ : totBad++;
    for (const r of D.runs) tMax = Math.max(tMax, D.pts[(r.start + r.n - 1) * 4 + 3]);
    layout(); render(true);
  }).catch(e => { console.warn('ghosts', e); section.classList.add('td-failed'); });

  // ---------- geometry
  function layout() {
    if (!D || !cTr) return;
    const w = cTr.clientWidth, h = cTr.clientHeight;
    if (!w || !h) return;
    dpr = Math.min(2, window.devicePixelRatio || 1); W = w; H = h;
    for (const c of [cBg, cTr, cFg]) { if (!c) continue; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const [tx, ty, tz] = D.head.target_xyz_m, P = D.pts, n = P.length / 4;
    XY = new Float32Array(n * 2);
    const us = [], wsv = [];
    for (let i = 0; i < n; i++) {
      const [u, v] = proj(P[i * 4] - tx, P[i * 4 + 1] - ty, P[i * 4 + 2] - tz);
      XY[i * 2] = u; XY[i * 2 + 1] = v; us.push(u); wsv.push(v);
    }
    // fit the central 98 % of points (a few open-loop runs fly far off; they leave the frame, as they leave the site)
    const q = (a, f) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(f * (s.length - 1))]; };
    const u0 = Math.min(q(us, 0.01), -900), u1 = Math.max(q(us, 0.99), 900), v0 = Math.min(q(wsv, 0.005), -500), v1 = q(wsv, 0.995);
    const padT = 34, padB = 8, hh = Math.max(120, h - padT - padB);   // keep clear of clock + pills
    S = Math.min(w * 0.94 / (u1 - u0), hh * 0.94 / (v1 - v0));
    OX = w / 2 - (u0 + u1) / 2 * S; OY = padT + hh * 0.5 + (v0 + v1) / 2 * S;
    for (let i = 0; i < n; i++) { XY[i * 2] = OX + XY[i * 2] * S; XY[i * 2 + 1] = OY - XY[i * 2 + 1] * S; }
    drawBg();
    T = -1;
  }
  const scr = (dx, dy, dz) => { const [u, v] = proj(dx, dy, dz); return [OX + u * S, OY - v * S]; };

  function drawBg() {
    if (!cBg) return;
    const g = cBg.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    g.lineWidth = 1; g.font = '10.5px "JetBrains Mono", monospace';
    // ground grid (250 m) inside 2.5 km, faint
    g.strokeStyle = 'rgba(240,238,230,.06)';
    g.beginPath();
    for (let k = -2500; k <= 2500; k += 250) {
      const L = Math.sqrt(Math.max(0, 2500 * 2500 - k * k));
      let [a, b] = scr(k, -L, 0), [c, d] = scr(k, L, 0); g.moveTo(a, b); g.lineTo(c, d);
      [a, b] = scr(-L, k, 0); [c, d] = scr(L, k, 0); g.moveTo(a, b); g.lineTo(c, d);
    }
    g.stroke();
    // range rings
    for (const r of [250, 500, 1000, 2000]) {
      g.strokeStyle = r === 2000 ? 'rgba(240,238,230,.2)' : 'rgba(240,238,230,.13)';
      g.setLineDash(r === 250 ? [2, 4] : []);
      g.beginPath();
      for (let i = 0; i <= 96; i++) { const a = i / 96 * Math.PI * 2, [x, y] = scr(r * Math.cos(a), r * Math.sin(a), 0); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      const [lx, ly] = scr(r * Math.cos(-2.2), r * Math.sin(-2.2), 0);
      g.fillStyle = 'rgba(207,203,192,.6)'; g.fillText(r >= 1000 ? `${r / 1000} km` : `${r} m`, lx + 4, ly + 12);
    }
    g.setLineDash([]);
    // altitude mast at the target: 0 .. 3 km
    const [bx, by] = scr(0, 0, 0), [mx, my] = scr(0, 0, 3000);
    g.strokeStyle = 'rgba(255,181,71,.28)'; g.setLineDash([3, 5]); g.beginPath(); g.moveTo(bx, by); g.lineTo(mx, my); g.stroke(); g.setLineDash([]);
    g.fillStyle = 'rgba(255,205,140,.7)';
    for (const z of [1000, 2000, 3000]) { const [x, y] = scr(0, 0, z); g.fillRect(x - 4, y - 0.5, 8, 1); g.fillText(`${z / 1000} km`, x + 8, y + 3.5); }
  }

  // ---------- tracks, incremental
  const STYLE = r => (r.ok ? [143, 207, 166] : r.policy === 'open_loop' ? [214, 122, 118] : [229, 128, 122]);   // --ok / --bad, desaturated
  function strokeRange(g, T0, T1) {
    const P = D.pts, nRuns = D.runs.length;
    const base = clamp(7 / Math.sqrt(nRuns), 0.09, 0.5) * clamp(Math.sqrt(W * H) / 950, 0.45, 1);   // small canvases: same glow, less burn-out
    g.globalCompositeOperation = 'lighter';
    g.lineCap = 'round'; g.lineJoin = 'round';
    for (const r of D.runs) {
      if (r.best) continue;
      const i0 = r.start, i1 = r.start + r.n - 1, tEnd = P[i1 * 4 + 3];
      if (tEnd <= T0 && T0 >= 0) continue;
      const [cr, cg, cb] = STYLE(r), a = base * (r.ok ? 0.62 : r.policy === 'open_loop' ? 0.85 : 1.6);
      const path = () => {
        g.beginPath();
        let started = false;
        for (let i = i0; i < i1; i++) {
          const ta = P[i * 4 + 3], tb = P[i * 4 + 7];
          if (tb <= T0) continue;
          if (ta >= T1) break;
          const fa = ta < T0 ? (T0 - ta) / (tb - ta) : 0, fb = tb > T1 ? (T1 - ta) / (tb - ta) : 1;
          const xa = XY[i * 2], ya = XY[i * 2 + 1], xb = XY[i * 2 + 2], yb = XY[i * 2 + 3];
          if (!started) { g.moveTo(xa + (xb - xa) * fa, ya + (yb - ya) * fa); started = true; }
          g.lineTo(xa + (xb - xa) * fb, ya + (yb - ya) * fb);
        }
        return started;
      };
      if (!path()) continue;
      g.strokeStyle = `rgba(${cr},${cg},${cb},${r2(a * 0.28)})`; g.lineWidth = 3.2; g.stroke();
      g.strokeStyle = `rgba(${cr},${cg},${cb},${r2(a)})`; g.lineWidth = 0.9; g.stroke();
      if (tEnd > T0 && tEnd <= T1) {
        g.fillStyle = `rgba(${cr},${cg},${cb},${r.ok ? 0.5 : 0.85})`;
        g.beginPath(); g.arc(XY[i1 * 2], XY[i1 * 2 + 1], r.ok ? 1.3 : 1.8, 0, 7); g.fill();
      }
    }
    g.globalCompositeOperation = 'source-over';
  }

  // dots: every simulated lander at simulated time Tn (moving = warm white; landed = settles green; crashed = flashes red, stops)
  function drawFg(Tn, fade) {
    if (!cFg) return;
    const g = cFg.getContext('2d'), P = D.pts;
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const [x, y] = scr(0, 0, 0);
    g.strokeStyle = AMBER; g.lineWidth = 1.5;
    g.beginPath(); g.ellipse(x, y, 9, 9 * SE, 0, 0, 7); g.stroke();
    const dk = 1 - fade * 0.9, flash = tSpan * 0.06;
    if (Tn > 0 && dk > 0.02) {
      g.globalCompositeOperation = 'lighter';
      for (const r of D.runs) {
        if (r.best) continue;
        const i1 = r.start + r.n - 1, tEnd = P[i1 * 4 + 3];
        let X, Y;
        if (Tn >= tEnd) {
          X = XY[i1 * 2]; Y = XY[i1 * 2 + 1];
          const [cr, cg, cb] = STYLE(r);
          if (!r.ok && Tn - tEnd < flash) {                       // crash flash: a red ring that shrinks
            const k = 1 - (Tn - tEnd) / flash;
            g.strokeStyle = `rgba(${cr},${cg},${cb},${r2(0.9 * k * dk)})`; g.lineWidth = 1.2;
            g.beginPath(); g.arc(X, Y, 2 + 7 * k, 0, 7); g.stroke();
          }
          g.fillStyle = `rgba(${cr},${cg},${cb},${r2((r.ok ? 0.55 : 0.9) * dk)})`;
          g.beginPath(); g.arc(X, Y, r.ok ? 1.4 : 1.9, 0, 7); g.fill();
          continue;
        }
        let i = r.start;
        while (i < i1 && P[i * 4 + 7] < Tn) i++;
        const ta = P[i * 4 + 3], tb = P[i * 4 + 7], f = tb > ta ? clamp((Tn - ta) / (tb - ta)) : 0;
        X = XY[i * 2] + (XY[i * 2 + 2] - XY[i * 2]) * f; Y = XY[i * 2 + 1] + (XY[i * 2 + 3] - XY[i * 2 + 1]) * f;
        g.fillStyle = `rgba(240,238,230,${r2(0.75 * dk)})`;
        g.beginPath(); g.arc(X, Y, 1.6, 0, 7); g.fill();
      }
      g.globalCompositeOperation = 'source-over';
    }
    // the chosen ("best") run, amber, on top; it stays when the others fade
    const b = D.runs.find(r => r.best);
    if (b && Tn > 0) {
      g.save(); g.shadowColor = 'rgba(255,181,71,.9)'; g.shadowBlur = 8 + 8 * fade;
      g.strokeStyle = AMBER; g.lineWidth = 2.2 + 1.2 * fade; g.lineCap = 'round'; g.lineJoin = 'round';
      g.beginPath();
      let px = 0, py = 0;
      for (let i = b.start; i < b.start + b.n; i++) {
        const t = P[i * 4 + 3];
        let X = XY[i * 2], Y = XY[i * 2 + 1];
        if (i > b.start && t > Tn) {
          const tp = P[i * 4 - 1], f = (Tn - tp) / (t - tp);
          X = px + (X - px) * f; Y = py + (Y - py) * f; g.lineTo(X, Y); px = X; py = Y; break;
        }
        i === b.start ? g.moveTo(X, Y) : g.lineTo(X, Y); px = X; py = Y;
      }
      g.stroke();
      g.fillStyle = '#ffe0a8'; g.beginPath(); g.arc(px, py, 3.2, 0, 7); g.fill();
      g.restore();
    }
  }

  let lastFade = -1;
  function render(force) {
    if (!D || !XY) return;
    const k = live ? easeIO(seg(sv, 0.08, 0.62)) : 1;
    const fade = live ? Math.round(ease(seg(sv, 0.68, 0.86)) * 100) / 100 : 0.5;   // static: trails half faded under the chosen path
    let Tn = k * tMax;
    Tn = Math.round(Tn / tMax * 400) / 400 * tMax;          // quantised: at most 400 distinct states
    if (!force && Tn === T && fade === lastFade) return;
    if (force || Tn !== T) {
      const g = cTr.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (Tn < T || T < 0 || force) { g.clearRect(0, 0, W, H); if (Tn > 0) strokeRange(g, -1, Tn); }
      else if (Tn > T) strokeRange(g, T, Tn);
    }
    T = Tn; lastFade = fade;
    set(cTr, 'opacity', String(r2(1 - fade * 0.93)));        // the trails fade (CSS opacity, no redraw)
    drawFg(Tn, fade);
    // live tally: a run counts once its dot has reached its end (landed / crashed); trails show the final colour,
    // so the tally is shown as "x of N" to read correctly while the dots are still falling
    let ok = 0, bad = 0;
    for (const r of D.runs) { if (D.pts[(r.start + r.n - 1) * 4 + 3] <= Tn) r.ok ? ok++ : bad++; }
    const pc = {};
    for (const r of D.runs) { const q = pc[r.policy] || (pc[r.policy] = 0); if (r.ok && D.pts[(r.start + r.n - 1) * 4 + 3] <= Tn) pc[r.policy] = q + 1; }
    for (const pol of Object.keys(polTot)) {
      const e = section.querySelector(`[data-td="ok-${pol}"]`); if (!e) continue;
      const v = `${(pc[pol] || 0).toLocaleString('en')} / ${polTot[pol].n.toLocaleString('en')}`;
      if (lastPol[pol] !== v) { e.textContent = v; lastPol[pol] = v; }
    }
    if (stageEl) {
      const si = Math.min(2, Math.floor(Tn / tSpan)), names = ['no guidance', 'classic guidance', 'AI pilot'];
      const v = live ? (Tn <= 0 ? 'Stage 1 of 3 · no guidance' : `Stage ${si + 1} of 3 · ${names[si]}`) : 'Three policies, same starts';
      if (v !== lastStage) { stageEl.textContent = v; lastStage = v; }
    }
    const okT = `${ok.toLocaleString('en')} / ${totOk.toLocaleString('en')}`, badT = `${bad.toLocaleString('en')} / ${totBad.toLocaleString('en')}`;
    if (okT !== lastOk && nOk) { nOk.textContent = okT; lastOk = okT; }
    if (badT !== lastBad && nBad) { nBad.textContent = badT; lastBad = badT; }
  }

  if (fig && 'ResizeObserver' in window) {
    let pend = false;
    new ResizeObserver(() => { if (pend) return; pend = true; requestAnimationFrame(() => { pend = false; layout(); render(true); }); }).observe(fig);
  }

  // when the section also holds content after the pinned part (.td-pin), map p to the pinned range only
  const pin = section.querySelector('.td-pin');
  let fac = 1;
  const measurePin = () => { if (!pin) return; const sh = section.offsetHeight, ph = pin.offsetHeight, vh = innerHeight; fac = ph > vh ? Math.max(1, (sh - vh) / (ph - vh)) : 1; };
  if (pin && 'ResizeObserver' in window) new ResizeObserver(measurePin).observe(section);
  measurePin();
  function update(p, vis) {
    if (!live) return;
    const target = clamp(p * fac);
    sv = ctx.shot ? target : sv + (target - sv) * 0.12;
    if (Math.abs(target - sv) < 1e-4) sv = target;
    // headline words, figure fly-in, dim-to-black, pills
    revealWords(ws, Math.max(seg(vis, 0.35, 0.95), seg(sv, 0, 0.04)));
    const f = ease(clamp(seg(vis, 0.5, 1) * 0.55 + seg(sv, 0, 0.1) * 0.45));
    if (fig) {
      set(fig, 'transform', f >= 1 ? 'none' : `translate3d(0,${r2((1 - f) * 60)}px,0) scale(${r2(0.9 + 0.1 * f)})`);
      set(fig, 'opacity', String(r2(clamp(f * 1.6))));
      set(fig, 'filter', f >= 0.98 ? 'none' : `blur(${r2((1 - f) * 8)}px)`);
    }
    if (dimEl) set(dimEl, "opacity", String(r2(clamp(vis * 1.4) * 0.85)));
    render(false);
  }
  return { update };
}

function splitWords(h) {
  if (!h) return [];
  const split = node => [...node.childNodes].forEach(c => {
    if (c.nodeType === 3) {
      const frag = document.createDocumentFragment();
      c.textContent.split(/(\s+)/).forEach(t => {
        if (!t) return;
        if (/^\s+$/.test(t)) { frag.append(t); return; }
        const w = document.createElement('span'); w.className = 'tdw'; w.textContent = t; frag.append(w);
      });
      c.replaceWith(frag);
    } else if (c.nodeType === 1 && c.hasAttribute('data-g')) {           // a number filled from the data: one word
      const w = document.createElement('span'); w.className = 'tdw'; c.replaceWith(w); w.append(c);
    } else if (c.nodeType === 1) split(c);
  });
  split(h);
  return [...h.querySelectorAll('.tdw')];
}
function revealWords(ws, k) {
  const n = ws.length;
  for (let i = 0; i < n; i++) {
    const e = ease(seg(k * (n + 2) / 3 - i / 3, 0, 1)), w = ws[i];
    set(w, 'opacity', String(r2(e)));
    set(w, 'transform', e >= 1 ? 'none' : `translate3d(0,${r2((1 - e) * 10)}px,0)`);
    set(w, 'filter', e >= 1 ? 'none' : `blur(${r2((1 - e) * 6)}px)`);
  }
}
