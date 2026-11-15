// Rover data blocks from assets/data/rover_stats.json (tools/export_rover_stats.py) and the 600 rover routes in
// assets/data/ghosts_rover_site23.* (tools/export_ghosts_web.py). Real simulation files only. Charts draw in once
// they enter the view (IntersectionObserver; no scroll listener). mountRoverStats(root, { full })
import { loadGhosts } from './ghostdata.js';

const NS = 'http://www.w3.org/2000/svg';
const sv = (t, a = {}, p) => { const e = document.createElementNS(NS, t); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (p) p.appendChild(e); return e; };
const h = (t, cls, html, p) => { const e = document.createElement(t); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (p) p.appendChild(e); return e; };
const fmt = (v, d = 0) => Number(v).toLocaleString('en', { minimumFractionDigits: d, maximumFractionDigits: d });
const PCOL = { naive: 'var(--peach)', cautious: 'var(--sage)', wait_light: 'var(--lavender)' };
const OCOL = { returned: [143, 207, 166], died_battery: [229, 128, 122], timeout: [168, 164, 154] };
const SITE = s => s.replace('lola5m_site', 'Site ');

function card(root, title, sub, cls = '') {
  const c = h('figure', 'rs-card ' + cls, null, root);
  h('figcaption', null, `<b>${title}</b>${sub ? `<span>${sub}</span>` : ''}`, c);
  return c;
}

// success (reached an ice-edge target AND returned) by driver and world
function successChart(c, S) {
  const W = 360, H = 200, P = { l: 30, r: 6, t: 22, b: 44 };
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'rs-svg', role: 'img', 'aria-label': 'Share of rovers that reached an ice-edge target and returned, by driver and world' }, c);
  const worlds = S.worlds, pols = S.policies;
  const vals = worlds.flatMap(w => pols.map(p => w.by_policy[p] ? w.by_policy[p].ice_and_returned / w.by_policy[p].n : 0));
  const top = Math.max(0.1, Math.ceil(Math.max(...vals) * 10) / 10);
  const Y = v => H - P.b - v / top * (H - P.t - P.b);
  for (let k = 0; k <= 2; k++) { const v = top * k / 2; sv('line', { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: 'rs-grid' }, svg); const t = sv('text', { x: P.l - 4, y: Y(v) + 3, class: 'rs-ax', 'text-anchor': 'end' }, svg); t.textContent = `${Math.round(v * 100)}%`; }
  const gw = (W - P.l - P.r) / worlds.length, bw = gw / (pols.length + 1.2);
  worlds.forEach((w, i) => {
    pols.forEach((p, j) => {
      const b = w.by_policy[p]; if (!b) return;
      const v = b.ice_and_returned / b.n, x = P.l + i * gw + (j + 0.6) * bw;
      const r = sv('rect', { x: x, y: Y(v), width: bw * 0.86, height: Math.max(0.5, H - P.b - Y(v)), class: 'rs-bar', style: `fill:${PCOL[p]};transform-origin:${x}px ${H - P.b}px` }, svg);
      sv('title', {}, r).textContent = `${SITE(w.site)} ${w.date} · ${S.policy_names[p]}: ${b.ice_and_returned} / ${b.n}`;
      const t = sv('text', { x: x + bw * 0.43, y: Y(v) - 3 - (j % 2) * 9, class: 'rs-ax rs-bl', 'text-anchor': 'middle' }, svg); t.textContent = `${b.ice_and_returned}/${b.n}`;
    });
    const t1 = sv('text', { x: P.l + (i + 0.5) * gw, y: H - P.b + 14, class: 'rs-ax2', 'text-anchor': 'middle' }, svg); t1.textContent = SITE(w.site);
    const t2 = sv('text', { x: P.l + (i + 0.5) * gw, y: H - P.b + 25, class: 'rs-ax', 'text-anchor': 'middle' }, svg); t2.textContent = w.date;
    const t3 = sv('text', { x: P.l + (i + 0.5) * gw, y: H - P.b + 35, class: 'rs-ax', 'text-anchor': 'middle' }, svg); t3.textContent = w.site.endsWith('23') ? 'unseen site' : 'training site';
  });
  legend(c, S);
}
function legend(c, S) { h('p', 'rs-legend', S.policies.map(p => `<i style="background:${PCOL[p]}"></i>${S.policy_names[p]}`).join(''), c); }

function distChart(c, S) {
  const D = S.distance, W = 360, H = 170, P = { l: 28, r: 6, t: 10, b: 26 };
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'rs-svg', role: 'img', 'aria-label': 'Histogram of distance driven per run, by driver' }, c);
  const nb = D.edges_m.length - 1, mx = Math.max(...S.policies.flatMap(p => D.hist[p]));
  const bw = (W - P.l - P.r) / nb, Y = v => H - P.b - v / mx * (H - P.t - P.b);
  for (let k = 0; k <= 2; k++) { const v = mx * k / 2; sv('line', { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: 'rs-grid' }, svg); const t = sv('text', { x: P.l - 4, y: Y(v) + 3, class: 'rs-ax', 'text-anchor': 'end' }, svg); t.textContent = fmt(v); }
  S.policies.forEach((p, j) => D.hist[p].forEach((v, i) => {
    const x = P.l + i * bw + j * bw / 3.3 + 1;
    sv('rect', { x, y: Y(v), width: bw / 3.6, height: Math.max(0, H - P.b - Y(v)), class: 'rs-bar', style: `fill:${PCOL[p]};transform-origin:${x}px ${H - P.b}px` }, svg);
  }));
  [0, Math.floor(nb / 2), nb].forEach(i => { const t = sv('text', { x: P.l + i * bw, y: H - 8, class: 'rs-ax', 'text-anchor': i ? (i === nb ? 'end' : 'middle') : 'start' }, svg); t.textContent = `${fmt(D.edges_m[i] / 1000, 1)} km`; });
  legend(c, S);
  h('p', 'rs-note', `Median distance: ${S.policies.map(p => `${S.policy_names[p]} ${fmt(D.median_m[p])} m`).join(' · ')}.`, c);
}

function lineChart(c, xs, series, { yMax = null, yLabel = '', xLabel = 'hour' } = {}) {
  const W = 360, H = 160, P = { l: 30, r: 6, t: 10, b: 22 };
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'rs-svg', role: 'img', 'aria-label': yLabel }, c);
  const x1 = Math.max(...xs.flat()), y1 = yMax ?? Math.max(...series.flatMap(s => s.y));
  const X = x => P.l + x / x1 * (W - P.l - P.r), Y = y => H - P.b - y / y1 * (H - P.t - P.b);
  for (let k = 0; k <= 2; k++) { const v = y1 * k / 2; sv('line', { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: 'rs-grid' }, svg); const t = sv('text', { x: P.l - 4, y: Y(v) + 3, class: 'rs-ax', 'text-anchor': 'end' }, svg); t.textContent = fmt(v, v < 2 && v > 0 ? 1 : 0); }
  const tx = sv('text', { x: W - P.r, y: H - 6, class: 'rs-ax', 'text-anchor': 'end' }, svg); tx.textContent = `${xLabel} ${fmt(x1)}`;
  series.forEach((s, i) => {
    const x = Array.isArray(xs[0]) ? xs[i] : xs;
    const d = s.y.map((v, k) => `${k ? 'L' : 'M'}${X(x[k]).toFixed(1)} ${Y(v).toFixed(1)}`).join('');
    sv('path', { d, class: 'rs-line', pathLength: 1, style: `stroke:${s.color};stroke-width:${s.w || 1.8};opacity:${s.o ?? 1}` }, svg);
  });
  return svg;
}

function routeMap(c) {
  const wrap = h('div', 'rs-map', null, c);
  const cv = h('canvas', null, null, wrap); cv.setAttribute('role', 'img');
  cv.setAttribute('aria-label', 'All 600 simulated rover routes on Site 23: green returned, red battery died, grey timed out');
  h('p', 'rs-legend', '<i style="background:var(--ok)"></i>returned <i style="background:var(--bad)"></i>battery died <i style="background:var(--fg-3)"></i>timed out <i class="ring"></i>lander', c);
  let drawn = false;
  c.__draw = () => {
    if (drawn) return; drawn = true;
    Promise.all([loadGhosts('rover', 'site23'), new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = 'assets/models3d/a_elev_tex.jpg'; })])
      .then(([G, im]) => {
        const w = wrap.clientWidth, dpr = Math.min(2, devicePixelRatio || 1); cv.width = w * dpr; cv.height = w * dpr;
        const g = cv.getContext('2d'); g.scale(dpr, dpr);
        const P = G.pts; let ext = 0;
        for (let i = 0; i < P.length; i += 4) ext = Math.max(ext, Math.abs(P[i]), Math.abs(P[i + 1]));
        ext = Math.ceil(ext * 1.08 / 100) * 100;                      // metres from the tile centre (= lander)
        const s = w / (2 * ext), X = x => w / 2 + x * s, Y = y => w / 2 - y * s;
        // terrain: the observed LOLA tile texture (5.12 km, row 0 = north), cropped to the routes, greyed and dimmed
        const k = im.width / 5120;
        g.filter = 'grayscale(1) brightness(0.5) contrast(1.1)';
        g.drawImage(im, (2560 - ext) * k, (2560 - ext) * k, 2 * ext * k, 2 * ext * k, 0, 0, w, w);
        g.filter = 'none';
        g.globalCompositeOperation = 'lighter'; g.lineWidth = 1; g.lineJoin = 'round';
        for (const r of [...G.runs].sort((a, b) => (a.outcome === 'returned') - (b.outcome === 'returned'))) {
          const [cr, cg, cb] = OCOL[r.outcome] || OCOL.timeout;
          g.strokeStyle = `rgba(${cr},${cg},${cb},${r.outcome === 'returned' ? 0.75 : 0.22})`;
          g.beginPath();
          for (let j = 0; j < r.n; j++) { const a = (r.start + j) * 4; j ? g.lineTo(X(P[a]), Y(P[a + 1])) : g.moveTo(X(P[a]), Y(P[a + 1])); }
          g.stroke();
        }
        g.globalCompositeOperation = 'source-over';
        const ice = G.head.ice_targets_xyz_m || [];
        g.fillStyle = '#9fc3d6';
        for (const q of ice) if (Math.abs(q[0]) < ext && Math.abs(q[1]) < ext) { g.beginPath(); g.arc(X(q[0]), Y(q[1]), 3, 0, 7); g.fill(); }
        g.strokeStyle = '#ffb547'; g.lineWidth = 2; g.beginPath(); g.arc(X(0), Y(0), 6, 0, 7); g.stroke();
        g.fillStyle = 'rgba(240,238,230,.75)'; g.font = '11px "JetBrains Mono", monospace';
        g.fillText(`${fmt(ext * 2 / 1000, 1)} km across · north up · blue = ice-edge targets`, 10, w - 10);
      }).catch(e => console.warn('rover map', e));
  };
}

export function mountRoverStats(root, { full = false } = {}) {
  if (!root) return;
  fetch('assets/data/rover_stats.json').then(r => r.json()).then(S => {
    const W0 = S.worlds[0], tot = W0.runs, R = S.shown_run;
    const best = S.policies.map(p => [p, W0.by_policy[p]]).filter(x => x[1]).sort((a, b) => b[1].ice_and_returned / b[1].n - a[1].ice_and_returned / a[1].n)[0];
    // big numbers
    const nums = h('div', 'rs-nums', null, root);
    const nb = (v, l) => h('div', 'rs-num', `<b>${v}</b><span>${l}</span>`, nums);
    nb(fmt(tot), `simulated drives on unseen ${SITE(W0.site)}, ${W0.date}`);
    nb(`${best[1].ice_and_returned} / ${best[1].n}`, `best rule driver (${S.policy_names[best[0]]}) reached an ice edge and came home`);
    nb(`${fmt(R.total_dist_m / 1000, 1)} km`, `driven by the shown run ${R.run_id} in ${fmt(R.duration_h)} h`);
    nb(`${fmt(R.hours_mostly_dark)} h`, `of ${fmt(R.hours_total)} with less than half the Sun visible (same run)`);
    const grid = h('div', 'rs-grid-wrap', null, root);
    successChart(card(grid, 'Reached an ice edge and returned', 'by driver and world'), S);
    routeMap(card(grid, 'All 600 routes, Site 23', 'on the observed LOLA terrain', 'rs-map-card'));
    {
      const c = card(grid, `Battery and sunlight · run ${R.run_id}`, `${R.policy} driver, ${R.date}, outcome ${R.outcome}`);
      lineChart(c, R.h, [{ y: R.battery_wh, color: 'var(--lavender)', w: 2 }, { y: R.sun_frac.map(v => v * R.battery_cap_wh), color: 'var(--sand)', w: 1.2, o: 0.7 }], { yMax: R.battery_cap_wh, yLabel: 'battery Wh and sunlit fraction' });
      h('p', 'rs-legend', `<i style="background:var(--lavender)"></i>battery, Wh (cap ${fmt(R.battery_cap_wh)}) <i style="background:var(--sand)"></i>sunlit fraction × ${fmt(R.battery_cap_wh)}`, c);
      h('p', 'rs-note', `Lowest charge ${fmt(R.min_battery_wh, 1)} Wh; ice edge reached at hour ${fmt(R.ice_reach_h)}; ${fmt(R.hours_dark)} h fully dark.`, c);
    }
    if (full) {
      distChart(card(grid, 'Distance driven per run', `${fmt(tot)} runs, ${SITE(W0.site)}`), S);
      const F = S.range_fan.runs;
      const c = card(grid, 'Distance from the lander over time', `${F.length} runs (${S.range_fan.note.split(',')[0]}); battery is recorded only for ${R.run_id}`);
      lineChart(c, F.map(r => r.h), F.map(r => { const [cr, cg, cb] = OCOL[r.outcome] || OCOL.timeout; return { y: r.range_m, color: `rgb(${cr},${cg},${cb})`, w: 1, o: r.outcome === 'returned' ? 0.9 : 0.3 }; }), { yLabel: 'distance from lander, m' });
      h('p', 'rs-legend', '<i style="background:var(--ok)"></i>returned <i style="background:var(--bad)"></i>battery died <i style="background:var(--fg-3)"></i>timed out', c);
      const d = S.driver, c2 = card(grid, 'Learned driver vs the rule drivers', `held-out Site 23, ${d.rovers_per_world} rovers each`, 'rs-driver');
      const rows = Object.entries(d.rule_drivers || {}).map(([k, v]) => `<tr><td>${k.replace(/_/g, ' ')}</td><td>${fmt(v.reached_ice * 100, 1)} %</td><td>${fmt(v.returned * 100, 1)} %</td><td>${fmt(v.died * 100, 1)} %</td></tr>`).join('');
      h('table', 'rs-table', `<thead><tr><th>driver</th><th>reached ice edge</th><th>returned</th><th>battery died</th></tr></thead><tbody><tr class="nt"><td>learned (vision transformer)</td><td colspan="3">not trained in this development run; trained later and FAILED (PPO 0.0 %, imitation 1.2 % returned on held-out worlds, see the rover post)</td></tr>${rows}</tbody>`, c2);
    }
    // draw-in on view
    const cards = [...root.querySelectorAll('.rs-card, .rs-nums')];
    const go = el => { el.classList.add('in'); el.__draw && el.__draw(); };
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window) || document.documentElement.classList.contains('shot')) { cards.forEach(go); return; }
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { go(e.target); io.unobserve(e.target); } }), { threshold: 0.25 });
    cards.forEach(c => io.observe(c));
  }).catch(e => console.warn('rover stats', e));
}
