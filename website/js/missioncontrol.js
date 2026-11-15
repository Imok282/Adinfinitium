// Mission control: replays ONE simulated landing (assets/data/mission_control_lander.json, run pilot_02 from
// runs/ghosts_web/lander_lola5m_site06.json) and ONE simulated rover drive (assets/data/rover_route_site23.json,
// run cautious_15). Every number shown comes from those files; speed is derived from consecutive positions.
const $ = (s, r = document) => r.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const LANDER_SHARE = 0.35;                       // first 35 % of the replay timeline = lander, rest = rover
const STEPS = ['5 · Hover · scan · re-target', '6 · Terminal descent · touchdown', '7 · Lander becomes the base',
  '8 · Rover drives · scans · relays', '9 · Rover home before sunset'];   // mission steps from js/gauge.js (the replay covers 5–9)

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const fmt = (v, d = 0) => Number(v).toLocaleString('en', { minimumFractionDigits: d, maximumFractionDigits: d });

// a line chart: faint full series + bright part clipped to the replay cursor
function chart(svg, xs, ys, { color = 'var(--sage)', yMin = null, yMax = null } = {}) {
  const W = 400, H = 140, pad = 6;
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const lo = yMin ?? Math.min(...ys), hi = yMax ?? Math.max(...ys);
  const X = x => pad + (x - x0) / (x1 - x0 || 1) * (W - 2 * pad), Y = y => H - pad - (y - lo) / (hi - lo || 1) * (H - 2 * pad);
  const d = xs.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)} ${Y(ys[i]).toFixed(1)}`).join('');
  for (let k = 1; k < 4; k++) el('line', { x1: 0, x2: W, y1: k * H / 4, y2: k * H / 4, class: 'grid' }, svg);
  const id = 'c' + Math.random().toString(36).slice(2);
  const cp = el('clipPath', { id }, el('defs', {}, svg));
  const rect = el('rect', { x: 0, y: 0, width: 0, height: H }, cp);
  el('path', { d, class: 'ghost' }, svg);
  el('path', { d, class: 'live', 'clip-path': `url(#${id})`, style: `stroke:${color}` }, svg);
  const dot = el('circle', { r: 3.2, class: 'dot', style: `fill:${color}` }, svg);
  const lbl = el('text', { x: W - 4, y: 12, class: 'ax', 'text-anchor': 'end' }, svg); lbl.textContent = fmt(hi);
  const lbl0 = el('text', { x: W - 4, y: H - 4, class: 'ax', 'text-anchor': 'end' }, svg); lbl0.textContent = fmt(lo);
  return f => {                                   // f = 0..1 position in this series
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(f * (xs.length - 1))));
    rect.setAttribute('width', X(xs[i]).toFixed(1));
    dot.setAttribute('cx', X(xs[i]).toFixed(1)); dot.setAttribute('cy', Y(ys[i]).toFixed(1));
  };
}

function gauge(root, value, max) {
  const fg = $('.fg', root), b = $('b', root);
  return show => { fg.style.strokeDashoffset = String(show ? 1 - Math.min(1, value / max) : 1); b.textContent = show ? fmt(value, 1) : '—'; };
}

document.querySelectorAll('.mc-page .mc-bar, .mc-page .mc-card').forEach((t, i) => t.style.setProperty('--i', String(i)));   // intro stagger order

Promise.all([
  fetch('assets/data/mission_control_lander.json').then(r => r.json()),
  // real rover brain v3.1 traverse from the end-to-end test on site DM1 (tools/export_rover_e2e.py)
  fetch('assets/data/rover_route_dm1_v31.json').then(r => r.json()),
]).then(([L, R]) => build(L, R)).catch(e => { console.warn('mission control', e); document.body.classList.add('mc-failed'); });

function build(L, R) {
  // ---------- lander series
  const [tx, ty, tz] = L.target_xyz_m;
  const lt = L.t_s, P = L.xyz_m, n = lt.length;
  const alt = P.map(p => p[2] - tz);
  const dist = P.map(p => Math.hypot(p[0] - tx, p[1] - ty));
  const spd = P.map((p, i) => i ? Math.hypot(p[0] - P[i - 1][0], p[1] - P[i - 1][1], p[2] - P[i - 1][2]) / ((lt[i] - lt[i - 1]) || 1) : 0);
  spd[0] = spd[1] ?? 0;
  $('[data-mc="lander-id"]').textContent = `run ${L.run_id} · ${L.policy === 'pilot' ? 'AI pilot on classic guidance' : L.policy} · NASA LOLA Site 06`;
  const cAlt = chart($('[data-chart="alt"]'), lt, alt, { color: 'var(--sage)', yMin: 0 });
  const cSpd = chart($('[data-chart="spd"]'), lt, spd, { color: 'var(--peach)', yMin: 0 });
  // top-down ground track, metres from target
  const lm = $('[data-map="lander"]');
  const ext = Math.max(50, ...P.map(p => Math.max(Math.abs(p[0] - tx), Math.abs(p[1] - ty)))) * 1.15, k = 90 / ext;
  for (const r of [0.25, 0.5, 1]) el('circle', { cx: 0, cy: 0, r: 90 * r, class: 'ring' }, lm);
  const lTxt = el('text', { x: 0, y: 97, class: 'ax', 'text-anchor': 'middle' }, lm); lTxt.textContent = `ring = ${fmt(ext / 1.15)} m`;
  el('circle', { cx: 0, cy: 0, r: 4, class: 'tgt' }, lm);
  const trk = P.map(p => [(p[0] - tx) * k, -(p[1] - ty) * k]);
  el('polyline', { points: trk.map(q => q.join(',')).join(' '), class: 'ghost' }, lm);
  const lLive = el('polyline', { points: '', class: 'live amber' }, lm);
  const lDot = el('circle', { r: 3.5, class: 'dot amber' }, lm);
  // decision log: only from real fields of this run
  const log = [[0, `descent starts ${fmt(alt[0])} m above the target, ${fmt(dist[0])} m out`]];
  if (L.engine_failed) log.push([0, `engine-out injected in this run → outcome: ${L.outcome === 'ok' ? 'still landed' : L.outcome}`]);
  if (L.nav_drift) log.push([0, `navigation drift injected in this run → outcome: ${L.outcome === 'ok' ? 'still landed' : L.outcome}`]);
  if (L.camera_blind) log.push([0, `camera blinded in this run → outcome: ${L.outcome === 'ok' ? 'still landed' : L.outcome}`]);
  log.push([lt[n - 1], `touchdown ${fmt(L.touchdown_vz_m_s, 2)} m/s, ${fmt(L.target_dist_m, 2)} m from target · outcome: ${L.outcome}`]);
  log.push([lt[n - 1], `fuel used ${fmt(L.fuel_used_kg, 1)} kg`]);
  const logUl = $('[data-mc="log"]');
  const logLis = log.map(([t, s]) => { const li = document.createElement('li'); li.innerHTML = `<code>T+${fmt(t)} s</code> ${s}`; li.hidden = true; logUl.appendChild(li); return li; });
  const gFuel = gauge($('[data-g="fuel"]'), L.fuel_used_kg, 100);     // dial full scale 100 kg (display only)
  const gVz = gauge($('[data-g="vz"]'), L.touchdown_vz_m_s, 3);       // dial full scale 3 m/s (display only)

  // ---------- rover series
  const rt = R.t_s.map(s => s / 3600), rn = rt.length;
  const site = R.site || 'Site 23';
  $('[data-mc="rover-id"]').textContent = R.brain ? `${R.run_id} of ${R.fleet.rovers} · ${R.policy} · ${site} · ${R.date}` : `run ${R.run_id} · ${R.policy} planner · ${site} · ${R.date}`;
  const rh = $('.mc-route h3'); if (rh) rh.textContent = `Route over ${site} (hazard cells, ice-edge targets)`;
  const sh = $('[data-chart="sun"]')?.closest('.mc-card')?.querySelector('h3'); if (sh && R.sun_label) sh.textContent = R.sun_label[0].toUpperCase() + R.sun_label.slice(1);
  const fl = $('[data-mc="fleet"]');
  if (fl && R.fleet) fl.textContent = `Whole fleet on this world (${R.fleet.rovers} rovers, strict rule): ${R.fleet.strict_returned} returned, ${R.fleet.strict_lost} lost, ${R.fleet.alive_without_completed_trip} alive without a completed trip. The rover shown is the lowest-index rover that returned, not picked by eye.`;
  const cBatt = chart($('[data-chart="batt"]'), rt, R.battery_wh, { color: 'var(--lavender)', yMin: 0, yMax: R.battery_cap_wh });
  const cSun = chart($('[data-chart="sun"]'), rt, R.sun_frac, { color: 'var(--sand)', yMin: 0, yMax: 1 });
  const rm = $('[data-map="rover"]');
  // frame the route (plus margin); ice targets / hazard cells outside that window are left out of the map
  const px = R.points.map(p => p[0]), pz = R.points.map(p => p[2]);
  const m = Math.max(Math.max(...px) - Math.min(...px), Math.max(...pz) - Math.min(...pz)) * 0.25 + 40;
  const xs = [Math.min(...px) - m, Math.max(...px) + m], zs = [Math.min(...pz) - m, Math.max(...pz) + m];
  const rx0 = Math.min(...xs), rx1 = Math.max(...xs), rz0 = Math.min(...zs), rz1 = Math.max(...zs);
  const rs = 180 / Math.max(rx1 - rx0, rz1 - rz0, 1), ox = 100 - (rx0 + rx1) / 2 * rs, oz = 100 - (rz0 + rz1) / 2 * rs;
  const RX = x => ox + x * rs, RZ = z => oz + z * rs;                 // frame: x east, z south (screen down = south)
  const cell = Math.max(1.2, (R.hazard_cell_m || 5) * rs);
  for (const [x, z] of R.hazard_cells_xz) if (x >= rx0 && x <= rx1 && z >= rz0 && z <= rz1) el('rect', { x: RX(x) - cell / 2, y: RZ(z) - cell / 2, width: cell, height: cell, class: 'haz' }, rm);
  for (const p of R.ice_targets) if (p[0] >= rx0 && p[0] <= rx1 && p[2] >= rz0 && p[2] <= rz1) el('circle', { cx: RX(p[0]), cy: RZ(p[2]), r: 2.6, class: 'ice' }, rm);
  el('circle', { cx: RX(0), cy: RZ(0), r: 4, class: 'tgt' }, rm);
  const rpts = R.points.map(p => [RX(p[0]).toFixed(1), RZ(p[2]).toFixed(1)]);
  el('polyline', { points: rpts.map(q => q.join(',')).join(' '), class: 'ghost' }, rm);
  const rLive = el('polyline', { points: '', class: 'live amber' }, rm);
  const rDot = el('circle', { r: 3.5, class: 'dot amber' }, rm);

  // ---------- steps + clock
  const stepsOl = $('.mc-steps');
  const stepLis = STEPS.map(s => { const li = document.createElement('li'); li.textContent = s; stepsOl.appendChild(li); return li; });
  $('[data-mc="src"]').textContent = `${L.meta.source}; ${R.meta.source.replace('runs/ghosts/', 'runs/ghosts/')}`;

  const num = (k, v) => { const e = $(`[data-mc="${k}"]`); if (e.textContent !== v) e.textContent = v; };
  let last = -1;
  function render(f) {
    if (f === last) return; last = f;
    if (f <= LANDER_SHARE) {
      const q = f / LANDER_SHARE, i = Math.round(q * (n - 1));
      cAlt(q); cSpd(q); cBatt(0); cSun(0);
      num('alt', `${fmt(alt[i])} m`); num('spd', `${fmt(spd[i], 1)} m/s`); num('dist', `${fmt(dist[i])} m`);
      num('rdist', '—'); num('batt', '—'); num('sun', '—'); num('ice', '—');
      lLive.setAttribute('points', trk.slice(0, i + 1).map(p => p.join(',')).join(' '));
      lDot.setAttribute('cx', trk[i][0]); lDot.setAttribute('cy', trk[i][1]);
      rLive.setAttribute('points', ''); rDot.setAttribute('cx', RX(0)); rDot.setAttribute('cy', RZ(0));
      logLis.forEach((li, j) => { li.hidden = log[j][0] > lt[i]; });
      gFuel(i === n - 1); gVz(i === n - 1);
      const step = alt[i] > 30 ? 0 : 1;
      stepLis.forEach((li, j) => li.classList.toggle('on', j === step));
      $('.mc-phase').textContent = STEPS[step]; $('.mc-t').textContent = `T+${fmt(lt[i])} s`;
    } else {
      const q = (f - LANDER_SHARE) / (1 - LANDER_SHARE), i = Math.round(q * (rn - 1));
      cAlt(1); cSpd(1); cBatt(q); cSun(q);
      num('alt', '0 m'); num('spd', '0.0 m/s'); num('dist', `${fmt(L.target_dist_m, 2)} m`);
      lLive.setAttribute('points', trk.map(p => p.join(',')).join(' '));
      lDot.setAttribute('cx', trk[n - 1][0]); lDot.setAttribute('cy', trk[n - 1][1]);
      logLis.forEach(li => { li.hidden = false; }); gFuel(true); gVz(true);
      num('rdist', `${fmt(R.dist_m[i])} m`); num('batt', `${fmt(R.battery_wh[i] / R.battery_cap_wh * 100)} %`);
      num('sun', `${fmt(R.sun_frac[i] * 100)} %`); num('ice', i >= R.ice_reach_index ? 'reached' : 'not yet');
      rLive.setAttribute('points', rpts.slice(0, i + 1).map(p => p.join(',')).join(' '));
      rDot.setAttribute('cx', rpts[i][0]); rDot.setAttribute('cy', rpts[i][1]);
      const step = q < 0.02 ? 2 : (i >= rn - 1 && R.outcome === 'returned') ? 4 : 3;
      stepLis.forEach((li, j) => li.classList.toggle('on', j === step));
      $('.mc-phase').textContent = STEPS[step]; $('.mc-t').textContent = `rover hour ${fmt(rt[i])}`;
    }
  }

  // ---------- play / scrub
  const scrub = $('.mc-scrub'), btn = $('.mc-play');
  let f = reduced ? 1 : 0, playing = false, prev = 0;
  const DUR = 40;                                             // seconds for the whole replay
  const setPlaying = p => { playing = p; btn.innerHTML = p ? '&#10074;&#10074;' : '&#9654;'; btn.setAttribute('aria-label', p ? 'Pause' : 'Play'); if (p) { prev = 0; requestAnimationFrame(tick); } };
  function tick(now) {
    if (!playing) return;
    const dt = prev ? Math.min(0.1, (now - prev) / 1000) : 0; prev = now;
    f = Math.min(1, f + dt / DUR);
    scrub.value = String(Math.round(f * 1000)); render(f);
    if (f >= 1) { setPlaying(false); return; }
    requestAnimationFrame(tick);
  }
  btn.addEventListener('click', () => { if (!playing && f >= 1) f = 0; setPlaying(!playing); });
  scrub.addEventListener('input', () => { setPlaying(false); f = Number(scrub.value) / 1000; render(f); });
  scrub.value = String(Math.round(f * 1000));
  render(f);
  const qs = new URLSearchParams(location.search);
  if (qs.has('t')) { f = Math.min(1, Math.max(0, parseFloat(qs.get('t')) || 0)); scrub.value = String(Math.round(f * 1000)); render(f); }
  else if (!reduced) setPlaying(true);
}
