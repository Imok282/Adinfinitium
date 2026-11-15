// "Replay a real landing": plays back recorded descents from assets/examples/pilot/pilot_examples.json.
// Three Site 11 (validation map) missions, each flown by the learned pilot and by classic guidance on the
// identical mission. Nothing is computed here beyond interpolating the recorded 1 s samples.
// Usage: mountReplay(el) -> lazy-loads when el scrolls into view.

const DATA_URL = 'assets/examples/pilot/pilot_examples.json';
const IMG_DIR = 'assets/examples/pilot/';
const LABEL = { nominal: 'Nominal', engine_out: 'Engine out', nav_drift: 'Nav drift' };
const BAR_FULL = 0.05; // bar end = 5 % of full thrust
const C = { pilot: '#d97757', classic: '#a8a49a', grid: 'rgba(240,238,230,.08)', ax: '#a8a49a', fg: '#f0eee6', bad: '#e5807a', ok: '#8fcfa6' };

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = (v, d = 1) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

// linear interpolation in a 1 s series; holds the last sample after the end
function at(s, key, t) {
  const n = s.t.length, last = n - 1;
  if (t >= s.t[last]) return s[key][last];
  if (t <= 0) return s[key][0];
  const i = Math.min(last - 1, Math.floor(t)), f = (t - s.t[i]) / (s.t[i + 1] - s.t[i]);
  const a = s[key][i], b = s[key][i + 1];
  return Array.isArray(a) ? a.map((v, k) => v + (b[k] - v) * f) : a + (b - a) * f;
}
const hdist = (s, t) => Math.hypot(at(s, 'dx_m', t), at(s, 'dy_m', t));

export function mountReplay(root) {
  if (!root) return;
  let started = false;
  const io = new IntersectionObserver(es => {
    if (es.some(e => e.isIntersecting) && !started) { started = true; io.disconnect(); init(root).catch(e => { console.warn('replay', e); root.textContent = 'Recorded descents could not be loaded.'; }); }
  }, { rootMargin: '300px 0px' });
  io.observe(root);
}

async function init(root) {
  const d = await (await fetch(DATA_URL)).json();
  const missions = d.missions;
  root.innerHTML = `
    <div class="rp-tabs" role="tablist" aria-label="Scenario">${missions.map((m, i) => `<button type="button" role="tab" class="rp-tab" data-i="${i}" aria-selected="${i === 0}">${LABEL[m.kind] || m.kind}</button>`).join('')}</div>
    <div class="rp-bar">
      <button type="button" class="rp-play" aria-label="Play">▶</button>
      <input type="range" class="rp-scrub" min="0" max="1000" step="1" value="0" aria-label="Replay time">
      <div class="rp-speed" role="group" aria-label="Playback speed"><button type="button" data-s="1">1×</button><button type="button" data-s="4" aria-pressed="true">4×</button></div>
      <b class="rp-clock">t = 0.0 s</b>
    </div>
    <div class="rp-main">
      <figure class="rp-view">
        <canvas class="rp-cv" role="img" aria-label="Side view of the descent: altitude against horizontal distance to the target, pilot path in terracotta, classic path in grey."></canvas>
        <figcaption><span><i class="k-p"></i>AI pilot</span><span><i class="k-c"></i>classic guidance</span><span class="rp-inj"></span></figcaption>
      </figure>
      <div class="rp-side">
        <table class="rp-read" aria-label="Live readouts">
          <thead><tr><th></th><th class="p">Pilot</th><th class="c">Classic</th></tr></thead>
          <tbody>
            <tr><th>Altitude</th><td data-r="p-alt"></td><td data-r="c-alt"></td></tr>
            <tr><th>Vertical speed</th><td data-r="p-vz"></td><td data-r="c-vz"></td></tr>
            <tr><th>Horizontal speed</th><td data-r="p-vh"></td><td data-r="c-vh"></td></tr>
            <tr><th>To target</th><td data-r="p-dh"></td><td data-r="c-dh"></td></tr>
            <tr><th>Fuel left</th><td data-r="p-fuel"></td><td data-r="c-fuel"></td></tr>
            <tr><th>Engines working</th><td data-r="p-eng"></td><td data-r="c-eng"></td></tr>
          </tbody>
        </table>
        <div class="rp-corr">
          <p class="rp-h">Pilot's correction <small>on top of the classic command, share of full thrust</small></p>
          <div class="rp-bars">${['x east', 'y north', 'z up'].map((n, k) => `<div class="rp-bx"><span>${n}</span><i class="rp-track"><i class="rp-fill" data-b="${k}"></i></i><b data-bv="${k}"></b></div>`).join('')}</div>
          <figure class="rp-cam"><img alt="" width="392" height="128" decoding="async"><figcaption class="rp-camcap"></figcaption></figure>
        </div>
      </div>
    </div>
    <p class="rp-out" aria-live="polite"></p>`;

  const $ = s => root.querySelector(s), $$ = s => [...root.querySelectorAll(s)];
  const cv = $('.rp-cv'), ctx = cv.getContext('2d'), scrub = $('.rp-scrub'), playB = $('.rp-play');
  const img = $('.rp-cam img'), camcap = $('.rp-camcap'), out = $('.rp-out'), clock = $('.rp-clock'), inj = $('.rp-inj');
  const R = Object.fromEntries($$('[data-r]').map(e => [e.dataset.r, e]));
  const bars = $$('.rp-fill'), barv = $$('[data-bv]');
  let mi = 0, m, T, t = 0, playing = false, speed = 4, last = 0, raf = 0, curImg = '';
  let W = 0, H = 0, sc;

  function pick(i) {
    mi = i; m = missions[i];
    T = Math.max(m.outcome.pilot.touchdown_time_s, m.outcome.classic.touchdown_time_s);
    const ps = m.pilot_series, cs = m.classic_series;
    const xs = [...ps.t.map((_, k) => Math.hypot(ps.dx_m[k], ps.dy_m[k])), ...cs.t.map((_, k) => Math.hypot(cs.dx_m[k], cs.dy_m[k]))];
    sc = { xmax: niceMax(Math.max(...xs)), ymax: niceMax(Math.max(...ps.alt_m, ...cs.alt_m)) };
    const j = m.injected;
    inj.textContent = j.engine_out_at_s != null ? `engine out at ${j.engine_out_at_s} s` : (j.nav_drift_bias_m_s.some(v => v) ? `nav drift ${j.nav_drift_bias_m_s.slice(0, 2).map(v => fmt(v, 2)).join(' / ')} m/s (x / y)` : 'no failure injected');
    $$('.rp-tab').forEach((b, k) => b.setAttribute('aria-selected', String(k === i)));
    setT(reduced() ? T : 0);
  }
  function niceMax(v) { const p = 10 ** Math.floor(Math.log10(v)); return Math.ceil(v / p * 2) / 2 * p; }

  function size() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }

  function draw() {
    if (!W || !m) return;
    const L = 44, Rm = 14, Tp = 14, B = 34, pw = W - L - Rm, ph = H - Tp - B;
    // square-root scales on both axes so the last few hundred metres (where the landing is decided) stay visible
    const X = x => L + Math.sqrt(Math.max(0, x) / sc.xmax) * pw, Y = y => Tp + ph - Math.sqrt(Math.max(0, y) / sc.ymax) * ph;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '10px "JetBrains Mono", monospace'; ctx.lineWidth = 1;
    for (const yv of [0, 10, 50, 200, 500, 1000, 2000, 3000].filter(v => v <= sc.ymax)) {
      ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(L, Y(yv) + .5); ctx.lineTo(L + pw, Y(yv) + .5); ctx.stroke();
      ctx.fillStyle = C.ax; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(yv + '', L - 6, Y(yv));
    }
    for (const xv of [0, 10, 50, 100, 200, 400, 600, 800, 1000].filter(v => v <= sc.xmax && (W > 480 || v !== 10))) {
      ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(X(xv) + .5, Tp); ctx.lineTo(X(xv) + .5, Tp + ph); ctx.stroke();
      ctx.fillStyle = C.ax; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(xv + '', X(xv), Tp + ph + 6);
    }
    ctx.fillStyle = C.ax; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('altitude, m (\u221A scale)', L + 6, Tp + 2);
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(W > 480 ? 'horizontal distance to target, m (\u221A scale)' : 'distance to target, m (\u221A)', L + pw, H - 1);
    // target
    ctx.fillStyle = C.fg; ctx.beginPath(); ctx.moveTo(X(0), Y(0) - 7); ctx.lineTo(X(0) - 5, Y(0)); ctx.lineTo(X(0) + 5, Y(0)); ctx.fill();
    const eo = m.injected.engine_out_at_s;
    for (const who of ['classic', 'pilot']) {
      const s = m[who + '_series'], col = C[who], end = Math.min(t, s.t[s.t.length - 1]);
      ctx.strokeStyle = col; ctx.lineWidth = who === 'pilot' ? 2.2 : 1.6; ctx.lineJoin = 'round';
      ctx.beginPath(); ctx.moveTo(X(hdist(s, 0)), Y(at(s, 'alt_m', 0)));
      for (let k = 1; k <= end; k++) ctx.lineTo(X(Math.hypot(s.dx_m[k], s.dy_m[k])), Y(s.alt_m[k]));
      ctx.lineTo(X(hdist(s, end)), Y(at(s, 'alt_m', end))); ctx.stroke();
      if (eo != null && t >= eo) {
        const ex = X(hdist(s, eo)), ey = Y(at(s, 'alt_m', eo));
        ctx.strokeStyle = C.bad; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(ex, ey, 5, 0, 7); ctx.stroke();
        if (who === 'pilot') { ctx.fillStyle = C.bad; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(`engine out ${eo} s`, ex + 8, ey - 6); }
      }
      lander(X(hdist(s, end)), Y(at(s, 'alt_m', end)), col, t >= m.outcome[who].touchdown_time_s);
    }
  }
  function lander(x, y, col, down) {
    ctx.save(); ctx.translate(x, y); ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(-5, -12); ctx.lineTo(5, -12); ctx.lineTo(6, -5); ctx.lineTo(-6, -5); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(-4, -5); ctx.lineTo(-8, 0); ctx.moveTo(4, -5); ctx.lineTo(8, 0); ctx.stroke();
    if (!down) { ctx.globalAlpha = .55; ctx.beginPath(); ctx.moveTo(-2, -4); ctx.lineTo(0, 2); ctx.lineTo(2, -4); ctx.fill(); }
    ctx.restore();
  }

  function readouts() {
    for (const [k, who] of [['p', 'pilot'], ['c', 'classic']]) {
      const s = m[who + '_series'], o = m.outcome[who], down = t >= o.touchdown_time_s;
      const v = [at(s, 'vx', t), at(s, 'vy', t), at(s, 'vz', t)];
      R[k + '-alt'].textContent = down ? 'landed' : at(s, 'alt_m', t).toFixed(0) + ' m';
      R[k + '-vz'].textContent = (down ? o.td_vertical_speed_m_s * -1 : v[2]).toFixed(1).replace('-', '−') + ' m/s';
      R[k + '-vh'].textContent = (down ? o.td_horizontal_speed_m_s : Math.hypot(v[0], v[1])).toFixed(1) + ' m/s';
      R[k + '-dh'].textContent = (down ? o.miss_distance_m : hdist(s, t)).toFixed(down ? 1 : 0) + ' m';
      R[k + '-fuel'].textContent = at(s, 'fuel_kg', t).toFixed(1) + ' kg';
      R[k + '-eng'].textContent = Math.round(at(s, 'engines_ok', Math.floor(t))) + '';
    }
    // pilot's correction = final command minus classic command, recorded per second
    const ps = m.pilot_series, pdown = t >= m.outcome.pilot.touchdown_time_s;
    const r = pdown ? [0, 0, 0] : at(ps, 'residual_applied', t);
    r.forEach((v, k) => {
      const f = Math.max(-1, Math.min(1, v / BAR_FULL));
      bars[k].style.transform = `scaleX(${Math.abs(f)})`; bars[k].style.transformOrigin = f < 0 ? 'right' : 'left';
      bars[k].style.left = f < 0 ? '0' : '50%';
      barv[k].textContent = (v >= 0 ? '+' : '−') + Math.abs(v * 100).toFixed(1) + ' %';
    });
    const cr = m.camera_crops.reduce((a, b) => Math.abs(b.t_s - t) < Math.abs(a.t_s - t) ? b : a);
    if (cr.file !== curImg) {
      curImg = cr.file; img.src = IMG_DIR + cr.file;
      img.alt = `What the pilot's camera saw at ${cr.t_s} s: slope, hazard belief and relative height of the ${Math.round(cr.ground_span_m)} m patch under the lander.`;
      camcap.textContent = `Camera at t = ${cr.t_s} s, ${cr.altitude_m.toFixed(0)} m up, ${Math.round(cr.ground_span_m)} m patch · slope · hazard belief · height`;
    }
    const line = who => { const o = m.outcome[who]; return t >= o.touchdown_time_s ? `${o.success ? 'landed' : o.outcome} at ${o.touchdown_time_s} s, miss ${o.miss_distance_m.toFixed(2)} m, ${o.fuel_used_kg.toFixed(1)} kg fuel used` : 'descending'; };
    out.innerHTML = `<span class="p"><b>Pilot</b> ${line('pilot')}</span><span class="c"><b>Classic</b> ${line('classic')}</span>`;
    clock.textContent = `t = ${t.toFixed(1)} s`;
  }

  function setT(v) { t = Math.max(0, Math.min(T, v)); scrub.value = String(Math.round(t / T * 1000)); draw(); readouts(); }
  function loop(now) {
    const dt = Math.min(.1, (now - last) / 1000); last = now;
    setT(t + dt * speed);
    if (t >= T) { setPlay(false); return; }
    raf = requestAnimationFrame(loop);
  }
  function setPlay(on) {
    playing = on; playB.textContent = on ? '❚❚' : '▶'; playB.setAttribute('aria-label', on ? 'Pause' : 'Play');
    cancelAnimationFrame(raf);
    if (on) { if (t >= T) setT(0); last = performance.now(); raf = requestAnimationFrame(loop); }
  }

  playB.addEventListener('click', () => setPlay(!playing));
  scrub.addEventListener('input', () => { setPlay(false); setT(scrub.value / 1000 * T); });
  $$('.rp-speed button').forEach(b => b.addEventListener('click', () => { speed = +b.dataset.s; $$('.rp-speed button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); }));
  $$('.rp-tab').forEach(b => b.addEventListener('click', () => { const was = playing; setPlay(false); pick(+b.dataset.i); if (was) setPlay(true); }));
  new ResizeObserver(size).observe(cv);
  // pause when off screen; autoplay once when first seen (not with reduced motion)
  let auto = !reduced();
  new IntersectionObserver(es => {
    const vis = es[0].isIntersecting;
    if (!vis && playing) setPlay(false);
    else if (vis && auto) { auto = false; setPlay(true); }
  }, { threshold: .35 }).observe(root);
  pick(0); size();
}
