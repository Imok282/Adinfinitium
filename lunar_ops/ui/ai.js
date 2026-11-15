/* Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in ai.html. */
/* AI view: the learned pilot's inputs, its transformer (real attention from scripts/live_mission.py) and its thrust command.
   Everything drawn comes from the live stream; nothing is decorative. Token order: 0 output (CLS), 1 sensors + classic command,
   2..65 the 8 x 8 camera patches (row-major, north up). */
(() => {
  const $ = (id) => document.getElementById(id), clamp = (v, a, b) => Math.min(b, Math.max(a, v)), lerp = (a, b, t) => a + (b - a) * t;
  const fmt = (x, d = 0) => (x == null || !isFinite(x) ? '–' : x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));
  const MAX_THRUST = 4000; // [N] 5 engines x 800 N (config/mission.yaml)
  const RESIDUAL_SCALE = 0.3; // [-] worldmodel/lander/pilot.py

  // fit the 1920 x 1080 stage to the window
  const stage = $('stage'); const fit = () => { const s = Math.min(innerWidth / 1920, innerHeight / 1080); stage.style.transform = `translate(${-960 * s}px, ${-540 * s}px) scale(${s})`; }; addEventListener('resize', fit); fit();

  // ---------------------------------------------------------------------------------------------------- sensors (the 21 inputs)
  const SENS = [ // name, value from obs, unit
    ['East to target', (o) => o[0] * 1000, 'm'], ['North to target', (o) => o[1] * 1000, 'm'], ['Altitude', (o) => o[2] * 1000, 'm'],
    ['East speed', (o) => o[3] * 50, 'm/s'], ['North speed', (o) => o[4] * 50, 'm/s'], ['Up speed', (o) => o[5] * 50, 'm/s'],
    ['Fuel', (o) => o[6] * 100, '%'], ['Engines working', (o) => o[7] * 5, ''], ['Mass', (o) => o[8] * 100, '%'], ['Hazard below', (o) => o[9], ''],
    ['Slope below', (o) => o[10] * 12, '°'], ['Camera working', (o) => o[11], ''], ['Mission time', (o) => o[12] * 240, 's'],
    ...['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'].map((d, i) => [`Unsafe 30 m ${d}`, (o) => o[13 + i], '']),
  ];
  const sens = $('sens'), rows = SENS.map(([nm]) => { const r = document.createElement('div'); r.className = 'row'; r.innerHTML = `<span class="nm">${nm}</span><span class="vl n">–</span><span class="bar"><i></i></span>`; sens.appendChild(r); return r; });
  function showSensors(obs) {
    SENS.forEach(([, f, u], i) => { const v = f(obs), r = rows[i]; r.children[1].textContent = u === '' ? fmt(v, Math.abs(v) < 10 && v % 1 ? 2 : 0) : `${fmt(v, Math.abs(v) >= 100 ? 0 : 1)} ${u}`;
      const b = clamp(obs[i], -1, 1), bar = r.children[2].firstChild; bar.className = b < 0 ? 'neg' : ''; bar.style.left = `${b >= 0 ? 50 : 50 + b * 50}%`; bar.style.width = `${Math.abs(b) * 50}%`; });
  }

  // ---------------------------------------------------------------------------------------------------- camera
  let camRGB = null, sal = new Float32Array(64);
  function decodeCam(b64, px) { const bin = atob(b64), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); camRGB = { a, px }; }
  function camCanvas(kind) { // 32 x 32 image: 'shade' (height hillshade + hazard tint) or a single channel
    const { a, px } = camRGB, c = document.createElement('canvas'); c.width = c.height = px; const g = c.getContext('2d'), im = g.createImageData(px, px);
    const H = (x, y) => a[(clamp(y, 0, px - 1) * px + clamp(x, 0, px - 1)) * 3 + 2] / 255 * 8 - 4; // relative height /50 m
    for (let y = 0; y < px; y++) for (let x = 0; x < px; x++) { const k = (y * px + x) * 3, o = (y * px + x) * 4; let r, gg, b;
      if (kind === 'shade') { const dx = (H(x + 1, y) - H(x - 1, y)) * 25, dy = (H(x, y + 1) - H(x, y - 1)) * 25, l = clamp(0.62 + (-dx * 0.7 + dy * 0.7) * 0.9, 0.08, 1), hz = a[k + 1] / 255;
        r = lerp(5 + 213 * l, 142, 0.6 * hz); gg = lerp(31 + 210 * l, 182, 0.6 * hz); b = lerp(32 + 190 * l, 155, 0.6 * hz); /* grey-green relief, hazard toward the accent */ }
      else { const v = a[k + kind] / 255; r = lerp(5, 218, v); gg = lerp(31, 241, v); b = lerp(32, 222, v); /* palette ramp */ }
      im.data[o] = r; im.data[o + 1] = gg; im.data[o + 2] = b; im.data[o + 3] = 255; }
    g.putImageData(im, 0, 0); return c;
  }
  function drawCam() {
    if (!camRGB) return; const cv = $('cam'), g = cv.getContext('2d'), S = cv.width; g.imageSmoothingEnabled = false; g.drawImage(camCanvas('shade'), 0, 0, S, S);
    const lo = Math.min(...sal), m = Math.max(...sal) - lo || 1e-9, cell = S / 8; for (let i = 0; i < 64; i++) { const v = (sal[i] - lo) / m; if (v < 0.35) continue; g.fillStyle = `rgba(142,182,155,${0.55 * (v - 0.35) / 0.65})`; g.fillRect((i % 8) * cell, Math.floor(i / 8) * cell, cell, cell); }
    for (let i = 0; i < 64; i++) { const v = (sal[i] - lo) / m; if (v > 0.8) { g.strokeStyle = 'rgba(218,241,222,.95)'; g.lineWidth = 2; g.strokeRect((i % 8) * cell + 1, Math.floor(i / 8) * cell + 1, cell - 2, cell - 2); } }
    g.strokeStyle = 'rgba(142,182,155,.9)'; g.lineWidth = 2; g.beginPath(); g.arc(S / 2, S / 2, 9, 0, 7); g.stroke(); // the lander is at the centre
    for (const [id, ch] of [['ch0', 0], ['ch1', 1]]) { const c = $(id), cg = c.getContext('2d'); cg.imageSmoothingEnabled = false; cg.drawImage(camCanvas(ch), 0, 0, c.width, c.height); }
  }

  // ---------------------------------------------------------------------------------------------------- network
  const net = $('net'), NW = 1060, NH = 760, DPR = 2; net.width = NW * DPR; net.height = NH * DPR; const ng = net.getContext('2d'); ng.scale(DPR, DPR);
  const bg = document.createElement('canvas'); bg.width = NW * DPR; bg.height = NH * DPR; const bgc = bg.getContext('2d'); bgc.scale(DPR, DPR);
  const GRID = { x0: 26, y0: 300, cell: 24 }, COLX = (l) => 300 + (l - 1) * 82, OUT = { x: 990, y: 0 };
  const tokY = (t) => (t < 2 ? 40 + t * 22 : 96 + (t - 2) * (600 / 63));
  OUT.y = tokY(0);
  function pos(l, t) { // l = 0 input, 1..8 output of each layer
    if (l > 0) return [COLX(l), tokY(t)];
    if (t === 0) return [GRID.x0 + GRID.cell * 4, tokY(0)]; if (t === 1) return [GRID.x0 + GRID.cell * 4, tokY(1) + 38];
    const p = t - 2; return [GRID.x0 + (p % 8 + 0.5) * GRID.cell, GRID.y0 + (Math.floor(p / 8) + 0.5) * GRID.cell];
  }
  const curve = (g, a, b) => { const mx = (a[0] + b[0]) / 2; g.moveTo(a[0], a[1]); g.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]); };
  let TR = null, chainMax = 1;
  function drawBackground(tr) {
    const g = bgc; g.clearRect(0, 0, NW, NH); g.globalCompositeOperation = 'lighter'; g.lineWidth = 0.7;
    for (let li = 0; li < tr.layers; li++) for (let q = 0; q < tr.tokens; q++) { const row = tr.top[li][q]; for (let j = 0; j < 4; j++) { const k = row[j], w = row[4 + j];
      g.strokeStyle = `rgba(142,182,155,${clamp(w * 0.8, 0.02, 0.32)})`; g.beginPath(); curve(g, pos(li, k), pos(li + 1, q)); g.stroke(); } }
    g.globalCompositeOperation = 'source-over';
    // input: camera patches with how much each one reaches the output (rollout)
    if (camRGB) { g.imageSmoothingEnabled = false; g.save(); g.beginPath(); g.roundRect(GRID.x0, GRID.y0, GRID.cell * 8, GRID.cell * 8, 8); g.clip(); g.globalAlpha = 0.75; g.drawImage(camCanvas('shade'), GRID.x0, GRID.y0, GRID.cell * 8, GRID.cell * 8); g.globalAlpha = 1;
      const lo = Math.min(...sal), m = Math.max(...sal) - lo || 1e-9; for (let i = 0; i < 64; i++) { const v = (sal[i] - lo) / m; g.fillStyle = `rgba(142,182,155,${0.6 * v * v})`; g.fillRect(GRID.x0 + (i % 8) * GRID.cell, GRID.y0 + Math.floor(i / 8) * GRID.cell, GRID.cell, GRID.cell); } g.restore(); }
    g.strokeStyle = 'rgba(142,182,155,.18)'; g.lineWidth = 1; for (let k = 0; k <= 8; k++) { g.beginPath(); g.moveTo(GRID.x0 + k * GRID.cell, GRID.y0); g.lineTo(GRID.x0 + k * GRID.cell, GRID.y0 + 8 * GRID.cell); g.moveTo(GRID.x0, GRID.y0 + k * GRID.cell); g.lineTo(GRID.x0 + 8 * GRID.cell, GRID.y0 + k * GRID.cell); g.stroke(); }
    // tokens: dot size = activity of that token in that layer
    for (let l = 1; l <= tr.layers; l++) { const act = tr.activity[l - 1], m = Math.max(...act, 1e-9);
      for (let t = 0; t < tr.tokens; t++) { const [x, y] = pos(l, t), v = act[t] / m; g.fillStyle = t < 2 ? `rgba(218,241,222,${0.35 + 0.6 * v})` : `rgba(218,241,222,${0.18 + 0.7 * v})`; g.beginPath(); g.arc(x, y, 1.3 + 2.6 * v, 0, 7); g.fill(); }
      // MLP neurons of the output token: the 32 strongest of the hidden width, as ticks under the column
      const sx = COLX(l) - 30, sy = 728; g.fillStyle = 'rgba(142,182,155,.08)'; g.fillRect(sx, sy, 60, 8);
      const tn = tr.top_neurons[l - 1], mv = Math.max(...tn.map((p) => Math.abs(p[1])), 1e-9); for (const [i, v] of tn) { g.fillStyle = v >= 0 ? `rgba(218,241,222,${0.4 + 0.6 * Math.abs(v) / mv})` : `rgba(218,241,222,${0.4 + 0.6 * Math.abs(v) / mv})`; g.fillRect(sx + i / tr.mlp_width * 60, sy, 1.4, 8); } }
    for (const [t, c] of [[0, '#DAF1DE'], [1, '#DAF1DE']]) { const [x, y] = pos(0, t); g.fillStyle = '#000000'; g.strokeStyle = c; g.lineWidth = 1.6; g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); g.stroke(); }
    g.font = '12px "Instrument Sans"'; g.fillStyle = 'rgba(255,255,255,.62)'; g.textAlign = 'left'; g.fillText('output token', pos(0, 0)[0] + 14, pos(0, 0)[1] + 4); g.fillText('sensors + classic', pos(0, 1)[0] + 14, pos(0, 1)[1] + 4); g.fillText('64 camera patches', GRID.x0, GRID.y0 - 10);
    g.textAlign = 'center'; for (let l = 1; l <= tr.layers; l++) g.fillText(`Layer ${l}`, COLX(l), 752);
    g.textAlign = 'right'; g.fillText('top neurons', COLX(1) - 38, 736);
  }
  function drawNet(wall) {
    ng.clearRect(0, 0, NW, NH); ng.drawImage(bg, 0, 0, NW, NH); if (!TR) return;
    ng.globalCompositeOperation = 'lighter';
    for (const [li, q, k, w] of TR.chain) { const a = pos(li, k), b = pos(li + 1, q), s = clamp(w / chainMax, 0.25, 1);
      for (const [lw, al] of [[7, 0.07], [3, 0.2], [1.4, 0.85]]) { ng.strokeStyle = `rgba(218,241,222,${al * s})`; ng.lineWidth = lw; ng.beginPath(); curve(ng, a, b); ng.stroke(); }
      ng.setLineDash([3, 14]); ng.lineDashOffset = -wall * 40; ng.strokeStyle = `rgba(218,241,222,${0.8 * s})`; ng.lineWidth = 1.6; ng.beginPath(); curve(ng, a, b); ng.stroke(); ng.setLineDash([]); }
    const lit = new Map(); for (const [li, q, k] of TR.chain) { lit.set(`${li + 1}:${q}`, 1); lit.set(`${li}:${k}`, 1); }
    for (const key of lit.keys()) { const [l, t] = key.split(':').map(Number), [x, y] = pos(l, t), r = 11, gr = ng.createRadialGradient(x, y, 0, x, y, r); gr.addColorStop(0, 'rgba(218,241,222,.95)'); gr.addColorStop(0.35, 'rgba(142,182,155,.45)'); gr.addColorStop(1, 'rgba(142,182,155,0)'); ng.fillStyle = gr; ng.beginPath(); ng.arc(x, y, r, 0, 7); ng.fill(); }
    // last layer's output token -> thrust
    const a = pos(TR.layers, 0); ng.strokeStyle = 'rgba(142,182,155,.8)'; ng.lineWidth = 2; ng.beginPath(); ng.moveTo(a[0], a[1]); ng.lineTo(OUT.x - 18, OUT.y); ng.stroke();
    ng.globalCompositeOperation = 'source-over';
    const gr = ng.createRadialGradient(OUT.x, OUT.y, 0, OUT.x, OUT.y, 40); gr.addColorStop(0, 'rgba(142,182,155,.45)'); gr.addColorStop(1, 'rgba(142,182,155,0)'); ng.fillStyle = gr; ng.beginPath(); ng.arc(OUT.x, OUT.y, 40, 0, 7); ng.fill();
    ng.fillStyle = '#000000'; ng.strokeStyle = '#DAF1DE'; ng.lineWidth = 2; ng.beginPath(); ng.arc(OUT.x, OUT.y, 16, 0, 7); ng.fill(); ng.stroke();
    ng.font = '12px "Instrument Sans"'; ng.fillStyle = 'rgba(255,255,255,.62)'; ng.textAlign = 'center'; ng.fillText('Thrust', OUT.x, OUT.y + 34);
  }

  // ---------------------------------------------------------------------------------------------------- does
  const H = { t: [], ai: [], cl: [] }, LOG = [];
  function chart() {
    const svg = $('chart'), X = 62, Y = 12, W = 254, Hh = 160, t1 = H.t.length ? H.t[H.t.length - 1] : 60, t0 = Math.max(0, t1 - 60);
    const xm = (t) => X + (t - t0) / Math.max(t1 - t0, 1) * W, ym = (v) => Y + Hh - clamp(v, 0, 100) / 100 * Hh;
    const path = (arr) => H.t.map((t, i) => (t < t0 ? '' : `${i && H.t[i - 1] >= t0 ? 'L' : 'M'}${xm(t).toFixed(1)},${ym(arr[i]).toFixed(1)}`)).join('');
    svg.innerHTML = [0, 50, 100].map((v) => `<line x1="${X}" x2="${X + W}" y1="${ym(v)}" y2="${ym(v)}" stroke="rgba(142,182,155,.08)"/><text x="${X - 10}" y="${ym(v) + 4}" text-anchor="end" font-size="11" fill="rgba(255,255,255,.38)" font-family="JetBrains Mono">${v}%</text>`).join('')
      + `<text x="${X}" y="${Y + Hh + 20}" font-size="11" fill="rgba(255,255,255,.38)" font-family="JetBrains Mono">${Math.round(t0)} s</text><text x="${X + W}" y="${Y + Hh + 20}" text-anchor="end" font-size="11" fill="rgba(255,255,255,.38)" font-family="JetBrains Mono">${Math.round(t1)} s</text>`
      + `<path d="${path(H.cl)}" fill="none" stroke="rgba(255,255,255,.62)" stroke-width="2" stroke-dasharray="6 6" opacity=".7"/><path d="${path(H.ai)}" fill="none" stroke="#DAF1DE" stroke-width="2.4" stroke-linejoin="round"/>`;
  }
  const norm = (v) => Math.hypot(v[0], v[1], v[2]);
  function onFrame(m) {
    if (m.lander) { const L = m.lander; $('s-t').textContent = `T+ ${String(Math.floor(m.t_s / 60)).padStart(2, '0')}:${String(Math.floor(m.t_s % 60)).padStart(2, '0')}`; $('s-alt').textContent = fmt(L.alt_m); $('s-vd').textContent = fmt(-L.vz_m_s, 1); $('s-eng').textContent = `${L.engines_ok} of 5`;
      $('s-src').innerHTML = L.source === 'pilot' ? '<span class="dot"></span>AI pilot' : '<span class="dot o"></span>Classic backup'; }
    const P = m.pilot; if (P) {
      const ai = norm(P.thrust_enu_n) / MAX_THRUST * 100, cl = norm(P.classic_enu_n) / MAX_THRUST * 100; $('thr').textContent = fmt(ai); $('thrc').textContent = `${fmt(cl)} %`;
      const r = P.residual.map((v) => RESIDUAL_SCALE * clamp(v, -1, 1) * MAX_THRUST); $('res').textContent = `east ${r[0] >= 0 ? '+' : '−'}${fmt(Math.abs(r[0]))} N · north ${r[1] >= 0 ? '+' : '−'}${fmt(Math.abs(r[1]))} N · up ${r[2] >= 0 ? '+' : '−'}${fmt(Math.abs(r[2]))} N`;
      $('lat').textContent = `${fmt(P.ms, 1)} ms per decision on the CPU · value estimate ${fmt(P.value, 2)}`;
      if (!H.t.length || P.t_s > H.t[H.t.length - 1]) { H.t.push(P.t_s); H.ai.push(ai); H.cl.push(cl); if (H.t.length > 600) for (const k in H) H[k].shift(); } else if (P.t_s < H.t[H.t.length - 1]) { for (const k in H) H[k].length = 0; LOG.length = 0; }
      chart(); showSensors(P.obs);
      const mp = (P.parameters / 1e6).toLocaleString('en-US', { maximumFractionDigits: 1 });
      $('prov').innerHTML = P.trained ? `Pilot network: trained checkpoint · ${mp} M parameters` : `<span style="color:var(--or)">Pilot network: UNTRAINED</span> · ${mp} M parameters · it flies exactly like the classic law until the trained checkpoint is loaded`;
      $('wait').style.opacity = '0'; }
    if (!m.lander && m.phase !== 'TOUCHDOWN') return; // events of the built-in rover simulator are not part of the live run
    for (const e of m.events || []) { LOG.unshift(e); if (LOG.length > 9) LOG.pop(); }
    if ((m.events || []).length) $('log').innerHTML = LOG.map((e) => `<div><span class="n">${String(Math.floor(e.t_s / 60)).padStart(2, '0')}:${String(Math.floor(e.t_s % 60)).padStart(2, '0')}</span><span>${e.msg.replace(/</g, '&lt;')}</span></div>`).join('');
  }
  function onAI(m) {
    TR = m.trace; decodeCam(m.camera_rgb_b64, m.camera_px); sal = Float32Array.from(TR.rollout.slice(2)); chainMax = Math.max(...TR.chain.map((e) => e[3]), 1e-9);
    $('netsub').textContent = `${TR.layers} layers × ${TR.heads} heads · ${TR.tokens} tokens: output, sensors, 64 camera patches · ${TR.mlp_width.toLocaleString('en-US')} neurons per layer`;
    drawBackground(TR); drawCam();
  }
  let wall = 0, last = performance.now(); const loop = (now) => { wall += (now - last) / 1000; last = now; drawNet(wall); requestAnimationFrame(loop); }; requestAnimationFrame(loop);
  function connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ui`);
    ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === 'ai') onAI(m); else if (m.type === 'frame') onFrame(m); };
    ws.onclose = () => setTimeout(connect, 1500);
  }
  connect();
})();
