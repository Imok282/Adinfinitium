// Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in index.html.
// Charts drawn by OpenUI (vendor/openui, MIT, https://github.com/thesysdev/openui): its React chart components, with their
// built-in animation, are fed from the mission data through OpenUI Lang, a small declarative language.
(function () {
'use strict';
const MC = (window.MC = window.MC || {}), P = MC.panels, q = (s) => JSON.stringify(String(s)), r2 = (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : 0);
const lim = () => MC.S.W?.limits || { battery_capacity_wh: 350, solar_peak_w: 50, slope_limit_deg: 20 };
const wait = (msg) => `root = Card([t])\nt = TextContent(${q(msg)}, "small")`;
// Average of an array over n equal slices of [i0, i1] (NaN slices become 0).
function slices(arr, i0, i1, n) { const out = [], step = (i1 - i0 + 1) / n; for (let k = 0; k < n; k++) { let s = 0, c = 0; for (let i = Math.floor(i0 + k * step); i < Math.floor(i0 + (k + 1) * step) || (c === 0 && i <= i1 && i < Math.floor(i0 + (k + 1) * step) + 1); i++) { const v = arr[i]; if (Number.isFinite(v)) { s += v; c++; } } out.push(c ? s / c : 0); } return out; }
function span(key) { const a = MC.cols[key]; let i0 = -1, i1 = -1; for (let i = 0; i <= MC.cur && i < a.length; i++) if (Number.isFinite(a[i])) { if (i0 < 0) i0 = i; i1 = i; } return i0 >= 0 && i1 > i0 ? [i0, i1] : null; }
const labels = (i0, i1, n) => { const t = MC.cols.t, out = [], step = (i1 - i0) / n, short = t[i1] - t[i0] < 3600, mmss = (s) => `${String(Math.floor(s % 3600 / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`; for (let k = 0; k < n; k++) { const s = t[Math.floor(i0 + k * step)]; out.push(short ? mmss(s) : MC.fmtHMS(s).slice(0, 5)); } return out; };
const arr = (a) => '[' + a.map(r2).join(',') + ']', strs = (a) => '[' + a.map(q).join(',') + ']';
const ch = (p) => Math.max(70, (p.el.clientHeight || 200) - 62);

const oui = (id, title, src, lang) => { P[id] = { title, src, oui: true, lang }; };
oui('ouiPower', 'Solar vs load', 'feed', (p) => { const s = span('solar'); if (!s) return wait('Rover power appears after landing.'); const n = 14, L = labels(s[0], s[1], n); return `root = Card([c])\nc = LineChart(${strs(L)}, [a, b], "natural", "time since landing (h:min)", "W", ${ch(p)})\na = Series("Solar W", ${arr(slices(MC.cols.solar, s[0], s[1], n))})\nb = Series("Load W", ${arr(slices(MC.cols.load, s[0], s[1], n))})`; });
oui('ouiBattery', 'Battery', 'feed', (p) => { const s = span('soc'); if (!s) return wait('Battery appears after landing.'); const n = 14, L = labels(s[0], s[1], n); return `root = Card([c])\nc = AreaChart(${strs(L)}, [a, b], "natural", "time since landing (h:min)", "%", ${ch(p)})\na = Series("Battery %", ${arr(slices(MC.cols.soc.map((v) => v * 100), s[0], s[1], n))})\nb = Series("Save-power level", ${arr(new Array(n).fill(20))})`; });
oui('ouiThermal', 'Temperatures', 'simulated', (p) => { const n = 14, i1 = MC.cur; if (i1 < 3) return wait('Waiting for data…'); const L = labels(0, i1, n), ser = MC.THERMAL.map((t, k) => `s${k} = Series(${q(t.name + ' °C')}, ${arr(slices(MC.sim.temp[t.k], 0, i1, n))})`); return `root = Card([c])\nc = LineChart(${strs(L)}, [${MC.THERMAL.map((_, k) => 's' + k).join(', ')}], "natural", "time since start (h:min)", "°C", ${ch(p)})\n${ser.join('\n')}`; });
oui('ouiEnergy', 'Energy by subsystem', 'simulated split', (p) => { const s = span('load'); if (!s) return wait('Rover energy appears after landing.'); const n = 8, c = MC.cols, L = labels(s[0], s[1], n), cats = Object.keys(MC.SPLIT.drive), acc = cats.map(() => new Array(n).fill(0)), step = (s[1] - s[0] + 1) / n;
  for (let i = s[0]; i <= s[1]; i++) { const k = Math.min(n - 1, Math.floor((i - s[0]) / step)), dt = i ? Math.max(0, c.t[i] - c.t[i - 1]) : 0, sp = MC.SPLIT[MC.MODES[c.mode[i]] || 'hold']; if (!Number.isFinite(c.load[i])) continue; cats.forEach((nm, j) => { acc[j][k] += c.load[i] * sp[nm] * dt / 3600; }); }
  return `root = Card([c])\nc = BarChart(${strs(L)}, [${cats.map((_, j) => 's' + j).join(', ')}], "stacked", "time since landing (h:min)", "Wh", ${ch(p)})\n${cats.map((nm, j) => `s${j} = Series(${q(nm + ' Wh')}, ${arr(acc[j])})`).join('\n')}`; });
oui('ouiSplit', 'Power split now', 'simulated split', () => { const load = MC.at('load'); if (!Number.isFinite(load)) return wait('Rover power appears after landing.'); const sp = MC.SPLIT[MC.MODES[MC.at('mode')] || 'hold'], cats = Object.keys(sp).filter((k) => sp[k] > 0); return `root = Card([c])\nc = PieChart(${strs(cats)}, ${arr(cats.map((k) => load * sp[k]))}, "donut")`; });
oui('ouiRadar', 'Health check (100 = best)', 'derived', () => { if (!Number.isFinite(MC.at('soc'))) return wait('Health check appears after landing.'); const L = lim(), lk = MC.link(), tb = MC.sim.temp.battery[MC.cur];
  const names = ['Battery', 'Solar input', 'Earth link', 'Thermal margin', 'Slope margin', 'Hazard margin'], vals = [MC.at('soc') * 100, MC.at('solar') / L.solar_peak_w * 100, lk.margin / 14 * 100, (tb + 10) / 40 * 100, (1 - MC.at('slope') / L.slope_limit_deg) * 100, (1 - MC.at('haz')) * 100].map((v) => Math.max(0, Math.min(100, v)));
  return `root = Card([c])\nc = RadarChart(${strs(names)}, [s])\ns = Series("Health", ${arr(vals)})`; });
oui('ouiTargets', 'Round trip to each target', 'derived', (p) => { const W = MC.S.W; if (!W || MC.cur < 0) return wait('Waiting…'); const x = MC.lastFinite('x'), y = MC.lastFinite('y'), px = Number.isFinite(x) ? x : W.start.x_m, py = Number.isFinite(y) ? y : W.start.y_m, used = MC.lastFinite('used'), dist = MC.lastFinite('dist'), k = Number.isFinite(used) && dist > 50 ? Math.min(3, Math.max(0.1, used / dist)) : 0.35, usable = Math.max(0, (MC.lastFinite('wh') || 0) - 0.2 * lim().battery_capacity_wh);
  const T = (W.targets || []).map((t) => ({ id: t.id, rt: 2 * Math.hypot(t.x_m - px, t.y_m - py) * k })).sort((a, b) => a.rt - b.rt).slice(0, 5); return `root = Card([c, n])\nc = HorizontalBarChart(${strs(T.map((t) => t.id))}, [a], "grouped", "Wh", "")\na = Series("Round trip Wh (estimate)", ${arr(T.map((t) => t.rt))})\nn = TextContent(${q('The 5 nearest targets. Usable energy now: ' + usable.toFixed(0) + ' Wh (battery above the 20 % floor).')}, "small")`; });
oui('ouiScatter', 'Targets: distance vs ice score', 'derived', () => { const W = MC.S.W, f = MC.S.last; if (!W || !f) return wait('Waiting…'); const x = MC.lastFinite('x'), y = MC.lastFinite('y'), px = Number.isFinite(x) ? x : W.start.x_m, py = Number.isFinite(y) ? y : W.start.y_m, open = [], done = [];
  for (const t of W.targets || []) (f.targets?.find((q2) => q2.id === t.id)?.sampled ? done : open).push(`Point(${r2(Math.hypot(t.x_m - px, t.y_m - py))}, ${r2(t.ice_score)}, ${r2(t.depth_m ?? 40)})`);
  const sets = [open.length ? `o = ScatterSeries("Open", [${open.join(', ')}])` : '', done.length ? `d = ScatterSeries("Sampled", [${done.join(', ')}])` : ''].filter(Boolean); return `root = Card([c])\nc = ScatterChart([${[open.length ? 'o' : '', done.length ? 'd' : ''].filter(Boolean).join(', ')}], "distance from rover (m)", "ice score")\n${sets.join('\n')}`; });
oui('ouiLander', 'Descent altitude', 'feed', (p) => { const s = span('alt'); if (!s) return wait('Descent altitude appears during the landing.'); const n = 16, L = labels(s[0], s[1], n); return `root = Card([c])\nc = AreaChart(${strs(L)}, [a], "natural", "time (min:s)", "m", ${ch(p)})\na = Series("Altitude m", ${arr(slices(MC.cols.alt, s[0], s[1], n))})`; });
oui('ouiEngines', 'Engine throttle', 'feed', () => { const eng = MC.lastFinite('eng'), th = MC.lastFinite('thrust'); if (!Number.isFinite(eng)) return wait('Engines appear during the landing.'); const per = th / Math.max(eng, 1) / 800 * 100, v = [0, 1, 2, 3, 4].map((i) => (eng < 5 && [1, 3].includes(i) ? 0 : per)); return `root = Card([c])\nc = RadialChart(["E1","E2","E3","E4","E5"], ${arr(v)})`; });
oui('ouiTime', 'Time in each activity', 'derived', () => { const c = MC.cols, cnt = [0, 0, 0, 0]; for (let i = 0; i <= MC.cur; i++) if (c.mode[i] >= 0) cnt[c.mode[i]]++; if (cnt.every((v) => v === 0)) return wait('Activity appears after landing.'); return `root = Card([c])\nc = SingleStackedBarChart(["Driving","Waiting","Sleeping","Drilling"], ${arr(cnt)})`; });

// OpenUI replacements for the canvas telemetry stacks (one measure per chart: never two y-scales in one chart).
// k charts in one card: side by side when the panel is wide, stacked when it is tall, else OpenUI tabs (one chart at a time,
// the operator picks). Each chart keeps ~86 px for its legend and x axis.
const grid = (p, k) => { const W = p.el.clientWidth || 600, H = ch(p), colH = Math.floor((H + 62) / k) - 86, mode = k === 1 ? 'one' : W / k >= 240 ? 'row' : colH >= 90 ? 'col' : 'tabs';
  return { h: mode === 'col' ? colH : mode === 'tabs' ? Math.max(70, H - 44) : H,
    root: (ids, names = ids) => (mode === 'row' ? `root = Card([g])\ng = Stack([${ids.join(', ')}], "row", "s")`
      : mode === 'tabs' ? `root = Card([g])\ng = Tabs([${ids.map((_, i) => `t${i}`).join(', ')}])\n` + ids.map((id, i) => `t${i} = TabItem(${q(id)}, ${q(names[i])}, [${id}])`).join('\n')
      : `root = Card([${ids.join(', ')}])`) }; };
function series1(key, scale = 1) { const s = span(key); return s ? { s, v: (n) => slices(MC.cols[key].map((v) => v * scale), s[0], s[1], n) } : null; }
oui('ouiTraces', 'Rover telemetry', 'feed', (p) => {
  const sp = series1('speed', 100), bt = series1('soc', 100), so = series1('solar'), ld = series1('load'); if (!sp && !bt) return wait('Rover telemetry appears after landing.');
  const ref = (sp || bt).s, n = 16, L = labels(ref[0], ref[1], n), G = grid(p, 3), h = G.h, out = [G.root(['a', 'b', 'c'], ['Speed', 'Battery', 'Power'])];
  out.push(`a = LineChart(${strs(L)}, [sa], "natural", "time since landing", "cm/s", ${h})`, `sa = Series("Speed cm/s", ${arr(sp ? sp.v(n) : [])})`);
  out.push(`b = AreaChart(${strs(L)}, [sb], "natural", "time since landing", "%", ${h})`, `sb = Series("Battery %", ${arr(bt ? bt.v(n) : [])})`);
  out.push(`c = LineChart(${strs(L)}, [sc, sd], "natural", "time since landing", "W", ${h})`, `sc = Series("Solar W", ${arr(so ? so.v(n) : [])})`, `sd = Series("Load W", ${arr(ld ? ld.v(n) : [])})`);
  return out.join('\n');
});
oui('ouiDescent', 'Landing telemetry', 'feed', (p) => {
  const al = series1('alt'); if (!al) return wait('Landing telemetry appears during the descent.'); const n = 18, L = labels(al.s[0], al.s[1], n), G = grid(p, 3), h = G.h;
  const vz = series1('vz'), vh = series1('vh'), fu = series1('fuel');
  return [G.root(['a', 'b', 'c'], ['Altitude', 'Speeds', 'Propellant']), `a = AreaChart(${strs(L)}, [sa], "natural", "time (min:s)", "m", ${h})`, `sa = Series("Altitude m", ${arr(al.v(n))})`,
    `b = LineChart(${strs(L)}, [sb, sc], "natural", "time (min:s)", "m/s", ${h})`, `sb = Series("Vertical speed m/s", ${arr(vz ? vz.v(n) : [])})`, `sc = Series("Sideways speed m/s", ${arr(vh ? vh.v(n) : [])})`,
    `c = LineChart(${strs(L)}, [sd], "natural", "time (min:s)", "kg", ${h})`, `sd = Series("Propellant kg", ${arr(fu ? fu.v(n) : [])})`].join('\n');
});
oui('ouiOrbitTr', 'Orbit telemetry', 'feed', (p) => {
  const oa = series1('orbAlt'), os = series1('orbSpd'); if (!oa) return wait('No orbit data in this feed.'); const n = 18, L = labels(oa.s[0], oa.s[1], n), G = grid(p, 2), h = G.h;
  return [G.root(['a', 'b'], ['Height', 'Speed']), `a = AreaChart(${strs(L)}, [sa], "natural", "mission time (h:min)", "km", ${h})`, `sa = Series("Height above the Moon km", ${arr(oa.v(n))})`,
    `b = LineChart(${strs(L)}, [sb], "natural", "mission time (h:min)", "m/s", ${h})`, `sb = Series("Orbital speed m/s", ${arr(os ? os.v(n) : [])})`].join('\n');
});
let ouiHistCache = null;
oui('ouiHist', 'Terrain statistics (whole map)', 'derived', (p) => { // histograms: how much of the map has each slope, ice-evidence and hazard value
  const W = MC.S.W; if (!W) return wait('Waiting…');
  if (!ouiHistCache || ouiHistCache.W !== W) { const sl = new Array(15).fill(0), ic = new Array(10).fill(0), hz = new Array(10).fill(0);
    for (let i = 0; i < W.slope.length; i++) { sl[Math.min(14, Math.floor(W.slope[i] / 2))]++; ic[Math.min(9, Math.floor(W.ice[i] / 255 * 10))]++; hz[Math.min(9, Math.floor(W.hazard[i] / 255 * 10))]++; }
    const pct = (b) => b.map((v) => v / W.slope.length * 100); ouiHistCache = { W, sl: pct(sl), ic: pct(ic), hz: pct(hz) }; }
  const H = ouiHistCache, G = grid(p, 3), h = G.h;
  return [G.root(['a', 'b', 'c'], ['Slope', 'Ice evidence', 'Hazard']), `a = BarChart(${strs(H.sl.map((_, i) => `${i * 2}°`))}, [sa], "grouped", "slope", "% of map", ${h})`, `sa = Series("Slope (limit ${lim().slope_limit_deg}°)", ${arr(H.sl)})`,
    `b = BarChart(${strs(H.ic.map((_, i) => (i / 10).toFixed(1)))}, [sb], "grouped", "score", "% of map", ${h})`, `sb = Series("Ice evidence score", ${arr(H.ic)})`,
    `c = BarChart(${strs(H.hz.map((_, i) => (i / 10).toFixed(1)))}, [sc], "grouped", "probability", "% of map", ${h})`, `sc = Series("Hazard", ${arr(H.hz)})`].join('\n');
});
oui('ouiProfile', 'Elevation profile along the route', 'derived', (p) => {
  const W = MC.S.W, c = MC.cols, n = MC.n(); if (!W) return wait('Waiting…'); const d = [], z = []; let acc = 0, px = null, py = null;
  for (let i = 0; i <= MC.cur && i < n; i++) { if (!Number.isFinite(c.x[i])) continue; if (px !== null) acc += Math.hypot(c.x[i] - px, c.y[i] - py); px = c.x[i]; py = c.y[i];
    const col = Math.floor((c.x[i] + W.half_x) / W.mpp), row = Math.floor((W.half_y - c.y[i]) / W.mpp); if (col < 0 || row < 0 || col >= W.cols || row >= W.rows) continue; d.push(acc); z.push(W.height[row * W.cols + col]); }
  if (d.length < 2) return wait('The route appears once the rover moves.'); const k = Math.min(24, d.length), idx = Array.from({ length: k }, (_, j) => Math.round(j * (d.length - 1) / (k - 1)));
  return `root = Card([a])\na = LineChart(${strs(idx.map((i) => `${d[i].toFixed(0)} m`))}, [s], "natural", "distance driven", "m", ${ch(p)})\ns = Series("Ground height vs landing point m", ${arr(idx.map((i) => z[i] - z[0]))})`;
});
oui('ouiTrain', 'Training curves', 'feed', (p) => {
  const T = MC.S.training; if (!T || !T.rows?.length) return wait('No training log found (runs/rover/train_log.jsonl).'); const rows = T.rows, last = rows[rows.length - 1], pick = (ks) => ks.find((k) => typeof last[k] === 'number');
  const kr = pick(['reward', 'mean_reward', 'return', 'ep_return', 'mean_return']), kl = pick(['loss', 'policy_loss', 'value_loss']); if (!kr && !kl) return wait('The training log has no reward or loss column.');
  const k = Math.min(30, rows.length), idx = Array.from({ length: k }, (_, j) => Math.round(j * (rows.length - 1) / Math.max(k - 1, 1))), L = idx.map((i) => String(i)), G = grid(p, kr && kl ? 2 : 1), h = G.h, out = [G.root(...[[kr ? 'a' : '', kl ? 'b' : ''], ['Reward', 'Loss']].map((x) => x.filter((_, i) => [kr, kl][i])))];
  if (kr) out.push(`a = LineChart(${strs(L)}, [sa], "natural", "log row", ${q(kr)}, ${h})`, `sa = Series(${q(kr)}, ${arr(idx.map((i) => rows[i][kr]))})`);
  if (kl) out.push(`b = LineChart(${strs(L)}, [sb], "natural", "log row", ${q(kl)}, ${h})`, `sb = Series(${q(kl)}, ${arr(idx.map((i) => rows[i][kl]))})`);
  return out.join('\n');
});

oui('ouiSkyEl', 'Sun and Earth above the horizon', 'feed', (p) => { // same unit (degrees) for both lines, so one axis is honest; 0 is the horizon
  const su = series1('sunEl'), ea = series1('earthEl'); if (!su && !ea) return wait('Sky angles appear once the feed has a site.'); const ref = (su || ea).s, n = 18, L = labels(ref[0], ref[1], n);
  return [`root = Card([c])`, `c = LineChart(${strs(L)}, [b, a], "natural", "mission time", "degrees", ${ch(p)})`, `a = Series("Sun elevation °", ${arr(su ? su.v(n) : [])})`,
    `b = Series("Earth elevation °", ${arr(ea ? ea.v(n) : [])})`].join('\n'); // Earth first so it is blue and the Sun orange, as in the sky view
});

// A short notice built from OpenUI's Callout component; it slides in with OpenUI's animation and fades out with OpenUI's callout fade.
MC.toast = function (variant, title, desc) {
  const O = window.__OpenUI, host = document.getElementById('toasts'); if (!O || !host) return;
  const el = document.createElement('div'); el.className = 'toast'; host.appendChild(el); const root = O.createRoot(el);
  root.render(O.React.createElement(O.Renderer, { response: `root = Card([c])\nc = Callout(${q(variant)}, ${q(title)}, ${q(desc)})`, library: O.openuiChatLibrary }));
  setTimeout(() => { el.classList.add('out'); setTimeout(() => { try { root.unmount(); } catch (e) { /* already gone */ } el.remove(); }, 700); }, 4600);
  while (host.children.length > 4) host.firstChild.remove();
};
// Render one OpenUI panel; called by the app loop. Re-renders only when the Lang text changed, so the chart animates between states.
MC.ouiRender = function (p, now) {
  const O = window.__OpenUI; if (!O) { if (!p.msgd) { p.el.innerHTML = '<div class="msg">Loading chart library…</div>'; p.msgd = true; } return; }
  if (now - (p.t || 0) < 900 && p.last) return; p.t = now; const lang = p.def.lang(p); if (lang === p.last) return; p.last = lang;
  if (!p.root) { p.el.innerHTML = ''; p.root = O.createRoot(p.el); } p.el.style.setProperty('--oui-round', `${Math.max(60, p.el.clientHeight - 58)}px`); // legend below the round charts p.root.render(O.React.createElement(O.Renderer, { response: lang, library: /\bStack\(/.test(lang) && O.openuiLibrary ? O.openuiLibrary : O.openuiChatLibrary }));
};
})();
