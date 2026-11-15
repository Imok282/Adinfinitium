// #built beat: multiplayer cursors (two people + three AI coding agents) assemble the simulation software
// piece by piece inside a mock IDE. Everything is a pure function of the smoothed section progress s.
// The cursors are a dramatisation; the code lines, images, chart numbers and commit counts are real:
//   code      copied verbatim from worldmodel/hazard/features.py, worldmodel/lander/pilot.py, worldmodel/lander/guidance.py
//   map       assets/img/terrain_site23.jpg (OBSERVED, NASA LOLA 5 m, Site 23)
//   sim       assets/models3d/still_ghosts.jpg (MODELLED ghost trajectories, lander twin on LOLA terrain)
//   chart     runs/hazard/eval.json, region lola5m_site23 (role TEST): U-Net F1 0.950, slope rule F1 0.814
//   commits   assets/build_log.json (git log --all via tools/export_build_log.py), shown by js/buildlog.js
import { mountBuildLog } from './buildlog.js';

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = x => (x = clamp(x), x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const easeOut = x => 1 - Math.pow(1 - clamp(x), 3);
const lerp = (a, b, k) => a + (b - a) * k;
const seg = (s, a, b) => clamp((s - a) / (b - a));

// ---------- real code (verbatim lines, real line numbers) ----------
const CODE = [
  { tab: 'features.py', path: 'worldmodel/hazard/features.py', start: 36, lines: [
    'def gap_fill(height: np.ndarray, measured: np.ndarray, sigmas: list[float]) -> np.ndarray:',
    { gap: 46 },
    '    ref = float(h[measured].mean())',
    '    hm = np.where(measured, h - ref, 0.0)',
    '    m = measured.astype(np.float64)',
    '    out = np.full(h.shape, np.nan)',
    '    for s in sigmas:',
    '        den = gaussian_filter(m, s, mode="nearest")',
    '        num = gaussian_filter(hm, s, mode="nearest")',
    '        ok = np.isnan(out) & (den >= MIN_WEIGHT)',
    '        out[ok] = num[ok] / den[ok]',
    '    out[np.isnan(out)] = 0.0',
    '    return np.where(measured, hm, out) + ref',
  ] },
  { tab: 'pilot.py', path: 'worldmodel/lander/pilot.py', start: 23, lines: [
    'RESIDUAL_SCALE = 0.3',
    { gap: 104 },
    'def residual_action(classic: torch.Tensor, r: torch.Tensor, w: torch.Tensor | float = 1.0) -> torch.Tensor:',
    '    return (classic + RESIDUAL_SCALE * w * r.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)',
  ] },
  { tab: 'guidance.py', path: 'worldmodel/lander/guidance.py', start: 32, lines: [
    'def classic_action(gp: GuidanceParams, obs: torch.Tensor) -> torch.Tensor:',
    '    r_h = obs[:, 0:2] * 1000.0',
    '    alt = obs[:, 2] * 1000.0',
    '    v = obs[:, 3:6] * 50.0',
    '    mass = obs[:, 8] * gp.start_mass',
    '    g_vec = torch.tensor([0.0, 0.0, -gp.g], device=obs.device)',
    '    h = alt - gp.gate_alt',
    '    tgo = (2.0 * h.clamp(min=0.0) / (-v[:, 2] + gp.v_final).clamp(min=1.0)).clamp(gp.tgo_min, gp.tgo_max)[:, None]',
    '    r = torch.cat([r_h, h[:, None]], dim=1)',
    '    v_f = torch.tensor([0.0, 0.0, -gp.v_final], device=obs.device)',
    '    zem = -(r + v * tgo + 0.5 * g_vec * tgo**2)',
    '    zev = v_f - (v + g_vec * tgo)',
    '    a_zem = 6.0 * zem / tgo**2 - 2.0 * zev / tgo',
    '    a_h = -gp.kp_h * r_h - gp.kv_h * v[:, :2]',
    '    a_z = gp.kv_z * (-gp.v_final - v[:, 2])',
    '    a_term = torch.cat([a_h, a_z[:, None]], dim=1)',
    '    a_ctrl = torch.where((h > gp.switch_h)[:, None], a_zem, a_term - g_vec)',
    '    return mass[:, None] * a_ctrl / gp.max_total_thrust',
  ] },
];
// typing windows (smoothed progress) per tab, in order; the last tab stays open at the end
const TYPE = [[0.36, 0.47], [0.47, 0.56], [0.56, 0.76]];
const VIS_CH = 72;   // typing time per line is capped: long lines run off the panel edge anyway

// ---------- cast ----------
const CURSORS = [
  // generic names (the user's choice); the real per-person counts live in the ticker
  { id: 'imok', name: 'Bob', c: '#d97757' },
  { id: 'nimansh', name: 'Maya', c: '#b7cdbe' },
  { id: 'lander', name: 'Harry', c: '#c8c3e0' },
  { id: 'models', name: 'Leo', c: '#e8b9a0' },
  { id: 'web', name: 'Ada', c: '#9fc3d6' },
];
// Pieces: owner drags it from `from` (centre, normalised main-area coords) between t0 and t1.
const PIECES = [
  { k: 'map', owner: 'imok', t0: 0.12, t1: 0.27, from: [-0.75, 0.28], files: ['contracts/tiles.py'] },
  { k: 'code', owner: 'lander', t0: 0.18, t1: 0.34, from: [1.75, 0.3], files: ['worldmodel/hazard/features.py', 'worldmodel/lander/pilot.py', 'worldmodel/lander/guidance.py'] },
  { k: 'sim', owner: 'nimansh', t0: 0.27, t1: 0.43, from: [0.2, 1.9], files: ['contracts/runs.py'] },
  { k: 'chart', owner: 'models', t0: 0.42, t1: 0.58, from: [0.71, 1.9], files: ['runs/hazard/eval.json'] },
];
// Free keyframes [s, u, v] per cursor (u, v: fraction of the IDE main area; may go outside it).
const KEYS = {
  web: [[0, 0.5, -0.45], [0.03, 0.01, 0.02], [0.06, 0.4, 0.56], [0.085, 0.43, 0.02], [0.11, 1.0, 0.63], [0.135, 0.42, 1.0],
        [0.2, 0.08, 1.02], [0.5, 0.97, 0.98], [0.7, 0.42, 0.99], [0.84, 0.405, 0.56], [1, 0.405, 0.56]],
  imok: [[0, -0.25, -0.25], [0.08, -0.3, 0.1], [0.34, 0.27, 0.32], [0.5, 0.2, 0.18], [0.66, 0.28, 0.4], [0.8, 0.33, 0.44], [1, 0.33, 0.44]],
  nimansh: [[0, -0.3, 1.35], [0.2, -0.1, 1.3], [0.5, 0.24, 0.8], [0.555, 0.6, 0.03], [0.6, 0.6, 0.04], [0.76, 0.3, 0.72], [1, 0.27, 0.74]],
  lander: [[0, 1.3, -0.35], [0.12, 1.3, 0.1], [0.9, 0.9, 0.45], [1, 0.9, 0.46]],
  models: [[0, 1.3, 1.35], [0.3, 1.25, 1.2], [0.78, 0.62, 0.7], [1, 0.62, 0.7]],
};
const SLOT_AT = { map: 0.05, code: 0.085, chart: 0.11, sim: 0.135 };   // web agent lays out the grid first
const BARS = [0.6, 0.72];   // F1 bars grow (models agent drags their ends)
const HAZ = [
  { label: 'U-Net hazard model', v: 0.9502803888980582, c: 'var(--sage)' },
  { label: 'Slope rule (physics backup)', v: 0.8141130672992294, c: 'var(--peach)' },
];

function h(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

// tiny tokenizer for colour: comments, strings, numbers, keywords, names before "(" or ":" (yaml keys)
function colorLine(line, yaml) {
  const frag = document.createDocumentFragment();
  let rest = line, comment = '';
  const ci = rest.indexOf('#');
  if (ci >= 0) { comment = rest.slice(ci); rest = rest.slice(0, ci); }
  const rx = yaml
    ? /(^\s*[\w-]+(?=:))|(\d+(?:\.\d+)?)|([^\d]+?)/g
    : /\b(def|return|float|None)\b|(\b\d+(?:\.\d+)?\b)|(\b\w+(?=\())|([^\w]+|\w+)/g;
  let m;
  while ((m = rx.exec(rest))) {
    const t = m[0];
    let cls = null;
    if (yaml) cls = m[1] ? 'tk-k' : m[2] ? 'tk-n' : null;
    else cls = m[1] ? 'tk-kw' : m[2] ? 'tk-n' : m[3] ? 'tk-f' : null;
    if (cls) frag.appendChild(h('span', cls, t)); else frag.appendChild(document.createTextNode(t));
  }
  if (comment) frag.appendChild(h('span', 'tk-c', comment));
  return frag;
}

function arrowSVG() {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 16 20'); s.setAttribute('width', '16'); s.setAttribute('height', '20'); s.setAttribute('aria-hidden', 'true');
  s.innerHTML = '<path d="M1.5 1.5 L1.5 16.5 L5.6 12.8 L8.4 19 L11 17.8 L8.3 11.7 L14 11.7 Z" fill="var(--c)" stroke="#1f1f1d" stroke-width="1.4" stroke-linejoin="round"/>';
  return s;
}

export function mount(sec, ctx) {
  const ide = sec.querySelector('#cur-ide');
  if (!ide) return null;
  const isStatic = () => ctx.reducedMotion || document.body.classList.contains('no-webgl');

  // ---------- headline: word-by-word reveal ----------
  const words = [];
  const hd = sec.querySelector('.cur-h');
  if (hd) {
    const walk = (node, into) => {
      for (const n of [...node.childNodes]) {
        if (n.nodeType === 3) {
          const parts = n.textContent.split(/(\s+)/);
          const f = document.createDocumentFragment();
          for (const p of parts) {
            if (!p) continue;
            if (/^\s+$/.test(p)) { f.appendChild(document.createTextNode(p)); continue; }
            const w = h('span', 'cur-w', p); words.push(w); f.appendChild(w);
          }
          node.replaceChild(f, n);
        } else if (n.nodeType === 1) walk(n, into);
      }
    };
    walk(hd);
  }

  // ---------- IDE ----------
  ide.textContent = '';
  const win = h('div', 'ide-win');
  const bar = h('div', 'ide-bar');
  bar.append(h('i', 'd'), h('i', 'd'), h('i', 'd'));
  const ttl = h('span', 'ide-ttl'); ttl.append(h('b', null, 'adinfinitium'), ' · lunar landing site navigator');
  bar.appendChild(ttl);
  const pres = h('span', 'ide-pres');
  CURSORS.forEach(c => { const d = h('i'); d.style.setProperty('--c', c.c); d.title = c.name; pres.appendChild(d); });
  bar.appendChild(pres);
  win.appendChild(bar);

  const body = h('div', 'ide-body');
  const side = h('nav', 'ide-side'); side.setAttribute('aria-label', 'Repository files (real paths)');
  const TREE = [
    ['contracts/', 0], ['tiles.py', 1, 'contracts/tiles.py'], ['light.py', 1, 'contracts/light.py'], ['runs.py', 1, 'contracts/runs.py'], ['sites.py', 1, 'contracts/sites.py'],
    ['worldmodel/', 0], ['lander/', 1], ['guidance.py', 2, 'worldmodel/lander/guidance.py'], ['pilot.py', 2, 'worldmodel/lander/pilot.py'],
    ['hazard/', 1], ['features.py', 2, 'worldmodel/hazard/features.py'], ['light/', 1], ['terrain/', 1], ['rover/', 1],
    ['runs/hazard/', 0], ['eval.json', 1, 'runs/hazard/eval.json'],
    ['sim_jax/', 0], ['website/', 0],
  ];
  const fileEl = {};
  TREE.forEach(([n, d, path]) => {
    const r = h('div', 'tr' + (path ? ' f' : ' dir'), n); r.style.setProperty('--d', d);
    if (path) fileEl[path] = r;
    side.appendChild(r);
  });
  body.appendChild(side);

  const main = h('div', 'ide-main');
  body.appendChild(main);
  win.appendChild(body);
  ide.appendChild(win);

  const slots = {}, pieces = {}, sels = {};
  for (const P of PIECES) {
    const sl = h('div', 'slot'); sl.dataset.k = P.k; main.appendChild(sl); slots[P.k] = sl;
    const pc = h('div', 'piece'); pc.dataset.k = P.k; pieces[P.k] = pc;
    const sel = h('div', 'pc-sel'); const own = CURSORS.find(c => c.id === P.owner);
    sel.style.setProperty('--c', own.c);
    sels[P.k] = sel;
    main.appendChild(pc);
  }

  // map piece (observed)
  {
    const pc = pieces.map;
    const hd2 = h('div', 'pc-h'); hd2.append(h('span', 'pc-t', 'terrain_site23.jpg'));
    const img = h('div', 'map-img'); img.setAttribute('role', 'img');
    img.setAttribute('aria-label', 'Site 23 near the lunar south pole, NASA LOLA 5 m: hillshaded height, and slope of 12 degrees or more in red with permanently shadowed regions in blue.');
    const cap = h('div', 'pc-cap', 'Site 23 · LOLA 5 m · height (hillshade) | slope ≥ 12° red, permanent shadow blue');
    pc.append(hd2, img, sels.map);
  }
  // sim piece (modelled)
  {
    const pc = pieces.sim;
    const hd2 = h('div', 'pc-h'); hd2.append(h('span', 'pc-t', 'lander twin · ghost runs'));
    const img = h('div', 'sim-img'); img.setAttribute('role', 'img');
    img.setAttribute('aria-label', 'Simulated lander trajectories over real lunar terrain: green successes converge on the target, red failures scatter.');
    const cap = h('div', 'pc-cap', 'simulated successes (green) and failures (red) on LOLA terrain');
    pc.append(hd2, img, sels.sim);
  }
  // code piece
  const tabs = [], codeBodies = [], typed = [];
  {
    const pc = pieces.code;
    const tb = h('div', 'code-tabs');
    CODE.forEach((f, i) => { const t = h('span', 'code-tab', f.tab); t.title = f.path; tb.appendChild(t); tabs.push(t); });
    const path = h('div', 'code-path');
    pc.append(tb, path);
    const view = h('div', 'code-view');
    CODE.forEach((f, i) => {
      const cb = h('div', 'code-body'); cb.dataset.i = i;
      const yaml = f.tab.endsWith('.yaml');
      let no = f.start; const lines = [];
      for (const ln of f.lines) {
        if (typeof ln === 'object') {
          const g = h('div', 'cl gap'); g.append(h('span', 'no', '⋮'), h('span', 'tx', ''));
          cb.appendChild(g); no = ln.gap; lines.push({ el: null, n: 0 }); continue;
        }
        const row = h('div', 'cl');
        const tx = h('span', 'tx'); tx.appendChild(colorLine(ln, yaml));
        row.append(h('span', 'no', String(no)), tx);
        cb.appendChild(row);
        lines.push({ el: row, tx, n: Math.min(ln.length, VIS_CH), indent: ln.length - ln.trimStart().length });
        no++;
      }
      const total = lines.reduce((a, l) => a + l.n, 0);
      typed.push({ lines, total, shown: -1, path: f.path });
      view.appendChild(cb); codeBodies.push(cb);
    });
    pc.append(view, sels.code);
    pc._path = path;
  }
  // chart piece (hazard F1, real)
  const barEls = [], barVals = [];
  {
    const pc = pieces.chart;
    const hd2 = h('div', 'pc-h'); hd2.append(h('span', 'pc-t', 'Hazard F1 · Site 23 test tile'));
    const bars = h('div', 'bars');
    HAZ.forEach(b => {
      const r = h('div', 'bar-r');
      const lab = h('span', 'bar-l', b.label);
      const tr = h('span', 'bar-t'); const fill = h('i'); fill.style.background = b.c; tr.appendChild(fill);
      const val = h('b', 'bar-v', b.v.toFixed(3));
      r.append(lab, tr, val); bars.appendChild(r); barEls.push(fill); barVals.push(val);
    });
    const src = h('div', 'pc-cap', 'Source: runs/hazard/eval.json · region lola5m_site23 (TEST, unseen in training) · label: LOLA slope ≥ 12°');
    pc.append(hd2, bars, sels.chart);
  }

  // cursors
  const curLayer = h('div', 'cur-layer'); curLayer.setAttribute('aria-hidden', 'true');
  const curs = {};
  CURSORS.forEach(c => {
    const e = h('div', 'cur'); e.style.setProperty('--c', c.c);
    e.append(arrowSVG(), h('span', 'cur-pill', c.name));
    curLayer.appendChild(e);
    curs[c.id] = { el: e, x: 0, y: 0, init: false, k: { imok: 0.16, nimansh: 0.14, lander: 0.22, models: 0.18, web: 0.2 }[c.id] };
  });
  main.appendChild(curLayer);

  // git history ticker
  const tick = sec.querySelector('#cur-tick');
  let bl = null;
  if (tick) fetch('assets/build_log.json').then(r => r.json()).then(d => { bl = mountBuildLog(tick, d, { reducedMotion: false }); last = -1; })
    .catch(e => console.warn('build log', e));

  // ---------- layout cache (read only on resize) ----------
  let s = 0, last = -1, lastHead = -1;
  let W = 1, H = 1, R = {}, chW = 6.6, lineH = 15, codeTop = 0, codeLeft = 0;
  function measure() {
    W = main.clientWidth || 1; H = main.clientHeight || 1;
    for (const P of PIECES) {
      const e = slots[P.k]; R[P.k] = { x: e.offsetLeft, y: e.offsetTop, w: e.offsetWidth, h: e.offsetHeight };
      sels[P.k].dataset.wh = `${R[P.k].w} × ${R[P.k].h}`;
    }
    const cb = codeBodies[0], row = cb && cb.querySelector('.cl:not(.gap) .tx');
    const view = pieces.code.querySelector('.code-view');
    if (row) { const fs = parseFloat(getComputedStyle(row).fontSize) || 11; chW = fs * 0.6; lineH = cb.querySelector('.cl').offsetHeight || 15; codeLeft = (view ? view.offsetLeft : 0) + row.offsetLeft; }
    codeTop = view ? view.offsetTop : 30;
    last = -1;
  }
  addEventListener('resize', measure);
  if (window.ResizeObserver) new ResizeObserver(measure).observe(main);
  measure();

  // ---------- per-frame ----------
  const pieceOf = k => PIECES.find(P => P.k === k);
  const grab = (P, k) => { const r = R[P.k]; return [r.x + r.w * 0.5, r.y + 14]; };   // grab the piece by its header
  function pieceState(P, s) {
    const r = R[P.k];
    const k = ease(seg(s, P.t0, P.t1));
    const fx = P.from[0] * W - (r.x + r.w / 2), fy = P.from[1] * H - (r.y + r.h / 2);
    return { k, dx: fx * (1 - k), dy: fy * (1 - k), on: s >= P.t0 };
  }
  function freeTarget(id, s) {
    const ks = KEYS[id].slice();
    for (const P of PIECES) if (P.owner === id) {
      const g = grab(P);
      const st0 = pieceState(P, P.t0);
      ks.push([P.t0, (g[0] + st0.dx) / W, (g[1] + st0.dy) / H], [P.t1, g[0] / W, g[1] / H]);
    }
    ks.sort((a, b) => a[0] - b[0]);
    if (s <= ks[0][0]) return [ks[0][1] * W, ks[0][2] * H];
    for (let i = 1; i < ks.length; i++) {
      if (s <= ks[i][0]) {
        const a = ks[i - 1], b = ks[i], k = ease((s - a[0]) / Math.max(1e-6, b[0] - a[0]));
        return [lerp(a[1], b[1], k) * W, lerp(a[2], b[2], k) * H];
      }
    }
    const z = ks[ks.length - 1];
    return [z[1] * W, z[2] * H];
  }
  let caret = null;   // caret px (main coords) while typing, for the lander agent

  function render(s, snap) {
    const key = Math.round(s * 4000);
    const changed = key !== last;
    last = key;
    // pieces
    if (changed) {
      for (const P of PIECES) {
        const st = pieceState(P, s), pc = pieces[P.k];
        const a = clamp((s - P.t0) / ((P.t1 - P.t0) * 0.22));
        const lift = 1 - st.k;
        pc.style.opacity = st.on ? a.toFixed(3) : '0';
        pc.style.visibility = st.on ? 'visible' : 'hidden';
        pc.style.transform = `translate3d(${st.dx.toFixed(1)}px,${st.dy.toFixed(1)}px,0) scale(${(1 + 0.07 * lift).toFixed(4)}) rotate(${(lift * (P.from[0] < 0 ? -4 : 4)).toFixed(2)}deg)`;
        pc.style.filter = a < 1 ? `blur(${((1 - a) * 6).toFixed(2)}px)` : '';
        pc.classList.toggle('lifted', st.on && st.k < 0.98);
        // selection frame: shown while dragging and briefly after the drop
        sels[P.k].style.opacity = st.on ? (st.k < 1 ? 1 : clamp(1 - (s - P.t1) / 0.07)).toFixed(3) : '0';
        const sl = slots[P.k];
        const so = clamp((s - SLOT_AT[P.k]) / 0.025);
        sl.style.opacity = (so * (st.k >= 1 ? 0 : 1)).toFixed(3);
        sl.style.transform = `scale(${(0.96 + 0.04 * so).toFixed(4)})`;
        for (const f of P.files) if (fileEl[f]) fileEl[f].classList.toggle('on', st.k >= 1);
      }
      // code typing
      caret = null;
      let active = 0;
      TYPE.forEach((w, i) => { if (s >= w[0]) active = i; });
      tabs.forEach((t, i) => { t.classList.toggle('on', i === active); t.classList.toggle('open', s >= TYPE[i][0] || i === 0); });
      codeBodies.forEach((cb, i) => cb.classList.toggle('on', i === active));
      pieces.code._path.textContent = CODE[active].path;
      const T = typed[active];
      const kk = seg(s, TYPE[active][0], TYPE[active][1]);
      const nch = Math.round(kk * T.total);
      if (nch !== T.shown) {
        T.shown = nch;
        let left = nch, li = 0;
        T.lines.forEach((L, i) => {
          if (!L.el) return;
          const n = clamp(left, 0, L.n);
          left -= L.n;
          L.el.style.visibility = n > 0 ? 'visible' : 'hidden';
          L.tx.style.width = n >= L.n ? '' : `${n}ch`;
          L.el.classList.toggle('typing', n > 0 && n < L.n);
          if (n > 0) li = i;
        });
        T.caretLine = li; T.caretCh = (() => { let l2 = nch; for (let i = 0; i < T.lines.length; i++) { const L = T.lines[i]; if (!L.el) continue; if (l2 <= L.n) return i === li ? l2 : L.n; l2 -= L.n; } return 0; })();
      }
      // other tabs: fully typed once their window passed (so switching back would show them complete)
      typed.forEach((T2, i) => {
        if (i === active) return;
        const full = s >= TYPE[i][1];
        const want = full ? T2.total : 0;
        if (T2.shown === want) return;
        T2.shown = want;
        T2.lines.forEach(L => { if (!L.el) return; L.el.style.visibility = full ? 'visible' : 'hidden'; L.tx.style.width = ''; L.el.classList.remove('typing'); });
      });
      if (s >= TYPE[0][0] && s <= TYPE[TYPE.length - 1][1] + 0.02) {
        const r = R.code;
        caret = [r.x + codeLeft + (T.caretCh || 0) * chW, r.y + codeTop + (T.caretLine || 0) * lineH + lineH * 0.9];
      }
      // chart bars
      const bk = easeOut(seg(s, BARS[0], BARS[1]));
      barEls.forEach((b, i) => { b.style.transform = `scaleX(${(HAZ[i].v * bk).toFixed(4)})`; });
      barVals.forEach((v, i) => { v.style.opacity = clamp((s - BARS[1] + 0.03) / 0.03).toFixed(2); });
      // window: rises and un-tilts
      const wk = easeOut(seg(s, 0, 0.12));
      win.style.transform = `perspective(1600px) rotateX(${(14 * (1 - wk)).toFixed(2)}deg) translate3d(0,${(40 * (1 - wk)).toFixed(1)}px,0) scale(${(0.94 + 0.06 * wk).toFixed(4)})`;
      win.style.opacity = (0.35 + 0.65 * easeOut(seg(s, -0.02, 0.06))).toFixed(3);
      bl && bl.setProgress(seg(s, 0.06, 0.86));
    }
    // cursors (own light damping for a hand-driven feel; snapped for screenshots / static)
    for (const c of CURSORS) {
      const C = curs[c.id];
      let tx, ty;
      const drag = PIECES.find(P => P.owner === c.id && s >= P.t0 && s <= P.t1);
      if (drag) { const st = pieceState(drag, s), g = grab(drag); tx = g[0] + st.dx; ty = g[1] + st.dy; }
      else if (c.id === 'lander' && caret && s < TYPE[TYPE.length - 1][1]) { tx = caret[0] + 4; ty = caret[1] - 4; }
      else if (c.id === 'models' && s >= BARS[0] && s <= BARS[1] + 0.03) {
        const r = R.chart, bk = easeOut(seg(s, BARS[0], BARS[1])), i = s < (BARS[0] + BARS[1]) / 2 ? 0 : 1;
        const bx = r.x + r.w * (0.38 + 0.47 * HAZ[i].v * bk), by = r.y + r.h * (0.36 + 0.24 * i);
        tx = bx; ty = by;
      } else if (c.id === 'imok' && W < 520 && s > 0.8) { tx = 0.55 * W; ty = 0.3 * H; }   // phone: park inside the window, not over the counts
      else { [tx, ty] = freeTarget(c.id, s); }
      if (snap || !C.init) { C.x = tx; C.y = ty; C.init = true; }
      else { C.x += (tx - C.x) * C.k * 2.2; C.y += (ty - C.y) * C.k * 2.2; }
      const tr = `translate3d(${C.x.toFixed(1)}px,${C.y.toFixed(1)}px,0)`;
      if (tr !== C.tr) { C.tr = tr; C.el.style.transform = tr; }
      const op = clamp((s + 0.02) / 0.04).toFixed(2);
      if (op !== C.op) { C.op = op; C.el.style.opacity = op; }
    }
  }

  function heads(r) {
    const n = words.length;
    const key = Math.round(r * 400);
    if (key === lastHead) return;
    lastHead = key;
    words.forEach((w, i) => {
      const e = easeOut(clamp(r * (n + 4) / 4 - i / 4));
      w.style.opacity = e.toFixed(3);
      w.style.transform = `translate3d(0,${((1 - e) * 10).toFixed(1)}px,0)`;
      w.style.filter = e < 1 ? `blur(${((1 - e) * 5).toFixed(1)}px)` : '';
    });
  }

  let wasStatic = null;
  function update(p, vis, t) {
    const st = isStatic();
    if (st !== wasStatic) { wasStatic = st; sec.classList.toggle('cur-static', st); measure(); if (st) { s = 1; render(1, true); heads(1); } }
    if (st) return;
    if (ctx.shot) {   // screenshots: read the real position (main.js caches section rects; exact frames matter more here)
      const r = sec.getBoundingClientRect();
      p = clamp(-r.top / Math.max(1, r.height - innerHeight));
      s = p;
    } else { s += (p - s) * 0.3; if (Math.abs(p - s) < 1e-4) s = p; }   // main.js already lerps scroll
    heads(ctx.shot ? 1 : clamp(vis * 1.25));
    render(s, ctx.shot);
  }
  // first paint
  render(0, true);
  return { update };
}
