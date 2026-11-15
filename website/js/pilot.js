// #ai — the pilot beat. A pure function of section progress p (scrubbed, reversible):
//   0.00–0.16  the camera dives into the lander (scene.setPilotFocus; DOM lander zoom-through as the fallback)
//   0.12–0.50  a river of real 32×32 ground-camera patches (assets/img/pilot_patches.png, tools/export_patches.py) flows in from the left
//   0.34–0.84  real-weight fly-through of the network (scene.setNet, js/net3d.js)
//   0.82–1.00  three thrust corrections burst out of the lander + the benchmark result
// Static fallback (reduced motion / no WebGL / ?static): CSS lays the settled frame out as a flow; the stream canvas becomes a patch grid.
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const sm = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const backOut = t => { const c = 2.2; t = clamp(t) - 1; return 1 + (c + 1) * t * t * t + c * t * t; };   // overshoot = "burst"
const ARROWS = [0.9, -0.7, 1.0];     // illustrative directions for the 3D arrows (east, north, up), NOT a recorded command

export function mount(sec, ctx) {
  const $ = s => sec.querySelector(s);
  const $$ = s => Array.from(sec.querySelectorAll(s));
  const stage = $('.pl-stage');
  const el = {
    dim: $('.pl-dim'), head: $('.pl-head'), lander: $('.pl-lander'),
    capIn: $('.pl-cap-in'), srcP: $('.pl-src-patches'),
    capNet: $('.pl-cap-net'), count: $('.pl-count'), netnote: $('.pl-netnote'),
    out: $('.pl-out'), capOut: $('.pl-cap-out'), rig: $('.pl-rig'), arrows: $$('.pl-ar'), result: $('.pl-result'),
  };
  const cv = $('.pl-stream'), g = cv.getContext('2d');
  const isStatic = () => ctx.reducedMotion || document.body.classList.contains('no-webgl');

  // ---- atlas (real crops) ----
  let meta = { px: 32, cols: 10, rows: 8, n_patches: 72 }, ready = false;
  let W = 1, H = 1, dpr = 1, dirty = true;
  let s = 0, last = -1, wasStatic = null, sceneOff = true, lastCount = '';
  const atlas = new Image();
  atlas.onload = () => { ready = true; dirty = true; };
  atlas.src = 'assets/img/pilot_patches.png';
  fetch('assets/img/pilot_patches.json').then(r => r.json()).then(d => { meta = d; dirty = true; }).catch(() => {});

  // ---- sizes (cached; never read layout in update) ----
  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = cv.clientWidth || innerWidth; H = cv.clientHeight || innerHeight;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    dirty = true; last = -1;
  }
  addEventListener('resize', resize);
  resize();

  // per-patch lanes inside one curved river band (deterministic)
  const rnd = i => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
  const lanes = Array.from({ length: 96 }, (_, i) => ({ o: rnd(i) - 0.5, s: 0.7 + 0.6 * rnd(i + 99), d: rnd(i + 7) }));

  // river: cubic from off-screen left (low), sweeping up through the middle, into the network mouth (centre)
  function river(u, o, out) {
    const narrow = W < 700, v = 1 - u;
    const P0 = [-0.12 * W, (narrow ? 0.80 : 0.86) * H], P1 = [0.18 * W, 0.92 * H], P2 = [0.22 * W, 0.40 * H], P3 = [0.5 * W, (narrow ? 0.52 : 0.58) * H];
    const b = [v * v * v, 3 * v * v * u, 3 * v * u * u, u * u * u];
    const x = b[0] * P0[0] + b[1] * P1[0] + b[2] * P2[0] + b[3] * P3[0];
    const y = b[0] * P0[1] + b[1] * P1[1] + b[2] * P2[1] + b[3] * P3[1];
    const width = (narrow ? 0.16 : 0.22) * H * Math.pow(v, 0.8);          // the band narrows into the mouth
    out[0] = x + o * width * 0.6; out[1] = y + o * width;
    return out;
  }
  const pt = [0, 0];
  function drawStream(pp) {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!ready || pp <= 0 || pp >= 1) return;
    g.imageSmoothingEnabled = false;
    const N = meta.n_patches, P = meta.px, base = W < 700 ? 46 : 84;
    for (let i = 0; i < N; i++) {
      const L = lanes[i];
      const u0 = (pp * 1.5 - (i / N) * 1.0 - L.d * 0.06) / 0.45;      // 0 = enters at the left edge, 1 = swallowed by the network
      for (let tr = 3; tr >= 0; tr--) {                               // faint trails behind each patch = motion blur
        const u = u0 - tr * 0.025;
        if (u <= 0 || u >= 1) continue;
        river(u, L.o, pt);
        const sz = base * L.s * Math.pow(1 - u, 0.55);
        if (sz < 1.5) continue;
        g.globalAlpha = (tr === 0 ? 1 : 0.4 / tr) * Math.min(1, u * 8) * Math.min(1, (1 - u) * 5);
        g.drawImage(atlas, (i % meta.cols) * P, Math.floor(i / meta.cols) * P, P, P, pt[0] - sz / 2, pt[1] - sz / 2, sz, sz);
      }
    }
    g.globalAlpha = 1;
  }

  function drawGrid() {                 // static fallback: every real patch as a grid
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!ready) return;
    g.imageSmoothingEnabled = false;
    const N = meta.n_patches, P = meta.px;
    const cols = W < 600 ? 8 : 18, gap = 4, sz = (W - gap * (cols - 1)) / cols;
    for (let i = 0; i < N; i++) {
      const x = (i % cols) * (sz + gap), y = Math.floor(i / cols) * (sz + gap);
      if (y > H) break;
      g.drawImage(atlas, (i % meta.cols) * P, Math.floor(i / meta.cols) * P, P, P, x, y, sz, sz);
    }
  }

  const show = (node, a, dy = 0, sc = 1, blur = 0) => {
    if (!node) return;
    node.style.opacity = a.toFixed(3);
    node.style.visibility = a < 0.005 ? 'hidden' : 'visible';
    node.style.transform = `translate3d(0, ${dy.toFixed(1)}px, 0) scale(${sc.toFixed(4)})`;
    node.style.filter = blur > 0.05 ? `blur(${blur.toFixed(1)}px)` : '';
  };
  const fly = (node, k, from = 24) => show(node, k, (1 - k) * from, 0.92 + 0.08 * k, (1 - k) * 6);   // blur -> sharp, rising
  const win = (x, a, b, c, d) => Math.min(sm(a, b, x), 1 - sm(c, d, x));

  function update(p, vis) {
    const st = isStatic();
    if (st !== wasStatic) {
      wasStatic = st;
      sec.classList.toggle('pl-static', st);
      stage.querySelectorAll('[style]').forEach(n => n.removeAttribute('style'));
      el.arrows.forEach(a => a.removeAttribute('transform'));
      if (st && el.count) el.count.textContent = '14,842,567';
      requestAnimationFrame(() => { resize(); if (st) drawGrid(); });
    }
    if (st) { if (dirty && ready) { dirty = false; drawGrid(); } return; }

    s = ctx.shot ? p : s + (p - s) * 0.12;
    if (Math.abs(p - s) < 1e-4) s = p;
    const scn = ctx.getScene?.();
    if (vis <= 0.001) {
      if (!sceneOff && scn) { scn.setNet?.(0, 0); scn.setPilotFocus?.(0); scn.setThrustArrows?.(null, 0); sceneOff = true; }
      return;
    }
    sceneOff = false;
    stage.classList.toggle('pl-3d', !!(scn && typeof scn.setPilotFocus === 'function'));
    stage.classList.toggle('pl-3d-arrows', !!(scn && typeof scn.setThrustArrows === 'function'));

    // ---- A: dive into the lander ----
    show(el.head, 1 - sm(0.07, 0.13, s), -sm(0.07, 0.13, s) * 30);
    const kIn = sm(0.0, 0.16, s);
    show(el.lander, (1 - sm(0.10, 0.17, s)), 0, 1 + Math.pow(kIn, 2.2) * 11, kIn * kIn * 10);
    show(el.dim, win(s, 0.11, 0.2, 0.74, 0.81) * 0.9);

    // ---- B: the river of real camera patches ----
    const pp = clamp((s - 0.12) / 0.38);
    const key = Math.round(pp * 2000);
    if (key !== last || dirty) { last = key; dirty = false; drawStream(pp); }
    fly(el.capIn, win(s, 0.15, 0.21, 0.36, 0.42), 18);
    show(el.srcP, win(s, 0.18, 0.24, 0.38, 0.42));

    // ---- C: network fly-through (real weights) ----
    const q = clamp((s - 0.38) / 0.38) * 0.93;      // stop with the thrust head + output columns still in view
    scn?.setNet?.(q, Math.min(sm(0.32, 0.44, s), 1 - sm(0.74, 0.81, s)) * clamp(vis * 1.5));
    const netA = win(s, 0.44, 0.50, 0.72, 0.78);
    fly(el.capNet, netA, 18);
    show(el.netnote, win(s, 0.48, 0.54, 0.72, 0.78));
    const txt = Math.round(14842567 * sm(0.45, 0.58, s)).toLocaleString('en-US');
    if (txt !== lastCount) { lastCount = txt; el.count.textContent = txt; }

    // ---- D: thrust corrections burst out ----
    const oA = sm(0.77, 0.83, s);
    // tail: everything in the output frame is gone before the next section (#built) enters (raw p, so fast scrolling cannot leave it behind)
    const tail = (1 - sm(0.955, 0.995, Math.max(s, p))) * clamp((vis - 0.92) / 0.08);
    show(el.out, oA * tail > 0.005 ? tail : 0);
    fly(el.capOut, oA, 18);
    show(el.rig, oA, (1 - oA) * 30, 0.7 + 0.3 * oA, (1 - oA) * 8);
    for (let i = 0; i < 3; i++) {
      const k = clamp((s - 0.80 - i * 0.015) / 0.06);
      el.arrows[i].setAttribute('transform', `scale(${Math.max(0.001, backOut(k)).toFixed(3)})`);
      el.arrows[i].style.opacity = Math.min(1, k * 3).toFixed(3);
    }
    fly(el.result, sm(0.83, 0.88, s), 24);

    // 3D lander hooks (agent F): centre the lander, fly into it, then arrows on it
    const outK = W < 700 ? 0.15 : 0.55;   // scene.js: 0..0.65 centres the lander, 0.65..1 flies into it (hidden at ~1)
    scn?.setPilotFocus?.(s < 0.5 ? kIn * clamp(vis * 1.4) : (1 - sm(0.74, 0.82, s) * (1 - outK)) * clamp((vis - 0.92) / 0.08));
    const ak = clamp((s - 0.80) / 0.06);
    scn?.setThrustArrows?.(ak > 0.001 && tail > 0.01 ? ARROWS.map(v => v * Math.min(1.15, backOut(ak))) : null, Math.min(1, ak * 3) * tail);
  }
  let warned = false;
  return { update: (p, vis, t) => { try { update(p, vis, t); } catch (e) { if (!warned) { warned = true; console.warn('pilot', e); } } } };
}
