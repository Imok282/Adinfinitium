// "World models" section: four interactive 3D terrain views built from real repo data
// (assets/models3d/, made by tools/export_models3d.py). Each view is lazy: created when it nears the
// viewport, paused off-screen, disposed when far away, so at most MAX_LIVE WebGL contexts exist.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

import { loadGhosts, loadGhostHead, fillCounts } from './ghostdata.js';

const BASE = 'assets/models3d/';
let ghostMask = [1, 1, 1];          // classic, pilot, open_loop (ghosts view toggles)
let ghostOnMask = null;
const MAX_LIVE = 2;
const BG = 0x262624;           // warm charcoal (css --bg)
const params = new URLSearchParams(location.search);
const SHOT = params.get('mvshot');           // ?mvshot=elev renders one view full-window (used to make the still images)

let metaP = null;
const getMeta = () => (metaP ||= fetch(BASE + 'models3d.json').then(r => r.json()));
const heightP = {};
function getHeights(tag, m) {
  return (heightP[tag] ||= fetch(BASE + tag + '_height.bin').then(r => r.arrayBuffer()).then(buf => {
    const v = new Uint16Array(buf), out = new Float32Array(v.length), k = (m.max_m - m.min_m) / 65535;
    for (let i = 0; i < v.length; i++) out[i] = m.min_m + v[i] * k;
    return out;
  }));
}
const texLoader = new THREE.TextureLoader();
const loadTex = url => new Promise((res, rej) => texLoader.load(url, t => { t.colorSpace = THREE.SRGBColorSpace; res(t); }, undefined, rej));

// ---------------------------------------------------------------- terrain mesh (km units, north = -z, east = +x)
export function terrainMesh(h, m, exag, material) {
  const n = m.n, size = m.size_m / 1000;
  const g = new THREE.PlaneGeometry(size, size, n - 1, n - 1);
  g.rotateX(-Math.PI / 2);
  const mid = (m.min_m + m.max_m) / 2;
  const pos = g.attributes.position;
  for (let i = 0; i < n * n; i++) pos.setY(i, (h[i] - mid) * exag / 1000);
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, material);
  // a dark plinth under the tile so it reads as a block of terrain
  const lo = (m.min_m - mid) * exag / 1000 - 0.12;
  const skirt = new THREE.Group();
  const sideMat = new THREE.MeshLambertMaterial({ color: 0x3a3934 });
  const edges = [[0, 1, n], [n * (n - 1), 1, n], [0, n, n], [n - 1, n, n]];
  for (const [start, step, count] of edges) {
    const pts = [];
    for (let k = 0; k < count; k++) { const i = start + k * step; pts.push(pos.getX(i), pos.getY(i), pos.getZ(i), pos.getX(i), lo, pos.getZ(i)); }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const idx = [];
    for (let k = 0; k < count - 1; k++) { const a = 2 * k; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    sg.setIndex(idx); sg.computeVertexNormals();
    skirt.add(new THREE.Mesh(sg, new THREE.MeshLambertMaterial({ color: 0x3a3934, side: THREE.DoubleSide })));
  }
  sideMat.dispose();
  const grp = new THREE.Group(); grp.add(mesh, skirt);
  grp.userData = { mid, exag, size, lo };
  return grp;
}

// height (m) at tile ENU (x east, y north, metres from tile centre), bilinear on the 256 grid
export function heightAt(h, m, x, y) {
  const n = m.n, mpp = m.m_per_px;
  const c = Math.min(n - 1.001, Math.max(0, x / mpp + (n - 1) / 2));
  const r = Math.min(n - 1.001, Math.max(0, (n - 1) / 2 - y / mpp));
  const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0;
  const H = (rr, cc) => h[rr * n + cc];
  return (H(r0, c0) * (1 - fc) + H(r0, c0 + 1) * fc) * (1 - fr) + (H(r0 + 1, c0) * (1 - fc) + H(r0 + 1, c0 + 1) * fc) * fr;
}

// ---------------------------------------------------------------- view builders
const BUILD = {
  async elev(ctx) {
    const meta = await getMeta(), m = meta.a;
    const [h, tex] = await Promise.all([getHeights('site23', m), loadTex(BASE + m.texture)]);
    ctx.add(terrainMesh(h, m, 1.5, new THREE.MeshLambertMaterial({ map: tex })));
    ctx.frame({ dist: 8.9, elev: 0.6, az: -0.55, targetY: -0.12 });
  },
  async hazard(ctx) {
    const meta = await getMeta(), m = meta.a, b = meta.b;
    const [h, tex] = await Promise.all([getHeights('site23', m), loadTex(BASE + b.texture)]);
    ctx.add(terrainMesh(h, m, 1.5, new THREE.MeshLambertMaterial({ map: tex })));
    ctx.frame({ dist: 8.6, elev: 0.82, az: 0.35, targetY: -0.12 });
  },
  async light(ctx) {
    const meta = await getMeta(), m = meta.a, c = meta.c;
    const [h, tex] = await Promise.all([getHeights('site23', m), loadTex(BASE + c.atlas)]);
    tex.generateMipmaps = false; tex.minFilter = THREE.LinearFilter;   // no bleeding between atlas frames
    const [gx, gy] = c.grid;
    tex.repeat.set(1 / gx, 1 / gy);
    const terr = terrainMesh(h, m, 1.5, new THREE.MeshBasicMaterial({ map: tex }));
    ctx.add(terr);
    // Sun marker: direction from SPICE (compass azimuth, true elevation), drawn at 4.2 km from the centre
    const sun = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 12), new THREE.MeshBasicMaterial({ color: 0xffd27a }));
    const glow = new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 12), new THREE.MeshBasicMaterial({ color: 0xffb547, transparent: true, opacity: 0.25 }));
    sun.add(glow);
    const rayGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const ray = new THREE.Line(rayGeo, new THREE.LineDashedMaterial({ color: 0xffb547, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.7 }));
    ctx.add(sun); ctx.add(ray);
    if (ctx.dead) return;
    const ui = ctx.el.parentElement.querySelector(`.mv-time[data-for="${ctx.el.id}"]`);
    const range = ui.querySelector('input'), out = ui.querySelector('output'), btn = ui.querySelector('.mv-play');
    ui.hidden = false;
    const setFrame = i => {
      const f = c.frames[i];
      tex.offset.set((i % gx) / gx, 1 - (Math.floor(i / gx) + 1) / gy);
      const az = THREE.MathUtils.degToRad(f.sun_az_deg), el = THREE.MathUtils.degToRad(f.sun_elev_deg);
      const R = 4.2, top = (m.max_m - (m.min_m + m.max_m) / 2) * 1.5 / 1000;
      const p = new THREE.Vector3(Math.sin(az) * Math.cos(el) * R, top * 0.4 + Math.sin(el) * R, -Math.cos(az) * Math.cos(el) * R);
      sun.position.copy(p);
      rayGeo.setFromPoints([p, new THREE.Vector3(0, top * 0.4, 0)]); ray.computeLineDistances();
      const d = f.utc.replace('T', ' ').slice(0, 16);
      out.textContent = `${d} UTC · Sun ${f.sun_elev_deg >= 0 ? '+' : ''}${f.sun_elev_deg.toFixed(1)}° · ${Math.round(f.lit_fraction_tile * 100)} % of tile lit`;
      range.value = i; ctx.dirty();
    };
    let frame = Number(range.value) || 0, playing = !SHOT, acc = 0;
    const setPlaying = p => { playing = p; btn.textContent = p ? '❚❚' : '▶'; btn.setAttribute('aria-label', p ? 'Pause' : 'Play'); };
    setPlaying(playing);
    const onInput = () => { setPlaying(false); frame = Number(range.value); setFrame(frame); };
    const onBtn = () => setPlaying(!playing);
    range.addEventListener('input', onInput); btn.addEventListener('click', onBtn);
    ctx.onTick(dt => { if (!playing) return; acc += dt; if (acc > 1.1) { acc = 0; frame = (frame + 1) % c.frames.length; setFrame(frame); } });
    ctx.onDispose(() => { range.removeEventListener('input', onInput); btn.removeEventListener('click', onBtn); ui.hidden = true; });
    setFrame(frame);
    ctx.frame({ dist: 9.4, elev: 0.5, az: 0.9, targetY: 0.05 });
  },
  // Hundreds of real simulated descents (assets/data/ghosts_lander_site06.*, tools/export_ghosts_web.py) as ONE
  // additive LineSegments geometry + one Points geometry for the end points; the best run as one thick amber Line2.
  // Per-vertex simulated time "tt" drives the replay (fragment discard), per-vertex policy index drives the toggles.
  async ghosts(ctx) {
    const meta = await getMeta(), m = meta.d;
    const [h, tex, G] = await Promise.all([getHeights('site06', m), loadTex(BASE + m.texture), loadGhosts('lander', 'site06')]);
    const terr = terrainMesh(h, m, 1, new THREE.MeshLambertMaterial({ map: tex, color: 0xa6a39b }));
    ctx.add(terr);
    const { mid } = terr.userData;
    const P = G.pts;
    const W = i => [P[i] / 1000, (P[i + 2] - mid) / 1000, -P[i + 1] / 1000];     // tile ENU m -> mesh km
    const POL = { classic: 0, pilot: 1, open_loop: 2 };
    const COL = { ok: [0.56, 0.86, 0.66], bad: [0.92, 0.48, 0.46] };   // --ok / --bad, desaturated
    const nSeg = G.runs.reduce((a, r) => a + (r.best ? 0 : r.n - 1), 0);
    const pos = new Float32Array(nSeg * 6), col = new Float32Array(nSeg * 6), tt = new Float32Array(nSeg * 2), pol = new Float32Array(nSeg * 2);
    const ePos = [], eCol = [], eT = [], ePol = [];
    let s = 0, tMax = 0, best = null;
    for (const r of G.runs) {
      if (r.best) { best = r; continue; }
      const pi = POL[r.policy] ?? 0;
      // additive intensity per track, scaled so ~600 overlapping successes glow without burning out to white
      const c = r.ok ? COL.ok : COL.bad, k = r.ok ? Math.min(0.34, 5 / Math.sqrt(G.runs.length)) : (r.policy === 'open_loop' ? 0.3 : 0.6);
      const c0 = c[0] * k, c1 = c[1] * k, c2 = c[2] * k;
      for (let j = 0; j < r.n - 1; j++, s++) {
        const a = (r.start + j) * 4, b = a + 4, o = s * 6;
        pos[o] = P[a] / 1000; pos[o + 1] = (P[a + 2] - mid) / 1000; pos[o + 2] = -P[a + 1] / 1000;
        pos[o + 3] = P[b] / 1000; pos[o + 4] = (P[b + 2] - mid) / 1000; pos[o + 5] = -P[b + 1] / 1000;
        col[o] = col[o + 3] = c0; col[o + 1] = col[o + 4] = c1; col[o + 2] = col[o + 5] = c2;
        pol[s * 2] = pol[s * 2 + 1] = pi;
        tt[s * 2] = P[a + 3]; tt[s * 2 + 1] = P[b + 3];
      }
      const e = (r.start + r.n - 1) * 4;
      ePos.push(...W(e)); eCol.push(...c.map(x => x * 0.9)); eT.push(P[e + 3]); ePol.push(pi);
      tMax = Math.max(tMax, P[e + 3]);
    }
    const uni = { uTime: { value: SHOT ? 1e9 : 0 }, uMask: { value: new THREE.Vector3(...ghostMask) }, uPx: { value: Math.min(2, devicePixelRatio || 1) } };
    const VS = `attribute vec3 acol; attribute float tt; attribute float pol; uniform float uPx; varying vec3 vC; varying float vT; varying float vP;
      void main(){ vC=acol; vT=tt; vP=pol; vec4 mv=modelViewMatrix*vec4(position,1.); gl_Position=projectionMatrix*mv; gl_PointSize=uPx*clamp(9./-mv.z,1.5,4.); }`;
    const FS = pts => `uniform float uTime; uniform vec3 uMask; varying vec3 vC; varying float vT; varying float vP;
      void main(){ float m = vP<.5?uMask.x:(vP<1.5?uMask.y:uMask.z); if(m<.5||vT>uTime) discard;
      ${pts ? 'vec2 d=gl_PointCoord-.5; float r=dot(d,d); if(r>.25) discard; gl_FragColor=vec4(vC*(1.-r*3.),1.);' : 'gl_FragColor=vec4(vC,1.);'} }`;
    const mk = pts => new THREE.ShaderMaterial({ uniforms: uni, vertexShader: VS, fragmentShader: FS(pts), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('acol', new THREE.BufferAttribute(col, 3));
    geo.setAttribute('tt', new THREE.BufferAttribute(tt, 1)); geo.setAttribute('pol', new THREE.BufferAttribute(pol, 1));
    const segs = new THREE.LineSegments(geo, mk(false)); segs.frustumCulled = false; segs.renderOrder = 2;
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(ePos, 3)); eg.setAttribute('acol', new THREE.Float32BufferAttribute(eCol, 3));
    eg.setAttribute('tt', new THREE.Float32BufferAttribute(eT, 1)); eg.setAttribute('pol', new THREE.Float32BufferAttribute(ePol, 1));
    const ends = new THREE.Points(eg, mk(true)); ends.frustumCulled = false; ends.renderOrder = 3;
    ctx.add(segs); ctx.add(ends);
    // the chosen ("best") run: one thick amber line, grows with the same clock
    let bestGeo = null, bestT = [];
    if (best) {
      const bp = [];
      for (let j = 0; j < best.n; j++) { const a = (best.start + j) * 4; bp.push(...W(a)); bestT.push(P[a + 3]); }
      bestGeo = new LineGeometry(); bestGeo.setPositions(bp);
      const bm = new LineMaterial({ color: 0xffb547, linewidth: 3, transparent: true, opacity: 1, depthWrite: false });
      ctx.lineMats.push(bm);
      const bl = new Line2(bestGeo, bm); bl.renderOrder = 4; ctx.add(bl);
      tMax = Math.max(tMax, bestT[bestT.length - 1]);
    }
    // target: ring on the ground + faint beam (tile ENU target from the header)
    const [tx, ty, tz] = G.head.target_xyz_m;
    const tg = new THREE.Vector3(tx / 1000, (tz - mid) / 1000, -ty / 1000);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.085, 40), new THREE.MeshBasicMaterial({ color: 0xffb547, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.copy(tg).y += 0.006;
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 2.3, 8), new THREE.MeshBasicMaterial({ color: 0xffb547, transparent: true, opacity: 0.25 }));
    beam.position.copy(tg).y += 1.15;
    ctx.add(ring); ctx.add(beam);
    // toggles (classic / pilot / open loop) live in the caption; state is module-level so it survives tab switches
    ghostOnMask = () => { uni.uMask.value.set(...ghostMask); ctx.dirty(); };
    ctx.onDispose(() => { ghostOnMask = null; });
    // replay: every run starts at simulated t = 0 and grows at the same clock; hold, repeat
    const SPAN = 7, HOLD = 3.5, rate = tMax / SPAN;
    // screenshot mode (?scroll=..., html.shot): start on the finished picture
    let t = SHOT ? 1e9 : document.documentElement.classList.contains('shot') ? SPAN : 0;
    const apply = () => {
      const st = Math.min(t, SPAN) * rate;
      uni.uTime.value = SHOT ? 1e9 : st;
      if (bestGeo) { let k = 0; while (k < bestT.length && bestT[k] <= st + 1e-6) k++; bestGeo.instanceCount = Math.max(0, k - 1); }
    };
    apply();
    if (!SHOT && t === 0) ctx.onTick(dt => { t += dt; if (t > SPAN + HOLD) t = 0; apply(); ctx.dirty(); });
    ctx.frame({ dist: 6.6, elev: 0.26, az: 0.62, target: new THREE.Vector3(tg.x * 0.8, tg.y + 1.05, tg.z * 0.8) });
  },
};

// ---------------------------------------------------------------- one viewer
function createViewer(el, kind) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power', preserveDrawingBuffer: !!SHOT });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(BG, 1);
  const canvas = renderer.domElement;
  canvas.className = 'mv-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  el.appendChild(canvas);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(BG, 14, 30);
  scene.add(new THREE.HemisphereLight(0xf2efe6, 0x3a3833, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(-3, 4, 2.5); scene.add(sun);
  const camera = new THREE.PerspectiveCamera(36, 1.6, 0.05, 80);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.enableZoom = false; controls.enablePan = false;
  controls.minDistance = 2.5; controls.maxDistance = 16; controls.maxPolarAngle = Math.PI * 0.48;
  controls.autoRotate = !SHOT; controls.autoRotateSpeed = 0.55;
  controls.touches.ONE = THREE.TOUCH.ROTATE; controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE;
  canvas.style.touchAction = 'pan-y';          // until activated, vertical swipes still scroll the page
  controls.addEventListener('start', () => { controls.autoRotate = false; });

  const ticks = [], disposers = [];
  let needs = true;
  const ctx = {
    el,
    add: o => scene.add(o),
    dirty: () => { needs = true; },
    onTick: f => ticks.push(f),
    onDispose: f => disposers.push(f),
    lineMats: [],
    frame({ dist, elev, az, targetY = 0, target }) {
      const tgt = target || new THREE.Vector3(0, targetY, 0);
      controls.target.copy(tgt);
      camera.position.set(tgt.x + Math.sin(az) * Math.cos(elev) * dist, tgt.y + Math.sin(elev) * dist, tgt.z + Math.cos(az) * Math.cos(elev) * dist);
      controls.update(); controls.saveState(); needs = true;
    },
  };
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); needs = true;
    ctx.lineMats.forEach(mt => mt.resolution.set(w, h));
  };
  const ro = new ResizeObserver(resize); ro.observe(el); resize();

  let active = false;
  const activate = () => { if (active) return; active = true; controls.enableZoom = true; canvas.style.touchAction = 'none'; el.classList.add('active'); };
  const deactivate = () => { if (!active) return; active = false; controls.enableZoom = false; canvas.style.touchAction = 'pan-y'; el.classList.remove('active'); };
  const onDocDown = e => { if (!el.contains(e.target)) deactivate(); };
  const onKey = e => { if (e.key === 'Escape') deactivate(); };
  canvas.addEventListener('pointerdown', activate);
  const onBlur = e => { if (!el.contains(e.relatedTarget)) deactivate(); };
  el.addEventListener('focus', activate);
  el.addEventListener('blur', onBlur);
  document.addEventListener('pointerdown', onDocDown, true);
  el.addEventListener('keydown', onKey);

  const reset = document.createElement('button');
  reset.type = 'button'; reset.className = 'mv-reset'; reset.textContent = 'Reset view';
  reset.addEventListener('click', () => { controls.reset(); controls.autoRotate = true; needs = true; });
  const hint = document.createElement('span');
  hint.className = 'mv-hint'; hint.setAttribute('aria-hidden', 'true');
  el.append(reset, hint);

  let ready = false, failed = false;
  const built = BUILD[kind](ctx).then(() => { if (ctx.dead) return; resize(); ready = true; el.classList.add('ready'); needs = true; })
    .catch(e => { if (ctx.dead) return; failed = true; console.warn('models3d', kind, e); el.classList.add('failed'); });

  return {
    el, built,
    get ready() { return ready; },
    deactivate,
    render(dt) {
      if (!ready || failed) return;
      for (const f of ticks) f(dt);
      const moved = controls.update();
      if (moved || needs || controls.autoRotate || SHOT) { renderer.render(scene, camera); needs = false; if (SHOT) el.dataset.frames = String((+el.dataset.frames || 0) + 1); }   // SHOT: keep presenting frames (and touch the DOM) so the headless capture sees them
    },
    dispose() {
      ctx.dead = true;
      disposers.forEach(f => f());
      ro.disconnect(); controls.dispose();
      document.removeEventListener('pointerdown', onDocDown, true);
      el.removeEventListener('keydown', onKey);
      el.removeEventListener('focus', activate); el.removeEventListener('blur', onBlur);
      canvas.removeEventListener('pointerdown', activate);
      scene.traverse(o => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) [].concat(o.material).forEach(mt => { if (mt.map) mt.map.dispose(); mt.dispose(); });
      });
      renderer.dispose(); renderer.forceContextLoss();
      canvas.remove(); reset.remove(); hint.remove();
      el.classList.remove('ready', 'active', 'failed');
    },
  };
}

// ---------------------------------------------------------------- section manager: tabs + one stage
// One WebGL context at most: the stage's viewer is disposed on every tab switch and when the stage is far away.
const TAB_BY_HASH = { 'view-elevation': 'elev', 'view-hazard': 'hazard', 'view-sunlight': 'light', 'view-ghosts': 'ghosts' };
const TAB_BY_PARAM = { elevation: 'elev', elev: 'elev', hazard: 'hazard', sunlight: 'light', light: 'light', ghosts: 'ghosts' };

export function mountModels(section, { reducedMotion = false, use3D = true } = {}) {
  if (!section) return {};
  const stage = section.querySelector('#mv-stage');
  const tabs = [...section.querySelectorAll('.mv-tab')];
  const caps = [...section.querySelectorAll('.mv-cap')];
  if (!stage || !tabs.length) return {};
  const still = stage.querySelector('.mv-still'), label = stage.querySelector('.mv-label');
  const live3D = use3D && !reducedMotion;
  section.classList.add(live3D ? 'mv-live' : 'mv-static');

  let kind = null, viewer = null, visible = false, dist = 1e9, managed = false, raf = 0, last = 0;

  function select(k, { focus = false } = {}) {
    if (!k || k === kind) return;
    kind = k;
    tabs.forEach(t => {
      const on = t.dataset.view === k;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      if (on) { stage.setAttribute('aria-labelledby', t.id); if (focus) t.focus(); }
    });
    let cap = null;
    caps.forEach(c => { const on = c.dataset.view === k; c.hidden = !on; if (on) cap = c; });
    stage.dataset.view = k;
    if (cap) {
      still.src = cap.dataset.still; still.alt = cap.dataset.alt || '';
      label.textContent = cap.dataset.label || '';
      stage.setAttribute('aria-label', (cap.dataset.alt || '') + (live3D ? ' Drag to rotate; click or focus to enable zoom.' : ''));
    }
    if (viewer) { viewer.dispose(); viewer = null; }
    if (managed) check();
  }

  // hand-drawn ring on each tab (drawn in by CSS when the tab is selected)
  const RING = 'M 8 9 C 30 1, 78 0, 94 10 C 102 17, 98 31, 78 36 C 52 41, 18 40, 6 31 C -1 25, 1 14, 14 7';
  tabs.forEach(t => {
    if (t.querySelector('.mv-ring')) return;
    t.insertAdjacentHTML('beforeend', `<svg class="mv-ring" viewBox="0 0 100 40" preserveAspectRatio="none" aria-hidden="true"><path pathLength="1" d="${RING}"/></svg>`);
  });
  // tabs: click + arrow keys (roving tabindex)
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t.dataset.view));
    t.addEventListener('keydown', e => {
      let j = null;
      if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = tabs.length - 1;
      if (j === null) return;
      e.preventDefault(); select(tabs[j].dataset.view, { focus: true });
    });
  });

  // ghosts tab: policy toggles (classic / pilot / open loop) + every count from the data header
  section.querySelectorAll('.mv-gtog').forEach(b => {
    const i = { classic: 0, pilot: 1, open_loop: 2 }[b.dataset.pol];
    if (i == null) return;
    b.setAttribute('aria-pressed', ghostMask[i] ? 'true' : 'false');
    b.addEventListener('click', () => {
      ghostMask[i] = ghostMask[i] ? 0 : 1;
      b.setAttribute('aria-pressed', ghostMask[i] ? 'true' : 'false');
      ghostOnMask && ghostOnMask();
    });
  });
  const gcap = section.querySelector('.mv-cap[data-view="ghosts"]');
  if (gcap) loadGhostHead('lander', 'site06').then(hd => fillCounts(gcap, hd)).catch(e => console.warn('ghost counts', e));

  // deep links: #view-hazard etc. (nav dropdown), also when the hash is already set
  const fromHash = (scroll) => {
    const k = TAB_BY_HASH[location.hash.slice(1)];
    if (!k) return;
    select(k);
    if (scroll) section.querySelector('.mv-tabs').scrollIntoView({ block: 'start', behavior: 'auto' });
  };
  window.addEventListener('hashchange', () => fromHash(true));
  document.addEventListener('click', e => {
    const a = e.target.closest && e.target.closest('a[href^="#view-"]');
    if (a && a.getAttribute('href') === location.hash) fromHash(true);   // same hash: no hashchange event
    else if (a) select(TAB_BY_HASH[a.getAttribute('href').slice(1)]);
  });
  const qp = TAB_BY_PARAM[params.get('tab')] || (SHOT && TAB_BY_PARAM[SHOT]) || TAB_BY_HASH[location.hash.slice(1)] || 'elev';
  select(qp);

  if (!live3D) return { select };

  if (SHOT) {                                   // still-image mode: the stage fills the window, no motion
    document.documentElement.classList.add('mv-shot-mode');
    stage.classList.add('mv-shot');
    viewer = createViewer(stage, kind);
    const loop = () => { viewer.render(0.016); requestAnimationFrame(loop); };
    viewer.built.then(loop);
    return { select };
  }

  function check() {
    const vh = window.innerHeight;
    const r = stage.getBoundingClientRect();
    dist = r.bottom < 0 ? -r.bottom : r.top > vh ? r.top - vh : 0;
    visible = dist === 0 && r.width > 0;
    if (viewer && dist > 1.5 * vh) { viewer.dispose(); viewer = null; }
    if (!viewer && dist < 0.6 * vh) viewer = createViewer(stage, kind);
    if (viewer && !visible) viewer.deactivate();
    if (viewer && visible) start();
  }

  function loop(t) {
    raf = 0;
    const dt = last ? Math.min(0.1, (t - last) / 1000) : 0.016; last = t;
    if (viewer && visible) { viewer.render(dt); raf = requestAnimationFrame(loop); } else last = 0;
  }
  function start() { if (!raf) raf = requestAnimationFrame(loop); }

  let pending = false;
  const onScroll = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; check(); }); };
  // capture: also catches the body scroller used in screenshot mode (?scroll=...)
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  window.addEventListener('resize', onScroll);
  setInterval(check, 700);
  managed = true;
  check();
  return { select, check };
}
