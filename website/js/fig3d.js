// Interactive 3D figures inside research posts. Reuses the terrain helpers of js/models3d.js.
//
// Markup (one or two per post):
//   <figure class="wide f3d" id="f3d-hazard" data-f3d="hazard:drape">
//     <div class="f3d-stage" tabindex="0" role="img" aria-label="...">
//       <img class="f3d-still" src="assets/fig3d/hazard/still_drape.jpg" width="1280" height="720" loading="lazy" alt="...">
//     </div>
//     <div class="f3d-ctl"></div>
//     <figcaption>... Source: ...</figcaption>
//   </figure>
// data-f3d="<module>:<builder>" loads js/fig3d/<module>.js and calls its exported async <builder>(ctx).
//
// Rules: ONE WebGL renderer per page (its canvas moves to whichever figure is nearest the viewport centre; the
// other figure shows its still image), built lazily when a figure comes within ~0.6 viewport heights, scene
// disposed when it leaves. Reduced motion, phones and no-WebGL get the still image only (controls hidden).
// ?f3dshot=<figure id> renders that one figure full-window without motion (used to make the still images).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { terrainMesh, heightAt } from './models3d.js';

export { THREE, terrainMesh, heightAt };

const BG = 0x262624;
const params = new URLSearchParams(location.search);
const SHOT = params.get('f3dshot');

// ---------------------------------------------------------------- shared loaders (cached per URL)
const cache = {};
export const loadJSON = url => (cache[url] ||= fetch(url).then(r => { if (!r.ok) throw new Error(r.status + ' ' + url); return r.json(); }));
export const loadBin = url => (cache[url] ||= fetch(url).then(r => { if (!r.ok) throw new Error(r.status + ' ' + url); return r.arrayBuffer(); }));
// uint16 little-endian heights, row 0 = north, h = min + v/65535*(max-min)  (same encoding as assets/models3d/*_height.bin)
export async function loadHeights(url, m) {
  const v = new Uint16Array(await loadBin(url)), out = new Float32Array(v.length), k = (m.max_m - m.min_m) / 65535;
  for (let i = 0; i < v.length; i++) out[i] = m.min_m + v[i] * k;
  return out;
}
const texLoader = new THREE.TextureLoader();
export const loadTex = url => new Promise((res, rej) => texLoader.load(url, t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; res(t); }, undefined, rej));

// tile ENU metres (x east, y north, z height) -> mesh km (north = -z), relative to the mesh's mid height
export const toMesh = (terr, x, y, z, lift = 0) => new THREE.Vector3(x / 1000, (z - terr.userData.mid) * terr.userData.exag / 1000 + lift, -y / 1000);

// a fat polyline (Line2). points: flat [x,y,z,...] in mesh units. colors: optional flat [r,g,b,...] (0..1)
export function fatLine(ctx, points, { color = 0xffffff, colors = null, width = 2.5, opacity = 1 } = {}) {
  const g = new LineGeometry(); g.setPositions(points); if (colors) g.setColors(colors);
  const mat = new LineMaterial({ color: colors ? 0xffffff : color, vertexColors: !!colors, linewidth: width, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
  ctx.lineMats.push(mat); ctx.resizeLines();
  const l = new Line2(g, mat); l.computeLineDistances(); return l;
}
// a small upright marker: pole + head, at mesh position p
export function marker(p, color = 0xffb547, h = 0.35, r = 0.045) {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, h, 6), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 }));
  pole.position.set(p.x, p.y + h / 2, p.z);
  const head = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 10), new THREE.MeshBasicMaterial({ color }));
  head.position.set(p.x, p.y + h, p.z);
  g.add(pole, head); return g;
}

// ---------------------------------------------------------------- the one renderer
let R = null;
function renderer() {
  if (R) return R;
  const r = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power', preserveDrawingBuffer: !!SHOT });
  r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  r.setClearColor(BG, 1);
  r.domElement.className = 'f3d-canvas';
  r.domElement.setAttribute('aria-hidden', 'true');
  return (R = r);
}

// ---------------------------------------------------------------- one live figure (scene + camera + controls + UI)
function createScene(fig) {
  const stage = fig.querySelector('.f3d-stage'), ctl = fig.querySelector('.f3d-ctl');
  const [mod, fn] = fig.dataset.f3d.split(':');
  const state = (fig._f3dState ||= {});
  const rnd = renderer(), canvas = rnd.domElement;
  stage.appendChild(canvas);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(BG, 16, 34);
  scene.add(new THREE.HemisphereLight(0xf2efe6, 0x3a3833, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(-3, 4, 2.5); scene.add(sun);
  const camera = new THREE.PerspectiveCamera(36, 16 / 9, 0.02, 90);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.enableZoom = false; controls.enablePan = false;
  controls.minDistance = 1.2; controls.maxDistance = 18; controls.maxPolarAngle = Math.PI * 0.48;
  controls.autoRotate = !SHOT && !state.userMoved; controls.autoRotateSpeed = 0.45;
  controls.touches.ONE = THREE.TOUCH.ROTATE; controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE;
  canvas.style.touchAction = 'pan-y';
  const onStart = () => { controls.autoRotate = false; state.userMoved = true; };
  controls.addEventListener('start', onStart);

  const ticks = [], disposers = [], uiNodes = [];
  let needs = true, dead = false;
  const ctx = {
    THREE, fig, stage, scene, camera, controls, state, shot: !!SHOT,
    get dead() { return dead; },
    add: o => scene.add(o),
    remove: o => scene.remove(o),
    sun,
    dirty: () => { needs = true; },
    onTick: f => ticks.push(f),
    onDispose: f => disposers.push(f),
    lineMats: [],
    resizeLines() { const w = stage.clientWidth, h = stage.clientHeight; ctx.lineMats.forEach(m => m.resolution.set(w, h)); },
    // camera framing; saved as the Reset view state. Later calls (e.g. after the user moved) keep the user's view.
    frame({ dist, elev, az, targetY = 0, target }) {
      const tgt = target || new THREE.Vector3(0, targetY, 0);
      controls.target.copy(tgt);
      camera.position.set(tgt.x + Math.sin(az) * Math.cos(elev) * dist, tgt.y + Math.sin(elev) * dist, tgt.z + Math.cos(az) * Math.cos(elev) * dist);
      controls.update(); controls.saveState();
      if (state.cam) { camera.position.fromArray(state.cam.p); controls.target.fromArray(state.cam.t); controls.update(); }
      needs = true;
    },
    // segmented toggle; value persists for this figure across rebuilds. Returns the current value.
    toggle(name, label, options, onChange, initial) {
      if (state[name] === undefined) state[name] = initial ?? options[0][0];
      const wrap = document.createElement('div');
      wrap.className = 'f3d-seg'; wrap.setAttribute('role', 'group'); wrap.setAttribute('aria-label', label);
      const lab = document.createElement('span'); lab.className = 'f3d-seg-l'; lab.textContent = label; wrap.appendChild(lab);
      const btns = options.map(([v, text]) => {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.dataset.v = v;
        b.setAttribute('aria-pressed', String(state[name] === v));
        b.addEventListener('click', () => {
          if (state[name] === v) return;
          state[name] = v; btns.forEach(x => x.setAttribute('aria-pressed', String(x.dataset.v === String(v))));
          onChange(v); needs = true;
        });
        wrap.appendChild(b); return b;
      });
      ctl.appendChild(wrap); uiNodes.push(wrap);
      onChange(state[name]);
      return state[name];
    },
    // on/off checkbox-style button
    check(name, text, onChange, initial = true) {
      if (state[name] === undefined) state[name] = initial;
      const b = document.createElement('button'); b.type = 'button'; b.className = 'f3d-chk'; b.textContent = text;
      b.setAttribute('aria-pressed', String(!!state[name]));
      b.addEventListener('click', () => { state[name] = !state[name]; b.setAttribute('aria-pressed', String(state[name])); onChange(state[name]); needs = true; });
      ctl.appendChild(b); uiNodes.push(b);
      onChange(state[name]);
      return state[name];
    },
    // time slider with play/pause. format(i) -> readout text. onInput(i). Plays at `fps` steps per second.
    slider(name, label, { min = 0, max, step = 1, value, fps = 1, play = true, format, onInput }) {
      if (state[name] === undefined) state[name] = value ?? min;
      const wrap = document.createElement('div'); wrap.className = 'f3d-time';
      const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'f3d-play';
      const rng = document.createElement('input'); rng.type = 'range'; rng.min = min; rng.max = max; rng.step = step; rng.value = state[name];
      rng.setAttribute('aria-label', label);
      const out = document.createElement('output'); out.className = 'f3d-out';
      wrap.append(btn, rng, out); ctl.appendChild(wrap); uiNodes.push(wrap);
      let playing = play && !SHOT && state[name + ':playing'] !== false, acc = 0;
      const setP = p => { playing = p; state[name + ':playing'] = p; btn.textContent = p ? '❚❚' : '▶'; btn.setAttribute('aria-label', p ? 'Pause' : 'Play'); };
      const set = v => { state[name] = v; rng.value = v; out.textContent = format ? format(v) : String(v); onInput(v); needs = true; };
      setP(playing);
      rng.addEventListener('input', () => { setP(false); set(Number(rng.value)); });
      btn.addEventListener('click', () => setP(!playing));
      ticks.push(dt => { if (!playing) return; acc += dt; if (acc >= 1 / fps) { acc = 0; let v = state[name] + step; if (v > max + 1e-9) v = min; set(v); } });
      set(state[name]);
      return { set, setPlaying: setP };
    },
    // colour key: [[css colour, text], ...]
    legend(items) {
      const d = document.createElement('div'); d.className = 'f3d-key';
      d.innerHTML = items.map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join('');
      ctl.appendChild(d); uiNodes.push(d); return d;
    },
    readout() { const o = document.createElement('p'); o.className = 'f3d-read'; ctl.appendChild(o); uiNodes.push(o); return o; },
  };

  const resize = () => {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    rnd.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); ctx.resizeLines(); needs = true;
  };
  const ro = new ResizeObserver(resize); ro.observe(stage); resize();

  // zoom only after the figure is clicked / focused, so the page still scrolls over it
  let active = false;
  const activate = () => { if (active) return; active = true; controls.enableZoom = true; canvas.style.touchAction = 'none'; fig.classList.add('f3d-active'); };
  const deactivate = () => { if (!active) return; active = false; controls.enableZoom = false; canvas.style.touchAction = 'pan-y'; fig.classList.remove('f3d-active'); };
  const onDocDown = e => { if (!stage.contains(e.target)) deactivate(); };
  const onKey = e => { if (e.key === 'Escape') deactivate(); };
  const onBlur = e => { if (!stage.contains(e.relatedTarget)) deactivate(); };
  canvas.addEventListener('pointerdown', activate);
  stage.addEventListener('focus', activate); stage.addEventListener('blur', onBlur); stage.addEventListener('keydown', onKey);
  document.addEventListener('pointerdown', onDocDown, true);

  const reset = document.createElement('button');
  reset.type = 'button'; reset.className = 'f3d-reset'; reset.textContent = 'Reset view';
  reset.addEventListener('click', () => { delete state.cam; state.userMoved = false; controls.reset(); controls.autoRotate = !SHOT; needs = true; });
  const hint = document.createElement('span'); hint.className = 'f3d-hint'; hint.setAttribute('aria-hidden', 'true');
  const spin = document.createElement('span'); spin.className = 'f3d-load'; spin.textContent = 'Loading 3D…';
  stage.append(reset, hint, spin);

  let ready = false, failed = false;
  const built = import(`./fig3d/${mod}.js`).then(M => {
    if (typeof M[fn] !== 'function') throw new Error(`fig3d: ${mod}.${fn} missing`);
    return M[fn](ctx);
  }).then(() => { if (dead) return; resize(); ready = true; fig.classList.add('f3d-ready'); spin.remove(); needs = true; })
    .catch(e => { if (dead) return; failed = true; console.warn('fig3d', fig.id, e); fig.classList.add('f3d-failed'); spin.remove(); });

  return {
    fig, built,
    deactivate,
    render(dt) {
      if (!ready || failed) return;
      for (const f of ticks) f(dt);
      const moved = controls.update();
      if (moved || needs || controls.autoRotate || SHOT) {
        rnd.render(scene, camera); needs = false;
        if (SHOT) stage.dataset.frames = String((+stage.dataset.frames || 0) + 1);
      }
    },
    dispose() {
      dead = true;
      if (state.userMoved) state.cam = { p: camera.position.toArray(), t: controls.target.toArray() };
      disposers.forEach(f => { try { f(); } catch (e) { console.warn(e); } });
      ro.disconnect(); controls.removeEventListener('start', onStart); controls.dispose();
      document.removeEventListener('pointerdown', onDocDown, true);
      stage.removeEventListener('focus', activate); stage.removeEventListener('blur', onBlur); stage.removeEventListener('keydown', onKey);
      canvas.removeEventListener('pointerdown', activate);
      scene.traverse(o => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) [].concat(o.material).forEach(mt => { for (const k of ['map', 'alphaMap', 'emissiveMap']) if (mt[k]) mt[k].dispose(); mt.dispose(); });
      });
      uiNodes.forEach(n => n.remove());
      reset.remove(); hint.remove(); spin.remove();
      if (canvas.parentNode === stage) canvas.remove();
      fig.classList.remove('f3d-ready', 'f3d-active', 'f3d-failed');
    },
  };
}

// ---------------------------------------------------------------- page manager
function webglOK() {
  try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
}

export function mountFig3D(root = document) {
  const figs = [...root.querySelectorAll('figure.f3d[data-f3d]')];
  if (!figs.length) return;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const phone = matchMedia('(max-width: 720px), (pointer: coarse) and (max-width: 1024px)').matches;
  const live = !!SHOT || (!reduce && !phone && webglOK());
  figs.forEach(f => f.classList.add(live ? 'f3d-live' : 'f3d-static'));
  if (!live) return;

  if (SHOT) {
    const fig = figs.find(f => f.id === SHOT);
    if (!fig) return;
    document.documentElement.classList.add('f3d-shot-mode'); fig.classList.add('f3d-shot');
    // a transformed ancestor (figure.wide uses translateX) would trap position: fixed inside it
    for (let a = fig; a && a !== document.documentElement; a = a.parentElement) { a.style.transform = 'none'; a.style.filter = 'none'; a.style.contain = 'none'; a.style.animation = 'none'; }
    const v = createScene(fig);
    const loop = () => { v.render(0.016); requestAnimationFrame(loop); };
    v.built.then(loop);
    return;
  }

  let cur = null, raf = 0, last = 0, visible = false;
  function check() {
    const vh = innerHeight, mid = vh / 2;
    let best = null, bestD = Infinity;
    for (const f of figs) {
      const r = f.querySelector('.f3d-stage').getBoundingClientRect();
      const d = r.bottom < 0 ? -r.bottom : r.top > vh ? r.top - vh : 0;
      if (d > 0.6 * vh || !r.width) continue;
      const c = Math.abs((r.top + r.bottom) / 2 - mid);
      if (c < bestD) { bestD = c; best = f; }
    }
    if (cur) {
      const r = cur.fig.querySelector('.f3d-stage').getBoundingClientRect();
      const d = r.bottom < 0 ? -r.bottom : r.top > vh ? r.top - vh : 0;
      const c = Math.abs((r.top + r.bottom) / 2 - mid);
      // switch when another figure is clearly nearer the centre (hysteresis), or this one is far away
      if (d > 1.5 * vh || (best && best !== cur.fig && (d > 0 || c - bestD > 0.35 * vh))) { cur.dispose(); cur = null; }
    }
    if (!cur && best) cur = createScene(best);
    if (cur) {
      const r = cur.fig.querySelector('.f3d-stage').getBoundingClientRect();
      visible = r.bottom > 0 && r.top < vh;
      if (!visible) cur.deactivate(); else start();
    }
  }
  function loop(t) {
    raf = 0;
    const dt = last ? Math.min(0.1, (t - last) / 1000) : 0.016; last = t;
    if (cur && visible && !document.hidden) { cur.render(dt); raf = requestAnimationFrame(loop); } else last = 0;
  }
  function start() { if (!raf) raf = requestAnimationFrame(loop); }
  let pending = false;
  const onScroll = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; check(); }); };
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  addEventListener('resize', onScroll);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) check(); });
  setInterval(check, 800);
  check();
}

mountFig3D();
