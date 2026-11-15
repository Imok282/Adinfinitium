// Home hero (company site): the lander over the Moon limb, cursor-reactive, own small three.js scene in its own canvas.
//   mountHero3D(el): adds a canvas inside el (the poster / existing content stays underneath and is kept for reduced
//   motion or no WebGL). One rAF loop, paused when el is off-screen (IntersectionObserver) or the tab is hidden. DPR <= 2.
//   Pointer -> target rotation (rx = -y * 0.08, ry = x * 0.12) + small camera offset, lerped with k = 1 - exp(-dt * 4).
//   Depth layers: far stars, Moon limb, propulsion module + lander, near dust (5th layer = page text, js/home.js).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));

function hasWebGL() {
  try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
}

export function mountHero3D(el, opts = {}) {
  if (!el) return null;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced || !hasWebGL() || new URLSearchParams(location.search).has('static')) return null;   // keep the poster
  try {
    return start(el, opts);
  } catch (e) { console.warn('hero3d', e); return null; }
}

function start(el, opts) {
  if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', display: 'block', opacity: '0', transition: 'opacity .9s cubic-bezier(.2,.7,.2,1)' });
  el.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(0x262624, 0);              // transparent: the page's warm charcoal shows through

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(30, 1, 0.1, 400);
  const SUN = new THREE.Vector3(1, 0.16, 0.35).normalize();   // grazing light, like the lunar south pole
  const sun = new THREE.DirectionalLight(0xfff1dd, 4.0); sun.position.copy(SUN).multiplyScalar(30); scene.add(sun);
  scene.add(new THREE.AmbientLight(0xc9c6bf, 0.38));
  const rim = new THREE.DirectionalLight(0xe8dccb, 0.8); rim.position.set(-6, 3, -6); scene.add(rim);

  // Moon limb: a big grey sphere below the lander
  const tl = new THREE.TextureLoader();
  const moonMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0, bumpScale: 1.5 });
  const moon = new THREE.Mesh(new THREE.SphereGeometry(30, 160, 100), moonMat);
  moon.position.set(0, -32.4, -6);
  moon.rotation.set(0.5, 0, 0.2);
  scene.add(moon);
  tl.load('assets/img/moon_color_2k.jpg', t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; moonMat.map = t; moonMat.needsUpdate = true; });
  tl.load('assets/img/moon_bump_1k.jpg', t => { moonMat.bumpMap = t; moonMat.needsUpdate = true; });

  { // a few stars
    const n = 600, p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const v = new THREE.Vector3().randomDirection(); v.z = -Math.abs(v.z); v.y = Math.abs(v.y) * 0.9 + 0.05; v.multiplyScalar(150); p.set([v.x, v.y, v.z], i * 3); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xf0eee6, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0.75, depthWrite: false })));
  }

  // lander (feet at y = 0 of `pivot`)
  const rig = new THREE.Group(), pivot = new THREE.Group();
  rig.add(pivot); scene.add(rig);
  let landerH = 2.9;
  new GLTFLoader().load('assets/models/lander.glb', g => {
    const m = g.scene;
    const box = new THREE.Box3().setFromObject(m), c = box.getCenter(new THREE.Vector3());
    m.position.set(-c.x, -box.min.y, -c.z);
    landerH = box.max.y - box.min.y;
    pivot.add(m);
    pivot.position.y = -landerH * 0.5;            // rotate about the lander's middle
    puff.points.position.y = landerH * 0.55;
    canvas.style.opacity = '1';
  }, undefined, () => {});
  // propulsion module stacked on top of the lander (the attached stack)
  new GLTFLoader().load('assets/models/propulsion_module.glb', g => {
    const m = g.scene, box = new THREE.Box3().setFromObject(m), c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
    const k = 0.62 * landerH / Math.max(sz.y, 0.001);
    m.scale.setScalar(k); m.position.set(-c.x * k, landerH - box.min.y * k, -c.z * k);
    pivot.add(m);
  }, undefined, () => {});

  // near dust: slow drifting motes close to the camera (strongest parallax with scroll and cursor)
  const dust = (() => {
    const n = 220, p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) p.set([(Math.random() - 0.5) * 16, (Math.random() - 0.5) * 9, 2 + Math.random() * 6], i * 3);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    const pts = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xe6dcc8, size: 0.035, transparent: true, opacity: 0.55, depthWrite: false }));
    scene.add(pts); return pts;
  })();

  // RCS puff on click: a short burst of soft particles from four thruster quads
  const puff = (() => {
    const n = 96, p = new Float32Array(n * 3), seed = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) { const q = (i % 4) * Math.PI / 2 + 0.785; seed.set([q + (Math.random() - 0.5) * 0.5, 0.6 + Math.random(), (Math.random() - 0.5) * 0.6, Math.random()], i * 4); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const cx = cv.getContext('2d'), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
    const mat = new THREE.PointsMaterial({ color: 0xe6dcc8, size: 0.35, map: new THREE.CanvasTexture(cv), transparent: true, opacity: 0, depthWrite: false });
    const points = new THREE.Points(g, mat); points.frustumCulled = false; points.visible = false;
    pivot.add(points);
    let t0 = -1;
    return {
      points,
      fire(t) { t0 = t; },
      update(t) {
        const k = t0 < 0 ? 1 : (t - t0) / 0.7;
        points.visible = k < 1; if (!points.visible) return;
        mat.opacity = 0.8 * (1 - k);
        for (let i = 0; i < n; i++) {
          const a = seed[i * 4], r = 1.1 + seed[i * 4 + 1] * 1.6 * Math.sqrt(k);
          p.set([Math.cos(a) * r, seed[i * 4 + 2] * k, Math.sin(a) * r], i * 3);
        }
        g.attributes.position.needsUpdate = true;
      },
    };
  })();

  // ---------- pointer (relative to el), lerped ----------
  let W = 1, H = 1, px = 0, py = 0;
  const cur = { rx: 0, ry: 0, cx: 0, cy: 0 };
  el.addEventListener('pointermove', e => {
    const r = el.getBoundingClientRect();
    px = clamp(((e.clientX - r.left) / Math.max(1, r.width)) * 2 - 1, -1, 1);
    py = clamp(((e.clientY - r.top) / Math.max(1, r.height)) * 2 - 1, -1, 1);
  }, { passive: true });
  el.addEventListener('pointerleave', () => { px = 0; py = 0; }, { passive: true });

  const base = { x: 2.7, y: 0.85, s: 0.55 };
  // Moon effects (js/moonfx.js): one element per page segment, sequenced by js/home.js via setSegments()
  let fx = null, baseQ = null;
  const seg = {}, UP = new THREE.Vector3(0, 1, 0), spinQ = new THREE.Quaternion();
  import('./moonfx.js').then(m => {
    fx = m.createMoonFX(THREE, scene, moon, { sunDir: SUN, radius: 30, patchDeg: 6, swarm: "closed" });
    baseQ = fx.orientToSite(-87.5, 20, new THREE.Vector3(0, 0.93, -0.36));   // south-pole sites on the upper limb, foreshortened toward the horizon
  }).catch(e => console.warn('moonfx', e));
  // page state from js/home.js: hero 0..1 (card opening), f 0..1 (whole page), big 0..1 (big-type section), end 0..1 (closing reveal)
  const st = { hero: 0, f: 0, big: 0, end: 0 }, sm = { hero: 0, f: 0, big: 0, end: 0 };
  function resize() {
    const r = el.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
    renderer.setSize(W, H, false);
    cam.aspect = W / H; cam.updateProjectionMatrix();
    const narrow = W / H < 0.9;
    base.x = narrow ? 0 : 2.7; base.y = narrow ? 0.75 : -0.1; base.s = narrow ? 0.26 : 0.36;   // right of the centred headline, fully in frame
    rig.position.set(base.x, base.y, 0); rig.scale.setScalar(base.s);
  }
  new ResizeObserver(resize).observe(el);
  resize();

  const clock = new THREE.Clock();
  let onScreen = true, raf = 0, scrollS = 0;
  function step() {
    const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
    const k = 1 - Math.exp(-dt * 4);
    cur.rx += (-py * 0.08 - cur.rx) * k; cur.ry += (px * 0.12 - cur.ry) * k;
    cur.cx += (px * 0.35 - cur.cx) * k; cur.cy += (-py * 0.2 - cur.cy) * k;
    for (const key in st) sm[key] += (st[key] - sm[key]) * Math.min(1, k * 2);
    const big = clamp(sm.big), end = sm.end, f = sm.f;
    // the lander has its own zone: the hero (right side) and the big-type moment (centre, below the words); in every
    // other segment it flies up and out of frame so it never covers text or cards
    const away = clamp(sm.hero * 1.4 - 0.25) * (1 - big);
    // the lander turns slowly with the page; in the big-type moment it comes to the centre, low over the limb
    rig.rotation.set(cur.rx + 0.05 * Math.sin(t * 0.6), -0.7 + cur.ry + t * 0.05 + f * Math.PI, 0.03 * Math.sin(t * 0.8));
    const bob = Math.sin(t * 1.1) * 0.06;
    rig.position.set(base.x * (1 - big) + away * 4, base.y + (-2.05 - base.y) * big + bob + away * 9 - end * 6, 0);
    rig.scale.setScalar(base.s * (1 - 0.15 * big));
    // the Moon is the fixed base of the page: slow turn, light swings with the page, it sinks away at the very end
    if (baseQ) { spinQ.setFromAxisAngle(UP, t * 0.004 + f * 0.5); moon.quaternion.copy(spinQ).multiply(baseQ); }
    else moon.rotation.y = t * 0.004 + f * 0.5;
    if (fx) for (const name in seg) { const e = fx[name]; if (e && e.update) e.update(seg[name]); }
    moon.position.y = -32.4 - end * 26;
    sun.position.set(Math.cos(f * 1.6 - 0.2) * 30, (0.16 + f * 0.25) * 30, Math.sin(f * 1.6 - 0.2) * 30);
    // scroll depth: the camera sinks toward the limb, so near layers (lander) move more than the Moon limb and the stars (true parallax)
    const sink = Math.max(scrollS, sm.hero);
    cam.position.set(cur.cx, 0.5 + cur.cy - sink * 2.4, 11 - sink * 1.5);
    cam.lookAt(0, 0.1 - sink * 1.2, 0);
    puff.update(t);
    dust.rotation.z = t * 0.01; dust.position.y = Math.sin(t * 0.2) * 0.15;
    renderer.render(scene, cam);
  }
  function frame() {
    raf = 0;
    if (!onScreen || document.hidden) return;
    step();
    raf = requestAnimationFrame(frame);
  }
  const kick = () => { if (!raf && onScreen && !document.hidden) { clock.getDelta(); raf = requestAnimationFrame(frame); } };
  new IntersectionObserver(es => { onScreen = es.some(e => e.isIntersecting); if (!opts.external) kick(); }, { threshold: 0 }).observe(el);
  if (!opts.external) { document.addEventListener('visibilitychange', kick); kick(); }
  // opts.external: the page loop (js/home.js) calls tick() each frame instead of a second rAF loop
  return { renderer, tick() { if (onScreen && !document.hidden) step(); }, setScroll(v) { scrollS = clamp(v); }, setState(o) { Object.assign(st, o); }, setSegments(o) { Object.assign(seg, o); }, destroy() { onScreen = false; renderer.dispose(); canvas.remove(); } };
}
