// One WebGL canvas, three layers rendered in order:
//   A) space: Moon (LROC colour + LOLA bump), stars, Earth. ONE camera: orbit view in the hero, then it converges on
//      the real landing site (Site 23, Malapert massif, lat/lon from the LOLA tile) and follows the lander down.
//   C) the pilot network (real sampled weights), shown while the AI section is on screen
//   B) the lander + the site in metres: the same camera pose re-expressed around the site, so the real LOLA terrain
//      (1280 m near patch over a 5120 m surround) sits exactly where the sphere shows the site and fades in as the
//      camera gets low: no scene swap. Hero: the lander + propulsion module stack orbits, separates, the lander rides
//      a side band left of the altitude rail, then converges on its true position above the site, lands, opens its
//      ramp; in #rover the rover drives the real planned route (assets/data/rover_route_site23.json).
// Timeline comes from js/gauge.js (timeline()) so the gauge readout and the 3D scene always agree.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { buildNet } from './net3d.js';
import { timeline, altAt, TD } from './gauge.js';

const MOON_KM = 1737.4;
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const BG = new THREE.Color(0x262624);      // warm dark grey = page --bg #262624 (no blue cast)

export async function createScene(canvas, { onReady } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, logarithmicDepthBuffer: true, powerPreference: 'high-performance', preserveDrawingBuffer: new URLSearchParams(location.search).has('scroll') });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));   // full-screen terrain is fill-rate bound: 1.5 x is ~45 % fewer pixels than 2 x on retina
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.autoClear = false;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.setClearColor(BG, 1);

  const texLoader = new THREE.TextureLoader();
  const gltf = new GLTFLoader();

  // hero video backdrop (js/media.js): transparent space during the hero so the video shows through
  const clean = new URLSearchParams(location.search).has('clean');
  const DBG = new URLSearchParams(location.search).has('debug');
  const FORCE = (new URLSearchParams(location.search).get('force') || '').split(',').filter(Boolean).map(Number); if (!FORCE.length) FORCE.length = 0;
  let dbgDiv = null; const dbgEl = () => dbgDiv || (dbgDiv = Object.assign(document.body.appendChild(document.createElement('pre')), { style: 'position:fixed;left:8px;top:70px;z-index:2147483647;background:#000;color:#0f0;font:11px monospace;white-space:pre-wrap;max-width:90vw;visibility:visible' }));   // poster renders: never show the video
  let backdrop = !clean && document.body.classList.contains('has-hero-video');
  window.addEventListener('adi:backdrop', e => { backdrop = !clean && !!(e.detail && e.detail.on); });

  // ================= A: space =================
  const sceneA = new THREE.Scene();
  sceneA.background = null;                 // cleared manually (alpha for the hero video)
  const camA = new THREE.PerspectiveCamera(38, 1, 1e-6, 3000);
  const SUN_DIR = new THREE.Vector3(1, 0.1, 0.42).normalize();   // grazing light over the pole: the south-pole problem
  const sunA = new THREE.DirectionalLight(0xfff3e2, 4.2);
  sunA.position.copy(SUN_DIR).multiplyScalar(10);
  sceneA.add(sunA, new THREE.AmbientLight(0xc9c6bf, 0.4));
  const fillA = new THREE.DirectionalLight(0xbfbab0, 0.3);  // earthshine-ish fill so the night side keeps faint detail
  fillA.position.set(-6, 2, 5);
  sceneA.add(fillA);

  const moonMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0, bumpScale: 2.0 });
  const moon = new THREE.Mesh(new THREE.SphereGeometry(1, 192, 128), moonMat);
  moon.rotation.x = Math.PI;                 // south pole on top (+Y)
  sceneA.add(moon);
  texLoader.load('assets/img/moon_color_2k.jpg', t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; moonMat.map = t; moonMat.needsUpdate = true; });
  texLoader.load('assets/img/moon_bump_1k.jpg', t => { moonMat.bumpMap = t; moonMat.needsUpdate = true; });

  let starMat;
  { // stars
    const n = 2800, p = new Float32Array(n * 3), c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(900);
      p.set([v.x, v.y, v.z], i * 3);
      const b = 0.55 + Math.random() * 0.45, warm = Math.random() < 0.2;
      c.set([b, b * (warm ? 0.9 : 0.97), b * (warm ? 0.75 : 1.0)], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('color', new THREE.BufferAttribute(c, 3));
    starMat = new THREE.PointsMaterial({ size: 1.9, sizeAttenuation: false, vertexColors: true, depthWrite: false, transparent: true, opacity: 1 });
    sceneA.add(new THREE.Points(g, starMat));
  }
  // Earth sits low on the horizon as seen from the south pole (angular size ~1.9 deg)
  const earth = new THREE.Mesh(new THREE.SphereGeometry(3.67, 48, 32), new THREE.MeshStandardMaterial({ map: earthTexture(), roughness: 0.8, emissive: 0x0a1428 }));
  earth.rotation.set(0.4, 2.2, 0.2);
  earth.position.set(-45, 9, -215);
  // Earth removed (round 3): read as a flat sticker on the horizon
  earth.visible = false;

  // polar orbit ring for the hero, with a soft glow following the lander
  const ORBIT_R = 1.34;
  // Orbit plane chosen in the hero camera's frame (right r, up u, towards camera d) so the near arc runs over the
  // upper-left of the Moon: the lander parks on that arc (theta near 0) and is never hidden behind the Moon.
  const hd = new THREE.Vector3(0.3, 0.52, 1).normalize();
  const hr = new THREE.Vector3(hd.z, 0, -hd.x).normalize(), hu = new THREE.Vector3().crossVectors(hd, hr).normalize();
  const orbitU = new THREE.Vector3().addScaledVector(hr, -0.62).addScaledVector(hu, 0.62).addScaledVector(hd, 0.48).normalize();
  const orbitV = new THREE.Vector3().addVectors(hr, hu).normalize();
  const ringPts = [], ringAng = new Float32Array(257);
  for (let i = 0; i <= 256; i++) { const a = i / 256 * Math.PI * 2; ringAng[i] = a; ringPts.push(orbitU.clone().multiplyScalar(Math.cos(a) * ORBIT_R).addScaledVector(orbitV, Math.sin(a) * ORBIT_R)); }
  const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPts);
  ringGeo.setAttribute('ang', new THREE.BufferAttribute(ringAng, 1));
  const ringMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uTheta: { value: 0 }, uOpacity: { value: 1 } },
    vertexShader: 'attribute float ang; varying float vA; void main(){ vA = ang; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'uniform float uTheta; uniform float uOpacity; varying float vA; void main(){ float d = abs(mod(vA - uTheta + 3.14159265, 6.2831853) - 3.14159265); float g = exp(-d / 0.45); gl_FragColor = vec4(mix(vec3(0.82,0.80,0.75), vec3(0.85,0.47,0.34), g * 0.8), (0.3 + 0.6 * g) * uOpacity); }',
  });
  const ring = new THREE.Line(ringGeo, ringMat);
  sceneA.add(ring);

  // ================= C: network =================
  let net = null;
  const labelRoot = document.getElementById('net-labels');
  fetch('assets/net.json').then(r => r.json()).then(d => { net = buildNet(d, { labelRoot }); resize(); }).catch(() => {});

  // ================= B: lander + landing site (metres) =================
  // sceneB shares sceneA's world axes but is in metres, origin = the landing site ON the sphere (Site 23, Malapert
  // massif; lat/lon from the tile header). camB = camA's pose with position (camA - site) * RM, so the LOLA terrain
  // lies exactly where the sphere shows the site: one continuous camera from orbit to touchdown, no scene swap.
  const RM = MOON_KM * 1000;
  const DEG = Math.PI / 180;
  const SITE_LAT = -85.99478, SITE_LON = 359.76422;     // data/tiles/lola5m_site23.npz header (tile centre = lander)
  const siteLocal = (() => { const th = (90 - SITE_LAT) * DEG, ph = Math.PI + SITE_LON * DEG; return new THREE.Vector3(-Math.cos(ph) * Math.sin(th), Math.cos(th), Math.sin(ph) * Math.sin(th)); })();
  // Moon spin at which the site faces the sun side (the descent converges there; lighting is SUN_DIR for both scenes)
  const SPIN_T = (() => { const e = new THREE.Euler(), v = new THREE.Vector3(); let best = 0, bd = -2; for (let i = 0; i < 720; i++) { const a = i / 720 * Math.PI * 2; e.set(Math.PI, a, 0); v.copy(siteLocal).applyEuler(e); const d = v.dot(SUN_DIR); if (d > bd) { bd = d; best = a; } } return best; })();
  // terrain tone: a fixed grey matched by eye to the sphere's filtered LROC colour around the site under the same lights
  const siteColor = new THREE.Color(0xa29e97), terrMats = [];

  const sceneB = new THREE.Scene();
  const camB = new THREE.PerspectiveCamera(38, 1, 0.05, 4e7);
  const TANF = Math.tan(19 * DEG);
  const sunB = new THREE.DirectionalLight(0xfff1dd, 4.2);
  sunB.castShadow = true;
  sunB.shadow.mapSize.set(1024, 1024);
  Object.assign(sunB.shadow.camera, { left: -70, right: 70, top: 70, bottom: -70, near: 1, far: 4000 });
  sunB.shadow.bias = -0.0005;
  sceneB.add(sunB, sunB.target);
  // same ambient + fill as the sphere's scene so the LOLA terrain keeps the sphere's tone; the cool rim is for the lander
  sceneB.add(new THREE.AmbientLight(0xc9c6bf, 0.4));
  const fillB = new THREE.DirectionalLight(0xbfbab0, 0.3); fillB.position.set(-6, 2, 5); sceneB.add(fillB);
  const rim = new THREE.DirectionalLight(0xe8dccb, 0.9); rim.position.set(-6, 3, -6); sceneB.add(rim);

  const landerRoot = new THREE.Group();  // feet at local y = 0
  sceneB.add(landerRoot);
  const landerPivot = new THREE.Group();
  landerRoot.add(landerPivot);
  let landerH = 2.86, landerNode = null, rampHinge = null;

  // flame (additive cone, gradient)
  const flameMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { k: { value: 1 } },
    vertexShader: 'varying float vY; void main(){ vY = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'varying float vY; uniform float k; void main(){ float a = pow(vY, 1.6) * k; gl_FragColor = vec4(vec3(1.0,0.72,0.35)*a + vec3(0.4,0.6,1.0)*pow(vY,8.0)*k, a); }',
  });
  const flameGeo = new THREE.ConeGeometry(0.42, 3.2, 28, 1, true);
  flameGeo.rotateX(Math.PI);          // tip down
  flameGeo.translate(0, -1.6, 0);
  const flame = new THREE.Mesh(flameGeo, flameMat);
  const flameLight = new THREE.PointLight(0xffa040, 0, 14, 1.6);
  landerPivot.add(flame, flameLight);

  // thrust-residual arrows for the pilot beat (setThrustArrows): x / y lateral, z vertical, in the lander frame
  const arrows = [new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1.4, 0), 1, 0xd97757, 0.35, 0.2),
    new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1.4, 0), 1, 0xb7cdbe, 0.35, 0.2),
    new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 1.4, 0), 1, 0xf0eee6, 0.35, 0.2)];
  const AX = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0)];
  for (const a of arrows) { a.visible = false; for (const m of [a.line.material, a.cone.material]) { m.transparent = true; m.depthTest = false; } a.renderOrder = 9; landerPivot.add(a); }
  let thrustV = null, thrustK = 0, pilotFocus = 0;

  // ground: real LOLA Site 23 terrain at the site (assets/data/landing_site23.json): a 1280 m 5 m/px near patch
  // over a 5120 m 20 m/px surround; both fade at their edges into the sphere
  const ground = new THREE.Group();
  sceneB.add(ground);
  ground.visible = false;
  let heightAt = () => 0, nearMat = null, wideMat = null, siteMeta = null;
  const dust = makeDust();
  const puff = makePuff();
  ground.add(dust.points);
  let rover = null, wheels = [];
  const roverPath = new THREE.Group(), roverYaw = new THREE.Group();
  roverPath.add(roverYaw); roverYaw.rotation.y = Math.PI; roverPath.visible = false;
  // propulsion module: the hero shows the stack (lander on top of it); it separates early and drifts away
  const pmGroup = new THREE.Group(), pmMats = [];
  pmGroup.position.y = -1.0;            // lander interface ring bottom (lander-node frame)
  gltf.load('assets/models/propulsion_module.glb', g => {
    g.scene.traverse(o => { if (o.isMesh) { o.castShadow = true; o.material = o.material.clone(); o.material.transparent = true; pmMats.push(o.material); } });
    pmGroup.add(g.scene);
  }, undefined, () => {});

  const loadImg = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  function decode16(img, n, lo, hi) {
    const cv = document.createElement('canvas'); cv.width = cv.height = n;
    const cx = cv.getContext('2d'); cx.drawImage(img, 0, 0);
    const px = cx.getImageData(0, 0, n, n).data, Hh = new Float32Array(n * n), Sl = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) { Hh[i] = lo + (px[i * 4] * 256 + px[i * 4 + 1]) / 65535 * (hi - lo); Sl[i] = px[i * 4 + 2] / 8; }
    return { Hh, Sl };
  }
  function sampler(Hh, n, mpp, c0) {
    const size = n * mpp;
    return (x, z) => {
      const fx = clamp((x + size / 2) / mpp - 0.5, 0, n - 1.001), fz = clamp((z + size / 2) / mpp - 0.5, 0, n - 1.001);
      const i = Math.floor(fx), j = Math.floor(fz), u = fx - i, v = fz - j, h = (a, b) => Hh[b * n + a];
      return lerp(lerp(h(i, j), h(i + 1, j), u), lerp(h(i, j + 1), h(i + 1, j + 1), u), v) - c0;
    };
  }
  function terrainMesh(Hh, n, mpp, c0, drop, fadeIn, order) {
    const size = n * mpp, geo = new THREE.PlaneGeometry(size, size, n - 1, n - 1);
    geo.rotateX(-Math.PI / 2);
    const P = geo.attributes.position, col = new Float32Array(P.count * 4);
    for (let k = 0; k < P.count; k++) {
      const x = P.getX(k), z = P.getZ(k), r2 = x * x + z * z;
      P.setY(k, Hh[k] - c0 - drop - r2 / (2 * RM));                       // + Moon curvature
      const e = Math.max(Math.abs(x), Math.abs(z)) / (size / 2), rr = Math.sqrt(r2) / (size / 2);
      const a = 1 - smooth(fadeIn, 1, Math.max(e * 0.96, rr * 0.86));
      col.set([1, 1, 1, a], k * 4);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 4));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ color: siteColor, vertexColors: true, roughness: 1, metalness: 0, map: grainTexture(), transparent: true, opacity: 0 });
    mat.map.repeat.set(size / 8, size / 8);
    terrMats.push(mat);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true; mesh.renderOrder = order;
    ground.add(mesh);
    return mat;
  }
  loadTerrain().catch(e => console.warn('terrain', e));
  async function loadTerrain() {
    const meta = siteMeta = await (await fetch('assets/data/landing_site23.json')).json();
    const [iN, iW] = await Promise.all([loadImg(meta.near.png), loadImg(meta.wide.png)]);
    const N = decode16(iN, meta.near.size_px, meta.height_lo_m, meta.height_hi_m), Wd = decode16(iW, meta.wide.size_px, meta.height_lo_m, meta.height_hi_m);
    const c0 = meta.lander_height_m;
    const hN = sampler(N.Hh, meta.near.size_px, meta.near.m_per_px, c0), hW = sampler(Wd.Hh, meta.wide.size_px, meta.wide.m_per_px, c0);
    const halfN = meta.near.size_px * meta.near.m_per_px / 2 - 5;
    heightAt = (x, z) => (Math.abs(x) < halfN && Math.abs(z) < halfN ? hN(x, z) : hW(x, z) - 1.5);
    wideMat = terrainMesh(Wd.Hh, meta.wide.size_px, meta.wide.m_per_px, c0, 1.5, 0.3, 1);   // long radial fade into the sphere
    nearMat = terrainMesh(N.Hh, meta.near.size_px, meta.near.m_per_px, c0, 0, 0.8, 2);
    if (routeData) buildRoute(routeData);
  }

  // ---------- rover route (assets/data/rover_route_site23.json; real planned run, see tools/export_rover_route.py) ----------
  let routeData = null, routeObj = null;
  const EXAG = 16;                         // lander + rover drawn x16 when the camera pulls back over the route (labelled)
  fetch('assets/data/rover_route_site23.json').then(r => r.json()).then(d => { routeData = d; if (nearMat) buildRoute(d); }).catch(() => {});
  function buildRoute(d) {
    if (routeObj) return;
    const P = d.points, n = P.length, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) pos.set([P[i][0], heightAt(P[i][0], P[i][2]) + 2.5, P[i][2]], i * 3);
    const gFull = new THREE.BufferGeometry(); gFull.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const gDone = new THREE.BufferGeometry(); gDone.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mFull = new THREE.LineBasicMaterial({ color: 0xd97757, transparent: true, opacity: 0, depthTest: false });
    const mDone = new THREE.LineBasicMaterial({ color: 0xe8b9a0, transparent: true, opacity: 0, depthTest: false });
    const full = new THREE.Line(gFull, mFull), done = new THREE.Line(gDone, mDone);
    full.renderOrder = done.renderOrder = 6;
    // ice targets (PSR edges: dark and cold, NOT confirmed ice) as flat rings; the one the run reached is brighter
    const rp = P[d.ice_reach_index]; let reach = -1, best = 1e9;
    d.ice_targets.forEach((t, i) => { const dd = Math.hypot(t[0] - rp[0], t[2] - rp[2]); if (dd < best) { best = dd; reach = i; } });
    const rings = [], ringM = new THREE.MeshBasicMaterial({ color: 0xc8c3e0, transparent: true, opacity: 0, side: THREE.DoubleSide, depthTest: false });
    const ringHot = ringM.clone();
    const rg = new THREE.RingGeometry(9, 13, 40); rg.rotateX(-Math.PI / 2);
    d.ice_targets.forEach((t, i) => {
      if (Math.abs(t[0]) > 630 || Math.abs(t[2]) > 630) return;
      const m = new THREE.Mesh(rg, i === reach ? ringHot : ringM); m.position.set(t[0], heightAt(t[0], t[2]) + 1.5, t[2]); m.renderOrder = 7; rings.push(m);
    });
    // hazard cells near the route (official slope >= 12 deg), 5 m squares
    const H = d.hazard_cells_xz, hzM = new THREE.MeshBasicMaterial({ color: 0xe0705e, transparent: true, opacity: 0, depthTest: false, side: THREE.DoubleSide });
    const hg = new THREE.PlaneGeometry(5, 5); hg.rotateX(-Math.PI / 2);
    const haz = new THREE.InstancedMesh(hg, hzM, Math.max(1, H.length)), mtx = new THREE.Matrix4();
    H.forEach((c, i) => { mtx.makeTranslation(c[0], heightAt(c[0], c[1]) + 1.2, c[1]); haz.setMatrixAt(i, mtx); });
    haz.count = H.length; haz.renderOrder = 5;
    const g = new THREE.Group(); g.add(haz, full, done, ...rings); ground.add(g);
    // comm link lander -> rover (updated per frame, in sceneB world)
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const lm = new THREE.LineDashedMaterial({ color: 0xe6dcc8, dashSize: 6, gapSize: 5, transparent: true, opacity: 0, depthTest: false });
    const link = new THREE.Line(lg, lm); link.frustumCulled = false; link.renderOrder = 8; sceneB.add(link);
    let cx = 0, cz = 0, mnx = 1e9, mxx = -1e9, mnz = 1e9, mxz = -1e9;
    for (const p of P) { mnx = Math.min(mnx, p[0]); mxx = Math.max(mxx, p[0]); mnz = Math.min(mnz, p[2]); mxz = Math.max(mxz, p[2]); }
    cx = (mnx + mxx) / 2; cz = (mnz + mxz) / 2;
    const Dc = new Float64Array(n); for (let i = 1; i < n; i++) Dc[i] = Dc[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][2] - P[i - 1][2]);
    routeObj = { D: Dc, total: Dc[n - 1], n, P, gDone, mFull, mDone, ringM, ringHot, hzM, link, lm, lg, centre: new THREE.Vector3(cx, 0, cz), span: Math.max(mxx - mnx, mxz - mnz), reachT: d.ice_targets[reach] };
  }

  const landerMats = [];
  gltf.load('assets/models/lander.glb', g => {
    const m = g.scene;
    const ln = m.getObjectByName('lander'); if (ln) ln.rotation.y = Math.PI / 2;   // bay faces the viewer side as before
    m.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; landerMats.push(o.material); } });
    const box = new THREE.Box3().setFromObject(m);
    const c = box.getCenter(new THREE.Vector3());
    m.position.set(-c.x, -box.min.y, -c.z);
    landerH = box.max.y - box.min.y;
    landerPivot.add(m);
    landerNode = m.getObjectByName('lander') || m;
    // side bay ramp: hinge along the bottom edge of the bay door (lander-node frame: door at x = -1, hinge y = -0.78)
    const ramp = m.getObjectByName('ramp_panel');
    if (ramp && ramp.parent) {
      rampHinge = new THREE.Group(); rampHinge.position.set(-1.0, -0.78, 0);
      ramp.parent.add(rampHinge); rampHinge.add(ramp); ramp.position.set(1.0, 0.78, 0);
    }
    landerNode.add(roverPath);
    landerNode.add(pmGroup);
    puff.points.position.y = -1.0; landerNode.add(puff.points);   // separation puff at the interface ring
    m.updateMatrixWorld(true);
    const eng = m.getObjectByName('engine_centre');
    if (eng) { const v = eng.getWorldPosition(new THREE.Vector3()); landerPivot.worldToLocal(v); flame.position.set(v.x, v.y, v.z); flameLight.position.set(v.x, v.y - 1, v.z); }
    else { flame.position.y = 0.5; }
    for (const a of arrows) a.position.y = landerH * 0.5;
    onReady && onReady(true);
  }, undefined, () => { onReady && onReady(false); });

  function loadRover() {
    if (rover !== null) return;
    rover = false;
    gltf.load('assets/models/rover.glb', g => {
      const node = g.scene.getObjectByName('rover') || g.scene;
      node.removeFromParent && node.removeFromParent();
      node.position.set(0, 0, 0); node.quaternion.identity();   // rover-node frame: +x forward, wheel bottoms at y = -0.206
      node.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });   // receiving shadows at the 900 m box gave acne/flicker
      const box = new THREE.Box3().setFromObject(node);
      node.position.y = -box.min.y;                       // wheel bottoms at y = 0 of roverYaw
      wheels = ['FL', 'ML', 'RL', 'FR', 'MR', 'RR'].map(k => node.getObjectByName('wheel_' + k)).filter(Boolean);
      roverYaw.add(node); rover = node;
    }, undefined, () => {});
  }

  // ================= state =================
  let W = 1, H = 1, progress = 0, netQ = 0, netMix = 0, roverP = null, labels = null, hudFn = null;
  const clock = new THREE.Clock();
  let roverS = null, heroTheta = null, orbitSpin = 0, frameN = 0, T = timeline(0), tlDirty = true, tdTime = -1, lastStep = -1, roverYawS = null;
  const S = { q: 0, f: 0, hero: 0, foot: 0, rov: 0, pf: 0 };
  function resize() {
    W = canvas.clientWidth || window.innerWidth; H = canvas.clientHeight || window.innerHeight;
    renderer.setSize(W, H, false);
    camA.aspect = camB.aspect = W / H;
    camA.updateProjectionMatrix(); camB.updateProjectionMatrix();
    net && net.resize(W, H);
  }
  window.addEventListener('resize', resize);
  resize();

  // scratch objects (no per-frame allocation)
  const v3 = new THREE.Vector3(), v4 = new THREE.Vector3(), vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3();
  const heroDir = new THREE.Vector3(0.3, 0.52, 1).normalize();
  const lookA = new THREE.Vector3(), lookH = new THREE.Vector3(), orbitP = new THREE.Vector3(), heroPos = new THREE.Vector3(), chasePos = new THREE.Vector3();
  const siteN = new THREE.Vector3(), siteX = new THREE.Vector3(), siteZ = new THREE.Vector3(), camZ = new THREE.Vector3(), basis = new THREE.Matrix4();
  const tgtM = new THREE.Vector3(), camM = new THREE.Vector3(), physP = new THREE.Vector3(), ovP = new THREE.Vector3();
  const qOv = new THREE.Quaternion(), qTilt = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.17), qInv = new THREE.Quaternion();
  function ndcToB(x, y, dist, out) {
    v3.set(x, y, 0.5).unproject(camB).sub(camB.position).normalize();
    return out.copy(camB.position).addScaledVector(v3, dist);
  }
  function setLensShift(cam, sx, sy) { cam.updateProjectionMatrix(); cam.projectionMatrix.elements[8] = -sx; cam.projectionMatrix.elements[9] = -sy; cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert(); }
  // lander scale so that it is `px` pixels tall at distance `dist` from camB
  const scaleForPx = (px, dist) => px * 2 * dist * TANF / (landerH * H);
  // site frame (metres, x east, z south, y up) -> sceneB world
  const siteToB = (x, y, z, out) => out.set(0, 0, 0).addScaledVector(siteX, x).addScaledVector(siteN, y).addScaledVector(siteZ, z);
  // route position at arc length s (metres) in the site frame (x, z), binary search on the cumulative distances
  const rpos = new THREE.Vector3(), rfwd = new THREE.Vector3(), rback = new THREE.Vector3(), roverTan = new THREE.Vector3(), roverTanS = new THREE.Vector3();
  const rUp = new THREE.Vector3(), rUpS = new THREE.Vector3(), rF = new THREE.Vector3(), rU = new THREE.Vector3(), rX = new THREE.Vector3(), rZ = new THREE.Vector3();
  const rM = new THREE.Matrix4(), rQ = new THREE.Quaternion(), qTmp = new THREE.Quaternion(), roverScreen = new THREE.Vector3();
  function routeAt(rt, sArc, out) {
    const D = rt.D; let lo = 0, hi = rt.n - 1;
    if (sArc <= 0) return out.set(rt.P[0][0], 0, rt.P[0][2]);
    if (sArc >= D[hi]) return out.set(rt.P[hi][0], 0, rt.P[hi][2]);
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (D[m] <= sArc) lo = m; else hi = m; }
    const f = (sArc - D[lo]) / Math.max(1e-6, D[hi] - D[lo]);
    return out.set(lerp(rt.P[lo][0], rt.P[hi][0], f), 0, lerp(rt.P[lo][2], rt.P[hi][2], f));
  }
  // ground height under a point given in lander-node coordinates, returned in lander-node y
  function groundYAtNode(x, z) {
    v4.set(x, 0, z); landerNode.localToWorld(v4); ground.worldToLocal(v4);
    v4.y = heightAt(v4.x, v4.z); ground.localToWorld(v4); landerNode.worldToLocal(v4);
    return v4.y;
  }

  // rover drive: bay floor -> ramp -> terrain, all in lander-node coordinates (bay opens towards -x)
  const RAMP_LEN = 1.03, HINGE_X = -1.0, HINGE_Y = -0.78, FOOT_Y = -1.45;
  const RAMP_A = Math.asin(clamp((HINGE_Y - FOOT_Y) / RAMP_LEN, 0.1, 0.95));
  const L0 = 0.55, L1 = RAMP_LEN, L2 = 3.2;
  function placeRover(s) {
    let x, y, pitch;
    if (s <= L0) { x = -0.45 - s; y = -0.82; pitch = 0; }
    else if (s <= L0 + L1) { const u = s - L0; x = HINGE_X - u * Math.cos(RAMP_A); y = HINGE_Y - u * Math.sin(RAMP_A); pitch = RAMP_A; }
    else {
      const u = s - L0 - L1; x = HINGE_X - L1 * Math.cos(RAMP_A) - u; const zz = -0.25 * u * u / L2;
      const yFront = groundYAtNode(x - 0.35, zz), yBack = groundYAtNode(x + 0.35, zz);   // follow the real terrain
      y = 0.5 * (yFront + yBack); roverPath.position.z = zz;
      pitch = Math.atan2(yBack - yFront, 0.7);             // +z rotation = nose down while driving towards -x
    }
    // ease pitch at the top and bottom of the ramp
    if (s > L0 - 0.25 && s < L0 + 0.25) pitch = RAMP_A * smooth(L0 - 0.25, L0 + 0.25, s);
    if (s > L0 + L1 - 0.3 && s < L0 + L1 + 0.3) pitch = lerp(RAMP_A, pitch, smooth(L0 + L1 - 0.3, L0 + L1 + 0.3, s));
    if (s <= L0 + L1) roverPath.position.z = 0;
    roverPath.position.x = x; roverPath.position.y = y;
    roverPath.rotation.set(0, 0, pitch);
    roverPath.scale.setScalar(1);
    for (let i = 0; i < wheels.length; i++) wheels[i].rotation.z = -s / 0.086;
  }
  const rampEnd = new THREE.Vector3(), routeP = new THREE.Vector3(), routeD = new THREE.Vector3(), roverW = new THREE.Vector3(), landerTop = new THREE.Vector3();
  const label = (el, p, on) => {   // project a sceneB world point to a DOM label (transform only)
    if (!el) return;
    v3.copy(p).project(camB);
    const ok = on > 0.02 && v3.z < 1 && Math.abs(v3.x) < 1.1 && Math.abs(v3.y) < 1.1;
    const o = ok ? on.toFixed(2) : '0';
    if (el.__o !== o) { el.__o = o; el.style.opacity = o; }                        // write only on change
    if (ok) { const tr = `translate3d(${Math.round((v3.x + 1) / 2 * W)}px, ${Math.round((1 - v3.y) / 2 * H)}px, 0)`; if (el.__t !== tr) { el.__t = tr; el.style.transform = tr; } }
  };

  function frame() {
    const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
    if (tlDirty || (frameN & 15) === 0) { T = timeline(progress); tlDirty = false; }   // DOM reads only when scrolled (+ a slow refresh)
    if (FORCE.length) { T.q = FORCE[0]; T.f = FORCE[1]; T.rover = FORCE[2] || 0; T.hero = FORCE[0] > 0.1 ? 1 : 0; T.foot = 0; }   // ?force=q,f,rover (debug renders)
    const rvT = roverP == null ? T.rover : roverP;
    // camera scrub damped (lerp 0.08 per frame), no overshoot
    if (frameN === 0 || document.documentElement.classList.contains('shot')) { S.q = T.q; S.f = T.f; S.hero = T.hero; S.foot = T.foot; S.rov = rvT; S.pf = pilotFocus; }
    else { S.q += (T.q - S.q) * 0.08; S.f += (T.f - S.f) * 0.08; S.hero += (T.hero - S.hero) * 0.08; S.foot += (T.foot - S.foot) * 0.08; S.rov += (rvT - S.rov) * 0.08; S.pf += (pilotFocus - S.pf) * 0.15; }
    const q = S.q, f = S.f, altKm = altAt(q, f), rv = S.rov, pf = S.pf;
    const narrow = W / H < 0.8 || W < 760;
    frameN++;

    // ---------- Moon spin: free in the hero, settles so the landing site is under the descent ----------
    const s = smooth(0.06, 0.3, q);                         // hero -> descent blend
    orbitSpin += dt * 0.012 * (1 - s);
    const spinFree = orbitSpin + q * 0.9, TAU = Math.PI * 2;
    const spinT = SPIN_T + TAU * Math.round((spinFree - SPIN_T) / TAU);
    moon.rotation.y = lerp(spinFree, spinT, smooth(0.03, 0.3, q));
    moon.updateMatrixWorld();
    siteN.copy(siteLocal).applyMatrix4(moon.matrixWorld).normalize();
    siteZ.set(0, 1, 0).addScaledVector(siteN, -siteN.y);   // towards the pole = south
    if (siteZ.lengthSq() < 1e-12) siteZ.set(0, 0, 1);
    siteZ.normalize(); siteX.crossVectors(siteN, siteZ).normalize();
    basis.makeBasis(siteX, siteN, siteZ); ground.quaternion.setFromRotationMatrix(basis);
    camZ.set(0, 0, 1).addScaledVector(siteN, -siteN.z).normalize();   // chase camera sits on the viewer side of the site

    // ---------- A: one camera, orbit -> site chase (metres converted to Moon units) ----------
    const Lm = altKm * 1000;
    const zo = smooth(0.02, 0.24, rv);                       // rover beat: pull back over the route
    let e = DEG * lerp(5, 26, smooth(Math.log(40), Math.log(30000), Math.log(Lm + 10)));
    // chase camera looks between the lander (above) and its landing site (below), so both stay in view and converge
    let R = 18 + 2.2 * Lm;
    tgtM.copy(siteN).multiplyScalar(0.35 * Lm + 1.2);
    if (zo > 0 && routeObj) {
      siteToB(routeObj.centre.x * (narrow ? 0.45 : 0.75), 0, routeObj.centre.z * (narrow ? 0.45 : 0.75), vA);   // route + lander both in frame
      tgtM.lerp(vA, zo); R = lerp(R, routeObj.span * (narrow ? 2.6 : 1.75), zo); e = lerp(e, 44 * DEG, zo);
    }
    camM.copy(tgtM).addScaledVector(siteN, R * Math.sin(e)).addScaledVector(camZ, R * Math.cos(e));
    chasePos.copy(siteN).addScaledVector(camM, 1 / RM);
    const heroDist = narrow ? 5.4 : 4.3;
    heroPos.copy(heroDir).multiplyScalar(heroDist);
    const sc2 = smooth(0, 1, s);
    // blend in log-distance so the approach feels like one continuous move
    const dH = heroPos.length(), dC = chasePos.length();
    camA.position.copy(heroPos).lerp(chasePos, sc2);
    camA.position.setLength(Math.exp(lerp(Math.log(dH - 0.999), Math.log(dC - 0.999), sc2)) + 0.999);
    lookH.copy(camA.position).negate().normalize();
    lookA.copy(siteN).addScaledVector(tgtM, 1 / RM).sub(camA.position).normalize();
    lookA.lerp(lookH, 1 - sc2).normalize();
    camA.up.set(0, 1, 0).lerp(siteN, sc2);
    camA.lookAt(v4.copy(camA.position).add(lookA));
    const fx = narrow ? 0.05 : 0.35, fy = narrow ? -0.1 : -0.1;   // touchdown on the right, where the side lander rides
    let lsx = lerp(0, fx, sc2), lsy = lerp(narrow ? -0.32 : -0.22, fy, sc2);   // hero: Moon centred, a little below the headline
    lsx = lerp(lsx, narrow ? 0 : -0.12, zo); lsy = lerp(lsy, narrow ? 0.32 : -0.1, zo);   // phones: route above the tiles
    const lift = clean ? 0 : Math.min(0.95 * S.foot, 0.4) * (1 - zo);   // after the rover beat the scene holds still under the closing band
    setLensShift(camA, lsx, lsy + lift);
    camA.updateMatrixWorld();
    const ringO = 1 - smooth(0.3, 0.9, S.hero);
    ringMat.uniforms.uOpacity.value = ringO;
    ring.visible = ringO > 0.01;
    // camB: same pose in metres around the site
    camB.quaternion.copy(camA.quaternion);
    camB.position.copy(camA.position).sub(siteN).multiplyScalar(RM);
    setLensShift(camB, lsx, lsy + lift);
    camB.updateMatrixWorld();
    const camAltKm = (camA.position.length() - 1) * MOON_KM;

    // ---------- B: lander placement ----------
    const D = 26;
    // 1) hero: on the near arc of the polar orbit (never behind the Moon), slow drift
    // park on the right-most point of the near (camera-side) arc, clear of the centred hero text; slow drift
    if ((frameN & 31) === 1 || heroTheta == null) {
      let bx = -9;
      for (let i = 0; i < 48; i++) {
        const a = i / 48 * Math.PI * 2;
        orbitP.copy(orbitU).multiplyScalar(Math.cos(a) * ORBIT_R).addScaledVector(orbitV, Math.sin(a) * ORBIT_R);
        if (orbitP.dot(camA.position) < 0.2) continue;                 // behind / beside the Moon
        orbitP.project(camA); if (orbitP.x > bx && orbitP.y < 0.5) { bx = orbitP.x; heroTheta = a; }
      }
      if (heroTheta == null) heroTheta = -0.2;
    }
    const theta = heroTheta + 0.08 * Math.sin(t * 0.05);
    ringMat.uniforms.uTheta.value = theta;
    orbitP.copy(orbitU).multiplyScalar(Math.cos(theta) * ORBIT_R).addScaledVector(orbitV, Math.sin(theta) * ORBIT_R);
    orbitP.project(camA);
    orbitP.x = Math.min(orbitP.x, narrow ? 0.55 : 0.7); orbitP.y = clamp(orbitP.y, -0.6, narrow ? -0.28 : 0.6);   // fully on screen; phones: below the hero buttons
    const heroPx = narrow ? 0.06 * H : 0.1 * H;
    // 2) side band: between the content (left ~70 %) and the rail, upper-middle, descending slowly with scroll
    const bob = Math.sin(t * 1.1) * 2;
    const desc = smooth(0.1, 0.8, q);
    let anx, any, markPx;
    if (!narrow) { anx = 0.62; any = lerp(0.32, 0.12, desc) - bob / H * 2; markPx = clamp(0.13 * H, 95, 140); }   // right side, left of the rail, upper-middle
    else { anx = 0.62; any = 0.38 - bob / H * 2; markPx = 46; }   // phones: right side, below the nav, fully on screen
    const toGauge = smooth(0.2, 0.85, S.hero);
    // 3) finale: the marker converges on the physical lander above the site (same NDC as the camera target)
    const b = smooth(0.0, 0.35, f / TD);
    physP.copy(siteN).multiplyScalar(Lm + 1.2);                   // lander centre-ish, metres around the site
    const Dp = Math.max(0.5, physP.distanceTo(camB.position));
    const physPx = landerH * H / (2 * Dp * TANF);
    v3.copy(physP).project(camB);
    const pnx = v3.x, pny = v3.y;
    let nx = lerp(lerp(orbitP.x, anx, toGauge), pnx, b), ny = lerp(lerp(orbitP.y, any, toGauge), pny, b);
    let px = lerp(lerp(heroPx, markPx, toGauge), Math.max(physPx, markPx * (1 - smooth(0.6, 1, f / TD))), b);
    let Dd = lerp(D, Dp, b);
    // pilot beat: bring the lander big to screen centre, then fly into it (setPilotFocus)
    if (pf > 0.001) {
      // brief dive-in bump: centre + bigger while pf rises, then back to the small right-side lander at pf = 1
      // (js/pilot.js holds pf = 1 for the whole network fly-through, so the side lander stays visible there)
      const a = smooth(0, 0.45, pf) * (1 - smooth(0.75, 1, pf));
      nx = lerp(nx, 0, a); ny = lerp(ny, 0.05, a); px = lerp(px, 0.3 * H, a);
    }
    let sc = scaleForPx(px, Dd);
    if (b >= 0.999) { sc = Math.max(1, markPx * (1 - smooth(0.6, 1, f / TD)) / physPx); }
    const scPre = sc;
    sc *= lerp(1, EXAG, zo);
    ndcToB(nx, ny, Dd, ovP);
    if (b >= 0.999) ovP.copy(physP);
    // feet below the NDC point: lander centre for the overlay, 1.2 m (x scale) for the physical lander
    qOv.copy(camB.quaternion).multiply(qTilt);
    landerRoot.quaternion.slerpQuaternions(qOv, ground.quaternion, b);
    v4.set(0, 1, 0).applyQuaternion(landerRoot.quaternion);
    landerRoot.position.copy(ovP).addScaledVector(v4, -lerp(landerH * 0.5, 1.2, b) * sc);
    if (b >= 0.999) landerRoot.position.copy(siteN).multiplyScalar(Lm + 1.2 * (1 - scPre));
    landerRoot.scale.setScalar(sc);
    landerRoot.visible = !(narrow && f <= 0.02);   // phones: the side lander is a DOM sprite above the content (index.html #side-lander) until the finale
    // attitude: tumble gently in orbit, upright marker in descent, bay towards the viewer in the finale
    const yawMark = 1.25 + 0.08 * Math.sin(t * 0.3), yawFin = -1.2;
    const neutral = smooth(0, 0.45, pf) * (1 - smooth(0.75, 1, pf));
    landerPivot.rotation.set(
      lerp(lerp(lerp(0.45 * Math.sin(t * 0.3), 0.04 * Math.sin(t * 0.7), toGauge), 0, b), 0.12, neutral),
      lerp(lerp(lerp(t * 0.22, yawMark, toGauge), yawFin, b), -0.6, neutral),
      lerp(lerp(lerp(0.3, 0.03 * Math.sin(t * 0.9), toGauge), 0, b), 0, neutral));
    // thrust-residual arrows (pilot beat)
    for (let i = 0; i < 3; i++) {
      const a = arrows[i], c = thrustV ? thrustV[i] || 0 : 0;
      a.visible = thrustK > 0.01 && Math.abs(c) > 0.02;
      if (!a.visible) continue;
      v3.copy(AX[i]).multiplyScalar(Math.sign(c)); a.setDirection(v3);
      a.setLength(0.6 + 1.8 * Math.min(1, Math.abs(c)), 0.35, 0.2);
      a.line.material.opacity = a.cone.material.opacity = thrustK;
    }

    // ---------- landing site ----------
    const landed = f >= TD * 0.999;
    if (q > 0.45 || f > 0) loadRover();
    ground.visible = camAltKm < 70 && !!nearMat;
    if (wideMat) wideMat.opacity = 1 - smooth(22, 60, camAltKm);
    if (nearMat) nearMat.opacity = 1 - smooth(3, 9, camAltKm);
    ground.position.set(0, 0, 0);
    const sep = smooth(0.025, 0.075, q);   // separation early in the scroll
    const thr = landed ? 0 : f > 0 ? 1 : (sep > 0.7 && toGauge > 0.6 ? 0.7 : 0);
    flame.visible = thr > 0;
    flame.scale.set(1, (0.7 + 0.3 * thr) * (0.92 + 0.08 * Math.sin(t * 40) + 0.05 * Math.sin(t * 73)), 1);
    flameMat.uniforms.k.value = 0.3 * thr;          // subtle plume (20-30 %)
    flameLight.intensity = 6 * thr;
    // separation: the propulsion module drops away, tilts and dims out
    pmGroup.visible = sep < 0.999;
    puff.update(sep);
    if (pmGroup.visible) {
      // calm release (user request): the module keeps the world pose it had at separation and only drifts slowly
      // (linear drift + a very gentle damped tumble, a few degrees over the whole range); it no longer inherits the lander's spin.
      const pn = pmGroup.parent;
      if (pn && sep < 0.002) {                              // attached: remember its world pose every frame
        pmGroup.position.set(0, -1.0, 0); pmGroup.quaternion.identity();
        pn.updateWorldMatrix(true, false); pmGroup.updateMatrixWorld(true);
        pmGroup.getWorldPosition(pmP0); pmGroup.getWorldQuaternion(pmQ0); pmS0 = landerRoot.scale.x;
      } else if (pn) {
        const k = 1 - Math.exp(-sep * 3);                     // damped: fast at first, then settles
        pmW.copy(pmP0).add(pmD.set(-1.6 * sep, -2.6 * sep, -0.8 * sep).multiplyScalar(pmS0));
        pn.updateWorldMatrix(true, false);
        pmGroup.position.copy(pn.worldToLocal(pmW));
        pmQ.copy(pmQ0).multiply(pmQt.setFromEuler(pmE.set(0.09 * k, 0.05 * k, -0.07 * k)));   // ≤ ~5° total tumble
        pmGroup.quaternion.copy(pn.getWorldQuaternion(pmQi).invert().multiply(pmQ));
      }
      const o = 1 - smooth(0.25, 1, sep); for (let i = 0; i < pmMats.length; i++) pmMats[i].opacity = o;
    }
    sunB.castShadow = b > 0.2;
    rim.intensity = lerp(0.9, 0.1, b);                     // overlay phases sit ~7000 km from the site origin: no shadow maps there
    // sun: same direction as the sphere's; shadow box widens when the camera pulls back over the route
    const sh = lerp(70, 900, zo);
    sunB.target.position.copy(b > 0.5 ? (zo > 0 ? siteToB(lerp(0, routeObj ? routeObj.centre.x : 0, zo), 0, lerp(0, routeObj ? routeObj.centre.z : 0, zo), vC) : vC.set(0, 0, 0)) : landerRoot.position);
    sunB.position.copy(sunB.target.position).addScaledVector(SUN_DIR, 85 + sh * 2);
    if (sunB.shadow.camera.right !== sh) { Object.assign(sunB.shadow.camera, { left: -sh, right: sh, top: sh, bottom: -sh, far: 400 + sh * 6 }); sunB.shadow.camera.updateProjectionMatrix(); }
    // dust: one soft ring spreading out for ~1.2 s at touchdown, then stillness
    if (landed && tdTime < 0) tdTime = t; else if (!landed) tdTime = -1;
    const de = tdTime < 0 ? 1 : (t - tdTime) / 1.2;
    dust.update(de < 1 ? 1 - de * de : 0, de);
    // ramp opens, rover drives out of the side bay, down the ramp and onto the real terrain
    if (rampHinge) rampHinge.rotation.z = (Math.PI / 2 + RAMP_A) * smooth(TD + 0.02, TD + 0.09, f);
    const route = routeObj;
    if (rover && landerNode) {
      roverPath.visible = f > TD + 0.03;
      if (roverPath.visible) {
        landerRoot.updateMatrixWorld(true); ground.updateMatrixWorld(true);
        placeRover((L0 + L1 + L2) * smooth(TD + 0.09, Math.min(0.95, TD + 0.4), f));
        if (rv > 0.0005 && route) {
          // rover beat: autonomous drive along the real planned route (site frame), from the ramp foot.
          // Position by ARC LENGTH along the route with a damped parameter (no stalls on the hourly waiting samples, no
          // jitter); heading from a look-ahead tangent; attitude from the terrain normal under the (drawn-size) footprint.
          roverPath.getWorldPosition(rampEnd);
          const sT = smooth(0.12, 0.95, rv) * route.total;
          if (roverS == null || document.documentElement.classList.contains('shot')) roverS = sT; else roverS += (sT - roverS) * 0.18;
          routeAt(route, roverS, rpos);
          routeAt(route, Math.min(route.total, roverS + 25), rfwd); routeAt(route, Math.max(0, roverS - 25), rback);
          const ex = lerp(1, EXAG, zo), half = 0.65 * ex;          // half footprint of the drawn rover (m)
          let tx = rfwd.x - rback.x, tz = rfwd.z - rback.z; const tl = Math.hypot(tx, tz);
          if (tl > 1e-3) { tx /= tl; tz /= tl; roverTan.set(tx, 0, tz); } else roverTan.set(-1, 0, 0);
          if (roverTanS.lengthSq() < 1e-6 || document.documentElement.classList.contains('shot')) roverTanS.copy(roverTan);
          else roverTanS.lerp(roverTan, 0.12).normalize();          // short-way blend of the heading vector (no angle wrap flips)
          const fx2 = roverTanS.x, fz2 = roverTanS.z, sx = -fz2, sz = fx2;
          // terrain under the four wheel corners (site frame); sit on the highest so it never sinks in
          const hF = heightAt(rpos.x + fx2 * half, rpos.z + fz2 * half), hB = heightAt(rpos.x - fx2 * half, rpos.z - fz2 * half);
          const hL = heightAt(rpos.x + sx * half * 0.7, rpos.z + sz * half * 0.7), hR = heightAt(rpos.x - sx * half * 0.7, rpos.z - sz * half * 0.7);
          const hy = Math.max((hF + hB) / 2, (hL + hR) / 2, heightAt(rpos.x, rpos.z));
          rUp.set(-(hF - hB) / (2 * half) * fx2 - (hL - hR) / (1.4 * half) * sx, 1, -(hF - hB) / (2 * half) * fz2 - (hL - hR) / (1.4 * half) * sz).normalize();
          if (rUpS.lengthSq() < 1e-6 || document.documentElement.classList.contains('shot')) rUpS.copy(rUp); else rUpS.lerp(rUp, 0.15).normalize();
          siteToB(rpos.x, hy, rpos.z, routeP);
          const k0 = smooth(0, 0.12, rv);
          roverW.copy(rampEnd).lerp(routeP, k0);
          // world attitude: forward (site tangent), up (terrain normal) -> roverPath frame (-x forward, +y up)
          siteToB(fx2, 0, fz2, rF); siteToB(rUpS.x, rUpS.y, rUpS.z, rU);
          rF.addScaledVector(rU, -rF.dot(rU)).normalize();
          rX.copy(rF).negate(); rZ.crossVectors(rX, rU);
          rM.makeBasis(rX, rU, rZ); rQ.setFromRotationMatrix(rM);
          landerNode.getWorldQuaternion(qInv).invert();
          landerNode.worldToLocal(v4.copy(roverW));
          roverPath.position.copy(v4);
          qTmp.copy(roverPath.quaternion);                                        // ramp pose from placeRover()
          roverPath.quaternion.copy(qInv).multiply(rQ);
          if (k0 < 0.999) roverPath.quaternion.slerpQuaternions(qTmp, roverPath.quaternion.clone(), k0);   // ease off the ramp
          landerNode.getWorldScale(vB);
          roverPath.scale.setScalar(ex / Math.max(1e-6, vB.x));
          for (let w = 0; w < wheels.length; w++) wheels[w].rotation.z = -roverS / 0.086;   // wheels turn with distance driven
          roverScreen.copy(roverW);
        } else { roverS = null; roverTanS.set(0, 0, 0); rUpS.set(0, 0, 0); }
      }
    }
    if (route) {
      const on = zo;
      route.mFull.opacity = 0.3 * on; route.mDone.opacity = 0.95 * on; route.ringM.opacity = 0.55 * on; route.ringHot.opacity = 0.95 * on; route.hzM.opacity = 0.5 * on;
      let iDone = 0; if (rv > 0 && roverS != null) { const D = route.D; while (iDone < route.n - 1 && D[iDone + 1] <= roverS) iDone++; }
      route.gDone.setDrawRange(0, rv > 0 ? Math.min(route.n, iDone + 1) : 0);
      route.link.visible = on > 0.02 && roverPath.visible;
      if (route.link.visible) {
        landerRoot.getWorldPosition(landerTop).addScaledVector(siteN, landerH * landerRoot.scale.x);
        roverPath.getWorldPosition(roverW).addScaledVector(siteN, 1.2 * lerp(1, EXAG, zo));
        const a = route.lg.attributes.position; a.setXYZ(0, landerTop.x, landerTop.y, landerTop.z); a.setXYZ(1, roverW.x, roverW.y, roverW.z); a.needsUpdate = true;
        if (frameN % 3 === 0) route.link.computeLineDistances();
        route.lm.opacity = 0.7 * on;
      }
      if (labels) {
        const lo = smooth(0.2, 0.3, rv) * (1 - smooth(0.97, 1, rv));
        roverPath.getWorldPosition(roverW).addScaledVector(siteN, 20);
        label(labels.rover, roverW, lo);
        landerRoot.getWorldPosition(landerTop).addScaledVector(siteN, landerH * landerRoot.scale.x + 6);
        label(labels.lander, landerTop, lo);
        if (route.reachT) { siteToB(route.reachT[0], heightAt(route.reachT[0], route.reachT[2]) + 14, route.reachT[2], vC); label(labels.ice, vC, lo); }
        if (labels.hazard && route.P.length) { const hz = routeData.hazard_cells_xz; if (hz.length) { const c = hz[Math.floor(hz.length / 2)]; siteToB(c[0], heightAt(c[0], c[1]) + 6, c[1], vC); label(labels.hazard, vC, lo); } }
      }
    }

    if (DBG && (frameN % 20) === 0) dbgEl().textContent = JSON.stringify({ q: +q.toFixed(3), f: +f.toFixed(3), rv: +rv.toFixed(3), alt: +altKm.toFixed(3), cam: +camAltKm.toFixed(3), near: nearMat ? +nearMat.opacity.toFixed(2) : -1, gv: ground.visible, b: +b.toFixed(2), sc: +sc.toFixed(2), y: document.body.scrollTop, kids: ground.children.length, h0: +heightAt(0,0).toFixed(2), h1: +heightAt(300,-300).toFixed(2), lr: landerRoot.position.toArray().map(v=>+v.toFixed(2)), cb: camB.position.toArray().map(v=>+v.toFixed(2)), gq: ground.quaternion.toArray().map(v=>+v.toFixed(3)) });
    // ---------- render ----------
    // hero video backdrop: transparent space while the hero is on screen, back to opaque navy as you scroll
    const bgA = backdrop ? smooth(0.25, 0.95, S.hero) : 1;
    renderer.setClearColor(BG, bgA);
    starMat.opacity = lerp(0.35, 1, bgA);
    renderer.clear();
    renderer.render(sceneA, camA);
    if (net && netMix > 0.001) {
      net.update(netQ, netMix, t);
      renderer.render(net.scene, net.camera);
    } else if (net) net.update(netQ, 0, t);
    renderer.clearDepth();
    renderer.render(sceneB, camB);
    // QA hook (tools/qa/qa_scroll.mjs): where the 3D lander is on screen and how big
    if ((frameN & 7) === 0) {
      const v = landerRoot.getWorldPosition(qaV).project(camB);
      const top = qaV2.copy(landerRoot.position).addScaledVector(camB.up, landerH * landerRoot.scale.x).project(camB);
      let rvq = null; if (roverPath.visible && roverS != null) { roverPath.getWorldPosition(qaV3).project(camB); rvq = { x: (qaV3.x + 1) / 2 * W, y: (1 - qaV3.y) / 2 * H, s: +roverS.toFixed(1) }; }
      window.__qa = { rover: rvq, lander: { x: (v.x + 1) / 2 * W, y: (1 - v.y) / 2 * H, hPx: Math.abs(top.y - v.y) / 2 * H, visible: landerRoot.visible && v.z < 1, inView: v.x > -1 && v.x < 1 && v.y > -1 && v.y < 1 }, q: +q.toFixed(3), f: +f.toFixed(3) };
    }
  }

  const qaV = new THREE.Vector3(), qaV2 = new THREE.Vector3(), qaV3 = new THREE.Vector3();
  const pmP0 = new THREE.Vector3(), pmW = new THREE.Vector3(), pmD = new THREE.Vector3(), pmQ0 = new THREE.Quaternion(), pmQ = new THREE.Quaternion(), pmQt = new THREE.Quaternion(), pmQi = new THREE.Quaternion(), pmE = new THREE.Euler();
  let pmS0 = 1;
  let running = true;
  let lastFrameMs = 0;
  const tick = () => { if (!running) return; lastFrameMs = performance.now(); try { frame(); } catch (e) { if (DBG) dbgEl().textContent = 'ERR ' + e.message + ' ' + (e.stack || '').split('\n')[1]; else if (frameN % 300 === 1) console.warn('scene frame', e); frameN++; } };
  renderer.setAnimationLoop(tick);
  // screenshot mode (?scroll=): headless Chrome stops firing rAF under virtual time; keep the scene current with a slow timer
  if (new URLSearchParams(location.search).has('scroll')) setInterval(() => { if (performance.now() - lastFrameMs > 150) tick(); }, 100);
  document.addEventListener('visibilitychange', () => { running = !document.hidden; });

  return {
    setProgress(p) { progress = clamp(p); tlDirty = true; },
    setNet(q, mix) { netQ = clamp(q); netMix = clamp(mix); },
    // pilot beat (js/pilot.js): k 0..1 brings the lander to big screen centre, k -> 1 flies the camera into it
    setPilotFocus(k) { pilotFocus = clamp(+k || 0); },
    // v = [x, y, z] normalised residual (x / y lateral, z vertical) or null; k = opacity
    setThrustArrows(v, k) { thrustV = Array.isArray(v) ? [v[0] || 0, v[1] || 0, v[2] || 0] : null; thrustK = clamp(+k || 0); },
    // rover beat (js/rover.js): p = #rover section progress (null = read it from the timeline); labels = {rover, lander, ice, hazard} DOM nodes
    setRover(p) { roverP = p == null ? null : clamp(p); tlDirty = true; },
    setRoverLabels(l) { labels = l || null; },
    onRoverStep(fn) { hudFn = typeof fn === 'function' ? fn : null; lastStep = -1; },
    roverExaggeration: EXAG,
    renderer,
  };
}

function grainTexture() {
  const n = 128, cv = document.createElement('canvas'); cv.width = cv.height = n;
  const g = cv.getContext('2d'), im = g.createImageData(n, n);
  for (let i = 0; i < n * n; i++) { const v = 228 + Math.random() * 27; im.data.set([v, v, v * 0.98, 255], i * 4); }
  g.putImageData(im, 0, 0);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function makePuff() {   // separation: a brief soft ring of vented gas expanding from the interface ring, scrubbed by scroll
  const n = 160, p = new Float32Array(n * 3), seed = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) seed.set([Math.random() * Math.PI * 2, 0.6 + Math.random() * 0.8, (Math.random() - 0.5) * 0.5], i * 3);
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3));
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;
  const cx = cv.getContext('2d'), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
  const m = new THREE.PointsMaterial({ color: 0xe6dcc8, size: 0.5, map: new THREE.CanvasTexture(cv), transparent: true, opacity: 0, depthWrite: false });
  const points = new THREE.Points(g, m); points.frustumCulled = false; points.visible = false;
  let last = -1;
  return { points, update(sep) {
    const k = clamp((sep - 0.05) / 0.6);
    const a = Math.sin(Math.PI * k);
    points.visible = a > 0.01; m.opacity = 0.7 * a;
    if (!points.visible || Math.abs(k - last) < 1e-4) return; last = k;
    for (let i = 0; i < n; i++) { const r = seed[i * 3 + 1] * (0.4 + 2.6 * Math.sqrt(k)); p.set([Math.cos(seed[i * 3]) * r, seed[i * 3 + 2] * k, Math.sin(seed[i * 3]) * r], i * 3); }
    g.attributes.position.needsUpdate = true;
  } };
}

function makeDust() {
  const n = 700, p = new Float32Array(n * 3), seed = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) seed.set([Math.random() * Math.PI * 2, 1.5 + Math.random() * 9, Math.random() * 0.8, Math.random()], i * 4);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(p, 3));
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;     // soft round puff
  const cx = cv.getContext('2d'), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.5, 'rgba(255,255,255,.45)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
  const m = new THREE.PointsMaterial({ color: 0x9a907f, size: 1.0, map: new THREE.CanvasTexture(cv), transparent: true, opacity: 0, depthWrite: false });
  const points = new THREE.Points(g, m);
  points.frustumCulled = false;
  points.renderOrder = 5;          // after the (transparent) terrain, which would otherwise paint over it
  return {
    points,
    update(amt, t) {
      m.opacity = 0.65 * amt;
      points.visible = amt > 0.01;
      if (!points.visible) return;
      for (let i = 0; i < n; i++) {
        const a = seed[i * 4], r = seed[i * 4 + 1], h = seed[i * 4 + 2], ph = seed[i * 4 + 3];
        const rr = (1.2 + r * 0.6) * (0.35 + 1.1 * Math.sqrt(t)) * (0.85 + 0.3 * ph);   // t = ring age 0..1
        p[i * 3] = Math.cos(a) * rr; p[i * 3 + 1] = h * 0.6 * (1 - t); p[i * 3 + 2] = Math.sin(a) * rr;
      }
      g.attributes.position.needsUpdate = true;
    },
  };
}

function earthTexture() {   // small procedural Earth: oceans, a few land masses, cloud bands (decorative)
  const w = 256, h = 128, cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const g = cv.getContext('2d');
  g.fillStyle = '#1d4f9e'; g.fillRect(0, 0, w, h);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  g.fillStyle = '#5d7a45';
  for (let i = 0; i < 14; i++) { g.beginPath(); g.ellipse(rnd() * w, 20 + rnd() * 88, 8 + rnd() * 26, 5 + rnd() * 16, rnd() * 3, 0, 7); g.fill(); }
  g.fillStyle = 'rgba(255,255,255,.75)';
  for (let i = 0; i < 40; i++) { g.beginPath(); g.ellipse(rnd() * w, rnd() * h, 6 + rnd() * 30, 1.5 + rnd() * 4, (rnd() - 0.5) * 0.4, 0, 7); g.fill(); }
  g.fillStyle = '#f2f6fb'; g.fillRect(0, 0, w, 7); g.fillRect(0, h - 7, w, 7);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}
