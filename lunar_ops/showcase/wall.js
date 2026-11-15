// Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in wall.html.
// Mission wall (screen 1): QuickMap-style 3D terrain with data layers (terrain, AI hazard output, safe cells, SPICE light,
// Earth link, dark craters, ice evidence), live lander and target, plus trajectory, flight data, vehicle and events.
// Every value comes from /api/layers or the live feed (/ws/ui); nothing on this screen is decorative.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (id) => document.getElementById(id), clamp = (v, a, b) => Math.min(b, Math.max(a, v)), lerp = (a, b, t) => a + (b - a) * t;
const fmt = (x, d = 0) => (x == null || !isFinite(x) ? '—' : x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));
const R_MOON = 1737.4e3; // [m]
const PHOTO_MEAN = 0.46; // both LROC photos (tile 1 m, context 2 m) are scaled to this mean brightness so they match
function photoFilter(img) { // brightness gain from the image's own mean (measured on a 64 px copy)
  const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'); g.drawImage(img, 0, 0, 64, 64); const d = g.getImageData(0, 0, 64, 64).data;
  let s = 0; for (let i = 0; i < d.length; i += 4) s += (d[i] + d[i + 1] + d[i + 2]) / 765; const mean = s / (d.length / 4);
  return `brightness(${Math.min(4, PHOTO_MEAN / Math.max(mean, 0.02)).toFixed(3)}) contrast(1.12)`; }
const stage = $('stage'); const fit = () => { const s = Math.min(innerWidth / 1920, innerHeight / 1080); stage.style.transform = `translate(${-960 * s}px, ${-540 * s}px) scale(${s})`; }; addEventListener('resize', fit); fit();
const b64 = (s) => { const bin = atob(s), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; };
const bits = (u8, n) => { const o = new Uint8Array(n); for (let i = 0; i < n; i++) o[i] = (u8[i >> 3] >> (7 - (i & 7))) & 1; return o; };
function turbo(t) { t = clamp(t, 0, 1); const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05)))), g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56)))), b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66)))); return [clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)]; }
const ps = (latDeg, lonDeg) => { const k = 2 * R_MOON * Math.tan(Math.PI / 4 + latDeg * Math.PI / 360), lo = lonDeg * Math.PI / 180; return [k * Math.sin(lo), k * Math.cos(lo)]; };
const latlon = (x, y) => [(2 * Math.atan(Math.hypot(x, y) / (2 * R_MOON)) - Math.PI / 2) * 180 / Math.PI, Math.atan2(x, y) * 180 / Math.PI];

// ---------------------------------------------------------------------------------------------------- 3D scene
const host = $('terrain'), W = 1260, H = 900;
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true }); renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(W, H); host.prepend(renderer.domElement);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x02040a);
const camera = new THREE.PerspectiveCamera(38, W / H, 0.5, 1e8);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.minDistance = 40; controls.maxDistance = 9e6;
const sun = new THREE.DirectionalLight(0xfff1dc, 2.6); scene.add(sun, sun.target); scene.add(new THREE.HemisphereLight(0xa9c2ff, 0x2b241d, 0.75));

let L = null, world = null, terrainMesh = null, ctxMesh = null, h0 = 0, exag = 1.5, layer = new URLSearchParams(location.search).get('layer') || 'hazard', tIdx = 0, liveTime = true, lastFrame = null;
const tex = { canvas: document.createElement('canvas'), photo: null }; tex.canvas.width = tex.canvas.height = 1536; const ctex = new THREE.CanvasTexture(tex.canvas); ctex.colorSpace = THREE.SRGBColorSpace; ctex.anisotropy = 8;
const live = { group: new THREE.Group(), track: [], trackLine: null, lander: null, target: null, hist: [], events: [] }; scene.add(live.group);

const LAYERS = {
  imagery: { name: 'Imagery', title: 'LROC NAC photo mosaic', a: '', z: '', bar: 'linear-gradient(90deg,#222,#bbb)', prov: 'OBSERVED · NASA LRO LROC NAC, 1 m/px (averaged lighting)' },
  elevation: { name: 'Elevation', title: 'Height (LOLA)', bar: 'linear-gradient(90deg,' + [0, .2, .4, .6, .8, 1].map((t) => `rgb(${turbo(t).map(Math.round)})`).join(',') + ')', prov: 'OBSERVED · NASA LRO LOLA 5 m' },
  slope: { name: 'Slope', title: 'Slope · limit 12°', a: '0°', z: '30°', bar: 'linear-gradient(90deg,' + [0, .2, .4, .6, .8, 1].map((t) => `rgb(${turbo(t).map(Math.round)})`).join(',') + ')', prov: 'derived from LOLA 5 m' },
  hazard: { name: 'Hazard (AI)', title: 'Hazard probability', a: 'safe 0', z: '1 unsafe', bar: 'linear-gradient(90deg,rgba(255,165,102,0),#ffa566,#f43f5e)', prov: '' },
  safe: { name: 'Safe cells', title: 'Where a lander may aim', a: '', z: '', bar: 'linear-gradient(90deg,#2f7dff,#8db9ff)', prov: 'no hazard within 15 m and slope < 6° within 10 m' },
  sun: { name: 'Sunlight', title: 'Sun above the local horizon', a: 'shadow', z: 'sunlit', bar: 'linear-gradient(90deg,#0b1430,#ffd696)', prov: 'MODELLED · SPICE DE440 + LOLA horizon' },
  earth: { name: 'Earth link', title: 'Earth above the local horizon', a: 'no link', z: 'direct link', bar: 'linear-gradient(90deg,#0b1430,#5ee6ff)', prov: 'MODELLED · SPICE DE440 + LOLA horizon' },
  psr: { name: 'Dark craters', title: 'Permanently shadowed regions', a: '', z: '', bar: 'linear-gradient(90deg,#3b2a8f,#8c6cff)', prov: 'MODELLED · horizon test over the lunar year' },
  ice: { name: 'Ice evidence', title: 'Ice EVIDENCE score (not confirmed ice)', a: '0', z: '1', bar: 'linear-gradient(90deg,#123a8a,#2f7dff,#5ee6ff,#ffffff)', prov: 'MODELLED · PSR + cold-trap + neutron evidence' },
};
const lw = $('layers'); for (const [k, v] of Object.entries(LAYERS)) { const b = document.createElement('button'); b.className = 'lb' + (k === layer ? ' on' : ''); b.textContent = v.name; b.onclick = () => { layer = k; for (const x of lw.children) x.classList.toggle('on', x === b); paint(); }; lw.appendChild(b); }

function paint() { // compose photo + the active layer into the terrain texture
  if (!L) return; const g = tex.canvas.getContext('2d'), N = tex.canvas.width; g.globalAlpha = 1; g.fillStyle = '#4a4844'; g.fillRect(0, 0, N, N);
  if (tex.photo) { g.filter = tex.photoFilter || 'none'; g.drawImage(tex.photo, 0, 0, N, N); g.filter = 'none'; }
  const { rows, cols } = L, im = new ImageData(cols, rows), d = im.data, lay = layer;
  const hmin = L.hmin, hmax = L.hmax, sunT = L.sun ? L.sun.subarray(tIdx * rows * cols, (tIdx + 1) * rows * cols) : null, earthT = L.earth ? L.earth.subarray(tIdx * rows * cols, (tIdx + 1) * rows * cols) : null;
  for (let i = 0; i < rows * cols; i++) { let c = null, a = 0;
    if (lay === 'elevation') { c = turbo((L.height[i] - hmin) / (hmax - hmin)); a = 0.82; }
    else if (lay === 'slope') { c = turbo(L.slope[i] / 4 / 30); a = 0.8; }
    else if (lay === 'hazard') { const p = L.hazard[i] / 255; c = p > 0.5 ? [244, 63, 94] : [255, 165, 102]; a = p * 0.5; }
    else if (lay === 'safe') { if (L.safe[i]) { c = [47, 125, 255]; a = 0.6; } }
    else if (lay === 'sun') { c = sunT[i] ? [255, 214, 150] : [8, 16, 44]; a = sunT[i] ? 0.35 : 0.72; }
    else if (lay === 'earth') { c = earthT[i] ? [94, 230, 255] : [8, 16, 44]; a = earthT[i] ? 0.55 : 0.6; }
    else if (lay === 'psr') { if (L.psr[i]) { c = [140, 108, 255]; a = 0.7; } }
    else if (lay === 'ice') { const s = L.ice[i] / 255; if (s > 0.12) { c = s < 0.5 ? [lerp(18, 47, s * 2), lerp(58, 125, s * 2), lerp(138, 255, s * 2)] : [lerp(47, 255, (s - .5) * 2), lerp(125, 255, (s - .5) * 2), 255]; a = 0.35 + 0.55 * s; } }
    if (c) { d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = a * 255; } }
  const tmp = document.createElement('canvas'); tmp.width = cols; tmp.height = rows; tmp.getContext('2d').putImageData(im, 0, 0); g.imageSmoothingEnabled = lay !== 'safe' && lay !== 'psr'; g.drawImage(tmp, 0, 0, N, N);
  ctex.needsUpdate = true; legend();
}
function legend() {
  const v = LAYERS[layer]; $('lg-t').textContent = v.title; $('lg-b').style.background = v.bar;
  $('lg-a').textContent = layer === 'elevation' ? `${fmt(L.hmin - h0)} m` : v.a; $('lg-z').textContent = layer === 'elevation' ? `+${fmt(L.hmax - h0)} m` : v.z;
  $('lg-p').textContent = layer === 'hazard' ? `MODELLED · ${L.hazard_source}${/backup/.test(L.hazard_source) ? ' (the hazard U-Net replaces it when its checkpoint is loaded)' : ''}` : v.prov;
}

function buildTerrain() {
  const { rows, cols, mpp } = L, Wm = (cols - 1) * mpp, Hm = (rows - 1) * mpp;
  const geo = new THREE.PlaneGeometry(Wm, Hm, cols - 1, rows - 1); geo.rotateX(-Math.PI / 2);
  terrainMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: ctex, roughness: 1, metalness: 0 })); scene.add(terrainMesh); setHeights();
  // wide context 1: the 20 m LOLA context around the tile (from the world message)
  if (world?.context) { const X = world.context, n = X.rows, m = X.cols, Hc = new Float32Array(b64(X.height_f32).buffer), x0 = X.offset_x_m - (m - 1) / 2 * X.mpp, y0 = X.offset_y_m + (n - 1) / 2 * X.mpp;
    const pos = [], uv = [], idx = [], inside = (xa, ya) => Math.abs(xa) < Wm / 2 - 10 && Math.abs(ya) < Hm / 2 - 10;
    for (let r = 0; r < n; r++) for (let c = 0; c < m; c++) { pos.push(x0 + c * X.mpp, 0, -(y0 - r * X.mpp)); uv.push(c / (m - 1), 1 - r / (n - 1)); }
    for (let r = 0; r < n - 1; r++) for (let c = 0; c < m - 1; c++) { const xa = x0 + (c + .5) * X.mpp, ya = y0 - (r + .5) * X.mpp; if (inside(xa, ya)) continue; const a = r * m + c; idx.push(a, a + m, a + 1, a + 1, a + m, a + m + 1); }
    const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); cg.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); cg.setIndex(idx); cg.userData.H = Hc;
    const cv = document.createElement('canvas'); cv.width = cv.height = 2560; const ct = new THREE.CanvasTexture(cv); ct.colorSpace = THREE.SRGBColorSpace; ct.anisotropy = 8;
    const img = new Image(); img.onload = () => { const g = cv.getContext('2d'); g.filter = photoFilter(img); g.drawImage(img, 0, 0, cv.width, cv.height); ct.needsUpdate = true; }; img.src = '/' + (world?.photo?.context || 'assets/site01/photo_ctx_2m.jpg'); // same processing as the tile photo
    ctxMesh = new THREE.Mesh(cg, new THREE.MeshStandardMaterial({ map: ct, roughness: 1, color: new URLSearchParams(location.search).get('debug') === 'levels' ? 0xffff60 : 0xffffff })); scene.add(ctxMesh); setHeights(); }
  buildGlobe(); buildPolar().catch((e) => console.warn('polar terrain', e));
}
// ---------------------------------------------------------------------------------------------------- the whole Moon
// A globe in the Moon-fixed frame (MOON_PA: z = north pole), carrying the LROC global map and, south of 70 deg S, the sharp
// LROC WAC polar mosaic. Placed so that the tile's centre, its east (+x) and north (+y) axes line up with the 5 m terrain:
// the globe surface under the site sits at the mean height of the tile's edge, so the tile blends into it.
const globe = { capBox: { value: new THREE.Vector4(1, -1, 1, -1) }, levels: [], mesh: null, M: new THREE.Matrix4(), center: new THREE.Vector3(), track: [], trackLine: null, dot: null, ring: null, trueSun: false };
function pP(lat, lon, r = R_MOON) { const la = lat * Math.PI / 180, lo = lon * Math.PI / 180; return new THREE.Vector3(r * Math.cos(la) * Math.cos(lo), r * Math.cos(la) * Math.sin(lo), r * Math.sin(la)); }
function tileAxes() { // Moon-fixed unit vectors of the tile's east (+x), up and north (+y) at its centre (finite differences of the projection)
  const [cx, cy] = ps(L.center_lat_deg, L.center_lon_deg), at = (dx, dy) => { const [la, lo] = latlon(cx + dx, cy + dy); return pP(la, lo); };
  const up = pP(L.center_lat_deg, L.center_lon_deg).normalize(), ex = at(50, 0).sub(at(-50, 0)), ey = at(0, 50).sub(at(0, -50));
  ex.addScaledVector(up, -ex.dot(up)).normalize(); ey.addScaledVector(up, -ey.dot(up)).addScaledVector(ex, -ey.dot(ex)).normalize(); return { ex, up, ey };
}
function buildGlobe() {
  const A = tileAxes(); globe.A = A;
  globe.M.set(A.ex.x, A.ex.y, A.ex.z, 0, A.up.x, A.up.y, A.up.z, 0, -A.ey.x, -A.ey.y, -A.ey.z, 0, 0, 0, 0, 1); // Moon-fixed -> scene (x east, y up, z south)
  const TL = new THREE.TextureLoader(), mapE = TL.load('assets/moon/moon_color_4k.jpg'), mapP = TL.load('assets/moon/polar_south_wac.jpg'); for (const t of [mapE, mapP]) { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; }
  const geo = new THREE.SphereGeometry(R_MOON, 720, 360); geo.rotateX(Math.PI / 2); // tessellation poles on the Moon's spin axis
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.capBox = globe.capBox; sh.uniforms.siteP = { value: A.up.clone() }; sh.uniforms.holeCos = { value: Math.cos(2300 / R_MOON) }; sh.uniforms.mapE = { value: mapE }; sh.uniforms.mapP = { value: mapP }; sh.uniforms.polarHalf = { value: 612744.8896254691 }; sh.uniforms.moonR = { value: R_MOON };
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vMoonP;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvMoonP = position;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vMoonP; uniform vec4 capBox; uniform vec3 siteP; uniform float holeCos; uniform sampler2D mapE; uniform sampler2D mapP; uniform float polarHalf; uniform float moonR;')
      .replace('#include <map_fragment>', `vec3 nP = normalize(vMoonP); if (dot(nP, siteP) > holeCos) discard; // the real 5 m / 20 m terrain is shown here
        { float la = asin(clamp(nP.z, -1.0, 1.0)); if (la < -1.1) { float rr = 2.0 * moonR * tan(0.78539816 + la * 0.5), lo = atan(nP.y, nP.x), X = rr * sin(lo), Y = rr * cos(lo);
          if (X > capBox.x && X < capBox.y && Y > capBox.z && Y < capBox.w) discard; } } // the real LOLA polar cap is drawn here
        float lat = asin(clamp(nP.z, -1.0, 1.0)); float lon = atan(nP.y, nP.x);
        vec2 uvE = vec2(lon / 6.2831853 + 0.5, lat / 3.1415927 + 0.5); vec2 gx = dFdx(uvE), gy = dFdy(uvE);
        if (abs(gx.x) > 0.5) gx.x -= sign(gx.x); if (abs(gy.x) > 0.5) gy.x -= sign(gy.x);
        vec3 cE = textureGrad(mapE, uvE, gx, gy).rgb;
        float rho = 2.0 * moonR * tan(0.78539816 + lat * 0.5); vec2 uvP = vec2(0.5 + rho * sin(lon) / (2.0 * polarHalf), 0.5 + rho * cos(lon) / (2.0 * polarHalf));
        vec3 cP = texture2D(mapP, uvP).rgb; cP *= dot(cE, vec3(0.333)) / max(dot(textureGrad(mapP, uvP, dFdx(uvP) * 24.0, dFdy(uvP) * 24.0).rgb, vec3(0.333)), 0.02); // match the global map's local brightness
        float wP = smoothstep(-1.2217, -1.3439, lat); // blend 70 -> 77 deg S
        diffuseColor.rgb *= mix(cE, cP, wP);`);
  };
  globe.mesh = new THREE.Mesh(geo, mat); globe.mesh.matrixAutoUpdate = false; scene.add(globe.mesh);
  // site marker seen from far away
  globe.ring = new THREE.Mesh(new THREE.RingGeometry(9000, 13000, 64), new THREE.MeshBasicMaterial({ color: 0xffa566, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false }));
  globe.ring.rotation.x = -Math.PI / 2; globe.ring.renderOrder = 5; scene.add(globe.ring);
  globe.dot = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false })); globe.dot.renderOrder = 6; globe.dot.visible = false; scene.add(globe.dot);
  placeGlobe();
}
function placeGlobe() { // Moon-fixed -> scene: rotation M, then the site (at its LOLA height h0) to the origin
  if (!globe.mesh) return; globe.center.copy(globe.A.up).multiplyScalar(-(R_MOON + h0)).applyMatrix4(globe.M);
  globe.mesh.matrix.makeTranslation(globe.center.x, globe.center.y, globe.center.z).multiply(globe.M); globe.mesh.matrixWorldNeedsUpdate = true; globe.ring.position.set(0, 25, 0);
  for (const lv of globe.levels) placeLevel(lv);
}
// ---------------------------------------------------------------------------------------------------- nested real LOLA terrain (cap, region, local)
function latlonPS(x, y) { const rho = Math.hypot(x, y); return [2 * Math.atan(rho / (2 * R_MOON)) - Math.PI / 2, Math.atan2(x, y)]; }
async function buildPolar() {
  const r = await fetch(`/api/polar?lat=${L.center_lat_deg}&lon=${L.center_lon_deg}`); if (!r.ok) return; const J = await r.json(); globe.polar = J;
  const [cx, cy] = ps(L.center_lat_deg, L.center_lon_deg), wac = new THREE.TextureLoader().load('assets/moon/polar_south_wac.jpg'); wac.colorSpace = THREE.SRGBColorSpace; wac.anisotropy = 8;
  const half = 612744.8896254691, box = (lv) => [lv.x0_m - lv.step_m / 2, lv.x0_m + (lv.cols - 0.5) * lv.step_m, lv.y0_m - (lv.rows - 0.5) * lv.step_m, lv.y0_m + lv.step_m / 2];
  const lv = J.levels, hasLocal = lv.some((q) => q.name === 'local'), tileHalf = hasLocal ? (L.cols - 1) * L.mpp / 2 - 2 : (world?.context ? (world.context.cols - 1) / 2 * world.context.mpp - 30 : L.cols * L.mpp / 2 - 10);
  if (hasLocal && ctxMesh) { scene.remove(ctxMesh); ctxMesh = null; } // the old 20 m ring is off-centre (+1.28 km E, +1.28 km N); the 16 km DEM replaces it
  const ctxPhoto = await new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = '/' + (world?.photo?.context || 'assets/site01/photo_ctx_2m.jpg'); });
  let ctxTex = null, ctxGain = 1; if (ctxPhoto) { ctxTex = new THREE.Texture(ctxPhoto); ctxTex.colorSpace = THREE.SRGBColorSpace; ctxTex.anisotropy = 8; ctxTex.needsUpdate = true; const f = photoFilter(ctxPhoto).match(/brightness\(([\d.]+)\)/); ctxGain = f ? Number(f[1]) : 1; }
  lv.forEach((v, k) => { // hole = the next finer level's box (the finest one: the 5 m tile and its 20 m ring)
    const inner = k + 1 < lv.length ? box(lv[k + 1]) : [cx - tileHalf, cx + tileHalf, cy - tileHalf, cy + tileHalf];
    const h = new Int16Array(b64(v.height_i16).buffer), n = v.rows * v.cols, xy = new Float32Array(n * 2), uv = new Float32Array(n * 2);
    for (let r2 = 0; r2 < v.rows; r2++) for (let c = 0; c < v.cols; c++) { const i = r2 * v.cols + c, X = v.x0_m + c * v.step_m, Y = v.y0_m - r2 * v.step_m; xy[2 * i] = X; xy[2 * i + 1] = Y; uv[2 * i] = 0.5 + X / (2 * half); uv[2 * i + 1] = 0.5 + Y / (2 * half); }
    const idx = []; for (let r2 = 0; r2 < v.rows - 1; r2++) for (let c = 0; c < v.cols - 1; c++) { const a = r2 * v.cols + c; idx.push(a, a + v.cols, a + 1, a + 1, a + v.cols, a + v.cols + 1); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.setAttribute('psxy', new THREE.Float32BufferAttribute(xy, 2)); g.setIndex(idx);
    const DBG = new URLSearchParams(location.search).get('debug') === 'levels', mat = new THREE.MeshStandardMaterial({ map: wac, roughness: 1, metalness: 0, color: DBG ? { cap: 0xff6060, region: 0x60ff60, local: 0x6090ff }[v.name] : v.name === 'local' ? 0xd8d8d8 : 0xffffff });
    const useCtx = v.name === 'local' && ctxTex;
    mat.onBeforeCompile = (sh) => { sh.uniforms.hole = { value: new THREE.Vector4(...inner) }; sh.uniforms.ctxMap = { value: ctxTex }; sh.uniforms.ctxC = { value: new THREE.Vector2(cx, cy) }; sh.uniforms.ctxGain = { value: ctxGain };
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute vec2 psxy; varying vec2 vXY;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvXY = psxy;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 vXY; uniform vec4 hole; uniform sampler2D ctxMap; uniform vec2 ctxC; uniform float ctxGain;').replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\nif (vXY.x > hole.x && vXY.x < hole.y && vXY.y > hole.z && vXY.y < hole.w) discard;')
        .replace('#include <map_fragment>', useCtx ? '#include <map_fragment>\nvec2 dq = vXY - ctxC; float wc = 1.0 - smoothstep(2200.0, 2540.0, max(abs(dq.x), abs(dq.y))); if (wc > 0.0) { vec3 cc = texture2D(ctxMap, vec2(0.5 + dq.x / 5120.0, 0.5 + dq.y / 5120.0)).rgb * ctxGain; diffuseColor.rgb = mix(diffuseColor.rgb, cc, wc); }' : '#include <map_fragment>'); };
    mat.customProgramCacheKey = () => `level-${v.name}-${useCtx ? 'ctx' : 'wac'}`; // each level compiles its own shader (three caches by hook source)
    const mesh = new THREE.Mesh(g, mat); const L2 = { v, h, mesh, n }; scene.add(mesh); globe.levels.push(L2); placeLevel(L2);
    if (v.name === 'cap') globe.capBox.value.set(...box(v).map((q, i2) => q + (i2 % 2 ? -2 * v.step_m : 2 * v.step_m)));
  });
  const note = lv.map((v) => v.provenance).join(' | '); globe.provenance = note;
}
function placeLevel(L2) { // vertices on the sphere: radius R + h0 + (h - h0) * exaggeration, then into the scene
  const { v, h, mesh, n } = L2, pos = mesh.geometry.attributes.position, xy = mesh.geometry.attributes.psxy.array, M = globe.M, C = globe.center, e = M.elements, tmp = new THREE.Vector3();
  for (let i = 0; i < n; i++) { let hh = h[i] * v.height_scale_m; if (!isFinite(hh) || hh < -9000) hh = 0; const [la, lo] = latlonPS(xy[2 * i], xy[2 * i + 1]), r = R_MOON + h0 + (hh - h0) * exag;
    tmp.set(r * Math.cos(la) * Math.cos(lo), r * Math.cos(la) * Math.sin(lo), r * Math.sin(la)).applyMatrix4(M).add(C); pos.setXYZ(i, tmp.x, tmp.y, tmp.z); }
  pos.needsUpdate = true; mesh.geometry.computeVertexNormals(); mesh.geometry.computeBoundingSphere();
}
const moonToScene = (vP) => vP.clone().applyMatrix4(globe.M).add(globe.center);
function globeLive(m) { // lander position and ground track on the globe while in orbit / braking
  if (!globe.mesh || !m.orbit) return; const p = moonToScene(pP(m.orbit.lat_deg, m.orbit.lon_deg, R_MOON + m.orbit.alt_km * 1000)), d = camera.position.distanceTo(p);
  if (globe.track.length && m.t_s < globe.lastT) globe.track = []; globe.lastT = m.t_s;
  if (!m.lander) { globe.track.push(p); if (globe.track.length > 3000) globe.track.splice(0, 2); }
  globe.dot.visible = !m.lander; globe.dot.position.copy(p); globe.dot.scale.setScalar(Math.max(d * 0.006, 2));
  if (globe.trackLine) scene.remove(globe.trackLine); if (globe.track.length > 1) { globe.trackLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(globe.track), new THREE.LineBasicMaterial({ color: 0x8db9ff, transparent: true, opacity: 0.85 })); scene.add(globe.trackLine); }
  if (m.sun_dir_P) globe.sunP = new THREE.Vector3(...m.sun_dir_P);
}
function aimSun() { // the scene's light: the true SPICE Sun, or (readable) the Sun's azimuth raised 35 deg above the site's horizon
  if (!L) return; let dir;
  if (globe.trueSun && globe.sunP) dir = globe.sunP.clone().applyMatrix4(new THREE.Matrix4().extractRotation(globe.M)).normalize();
  else { const az = (L.sun_az_grid ? L.sun_az_grid[tIdx] : 135) * Math.PI / 180; dir = new THREE.Vector3(Math.sin(az) * Math.cos(0.61), Math.sin(0.61), -Math.cos(az) * Math.cos(0.61)); }
  sun.position.copy(dir).multiplyScalar(5e7); sun.target.position.set(0, 0, 0);
  $('sunmode').textContent = globe.trueSun ? 'True Sun (SPICE)' : 'Readable light'; $('sunmode').classList.toggle('on', globe.trueSun);
}
function fly(to) { // animate the camera between the landing site and the whole Moon
  globe.mode = to; controls.minDistance = to === 'moon' ? R_MOON * 1.08 : 40; for (const b of ['tomoon', 'topole', 'tosite']) $(b).classList.toggle('on', b === { moon: 'tomoon', pole: 'topole', site: 'tosite' }[to]);
  const from = { p: camera.position.clone(), t: controls.target.clone() }, goal = to === 'pole' ? { p: new THREE.Vector3(-0.35, 0.62, 0.7).normalize().multiplyScalar(160e3), t: new THREE.Vector3(0, -3000, 0) } : to === 'moon' ? { p: globe.center.clone().add(new THREE.Vector3(-0.9, 1.6, 2.1).normalize().multiplyScalar(R_MOON * 3.4)), t: globe.center.clone() } : { p: new THREE.Vector3(-1500, 1400, 1900), t: new THREE.Vector3() };
  const t0 = performance.now(), dur = 2600; (function step() { const u = Math.min((performance.now() - t0) / dur, 1), e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
    const lg = (a, b) => Math.exp(lerp(Math.log(a), Math.log(b), e)), da = from.p.distanceTo(from.t), db = goal.p.distanceTo(goal.t), dir = from.p.clone().sub(from.t).normalize().lerp(goal.p.clone().sub(goal.t).normalize(), e).normalize();
    controls.target.copy(from.t).lerp(goal.t, e); camera.position.copy(controls.target).addScaledVector(dir, lg(da, db)); if (u < 1) requestAnimationFrame(step); })();
}

function setHeights() {
  if (!terrainMesh) return; const p = terrainMesh.geometry.attributes.position; for (let i = 0; i < p.count; i++) p.setY(i, (L.height[i] - h0) * exag); p.needsUpdate = true; terrainMesh.geometry.computeVertexNormals();
  if (ctxMesh) { const q = ctxMesh.geometry.attributes.position, Hc = ctxMesh.geometry.userData.H; for (let i = 0; i < q.count; i++) q.setY(i, (Hc[i] - h0) * exag - 1.5); q.needsUpdate = true; ctxMesh.geometry.computeVertexNormals(); }
  placeGlobe(); placeTarget();
}
function heightAt(x, y) { const { rows, cols, mpp } = L, c = clamp(x / mpp + (cols - 1) / 2, 0, cols - 1.001), r = clamp((rows - 1) / 2 - y / mpp, 0, rows - 1.001), c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0, Hh = L.height;
  return lerp(lerp(Hh[r0 * cols + c0], Hh[r0 * cols + c0 + 1], fc), lerp(Hh[(r0 + 1) * cols + c0], Hh[(r0 + 1) * cols + c0 + 1], fc), fr); }
const toScene = (x, y, h) => new THREE.Vector3(x, (h - h0) * exag, -y);

// target ring + label
function placeTarget() {
  if (!L) return; const s = lastFrame?.site || { x_m: 0, y_m: 0 };
  if (!live.target) { const g = new THREE.Group(); const ring = new THREE.Mesh(new THREE.RingGeometry(14, 20, 64), new THREE.MeshBasicMaterial({ color: 0xffa566, side: THREE.DoubleSide, transparent: true, opacity: .95, depthTest: false }));
    ring.rotation.x = -Math.PI / 2; g.add(ring); const beam = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 300, 8, 1, true), new THREE.MeshBasicMaterial({ color: 0xffa566, transparent: true, opacity: .35, blending: THREE.AdditiveBlending, depthWrite: false })); beam.position.y = 150; g.add(beam); live.target = g; live.group.add(g); }
  live.target.position.copy(toScene(s.x_m, s.y_m, heightAt(s.x_m, s.y_m))).add(new THREE.Vector3(0, 2, 0));
}

// ---------------------------------------------------------------------------------------------------- readout under the cursor
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!terrainMesh) return; const r = renderer.domElement.getBoundingClientRect(); mouse.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObject(terrainMesh)[0]; if (!hit) return; const x = hit.point.x, y = -hit.point.z, [cx, cy] = ps(L.center_lat_deg, L.center_lon_deg), [la, lo] = latlon(cx + x, cy + y);
  const { rows, cols, mpp } = L, c = Math.round(x / mpp + (cols - 1) / 2), rr = Math.round((rows - 1) / 2 - y / mpp), i = clamp(rr, 0, rows - 1) * cols + clamp(c, 0, cols - 1);
  $('r-lat').textContent = `${la.toFixed(4)}°`; $('r-lon').textContent = `${lo.toFixed(4)}°`; $('r-h').textContent = `${fmt(L.height[i] - h0, 1)} m`; $('r-s').textContent = `${(L.slope[i] / 4).toFixed(1)}°`;
  $('r-z').textContent = (L.hazard[i] / 255).toFixed(2); const sunT = L.sun[tIdx * rows * cols + i], earthT = L.earth[tIdx * rows * cols + i]; $('r-l').textContent = `${sunT ? 'sunlit' : 'shadow'} · ${earthT ? 'Earth link' : 'no Earth'}`;
});

// ---------------------------------------------------------------------------------------------------- time of day (light layers)
function setTime(i, fromUser) { if (!L) return; tIdx = clamp(i, 0, L.times.length - 1); if (fromUser) liveTime = false; $('tslider').value = String(tIdx); $('ttime').textContent = L.times[tIdx].replace('T', ' ').replace('Z', ' UTC');
  $('tsun').textContent = `${L.sun_el[tIdx].toFixed(2)}°`; $('tearth').textContent = `${L.earth_el[tIdx].toFixed(2)}°`; $('tlive').classList.toggle('on', liveTime);
  aimSun();
  if (layer === 'sun' || layer === 'earth') paint(); lights(); }
$('tslider').oninput = (e) => setTime(Number(e.target.value), true); $('tlive').onclick = () => { liveTime = true; if (lastFrame?.utc) syncTime(lastFrame.utc); };
let playT = null; $('tplay').onclick = () => { if (playT) { clearInterval(playT); playT = null; $('tplay').textContent = '▶'; return; } liveTime = false; $('tplay').textContent = '❚❚'; playT = setInterval(() => setTime((tIdx + 1) % L.times.length, true), 400); };
function syncTime(utc) { const t = Date.parse(utc), t0 = Date.parse(L.times[0]), step = (Date.parse(L.times[1]) - t0) || 1; const i = clamp(Math.round((t - t0) / step), 0, L.times.length - 1); if (i !== tIdx) setTime(i, false); }
$('tomoon').onclick = () => fly('moon'); $('topole').onclick = () => fly('pole'); $('tosite').onclick = () => fly('site'); $('sunmode').onclick = () => { globe.trueSun = !globe.trueSun; aimSun(); };
$('exag').oninput = (e) => { exag = Number(e.target.value); $('exv').textContent = `${exag}×`; setHeights(); };

// ---------------------------------------------------------------------------------------------------- status lights (computed, never decorative)
function lights() {
  const f = lastFrame, out = [], add = (k, st, txt) => out.push(`<div class="lt ${st}" title="${txt}">${k}<b>${st === 'go' ? 'GO' : st === 'caution' ? 'CAUTION' : st === 'nogo' ? 'NO-GO' : '—'}</b></div>`);
  const pe = f?.attitude?.pointing_error_deg; add('GNC', pe == null ? '' : pe < 8 ? 'go' : pe < 20 ? 'caution' : 'nogo', `pointing error ${fmt(pe, 1)}°`);
  const fuel = f?.vehicle?.fuel_kg; add('PROP', fuel == null ? '' : fuel > 30 ? 'go' : fuel > 10 ? 'caution' : 'nogo', `${fmt(fuel, 1)} kg propellant`);
  const ok = f?.vehicle?.engines_ok; add('ENGINES', ok == null ? '' : ok === 5 ? 'go' : ok >= 3 ? 'caution' : 'nogo', `${ok ?? '—'} of 5`);
  if (L) { const s = f?.site || { x_m: 0, y_m: 0 }, { rows, cols, mpp } = L, c = Math.round(s.x_m / mpp + (cols - 1) / 2), r = Math.round((rows - 1) / 2 - s.y_m / mpp), i = clamp(r, 0, rows - 1) * cols + clamp(c, 0, cols - 1);
    add('COMMS', L.earth[tIdx * rows * cols + i] ? 'go' : 'nogo', 'Earth above the horizon at the target'); add('SUN', L.sun[tIdx * rows * cols + i] ? 'go' : 'caution', 'Sun above the horizon at the target');
    add('TERRAIN', L.safe[i] ? 'go' : 'nogo', 'target cell safe'); }
  const tr = f?.sim?.pilot_trained; add('AI PILOT', f?.sim == null ? '' : !f.sim.ai ? 'caution' : tr ? 'go' : 'caution', f?.sim ? (f.sim.ai ? (tr ? 'trained network' : 'UNTRAINED network, flies like classic') : 'off: classic guidance') : '');
  $('lights').innerHTML = out.join('');
}

// ---------------------------------------------------------------------------------------------------- right column
const NS = 'http://www.w3.org/2000/svg';
function trajectory() {
  const svg = $('trajsvg'), X0 = 60, X1 = 590, Y0 = 220, Y1 = 46, xm = (km) => X0 + (1 - Math.log10(1 + km) / Math.log10(6000)) * (X1 - X0), ym = (a) => Y0 - Math.sqrt(clamp(a, 0, 110) / 110) * (Y0 - Y1);
  let s = ''; for (const a of [0, 2, 15, 50, 100]) s += `<line x1="${X0}" x2="${X1}" y1="${ym(a)}" y2="${ym(a)}" stroke="rgba(141,185,255,.08)"/><text x="${X0 - 8}" y="${ym(a) + 4}" text-anchor="end" font-size="10.5" fill="#5d6b87" font-family="JetBrains Mono">${a} km</text>`;
  for (const k of [5000, 1000, 100, 10, 1, 0]) s += `<text x="${xm(k)}" y="${Y0 + 18}" text-anchor="middle" font-size="10.5" fill="#5d6b87" font-family="JetBrains Mono">${k ? fmt(k) : 'site'}</text>`;
  const col = { ORBIT: '#8db9ff', DEORBIT: '#2f7dff', DESCENT: '#ffa566', LANDED: '#34d399' }; let prev = null;
  for (const p of live.hist) { if (prev) s += `<line x1="${xm(prev.r).toFixed(1)}" y1="${ym(prev.a).toFixed(1)}" x2="${xm(p.r).toFixed(1)}" y2="${ym(p.a).toFixed(1)}" stroke="${col[p.ph] || '#8db9ff'}" stroke-width="2.2" stroke-linecap="round"/>`; prev = p; }
  if (prev) s += `<circle cx="${xm(prev.r)}" cy="${ym(prev.a)}" r="5" fill="#fff"/><circle cx="${xm(prev.r)}" cy="${ym(prev.a)}" r="11" fill="#fff" opacity=".18"/>`;
  s += `<text x="${X1}" y="${Y1 - 14}" text-anchor="end" font-size="11" fill="#8a99b6">orbit · deorbit · descent · landed</text>`; svg.innerHTML = s;
}
const ENG = [[0, 0], [0.8, 0], [0, 0.8], [-0.8, 0], [0, -0.8]], RCS = [[0.411, -0.974], [0.974, -0.411], [0.974, 0.411], [0.411, 0.974], [-0.411, 0.974], [-0.974, 0.411], [-0.974, -0.411], [-0.411, -0.974]], PADS = [[1.55, -1.55], [1.55, 1.55], [-1.55, 1.55], [-1.55, -1.55]];
function vehicle() {
  const f = lastFrame, sc = 46, cx = 220, cy = 140, P = (x, y) => [cx + y * sc, cy - x * sc]; // top view: body +x up the screen, +y to the right
  const th = f?.attitude?.engine_throttle || [], rc = f?.attitude?.rcs_on || [], fuel = f?.vehicle?.fuel_kg ?? 0, f0 = f?.vehicle?.fuel_start_kg || 740;
  let s = ''; for (const [x, y] of PADS) { const [a, b] = P(x, y); s += `<line x1="${cx}" y1="${cy}" x2="${a}" y2="${b}" stroke="#5d6b87" stroke-width="3"/><circle cx="${a}" cy="${b}" r="13" fill="#0c1426" stroke="#8a99b6" stroke-width="2"/>`; }
  const oct = Array.from({ length: 8 }, (_, k) => { const t = (k + .5) * Math.PI / 4, [a, b] = P(1.05 * Math.cos(t), 1.05 * Math.sin(t)); return `${a},${b}`; }).join(' ');
  s += `<polygon points="${oct}" fill="#0c1426" stroke="#8db9ff" stroke-width="2"/>`;
  ENG.forEach(([x, y], i) => { const [a, b] = P(x, y), t = th[i] || 0, dead = f && f.vehicle && i > 0 && t === 0 && (f.vehicle.engines_ok ?? 5) < 5 && [1, 3].includes(i); s += `<circle cx="${a}" cy="${b}" r="12" fill="rgba(255,165,102,${(0.12 + 0.88 * t).toFixed(2)})" stroke="${dead ? '#f43f5e' : '#ffa566'}" stroke-width="2"/><text x="${a}" y="${b + 4}" text-anchor="middle" font-size="10" fill="#02040a" font-family="JetBrains Mono">${i}</text>`; });
  RCS.forEach(([x, y], i) => { const [a, b] = P(x * 1.12, y * 1.12), on = rc[i]; s += `<rect x="${a - 5}" y="${b - 5}" width="10" height="10" rx="3" fill="${on ? '#5ee6ff' : '#14203a'}" stroke="${on ? '#5ee6ff' : '#5d6b87'}"/>`; });
  const fr = clamp(fuel / f0, 0, 1); s += `<rect x="420" y="40" width="34" height="180" rx="8" fill="#0c1426" stroke="#5d6b87"/><rect x="424" y="${44 + 172 * (1 - fr)}" width="26" height="${172 * fr}" rx="5" fill="${fr > .05 ? '#2f7dff' : '#f43f5e'}"/>`;
  s += `<text x="437" y="236" text-anchor="middle" font-size="11" fill="#8a99b6">tank</text>`;
  const lines = [['Engines', `${f?.vehicle?.engines_ok ?? '—'} of 5`], ['Throttle', th.length ? `${Math.round(Math.max(...th) * 100)} % max` : '—'], ['Attitude thrusters', rc.length ? `${rc.filter(Boolean).length} firing` : '—'], ['Pointing error', `${fmt(f?.attitude?.pointing_error_deg, 1)}°`], ['Propellant', `${fmt(fuel, 1)} kg`]];
  s += lines.map(([k, v], i) => `<text x="482" y="${70 + i * 34}" font-size="11" fill="#8a99b6">${k}</text><text x="482" y="${86 + i * 34}" font-size="14" fill="#e8eef9" font-family="JetBrains Mono">${v}</text>`).join('');
  $('vehsvg').innerHTML = s;
}
function gauges() {
  const f = lastFrame; if (!f) return; const alt = f.lander ? f.lander.alt_m : f.orbit.alt_km * 1000; $('g-alt').textContent = alt > 20000 ? fmt(alt / 1000, 1) : fmt(alt); $('g-altu').textContent = alt > 20000 ? 'km' : 'm';
  $('g-spd').textContent = fmt(f.lander ? Math.hypot(f.lander.vx_m_s, f.lander.vy_m_s, f.lander.vz_m_s) : f.orbit.speed_m_s, f.lander ? 1 : 0); $('g-fuel').textContent = fmt(f.vehicle.fuel_kg, 1); $('g-fuelb').style.width = `${clamp(f.vehicle.fuel_kg / f.vehicle.fuel_start_kg, 0, 1) * 100}%`;
  $('g-thr').textContent = fmt(f.vehicle.thrust_n); $('g-thrb').style.width = `${clamp(f.vehicle.thrust_n / 4000, 0, 1) * 100}%`;
  $('g-extra').textContent = `${fmt(f.orbit.ground_range_km, 1)} km to site` + (['ORBIT', 'DEORBIT'].includes(f.mission_phase) ? ` · periapsis ${fmt(f.orbit.periapsis_km, 1)} km` : '') + (f.vehicle.braking_tgo_s != null ? ` · ${fmt(f.vehicle.braking_tgo_s)} s to the 2 km gate` : '') + (f.lander ? ` · ${fmt(Math.hypot(f.lander.x_m - f.site.x_m, f.lander.y_m - f.site.y_m), 1)} m from target` : '');
}
function onFrame(m) {
  lastFrame = m; const h = Math.floor(m.t_s / 3600), mi = Math.floor(m.t_s % 3600 / 60), se = Math.floor(m.t_s % 60);
  $('utc').textContent = (m.utc || '—').replace('T', ' ').replace('Z', ''); $('met').textContent = `T+ ${String(h).padStart(2, '0')}:${String(mi).padStart(2, '0')}:${String(se).padStart(2, '0')}`;
  $('phase').textContent = { DOCKED: 'Orbit · docked', SEPARATED: 'Orbit · separated', DEORBIT: 'Deorbit burn', COAST: 'Coast to 15 km', BRAKING: 'Braking', TERMINAL: 'Terminal descent', LANDED: 'Landed', ENDED: 'Run ended' }[m.phase] || m.phase;
  $('warp').textContent = m.sim ? `${m.sim.playing ? '' : 'PAUSED · '}${m.sim.warp}×` : '—'; $('site').textContent = m.site ? `${m.site.region} · target ${fmt(m.site.x_m)} m E, ${fmt(m.site.y_m)} m N` : '';
  if (live.hist.length && m.t_s < live.hist[live.hist.length - 1].t) { live.hist = []; live.track = []; live.events = []; }
  live.hist.push({ t: m.t_s, r: m.orbit.ground_range_km, a: m.lander ? m.lander.alt_m / 1000 : m.orbit.alt_km, ph: m.mission_phase }); if (live.hist.length > 4000) live.hist.splice(0, 2);
  for (const e of m.events || []) { live.events.unshift(e); if (live.events.length > 12) live.events.pop(); }
  $('evlist').innerHTML = live.events.map((e) => `<div><span class="n">T+${fmt(e.t_s, 1)}</span><span>${String(e.msg).replace(/</g, '&lt;')}</span></div>`).join('');
  if (L && liveTime && m.utc) syncTime(m.utc);
  if (L && m.lander) { const p = toScene(m.lander.x_m, m.lander.y_m, heightAt(m.lander.x_m, m.lander.y_m) + m.lander.alt_m / exag * 1); p.y = (heightAt(m.lander.x_m, m.lander.y_m) - h0) * exag + m.lander.alt_m; live.track.push(p);
    if (!live.lander) { live.lander = new THREE.Mesh(new THREE.ConeGeometry(8, 22, 16), new THREE.MeshBasicMaterial({ color: 0xffffff })); live.lander.rotation.x = Math.PI; live.group.add(live.lander); }
    live.lander.position.copy(p).add(new THREE.Vector3(0, 11, 0)); if (live.trackLine) live.group.remove(live.trackLine);
    live.trackLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(live.track), new THREE.LineBasicMaterial({ color: 0xffa566 })); live.group.add(live.trackLine); }
  globeLive(m); if (globe.trueSun) aimSun(); placeTarget(); trajectory(); gauges(); vehicle(); lights();
}

// ---------------------------------------------------------------------------------------------------- boot
async function boot() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ui`);
  const gotWorld = new Promise((res) => { ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === 'world') { world = m; res(); } else if (m.type === 'frame' && m.orbit) { if (L) onFrame(m); else lastFrame = m; } }; setTimeout(res, 4000); });
  await gotWorld;
  const r = await fetch(`/api/layers?region=${encodeURIComponent(world?.region || lastFrame?.site?.region || 'lola5m_site01')}`); const J = await r.json();
  const n = J.rows * J.cols; L = { ...J, height: new Float32Array(b64(J.height_f32).buffer), slope: b64(J.slope_u8), hazard: b64(J.hazard_u8), ice: b64(J.ice_u8), safe: bits(b64(J.safe_bits), n), psr: bits(b64(J.psr_bits), n) };
  const T = J.times.length, sb = b64(J.sun_bits), eb = b64(J.earth_bits), rowBytes = Math.ceil(n / 8); L.sun = new Uint8Array(T * n); L.earth = new Uint8Array(T * n);
  for (let t = 0; t < T; t++) { L.sun.set(bits(sb.subarray(t * rowBytes, (t + 1) * rowBytes), n), t * n); L.earth.set(bits(eb.subarray(t * rowBytes, (t + 1) * rowBytes), n), t * n); }
  let mn = Infinity, mx = -Infinity; for (const v of L.height) { if (v < mn) mn = v; if (v > mx) mx = v; } L.hmin = mn; L.hmax = mx;
  const s = lastFrame?.site || { x_m: 0, y_m: 0 }; h0 = heightAt(s.x_m, s.y_m); $('tslider').max = String(T - 1);
  const photo = new Image(); photo.onload = () => { tex.photo = photo; tex.photoFilter = photoFilter(photo); paint(); }; photo.src = '/' + (world?.photo?.tile || 'assets/site01/photo_1m.jpg');
  buildTerrain(); paint(); setTime(0, false); camera.position.set(-1500, 1400, 1900); controls.target.set(0, 0, 0); controls.update(); $('wait').style.display = 'none';
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === 'frame' && m.orbit) onFrame(m); };
  if (lastFrame) onFrame(lastFrame); else { lights(); vehicle(); trajectory(); }
  const v0 = new URLSearchParams(location.search).get('view'); if (v0 === 'moon') { globe.mode = 'moon'; controls.target.copy(globe.center); camera.position.copy(globe.center).add(new THREE.Vector3(-0.9, 1.6, 2.1).normalize().multiplyScalar(R_MOON * 3.4)); controls.minDistance = R_MOON * 1.08; }
  else if (v0 === 'near') { const d = Number(new URLSearchParams(location.search).get('cam') || 12000); controls.target.set(0, 0, 0); camera.position.copy(new THREE.Vector3(-0.45, 0.55, 0.7).normalize().multiplyScalar(d)); }
  else if (v0 !== 'site') { globe.mode = 'pole'; controls.target.set(0, -3000, 0); camera.position.copy(new THREE.Vector3(-0.35, 0.62, 0.7).normalize().multiplyScalar(160e3)); } // opens on the south-pole region
  if (new URLSearchParams(location.search).get('sun') === 'true') { globe.trueSun = true; aimSun(); }
  // scale bar: metres per 120 screen px at the target distance
  setInterval(() => { const d = camera.position.distanceTo(controls.target), mPerPx = 2 * d * Math.tan(camera.fov * Math.PI / 360) / H, nice = [50, 100, 200, 500, 1000, 2000, 5000, 10000].find((v) => v / mPerPx > 70) || 10000;
    $('sbar').style.width = `${(nice / mPerPx).toFixed(0)}px`; $('stext').textContent = nice >= 1000 ? `${nice / 1000} km` : `${nice} m`; }, 300);
}
function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }
loop(); boot().catch((e) => { $('wait').textContent = 'Could not load the layers: ' + (e.message || e); console.error(e); });
window.__wall = { get L() { return L; }, setTime, paint, setLayer: (k) => { layer = k; for (const x of lw.children) x.classList.toggle('on', x.textContent === LAYERS[k].name); paint(); } };
