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
// The only colours on this screen (decided 15 Nov): main 1 #051F20, main 2 #163832 (steps #0B2B26, #235347), accent
// #8EB69B, highlight #DAF1DE, white text. Data colour scales are ramps through these six.
const PAL = { p0: '#000000', p1: '#0B2B26', p2: '#163832', p3: '#235347', acc: '#8EB69B', hi: '#DAF1DE', text: '#FFFFFF' };
const RAMP = [[5, 31, 32], [11, 43, 38], [22, 56, 50], [35, 83, 71], [142, 182, 155], [218, 241, 222]];
const rgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
function ramp(t, from = 0) { t = clamp(t, 0, 1); const R = RAMP.slice(from), x = t * (R.length - 1), i = Math.min(Math.floor(x), R.length - 2), f = x - i; return R[i].map((v, k) => lerp(v, R[i + 1][k], f)); }
// Data colours (statistical data and graphs; never green): validated dark-mode set on #000, fixed order.
const DATA = ['#3987e5', '#d95926', '#9085e9', '#c98500', '#d55181'];
const BLUE = [[13, 54, 107], [16, 66, 129], [28, 92, 171], [42, 120, 214], [57, 135, 229], [109, 167, 236], [158, 197, 244], [205, 226, 251]]; // magnitude: dark -> light
function blueRamp(t) { t = clamp(t, 0, 1); const x = t * (BLUE.length - 1), i = Math.min(Math.floor(x), BLUE.length - 2), f = x - i; return BLUE[i].map((v, k) => lerp(v, BLUE[i + 1][k], f)); }
const blueCss = 'linear-gradient(90deg,' + BLUE.map((c) => `rgb(${c})`).join(',') + ')';
const rampCss = (from = 0) => 'linear-gradient(90deg,' + RAMP.slice(from).map((c) => `rgb(${c})`).join(',') + ')';
const ps = (latDeg, lonDeg) => { const k = 2 * R_MOON * Math.tan(Math.PI / 4 + latDeg * Math.PI / 360), lo = lonDeg * Math.PI / 180; return [k * Math.sin(lo), k * Math.cos(lo)]; };
const latlon = (x, y) => [(2 * Math.atan(Math.hypot(x, y) / (2 * R_MOON)) - Math.PI / 2) * 180 / Math.PI, Math.atan2(x, y) * 180 / Math.PI];

// ---------------------------------------------------------------------------------------------------- 3D scene
const host = $('terrain'), W = 1260, H = 900;
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true }); renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(W, H); host.prepend(renderer.domElement);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); scene.background = new THREE.Color(PAL.p0);
const camera = new THREE.PerspectiveCamera(38, W / H, 0.5, 1e8);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.minDistance = 40; controls.maxDistance = 9e6;
const sun = new THREE.DirectionalLight(0xffffff, 2.6); scene.add(sun, sun.target); scene.add(new THREE.HemisphereLight(0xffffff, PAL.p0, 0.75)); // neutral light: colour comes from the data and the palette

let L = null, world = null, terrainMesh = null, ctxMesh = null, h0 = 0, exag = 1.5, layer = new URLSearchParams(location.search).get('layer') || 'imagery', tIdx = 0, liveTime = true, lastFrame = null;
const tex = { canvas: document.createElement('canvas'), photo: null }; tex.canvas.width = tex.canvas.height = 1536; const ctex = new THREE.CanvasTexture(tex.canvas); ctex.colorSpace = THREE.SRGBColorSpace; ctex.anisotropy = 8;
const live = { group: new THREE.Group(), track: [], trackLine: null, lander: null, target: null, hist: [], events: [] }; scene.add(live.group);

const LAYERS = {
  imagery: { name: 'Imagery', title: 'LROC NAC photo mosaic', a: '', z: '', bar: `linear-gradient(90deg,${PAL.p0},${PAL.hi})`, prov: 'OBSERVED · NASA LRO LROC NAC, 1 m/px (averaged lighting)' },
  elevation: { name: 'Elevation', title: 'Height (LOLA)', bar: blueCss, prov: 'OBSERVED · NASA LRO LOLA 5 m' },
  slope: { name: 'Slope', title: 'Slope · limit 12°', a: '0°', z: '30°', bar: blueCss, prov: 'derived from LOLA 5 m' },
  hazard: { name: 'Hazard (AI)', title: 'Hazard probability', a: 'safe 0', z: '1 unsafe', bar: 'linear-gradient(90deg,rgba(217,89,38,0),#d95926)', prov: '' },
  safe: { name: 'Safe cells', title: 'Where a lander may aim', a: '', z: '', bar: 'linear-gradient(90deg,#3987e5,#3987e5)', prov: 'no hazard within 15 m and slope < 6° within 10 m' },
  sun: { name: 'Sunlight', title: 'Sun above the local horizon', a: 'shadow', z: 'sunlit', bar: 'linear-gradient(90deg,#000000,#c98500)', prov: 'MODELLED · SPICE DE440 + LOLA horizon' },
  earth: { name: 'Earth link', title: 'Earth above the local horizon', a: 'no link', z: 'direct link', bar: 'linear-gradient(90deg,#000000,#3987e5)', prov: 'MODELLED · SPICE DE440 + LOLA horizon' },
  psr: { name: 'Dark craters', title: 'Permanently shadowed regions', a: '', z: '', bar: 'linear-gradient(90deg,#9085e9,#9085e9)', prov: 'MODELLED · horizon test over the lunar year' },
  ice: { name: 'Ice evidence', title: 'Ice EVIDENCE score (not confirmed ice)', a: '0', z: '1', bar: blueCss, prov: 'MODELLED · PSR + cold-trap + neutron evidence' },
};
const lw = $('layers'); for (const [k, v] of Object.entries(LAYERS)) { const b = document.createElement('button'); b.className = 'lb' + (k === layer ? ' on' : ''); b.textContent = v.name; b.onclick = () => chooseLayer(k); lw.appendChild(b); }

function paint() { // compose photo + the active layer into the terrain texture
  if (!L) return; const g = tex.canvas.getContext('2d'), N = tex.canvas.width; g.globalAlpha = 1; g.fillStyle = PAL.p2; g.fillRect(0, 0, N, N);
  if (tex.photo) { g.filter = tex.photoFilter || 'none'; g.drawImage(tex.photo, 0, 0, N, N); g.filter = 'none'; }
  const { rows, cols } = L, im = new ImageData(cols, rows), d = im.data, lay = layer;
  const hmin = L.hmin, hmax = L.hmax, sunT = L.sun ? L.sun.subarray(tIdx * rows * cols, (tIdx + 1) * rows * cols) : null, earthT = L.earth ? L.earth.subarray(tIdx * rows * cols, (tIdx + 1) * rows * cols) : null;
  let cov = 0; // cells the layer marks, for the legend's coverage line
  for (let i = 0; i < rows * cols; i++) { let c = null, a = 0;
    if (lay === 'elevation') { c = blueRamp((L.height[i] - hmin) / (hmax - hmin)); a = 0.85; }
    else if (lay === 'slope') { c = blueRamp(L.slope[i] / 4 / 30); a = 0.82; }
    else if (lay === 'hazard') { const p = L.hazard[i] / 255; c = [217, 89, 38]; a = p * 0.62; cov += p > 0.5; }
    else if (lay === 'safe') { if (L.safe[i]) { c = [57, 135, 229]; a = 0.75; cov++; } }
    else if (lay === 'sun') { c = sunT[i] ? [201, 133, 0] : [0, 0, 0]; a = sunT[i] ? 0.45 : 0.42; cov += sunT[i]; } // shadow only dims, so the ground stays readable
    else if (lay === 'earth') { c = earthT[i] ? [57, 135, 229] : [0, 0, 0]; a = earthT[i] ? 0.6 : 0.38; cov += earthT[i]; }
    else if (lay === 'psr') { if (L.psr[i]) { c = [144, 133, 233]; a = 0.85; cov++; } }
    else if (lay === 'ice') { const sc = L.ice[i] / 255; if (sc > 0.12) { c = blueRamp(0.4 + 0.6 * sc); a = 0.55 + 0.4 * sc; cov++; } }
    if (c) { d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = a * 255; } }
  const tmp = document.createElement('canvas'); tmp.width = cols; tmp.height = rows; tmp.getContext('2d').putImageData(im, 0, 0); g.imageSmoothingEnabled = lay !== 'safe' && lay !== 'psr'; g.drawImage(tmp, 0, 0, N, N);
  ctex.needsUpdate = true; legend(cov / (rows * cols));
}
const COVER = { hazard: 'Unsafe (above 0.5)', safe: 'Safe cells', sun: 'Sunlit', earth: 'Earth in view', psr: 'Permanently shadowed', ice: 'Ice evidence above 0.12' };
function legend(frac = null) {
  const v = LAYERS[layer], cv = $('lg-c'); if (cv) cv.innerHTML = frac == null || !COVER[layer] ? '' : frac === 0 ? `${COVER[layer]}: none on this tile${layer === 'sun' || layer === 'earth' ? ' at this time' : ''}` : `${COVER[layer]}: <span class="n">${(frac * 100).toFixed(frac < 0.01 ? 2 : 1)} %</span> of the tile`; $('lg-t').textContent = v.title; $('lg-b').style.background = v.bar;
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
    ctxMesh = new THREE.Mesh(cg, new THREE.MeshStandardMaterial({ map: ct, roughness: 1, color: new URLSearchParams(location.search).get('debug') === 'levels' ? 0xffffff : 0xffffff })); scene.add(ctxMesh); setHeights(); }
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
  globe.ring = new THREE.Mesh(new THREE.RingGeometry(9000, 13000, 64), new THREE.MeshBasicMaterial({ color: PAL.acc, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false }));
  globe.ring.rotation.x = -Math.PI / 2; globe.ring.renderOrder = 5; scene.add(globe.ring);
  globe.dot = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshBasicMaterial({ color: PAL.hi, depthTest: false })); globe.dot.renderOrder = 6; globe.dot.visible = false; scene.add(globe.dot);
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
    const DBG = new URLSearchParams(location.search).get('debug') === 'levels', mat = new THREE.MeshStandardMaterial({ map: wac, roughness: 1, metalness: 0, color: DBG ? { cap: PAL.p3, region: PAL.acc, local: PAL.hi }[v.name] : 0xffffff });
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
  if (globe.trackLine) scene.remove(globe.trackLine); if (globe.track.length > 1) { globe.trackLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(globe.track), new THREE.LineBasicMaterial({ color: PAL.hi, transparent: true, opacity: 0.85 })); scene.add(globe.trackLine); }
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
  if (!live.target) { const g = new THREE.Group(); const ring = new THREE.Mesh(new THREE.RingGeometry(14, 20, 64), new THREE.MeshBasicMaterial({ color: PAL.acc, side: THREE.DoubleSide, transparent: true, opacity: .95, depthTest: false }));
    ring.rotation.x = -Math.PI / 2; g.add(ring); const beam = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 300, 8, 1, true), new THREE.MeshBasicMaterial({ color: PAL.acc, transparent: true, opacity: .35, blending: THREE.AdditiveBlending, depthWrite: false })); beam.position.y = 150; g.add(beam); live.target = g; live.group.add(g); }
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
  if (!$('lights')) return; // the shared company banner (/banner.js) shows the lights on every screen
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
function trajectory() { // OpenUI charts, refreshed at most every 1.5 s (OpenUI animates each change)
  const now = performance.now(); if (now - (trajectory.t || 0) < 1500 || !window.OUI || !live.hist.length) return; trajectory.t = now;
  const H = live.hist, n = Math.min(24, H.length), pick = Array.from({ length: n }, (_, k) => H[Math.round(k * (H.length - 1) / Math.max(n - 1, 1))]);
  const hm = (t) => `${String(Math.floor(t / 3600))}:${String(Math.floor(t % 3600 / 60)).padStart(2, '0')}`;
  const hh = Math.max(80, ($('oc-alt').clientHeight || 170) - 64); // room for the x-axis title and legend
  OUI.render($('oc-alt'), `root = Card([c])\nc = AreaChart(${OUI.strs(pick.map((p) => hm(p.t)))}, [a], "natural", "mission time (h:min)", "km", ${hh})\na = Series("Altitude km", ${OUI.arr(pick.map((p) => p.a))})`);
  const groups = { Orbit: ['DOCKED', 'SEPARATED'], Deorbit: ['DEORBIT'], Coast: ['COAST'], Braking: ['BRAKING'], Descent: ['TERMINAL', 'LANDED', 'ENDED'] }, used = Object.fromEntries(Object.keys(groups).map((k) => [k, 0]));
  for (let k = 1; k < H.length; k++) { const d = (H[k - 1].fuel ?? 0) - (H[k].fuel ?? 0); if (d > 0) { const g = Object.keys(groups).find((G) => groups[G].includes(H[k].phase)); if (g) used[g] += d; } }
  const cats = Object.keys(used).filter((k) => used[k] > 0.05);
  OUI.render($('oc-prop'), cats.length ? `root = Card([c])\nc = PieChart(${OUI.strs(cats)}, ${OUI.arr(cats.map((k) => used[k]))}, "donut")` : 'root = Card([t])\nt = TextContent("No propellant used yet in this view.", "small")');
}
const ENG = [[0, 0], [0.8, 0], [0, 0.8], [-0.8, 0], [0, -0.8]], RCS = [[0.411, -0.974], [0.974, -0.411], [0.974, 0.411], [0.411, 0.974], [-0.411, 0.974], [-0.974, 0.411], [-0.974, -0.411], [-0.411, -0.974]], PADS = [[1.55, -1.55], [1.55, 1.55], [-1.55, 1.55], [-1.55, -1.55]];
function vehicle() {
  const f = lastFrame, sc = 46, cx = 220, cy = 140, P = (x, y) => [cx + y * sc, cy - x * sc]; // top view: body +x up the screen, +y to the right
  const th = f?.attitude?.engine_throttle || [], rc = f?.attitude?.rcs_on || [], fuel = f?.vehicle?.fuel_kg ?? 0, f0 = f?.vehicle?.fuel_start_kg || 740;
  let s = ''; for (const [x, y] of PADS) { const [a, b] = P(x, y); s += `<line x1="${cx}" y1="${cy}" x2="${a}" y2="${b}" stroke="${PAL.p3}" stroke-width="3"/><circle cx="${a}" cy="${b}" r="13" fill="${PAL.p0}" stroke="${PAL.acc}" stroke-width="2"/>`; }
  const oct = Array.from({ length: 8 }, (_, k) => { const t = (k + .5) * Math.PI / 4, [a, b] = P(1.05 * Math.cos(t), 1.05 * Math.sin(t)); return `${a},${b}`; }).join(' ');
  s += `<polygon points="${oct}" fill="${PAL.p0}" stroke="${PAL.acc}" stroke-width="2"/>`;
  ENG.forEach(([x, y], i) => { const [a, b] = P(x, y), t = th[i] || 0, dead = f && f.vehicle && i > 0 && t === 0 && (f.vehicle.engines_ok ?? 5) < 5 && [1, 3].includes(i); s += `<circle cx="${a}" cy="${b}" r="12" fill="${DATA[1]}" fill-opacity="${(0.08 + 0.92 * t).toFixed(2)}" stroke="${dead ? PAL.hi : DATA[1]}" stroke-width="2" ${dead ? 'stroke-dasharray="3 3"' : ''}/><text x="${a}" y="${b + 4}" text-anchor="middle" font-size="10" fill="${PAL.p0}" font-family="JetBrains Mono">${i}</text>`; });
  RCS.forEach(([x, y], i) => { const [a, b] = P(x * 1.12, y * 1.12), on = rc[i]; s += `<rect x="${a - 5}" y="${b - 5}" width="10" height="10" rx="3" fill="${on ? DATA[3] : PAL.p1}" stroke="${on ? DATA[3] : PAL.p3}"/>`; });
  const fr = clamp(fuel / f0, 0, 1); s += `<rect x="420" y="40" width="34" height="180" rx="8" fill="${PAL.p0}" stroke="${PAL.p3}"/><rect x="424" y="${44 + 172 * (1 - fr)}" width="26" height="${172 * fr}" rx="5" fill="${fr > .05 ? DATA[0] : DATA[1]}"/>`;
  s += `<text x="437" y="236" text-anchor="middle" font-size="11" fill="rgba(255,255,255,.62)">tank</text>`;
  const lines = [['Engines', `${f?.vehicle?.engines_ok ?? '—'} of 5`], ['Throttle', th.length ? `${Math.round(Math.max(...th) * 100)} % max` : '—'], ['Attitude thrusters', rc.length ? `${rc.filter(Boolean).length} firing` : '—'], ['Pointing error', `${fmt(f?.attitude?.pointing_error_deg, 1)}°`], ['Propellant', `${fmt(fuel, 1)} kg`]];
  s += lines.map(([k, v], i) => `<text x="482" y="${68 + i * 36}" font-size="13" fill="rgba(255,255,255,.62)">${k}</text><text x="482" y="${86 + i * 36}" font-size="17" fill="#FFFFFF" ${/\d/.test(v) ? 'font-family="JetBrains Mono"' : ''}>${v}</text>`).join('');
  $('vehsvg').innerHTML = s;
}
// ---------------------------------------------------------------------------------------------------- ISRO-style flight gauges
const tween = new Map(); // id -> {from, to, t0, d}: numbers roll to their new value
function roll(id, value, d = 0) { const el = $(id); if (!el) return; if (value == null || !isFinite(value)) { el.textContent = '—'; tween.delete(id); return; }
  const cur = tween.get(id), now = performance.now(), from = cur ? cur.val : value; tween.set(id, { from, to: value, t0: now, d, val: from, el }); }
function rollStep() { const now = performance.now(); for (const [id, t] of tween) { const u = Math.min((now - t.t0) / 500, 1), e = 1 - Math.pow(1 - u, 3); t.val = lerp(t.from, t.to, e); t.el.textContent = fmt(t.val, t.d); if (u >= 1) t.from = t.to; } requestAnimationFrame(rollStep); }
requestAnimationFrame(rollStep);
function dial(cx, cy, r, frac, label, unit, id) { // 240 deg dial: track, value arc, ticks and an eased needle
  const a0 = 150, sweep = 240, ang = (f) => (a0 + sweep * clamp(f, 0, 1)) * Math.PI / 180, pt = (f, rr) => [cx + rr * Math.cos(ang(f)), cy + rr * Math.sin(ang(f))];
  const arc = (f, rr) => { const [x0, y0] = pt(0, rr), [x1, y1] = pt(f, rr); return `M${x0},${y0} A${rr},${rr} 0 ${sweep * f > 180 ? 1 : 0} 1 ${x1},${y1}`; };
  let s = `<path d="${arc(1, r)}" fill="none" stroke="${PAL.p2}" stroke-width="9" stroke-linecap="round"/><path d="${arc(Math.max(clamp(frac, 0, 1), 0.001), r)}" fill="none" stroke="${DATA[0]}" stroke-width="9" stroke-linecap="round" style="filter:drop-shadow(0 0 5px rgba(57,135,229,.55))"/>`;
  for (let k = 0; k <= 10; k++) { const [x0, y0] = pt(k / 10, r - 14), [x1, y1] = pt(k / 10, r - (k % 5 ? 19 : 23)); s += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="${PAL.p3}" stroke-width="${k % 5 ? 1.2 : 2}"/>`; }
  const deg = a0 + sweep * clamp(frac, 0, 1);
  s += `<g class="dial-needle" style="transform-origin:${cx}px ${cy}px;transform:rotate(${deg}deg)"><line x1="${cx}" y1="${cy}" x2="${cx + r - 20}" y2="${cy}" stroke="${PAL.hi}" stroke-width="2.4" stroke-linecap="round"/></g><circle cx="${cx}" cy="${cy}" r="5" fill="${PAL.hi}"/>`;
  s += `<text x="${cx}" y="${cy + r * 0.62}" text-anchor="middle" font-size="22" font-family="JetBrains Mono" id="${id}">—</text><text x="${cx}" y="${cy + r * 0.62 + 17}" text-anchor="middle" font-size="10.5" fill="rgba(255,255,255,.62)" letter-spacing="1.5">${label} · ${unit}</text>`;
  return s;
}
function leds(id, frac, warnBelow) { const el = $(id); if (!el) return; if (!el.children.length) el.innerHTML = '<b></b>'.repeat(20); const n = Math.round(clamp(frac, 0, 1) * 20);
  [...el.children].forEach((b, k) => { b.className = k < n ? (warnBelow != null && frac < warnBelow ? 'warn' : 'on') : ''; }); }
function gauges() {
  const f = lastFrame; if (!f) return; const altM = f.lander ? f.lander.alt_m : f.orbit.alt_km * 1000, spd = f.lander ? Math.hypot(f.lander.vx_m_s, f.lander.vy_m_s, f.lander.vz_m_s) : f.orbit.speed_m_s;
  const km = altM > 2000, altFrac = Math.log10(1 + Math.max(altM, 0)) / Math.log10(1 + 110e3); // log scale: 0 m .. 110 km
  if (!gauges.built || gauges.km !== km) { $('dials').innerHTML = dial(78, 92, 62, altFrac, 'ALTITUDE', km ? 'km' : 'm', 'd-alt') + dial(226, 92, 62, spd / 1800, 'SPEED', 'm/s', 'd-spd'); gauges.built = true; gauges.km = km; tween.delete('d-alt'); tween.delete('d-spd'); }
  const needles = $('dials').querySelectorAll('.dial-needle'), arcs = $('dials').querySelectorAll('path:nth-of-type(2n)');
  const setDial = (k, frac) => { const deg = 150 + 240 * clamp(frac, 0, 1); needles[k].style.transform = `rotate(${deg}deg)`; const cx = k ? 226 : 78, cy = 92, r = 62, a = (v) => (150 + 240 * v) * Math.PI / 180, x0 = cx + r * Math.cos(a(0)), y0 = cy + r * Math.sin(a(0)), fr = Math.max(clamp(frac, 0, 1), 0.001), x1 = cx + r * Math.cos(a(fr)), y1 = cy + r * Math.sin(a(fr)); arcs[k].setAttribute('d', `M${x0},${y0} A${r},${r} 0 ${240 * fr > 180 ? 1 : 0} 1 ${x1},${y1}`); };
  setDial(0, altFrac); setDial(1, spd / 1800);
  roll('d-alt', km ? altM / 1000 : altM, km ? 1 : 0); roll('d-spd', spd, f.lander ? 1 : 0);
  roll('g-fuel', f.vehicle.fuel_kg, 1); leds('led-fuel', f.vehicle.fuel_kg / f.vehicle.fuel_start_kg, 0.05);
  roll('g-thr', f.vehicle.thrust_n, 0); leds('led-thr', f.vehicle.thrust_n / 4000);
  roll('t-range', f.orbit.ground_range_km, 1); roll('t-gate', f.vehicle.braking_tgo_s, 0);
  roll('t-miss', f.lander ? Math.hypot(f.lander.x_m - f.site.x_m, f.lander.y_m - f.site.y_m) : null, 1); roll('t-peri', ['ORBIT', 'DEORBIT'].includes(f.mission_phase) ? f.orbit.periapsis_km : null, 1);
}
function onFrame(m) {
  lastFrame = m; const h = Math.floor(m.t_s / 3600), mi = Math.floor(m.t_s % 3600 / 60), se = Math.floor(m.t_s % 60);
  void h; void mi; void se; // clocks are in the banner
  $('site').textContent = m.site ? `${m.site.region} · target ${fmt(m.site.x_m)} m E, ${fmt(m.site.y_m)} m N` : '';
  if (live.hist.length && m.t_s < live.hist[live.hist.length - 1].t) { live.hist = []; live.track = []; live.events = []; }
  live.hist.push({ t: m.t_s, r: m.orbit.ground_range_km, a: m.lander ? m.lander.alt_m / 1000 : m.orbit.alt_km, ph: m.mission_phase, phase: m.phase, fuel: m.vehicle.fuel_kg }); if (live.hist.length > 4000) live.hist.splice(0, 2);
  for (const e of m.events || []) { e._fresh = true; live.events.unshift(e); if (live.events.length > 12) live.events.pop(); }
  if ((m.events || []).length) { $('evlist').innerHTML = live.events.map((e) => `<div class="${e._fresh ? 'fresh' : ''}"><span class="n">T+${fmt(e.t_s, 1)}</span><span>${String(e.msg).replace(/</g, '&lt;')}</span></div>`).join(''); for (const e of live.events) e._fresh = false; }
  if (L && liveTime && m.utc) syncTime(m.utc);
  if (L && m.lander) { const p = toScene(m.lander.x_m, m.lander.y_m, heightAt(m.lander.x_m, m.lander.y_m) + m.lander.alt_m / exag * 1); p.y = (heightAt(m.lander.x_m, m.lander.y_m) - h0) * exag + m.lander.alt_m; live.track.push(p);
    if (!live.lander) { live.lander = new THREE.Mesh(new THREE.ConeGeometry(8, 22, 16), new THREE.MeshBasicMaterial({ color: PAL.hi })); live.lander.rotation.x = Math.PI; live.group.add(live.lander); }
    live.lander.position.copy(p).add(new THREE.Vector3(0, 11, 0)); if (live.trackLine) live.group.remove(live.trackLine);
    live.trackLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(live.track), new THREE.LineBasicMaterial({ color: PAL.hi })); live.group.add(live.trackLine); }
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
// Data layers exist only on the landing tile (a few km), which is a dot from the pole or Moon views: picking one flies to the site.
function chooseLayer(k) { layer = k; for (const x of lw.children) x.classList.toggle('on', x.textContent === LAYERS[k].name); paint(); if (k !== 'imagery' && globe.mode !== 'site') fly('site'); }
window.__wall = { get L() { return L; }, setTime, paint, setLayer: chooseLayer };
