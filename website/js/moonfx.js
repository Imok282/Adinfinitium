// Moon effects for the home hero (agent F). Plug-in elements the hero sequences inside ITS scene and loop.
//
//   import { createMoonFX } from './moonfx.js';
//   const fx = createMoonFX(THREE, scene, moonMesh, { sunDir, radius: 30 });
//   fx.scan.update(p)        // p 0..1 = that element's segment progress (each element fades in/out at its edges)
//   fx.scan.group            // THREE.Object3D (already added: surface elements as children of moonMesh, the rest to scene)
//   fx.dispose()             // or fx.scan.dispose()
//   fx.orientToSite(lat, lon, worldDir)   // quaternion for moonMesh that turns that site towards worldDir (south-pole sites
//                                         // sit at the bottom of an un-rotated sphere, so the hero usually wants this)
// Elements (each a different kind of motion):
//   separation  propulsion module + lander parting and drifting (tumble <= 5 deg/s at a 6 s segment)
//   relay       small relay satellite crossing the sky on an arc
//   scan        hazard-scan beam sweeping a surface patch at Site 23 and painting the REAL hazard map behind it
//               (assets/models3d/b_hazard_tex.jpg: U-Net P(unsafe) for held-out Site 23, 5.12 km tile, MODELLED)
//   terminator  sunlight terminator sweeping across the Moon (shader shell, rotating sun direction uniform)
//   pins        landing-site pins popping up one by one at the REAL tile-centre lat/lon of LOLA Sites 01 04 06 07 11 20 23
//               (data/tiles/lola5m_siteNN.npz header center_lat_deg / center_lon_deg)
//   swarm       tiny landers raining onto Site 06: green landed / red failed in the REAL proportion of
//               assets/data/ghosts_lander_site06.json (629 ok of 960 incl. open-loop contrast runs -> 63 of 96 drawn; opts.swarm = "closed": 63 of 64)
//   rover       tiny rover driving the REAL route (assets/data/rover_route_site23.json) as a dotted trail, enlarged around the lander
//   packets     data packets travelling rover -> lander -> Earth along a thin line
// Surface patches are drawn at an exaggerated size (patchDeg, default 10 deg across for the 5.12 km tile) so they are
// visible from orbit; positions inside the patch are true to the tile. No per-frame allocation; small geometry.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const SITES = [   // tile-centre lat/lon, data/tiles/lola5m_siteNN.npz headers
  ['01', -89.4632, 222.5104], ['04', -89.7668, 188.1301], ['06', -85.4381, 37.3667], ['07', -88.811, 123.6901],
  ['11', -88.6834, 292.0679], ['20', -85.4266, 31.7428], ['23', -85.9948, 359.7642],
];
const SITE23 = [-85.9948, 359.7642], SITE06 = [-85.4381, 37.3667];
// ghosts_lander_site06.json, drawn 1:10. counts_total = 629 ok of 960 INCLUDING 320 open-loop contrast runs (all fail);
// classic 314/320 + pilot 315/320 = 629 of 640 closed-loop. opts.swarm = "closed" draws only the closed-loop runs (63 of 64).
const SWARM_OK = 63, SWARM_N_ALL = 96, SWARM_N_CLOSED = 64;
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const edge = p => smooth(0, 0.1, p) * (1 - smooth(0.9, 1, p));

export function createMoonFX(THREE, scene, moon, opts = {}) {
  const R = opts.radius || (moon.geometry && moon.geometry.parameters && moon.geometry.parameters.radius) || 1;
  const PATCH = THREE.MathUtils.degToRad(opts.patchDeg || 10);
  const sunDir = (opts.sunDir && opts.sunDir.clone ? opts.sunDir.clone() : new THREE.Vector3(1, 0.16, 0.35)).normalize();
  const base = (opts.base || '').replace(/\/?$/, opts.base ? '/' : '');
  const tl = new THREE.TextureLoader();
  const els = {};

  // ---------- geometry helpers (moon-local, SphereGeometry uv convention: u = 0.5 at lon 0) ----------
  function siteDir(lat, lon, out = new THREE.Vector3()) {
    const th = THREE.MathUtils.degToRad(90 - lat), ph = Math.PI + THREE.MathUtils.degToRad(lon);
    return out.set(-Math.cos(ph) * Math.sin(th), Math.cos(th), Math.sin(ph) * Math.sin(th)).normalize();
  }
  function frame(lat, lon) {   // n up, e east, nn north (moon-local unit vectors)
    const n = siteDir(lat, lon), e = new THREE.Vector3(0, 1, 0).cross(n).normalize(), nn = new THREE.Vector3().crossVectors(n, e).normalize();
    return { n, e, nn };
  }
  // local patch point: (x east, z south) in units of the patch half-size (-1..1) -> moon-local position at height h (x R)
  function patchPoint(F, x, z, h, out) {
    out.copy(F.n).addScaledVector(F.e, x * PATCH / 2).addScaledVector(F.nn, -z * PATCH / 2).normalize();
    return out.multiplyScalar(R * (1 + h));
  }
  function patchMesh(F, mat, segs = 24, h = 0.0015) {
    const g = new THREE.PlaneGeometry(1, 1, segs, segs), P = g.attributes.position, v = new THREE.Vector3();
    for (let i = 0; i < P.count; i++) { patchPoint(F, P.getX(i) * 2, -P.getY(i) * 2, h, v); P.setXYZ(i, v.x, v.y, v.z); }
    g.computeVertexNormals();
    return new THREE.Mesh(g, mat);
  }
  const softDot = (() => {
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const c = cv.getContext('2d'), gr = c.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.45, 'rgba(255,255,255,.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    c.fillStyle = gr; c.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(cv);
  })();
  const disposables = [];
  const track = o => { o.traverse(x => { if (x.geometry) disposables.push(x.geometry); if (x.material) [].concat(x.material).forEach(m => disposables.push(m)); }); return o; };
  function element(name, group, parent, update) {
    group.name = 'moonfx-' + name; group.visible = false; parent.add(group); track(group);
    els[name] = { group, update(p) { const k = edge(clamp(p)); group.visible = p > 0 && p < 1 && k > 0.001; if (group.visible) update(clamp(p), k); }, dispose() { group.removeFromParent(); } };
  }
  const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3(), mtx = new THREE.Matrix4(), q = new THREE.Quaternion(), qT = new THREE.Quaternion(), sc = new THREE.Vector3();
  const F23 = frame(...SITE23), F06 = frame(...SITE06);

  // ---------- separation (scene space, above the limb; GLB models, ~3 m drawn at 0.05 R) ----------
  {
    const g = new THREE.Group(), lander = new THREE.Group(), pm = new THREE.Group(), mats = [];
    g.add(lander, pm);
    const gl = new GLTFLoader(), S = R * 0.05;
    const fit = (url, holder) => gl.load(base + url, m => {
      const o = m.scene, b = new THREE.Box3().setFromObject(o), c = b.getCenter(new THREE.Vector3()), h = b.max.y - b.min.y || 1;
      o.position.sub(c); const w = new THREE.Group(); w.add(o); w.scale.setScalar(S / h);
      o.traverse(x => { if (x.isMesh) { x.material = x.material.clone(); x.material.transparent = true; mats.push(x.material); disposables.push(x.material); } });
      holder.add(w);
    }, undefined, () => {});
    fit('assets/models/lander.glb', lander); fit('assets/models/propulsion_module.glb', pm);
    const anchor = new THREE.Vector3();
    element('separation', g, scene, (p, k) => {
      moon.updateMatrixWorld(); patchPoint(F23, -0.2, 0.2, 0.09, anchor); moon.localToWorld(anchor);
      g.position.copy(anchor);
      const s = smooth(0.15, 1, p);                                  // parting distance
      lander.position.set(-0.4 * S * s, 0.55 * S + 0.6 * S * s, 0); lander.rotation.set(0.04 * s, 0.3 * s, -0.05 * s);
      pm.position.set(0.5 * S * s, -0.55 * S - 0.4 * S * s, 0.2 * S * s); pm.rotation.set(0.35 * s, -0.2 * s, 0.5 * s);   // <= ~30 deg over the segment
      for (let i = 0; i < mats.length; i++) mats[i].opacity = k;
    });
  }

  // ---------- relay satellite (moon-local arc above the Site 23 region, so it is in frame wherever the hero points the site) ----------
  {
    const g = new THREE.Group(), mat = new THREE.MeshStandardMaterial({ color: 0xd9d4c7, roughness: 0.5, metalness: 0.4, transparent: true });
    const pan = new THREE.MeshStandardMaterial({ color: 0x2b3a66, roughness: 0.35, metalness: 0.6, emissive: 0x0b1020, transparent: true, side: THREE.DoubleSide });
    const S = R * 0.012, bus = new THREE.Mesh(new THREE.BoxGeometry(S, S, S), mat);
    const pg = new THREE.PlaneGeometry(S * 2.6, S * 0.9), p1 = new THREE.Mesh(pg, pan), p2 = new THREE.Mesh(pg, pan);
    p1.position.x = S * 1.9; p2.position.x = -S * 1.9;
    const dish = new THREE.Mesh(new THREE.ConeGeometry(S * 0.5, S * 0.4, 12, 1, true), mat); dish.position.y = S * 0.7; dish.rotation.x = Math.PI;
    g.add(bus, p1, p2, dish);
    // moon-local arc high above the Site 23 region (the part of the Moon the hero frames), west -> east
    element('relay', g, moon, (p, k) => {
      const x = THREE.MathUtils.lerp(-3.2, 3.2, p);
      patchPoint(F23, x, -1.2 + 0.25 * x * x / 3.2, 0.22 + 0.06 * Math.sin(Math.PI * p), g.position);
      q.setFromUnitVectors(tmp.set(0, 1, 0), tmp2.copy(g.position).normalize()); g.quaternion.copy(q);
      g.rotateY(0.6 + 0.15 * Math.sin(p * 6));
      mat.opacity = pan.opacity = k;
    });
  }

  // ---------- hazard scan: beam sweeps the Site 23 patch, the real hazard map fades in behind it ----------
  {
    const g = new THREE.Group();
    const tex = tl.load(base + 'assets/models3d/b_hazard_tex.jpg', t => { t.colorSpace = THREE.SRGBColorSpace; });
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
      uniforms: { map: { value: tex }, uReveal: { value: 0 }, uFade: { value: 1 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: 'uniform sampler2D map; uniform float uReveal, uFade; varying vec2 vUv; void main(){ float y = 1.0 - vUv.y; float a = smoothstep(uReveal, uReveal - 0.06, y); vec3 c = texture2D(map, vUv).rgb; float edge = smoothstep(0.0, 0.04, min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y))); float line = exp(-abs(y - uReveal) * 90.0) * step(y, uReveal + 0.02); gl_FragColor = vec4(mix(c, vec3(1.0, 0.78, 0.6), line), (0.85 * a * edge + line * 0.9) * uFade); }',
    });
    const patch = patchMesh(F23, mat); g.add(patch);
    const bg = new THREE.BufferGeometry(); bg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
    const bm = new THREE.MeshBasicMaterial({ color: 0xffc9a0, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending });
    const beam = new THREE.Mesh(bg, bm); beam.frustumCulled = false; g.add(beam);
    const apex = new THREE.Vector3(); patchPoint(F23, 0, 1.6, 0.3, apex);       // scanner in orbit, south of the patch
    element('scan', g, moon, (p, k) => {
      const r = smooth(0.05, 0.85, p);
      mat.uniforms.uReveal.value = r; mat.uniforms.uFade.value = k;
      const P = bg.attributes.position, z = r * 2 - 1;
      P.setXYZ(0, apex.x, apex.y, apex.z);
      patchPoint(F23, -1, z, 0.002, tmp); P.setXYZ(1, tmp.x, tmp.y, tmp.z);
      patchPoint(F23, 1, z, 0.002, tmp); P.setXYZ(2, tmp.x, tmp.y, tmp.z);
      P.needsUpdate = true;
      bm.opacity = 0.14 * k * (1 - smooth(0.85, 0.95, p));
    });
  }

  // ---------- terminator: night-side shell with a rotating sun direction uniform ----------
  {
    const g = new THREE.Group();
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uSun: { value: new THREE.Vector3() }, uFade: { value: 1 } },
      vertexShader: 'varying vec3 vN; void main(){ vN = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: 'uniform vec3 uSun; uniform float uFade; varying vec3 vN; void main(){ float d = dot(vN, uSun); float night = smoothstep(0.04, -0.12, d); float glow = exp(-abs(d) * 40.0) * 0.5; gl_FragColor = vec4(mix(vec3(0.03,0.03,0.03), vec3(1.0,0.8,0.6), glow / max(night + glow, 1e-3)), (night * 0.82 + glow * 0.35) * uFade); }',
    });
    const shell = new THREE.Mesh(new THREE.SphereGeometry(R * 1.006, 128, 64), mat);
    g.add(shell);
    const inv = new THREE.Quaternion(), axis = new THREE.Vector3(0, 1, 0);
    element('terminator', g, moon, (p, k) => {
      // sun direction in moon-local space, rotated about the spin axis: the terminator sweeps across the disc
      moon.getWorldQuaternion(inv).invert();
      mat.uniforms.uSun.value.copy(sunDir).applyQuaternion(inv).applyAxisAngle(axis, THREE.MathUtils.lerp(1.4, -1.4, p)).normalize();
      mat.uniforms.uFade.value = k;
    });
  }

  // ---------- pins at the real LOLA site lat/lons ----------
  {
    const g = new THREE.Group(), n = SITES.length;
    const stick = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.004 * R, 0.004 * R, 1, 6).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ color: 0xf0eee6, transparent: true }), n);
    const head = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.008 * R, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }), n);
    SITES.forEach((s, i) => head.setColorAt(i, new THREE.Color(s[0] === '23' ? 0xd97757 : s[0] === '06' ? 0xb7cdbe : 0xe6dcc8)));
    g.add(stick, head);
    const dirs = SITES.map(s => siteDir(s[1], s[2])), up = new THREE.Vector3(0, 1, 0);
    element('pins', g, moon, (p, k) => {
      for (let i = 0; i < n; i++) {
        const t0 = 0.08 + i * 0.09, s = smooth(t0, t0 + 0.1, p), pop = s * (1 + 0.25 * Math.sin(Math.PI * s));   // little overshoot
        const L = 0.07 * R * pop, d = dirs[i];
        q.setFromUnitVectors(up, d);
        mtx.compose(tmp.copy(d).multiplyScalar(R), q, sc.set(Math.max(pop, 1e-4), Math.max(L, 1e-4), Math.max(pop, 1e-4))); stick.setMatrixAt(i, mtx);
        mtx.compose(tmp.copy(d).multiplyScalar(R + L), q, sc.setScalar(Math.max(pop, 1e-4))); head.setMatrixAt(i, mtx);
      }
      stick.instanceMatrix.needsUpdate = head.instanceMatrix.needsUpdate = true;
      stick.material.opacity = head.material.opacity = k;
    });
  }

  // ---------- swarm: landers raining onto Site 06 (real landed / failed proportion) ----------
  {
    const g = new THREE.Group(), N = opts.swarm === "closed" ? SWARM_N_CLOSED : SWARM_N_ALL;
    const geo = new THREE.ConeGeometry(0.006 * R, 0.014 * R, 5); geo.rotateX(Math.PI);   // tip down
    const im = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }), N);
    const seed = new Float32Array(N * 5); let s1 = 11; const rnd = () => ((s1 = (s1 * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < N; i++) {
      const ok = i < SWARM_OK;
      seed.set([(rnd() - 0.5) * 0.9, (rnd() - 0.5) * 0.9, rnd() * 0.45, ok ? (rnd() - 0.5) * 0.12 : (rnd() - 0.5) * 0.9, ok ? (rnd() - 0.5) * 0.12 : (rnd() - 0.5) * 0.9], i * 5);
      im.setColorAt(i, new THREE.Color(ok ? 0x74dba6 : 0xff7381));
    }
    g.add(im);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), up = new THREE.Vector3(0, -1, 0);
    element('swarm', g, moon, (p, k) => {
      for (let i = 0; i < N; i++) {
        const o = i * 5, t = smooth(seed[o + 2], seed[o + 2] + 0.45, p);
        patchPoint(F06, seed[o], seed[o + 1], 0.22, a);                // start high and scattered
        patchPoint(F06, seed[o + 3], seed[o + 4], 0.004, b);           // end: on target (landed) or off (failed)
        tmp.copy(a).lerp(b, t);
        const ok = i < SWARM_OK, landedTilt = !ok && t >= 1 ? 1.2 : 0;
        q.setFromUnitVectors(up, tmp2.copy(b).negate().normalize());
        if (landedTilt) q.multiply(qT.setFromAxisAngle(F06.e, landedTilt));
        mtx.compose(tmp, q, sc.setScalar(t > 0 ? 1 : 1e-4)); im.setMatrixAt(i, mtx);
      }
      im.instanceMatrix.needsUpdate = true; im.material.opacity = k;
    });
  }

  // ---------- rover: dotted trail of the real planned route on the Site 23 patch ----------
  const roverEnd = new THREE.Vector3(), landerAt = new THREE.Vector3(); patchPoint(F23, 0, 0, 0.004, landerAt); roverEnd.copy(landerAt);
  {
    const g = new THREE.Group(), MAXN = 220;
    const pos = new Float32Array(MAXN * 3), pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); pg.setDrawRange(0, 0);
    const dots = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0xd97757, size: 0.02 * R, map: softDot, transparent: true, depthWrite: false }));
    const rover = new THREE.Mesh(new THREE.BoxGeometry(0.012 * R, 0.006 * R, 0.009 * R), new THREE.MeshBasicMaterial({ color: 0xf0eee6, transparent: true }));
    const base0 = new THREE.Mesh(new THREE.OctahedronGeometry(0.01 * R, 0), new THREE.MeshBasicMaterial({ color: 0xe6dcc8, transparent: true }));
    base0.position.copy(landerAt);
    const frameM = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, z]) => patchPoint(F23, x * 0.85, z * 0.85, 0.002, new THREE.Vector3()))), new THREE.LineBasicMaterial({ color: 0xa8a49a, transparent: true, opacity: 0.4 }));
    g.add(frameM, dots, base0, rover);
    let n = 0;   // route metres -> patch units: drawn ENLARGED around the lander (largest offset -> 0.75 of the patch half-size); shape true
    fetch(base + 'assets/data/rover_route_site23.json').then(r => r.json()).then(d => {
      n = Math.min(MAXN, d.points.length);
      const HALF = Math.max(1, ...d.points.map(P => Math.max(Math.abs(P[0]), Math.abs(P[2])))) / 0.75;
      for (let i = 0; i < n; i++) { const P = d.points[Math.floor(i * (d.points.length - 1) / Math.max(1, n - 1))]; patchPoint(F23, P[0] / HALF, P[2] / HALF, 0.003, tmp); pos.set([tmp.x, tmp.y, tmp.z], i * 3); }
      pg.attributes.position.needsUpdate = true;
    }).catch(() => {});
    element('rover', g, moon, (p, k) => {
      const u = smooth(0.05, 0.95, p) * Math.max(0, n - 1), i = Math.floor(u);
      pg.setDrawRange(0, n ? i + 1 : 0);
      if (n > 1) { const j = Math.min(n - 1, i + 1), f = u - i; tmp.fromArray(pos, i * 3).lerp(tmp2.fromArray(pos, j * 3), f); rover.position.copy(tmp); roverEnd.copy(tmp); rover.lookAt(tmp2.fromArray(pos, j * 3)); }
      dots.material.opacity = rover.material.opacity = base0.material.opacity = k; frameM.material.opacity = 0.4 * k;
    });
  }

  // ---------- packets: rover -> lander -> Earth ----------
  {
    const g = new THREE.Group(), NP = 10;
    const earth = new THREE.Vector3();
    const lineG = new THREE.BufferGeometry(); lineG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
    const line = new THREE.Line(lineG, new THREE.LineBasicMaterial({ color: 0xe6dcc8, transparent: true, opacity: 0.5 })); line.frustumCulled = false;
    const ppos = new Float32Array(NP * 3), pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(ppos, 3));
    const pts = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0xffc9a0, size: 0.02 * R, map: softDot, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    pts.frustumCulled = false;
    g.add(line, pts);
    const rv = new THREE.Vector3();
    element('packets', g, moon, (p, k) => {
      rv.copy(F23.n).addScaledVector(F23.e, -0.06).addScaledVector(F23.nn, 0.03).normalize().multiplyScalar(R * 1.004);   // rover, a little off the lander
      earth.copy(opts.earthDirLocal || tmp2.set(-0.3, -0.2, 1).normalize()).multiplyScalar(R * 2.2).add(tmp.copy(F23.n).multiplyScalar(R));
      const L = lineG.attributes.position; L.setXYZ(0, rv.x, rv.y, rv.z); L.setXYZ(1, landerAt.x, landerAt.y, landerAt.z); L.setXYZ(2, earth.x, earth.y, earth.z); L.needsUpdate = true;
      const d1 = rv.distanceTo(landerAt), d2 = landerAt.distanceTo(earth), tot = d1 + d2;
      for (let i = 0; i < NP; i++) {
        const s = ((p * 2.5 + i / NP) % 1) * tot;
        if (s < d1) tmp.copy(rv).lerp(landerAt, s / d1); else tmp.copy(landerAt).lerp(earth, (s - d1) / d2);
        ppos[i * 3] = tmp.x; ppos[i * 3 + 1] = tmp.y; ppos[i * 3 + 2] = tmp.z;
      }
      pg.attributes.position.needsUpdate = true;
      line.material.opacity = 0.5 * k; pts.material.opacity = k;
    });
  }

  return Object.assign(els, {
    names: Object.keys(els),
    siteDir,
    orientToSite(lat, lon, worldDir) { return new THREE.Quaternion().setFromUnitVectors(siteDir(lat, lon), worldDir.clone().normalize()); },
    sites: SITES.map(([id, lat, lon]) => ({ id, lat, lon })),
    dispose() { for (const k of Object.keys(els)) els[k].dispose && els[k].dispose(); disposables.forEach(d => d.dispose && d.dispose()); },
  });
}
