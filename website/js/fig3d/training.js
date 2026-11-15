// 3D figure for research/training.html (loaded by js/fig3d.js): the 960 simulated descents on Site 06 by training
// stage (no guidance = open_loop, classic guidance, AI pilot on top of classic). Ported from the `ghosts` view of
// js/models3d.js: all tracks in ONE additive LineSegments geometry + one Points geometry for the end points; per-vertex
// simulated time drives the replay (every track grows on the same clock), per-vertex stage index drives the toggle.
// Data: assets/data/ghosts_lander_site06.{json,bin} (tools/export_ghosts_web.py) over the Site 06 LOLA mesh
// (assets/models3d/, tools/export_models3d.py). Every count shown comes from the data header.
import { THREE, terrainMesh, loadJSON, loadHeights, loadTex } from '../fig3d.js';
import { loadGhosts, countOf } from '../ghostdata.js';

const M3 = 'assets/models3d/';
const STAGES = [['open_loop', 'No guidance'], ['classic', 'Classic'], ['pilot', 'AI pilot']];

export async function ghosts(ctx) {
  const meta = await loadJSON(M3 + 'models3d.json'), m = meta.d;
  const [h, tex, G] = await Promise.all([loadHeights(M3 + 'site06_height.bin', m), loadTex(M3 + m.texture), loadGhosts('lander', 'site06')]);
  if (ctx.dead) return;
  const terr = terrainMesh(h, m, 1, new THREE.MeshLambertMaterial({ map: tex, color: 0xa6a39b }));
  ctx.add(terr);
  const { mid } = terr.userData;
  const P = G.pts, head = G.head;
  const SI = Object.fromEntries(STAGES.map(([p], i) => [p, i]));
  const COL = { ok: [0.56, 0.86, 0.66], bad: [0.92, 0.48, 0.46] };   // --ok / --bad
  const nSeg = G.runs.reduce((a, r) => a + r.n - 1, 0);
  const pos = new Float32Array(nSeg * 6), col = new Float32Array(nSeg * 6), tt = new Float32Array(nSeg * 2), st = new Float32Array(nSeg * 2);
  const ePos = [], eCol = [], eT = [], eSt = [];
  let s = 0, tMax = 0;
  for (const r of G.runs) {
    const si = SI[r.policy] ?? 0;
    // additive intensity per track, scaled so ~300 overlapping landings glow without burning out to white
    const c = r.ok ? COL.ok : COL.bad, k = r.ok ? Math.min(0.34, 5 / Math.sqrt(G.runs.length)) : (r.policy === 'open_loop' ? 0.3 : 0.6);
    for (let j = 0; j < r.n - 1; j++, s++) {
      const a = (r.start + j) * 4, b = a + 4, o = s * 6;
      pos[o] = P[a] / 1000; pos[o + 1] = (P[a + 2] - mid) / 1000; pos[o + 2] = -P[a + 1] / 1000;
      pos[o + 3] = P[b] / 1000; pos[o + 4] = (P[b + 2] - mid) / 1000; pos[o + 5] = -P[b + 1] / 1000;
      for (let q = 0; q < 3; q++) col[o + q] = col[o + 3 + q] = c[q] * k;
      st[s * 2] = st[s * 2 + 1] = si;
      tt[s * 2] = P[a + 3]; tt[s * 2 + 1] = P[b + 3];
    }
    const e = (r.start + r.n - 1) * 4;
    ePos.push(P[e] / 1000, (P[e + 2] - mid) / 1000, -P[e + 1] / 1000);
    eCol.push(...c.map(x => x * 0.9)); eT.push(P[e + 3]); eSt.push(si);
    tMax = Math.max(tMax, P[e + 3]);
  }
  const uni = { uTime: { value: ctx.shot ? 1e9 : 0 }, uMask: { value: new THREE.Vector3(1, 1, 1) }, uPx: { value: Math.min(2, devicePixelRatio || 1) } };
  const VS = `attribute vec3 acol; attribute float tt; attribute float stg; uniform float uPx; varying vec3 vC; varying float vT; varying float vS;
    void main(){ vC=acol; vT=tt; vS=stg; vec4 mv=modelViewMatrix*vec4(position,1.); gl_Position=projectionMatrix*mv; gl_PointSize=uPx*clamp(9./-mv.z,1.5,4.); }`;
  const FS = pts => `uniform float uTime; uniform vec3 uMask; varying vec3 vC; varying float vT; varying float vS;
    void main(){ float m = vS<.5?uMask.x:(vS<1.5?uMask.y:uMask.z); if(m<.5||vT>uTime) discard;
    ${pts ? 'vec2 d=gl_PointCoord-.5; float r=dot(d,d); if(r>.25) discard; gl_FragColor=vec4(vC*(1.-r*3.),1.);' : 'gl_FragColor=vec4(vC,1.);'} }`;
  const mk = pts => new THREE.ShaderMaterial({ uniforms: uni, vertexShader: VS, fragmentShader: FS(pts), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('acol', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('tt', new THREE.BufferAttribute(tt, 1)); geo.setAttribute('stg', new THREE.BufferAttribute(st, 1));
  const segs = new THREE.LineSegments(geo, mk(false)); segs.frustumCulled = false; segs.renderOrder = 2;
  const eg = new THREE.BufferGeometry();
  eg.setAttribute('position', new THREE.Float32BufferAttribute(ePos, 3)); eg.setAttribute('acol', new THREE.Float32BufferAttribute(eCol, 3));
  eg.setAttribute('tt', new THREE.Float32BufferAttribute(eT, 1)); eg.setAttribute('stg', new THREE.Float32BufferAttribute(eSt, 1));
  const ends = new THREE.Points(eg, mk(true)); ends.frustumCulled = false; ends.renderOrder = 3;
  ctx.add(segs); ctx.add(ends);

  // landing target: ring on the ground + faint beam
  const [tx, ty, tz] = head.target_xyz_m;
  const tg = new THREE.Vector3(tx / 1000, (tz - mid) / 1000, -ty / 1000);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.085, 40), new THREE.MeshBasicMaterial({ color: 0xffb547, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.position.copy(tg).y += 0.006;
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 2.3, 8), new THREE.MeshBasicMaterial({ color: 0xffb547, transparent: true, opacity: 0.25 }));
  beam.position.copy(tg).y += 1.15;
  ctx.add(ring); ctx.add(beam);

  // stage toggle + readout (counts straight from the header)
  const fmt = n => Number(n).toLocaleString('en');
  const out = ctx.readout();
  const line = sel => {
    const show = sel === 'all' ? STAGES : STAGES.filter(([p]) => p === sel);
    const parts = show.map(([p, l]) => { const c = countOf(head, p); return `${l}: ${fmt(c.ok)} of ${fmt(c.total)} landed`; });
    const n = show.reduce((a, [p]) => a + countOf(head, p).total, 0);
    return `${fmt(n)} descents shown · ${parts.join(' · ')}`;
  };
  ctx.toggle('stage', 'stage', [['all', 'All three'], ...STAGES], v => {
    uni.uMask.value.set(...STAGES.map(([p]) => (v === 'all' || v === p ? 1 : 0)));
    out.textContent = line(v);
  }, 'all');
  ctx.legend([['#8fdba8', 'landed'], ['#eb7a75', 'crashed or failed'], ['#ffb547', 'landing target']]);

  // replay: every run starts at simulated t = 0 and grows on the same clock; hold, repeat
  const SPAN = 7, HOLD = 3.5, rate = tMax / SPAN;
  let t = ctx.shot ? 1e9 : 0;
  if (!ctx.shot) ctx.onTick(dt => { t += dt; if (t > SPAN + HOLD) t = 0; uni.uTime.value = Math.min(t, SPAN) * rate; ctx.dirty(); });
  ctx.frame({ dist: 6.6, elev: 0.26, az: 0.62, target: new THREE.Vector3(tg.x * 0.8, tg.y + 1.05, tg.z * 0.8) });
}
