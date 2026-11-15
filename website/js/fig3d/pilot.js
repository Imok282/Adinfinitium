// 3D figure for research/pilot-benchmark.html: paired descents on Site42, each mission flown by classic guidance (dashed)
// and by the learned pilot (solid); green = landed, red = crashed. Pairs where only one landed are drawn thick.
// Data: assets/fig3d/pilot/ (tools/export_fig3d_pilot.py: worldmodel/lander/fly.py loop, benchmark mission generator).
import { THREE, terrainMesh, heightAt, loadJSON, loadBin, loadHeights, loadTex, fatLine } from '../fig3d.js';

const D = 'assets/fig3d/pilot/';
const OK = [0.56, 0.86, 0.66], BAD = [0.92, 0.48, 0.46];          // --ok / --bad

export async function pairs(ctx) {
  const meta = await loadJSON(D + 'pilot3d.json');
  const tm = meta.terrain, T = meta.tracks;
  const [h, tex, buf] = await Promise.all([loadHeights(D + tm.file, tm), loadTex(D + tm.texture), loadBin(D + T.file)]);
  if (ctx.dead) return;
  const terr = terrainMesh(h, tm, 1, new THREE.MeshLambertMaterial({ map: tex }));
  ctx.add(terr);
  const mid = terr.userData.mid;
  const q = new Uint16Array(buf), off = T.offset, sc = T.scale;
  const P = i => [off[0] + q[i] * sc[0], off[1] + q[i + 1] * sc[1], off[2] + q[i + 2] * sc[2], off[3] + q[i + 3] * sc[3]];
  const W = p => [p[0] / 1000, (p[2] - mid) / 1000, -p[1] / 1000];       // tile ENU m -> mesh km (exag 1)
  const C = Object.fromEntries(T.columns.map((c, i) => [c, i]));
  const rows = T.rows;

  // all tracks of both conditions in ONE additive LineSegments; per-vertex cond/policy/differs/time/dash distance
  let nSeg = 0; for (const r of rows) nSeg += r[C.n_points] - 1;
  const pos = new Float32Array(nSeg * 6), col = new Float32Array(nSeg * 6), att = new Float32Array(nSeg * 2 * 4), dd = new Float32Array(nSeg * 2);
  const ePos = [], eCol = [], eAtt = [];
  let s = 0, tMax = 0;
  for (const r of rows) {
    const ok = r[C.ok] === 1, c = ok ? OK : BAD, k = ok ? 0.42 : 0.9;
    const a4 = [r[C.cond_idx], r[C.policy_idx], r[C.differs], 0];
    let dist = 0;
    for (let j = 0; j < r[C.n_points] - 1; j++, s++) {
      const pa = P((r[C.first_point] + j) * 4), pb = P((r[C.first_point] + j + 1) * 4);
      const wa = W(pa), wb = W(pb);
      pos.set(wa, s * 6); pos.set(wb, s * 6 + 3);
      for (let v = 0; v < 2; v++) { col[s * 6 + v * 3] = c[0] * k; col[s * 6 + v * 3 + 1] = c[1] * k; col[s * 6 + v * 3 + 2] = c[2] * k; }
      att.set([a4[0], a4[1], a4[2], pa[3]], s * 8); att.set([a4[0], a4[1], a4[2], pb[3]], s * 8 + 4);
      dd[s * 2] = dist; dist += Math.hypot(wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]); dd[s * 2 + 1] = dist;
    }
    const e = P((r[C.first_point] + r[C.n_points] - 1) * 4);
    ePos.push(...W(e)); eCol.push(...c); eAtt.push(a4[0], a4[1], a4[2], e[3]);
    tMax = Math.max(tMax, e[3]);
  }
  const uni = { uTime: { value: 1e9 }, uCond: { value: 0 }, uPol: { value: new THREE.Vector2(1, 1) }, uDiff: { value: 0 }, uPx: { value: Math.min(2, devicePixelRatio || 1) } };
  const VS = `attribute vec3 acol; attribute vec4 att; attribute float dd; uniform float uPx; varying vec3 vC; varying vec4 vA; varying float vD;
    void main(){ vC=acol; vA=att; vD=dd; vec4 mv=modelViewMatrix*vec4(position,1.); gl_Position=projectionMatrix*mv; gl_PointSize=uPx*clamp(16./-mv.z,2.,5.); }`;
  const FS = pts => `uniform float uTime; uniform float uCond; uniform vec2 uPol; uniform float uDiff; varying vec3 vC; varying vec4 vA; varying float vD;
    void main(){
      if (abs(vA.x-uCond)>.5) discard;
      float on = vA.y<.5 ? uPol.x : uPol.y; if (on<.5) discard;
      if (uDiff>.5 && vA.z<.5) discard;
      if (vA.w>uTime) discard;
      ${pts ? 'vec2 d=gl_PointCoord-.5; float r=dot(d,d); if(r>.25) discard; gl_FragColor=vec4(vC*(1.-r*2.5),1.);'
            : 'if (vA.y<.5 && fract(vD*9.)>.55) discard; gl_FragColor=vec4(vC,1.);'} }`;
  const mk = pts => new THREE.ShaderMaterial({ uniforms: uni, vertexShader: VS, fragmentShader: FS(pts), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('acol', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('att', new THREE.BufferAttribute(att, 4)); geo.setAttribute('dd', new THREE.BufferAttribute(dd, 1));
  const segs = new THREE.LineSegments(geo, mk(false)); segs.frustumCulled = false; segs.renderOrder = 2;
  const eg = new THREE.BufferGeometry();
  eg.setAttribute('position', new THREE.Float32BufferAttribute(ePos, 3)); eg.setAttribute('acol', new THREE.Float32BufferAttribute(eCol, 3));
  eg.setAttribute('att', new THREE.Float32BufferAttribute(eAtt, 4)); eg.setAttribute('dd', new THREE.Float32BufferAttribute(new Float32Array(ePos.length / 3), 1));
  const ends = new THREE.Points(eg, mk(true)); ends.frustumCulled = false; ends.renderOrder = 3;
  ctx.add(segs); ctx.add(ends);

  // landing cells: small amber dots on the ground
  const tp = [];
  for (const [x, y] of meta.targets_xy_m) tp.push(x / 1000, (heightAt(h, tm, x, y) - mid) / 1000 + 0.004, -y / 1000);
  const tg = new THREE.BufferGeometry(); tg.setAttribute('position', new THREE.Float32BufferAttribute(tp, 3));
  ctx.add(new THREE.Points(tg, new THREE.PointsMaterial({ color: 0xffb547, size: 3, sizeAttenuation: false, transparent: true, opacity: 0.8 })));

  // pairs where exactly one pilot landed: thick lines (pilot solid, classic dashed), rebuilt per condition
  let hl = [];
  const buildHL = cond => {
    hl.forEach(o => { ctx.remove(o.line); o.line.geometry.dispose(); o.line.material.dispose(); ctx.lineMats.splice(ctx.lineMats.indexOf(o.line.material), 1); });
    hl = [];
    for (const r of rows) {
      if (r[C.cond_idx] !== cond || !r[C.differs]) continue;
      const pts = [], ts = [];
      for (let j = 0; j < r[C.n_points]; j++) { const p = P((r[C.first_point] + j) * 4); pts.push(...W(p)); ts.push(p[3]); }
      const pilot = r[C.policy_idx] === 1, ok = r[C.ok] === 1;
      const line = fatLine(ctx, pts, { color: new THREE.Color(...(ok ? OK : BAD)), width: pilot ? 3.2 : 2.4 });
      if (!pilot) { line.material.dashed = true; line.material.dashSize = 0.05; line.material.gapSize = 0.035; line.material.needsUpdate = true; }
      line.renderOrder = 4; ctx.add(line);
      hl.push({ line, ts, pol: r[C.policy_idx] });
    }
  };
  const applyTime = () => {
    const t = uni.uTime.value;
    for (const o of hl) {
      let k = 0; while (k < o.ts.length && o.ts[k] <= t + 1e-6) k++;
      o.line.geometry.instanceCount = Math.max(0, k - 1);
      o.line.visible = (o.pol === 0 ? uni.uPol.value.x : uni.uPol.value.y) > 0.5;
    }
  };

  const read = ctx.readout();
  const say = () => {
    const cond = ctx.state.cond, c = meta.counts[cond];
    read.textContent = `${cond === 'nominal' ? 'Normal' : 'Harsh (3×)'} failure rates · ${c.n_pairs} paired missions · pilot landed ${c.pilot_landed}, classic ${c.classic_landed} · ` +
      (c.only_pilot + c.only_classic
        ? `only the pilot landed ${c.only_pilot}, only classic ${c.only_classic} (thick lines)`
        : 'no pair where only one landed in this subset');
  };
  ctx.toggle('cond', 'Failure rates', [['nominal', 'Normal'], ['harsh', 'Harsh (3×)']], v => {
    uni.uCond.value = v === 'nominal' ? 0 : 1; buildHL(uni.uCond.value); applyTime(); say(); ctx.dirty();
  });
  ctx.toggle('pol', 'Show', [['both', 'Both'], ['pilot', 'Pilot'], ['classic', 'Classic']], v => {
    uni.uPol.value.set(v === 'pilot' ? 0 : 1, v === 'classic' ? 0 : 1); applyTime(); ctx.dirty();
  });
  ctx.check('diff', 'Only pairs that differ', v => { uni.uDiff.value = v ? 1 : 0; ctx.dirty(); }, false);
  ctx.legend([['#8fcfa6', 'landed'], ['#e5807a', 'crashed'], ['#cfcbc0', 'solid = pilot · dashed = classic'], ['#ffb547', 'landing cell']]);
  const step = 2, max = Math.ceil(tMax / step) * step;
  ctx.slider('t', 'Simulated time', {
    min: 0, max, step, value: ctx.shot ? max : 0, fps: 14,
    format: v => `t = ${v} s of ${max} s simulated`,
    onInput: v => { uni.uTime.value = v >= max ? 1e9 : v; applyTime(); },
  });
  // frame on the cloud of descents (targets are spread over the tile; starts 1.5-2.5 km above them)
  ctx.frame({ dist: 9.2, elev: 0.32, az: 0.7, target: new THREE.Vector3(0, 0.75, 0) });
}
