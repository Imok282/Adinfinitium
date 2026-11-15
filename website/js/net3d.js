// The pilot network, drawn from REAL sampled weights (assets/net.json, made by tools/export_net.py).
// Columns of glowing dots (neurons) + thousands of thin additive bezier lines, coloured by weight sign/strength.
// Self-contained: restyle via the STYLE object or buildNet(..., { style }).
import * as THREE from 'three';
import { STYLE, colHeight } from './net-style.js';
export { STYLE };

export function buildNet(data, opts = {}) {
  const S = { ...STYLE, ...(opts.style || {}) };
  const layers = data.layers;
  const L = layers.length;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 400);

  // neuron positions
  const pos = layers.map((ly, k) => {
    const h = colHeight(ly.size);
    const out = [];
    for (let i = 0; i < ly.n; i++) {
      const y = ly.n === 1 ? 0 : (i / (ly.n - 1) - 0.5) * h;
      out.push(new THREE.Vector3(k * S.spacing, y, 0));
    }
    return out;
  });
  const xMax = (L - 1) * S.spacing;

  // ---- background gradient + faint stars (only drawn while the section is visible) ----
  const bgMat = new THREE.ShaderMaterial({
    depthTest: false, depthWrite: false, transparent: true,
    uniforms: { top: { value: new THREE.Color(S.bgTop) }, bottom: { value: new THREE.Color(S.bgBottom) }, uMix: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: `varying vec2 vUv; uniform vec3 top; uniform vec3 bottom; uniform float uMix;
      float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
      void main(){
        vec3 c = mix(bottom, top, smoothstep(0.0, 1.0, vUv.y + 0.15*sin(vUv.x*3.0)));
        vec2 g = floor(vUv * vec2(420.0, 260.0));
        float s = step(0.997, h(g)) * 0.55;
        gl_FragColor = vec4(c + s, uMix);
      }`,
  });
  const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat);
  bg.frustumCulled = false; bg.renderOrder = -10;
  scene.add(bg);

  // ---- lines: one big LineSegments, additive ----
  let nSeg = 0;
  for (const e of data.edges) nSeg += e.w.length * S.segments;
  const lp = new Float32Array(nSeg * 6), lc = new Float32Array(nSeg * 6), lu = new Float32Array(nSeg * 2);
  let o = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), p0 = new THREE.Vector3(), p1 = new THREE.Vector3();
  const bez = (t, A, B, out) => {   // horizontal-tangent cubic: forms bow-tie fans
    const dx = (B.x - A.x) * 0.5, u = 1 - t;
    out.x = u*u*u*A.x + 3*u*u*t*(A.x+dx) + 3*u*t*t*(B.x-dx) + t*t*t*B.x;
    out.y = u*u*u*A.y + 3*u*u*t*A.y + 3*u*t*t*B.y + t*t*t*B.y;
    out.z = 0;
    return out;
  };
  for (const e of data.edges) {
    const A = pos[e.from], B = pos[e.to];
    for (let m = 0; m < e.w.length; m++) {
      const w = e.w[m], aw = Math.abs(w);
      a.copy(A[e.src[m]]); b.copy(B[e.dst[m]]);
      const sc = w >= 0 ? S.pos : S.neg;
      const t2 = Math.min(1, aw * aw * 1.4);
      const tint = S.tint;
      const base = [0, 1, 2].map(c => S.lineBase[c] * (1 - tint) + (sc[c] * (1 - t2) + S.strong[c] * t2) * tint);
      const gain = S.lineGain * (0.25 + 0.75 * Math.pow(aw, 0.7));
      for (let s = 0; s < S.segments; s++) {
        bez(s / S.segments, a, b, p0); bez((s + 1) / S.segments, a, b, p1);
        lp.set([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z], o * 6);
        for (let v = 0; v < 2; v++) for (let c = 0; c < 3; c++) lc[o * 6 + v * 3 + c] = base[c] * gain;
        lu[o * 2] = p0.x / xMax; lu[o * 2 + 1] = p1.x / xMax;
        o++;
      }
    }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
  lg.setAttribute('color', new THREE.BufferAttribute(lc, 3));
  lg.setAttribute('u', new THREE.BufferAttribute(lu, 1));
  const pulseCol = new THREE.Color().fromArray(S.pulse);
  const lineMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, vertexColors: true,
    uniforms: { uMix: { value: 0 }, pulse: { value: -1 }, reveal: { value: 1 }, pulseCol: { value: pulseCol } },
    vertexShader: `attribute float u; varying vec3 vC; varying float vU;
      void main(){ vC = color; vU = u; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `varying vec3 vC; varying float vU; uniform float uMix; uniform float pulse; uniform float reveal; uniform vec3 pulseCol;
      void main(){
        float p = exp(-pow((vU - pulse) * 28.0, 2.0));
        float r = smoothstep(reveal + 0.02, reveal - 0.02, vU);
        gl_FragColor = vec4((vC * (2.0 + 4.0 * p) + pulseCol * p * 0.05) * uMix * r, 1.0);
      }`,
  });
  const lines = new THREE.LineSegments(lg, lineMat);
  lines.frustumCulled = false;
  scene.add(lines);

  // ---- dots ----
  const nDots = pos.reduce((s, c) => s + c.length, 0);
  const dp = new Float32Array(nDots * 3), du = new Float32Array(nDots);
  let d = 0;
  for (const col of pos) for (const v of col) { dp.set([v.x, v.y, v.z], d * 3); du[d] = v.x / xMax; d++; }
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.BufferAttribute(dp, 3));
  dg.setAttribute('u', new THREE.BufferAttribute(du, 1));
  const dotMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    uniforms: { uMix: { value: 0 }, pulse: { value: -1 }, reveal: { value: 1 }, col: { value: new THREE.Color().fromArray(S.dot) },
      pulseCol: { value: pulseCol }, scale: { value: 1 } },
    vertexShader: `attribute float u; varying float vU; uniform float scale;
      void main(){ vU = u; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        gl_PointSize = clamp(scale * 7.0 / -mv.z, 1.5, 9.0); }`,
    fragmentShader: `varying float vU; uniform float uMix; uniform float pulse; uniform float reveal; uniform vec3 col; uniform vec3 pulseCol;
      void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.0, r);
        float p = exp(-pow((vU - pulse) * 28.0, 2.0));
        float rv = smoothstep(reveal + 0.02, reveal - 0.02, vU);
        gl_FragColor = vec4(mix(col, pulseCol, p) * a * (1.2 + 1.5 * p) * uMix * rv, 1.0); }`,
  });
  const dots = new THREE.Points(dg, dotMat);
  dots.frustumCulled = false;
  scene.add(dots);

  // ---- HTML labels (tiny monospace boxes at the top of each column) ----
  let labelRoot = null, labelEls = [];
  const tops = layers.map((ly, k) => new THREE.Vector3(k * S.spacing, colHeight(ly.size) / 2 + 0.35, 0));
  if (opts.labelRoot) {
    labelRoot = opts.labelRoot;
    labelEls = layers.map(ly => {
      const el = document.createElement('div');
      el.className = 'net-label';
      el.textContent = ly.label;
      labelRoot.appendChild(el);
      return el;
    });
  }

  const tmp = new THREE.Vector3();
  let W = 1, H = 1;
  let labelSize = [];                 // cached [w, h] per label (re-measured after a resize: font size is responsive)
  function resize(w, h) {
    W = w; H = h; labelSize = [];
    camera.aspect = w / h; camera.updateProjectionMatrix();
    dotMat.uniforms.scale.value = h / 220;
  }

  // q: section progress 0..1 (camera flies input -> output). mix: 0..1 visibility. t: seconds.
  function update(q, mix, t) {
    const narrow = W / H < 0.8;
    const e = q * q * (3 - 2 * q);
    const x = -3.5 + e * (xMax + 1.0);
    const side = narrow ? 9.5 : 6.2;
    camera.position.set(x - (narrow ? 2.5 : 4.5), 2.0 - e * 0.8, side);
    camera.lookAt(x + (narrow ? 3.0 : 5.5), narrow ? 0.4 : 1.1, 0);   // look a little above the centre: the net sits below the page captions
    const reveal = Math.min(1.05, 0.12 + q * 1.3);
    const pulse = ((t * 0.12) % 1.25) - 0.1;
    for (const m of [lineMat, dotMat]) { m.uniforms.uMix.value = mix; m.uniforms.pulse.value = pulse; m.uniforms.reveal.value = reveal; }
    bgMat.uniforms.uMix.value = mix;
    if (labelRoot) {
      labelRoot.style.opacity = mix > 0.05 ? Math.min(1, mix * 1.5) : 0;
      camera.updateMatrixWorld();
      // Labels of neighbouring columns can land on top of each other on screen. Place them greedily
      // (nearest column first): try the column top, then lift by one or two rows; if every row collides, hide it.
      const cand = [];
      for (let k = 0; k < L; k++) {
        tmp.copy(tops[k]).project(camera);
        const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.1 && Math.abs(tmp.y) < 1.1 && (k / (L - 1)) <= reveal + 0.02;
        if (!vis) { labelEls[k].style.display = 'none'; continue; }
        const d = tmp.z;
        cand.push({ k, sx: (tmp.x + 1) / 2 * W, sy: (1 - tmp.y) / 2 * H, d });
      }
      cand.sort((p, q) => p.d - q.d);            // closest to the camera (largest on screen) wins
      const boxes = [], GAP = 4, TOP = Math.max(66, H * (opts.labelTop ?? 0.3));   // keep labels clear of the top bar and the section captions
      for (const c of cand) {
        const el = labelEls[c.k];
        if (!labelSize[c.k]) { el.style.display = 'block'; labelSize[c.k] = [el.offsetWidth || 60, el.offsetHeight || 14]; }
        const [w, h] = labelSize[c.k];
        let placed = false;
        for (const row of [0, 1, 2, -1, -2, -3]) {
          if (placed) break;
          const y1 = c.sy - row * (h + 3), y0 = y1 - h, x0 = c.sx - w / 2, x1 = c.sx + w / 2;
          if (y0 < TOP) continue;                       // keep clear of the fixed top bar
          if (x1 > W - (W > 900 ? 300 : 0)) continue;     // keep the right ~300 px clear for the side lander + altitude rail
          if (boxes.some(b => x0 < b[2] + GAP && x1 > b[0] - GAP && y0 < b[3] + 2 && y1 > b[1] - 2)) continue;
          boxes.push([x0, y0, x1, y1]);
          el.style.display = 'block';
          el.style.transform = `translate(${x0.toFixed(1)}px, ${y0.toFixed(1)}px)`;
          placed = true;
        }
        if (!placed) el.style.display = 'none';
      }
    }
  }
  return { scene, camera, update, resize };
}

