// 3D figure for research/terrain.html (loaded by js/fig3d.js): three hide-and-check patches of held-out Site 23 as surfaces
// (real LOLA height, kriging from the shown laser pixels, the old Gaussian backup), with an error heat overlay and the
// laser pixels. Data: assets/fig3d/terrain/ (tools/export_fig3d_terrain.py; numbers checked against records/weak_flags/terrain.json).
import { THREE, terrainMesh, loadJSON, loadBin } from '../fig3d.js';

const D = 'assets/fig3d/terrain/';
const SCALE = 12.5, SY = 1, EXAG = 2 * SCALE / SY;   // 320 m patch -> 4 scene units; heights x2 overall. terrainMesh's fixed
// 120 m plinth is scaled by SY only, so it stays thin
const ERR_MAX = 3;                       // m, top of the error colour scale

// viridis (matplotlib), 5 stops, as in the 2-D figure on the page
const VIR = [[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]];
// error: warm sequential, pale -> deep red
const ERR = [[0.96, 0.93, 0.85], [0.98, 0.78, 0.55], [0.94, 0.52, 0.33], [0.80, 0.25, 0.22], [0.55, 0.08, 0.15]];
function ramp(stops, t, out, o) {
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i, a = stops[i], b = stops[i + 1];
  out[o] = a[0] + (b[0] - a[0]) * f; out[o + 1] = a[1] + (b[1] - a[1]) * f; out[o + 2] = a[2] + (b[2] - a[2]) * f;
}
const css = c => `rgb(${c.map(v => Math.round(v * 255)).join(',')})`;

export async function patches(ctx) {
  const [meta, buf] = await Promise.all([loadJSON(D + 'patches.json'), loadBin(D + 'patches.bin')]);
  if (ctx.dead) return;
  const n = meta.n, nn = n * n, per = 3 * nn * 2 + nn;
  const KEYS = ['truth', 'kriging', 'backup'];
  const data = meta.patches.map((p, i) => {
    const o = i * per, s = {};
    KEYS.forEach((k, j) => {
      const v = new Uint16Array(buf, o + j * nn * 2, nn), a = new Float32Array(nn), q = (p.max_m - p.min_m) / 65535;
      for (let x = 0; x < nn; x++) a[x] = p.min_m + v[x] * q;
      s[k] = a;
    });
    return { p, s, mask: new Uint8Array(buf, o + 3 * nn * 2, nn) };
  });

  const mat = new THREE.MeshLambertMaterial({ vertexColors: true, color: 0xd2d2d2 });   // scene lights are bright; keep the colours true
  const dotMat = new THREE.PointsMaterial({ size: 0.075, vertexColors: true, sizeAttenuation: true, depthWrite: false, transparent: true, opacity: 0.95 });
  let grp = null, dots = null;
  let read = null, key = null;
  const W = meta.window, P23 = meta.site23_pooled;

  function build() {
    const { p, s, mask } = data[ctx.state.patch];
    const k = ctx.state.surf, h = s[k], err = ctx.state.err && k !== 'truth';
    if (grp) { ctx.remove(grp); grp.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material && o.material !== mat) o.material.dispose(); }); }
    if (dots) { ctx.remove(dots); dots.geometry.dispose(); dots = null; }
    const m = { n, size_m: meta.size_m, min_m: p.min_m, max_m: p.max_m };
    grp = terrainMesh(h, m, EXAG, mat);
    grp.scale.set(SCALE, SY, SCALE);
    const g = grp.children[0].geometry, col = new Float32Array(nn * 3), t = s.truth;
    for (let i = 0; i < nn; i++) {
      if (err) ramp(ERR, Math.abs(h[i] - t[i]) / ERR_MAX, col, i * 3);
      else ramp(VIR, (h[i] - p.min_m) / (p.max_m - p.min_m), col, i * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    ctx.add(grp);
    if (ctx.state.laser) {
      const pos = g.attributes.position, pts = [], pc = [];
      for (let i = 0; i < nn; i++) {
        if (!mask[i]) continue;
        pts.push(pos.getX(i) * SCALE, pos.getY(i) * SY + 0.03, pos.getZ(i) * SCALE);
        if (mask[i] === 1) pc.push(1, 1, 1); else pc.push(0.85, 0.47, 0.34);   // shown = white, hidden = accent
      }
      const dg = new THREE.BufferGeometry();
      dg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3)); dg.setAttribute('color', new THREE.Float32BufferAttribute(pc, 3));
      dots = new THREE.Points(dg, dotMat); dots.renderOrder = 3; ctx.add(dots);
    }
    // readout: the page's metric (RMSE on the hidden real laser heights)
    const where = `patch ${p.index + 1} of 3 (row ${p.origin_rc[0]}, col ${p.origin_rc[1]})`;
    if (k === 'truth') read.textContent = `Real LOLA height, ${where}: ${p.relief_m} m relief · ${p.n_shown} laser pixels shown, ${p.n_hidden} hidden.${ctx.state.err ? ' Error heat applies to kriging and the old backup.' : ''}`;
    else {
      const e = p.errors[k], net = p.errors.network.rmse_hidden_m;
      read.textContent = `${k === 'kriging' ? 'Kriging' : 'Old backup'}, ${where}: RMSE ${e.rmse_hidden_m.toFixed(2)} m (MAE ${e.mae_hidden_m.toFixed(2)} m) on the ${p.n_hidden} hidden laser pixels · diffusion network ${net.toFixed(2)} m on the same pixels · whole window: kriging ${W.kriging_rmse_m.toFixed(2)} m, network ${W.network_rmse_m.toFixed(2)} m.`;
    }
    key.innerHTML = err
      ? `<span><i style="background:linear-gradient(90deg,${ERR.map(css).join(',')})"></i>|error| 0 → ≥ ${ERR_MAX} m</span><span><i style="background:#fff"></i>shown</span><span><i style="background:rgb(217,120,87)"></i>hidden (scored)</span>`
      : `<span><i style="background:linear-gradient(90deg,${VIR.map(css).join(',')})"></i>height ${Math.round(p.min_m)} → ${Math.round(p.max_m)} m</span><span><i style="background:#fff"></i>shown laser pixel</span><span><i style="background:rgb(217,120,87)"></i>hidden (scored)</span>`;
    ctx.dirty();
  }

  ctx.state.patch ??= 1; ctx.state.surf ??= 'kriging';
  let ready = false;
  const rebuild = () => { if (ready) build(); };
  ctx.toggle('surf', 'Surface', [['truth', 'Real (LOLA)'], ['kriging', 'Kriging'], ['backup', 'Old backup']], rebuild);
  ctx.toggle('patch', 'Patch', [[0, '1'], [1, '2'], [2, '3']], rebuild);
  ctx.check('err', 'Error heat', rebuild, false);
  ctx.check('laser', 'Laser pixels', rebuild, true);
  key = ctx.legend([]);
  read = ctx.readout();
  ready = true; build();
  ctx.onDispose(() => { mat.dispose(); dotMat.dispose(); });
  ctx.frame({ dist: 8.4, elev: 0.72, az: 0.7, targetY: -0.35 });
}
