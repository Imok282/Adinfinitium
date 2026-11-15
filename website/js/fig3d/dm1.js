// 3D figures for site DM1 (research/rover.html: `route`, research/end-to-end.html: `e2e`), loaded by js/fig3d.js.
// Data: assets/fig3d/dm1/ (tools/export_fig3d_dm1.py): observed NASA LOLA terrain of the rover world (2.56 km centre crop
// of the DM1 tile, 10 m/px), the hazard U-Net cells, and rover 0's simulated v3.1 traverse from the end-to-end test.
import { THREE, terrainMesh, heightAt, loadJSON, loadHeights, loadTex, toMesh, fatLine, marker } from '../fig3d.js';

const D = 'assets/fig3d/dm1/';
const EXAG = 2.5;
const BAT_LO = 75, BAT_HI = 100;            // colour scale for battery (rover 0 spans 76.0 to 100 Wh)

const lerp = (a, b, t) => a + (b - a) * t;
// low -> amber -> full: --bad, amber, --ok
function batColor(wh) {
  const t = Math.min(1, Math.max(0, (wh - BAT_LO) / (BAT_HI - BAT_LO)));
  const A = [0.90, 0.50, 0.48], B = [1.0, 0.72, 0.30], C = [0.56, 0.81, 0.65];
  return t < 0.5 ? A.map((v, i) => lerp(v, B[i], t * 2)) : B.map((v, i) => lerp(v, C[i], (t - 0.5) * 2));
}
// dark -> sunlit (monthly mean sunlit fraction at the rover's cell)
function sunColor(f) {
  const A = [0.27, 0.33, 0.62], B = [1.0, 0.86, 0.48];
  return A.map((v, i) => lerp(v, B[i], Math.min(1, Math.max(0, f))));
}

async function base(ctx) {
  const d = await loadJSON(D + 'dm1.json');
  const m = d.terrain;
  const [h, relief, hazard] = await Promise.all([loadHeights(D + 'height.bin', m), loadTex(D + 'relief.jpg'), loadTex(D + 'hazard.jpg')]);
  const mat = new THREE.MeshLambertMaterial({ map: hazard });
  const terr = terrainMesh(h, m, EXAG, mat);
  ctx.add(terr);
  ctx.onDispose(() => { relief.dispose(); hazard.dispose(); });
  const R = d.route;
  // route draped on the mesh (heights from the same mesh, lifted a little so it never sinks into it)
  const P = R.xy.map(([x, y]) => toMesh(terr, x, y, heightAt(h, m, x, y), 0.012));
  const at = (x, y, lift = 0) => toMesh(terr, x, y, heightAt(h, m, x, y), lift);
  // lander + PSR-edge targets (rover 0's target bigger)
  ctx.add(marker(at(0, 0), 0xffb547, 0.42, 0.05));
  d.targets.forEach(([x, y], i) => {
    const mine = i === d.target_index;
    ctx.add(marker(at(x, y), mine ? 0x9ec9ff : 0x6f93c4, mine ? 0.34 : 0.16, mine ? 0.045 : 0.026));
  });
  return { d, h, m, terr, mat, relief, hazard, R, P, at };
}

function routeLine(ctx, P, cols, width = 4.5) {
  const pts = [], cc = [];
  P.forEach((p, i) => { pts.push(p.x, p.y, p.z); cc.push(...cols[i]); });
  return fatLine(ctx, pts, { colors: cc, width });
}

function roverDot() {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(new THREE.SphereGeometry(0.03, 18, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })));
  const halo = new THREE.Mesh(new THREE.SphereGeometry(0.06, 18, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, depthWrite: false }));
  g.add(halo);
  return g;
}

const fmtH = s => { const h = Math.round(s / 3600); return h < 48 ? `${h} h` : `${(h / 24).toFixed(1)} d`; };

// research/rover.html: rover 0's traverse coloured by battery, with a replay slider
export async function route(ctx) {
  const B = await base(ctx);
  if (ctx.dead) return;
  const { R, P, mat, relief } = B;
  mat.map = relief; mat.needsUpdate = true;          // terrain only; the hazard overlay is the end-to-end figure's toggle
  ctx.add(routeLine(ctx, P, R.battery_wh.map(batColor)));
  const dot = roverDot(); ctx.add(dot);
  const ice = R.ice_reach_index;
  ctx.legend([['linear-gradient(90deg, rgb(230,128,122), rgb(255,184,77), rgb(143,207,166));width:30px', `battery ${BAT_LO} → ${BAT_HI} Wh`], ['#ffb547', 'lander'], ['#9ec9ff', 'its PSR-edge target'], ['#6f93c4', 'other targets']]);
  const N = R.xy.length - 1;
  ctx.slider('i', 'Hour of the traverse', {
    min: 0, max: N, value: ctx.shot ? N : 0, fps: 22,
    format: i => `hour ${i}/${N} · ${Math.round(R.dist_m[i]).toLocaleString('en-US')} m · ${R.battery_wh[i].toFixed(1)} Wh · sunlit ${Math.round(R.sun_frac[i] * 100)}%${i >= ice ? ' · target reached' : ''}`,
    onInput: i => { dot.position.copy(P[i]); },
  });
  ctx.frame({ dist: 3.5, elev: 0.6, az: -0.3, target: new THREE.Vector3(0.5, -0.08, 0.5) });
}

// research/end-to-end.html: terrain + hazard U-Net cells + rover 0's route, with toggles
export async function e2e(ctx) {
  const B = await base(ctx);
  if (ctx.dead) return;
  const { d, R, P, mat, relief, hazard } = B;
  let line = null;
  const read = ctx.readout();
  const F = d.fleet;
  read.textContent = `Fleet on this world (${F.rovers} rovers, strict rule): ${F.strict_returned} returned, ${F.strict_lost} lost, ${F.alive_without_completed_trip} alive without a completed trip. Route shown: rover 0, the lowest-index rover that returned; ${Math.round(R.total_dist_m).toLocaleString('en-US')} m, ice edge reached at hour ${R.ice_reach_index}, back within ${R.end_distance_from_lander_m.toFixed(1)} m of the lander.`;
  ctx.toggle('ov', 'Overlay', [['hazard', 'Hazard U-Net'], ['none', 'Terrain only']], v => { mat.map = v === 'hazard' ? hazard : relief; mat.needsUpdate = true; ctx.dirty(); });
  let key = null;
  ctx.toggle('col', 'Route colour', [['bat', 'Battery'], ['sun', 'Sunlight']], v => {
    if (line) { ctx.remove(line); line.geometry.dispose(); line.material.dispose(); ctx.lineMats.splice(ctx.lineMats.indexOf(line.material), 1); }
    line = routeLine(ctx, P, v === 'bat' ? R.battery_wh.map(batColor) : R.sun_frac.map(sunColor));
    ctx.add(line);
    const items = v === 'bat'
      ? [['linear-gradient(90deg, rgb(230,128,122), rgb(255,184,77), rgb(143,207,166));width:30px', `battery ${BAT_LO} → ${BAT_HI} Wh`]]
      : [['linear-gradient(90deg, rgb(69,84,158), rgb(255,219,122));width:30px', 'sunlit share 0 → 100% (monthly mean)']];
    items.push(['rgb(230,107,97)', 'hazard (U-Net ≥ 0.5)'], ['#ffb547', 'lander'], ['#9ec9ff', 'rover 0\'s target']);
    if (key) key.remove();
    key = ctx.legend(items);
  });
  ctx.frame({ dist: 4.3, elev: 0.72, az: 0.35, target: new THREE.Vector3(0.3, -0.08, 0.3) });
}
