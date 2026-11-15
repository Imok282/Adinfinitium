// #rover beat: after touchdown the rover drives its real planned route on its own (3D in js/scene.js).
// DOM part only: HUD numbers from assets/data/rover_route_site23.json (tools/export_rover_route.py), projected labels
// handed to the scene (it writes their transforms in its own loop), and the static route map for ?static / reduced motion.
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };

export function mount(sec, ctx) {
  const q = k => sec.querySelector(`[data-rv="${k}"]`);
  const el = { t: q('t'), d: q('d'), bat: q('bat'), batbar: q('batbar'), sun: q('sun'), st: q('st'), id: q('id'), ev: q('eval') };
  const chart = sec.querySelector('.rv-chart'); let hudOn = false;
  // style writes only when the value changes (cached per element + property)
  const setVar = (e, k, v) => { const c = e.__rv || (e.__rv = {}); if (c[k] !== v) { c[k] = v; e.style.setProperty(k, v); } };
  const head = sec.querySelector('.rv-head'), hud = sec.querySelector('.rv-hud'), tiles = [...sec.querySelectorAll('.rv-tile')];
  let R = null, lastI = -2, labelsSet = false, s = 0, onNow = false;
  const fmtKm = m => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);
  function show(i, phaseRamp) {
    if (!R || i === lastI) return;
    lastI = i;
    const n = R.t_s.length, k = Math.max(0, Math.min(n - 1, i));
    el.t.textContent = `${Math.round(R.t_s[k] / 3600)} h on the surface`;
    el.d.textContent = fmtKm(R.dist_m[k]);
    if (R.battery_wh) {
      const f = R.battery_wh[k] / (R.battery_cap_wh || 100);
      el.bat.textContent = `${Math.round(f * 100)} %`;
      el.batbar.style.setProperty('--bat', f.toFixed(3));
      el.batbar.style.background = f < 0.2 ? 'var(--accent)' : 'var(--ink, #1f1f1d)';
    }
    if (R.sun_frac) el.sun.textContent = `sunlight ${Math.round(R.sun_frac[k] * 100)} %`;
    const reach = R.ice_reach_index;
    el.st.textContent = phaseRamp ? 'Down the ramp, leaving the lander'
      : k < reach - 2 ? (R.sun_frac && R.sun_frac[k] < 0.25 ? 'In shadow: slow, saving battery' : 'Driving out, steering around steep cells')
      : k <= reach + 1 ? 'At an ice-target cell (PSR edge)'
      : k < n - 1 ? 'Heading home before the battery runs out' : 'Back at the lander';
  }
  fetch('assets/data/rover_route_site23.json').then(r => r.json()).then(d => {
    R = d;
    if (el.id) el.id.textContent = d.run_id;
    // drive graphs from the real run: battery %, sunlight fraction, distance (normalised), 0..211 h
    try {
      const n = d.t_s.length, X = i => (i / (n - 1) * 240).toFixed(1), Y = v => (88 - v * 84).toFixed(1);
      const line = f => d.t_s.map((_, i) => `${i ? 'L' : 'M'}${X(i)} ${Y(f(i))}`).join('');
      const set = (k, f) => { const e = sec.querySelector(`[data-rv="${k}"]`); if (e) { e.setAttribute('d', line(f)); e.setAttribute('pathLength', '1'); } };
      if (d.battery_wh) set('g-bat', i => d.battery_wh[i] / (d.battery_cap_wh || 100));
      if (d.sun_frac) set('g-sun', i => d.sun_frac[i]);
      set('g-dist', i => d.dist_m[i] / d.total_dist_m);
    } catch (e) { /* charts optional */ }
    show(d.t_s.length - 1, false);          // static / initial state: the completed run
    lastI = -2;
  }).catch(() => {});

  const reduced = ctx.reducedMotion || document.body.classList.contains('no-webgl');
  return {
    update(p, vis) {
      try {
        const scn = ctx.getScene && ctx.getScene();
        if (reduced || !scn) return;
        if (!labelsSet && scn.setRoverLabels) {
          const L = k => sec.querySelector(`[data-lab="${k}"]`);
          scn.setRoverLabels({ rover: L('rover'), lander: L('lander') });   // ice + steep are explained by the tiles
          labelsSet = true;
        }
        const on = vis > 0.01;
        if (on !== onNow) { onNow = on; sec.classList.toggle('on', on); }
        scn.setRover && scn.setRover(on || p > 0 ? p : null);
        s = ctx.shot ? p : s + (p - s) * 0.5;
        if (!hudOn) { hud.style.setProperty('--hud', '1'); hudOn = true; }
        // heading only once the stage is pinned (it would otherwise scroll up across the big 3D lander)
        if (chart) setVar(chart, '--draw', smooth(0.12, 0.95, s).toFixed(3));   // on the svg only: a var on the section would restyle its whole subtree
        if (head) setVar(head, 'opacity', (p > 0 ? smooth(0.0, 0.05, s) : 0).toFixed(3));
        // tiles pop in one by one around the rover as it drives (scrubbed, reversible)
        for (let i = 0; i < tiles.length; i++) setVar(tiles[i], '--pop', smooth(0.1 + i * 0.07, 0.17 + i * 0.07, s).toFixed(2));
        if (R) {
          // same arc-length mapping as the 3D rover (js/scene.js): distance along the route -> sample index
          const dd = smooth(0.12, 0.95, s) * R.total_dist_m; let k = 0; while (k < R.dist_m.length - 1 && R.dist_m[k + 1] <= dd) k++;
          const u = k;
          show(Math.floor(u), s < 0.12);
        }
      } catch (e) { /* never break the page */ }
    },
  };
}
