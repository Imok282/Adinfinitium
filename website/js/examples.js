// "What the model sees and says": renders the compiled model outputs in assets/examples/ (made read-only from the
// trained checkpoints by website/tools/examples/*.py). Nothing is computed here beyond formatting the stored values.
//   mountHazard(el)   hazard U-Net windows (images + per-window P/R/F1)       research/hazard-model.html
//   mountHazardTeaser(el)  one window, one image                              pilot.html
//   mountLight(el)    skyline model vs ray-cast truth vs coarse backup        research/training.html
//   mountPlanner(el)  top-10 rankings for one Site 23 world                   navigator.html
//   mountRover(el)    rover v3.1 route trace excerpt                          research/rover.html
const E = 'assets/examples/';
const get = u => fetch(E + u).then(r => { if (!r.ok) throw new Error(r.status + ' ' + u); return r.json(); });
const h = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === 'class' ? (n.className = v) : k === 'text' ? (n.textContent = v) : n.setAttribute(k, v);
  for (const c of kids) if (c != null) n.append(c);
  return n;
};
const f = (v, d = 2) => (v == null ? '–' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(d));
const pct = (v, d = 0) => (v == null ? '–' : (v * 100).toFixed(d) + ' %');
const fail = (el, what) => e => { console.warn('examples', what, e); el.textContent = `The ${what} examples could not be loaded.`; };
function tabs(labels, onPick, cls = 'ex-tabs') {
  const bar = h('div', { class: cls, role: 'tablist' });
  const bs = labels.map(([k, t], i) => {
    const b = h('button', { class: 'ex-tab', type: 'button', role: 'tab', 'aria-selected': i ? 'false' : 'true', text: t });
    b.addEventListener('click', () => { bs.forEach(x => x.setAttribute('aria-selected', x === b ? 'true' : 'false')); onPick(k); });
    bar.append(b); return b;
  });
  return bar;
}
const lazy = (el, fn) => {
  if (!el) return;
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: '400px' });
  io.observe(el);
};

// ---------------------------------------------------------------- hazard
const HZ_KIND = { steep: 'Steep', moderate: 'Moderate', flat: 'Flat', hard_case: 'Hard case (Site 42)' };
const HZ_IMG = [
  ['input_hillshade', 'Input: hillshade', 'gap-filled from ~11 % real laser pixels'],
  ['unet_probability', 'U-Net probability', 'black safe … yellow unsafe'],
  ['unet_mask', 'U-Net mask', 'red unsafe · green safe'],
  ['backup_slope_rule', 'Slope rule (backup)', 'red unsafe · green safe'],
  ['truth', 'Truth', 'official LOLA slope ≥ 12°'],
  ['unet_error', 'Error: U-Net', 'red missed · blue false alarm'],
  ['backup_error', 'Error: slope rule', 'red missed · blue false alarm'],
];
export function mountHazard(el) {
  lazy(el, () => get('hazard/hazard_examples.json').then(d => {
    el.textContent = '';
    const grid = h('div', { class: 'ex-hz-grid' }), stats = h('div', { class: 'ex-stats' }), cap = h('p', { class: 'ex-cap' });
    const show = kind => {
      const w = d.windows.find(x => x.kind === kind);
      grid.textContent = '';
      for (const [k, t, s] of HZ_IMG) {
        grid.append(h('figure', { class: 'ex-hz-cell' },
          h('img', { src: E + 'hazard/' + w.images[k], width: 256, height: 256, loading: 'lazy', alt: `${t}, ${w.site} ${HZ_KIND[kind]} window` }),
          h('figcaption', {}, h('b', { text: t }), h('span', { text: s }))));
      }
      const row = (name, m, cls) => h('tr', { class: cls }, h('th', { text: name }), h('td', { text: f(m.precision) }), h('td', { text: f(m.recall) }), h('td', { text: f(m.f1) }), h('td', { text: pct(m.missed_hazard_rate, 1) }));
      stats.textContent = '';
      stats.append(h('table', { class: 'ex-table' },
        h('thead', {}, h('tr', {}, h('th', { text: 'This window' }), h('th', { text: 'Precision' }), h('th', { text: 'Recall' }), h('th', { text: 'F1' }), h('th', { text: 'Missed hazards' }))),
        h('tbody', {}, row('U-Net', w.unet, 'p'), row('Slope rule', w.backup_slope_rule, 'c'))));
      const [x, y, ww] = w.window_px;
      cap.textContent = `Tile ${w.region} (${w.site}), window at px ${x},${y}, ${ww} × ${ww} px = ${(w.size_m / 1000).toFixed(2)} km · ${w.split} · ${pct(w.unsafe_fraction_truth)} of the window is unsafe · laser pixels ${pct(w.laser_pixel_fraction, 1)}. Input observed (LOLA); maps modelled.`;
    };
    el.append(tabs(d.windows.map(w => [w.kind, HZ_KIND[w.kind] || w.kind]), show), grid, stats, cap);
    show(d.windows[0].kind);
  }).catch(fail(el, 'hazard')));
}
export function mountHazardTeaser(el) {
  lazy(el, () => get('hazard/hazard_examples.json').then(d => {
    const w = d.windows.find(x => x.kind === 'steep') || d.windows[0];
    el.textContent = '';
    el.append(h('div', { class: 'ex-teaser' },
      h('img', { src: E + 'hazard/' + w.images.input_hillshade, width: 256, height: 256, loading: 'lazy', alt: `Hillshade input, ${w.site} steep window` }),
      h('img', { src: E + 'hazard/' + w.images.unet_mask, width: 256, height: 256, loading: 'lazy', alt: `U-Net hazard mask, ${w.site} steep window: red unsafe, green safe` })),
      h('p', { class: 'ex-cap', text: `Held-out ${w.site}, 1.28 km window: input (left) and U-Net mask (right, red unsafe). F1 ${f(w.unet.f1)} vs slope rule ${f(w.backup_slope_rule.f1)}. Modelled.` }));
  }).catch(fail(el, 'hazard')));
}

// ---------------------------------------------------------------- light
const LT_KIND = k => ({ open_rim_low_skyline: 'Open rim, low skyline', typical_median_skyline: 'Typical cell', enclosed_high_skyline: 'Enclosed, high skyline', hard_case_p95_model_error: 'Hard case (p95 error)' }[k] || k.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase()));
const SVGNS = 'http://www.w3.org/2000/svg';
const s = (tag, a = {}) => { const n = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(a)) n.setAttribute(k, v); return n; };
export function mountLight(el) {
  lazy(el, () => get('light/light_examples.json').then(d => {
    el.textContent = '';
    const chart = h('div', { class: 'ex-light' }), stats = h('div', { class: 'ex-stats' });
    const sp = d.sun_earth_path;
    const show = i => {
      const c = d.cells[i], sk = c.skyline_deg;
      const all = [...sk.truth_raycast_5m, ...sk.model, ...sk.backup_coarse_raycast_20m, ...(sp ? sp.sun_elev_deg : [])];
      let lo = Math.floor(Math.min(...all)) - 0.5, hi = Math.ceil(Math.max(...all)) + 0.5;
      const W = 960, H = 340, L = 46, R = 12, T = 12, B = 34;
      const X = az => L + (az / 360) * (W - L - R), Y = v => T + (hi - v) / (hi - lo) * (H - T - B);
      const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'ex-svg', role: 'img', 'aria-label': `Skyline elevation by azimuth for the ${LT_KIND(c.kind)} cell: ray-cast truth, model and coarse backup, with the Sun path.` });
      const step = hi - lo > 12 ? 4 : hi - lo > 6 ? 2 : 1;
      for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
        svg.append(s('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: v === 0 ? 'ex-ax0' : 'ex-grid' }));
        const t = s('text', { x: L - 8, y: Y(v) + 4, 'text-anchor': 'end', class: 'ex-tick' }); t.textContent = v + '°'; svg.append(t);
      }
      for (let az = 0; az <= 360; az += 45) {
        const t = s('text', { x: X(az), y: H - 12, 'text-anchor': 'middle', class: 'ex-tick' }); t.textContent = az + '°'; svg.append(t);
      }
      if (sp) for (let k = 0; k < sp.sun_az_deg.length; k += 2) {
        const az = ((sp.sun_az_deg[k] % 360) + 360) % 360;
        svg.append(s('circle', { cx: X(az), cy: Y(sp.sun_elev_deg[k]), r: 1.7, class: 'ex-sun' }));
      }
      const path = arr => arr.map((v, a) => (a ? 'L' : 'M') + X(a).toFixed(1) + ' ' + Y(v).toFixed(1)).join('');
      svg.append(s('path', { d: path(sk.backup_coarse_raycast_20m), class: 'ex-l-backup' }));
      svg.append(s('path', { d: path(sk.truth_raycast_5m), class: 'ex-l-truth' }));
      svg.append(s('path', { d: path(sk.model), class: 'ex-l-model' }));
      chart.textContent = '';
      chart.append(svg, h('div', { class: 'ex-legend' },
        h('span', {}, h('i', { class: 'k-truth' }), 'ray-cast truth (5 m)'),
        h('span', {}, h('i', { class: 'k-model' }), 'skyline model'),
        h('span', {}, h('i', { class: 'k-backup' }), 'coarse backup (20 m)'),
        sp ? h('span', {}, h('i', { class: 'k-sun' }), 'Sun path, Sept 2027') : null,
        h('span', { class: 'ex-axis', text: 'x: grid azimuth · y: elevation above horizontal' })));
      const e = c.error_deg, sh = c.sunlit_hours;
      stats.textContent = '';
      stats.append(h('table', { class: 'ex-table' },
        h('thead', {}, h('tr', {}, h('th', { text: 'This cell' }), h('th', { text: 'Mean error' }), h('th', { text: 'Max error' }), h('th', { text: `Sunlit h of ${sp ? sp.window_hours : '–'}` }))),
        h('tbody', {},
          h('tr', { class: 'p' }, h('th', { text: 'Model' }), h('td', { text: f(e.model_mae) + '°' }), h('td', { text: f(e.model_max) + '°' }), h('td', { text: sh.model })),
          h('tr', { class: 'c' }, h('th', { text: 'Coarse backup' }), h('td', { text: f(e.backup_mae) + '°' }), h('td', { text: f(e.backup_max) + '°' }), h('td', { text: sh.backup })),
          h('tr', {}, h('th', { text: 'Ray-cast truth' }), h('td', { text: '–' }), h('td', { text: '–' }), h('td', { text: sh.truth })))));
    };
    const sm = d.summary_300_cells;
    el.append(tabs(d.cells.map((c, i) => [i, LT_KIND(c.kind)]), show), chart, stats,
      h('p', { class: 'ex-cap', text: `${d.site}, ${d.region} · ${d.split} · over 300 random cells the model's mean error is ${f(sm.model_mae_deg)}° vs ${f(sm.backup_mae_deg)}° for the backup (better on ${sm.cells_model_better_pct} %). Truth is window-limited to 1280 m, so far mountains are missing; Sun path from SPICE DE440, azimuth alignment unverified. Heights observed (LOLA); skylines modelled.` }));
    show(0);
  }).catch(fail(el, 'light')));
}

// ---------------------------------------------------------------- planner
const PL_TAB = [['navigator', 'Navigator ranking'], ['set', 'PlannerSetNet'], ['logistic', 'Logistic']];
const PL_PRED = { navigator: ['navigator_score', 'Score', 3], set: ['p_set', 'Predicted p', 3], logistic: ['p_logistic', 'Predicted p', 3] };
export function mountPlanner(el) {
  lazy(el, () => get('planner/planner_examples.json').then(d => {
    el.textContent = '';
    const wrap = h('div', { class: 'ex-scroll' });
    let cur = 'navigator', sortKey = 'rank', dir = 1;
    const best = new Set(d.simulated_best10_ids);
    const cols = () => [
      ['rank', 'Rank', r => r.rank, r => r.rank],
      ['cell', 'Cell · hour', r => r.id, r => `(${r.cell_rc[0]}, ${r.cell_rc[1]}) · h ${r.hour}`],
      ['pred', PL_PRED[cur][1], r => r[PL_PRED[cur][0]], r => f(r[PL_PRED[cur][0]], PL_PRED[cur][2])],
      ['sim', 'Simulated p(mission)', r => r.simulated.p_mission, r => f(r.simulated.p_mission, 3) + (best.has(r.id) ? ' ★' : '')],
      ['slope', 'Slope', r => r.factors.slope_deg, r => f(r.factors.slope_deg, 1) + '°'],
      ['sun', 'Sunlit', r => r.factors.sunlit_mean_frac, r => pct(r.factors.sunlit_mean_frac)],
      ['earth', 'Earth visible', r => r.factors.earth_visible_frac, r => pct(r.factors.earth_visible_frac)],
      ['ice', 'Ice target', r => r.factors.ice_target_distance_km, r => f(r.factors.ice_target_distance_km, 2) + ' km'],
      ['haz', 'Unsafe ≤150 m', r => r.factors.hazard_frac_150m, r => pct(r.factors.hazard_frac_150m, 1)],
    ];
    const draw = () => {
      const C = cols(), col = C.find(c => c[0] === sortKey) || C[0];
      const rows = [...d.top10[cur]].sort((a, b) => (col[2](a) - col[2](b)) * dir);
      const thead = h('tr', {});
      for (const [k, t] of C) {
        const b = h('button', { type: 'button', class: 'ex-sort', text: t + (k === sortKey ? (dir > 0 ? ' ↑' : ' ↓') : '') });
        b.addEventListener('click', () => { if (sortKey === k) dir = -dir; else { sortKey = k; dir = k === 'rank' || k === 'cell' || k === 'slope' || k === 'ice' || k === 'haz' ? 1 : -1; } draw(); });
        thead.append(h('th', { 'aria-sort': k === sortKey ? (dir > 0 ? 'ascending' : 'descending') : 'none' }, b));
      }
      const tb = h('tbody', {});
      for (const r of rows) tb.append(h('tr', {}, ...C.map(([k, , , fmt]) => h(k === 'rank' || k === 'cell' ? 'th' : 'td', { text: fmt(r) }))));
      wrap.textContent = '';
      wrap.append(h('table', { class: 'ex-table ex-pl' }, h('thead', {}, thead), tb));
    };
    const w = d.world, m = d.this_world_metrics, a = d.site23_means_7_worlds;
    const chosen = h('p', { class: 'ex-note' });
    const say = () => { const c = m[cur]; chosen.textContent = `Top pick's simulated p(mission) ${f(c.chosen_simulated_p, 3)} (best in this world ${f(m.best_simulated_p, 3)}, regret ${f(c.regret, 3)}) · rank correlation with simulated truth over all ${w.candidates} candidates ${f(c.rank_corr)}.`; };
    el.append(tabs(PL_TAB, k => { cur = k; sortKey = 'rank'; dir = 1; draw(); say(); }), wrap, chosen,
      h('p', { class: 'ex-note ex-honest', text: `This world flatters the learned planners. Averaged over all 7 Site 23 worlds, the top pick's simulated p(mission) is logistic ${f(a.logistic.mean_chosen_simulated_p, 3)}, Navigator ${f(a.navigator.mean_chosen_simulated_p, 3)}, PlannerSetNet ${f(a.set.mean_chosen_simulated_p, 3)}; on 2027-03-01 and 2027-06-01 the Navigator picks better.` }),
      h('p', { class: 'ex-cap', text: `${w.site}, ${w.date}, ${w.candidates} candidate (cell, landing hour) pairs · ${w.split} · offline ranking, no flight. ★ = among the 10 best by simulation (${d.label_provenance.n_lander_sims_per_candidate} lander + ${d.label_provenance.n_rover_sims_per_candidate} rover sims per candidate, classic pilot). Terrain observed (LOLA); outcomes modelled. Click a column to sort.` }));
    draw(); say();
  }).catch(fail(el, 'planner')));
}

// ---------------------------------------------------------------- rover
export function mountRover(el) {
  lazy(el, () => get('rover/rover_v31_trace_excerpt.json').then(d => {
    el.textContent = '';
    const cap = d.battery_cap_wh, maxD = d.total_dist_m;
    const tb = h('tbody', {});
    let prev = null;
    for (const r of d.rows) {
      if (prev != null && r.hour - prev > 1) tb.append(h('tr', { class: 'ex-gap' }, h('td', { colspan: 6, text: `… hours ${prev + 1}–${r.hour - 1} not in the excerpt …` })));
      prev = r.hour;
      const bar = (v, max, cls) => h('span', { class: 'ex-bar ' + cls }, h('i', { style: `width:${Math.max(0, Math.min(100, v / max * 100)).toFixed(1)}%` }));
      const ev = r.event || '';
      tb.append(h('tr', { class: ev ? 'ex-ev' : '' },
        h('th', { text: r.hour }),
        h('td', {}, bar(r.dist_travelled_m, maxD, 'd'), h('b', { text: Math.round(r.dist_travelled_m).toLocaleString('en') + ' m' })),
        h('td', {}, bar(r.battery_wh, cap, 'bt'), h('b', { text: f(r.battery_wh, 1) + ' Wh' })),
        h('td', {}, bar(r.sun_frac, 1, 's'), h('b', { text: pct(r.sun_frac) })),
        h('td', { text: Math.round(Math.hypot(r.x_east_m, r.z_south_m)).toLocaleString('en') + ' m' }),
        h('td', { class: 'ex-evt', text: ev })));
    }
    const fl = d.fleet_512;
    el.append(h('p', { class: 'ex-label', text: 'Route trace, not a decision log' }),
      h('div', { class: 'ex-scroll' }, h('table', { class: 'ex-table ex-rv' },
        h('thead', {}, h('tr', {}, h('th', { text: 'Hour' }), h('th', { text: 'Distance driven' }), h('th', { text: 'Battery' }), h('th', { text: 'Sunlit' }), h('th', { text: 'From lander' }), h('th', { text: 'Event' }))), tb)),
      h('p', { class: 'ex-cap', text: `${d.site}, ${d.date} · ${d.split} · rover 0: reached the ice target (${d.ice_source}) at hour ${d.ice_reach_hour}, outcome "${d.outcome}", ${Math.round(d.total_dist_m).toLocaleString('en')} m in ${d.duration_h} h. This rover is one of the ${fl.strict_returned} of ${fl.rovers} that returned under the strict rule; ${fl.strict_lost} were lost. Excerpt of the recorded state per hour, copied unchanged; no per-hour decision log exists. Modelled (simulation).` }));
  }).catch(fail(el, 'rover')));
}
