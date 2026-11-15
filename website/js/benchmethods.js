// Benchmarks page: headline gate card, Limits, Methodology, Reproduce and Downloads blocks. Plain factual text; every
// number is read from benchmarks.json (ctx.bench), stats.json (ctx.stats) or projections.json (ctx.proj).
import { esc } from './barcharts.js';
import { famOf, ltest, pfmt, praw } from './benchspecs.js';

const row = (ctx, fid, prefix) => famOf(ctx, fid)?.rows.find(r => r.name.startsWith(prefix));
const val = (ctx, fid, prefix, metric) => row(ctx, fid, prefix)?.values?.[metric];
const metric = (ctx, fid, id) => famOf(ctx, fid)?.metrics.find(m => m.id === id);
// value with its family's digits and unit
function v(ctx, fid, prefix, id) {
  const x = val(ctx, fid, prefix, id), m = metric(ctx, fid, id);
  if (x == null || !m) return '—';
  const dec = (String(x).split('.')[1] || '').length;
  const s = (+x).toFixed(Math.max(m.digits ?? 0, Math.min(dec, 4)));
  return m.unit === '%' || m.unit === '°' ? s + m.unit : m.unit ? `${s} ${m.unit}` : s;
}
const f1 = (x, d = 1) => (+x).toFixed(d);
const table = (head, rows, cls = '') => `<div class="bm-table-wrap"><table class="bm-table bm-mtable ${cls}"><thead><tr>${head.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, i) => i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;

/* ---------- headline gate card ---------- */
export function gateCard(ctx) {
  const r = (ctx.proj?.training_record || []).find(x => /^Lander pilot Dream-RSI/.test(x.model));
  if (!r) return '';
  const head = String(r.best).split(' (')[0];
  const rest = String(r.best).slice(head.length).replace(/^\s*\(|\)\s*$/g, '');
  return `<figure class="bc bc-tile">
    <div class="bc-top"><h4 class="bc-title">Safety gate · pilot self-improvement</h4></div>
    <p class="bc-big">${esc(head)}</p>
    <p class="bc-callout">No candidate that loses a mission the shipped pilot lands is accepted, so the shipped pilot is unchanged and no regression was shipped.</p>
    <p class="bc-n">Gate (Site11): ${esc(String(r.start).replace(/^gate:\s*/i, '').replace(/\s*\(Site11\)\s*$/, ''))}. ${rest ? 'Best candidate: ' + esc(rest.replace(/^best\s*/i, '')) + '.' : ''}</p>
    <details class="bc-how"><summary>How measured</summary><div class="bc-how-body"><dl>
      <div><dt>Job</dt><dd>${esc(r.amount)}; ${esc(r.machine)}, ${esc(r.when)}</dd></div>
      <div><dt>Set before / after</dt><dd>Set before the run: the acceptance rule (lose no gate mission the current pilot lands, land more in total).</dd></div>
      <div><dt>Source</dt><dd>${esc(r.source)}</dd></div></dl></div></details>
  </figure>`;
}

/* ---------- limits ---------- */
export function renderLimits(ctx) {
  const lt = ctx.stats?.lander_tests, un = ltest(ctx, 'unseen_nominal'), uh = ltest(ctx, 'unseen_harsh'), sh = ltest(ctx, 'site23_harsh');
  const s42 = (ctx.stats?.lander_per_map_post_hoc?.items || []).find(x => x.site === 'Site42' && x.condition === 'nominal');
  const set = row(ctx, 'planner', 'PlannerSetNet'), lg = row(ctx, 'planner', 'Logistic');
  const items = [
    un && `<b>Pilot.</b> On the Site23 final exam, flown once, the pilot tied classic guidance (${v(ctx, 'pilot', 'Lander pilot', 'success_exam')} each)${sh ? `; under harsh rates it was ${f1(sh.diff_pts, 2)} pts (p = ${pfmt(praw(sh))}, n.s.)` : ''}. The unseen-map gain has p = ${pfmt(praw(un))}, but after Holm adjustment over the ${lt.n_tests} lander tests it is p = ${pfmt(un.p_holm)}, above 0.05. ${uh ? `Under harsh rates the unseen-map gain is not significant (p = ${pfmt(praw(uh))}). ` : ''}${s42 ? `Looked at after the result, the gain comes from one map, Site42 (${s42.diff_pts > 0 ? '+' : ''}${f1(s42.diff_pts, 2)} pts, p = ${pfmt(s42.p_recomputed)}). ` : ''}On missions with an engine failure the pilot lands less often than classic (${v(ctx, 'pilot', 'Lander pilot', 'engine_out')} vs ${v(ctx, 'pilot', 'Classic', 'engine_out')}).`,
    `<b>Hazard.</b> On the fresh, flatter sites the hazard F1 falls to ${v(ctx, 'hazard', 'Hazard U-Net', 'f1_fresh')}, against ${v(ctx, 'hazard', 'Hazard U-Net', 'f1_heldout')} on the steep held-out sites; flat ground is missing from training.`,
    `<b>Light.</b> The light network is much slower than its backup (${v(ctx, 'light', 'Skyline transformer', 'runtime')} vs ${v(ctx, 'light', 'Coarse ray-cast', 'runtime')} per window), and the rover still uses the backup.`,
    `<b>Terrain.</b> Our diffusion network (${v(ctx, 'terrain', 'Diffusion U-Net', 'rmse_heldout')}) loses to classical kriging (${v(ctx, 'terrain', 'Kriging', 'rmse_heldout')}) on the held-out sites; kriging is what we present.`,
    set && `<b>Planner.</b> No learned planner beats the hand-set Navigator ranking with confidence. ${esc(set.note)} ${lg ? esc(lg.note) : ''}`,
    `<b>Rover.</b> Both learned rover drivers failed on held-out sites (${v(ctx, 'rover_learned', 'Rover v1', 'returned')} and ${v(ctx, 'rover_learned', 'Rover v2', 'returned')} returned, against ${v(ctx, 'rover_learned', 'Planner: naive', 'returned')} for the naive planner). The scripted v3.1 brain still loses ${v(ctx, 'rover', 'Rover brain v3.1', 'lost_strict')} of rovers under the strict rule.`,
    `<b>Test sites.</b> Every real test site has now been used at least once, so no result on this page is a new unseen test.${ctx.missionBench ? ' The procedural Moon maps are the only fresh terrain.' : ''}`,
    `<b>Simulation.</b> Every lander, rover and planner number is simulation: a simplified 3-DOF terminal-descent lander and a kinematic rover with hourly decisions, on NASA LOLA 5 m terrain.`,
    `<b>Not yet run.</b> The evolution swarm has no run. Projections on this page are curve fits to our training logs, not results.`,
  ].filter(Boolean);
  return `<ul class="bm-limits">${items.map(x => `<li>${x}</li>`).join('')}</ul>`;
}

/* ---------- methodology ---------- */
export function renderMethodology(ctx) {
  const st = ctx.stats || {}, fx = st.facts || {}, tl = fx.touchdown_limits || {}, lt = st.lander_tests;
  const lf = fx.laser_fraction_heldout_windows, lff = fx.laser_fraction_fresh_windows, ip = fx.hazard_label_interpolated_pct;
  const bt = st.meta?.bootstrap;
  const testsRows = (lt?.items || []).map(t => [esc(t.label), esc(String(t.n_missions ?? '—')), t.pilot_only_landed != null ? `${t.pilot_only_landed} vs ${t.classic_only_landed}` : '—',
    `${t.diff_pts != null ? (t.diff_pts > 0 ? '+' : '') + f1(t.diff_pts, 2) + ' pts' : '—'}${t.diff_ci95_pts ? ` <span class="bm-ci">${f1(t.diff_ci95_pts[0], 2)} to ${f1(t.diff_ci95_pts[1], 2)}</span>` : ''}`,
    pfmt(praw(t)) + (praw(t) < 0.05 ? '' : ' (n.s.)'), pfmt(t.p_holm) + (t.p_holm < 0.05 ? '' : ' (n.s.)'), esc(t.registered || '')]);
  const pm = (st.lander_per_map_post_hoc?.items || []).map(x => [esc(`${x.site}, ${x.condition === 'nominal' ? 'normal' : 'harsh'}`), `${x.pilot_only_landed} vs ${x.classic_only_landed}`, `${x.diff_pts > 0 ? '+' : ''}${f1(x.diff_pts, 2)} pts`, pfmt(x.p_recomputed)]);
  const effRows = (st.effects || []).map(e => [esc(e.label), `${f1(e.value, e.unit === 'x' && e.value < 2 ? 2 : 1)}${e.unit === 'x' ? '×' : e.unit}`,
    e.ci95 ? `${f1(e.ci95[0], e.unit === 'x' && e.value < 2 ? 2 : 1)}–${f1(e.ci95[1], e.unit === 'x' && e.value < 2 ? 2 : 1)}${e.unit === 'x' ? '×' : e.unit}` : 'none', esc(e.ci_method), esc(e.source)]);
  const mb = ctx.missionBench;
  const fixed = [
    ['Paired design', 'Set before the run', 'Every method in a table is scored on the same windows, missions or worlds; lander missions use fixed seeds.'],
    ['Held-out maps for the pilot', 'Set before the run', 'Site42, Shoemaker and DM2 were kept for benchmarking only.'],
    ['Site23 final exam', 'Set before the run', 'Flown once and never again.'],
    ['Zero-regression gate', 'Set before the run', 'A self-improvement candidate is accepted only if it loses no gate mission the current pilot lands.'],
    ['Rover v3.1 on untouched sites', 'Set before the run', 'Sites and pass conditions written before the single evaluation.'],
    ['Light 6,000-step network', 'Set before the run', 'Adoption margin pre-registered; the network missed it and was not adopted.'],
    ['Pilot gain comes from Site42', 'Chosen after (post hoc)', 'Per-map breakdown read after the pooled result; not part of the Holm family.'],
    ['Hazard flat-site weakness', 'Chosen after (post hoc)', 'Grouping fresh sites by flatness was done after seeing the scores.'],
    ['Rover v3 reference on Site23+42', 'Chosen after (post hoc)', 'v3 was designed after these sites had been looked at.'],
    ...(mb ? [['Landing-site rehearsal and procedural Moon maps', 'Chosen after (post hoc)', 'Flown after the Site23 exam.']] : []),
  ];
  const prov = [
    ['Terrain heights', 'NASA LRO LOLA 5 m south-pole elevation models (PGDA)', '5 m', lf && lff ? `Laser returns in ${f1(lff.min_pct)}–${f1(lff.max_pct)}% of pixels per fresh window and ${f1(lf.min_pct)}–${f1(lf.max_pct)}% per held-out window; the rest is NASA interpolation` : '', 'Observed + interpolated'],
    ['Hazard labels', `Slope from the same elevation models; unsafe at ${tl.max_slope_deg ?? '—'}° or more`, '5 m', ip ? `${f1(ip.value)}% of label pixels come from interpolated heights (${ip.windows} test windows)` : '', 'Derived from observed + interpolated'],
    ['Sun and Earth directions', 'NASA/JPL SPICE kernels, DE440 ephemeris', 'per hour', 'Computed, not measured', 'Modelled'],
    ['Shadow (PSR) mask', 'NASA permanently-shadowed-region map, resampled to the 5 m grid', '5 m (resampled)', '', 'Observed, resampled'],
    ['Moon imagery', 'LROC (NASA/GSFC/Arizona State University)', '—', 'Used for visuals only; no benchmark reads it', 'Observed'],
    ['Lander, rover and planner outcomes', 'Our simulators on the terrain above', '—', '', 'Simulation'],
    ['Terrain-network samples', 'Our diffusion network', '5 m', 'Shown only as GENERATED — NOT OBSERVED', 'Generated'],
  ];
  const hz = famOf(ctx, 'hazard');
  const plTest = famOf(ctx, 'planner')?.test_set || '';
  const nMiss = (plTest.match(/(\d+) lander-twin/) || [])[1];
  const biases = [
    `Steep test sites inflate hazard F1: ${v(ctx, 'hazard', 'Hazard U-Net', 'f1_heldout')} on held-out Site23+42 against ${v(ctx, 'hazard', 'Hazard U-Net', 'f1_fresh')} on the flatter fresh sites; flagging everything already scores ${v(ctx, 'hazard', 'Constant', 'f1_heldout')} on Site23+42.`,
    'The hazard comparison leans towards the network: the slope-rule backup\'s slope map is one of the network\'s inputs.',
    hz && ip ? `Hazard truth is mostly NASA interpolation (${f1(ip.value)}% of label pixels), so the all-pixel score measures agreement with that interpolation.` : '',
    'The terrain test hides laser pixels next to shown pixels on the same laser track, which favours interpolation; wide gaps were not tested.',
    nMiss ? `Planner labels are noisy: each candidate's label comes from ${nMiss} simulated missions, which is similar in size to the differences between methods.` : '',
    'The lander is a simplified 3-DOF terminal-descent simulation (no attitude dynamics); the rover is kinematic with one decision per hour.',
    'Light truth and model both cut rays at a fixed range, so far horizons are missing from both.',
  ].filter(Boolean);
  return `
    <h3 class="bm-sub">Design</h3>
    <p class="bm-prose">${esc(ctx.bench.meta?.note || '')} Lander missions use fixed seeds, so both controllers fly identical missions with identical injected failures.</p>
    <h3 class="bm-sub">Held-out and previously seen sites</h3>
    <p class="bm-prose">Learned models are trained on separate training sites and selected on a validation site (Site11 for hazard, light, terrain and the pilot). Site23 and Site42 were held out from training, but are "previously seen" for rover v3 and v3.1 and for every later check. The fresh sites (DM1, LM7, NPA–NPD) and the untouched rover sites (LM2–LM5) have now each been used once. All real sites have therefore been used at least once.${mb ? ' The procedural Moon maps are the only fresh terrain.' : ''}</p>
    <h3 class="bm-sub">What lander "success" means</h3>
    <p class="bm-prose">A mission succeeds when it touches down with vertical speed at most ${tl.max_vertical_speed_m_s ?? '—'} m/s, horizontal speed at most ${tl.max_horizontal_speed_m_s ?? '—'} m/s and ground slope at most ${tl.max_slope_deg ?? '—'}° (${esc(tl.source || 'config/mission.yaml')}). The thrust direction may tilt at most a fixed angle from vertical (a 3-DOF stand-in for attitude), which is not a touchdown condition. Missions inject engine-outs, navigation drift and blind cameras; "harsh" triples those rates.</p>
    <h3 class="bm-sub">Statistics</h3>
    <p class="bm-prose">95% intervals: bootstrap over windows (hazard, light, terrain), over worlds (planner) and over sites (rover); Wilson intervals for lander rates and a paired normal interval for lander differences. Lander comparisons use the exact McNemar test on the discordant missions (two-sided).${bt ? ` Effect-size intervals below use ${bt.resamples.toLocaleString('en-US')} bootstrap resamples (seed ${bt.seed}, ${esc(bt.interval)}).` : ''}</p>
    <h3 class="bm-sub">Multiple comparisons</h3>
    <p class="bm-prose">${lt ? `${lt.n_tests} lander tests are reported (${esc(lt.family)}). p-values are Holm-adjusted over that family. The headline unseen-map result has p = ${pfmt(praw(ltest(ctx, 'unseen_nominal')))} unadjusted and p = ${pfmt(ltest(ctx, 'unseen_nominal').p_holm)} after adjustment.` : ''} Other families report intervals rather than tests.</p>
    ${lt ? table(['Test', 'Missions', 'Pilot-only vs classic-only landings', 'Difference (95% CI)', 'p (exact McNemar)', 'p (Holm)', 'Registration'], testsRows) : ''}
    ${pm.length ? `<details class="bm-details"><summary>Per-map breakdown (post hoc, not adjusted)</summary>${table(['Map', 'Pilot-only vs classic-only', 'Difference', 'p (exact McNemar)'], pm)}<p class="bm-n">Source: ${esc(st.lander_per_map_post_hoc.source)}</p></details>` : ''}
    <h3 class="bm-sub">Effect sizes</h3>
    ${table(['Effect', 'Value', '95% CI', 'Interval method', 'Source'], effRows)}
    <h3 class="bm-sub">Set before the run vs chosen after</h3>
    ${table(['Item', 'Status', 'What'], fixed.map(r => r.map(esc)))}
    <h3 class="bm-sub">Data provenance</h3>
    ${table(['Data', 'Source', 'Resolution', 'Interpolated or derived', 'Provenance'], prov.map(r => r.map(esc)))}
    <h3 class="bm-sub">Known biases</h3>
    <ul class="bm-limits">${biases.map(b => `<li>${esc(b)}</li>`).join('')}</ul>
    <p class="src">Sources: assets/data/stats.json (written by website/tools/export_bench_csv.py from the hand-off records), assets/data/benchmarks.json, config/mission.yaml.</p>`;
}

/* ---------- reproduce + downloads ---------- */
export function renderReproduce(ctx) {
  const r = ctx.stats?.reproduce; if (!r) return '';
  const ck = ctx.stats.checkpoints || [];
  return `
    <p class="bm-prose">Code: branch <code>${esc(r.branch)}</code>. ${esc(r.warning)}</p>
    ${table(['What', 'Command', 'Source'], r.commands.map(c => [esc(c.what), `<code>${esc(c.cmd)}</code>`, esc(c.source)]), 'bm-cmds')}
    <h3 class="bm-sub">Checkpoints</h3>
    <ul class="bm-ck">${ck.map(c => `<li><b>${esc(c.name)}</b>${c.id ? ` · id <code>${esc(c.id)}</code>` : ''}${c.sha256 ? ` · sha256 <code>${esc(c.sha256.slice(0, 12))}…</code>` : ''}</li>`).join('')}</ul>
    <details class="bm-details"><summary>Full sha256 and sources</summary>
      <ul class="bm-ck">${ck.map(c => `<li><b>${esc(c.name)}</b>${c.sha256 ? `<br><code class="bm-sha">${esc(c.sha256)}</code>` : ''}<br><span class="bm-n">${esc(c.source)}</span></li>`).join('')}</ul>
    </details>`;
}

export function renderDownloads(ctx) {
  const files = [
    ['assets/data/benchmarks.json', 'Every benchmark row, interval, n, note and source'],
    ['assets/data/stats.json', 'Lander tests with Holm-adjusted p, effect sizes, methodology facts, checkpoints, commands'],
    ['assets/data/projections.json', 'Training record, projections (not measured) and targets'],
    ['assets/data/results.json', 'Numbers used elsewhere on the site'],
    ...(ctx.missionBench ? [['assets/data/mission_bench.json', 'Landing-site rehearsal and procedural-Moon results']] : []),
  ];
  const csvs = (ctx.bench.families || []).map(f => [`assets/data/csv/${f.id}.csv`, `${f.name} table`]).concat([['assets/data/csv/lander_tests.csv', 'Lander tests with raw and Holm-adjusted p']]);
  const li = ([href, what]) => `<li><a href="${esc(href)}" download>${esc(href.split('/').pop())}</a> <span class="bm-n">${esc(what)}</span></li>`;
  return `<div class="bm-dl"><div><h3 class="bm-sub">JSON</h3><ul>${files.map(li).join('')}</ul></div><div><h3 class="bm-sub">CSV</h3><ul>${csvs.map(li).join('')}</ul></div></div>
    <p class="src">CSV files and stats.json are regenerated with <code>python3 website/tools/export_bench_csv.py</code>.</p>`;
}
