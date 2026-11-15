// Chart specs shared by research/benchmarks.html, pilot.html, navigator.html and index.html (rendered by js/barcharts.js).
// Text only: every number is read at render time from benchmarks.json (ctx.bench), stats.json (ctx.stats, written by
// website/tools/export_bench_csv.py) or projections.json (ctx.proj).
import { esc } from './barcharts.js';

export const famOf = (ctx, id) => (ctx.bench?.families || []).find(f => f.id === id);
export const effect = (ctx, id) => (ctx.stats?.effects || []).find(e => e.id === id);
export const ltest = (ctx, id) => (ctx.stats?.lander_tests?.items || []).find(t => t.id === id);
const f = (x, d = 1) => (+x).toFixed(d);
const signed = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(+x).toFixed(d);
export const pfmt = p => (p >= 0.995 ? '1.00' : p >= 0.1 ? (+p).toFixed(2) : (+p).toPrecision(2));
export const praw = t => (t.p_recomputed ?? t.p_file);

/* ---------- callouts and significance lines ---------- */
const hazardRatio = (gid, d) => ctx => {
  const e = effect(ctx, 'hazard_f1_ratio_' + gid); if (!e) return '';
  return `${f(e.value, d)}× the slope rule's F1 (95% CI ${f(e.ci95[0], d)}–${f(e.ci95[1], d)}×, ${e.ci_method.split(' (')[0]}).`;
};
const lightRed = (gid, unitWord) => ctx => {
  const e = effect(ctx, 'light_reduction_' + gid); if (!e) return '';
  return `${f(e.value)}% lower error than the coarse ray-cast (95% CI ${f(e.ci95[0])}–${f(e.ci95[1])}%); lower in ${e.windows_better} of ${e.windows} ${unitWord}.`;
};
const roverRatio = ctx => {
  const e = effect(ctx, 'rover_returned_ratio'); if (!e) return '';
  return `${f(e.value)}× the best rule-based planner (95% CI ${f(e.ci95[0])}–${f(e.ci95[1])}×); ${signed(e.diff_pts, 1)} pts (95% CI ${signed(e.diff_ci95_pts[0], 1)} to ${signed(e.diff_ci95_pts[1], 1)}), higher at ${e.sites_higher} of ${e.sites} sites.`;
};
export const pilotSig = id => ctx => {
  const t = ltest(ctx, id), lt = ctx.stats?.lander_tests; if (!t) return '';
  const raw = praw(t);
  const ci = t.diff_ci95_pts ? ` (95% CI ${signed(t.diff_ci95_pts[0])} to ${signed(t.diff_ci95_pts[1])})` : '';
  return `Pilot − classic: ${signed(t.diff_pts)} pts${ci}. Exact McNemar p = ${pfmt(raw)}${raw < 0.05 ? '' : ' (n.s.)'}; Holm-adjusted over ${lt.n_tests} lander tests: p = ${pfmt(t.p_holm)}${t.p_holm < 0.05 ? '' : ' (n.s.)'}.`;
};
const pilotCallout = id => ctx => { const t = ltest(ctx, id); return t ? `${signed(t.diff_pts)} pts over classic guidance on the same missions.` : ''; };
const fuelCallout = ctx => { const e = effect(ctx, 'pilot_fuel_reduction'); return e ? `${f(e.value)}% less fuel than classic guidance on successful landings (no CI computable: ${e.ci_method.replace(/^no CI computable: /, '')}).` : ''; };
const diffNoTest = (metric, extra) => (ctx, fam) => {
  const rows = fam.rows; const a = rows.find(r => r.ours), b = rows.find(r => !r.ours);
  const g = r => extra ? r.extra_values?.[metric] : r.values?.[metric];
  if (g(a) == null || g(b) == null) return '';
  return `Pilot − classic: ${signed(g(a) - g(b), 1)} pts; no significance test is reported for this subset.`;
};

/* ---------- "How measured" ---------- */
const DESIGN = { hazard: 'windows', light: 'windows and cells', terrain: 'hidden laser pixels', pilot: 'fixed-seed missions', planner: 'worlds and candidates', rover: 'worlds and seeds', rover_learned: 'worlds and seeds' };
const CI = {
  hazard: 'bootstrap over windows', light: 'bootstrap over windows', terrain: 'bootstrap over windows',
  pilot: 'Wilson interval for each rate; paired normal interval for differences', planner: 'bootstrap over the test worlds',
  rover: 'bootstrap over sites (rough with few sites)', rover_learned: 'no interval reported',
};
const STATUS = {
  hazard: 'Set before scoring: test windows fixed by site; the six fresh sites were scored once, under a plan written first.',
  light: 'Set before scoring: test windows fixed by site; the six fresh sites were scored once, under a plan written first. The 6,000-step network had a pre-registered adoption margin and was not adopted.',
  terrain: 'Set before scoring: every classical method was tuned on Site11 only, then scored once on held-out and fresh sites.',
  pilot: 'Set before the run: paired design, benchmark-only unseen maps, and the Site23 exam flown once. Chosen after: the per-map breakdown (the gain comes from Site42).',
  planner: 'Set before training: held-out Site23 split; Site42 has no labels.',
  rover: 'Set before the run: four untouched sites and the pass conditions, written before the one evaluation.',
  rover_learned: 'Set before training: held-out sites; v2 verdict pre-registered. Chosen after: the v3 reference row was designed after these sites had been seen.',
};
const CKPT = { hazard: ['hazard_unet_big (id)', 'hazard_unet_big'], light: ['light_skyline_big (id)', 'light_skyline_big'], terrain: ['terrain_diffusion_mini'], pilot: ['lander_pilot'], rover_learned: ['rover_driver_mini'] };

export function howMeasured(ctx, fam, m, rows) {
  const ours = rows.find(r => r.ours) || rows[0];
  const ck = (CKPT[fam.id] || []).map(n => (ctx.stats?.checkpoints || []).find(c => c.name === n)).filter(Boolean);
  const ckTxt = ck.map(c => `${c.name.replace(' (id)', '')}: ${c.id ? 'id ' + c.id : ''}${c.id && c.sha256 ? ', ' : ''}${c.sha256 ? 'sha256 ' + c.sha256.slice(0, 12) + '…' : ''}`).join('; ');
  let test = '';
  if (fam.id === 'pilot') {
    const ids = m.id === 'success' ? ['unseen_nominal'] : m.id === 'success_harsh' ? ['unseen_harsh'] : m.id === 'success_exam' ? ['site23_nominal'] : m.id === 'site23_harsh_success_pct' ? ['site23_harsh'] : [];
    test = ids.map(id => pilotSig(id)(ctx)).join(' ');
  }
  const items = [
    ['Test set', fam.test_set], ['n', ours?.n], ['Design', `Paired: every row is scored on the same ${DESIGN[fam.id] || 'test items'}.`],
    ['Metric', m.definition], ['Interval', CI[fam.id]], ['Significance', test], ['Set before / after', STATUS[fam.id]],
    ['Checkpoint', ckTxt], ['Source', [fam.source, ours?.source].filter(Boolean).join(' · ')],
  ].filter(([, v]) => v);
  return `<dl>${items.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>`;
}

/* ---------- specs ---------- */
const H = howMeasured;
export const FAMILY_SPECS = {
  hazard: { min: 300, charts: [
    { how: H, metrics: [{ id: 'f1_fresh', tab: 'Fresh sites', title: 'Hazard F1 · fresh sites', callout: hazardRatio('fresh', 1) }, { id: 'f1_heldout', tab: 'Site23+42', title: 'Hazard F1 · held-out Site23+42', callout: hazardRatio('heldout', 2) }] },
    { how: H, metrics: ['recall_fresh'], title: 'Unsafe cells caught · fresh sites' },
    { how: H, metrics: ['false_alarm_fresh'], title: 'False alarms · fresh sites' }] },
  light: { min: 300, charts: [
    { how: H, metrics: [{ id: 'skyline_err', tab: 'Site23+42', title: 'Skyline error · held-out Site23+42', callout: lightRed('heldout', 'windows') }, { id: 'skyline_err_fresh', tab: 'Fresh sites', title: 'Skyline error · fresh sites', callout: lightRed('fresh', 'sites') }] },
    { how: H, metrics: ['hours_err'], title: 'Sunlit-hours error' },
    { how: H, metrics: ['runtime'], title: 'Runtime per window' }] },
  terrain: { min: 420, charts: [
    { how: H, exclude: ['Old gap-fill backup'], metrics: [{ id: 'rmse_heldout', tab: 'Site23+42', title: 'Height RMSE · held-out Site23+42' }, { id: 'rmse_fresh', tab: 'Fresh sites', title: 'Height RMSE · fresh sites' }] },
    { how: H, metrics: ['band_cov'], title: '90% band coverage (target 90%)' },
    { how: H, metrics: ['band_width'], title: 'Band width' }] },
  pilot: { min: 300, charts: [
    { how: H, metrics: [{ id: 'success', tab: 'Normal', title: 'Landing success · unseen maps', sig: pilotSig('unseen_nominal') }, { id: 'success_harsh', tab: 'Harsh', title: 'Landing success · unseen maps, harsh rates', sig: pilotSig('unseen_harsh') }] },
    { how: H, metrics: [{ id: 'success_exam', tab: 'Normal', title: 'Landing success · Site23 final exam', sig: pilotSig('site23_nominal') },
      { extra: 'site23_harsh_success_pct', tab: 'Harsh', title: 'Landing success · Site23 final exam, harsh rates', unit: '%', digits: 1, better: 'higher', definition: 'Share of the Site23 final-exam missions landed with failure rates tripled (flown once).', sig: pilotSig('site23_harsh') }] },
    { how: H, metrics: [{ id: 'engine_out', title: 'Success after engine-out · unseen maps', sig: diffNoTest('engine_out') }] },
    { how: H, metrics: [{ extra: 'fuel_kg_successful_unseen_normal', title: 'Fuel used · successful landings, unseen maps', unit: 'kg', digits: 1, better: 'lower', definition: 'Mean fuel burned on the missions each controller landed (unseen maps, normal rates); the two sets of successful missions differ slightly.', callout: fuelCallout }] }] },
  planner: { min: 300, charts: [
    { how: H, metrics: ['chosen_p'], title: 'Chosen-site mission success · Site23' },
    { how: H, metrics: ['regret'], title: 'Regret vs best candidate' },
    { how: H, metrics: ['brier'], title: 'Brier score (calibration)' }] },
  rover: { min: 420, charts: [
    { how: H, metrics: [{ id: 'return_strict', tab: 'Strict', title: 'Rover returned · strict', callout: roverRatio }, { id: 'return_lenient', tab: 'Lenient', title: 'Rover returned · lenient' }] },
    { how: H, metrics: ['lost_strict'], title: 'Rovers lost · strict' },
    { how: H, metrics: ['reached'], title: 'Reached target' }] },
  rover_learned: { min: 420, charts: [
    { how: H, metrics: ['returned'], title: 'Rover returned · held-out Site23+42' },
    { how: H, metrics: ['reached'], title: 'Reached target' },
    { how: H, metrics: ['died'], title: 'Rovers died' }] },
};

// Strongest real results first (each still with its CI, n, test and "How measured").
export const HEADLINE = [
  { family: 'hazard', spec: FAMILY_SPECS.hazard.charts[0] },
  { family: 'light', spec: FAMILY_SPECS.light.charts[0] },
  { family: 'rover', spec: { how: H, metrics: [{ id: 'return_strict', tab: 'Returned', title: 'Rover returned · strict, untouched sites', callout: roverRatio }, { id: 'lost_strict', tab: 'Lost', title: 'Rovers lost · strict, untouched sites' }] } },
  { family: 'pilot', spec: { how: H, metrics: [{ id: 'success', tab: 'Normal', title: 'Landing success · unseen maps', callout: pilotCallout('unseen_nominal'), sig: pilotSig('unseen_nominal') }, { id: 'success_harsh', tab: 'Harsh', title: 'Landing success · unseen maps, harsh rates', callout: pilotCallout('unseen_harsh'), sig: pilotSig('unseen_harsh') }] } },
  { family: 'pilot', spec: FAMILY_SPECS.pilot.charts[3] },
];

// Compact sets for the product pages and the home teaser.
export const HOME_SPECS = [
  { family: 'hazard', spec: { how: H, metrics: [{ id: 'f1_fresh', title: 'Hazard F1 · fresh sites', callout: hazardRatio('fresh', 1) }] } },
  { family: 'rover', spec: { how: H, metrics: [{ id: 'return_strict', title: 'Rover returned · strict', callout: roverRatio }], only: ['Rover brain v3.1', 'Rover brain v3', 'Planner: naive'] } },
  { family: 'pilot', spec: { how: H, metrics: [{ id: 'success', title: 'Landing success · unseen maps', sig: pilotSig('unseen_nominal') }] } },
];
