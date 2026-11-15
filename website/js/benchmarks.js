// Benchmarks page renderer (research/benchmarks.html). All numbers come from assets/data/benchmarks.json and
// assets/data/projections.json; this file only holds labels and which projected metric sits next to which benchmark column.
//
// benchmarks.json (copy of data/handoff_macmini_1115/benchmarks_final.json):
// {
//   "placeholder": false,                // true = sample figures, shows one muted notice at the top of the page
//   "meta": { "provenance", "source", "frame", "units", "built", "note" },
//   "families": [ {
//       "id", "name", "lede", "chart_metric", "test_set", "source",
//       "metrics": [ { "id", "label", "unit", "better": "higher" | "lower", "digits", "definition" } ],
//       "rows": [ { "name", "ours", "values": {metric: v}, "ci95": {metric: [lo, hi]}, "extra_values": {key: v},
//                   "n": "sample size text", "source", "note" } ]   // best value per metric is highlighted automatically
//   } ],
//   "not_benchmarked": [ { "family", "why" } ]
// }
// projections.json (copy of data/handoff_macmini_1115/projections.json): label, method, warning,
//   training_record[] {model, machine, when, hours, amount, start, best, source},
//   projections[] {family, model, label, metric_fitted, fit{n_points, residual_rmse}, measured_validation_now{},
//                  projected{"2x_training"|"5x_training": {metric: {point, band95}}}, caveats[], out_of_sample_check{}},
//   targets[] {family, model, label, target, why_no_projection}

// mission_bench.json (optional; read only if stats.json optional_files.mission_bench is true, so a missing file causes
// no request and no console error). Expected schema, any part may be missing:
// { "meta": {provenance, source, ...}, "families": [ <same family schema as benchmarks.json> ],
//   "headline": [ { "family": "<id>", "metrics": ["<metric id>", ...], "title": "..." } ],   // optional headline charts
//   "tests": [ { "id", "label", "p", "registered": "pre-specified" | "post hoc", "kind" } ] }      // joins the Holm family
// Anything that does not match is ignored silently.

import { mountBarCharts, barChart, projectionViews, loadCtx, loadJSON } from './barcharts.js';
import { FAMILY_SPECS, HEADLINE, howMeasured } from './benchspecs.js';
import { gateCard, renderLimits, renderMethodology, renderReproduce, renderDownloads } from './benchmethods.js';

const PASTELS = ['--sage', '--peach', '--lavender', '--sand'];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = (v, d) => (+v).toLocaleString('en-US', { minimumFractionDigits: d ?? 0, maximumFractionDigits: d ?? 0 });
const fmt = (v, m) => (v == null || Number.isNaN(+v)) ? '—' : num(v, m.digits);
const tight = u => u === '%' || u === '°';
const withUnit = (v, m) => { const s = fmt(v, m); return s === '—' || !m.unit ? s : (tight(m.unit) ? s + m.unit : `${s} ${m.unit}`); };
const range = (ci, m) => `${fmt(ci[0], m)}–${fmt(ci[1], m)}${m.unit && tight(m.unit) ? m.unit : ''}`;
const here = () => esc(location.href.split('#')[0]);

// Readable labels for the pilot's extra_values (labels only; values come from the JSON).
const EXTRA = {
  miss_m_successful_unseen_normal: ['Miss distance, successful landings', 'm'],
  fuel_kg_successful_unseen_normal: ['Fuel used, successful landings', 'kg'],
  touchdown_vz_m_s_unseen_normal: ['Touchdown vertical speed', 'm/s'],
  nav_drift_success_unseen_normal_pct: ['Success with navigation drift', '%'],
  camera_blind_success_unseen_normal_pct: ['Success with a blind camera', '%'],
  site23_harsh_success_pct: ['Site23 exam, harsh rates', '%'],
};

function best(rows, m) {
  const vals = rows.map(r => r.values?.[m.id]).filter(v => v != null && !Number.isNaN(+v)).map(Number);
  if (!vals.length) return null;
  return m.better === 'lower' ? Math.min(...vals) : Math.max(...vals);
}

function table(f) {
  const bests = Object.fromEntries(f.metrics.map(m => [m.id, best(f.rows, m)]));
  const head = f.metrics.map(m => `<th scope="col">${esc(m.label)}${m.unit && !tight(m.unit) ? ` <span class="bm-unit">(${esc(m.unit)})</span>` : ''}<span class="bm-dir">${m.better === 'lower' ? 'lower is better' : 'higher is better'}</span></th>`).join('');
  const body = f.rows.map(r => {
    const cells = f.metrics.map(m => {
      const v = r.values?.[m.id], ci = r.ci95?.[m.id];
      const isBest = v != null && bests[m.id] != null && +v === bests[m.id];
      const ciTxt = ci && v != null ? `<span class="bm-ci" title="95% interval">${esc(range(ci, m))}</span>` : '';
      return `<td class="${isBest ? 'best' : ''}">${esc(withUnit(v, m))}${isBest ? '<span class="sr-only"> (best)</span>' : ''}${ciTxt}</td>`;
    }).join('');
    const failed = /FAILED/.test(r.name);
    return `<tr class="${r.ours ? 'ours' : ''}${failed ? ' failed' : ''}"><th scope="row">${esc(r.name)}</th>${cells}</tr>`;
  }).join('');
  return `<div class="bm-table-wrap"><table class="bm-table"><thead><tr><th scope="col">Model</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function rowNotes(f) {
  const items = f.rows.map(r => {
    const extra = r.extra_values ? `<ul class="bm-extra">${Object.entries(r.extra_values).map(([k, v]) => {
      const [lab, u] = EXTRA[k] || [k.replace(/_/g, ' '), ''];
      return `<li>${esc(lab)}: <span class="mono">${esc(String(v))}${u ? (u === '%' ? '%' : ' ' + esc(u)) : ''}</span></li>`;
    }).join('')}</ul>` : '';
    return `<li><b>${esc(r.name)}</b>
      ${r.note ? `<p>${esc(r.note)}</p>` : ''}${extra}
      ${r.n ? `<p class="bm-n">n: ${esc(r.n)}</p>` : ''}
      ${r.source ? `<p class="bm-n">Source: ${esc(r.source)}</p>` : ''}</li>`;
  }).join('');
  return `<details class="bm-details"><summary>Notes, sample sizes and sources per row</summary><ul class="bm-rownotes">${items}</ul></details>`;
}

function family(f, i) {
  const defs = f.metrics.map(m => `<div><dt>${esc(m.label)}</dt><dd>${esc(m.definition)}</dd></div>`).join('');
  return `<section class="bm-family" id="${esc(f.id)}" style="--tint: var(${PASTELS[i % PASTELS.length]})">
    <header class="bm-fhead"><h2>${esc(f.name)}</h2><p>${esc(f.lede)}</p></header>
    ${f.test_set ? `<p class="bm-testset"><b>Test set.</b> ${esc(f.test_set)}</p>` : ''}
    <div class="bm-charts" data-fam="${esc(f.id)}"></div>
    <details class="bm-details bm-full"><summary>Full table</summary>
      ${table(f)}
      <p class="bm-cinote">Small grey ranges under a value are 95% intervals. The best value in each column is highlighted, including trivial baselines.</p>
      <dl class="bm-defs">${defs}</dl>
    </details>
    ${rowNotes(f)}
    ${f.source ? `<p class="src">Source: ${esc(f.source)}</p>` : ''}
  </section>`;
}

function notBenchmarked(list) {
  if (!list?.length) return '';
  return `<section class="bm-family bm-nb" id="not-benchmarked">
    <header class="bm-fhead"><h2>Not benchmarked</h2><p>What has no benchmark yet, and why.</p></header>
    <ul class="bm-nblist">${list.map(x => `<li><b>${esc(x.family)}</b><p>${esc(x.why)}</p></li>`).join('')}</ul>
  </section>`;
}

/* ---------- two days of training ---------- */
function timeline(rec) {
  const maxH = Math.max(...rec.map(r => +r.hours || 0)) || 1;
  const dash = s => s == null || s === '-' || s === '';
  return rec.map(r => {
    const h = +r.hours;
    const hours = r.hours == null ? 'hours not logged' : `${num(h, 2)} h`;
    const bar = r.hours ? `<span class="tl-bar" aria-hidden="true"><i style="width:${(h / maxH * 100).toFixed(1)}%"></i></span>` : '';
    return `<li class="tl-item">
      <div class="tl-top"><h3>${esc(r.model)}</h3><span class="tl-hours">${esc(hours)}</span></div>
      ${bar}
      <p class="tl-where">${esc([r.machine, r.when].filter(x => !dash(x)).join(' · '))}</p>
      <p class="tl-amount">${esc(r.amount)}</p>
      ${dash(r.start) && dash(r.best) ? '' : `<dl class="tl-sb"><div><dt>Start</dt><dd>${esc(r.start)}</dd></div><div><dt>Best</dt><dd>${esc(r.best)}</dd></div></dl>`}
      <p class="bm-n">Source: ${esc(r.source)}</p>
    </li>`;
  }).join('');
}

/* ---------- measured -> projected ---------- */
// Which projected metric goes with which measured value. measured() reads projections.json; bench names a
// benchmarks.json column (first "ours" row of that family, or the row whose name matches benchRow).
const PROJ = {
  hazard: { chart: 1, metrics: [
    { key: 'validation_f1', tab: 'Validation', label: 'F1, validation (Site11)', better: 'higher', digits: 4, measured: p => p.measured_validation_now?.f1 },
    { key: 'heldout_site23_42_f1', tab: 'Site23+42', label: 'F1, Site23+42', better: 'higher', digits: 4, bench: 'f1_heldout' },
    { key: 'fresh_f1', tab: 'Fresh', label: 'F1, fresh sites', better: 'higher', digits: 4, bench: 'f1_fresh' }] },
  light: { chart: 1, metrics: [
    { key: 'validation', tab: 'Validation', label: 'Skyline error, validation (Site11)', unit: '°', better: 'lower', digits: 4, measured: p => p.measured_validation_now?.mae_deg },
    { key: 'heldout_site23_42_mae_deg', tab: 'Site23+42', label: 'Skyline error, Site23+42', unit: '°', better: 'lower', digits: 4, bench: 'skyline_err' },
    { key: 'fresh_mae_deg', tab: 'Fresh', label: 'Skyline error, fresh', unit: '°', better: 'lower', digits: 4, bench: 'skyline_err_fresh' }] },
  terrain: { chart: 1, ref: { bench: 'rmse_heldout', row: /^Kriging/, label: 'Kriging, measured' }, metrics: [
    { key: 'validation', tab: 'Validation', label: 'Denoising MSE, validation (Site11)', better: 'lower', digits: 5, measured: p => p.measured_validation_now?.val_mse },
    { key: 'heldout_rmse_m_if_rmse_scaled_like_sqrt_val_mse', tab: 'Site23+42', label: 'Height RMSE, Site23+42 (if RMSE scales like √MSE)', unit: 'm', better: 'lower', digits: 2, bench: 'rmse_heldout', benchRow: /Diffusion/ }] },
};

function benchValue(bench, famId, metric, rowRe) {
  const f = bench.families?.find(x => x.id === famId);
  const row = f?.rows.find(r => rowRe ? rowRe.test(r.name) : r.ours);
  const m = f?.metrics.find(x => x.id === metric);
  return row && m ? { v: row.values?.[metric], digits: m.digits } : null;
}

// Bar views for one projection card: one view per metric in PROJ (measured value + 2x/5x projected), plus the
// measured extras the file records (light: the real 1.84x resume; terrain: kriging, the presented estimator).
function projBarViews(p, bench) {
  const spec = PROJ[p.family];
  if (!spec) return [];
  return spec.metrics.map((mm, k) => {
    const bv = mm.measured ? null : benchValue(bench, p.family, mm.bench, mm.benchRow);
    const mv = mm.measured ? mm.measured(p) : bv?.v;
    const extra = [];
    const oos = p.out_of_sample_check;
    if (oos && p.family === 'light') {
      const v = mm.key === 'validation' ? oos.measured_validation_best : mm.key === 'heldout_site23_42_mae_deg' ? oos.measured_heldout : null;
      if (v != null) extra.push({ label: '1.84x, resumed', sub: 'measured', value: v, kind: 'ours2', at: 'mid' });
    }
    if (spec.ref && k === spec.chart) {
      const r = benchValue(bench, p.family, spec.ref.bench, spec.ref.row);
      if (r?.v != null) extra.push({ label: 'Kriging', sub: 'measured, presented', value: r.v, kind: 'base' });
    }
    return { tab: mm.tab, title: mm.label, unit: mm.unit || '', digits: mm.digits, better: mm.better, measured: mv, measuredDigits: bv?.digits, key: mm.key, extra, first: k === spec.chart };
  }).sort((a, b) => b.first - a.first);
}

function projCard(p, bench, label, i) {
  const spec = PROJ[p.family];
  if (!spec) return '';
  const meas = spec.metrics.map(mm => {
    if (mm.measured) return { v: mm.measured(p), digits: mm.digits };
    return benchValue(bench, p.family, mm.bench, mm.benchRow);
  });
  const cell = (pr, mm) => pr ? `<td class="pj-pcell">${esc(num(pr.point, mm.digits))}${mm.unit && tight(mm.unit) ? esc(mm.unit) : ''}<span class="bm-ci">${esc(num(pr.band95[0], mm.digits))}–${esc(num(pr.band95[1], mm.digits))}</span></td>` : '<td class="pj-pcell">—</td>';
  const rows = spec.metrics.map((mm, k) => {
    const mv = meas[k];
    const mtxt = mv?.v == null ? '—' : num(mv.v, mv.digits) + (mm.unit && tight(mm.unit) ? mm.unit : '');
    return `<tr><th scope="row">${esc(mm.label)}${mm.unit && !tight(mm.unit) ? ` <span class="bm-unit">(${esc(mm.unit)})</span>` : ''}<span class="bm-dir">${mm.better === 'lower' ? 'lower is better' : 'higher is better'}</span></th>
      <td class="pj-mcell">${esc(mtxt)}</td>${cell(p.projected['2x_training']?.[mm.key], mm)}${cell(p.projected['5x_training']?.[mm.key], mm)}</tr>`;
  }).join('');
  const oos = p.out_of_sample_check;
  return `<article class="pj-card">
    <header><h4>${esc(p.model)}</h4><p class="pj-label">${esc(p.label || label)}</p></header>
    <div class="pj-body"><div class="bm-table-wrap"><table class="bm-table pj-table">
      <thead><tr><th scope="col">Metric</th><th scope="col">Measured<span class="bm-dir">now</span></th><th scope="col" class="pj-ph">2x training<span class="bm-dir">projected</span></th><th scope="col" class="pj-ph">5x training<span class="bm-dir">projected</span></th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div class="pj-bars" data-pj="${esc(p.family)}"></div></div>
    <p class="bm-n">Fitted metric: ${esc(p.metric_fitted)}; ${esc(String(p.fit?.n_points ?? '—'))} logged points.</p>
    ${oos ? `<p class="pj-check"><b>Checked once.</b> ${esc(oos.what)} Measured validation ${esc(String(oos.measured_validation_best))}°, held-out ${esc(String(oos.measured_heldout))}°. ${esc(oos.reading)}</p>` : ''}
    ${p.caveats?.length ? `<ul class="pj-cav">${p.caveats.map(c => `<li>${esc(c)}</li>`).join('')}</ul>` : ''}
  </article>`;
}

function targetCard(t) {
  return `<article class="tg-card">
    <header><h4>${esc(t.model)}</h4><p class="tg-label">Target</p></header>
    <p>${esc(t.target)}</p>
    <p class="bm-n"><b>Why no projection:</b> ${esc(t.why_no_projection)}</p>
  </article>`;
}

function defaultSpec(ctx, famId) {   // mission_bench families: first three metrics, how-measured from the family itself
  const f = (ctx.extraFamilies || []).find(x => x.id === famId);
  if (!f) return null;
  return { min: 300, charts: (f.metrics || []).slice(0, 3).map(m => ({ metrics: [m.id], how: howMeasured })) };
}

function loadMissionBench(ctx) {
  if (!ctx.stats?.optional_files?.mission_bench) return Promise.resolve(null);
  return loadJSON('assets/data/mission_bench.json').then(mb => (mb && typeof mb === 'object' ? mb : null)).catch(() => null);
}

export function mountBenchmarks() {
  const $ = id => document.getElementById(id);
  const root = $('bm-root'), notice = $('bm-notice'), toc = $('bm-toc');
  return loadCtx().then(ctx => loadMissionBench(ctx).then(mb => {
    const okFam = f => f && typeof f === 'object' && f.id && Array.isArray(f.metrics) && Array.isArray(f.rows);
    ctx.missionBench = mb;
    ctx.extraFamilies = (mb?.families || []).filter(okFam);
    return ctx;
  })).then(ctx => {
    const d = ctx.bench, pj = ctx.proj;
    const fams = d.families || [];
    if (notice) notice.hidden = !d.placeholder;
    if ($('bm-meta') && d.meta?.note) $('bm-meta').textContent = d.meta.note;
    const links = [['headline', 'Headline results'], ...fams.map(f => [f.id, f.name])];
    if (ctx.extraFamilies.length) links.push(['mission-bench', 'Mission rehearsal']);
    if (d.not_benchmarked?.length) links.push(['not-benchmarked', 'Not benchmarked']);
    links.push(['methodology', 'Methodology'], ['reproduce', 'Reproduce'], ['downloads', 'Raw data']);
    if (pj) links.push(['training', 'Two days of training'], ['projected', 'Where we\'re going']);
    links.push(['limits', 'Limits']);
    if (toc) toc.innerHTML = links.map(([id, n]) => `<a href="${here()}#${esc(id)}">${esc(n)}</a>`).join('');
    root.innerHTML = fams.map(family).join('') + (ctx.extraFamilies.length ? `<h2 class="bm-sub" id="mission-bench">Mission rehearsal</h2>` + ctx.extraFamilies.map((f, i) => family(f, i + fams.length)).join('') : '') + notBenchmarked(d.not_benchmarked);
    // headline row: strongest real results first (mission_bench headline specs appended if present)
    const hl = $('hl-root');
    if (hl) {
      hl.classList.add('bc-row');
      const extraHl = (ctx.missionBench?.headline || []).filter(h => h && h.family && Array.isArray(h.metrics)).map(h => ({ family: h.family, spec: { metrics: h.metrics, title: h.title, how: howMeasured } }));
      Promise.all(HEADLINE.concat(extraHl).map(h => mountBarCharts(hl, h.family, [h.spec], { ctx }))).then(() => hl.insertAdjacentHTML('beforeend', gateCard(ctx)));
    }
    const fill = (id, fn) => { const el = $(id); if (el) { try { el.innerHTML = fn(ctx); } catch (e) { console.warn('benchmarks', id, e); } } };
    fill('limits-root', renderLimits);
    fill('meth-root', renderMethodology);
    fill('repro-root', renderReproduce);
    fill('dl-root', renderDownloads);
    root.querySelectorAll('.bm-charts').forEach(el => { const c = FAMILY_SPECS[el.dataset.fam] || defaultSpec(ctx, el.dataset.fam); if (c) mountBarCharts(el, el.dataset.fam, c.charts, { min: c.min, ctx }); });
    if (pj) {
      $('tl-root').innerHTML = timeline(pj.training_record || []);
      $('pj-method').innerHTML = `<b>${esc(pj.label)}.</b> ${esc(pj.method)} ${esc(pj.warning)}`;
      $('pj-root').innerHTML = (pj.projections || []).map((p, i) => projCard(p, d, pj.label, i)).join('');
      (pj.projections || []).forEach(p => {
        const el = $('pj-root').querySelector(`.pj-bars[data-pj="${p.family}"]`);
        const views = el && projectionViews(p, projBarViews(p, d), pj.label);
        if (views?.length) barChart(el, views);
      });
      $('tg-root').innerHTML = (pj.targets || []).map(targetCard).join('');
    } else {
      for (const id of ['training', 'projected']) $(id).hidden = true;
    }
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }).catch(e => { console.warn('benchmarks', e); root.innerHTML = '<p class="src">Benchmarks could not be loaded.</p>'; });
}

mountBenchmarks();
