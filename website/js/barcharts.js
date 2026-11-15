// Vertical bar charts (launch-page style) for benchmark and projection numbers. No numbers live here: every bar comes
// from assets/data/benchmarks.json, assets/data/projections.json or assets/data/stats.json (see js/benchspecs.js for
// the chart specs and the callout / significance / "How measured" text built from those files).
//
//   import { mountBarCharts } from './js/barcharts.js';
//   mountBarCharts(el, 'pilot', [ spec, ... ], { min: 300, ink: false });
//
//   spec = { title?, metrics: [id | { id, tab, title, extra?, unit?, digits?, better?, definition? }],
//            only?: [row-name prefix], exclude?: [row-name prefix],
//            callout?(ctx, fam, m) -> string, sig?(ctx, fam, m) -> string, how?(ctx, fam, m, rows) -> html }
//     metric.extra reads row.extra_values[extra] instead of row.values[id] (then unit/digits/better/definition are given).
//     Excluded rows are named in the caption with their value ("Not shown (off scale): ..."), never silently dropped.
//   ctx = { bench, stats, proj } (loaded once by loadCtx()).
//
//   Lower-level: barChart(el, views, opts), views = [{ tab, title, unit, digits, better, caption, callout, sig, n, how,
//   bars: [{ label, sub, value, ci: [lo, hi], kind: 'ours' | 'ours2' | 'base' | 'proj', full, digits }] }].

const cache = {};
export const loadJSON = u => (cache[u] ||= fetch(u).then(r => { if (!r.ok) throw new Error(`${u} ${r.status}`); return r.json(); }));
export function loadCtx() {
  return Promise.all([loadJSON('assets/data/benchmarks.json'), loadJSON('assets/data/stats.json').catch(() => null),
    loadJSON('assets/data/projections.json').catch(() => null)]).then(([bench, stats, proj]) => ({ bench, stats, proj }));
}
const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
export const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tight = u => u === '%' || u === '°';
export const fmtVal = (v, digits, unit) => {
  if (v == null || Number.isNaN(+v)) return '—';
  const s = (+v).toLocaleString('en-US', { minimumFractionDigits: digits ?? 0, maximumFractionDigits: digits ?? 0 });
  return !unit ? s : tight(unit) ? s + unit : `${s} ${unit}`;
};
let uid = 0;

function shortName(name) { return String(name).split(' (')[0]; }
function subLabel(r) {
  if (/FAILED/.test(r.name)) return 'failed';
  if (/not adopted/i.test(r.name)) return 'ours · not adopted';
  return r.ours ? 'ours' : '';
}

/* ---------- generic chart ---------- */
export function barChart(el, views, opts = {}) {
  const id = ++uid;
  const fig = document.createElement('figure');
  fig.className = 'bc' + (opts.ink ? ' bc-ink' : '');
  const tabs = views.length > 1 ? `<div class="bc-tog" role="group" aria-label="Switch view">${views.map((v, i) =>
    `<button type="button" aria-pressed="${i === 0}" data-i="${i}">${esc(v.tab)}</button>`).join('')}</div>` : '';
  fig.innerHTML = `<div class="bc-top"><h4 class="bc-title" id="bc-t${id}"></h4>${tabs}</div>
    <p class="bc-callout"></p>
    <div class="bc-plot"><div class="bc-bars" role="img" aria-labelledby="bc-t${id}"></div></div>
    <p class="bc-sig"></p><p class="bc-n"></p>
    <figcaption class="bc-cap"></figcaption>
    <details class="bc-how"><summary>How measured</summary><div class="bc-how-body"></div></details>`;
  el.appendChild(fig);
  const $ = s => fig.querySelector(s);
  const bars = $('.bc-bars'), title = $('.bc-title');
  let seen = false;
  const setText = (node, txt) => { node.textContent = txt || ''; node.hidden = !txt; };

  function render(v) {
    const lower = v.better === 'lower';
    title.textContent = v.title + (lower && !/lower is better/.test(v.title) ? ' · lower is better' : '');
    const top = Math.max(...v.bars.flatMap(b => [+b.value || 0, ...(b.ci ? [+b.ci[1]] : [])])) || 1;
    const scale = top * 1.12;
    const pct = x => Math.max(0, Math.min(100, (+x / scale) * 100));
    fig.classList.toggle('bc-many', Math.max(...views.map(x => x.bars.length)) > 6);
    bars.style.setProperty('--n', v.bars.length);
    const f = (x, b) => fmtVal(x, b.digits ?? v.digits, v.unit);
    bars.setAttribute('aria-label', `${title.textContent}: ` + v.bars.map(b => `${b.full || b.label}${b.kind === 'proj' ? ' (projected, not measured)' : ''} ${f(b.value, b)}${b.ci ? ` (95% interval ${f(b.ci[0], b)} to ${f(b.ci[1], b)})` : ''}`).join('; '));
    bars.innerHTML = v.bars.map(b => {
      const h = pct(b.value), hi = b.ci ? pct(b.ci[1]) : h, lo = b.ci ? pct(b.ci[0]) : h;
      const lab = Math.max(h, hi);
      return `<div class="bc-col bc-${b.kind || 'base'}" title="${esc(b.full || b.label)}">
        <div class="bc-area">
          <span class="bc-bar" style="height:${h.toFixed(2)}%"></span>
          ${b.ci ? `<span class="bc-whisk" style="bottom:${lo.toFixed(2)}%;height:${Math.max(0, hi - lo).toFixed(2)}%"></span>` : ''}
          <span class="bc-val" style="bottom:${lab.toFixed(2)}%">${esc(f(b.value, b))}</span>
        </div>
        <span class="bc-name">${esc(b.label)}</span>${b.sub ? `<span class="bc-sub">${esc(b.sub)}</span>` : ''}
      </div>`;
    }).join('');
    setText($('.bc-callout'), v.callout);
    setText($('.bc-sig'), v.sig);
    setText($('.bc-n'), v.n ? `n: ${v.n}` : '');
    setText($('.bc-cap'), v.caption);
    const how = $('.bc-how');
    if (v.how) { how.hidden = false; $('.bc-how-body').innerHTML = v.how; } else how.hidden = true;
  }
  function show(i, animate) {
    fig.querySelectorAll('.bc-tog button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.i === i)));
    if (animate && !reduce()) { fig.classList.remove('bc-in'); render(views[i]); void fig.offsetWidth; fig.classList.add('bc-in'); }
    else render(views[i]);
  }
  fig.querySelectorAll('.bc-tog button').forEach(b => b.addEventListener('click', () => show(+b.dataset.i, true)));
  show(0, false);
  if (reduce() || !('IntersectionObserver' in window)) fig.classList.add('bc-in', 'bc-still');
  else {
    const io = new IntersectionObserver(es => { if (!seen && es.some(e => e.isIntersecting)) { seen = true; fig.classList.add('bc-in'); io.disconnect(); } }, { threshold: 0.2 });
    io.observe(fig);
  }
  return fig;
}

/* ---------- benchmark family charts ---------- */
export function familyViews(fam, spec, ctx = {}) {
  const mets = spec.metrics.map(m => typeof m === 'string' ? { id: m } : m);
  const excl = spec.exclude || [], only = spec.only;
  return mets.map(mm => {
    const m = mm.extra ? { id: mm.extra, label: mm.title, unit: mm.unit || '', digits: mm.digits, better: mm.better, definition: mm.definition || '' }
      : fam.metrics.find(x => x.id === mm.id);
    if (!m) return null;
    const val = r => mm.extra ? r.extra_values?.[mm.extra] : r.values?.[m.id];
    const rows = fam.rows.filter(r => val(r) != null && (!only || only.some(p => r.name.startsWith(p))));
    const shown = rows.filter(r => !excl.some(p => r.name.startsWith(p)));
    const hidden = rows.filter(r => excl.some(p => r.name.startsWith(p)));
    let firstOurs = true;
    const bars = shown.map(r => {
      let kind = 'base';
      if (r.ours) { kind = firstOurs ? 'ours' : 'ours2'; firstOurs = false; }
      return { label: shortName(r.name), sub: subLabel(r), full: r.name, value: val(r), ci: mm.extra ? null : r.ci95?.[m.id], kind };
    });
    const unit = m.unit || '';
    const anyCi = bars.some(b => b.ci);
    const allRows = fam.rows.filter(r => val(r) != null);
    const part = only && rows.length < allRows.length ? ` Showing ${rows.length} of ${allRows.length} rows; the full comparison is on the benchmarks page.` : '';
    const off = hidden.length ? ` Not shown (off scale): ${hidden.map(r => `${shortName(r.name)} ${fmtVal(val(r), m.digits, unit)}`).join(', ')}.` : '';
    const ours = shown.find(r => r.ours) || shown[0];
    const call = fn => { try { return fn ? fn(ctx, fam, m, shown) : ''; } catch (e) { console.warn('barcharts', e); return ''; } };
    return {
      tab: mm.tab || m.label, title: mm.title || spec.title || m.label, unit, digits: m.digits, better: m.better,
      caption: spec.caption === false ? '' : `${m.definition}${anyCi ? ' Whiskers: 95% interval.' : ''}${off}${part}`, bars,
      callout: call(mm.callout || spec.callout), sig: call(mm.sig || spec.sig), n: ours?.n || '',
      how: call(spec.how),
    };
  }).filter(Boolean);
}

export function mountBarCharts(el, familyId, charts, opts = {}) {
  if (!el) return Promise.resolve();
  el.classList.add('bc-row');
  if (opts.min) el.style.setProperty('--bc-min', opts.min + 'px');
  return (opts.ctx ? Promise.resolve(opts.ctx) : loadCtx()).then(ctx => {
    const fam = (ctx.bench.families || []).concat(ctx.extraFamilies || []).find(f => f.id === familyId);
    if (!fam) throw new Error('no family ' + familyId);
    for (const spec of charts) { const views = familyViews(fam, spec, ctx); if (views.length) barChart(el, views, opts); }
  }).catch(e => { console.warn('barcharts', e); el.innerHTML = '<p class="src">Charts could not be loaded.</p>'; });
}

/* ---------- measured -> projected (projections.json) ---------- */
export function projectionViews(p, views, label) {
  return views.map(v => {
    const p2 = p.projected?.['2x_training']?.[v.key], p5 = p.projected?.['5x_training']?.[v.key];
    if (v.measured == null || !p2 || !p5) return null;
    const bars = [
      { label: 'Measured', sub: 'now', value: v.measured, digits: v.measuredDigits, kind: 'ours' },
      ...(v.extra || []).filter(x => x.at === 'mid'),
      { label: '2x training', sub: 'projected', value: p2.point, ci: p2.band95, kind: 'proj' },
      { label: '5x training', sub: 'projected', value: p5.point, ci: p5.band95, kind: 'proj' },
      ...(v.extra || []).filter(x => x.at !== 'mid'),
    ];
    return { tab: v.tab, title: v.title, unit: v.unit, digits: v.digits, better: v.better, bars,
      caption: `${label || 'Projected — not measured'}: hatched bars are curve-fit extrapolations of our training logs; whiskers are the 95% band.${v.note ? ' ' + v.note : ''}` };
  }).filter(Boolean);
}
