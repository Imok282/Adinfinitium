// Results section, rendered from ONE file: assets/data/results.json (provenance: modelled).
// To change a number, edit that JSON and reload; nothing here needs to change.
// Card types: "paired" (classic vs pilot bars + delta), "kpis" (big numbers), "pending" (a slot waiting for a result).

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const rich = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');           // **bold** only
const num = (v, d = 2) => Number(v).toFixed(d);
const signed = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const pFmt = p => (p < 0.001 ? '< 0.001' : String(Math.round(p * 1000) / 1000));

function paired(c, axis) {
  const bar = (side, cls) => {
    const w = Math.max(0, Math.min(100, (side.success_pct - axis.min) / (axis.max - axis.min) * 100));
    return `<div class="bar${cls}"><span class="bl">${esc(side.label)}</span><span class="bv">${num(side.success_pct)} %</span><i style="--w:${w.toFixed(2)}%"></i></div>`;
  };
  const ci = c.ci95_points ? ` · 95 % CI [${signed(c.ci95_points[0])}, ${signed(c.ci95_points[1])}]` : '';
  const p = c.p_value != null ? ` · ${esc(c.test || 'p')} p = ${pFmt(c.p_value)}` : '';
  const sig = c.significant === false ? ' — <b>not significant</b>' : '';
  return `
    <div class="bars" data-axis="${esc(axis.label || '')}">${bar(c.baseline, '')}${bar(c.candidate, ' pilot')}</div>
    <p class="delta"><b>${signed(c.diff_points, 2)} points</b>${ci}${p}${sig}</p>`;
}

function kpis(list) {
  return `<div class="kpis">${(list || []).map(k => `<div><b>${esc(k.value)}</b><span>${esc(k.label)}</span></div>`).join('')}</div>`;
}

function pending(c) {
  if (c.value == null) return `<p class="pending-slot">${esc(c.pending_text || 'Result pending')}</p>`;
  return `<div class="kpis"><div><b>${esc(c.value)}</b><span>${esc(c.detail || '')}</span></div></div>`;
}

export function renderResults(root, data, opts = {}) {
  const full = !!opts.full;   // results.html: every card (paired bars, CIs, p-values) and the full tiles
  const axis = data.bar_axis || { min: 94, max: 100, label: 'bar axis 94–100 %' };
  const sec = root.closest('section') || document;
  const intro = sec.querySelector('[data-results-intro]');
  if (intro && data.intro) intro.innerHTML = rich(data.intro);
  // hero micro-stats and the big stat row (both optional in the JSON)
  const hero = document.querySelector('[data-hero-stats]');
  if (hero && data.hero_stats) hero.innerHTML = data.hero_stats.map(k => `<li><b>${esc(k.value)}</b> ${esc(k.label)}</li>`).join('');
  const row = sec.querySelector('[data-results-stats]');
  // pastel bento tiles: one real stat each (value + title); one muted source line for the whole block
  const tints = ['sage', 'peach', 'lavender', 'sand'];
  if (row && data.headline_stats) row.innerHTML = data.headline_stats.map((k, i) =>
    `<li class="${tints[i % 4]}"><b>${esc(k.value)}</b><span>${esc(k.title || k.label || '')}</span>` +
    (full && k.body ? `<p>${esc(k.body)}</p>` : '') + (full && k.source ? `<code>${esc(k.source)}</code>` : '') + `</li>`).join('');
  // the "ours" panel of the comparison strip (first headline stat)
  const scope = full ? document : sec;
  const big = scope.querySelector('[data-ours-big]'), txt = scope.querySelector('[data-ours-txt]'), h0 = (data.headline_stats || [])[0];
  if (big && h0) { big.textContent = big.dataset.final = h0.value; const m = String(h0.value).match(/[\d.]+/); if (m) { big.dataset.count = m[0]; big.dataset.unit = String(h0.value).slice(m.index + m[0].length); } }
  if (txt && h0) txt.textContent = `${h0.title} — ${data.hero_stats?.[2]?.value || ''} paired missions per condition on maps it never saw.`.replace(' —  paired', ' —');
  // only slots still waiting for a result are shown as tiles (Site 23 final exam); the detailed cards live in the JSON
  if (full) {
    root.innerHTML = (data.cards || []).map((c, i) => {
      const body = c.type === 'paired' ? paired(c, axis) : c.type === 'kpis' ? kpis(c.kpis) : pending(c);
      const isPending = c.type === 'pending' && c.value == null;
      return `<article class="res-card ${isPending ? 'pending' : tints[i % 4]}" id="res-${esc(c.id)}">
        <h3>${esc(c.title)}</h3>${body}
        ${c.note ? `<p class="small">${rich(c.note)}</p>` : ''}
        ${c.source ? `<p class="src">Source: ${esc(c.source)}</p>` : ''}
      </article>`;
    }).join('');
    requestAnimationFrame(() => root.querySelectorAll('.bars').forEach(b => b.classList.add('in')));
    const foot = document.querySelector('[data-results-foot]');
    if (foot && data.footnote) foot.textContent = data.footnote;
    return;
  }
  // compact mode: pending slots, plus the Site 23 final exam once it has a result
  root.innerHTML = (data.cards || []).filter(c => c.type === 'pending' || c.id === 'site23-exam').map(c => {
    if (c.type !== 'pending') {
      const k = (c.kpis || [])[0] || {};
      return `<article class="res-card" id="res-${esc(c.id)}"><b>${esc(k.value)}</b><span>${esc(c.title)} · ${esc(k.label)}${c.kpis && c.kpis[1] ? ' vs ' + esc(c.kpis[1].value) + ' classic (tie)' : ''}</span></article>`;
    }
    const v = c.value == null ? esc(c.pending_text || 'Result pending') : esc(c.value);
    return `<article class="res-card pending" id="res-${esc(c.id)}"><b>${v}</b><span>${esc(c.title)}</span></article>`;
  }).join('');
  const foot = sec.querySelector('[data-results-foot]');
  const srcs = [...new Set((data.headline_stats || []).map(k => String(k.source || '').split(' ')[0]).filter(Boolean))];
  if (foot) foot.textContent = `Source: ${srcs.join(' · ')} · all simulation`;
}

export function mountResults(root, opts = {}) {
  if (!root) return Promise.resolve();
  return fetch('assets/data/results.json', { cache: 'no-cache' })
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(d => renderResults(root, d, opts))
    .catch(e => {
      console.warn('results', e);
      root.innerHTML = '<p class="small">Results could not be loaded (assets/data/results.json).</p>';
    });
}
