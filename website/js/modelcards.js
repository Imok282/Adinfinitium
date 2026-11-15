// Model cards (#agents): eight pastel cards rise from below as a deck (each replacing the last), then settle into a grid.
// Each opens a panel (unfolding from its card) with diagrams drawn from the real code, or a user file in
// assets/diagrams/<key>.svg|.png, the training data and real metrics. All numbers: assets/data/models.json (each with its source).
// Owned by round-3 agent D.

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = x => 1 - Math.pow(1 - x, 3);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
const SVGNS = 'http://www.w3.org/2000/svg';

// ---------- diagram (inline SVG from models.json node / edge lists) ----------
function anchor(a, b) {
  if (b.x >= a.x + a.w - 2) return [[a.x + a.w, a.y + a.h / 2], [b.x, b.y + b.h / 2]];
  if (b.x + b.w <= a.x + 2) return [[a.x, a.y + a.h / 2], [b.x + b.w, b.y + b.h / 2]];
  if (b.y >= a.y + a.h) return [[a.x + a.w / 2, a.y + a.h], [b.x + b.w / 2, b.y]];
  return [[a.x + a.w / 2, a.y], [b.x + b.w / 2, b.y + b.h]];
}
let svgN = 0;
export function diagramSVG(d) {
  const ids = Object.fromEntries(d.nodes.map(n => [n.id, n]));
  const H = Math.max(...d.nodes.map(n => n.y + n.h), d.lane ? d.lane.y + (d.lane.h || 0) : 0) + 22;
  const mk = `mc-arr-${++svgN}`;
  let s = `<svg class="mc-svg" viewBox="0 0 960 ${H}" role="img" aria-label="${esc(d.title)}" xmlns="${SVGNS}">`;
  s += `<defs><marker id="${mk}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="mc-arrhead"/></marker></defs>`;
  if (d.lane) {
    const l = d.lane, x = l.x ?? 8, w = l.w ?? 944, h = l.h ?? (H - l.y - 6);
    s += `<rect class="mc-lane" x="${x}" y="${l.y}" width="${w}" height="${h}" rx="16"/><text class="mc-lane-t" x="${l.right ? x + w - 12 : x + 12}" y="${l.bottom ? l.y + h - 9 : l.y + 17}"${l.right ? ' text-anchor="end"' : ''}>${esc(l.label)}</text>`;
  }
  let beams = '', k = 0;
  for (const [f, t, kind] of d.edges) {
    const a = ids[f], b = ids[t];
    if (!a || !b) continue;
    const [[x1, y1], [x2, y2]] = anchor(a, b);
    const st = `style="--k:${k++}"`;
    s += `<line class="mc-edge${kind ? ' ' + kind : ''}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"${kind ? '' : ' pathLength="1"'} ${st} marker-end="url(#${mk})"/>`;
    if (!kind) beams += `<line class="mc-beam" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" pathLength="1" ${st}/>`;
  }
  s += `<g class="mc-beams" aria-hidden="true">${beams}</g>`;
  for (const n of d.nodes) {
    s += `<g class="mc-node t-${n.t}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.small ? 9 : 12}"/>`;
    s += n.small ? `<text class="mc-ns2" x="${n.x + 12}" y="${n.y + n.h / 2 + 4.5}">${esc(n.label)}</text>`
      : `<text class="mc-nl" x="${n.x + 12}" y="${n.y + 22}">${esc(n.label)}</text>`;
    String(n.sub || '').split('\n').filter(Boolean).forEach((line, i) => {
      s += `<text class="mc-ns" x="${n.x + 12}" y="${n.y + 40 + i * 15}">${esc(line)}</text>`;
    });
    s += '</g>';
  }
  return s + '</svg>';
}

// user diagram first (assets/diagrams/<key>.svg, then .png), else the drawn one. Only files listed in
// assets/diagrams/manifest.json are tried, so missing files never cause 404s.
let manifest = null;
function diagramManifest() {
  return manifest || (manifest = fetch('assets/diagrams/manifest.json').then(r => (r.ok ? r.json() : { files: [] }))
    .then(j => new Set(j.files || [])).catch(() => new Set()));
}
function findUserDiagram(key) {
  return diagramManifest().then(f => (f.has(`${key}.svg`) ? `assets/diagrams/${key}.svg` : f.has(`${key}.png`) ? `assets/diagrams/${key}.png` : null));
}

// ---------- modal ----------
function buildModal() {
  const m = document.createElement('div');
  m.className = 'mc-modal';
  m.hidden = true;
  m.innerHTML = `<div class="mc-backdrop" data-close></div>
    <div class="mc-dialog" role="dialog" aria-modal="true" aria-labelledby="mc-title" tabindex="-1">
      <header class="mc-head">
        <div><p class="mc-eyebrow" id="mc-kind"></p><h3 id="mc-title"></h3><p class="mc-one" id="mc-one"></p></div>
        <button type="button" class="mc-close" data-close aria-label="Close model details">
          <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15"/></svg></button>
      </header>
      <div class="mc-body" id="mc-body"></div>
    </div>`;
  document.body.appendChild(m);
  return m;
}

const LEGEND = '<p class="mc-legend" aria-hidden="true"><span class="t-in">input</span><span class="t-nn">learned</span><span class="t-phys">physics / simulation</span><span class="t-out">output</span></p>';
function diagramBlock(title, d, id) {
  return `<section class="mc-diagram tone-${id ? 'main' : 'alt'}"><div class="mc-dhead"><p class="mc-sub">${esc(title)}</p>${LEGEND}</div>
    <div class="mc-dwrap"${id ? ` id="${id}"` : ''} tabindex="0" aria-label="Diagram; scroll sideways on small screens">${diagramSVG(d)}</div>
    <p class="mc-pan" aria-hidden="true">Swipe the diagram sideways to see all of it.</p></section>`;
}
function tableHTML(t) {
  if (!t) return '';
  return `<div class="mc-table"><h4>${esc(t.title)}</h4><table><thead><tr>${t.head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>
    <tbody>${t.rows.map(r => `<tr${/accepted/.test(r.join(' ')) ? ' class="acc"' : ''}>${r.map(c => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>
    ${t.note ? `<p class="mc-tnote">${esc(t.note)}</p>` : ''}<p class="src">Source: ${esc(t.source)}</p></div>`;
}
function panelHTML(m) {
  if (m.slot) {
    return `<section class="mc-diagram tone-main"><div class="mc-dwrap user" id="mc-dwrap"><img class="mc-userdiagram" src="${esc(m.thumb)}" alt="${esc(m.thumb_alt)}"></div></section>
      <p class="mc-how">${md(m.how)}</p>`;
  }
  const metrics = m.metrics.map(x => `<li><span class="mc-ml">${esc(x.label)}</span>
      <span class="mc-mv">${esc(x.value)}${x.base ? ` <span class="mc-mb">vs ${esc(x.base)} ${esc(x.base_label || m.base_label || 'backup')}</span>` : ''}</span>
      <span class="src">${esc(x.source)}</span></li>`).join('');
  const train = m.training.map(r => `<div><dt>${esc(r.k)}</dt><dd>${esc(r.v)}</dd></div>`).join('');
  const extra = (m.extra || []).map(x => x.diagram ? diagramBlock(x.title, x.diagram)
    : `<section class="mc-diagram tone-alt"><div class="mc-dhead"><p class="mc-sub">${esc(x.title)}</p></div><img class="mc-userdiagram" src="${esc(x.img)}" alt="${esc(x.alt)}" loading="lazy"></section>`).join('');
  const fig = m.image ? `<figure class="mc-fig"><img src="${esc(m.image)}" alt="${esc(m.image_alt)}" loading="lazy" decoding="async">
      <figcaption>${esc(m.image_caption)}${m.provenance === 'generated' ? ' <span class="mc-gen">GENERATED — NOT OBSERVED</span>' : ''}<span class="src">${esc(m.image_src)}</span></figcaption></figure>` : '';
  return `${diagramBlock(m.diagram.title, m.diagram, 'mc-dwrap')}
    <div class="mc-cols">
      <div class="mc-left">
        <p class="mc-how">${md(m.how)}</p>
        <p class="mc-backup"><b>Physics backup, runs first:</b> ${esc(m.backup)}</p>
        <h4>Trained on</h4><dl class="mc-train">${train}</dl>
        <p class="src">${esc(m.training_source)}</p>
        ${fig}
      </div>
      <div class="mc-right"><h4>Evaluation</h4><ul class="mc-metrics">${metrics}</ul>${tableHTML(m.table)}</div>
    </div>${extra}`;
}

// ---------- mount ----------
export function mount(section, ctx = {}) {
  const grid = section.querySelector('#mc-grid');
  const stage = section.querySelector('.mc-stage');
  const sub = section.querySelector('.mc-subline');
  const arch = section.querySelector('.mc-arch');
  let archFrac = 0, archLast = -1;
  const gridMode = !!ctx.grid;                       // models.html: plain grid, light fly-in on first view, no pinned deck
  const staticMode = gridMode || !!(ctx.reducedMotion || document.body.classList.contains('no-webgl'));
  if (staticMode) section.classList.add('mc-static');
  const cards = [...grid.querySelectorAll('.mc-card')];
  const N = cards.length;
  let data = null, modal = null, opener = null, s = staticMode ? 1 : 0, last = -1;
  let flipFrom = null, closing = false, M = null;
  // Motion 14.6.0 for the shared-layout card -> panel open; without it the CSS unfold is used
  if (!ctx.reducedMotion && !ctx.shot) import('https://cdn.jsdelivr.net/npm/motion@14.6.0/+esm').then(mod => { M = mod; }).catch(() => { M = null; });
  let off = cards.map(() => ({ x: 0, y: 0 })), vh = innerHeight;

  // layout cache (resize only): offset from each card's grid slot to the deck position (centre of the grid area)
  function measure() {
    vh = innerHeight;
    archFrac = arch ? clamp(arch.offsetHeight / Math.max(1, section.offsetHeight - innerHeight)) : 0;
    const gw = grid.clientWidth, gh = grid.clientHeight;
    off = cards.map(c => ({ x: gw / 2 - (c.offsetLeft + c.offsetWidth / 2), y: gh * 0.42 - (c.offsetTop + c.offsetHeight / 2) }));
    last = -1;
  }
  measure();
  addEventListener('resize', () => { measure(); render(s); });

  function enrich(json) {
    data = json;
    const byKey = Object.fromEntries(json.models.map(m => [m.key, m]));
    cards.forEach(c => {
      const m = byKey[c.dataset.key];
      if (!m) return;
      const km = m.key_metric || {};
      c.classList.add(`tone-${m.tone || 'sand'}`);
      c.innerHTML = `<span class="mc-thumb"><img src="${esc(m.thumb)}" alt="" loading="lazy" decoding="async"></span>
        <span class="mc-ctext"><span class="mc-cname">${esc(m.name)}</span><span class="mc-cone">${esc(m.one_line)}</span>
        <span class="mc-cval">${esc(km.value)}</span><span class="mc-clab">${esc(km.label)}</span>
        ${km.source ? `<span class="mc-csrc">${esc(km.source)}</span>` : ''}</span>`;
      c.setAttribute('aria-label', `${m.name}: ${km.value}, ${km.label}. Open details`);
    });
    measure(); render(s);
    const want = new URLSearchParams(location.search).get('model');
    if (want && byKey[want]) setTimeout(() => open(want, null), 50);
  }

  function lock(on) {
    const b = document.body;
    if (on) {
      const sw = innerWidth - document.documentElement.clientWidth;
      b.dataset.mcPad = b.style.paddingRight || '';
      if (sw > 0) b.style.paddingRight = `${sw}px`;
      b.classList.add('mc-locked');
    } else {
      b.style.paddingRight = b.dataset.mcPad || '';
      b.classList.remove('mc-locked');
    }
  }
  function focusables() {
    return [...modal.querySelectorAll('button, a[href], [tabindex="0"]')].filter(e => !e.disabled && e.offsetParent !== null);
  }
  function onKey(e) {
    if (!modal || modal.hidden) return;
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const f = focusables();
    if (!f.length) return;
    const first = f[0], lastF = f[f.length - 1];
    if (e.shiftKey && (document.activeElement === first || document.activeElement === modal.querySelector('.mc-dialog'))) { e.preventDefault(); lastF.focus(); }
    else if (!e.shiftKey && document.activeElement === lastF) { e.preventDefault(); first.focus(); }
  }
  function open(key, from) {
    if (!data) return;
    const m = data.models.find(x => x.key === key);
    if (!m) return;
    if (!modal) {
      modal = buildModal();
      modal.addEventListener('click', e => { if (e.target.closest('[data-close]')) close(); });
      document.addEventListener('keydown', onKey);
    }
    opener = from || document.activeElement;
    modal.querySelector('#mc-kind').textContent = m.kind;
    modal.querySelector('#mc-title').textContent = m.name;
    modal.querySelector('#mc-one').textContent = m.one_line;
    const body = modal.querySelector('#mc-body');
    body.innerHTML = panelHTML(m);
    body.scrollTop = 0;
    const sc = parseFloat(new URLSearchParams(location.search).get('mcscroll'));   // screenshots only
    if (sc) setTimeout(() => { body.scrollTop = sc; }, 300);
    if (!m.slot) findUserDiagram(key).then(url => {
      const w = body.querySelector('#mc-dwrap');
      if (url && w && modal.querySelector('#mc-title').textContent === m.name) {
        w.innerHTML = `<img class="mc-userdiagram" src="${url}" alt="Architecture diagram of the ${esc(m.name)}">`;
        w.classList.add('user');
      }
    });
    modal.hidden = false;
    if (ctx.shot) modal.classList.add('mc-noanim');
    lock(true);
    const dlg = modal.querySelector('.mc-dialog');
    flipFrom = null;
    // shared-layout open (Motion FLIP): the dialog grows from the card's rectangle; the card's image + title cross-fade into the header
    if (from && M && !ctx.reducedMotion && !ctx.shot) {
      modal.classList.add('mc-flip', 'on');
      const a = from.getBoundingClientRect(), b = dlg.getBoundingClientRect();
      flipFrom = from;
      const { animate, spring } = M;
      dlg.style.transformOrigin = '0 0';
      animate(dlg, { x: [a.left - b.left, 0], y: [a.top - b.top, 0], scaleX: [a.width / b.width, 1], scaleY: [a.height / b.height, 1] },
        { type: spring, visualDuration: 0.45, bounce: 0.15 });
      const ghost = from.cloneNode(true);
      ghost.removeAttribute('aria-label'); ghost.setAttribute('aria-hidden', 'true'); ghost.tabIndex = -1;
      ghost.className = from.className + ' mc-ghost';
      const cs = getComputedStyle(from);
      Object.assign(ghost.style, { position: 'fixed', left: `${a.left}px`, top: `${a.top}px`, width: `${a.width}px`, height: `${a.height}px`, margin: 0, transform: 'none',
        opacity: 1, zIndex: 2, pointerEvents: 'none', background: cs.backgroundColor, borderRadius: cs.borderRadius, color: cs.color, padding: cs.padding, boxSizing: 'border-box' });
      modal.appendChild(ghost);
      animate(ghost, { x: [0, b.left - a.left], y: [0, b.top - a.top], opacity: [1, 0] }, { duration: 0.32, ease: [0.22, 1, 0.36, 1] }).finished.then(() => ghost.remove(), () => ghost.remove());
      animate(modal.querySelector('.mc-head'), { opacity: [0, 1] }, { delay: 0.12, duration: 0.3 });
      animate(body, { opacity: [0, 1] }, { delay: 0.18, duration: 0.3 });
      requestAnimationFrame(() => modal.querySelector('.mc-close').focus({ preventScroll: true }));
      return;
    }
    // fallback unfold (no Motion): start the dialog at the card's rectangle, then release to its own layout
    if (from && !ctx.reducedMotion) {
      const a = from.getBoundingClientRect(), b = dlg.getBoundingClientRect();
      dlg.style.transition = 'none';
      dlg.style.transformOrigin = '0 0';
      dlg.style.transform = `translate3d(${a.left - b.left}px,${a.top - b.top}px,0) scale(${a.width / b.width},${a.height / b.height})`;
      void dlg.offsetWidth;
      dlg.style.transition = '';
    }
    requestAnimationFrame(() => { modal.classList.add('on'); dlg.style.transform = ''; modal.querySelector('.mc-close').focus({ preventScroll: true }); });
  }
  function finishClose() {
    modal.classList.remove('on', 'mc-flip');
    const dlg = modal.querySelector('.mc-dialog');
    dlg.style.transform = ''; dlg.style.opacity = '';
    modal.hidden = true;
    lock(false);
    if (opener && opener.focus) opener.focus({ preventScroll: true });
  }
  function close() {
    if (!modal || modal.hidden || closing) return;
    const card = flipFrom;
    if (card && M && !ctx.reducedMotion && card.isConnected) {
      // close: animate back into the card it came from
      closing = true;
      const dlg = modal.querySelector('.mc-dialog');
      const a = card.getBoundingClientRect(), b = dlg.getBoundingClientRect();
      const { animate, spring } = M;
      modal.classList.remove('on');
      animate(dlg, { x: a.left - b.left, y: a.top - b.top, scaleX: a.width / b.width, scaleY: a.height / b.height, opacity: [1, 0] },
        { type: spring, visualDuration: 0.35, bounce: 0, opacity: { duration: 0.3, ease: 'easeIn' } })
        .finished.then(() => { closing = false; finishClose(); }, () => { closing = false; finishClose(); });
      return;
    }
    finishClose();
  }

  grid.addEventListener('click', e => { const c = e.target.closest('.mc-card'); if (c) open(c.dataset.key, c); });
  fetch('assets/data/models.json').then(r => r.json()).then(enrich).catch(e => console.warn('modelcards data', e));

  // v 0..1: deck phase (0.04 .. 0.62): card i rises from below to the deck spot, the previous one tucks back;
  // settle phase (0.66 .. 0.9): the whole deck spreads into the grid.
  const D0 = 0.04, D1 = 0.62, S0 = 0.66, S1 = 0.9, step = (D1 - D0) / N;
  function render(v) {
    if (gridMode) return;
    if (Math.abs(v - last) < 1e-4) return;
    last = v;
    if (sub) { const q = ease(clamp((v - 0.02) / 0.1)); sub.style.opacity = q; sub.style.transform = `translate3d(0,${(1 - q) * 8}px,0)`; }
    const spread = ease(clamp((v - S0) / (S1 - S0)));
    const active = clamp((v - D0) / step, 0, N);         // how many cards have risen (fractional)
    cards.forEach((c, i) => {
      const rise = ease(clamp(active - i));               // this card's rise 0..1
      const behind = clamp(active - i - 1, 0, 3);         // how many cards sit on top of it
      const o = off[i];
      // deck pose
      const dy = (1 - rise) * vh * 0.9 - behind * 14 * (1 - spread);
      const sc = (1 - behind * 0.04 * (1 - spread));
      const x = o.x * (1 - spread), y = o.y * (1 - spread) + dy * (1 - spread);
      c.style.transform = `translate3d(${x}px,${y}px,0) scale(${sc})`;
      c.style.opacity = rise <= 0 ? 0 : clamp(rise * 3) * (1 - Math.min(behind, 3) / 3 * (1 - spread) * 0.9);
      c.style.zIndex = String(i + 1);
      c.classList.toggle('settled', spread > 0.98);
    });
  }

  // the deck's timeline starts once the overview card has scrolled past
  function deckP(p) { return clamp((p - archFrac) / Math.max(0.01, 1 - archFrac)); }
  // overview card: far (scale .85, blurred, faint) -> near (scale 1, sharp) while the section enters the viewport
  function renderArch(q) {
    if (!arch || Math.abs(q - archLast) < 1e-3) return;
    archLast = q;
    const e = ease(q);
    arch.style.transform = `translate3d(0,${(1 - e) * 40}px,0) scale(${0.85 + 0.15 * e})`;
    arch.style.opacity = String(0.15 + 0.85 * e);
    arch.style.filter = e < 0.98 ? `blur(${(1 - e) * 8}px)` : '';
  }
  if (gridMode) {
    section.classList.add('mc-gridmode');
    cards.forEach((c, i) => c.style.setProperty('--i', i));
    if (ctx.reducedMotion || !('IntersectionObserver' in window)) section.classList.add('mc-in');
    else {
      const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { section.classList.add('mc-in'); io.disconnect(); } }, { threshold: 0.12 });
      io.observe(grid);
    }
    return { open, close, update() {} };
  }
  if (staticMode) render(1);
  // screenshot mode: headless capture can miss the last rAF frame, so also settle from timers (?scroll= only)
  if (ctx.shot && !staticMode) [2000, 4500, 7000, 8600, 9800].forEach(ms => setTimeout(() => {
    const r = section.getBoundingClientRect();
    const p = clamp(-r.top / Math.max(1, r.height - innerHeight)); s = clamp(deckP(p) * 1.15); render(s);
    renderArch(clamp((innerHeight - (r.top + (arch ? arch.offsetHeight * 0.35 : 0))) / (innerHeight * 0.9)));
  }, ms));
  return {
    update(p, vis) {
      if (staticMode) return;
      const target = clamp(deckP(p) * 1.15);
      renderArch(clamp(vis / 0.95));
      s = ctx.shot ? target : s + (target - s) * 0.12;
      if (Math.abs(target - s) < 1e-3) s = target;
      render(s);
      if (stage) stage.classList.toggle('mc-on', vis > 0.05);
    },
    open, close,
  };
}
