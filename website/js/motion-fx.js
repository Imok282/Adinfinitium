// Motion pass for the subpages (loaded from js/nav.js after fx.js). Uses Motion 14.6.0 (ESM, jsDelivr).
// Nothing runs on the story page (body.story) or under prefers-reduced-motion; if the Motion import fails, nothing is hidden.
//  1. Staggered reveals (inView + stagger, spring without bounce) for card grids and lists. Movement goes through the
//     CSS `translate` property (a --mfx-y variable), so it never touches `transform`, which fx.js uses for the tilt.
//  2. Number tickers for stats fx.js does not cover (pilot results, navigator tiles, research posts, mission-control
//     readouts): every number in the rendered text counts up, then the exact original text is restored. If another
//     script rewrites the element meanwhile (live readouts), the ticker stops and leaves that text alone.
//  3. (no custom cursor: the system pointer is never changed)
//  4. Path drawing for inline SVG diagrams in research posts (U-Net edges, Dream-RSI loop arrows) on entering view.
const body = document.body;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

if (!reduce && !body.classList.contains('story')) {
  import('https://cdn.jsdelivr.net/npm/motion@14.6.0/+esm').then(M => run(M)).catch(e => console.warn('motion-fx', e));
}

function run({ animate, inView, stagger, spring }) {
  const css = document.createElement('style');
  css.textContent = `
.mfx-r { translate: 0 calc(var(--mfx-y, 0) * 1px); }
`;
  document.head.appendChild(css);

  // ---------- 1. staggered reveals ----------
  const ITEMS = ['.tz', '.res-card', '.rs-item', '.rs-feature', '.cp-card', '.bm-card', '.nv-fig', '.pp-tile', '.post-stats > *',
    '.cp-team > *', '.co-grid > *', '.mc-grid > .mc-card', '.method-list > li', '.post-body figure'].join(',');
  const SKIP = '.topnav, .footer, .mc-modal, #built, .story';
  const groups = new Map();
  document.querySelectorAll(ITEMS).forEach(el => {
    if (el.closest(SKIP) || el.classList.contains('mfx-r')) return;
    if (el.parentElement && el.parentElement.closest(ITEMS)) return;   // only the outer card of nested matches
    const g = el.parentElement;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(el);
  });
  groups.forEach((els, parent) => {
    const below = parent.getBoundingClientRect().top > innerHeight * 0.92;   // only hide what has not been seen yet
    if (!below) return;
    els.forEach(el => { el.classList.add('mfx-r'); el.style.opacity = '0'; el.style.setProperty('--mfx-y', '26'); });
    inView(parent, () => {
      animate(els, { opacity: [0, 1], '--mfx-y': [26, 0] }, { delay: stagger(0.07), type: spring, bounce: 0, visualDuration: 0.6 })
        .finished?.then(() => els.forEach(el => { el.style.opacity = ''; el.style.removeProperty('--mfx-y'); }))
        .catch(() => {});
    }, { amount: 0.15 });
    // safety: never leave anything hidden (e.g. a group that never scrolls into view in a short window)
    setTimeout(() => els.forEach(el => { if (el.style.opacity === '0' && el.getBoundingClientRect().top < innerHeight) { el.style.opacity = ''; el.style.removeProperty('--mfx-y'); } }), 6000);
  });

  // ---------- 2. number tickers ----------
  const NUMS = /(?<![A-Za-z\d.])[−-]?\d[\d,]*(?:\.\d+)?/g;   // skip digits that are part of a name (F1, Site23)
  const tick = b => {
    if (b.dataset.ticked || b.children.length) return;
    const txt = b.textContent;
    const parts = [...txt.matchAll(NUMS)];
    if (!parts.length || txt.length > 24) return;
    const nums = parts.map(m => {
      const s = m[0].replace('−', '-'), v = parseFloat(s.replace(/,/g, ''));
      return { i: m.index, len: m[0].length, v, dec: (s.split('.')[1] || '').length, commas: s.includes(','), minus: m[0][0] === '−' };
    });
    if (nums.every(n => !isFinite(n.v) || n.v === 0)) return;
    b.dataset.ticked = '1';
    b.style.fontVariantNumeric = 'tabular-nums';
    const fmt = k => {
      let out = '', at = 0;
      for (const n of nums) {
        let s = Math.abs(n.v * k).toFixed(n.dec);
        if (n.commas) s = s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        out += txt.slice(at, n.i) + (n.v < 0 ? (n.minus ? '−' : '-') : '') + s;
        at = n.i + n.len;
      }
      return out + txt.slice(at);
    };
    let mine = fmt(0), stopped = false;
    b.textContent = mine;
    const ctl = animate(0, 1, {
      duration: 1.1, ease: [0.22, 1, 0.36, 1],
      onUpdate: k => {
        if (stopped) return;
        if (b.textContent !== mine) { stopped = true; ctl && ctl.stop(); return; }   // another script took over
        mine = fmt(k); b.textContent = mine;
      },
      onComplete: () => { if (!stopped && b.textContent === mine) b.textContent = txt; },
    });
  };
  const TICK = '.pp-res > b, .nv-tiles b, .post-stats b, .mc-nums b';
  const watch = b => {
    if (b.dataset.mfxW) return;
    b.dataset.mfxW = '1';
    const go = () => inView(b, () => { tick(b); }, { amount: 0.6 });
    if (/\d/.test(b.textContent)) go();
    else {   // live readouts start as "—": wait for the first real value
      const mo = new MutationObserver(() => { if (/\d/.test(b.textContent)) { mo.disconnect(); go(); } });
      mo.observe(b, { childList: true, characterData: true, subtree: true });
      setTimeout(() => mo.disconnect(), 10000);
    }
  };
  document.querySelectorAll(TICK).forEach(watch);

  // ---------- 3. (removed) no custom cursor: the user keeps the normal system pointer ----------

  // ---------- 4. path drawing ----------
  const draw = svg => {
    if (svg.dataset.mfxD) return;
    svg.dataset.mfxD = '1';
    if (svg.classList.contains('mc-svg')) {
      // U-Net diagram: modelcards.css already draws its edges with a CSS animation; hold it until the diagram is in view
      const els = svg.querySelectorAll('.mc-edge, .mc-beam');
      els.forEach(p => p.style.animationPlayState = 'paused');
      inView(svg, () => els.forEach(p => p.style.animationPlayState = ''), { amount: 0.3 });
      return;
    }
    const paths = svg.querySelectorAll('.re:not(.dash)');   // Dream-RSI loop arrows
    if (!paths.length) return;
    paths.forEach(p => p.style.opacity = '0');
    inView(svg, () => {
      paths.forEach(p => p.style.opacity = '');
      animate(paths, { pathLength: [0, 1] }, { duration: 0.8, delay: stagger(0.08), ease: [0.65, 0, 0.35, 1] });
    }, { amount: 0.3 });
  };
  const scanSvg = () => document.querySelectorAll('.post-body svg.mc-svg, .post-body svg.rsi-svg').forEach(draw);
  scanSvg();
  const mo = new MutationObserver(scanSvg);   // the hazard U-Net diagram is injected after a fetch
  mo.observe(body, { childList: true, subtree: true });
  setTimeout(() => mo.disconnect(), 8000);
}
