// Small page effects (loaded from js/nav.js on every page). Nothing runs for reduced motion.
// Scope: subpages (body:not(.home)) and the home page's .teaser-sec sections; the pinned home beats have their own motion.
//  - Number tickers: KPI values count up from 0 to the rendered value once, on entering the viewport
//    (results.html .stat-row b, benchmarks values); decimals, separators and units are kept, tabular-nums.
//  - Pointer tilt (max 6 deg, damped) on .tz, .tm, .res-card; desktop with a fine pointer only. .mc-card is skipped
//    (it already has its own transforms). Elements that carry an inline transform from another script are left alone.
// Headline word reveals already come from js/nav.js (word split, staggered, blur -> sharp), so they are not repeated here.
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const home = document.body.classList.contains('home');
const inScope = el => !home || !!el.closest('.teaser-sec');

if (!reduce && 'IntersectionObserver' in window) {
  // ---------- number tickers ----------
  const NUM = /^([^\d−-]*)([−-]?)([\d,]*\.?\d+)(.*)$/;
  const tick = b => {
    if (b.dataset.ticked) return;
    const txt = b.textContent.trim(), m = txt.match(NUM);
    if (!m || txt.length > 16) return;
    b.dataset.ticked = '1';
    const [, pre, sign, numS, post] = m, target = parseFloat(numS.replace(/,/g, ''));
    if (!isFinite(target) || target === 0) return;
    const dec = (numS.split('.')[1] || '').length, commas = numS.includes(',');
    const fmt = v => { let s = v.toFixed(dec); if (commas) s = s.replace(/\B(?=(\d{3})+(?!\d))/g, ','); return pre + sign + s + post; };
    b.style.fontVariantNumeric = 'tabular-nums';
    const t0 = performance.now(), dur = 1100;
    const step = now => {
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      b.textContent = k >= 1 ? txt : fmt(target * e);
      if (k < 1) requestAnimationFrame(step);
    };
    b.textContent = fmt(0); requestAnimationFrame(step);
  };
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); tick(e.target); } }), { threshold: 0.6 });
  const SEL = '.results-page .stat-row li b, .results-page .mis-big, .bm-root b, .bm-root [class*="val"], .bm-root [class*="big"]';
  const seen = new WeakSet();
  const scan = () => document.querySelectorAll(SEL).forEach(b => {
    if (seen.has(b) || !inScope(b) || b.children.length) return;
    seen.add(b); io.observe(b);
  });
  if (!home) {   // values are rendered by page scripts after load: rescan on DOM changes for a few seconds
    scan();
    const mo = new MutationObserver(scan); mo.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => mo.disconnect(), 8000);
  }

  // ---------- pointer tilt ----------
  if (matchMedia('(pointer: fine) and (hover: hover)').matches && innerWidth >= 1100) {
    const MAX = 6;
    let cur = null, tx = 0, ty = 0, rx = 0, ry = 0, raf = 0;
    const loop = () => {
      raf = 0;
      if (!cur) return;
      rx += (tx - rx) * 0.15; ry += (ty - ry) * 0.15;
      cur.style.transform = `perspective(900px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg)`;
      if (Math.abs(tx - rx) > 0.01 || Math.abs(ty - ry) > 0.01) raf = requestAnimationFrame(loop);
      else if (tx === 0 && ty === 0) { cur.style.transform = ''; cur.style.willChange = ''; cur = null; }
    };
    const kick = () => { if (!raf) raf = requestAnimationFrame(loop); };
    document.addEventListener('pointermove', e => {
      const el = e.target.closest && e.target.closest('.tz, .tm, .res-card');
      if (el && inScope(el) && (el === cur || !el.style.transform)) {
        if (cur && cur !== el) { cur.style.transform = ''; rx = ry = 0; }
        cur = el; el.style.willChange = 'transform';
        const r = el.getBoundingClientRect();
        ty = ((e.clientX - r.left) / r.width - 0.5) * 2 * MAX;
        tx = -((e.clientY - r.top) / r.height - 0.5) * 2 * MAX;
        kick();
      } else if (cur) { tx = ty = 0; kick(); }
    }, { passive: true });
    document.addEventListener('pointerleave', () => { if (cur) { tx = ty = 0; kick(); } });
  }
}
