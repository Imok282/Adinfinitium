// Top bar: pill dropdowns + the round "all sections" sheet. Keyboard accessible
// (buttons with aria-expanded, Escape closes and returns focus, outside click closes).
// The active-section highlight is driven by main.js (onScroll) via [data-nav] attributes.
const bar = document.getElementById('topnav');
if (bar) {
  const pairs = [...bar.querySelectorAll('button[aria-controls]')].map(btn => ({ btn, menu: document.getElementById(btn.getAttribute('aria-controls')) }));
  const close = (except) => pairs.forEach(p => { if (p !== except) { p.btn.setAttribute('aria-expanded', 'false'); p.menu.hidden = true; } });
  const open = p => { close(p); p.btn.setAttribute('aria-expanded', 'true'); p.menu.hidden = false; };

  pairs.forEach(p => {
    p.btn.addEventListener('click', e => {
      e.stopPropagation();
      p.btn.getAttribute('aria-expanded') === 'true' ? close() : open(p);
    });
    p.btn.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { e.preventDefault(); open(p); p.menu.querySelector('a')?.focus(); }
    });
    p.menu.addEventListener('keydown', e => {
      const items = [...p.menu.querySelectorAll('a')], i = items.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); items[(i + 1) % items.length].focus(); }
      if (e.key === 'ArrowUp') { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
    });
  });

  document.addEventListener('click', e => { if (!bar.contains(e.target)) close(); });
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    const p = pairs.find(q => q.btn.getAttribute('aria-expanded') === 'true');
    if (p) { close(); p.btn.focus(); }
  });
  bar.addEventListener('focusout', e => { if (e.relatedTarget && !bar.contains(e.relatedTarget)) close(); });

  // in-page links: close menus; fall back to the parent section when a deep anchor (e.g. #view-hazard) is missing;
  // data-off scrolls a little into a tall section (e.g. into the network fly-through)
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  bar.addEventListener('click', e => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    close();
    const id = a.getAttribute('href').slice(1);
    if (!id) { e.preventDefault(); return; }           // placeholder links (TODO URLs)
    let el = document.getElementById(id);
    const off = parseFloat(a.dataset.off || '0');
    if (el && !off) return;                              // normal anchor jump
    e.preventDefault();
    if (!el) el = document.getElementById(id.startsWith('view-') ? 'models' : 'top');
    const scroller = document.documentElement.classList.contains('shot') ? document.body : document.scrollingElement;
    const y = el.getBoundingClientRect().top + scroller.scrollTop + off * innerHeight;
    scroller.scrollTo({ top: y, behavior: reduce ? 'auto' : 'smooth' });
    history.replaceState(null, '', '#' + el.id);
  });
}

// Bar state: transparent at the very top, frosted with a hairline after 8 px (wordmark crossfades to "A∞").
if (bar) {
  const y = () => Math.max(document.scrollingElement?.scrollTop || 0, document.body.scrollTop || 0, window.scrollY || 0);
  let raf = 0;
  const upd = () => { raf = 0; const v = y(); bar.classList.toggle('scrolled', v > 8); bar.classList.toggle('brand-mini', v > 40); };
  // Brand morph (css/style.css "brand morph"): measure once where each letter has to travel (offsets ignore transforms)
  const brand = bar.querySelector('.brand'), word = brand && brand.querySelector('.brand-word'), inf = brand && brand.querySelector('.brand-inf');
  const measureBrand = () => {
    if (!word || !inf) return;
    const cx = word.offsetLeft + word.offsetWidth / 2;
    word.querySelectorAll('.bl').forEach(l => {
      const c = l.offsetLeft + l.offsetWidth / 2;   // .brand is the offsetParent of the letters too
      const ls = parseFloat(getComputedStyle(l).letterSpacing) || 0;   // trailing letter-spacing is not ink
      const target = l.classList.contains('bl-a') ? cx - (l.offsetWidth - ls) * 0.59 - 1 + ls / 2 : cx;
      l.style.setProperty('--dx', (target - c).toFixed(1) + 'px');
    });
    inf.style.left = (cx + 1).toFixed(1) + 'px';
  };
  measureBrand(); addEventListener('resize', measureBrand); addEventListener('load', measureBrand); document.fonts && document.fonts.ready.then(measureBrand);
  document.addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(upd); }, { passive: true, capture: true });
  upd(); setTimeout(upd, 400); addEventListener('load', () => setTimeout(upd, 100));
}

// Headlines: one serif phrase per H1/H2 gets a hand-drawn amber underline that draws in ~400 ms after the words.
const SCRIBBLE = '<svg class="scribble" viewBox="0 0 200 20" preserveAspectRatio="none" aria-hidden="true">' +
  '<path pathLength="1" d="M3 11 C 40 5, 95 3, 150 6 S 197 9, 196 7 C 160 9, 90 12, 30 14 C 70 15, 130 15, 188 13"/></svg>';
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window);
const heads = [...document.querySelectorAll('main h1, main h2')].filter(h => !h.closest('#models'));
heads.forEach(h => h.querySelectorAll('.serif').forEach(sp => {
  if (sp.parentElement.classList.contains('hl')) return;
  const wrap = document.createElement('span'); wrap.className = 'hl';
  sp.replaceWith(wrap); wrap.append(sp); wrap.insertAdjacentHTML('beforeend', SCRIBBLE);
}));
if (reduceMotion) heads.forEach(h => h.classList.add('no-anim'));

// Once-only reveal (opacity + 12 px rise, 500 ms); H1/H2 rise word by word with a blur -> sharp change.
// Added by JS so the page is fully visible without it; skipped for reduced motion. The models section is left alone.
(() => {
  if (reduceMotion) return;
  let n = 0;
  const splitWords = (node) => {
    [...node.childNodes].forEach(c => {
      if (c.nodeType === 3) {
        const frag = document.createDocumentFragment();
        c.textContent.split(/(\s+)/).forEach(t => {
          if (!t) return;
          if (/^\s+$/.test(t)) { frag.append(t); return; }
          const w = document.createElement('span'); w.className = 'w'; w.textContent = t; w.style.setProperty('--d', `${(n++ * 0.045).toFixed(3)}s`);
          frag.append(w);
        });
        c.replaceWith(frag);
      } else if (c.nodeType === 1 && !c.matches('svg')) splitWords(c);
    });
  };
  heads.forEach(h => { n = 0; splitWords(h); h.classList.add('words'); h.style.setProperty('--sd', `${(n * 0.045 + 0.4).toFixed(2)}s`); });
  // Checked on scroll (rAF-throttled) instead of IntersectionObserver: deterministic, and anything scrolled past
  // by an anchor jump is revealed too. Each element is revealed once and then dropped from the list.
  let pending = [...document.querySelectorAll('[data-reveal], .sec:not(#models):not(.hero) > .panel'), ...heads];
  pending.forEach(el => el.classList.add(el.matches('h1, h2') ? 'words' : 'reveal'));
  let raf = 0;
  const check = () => {
    raf = 0;
    const lim = innerHeight * 0.92;
    pending = pending.filter(el => {
      const r = el.getBoundingClientRect();
      if (r.top < lim && (r.width || r.height)) { el.classList.add('is-in'); return false; }
      return true;
    });
  };
  const queue = () => { if (!raf && pending.length) raf = requestAnimationFrame(check); };
  document.addEventListener('scroll', queue, { passive: true, capture: true });
  addEventListener('resize', queue);
  new ResizeObserver(queue).observe(document.body);
  check(); setTimeout(check, 300);
})();

// multi-page: highlight the pill of the current page
{
  const here = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
  document.querySelectorAll('.topnav [data-page]').forEach(a => {
    const on = (a.getAttribute('href') || '').toLowerCase() === here;
    a.classList.toggle('active', on); if (on) a.setAttribute('aria-current', 'page');
  });
}

// small interaction polish on every page (pointer devices, motion allowed):
// magnetic primary buttons and a soft spotlight that follows the pointer on cards ([data-spot] or the known card classes).
{
  const fine = matchMedia('(pointer: fine)').matches, calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (fine && !calm) {
    document.querySelectorAll('.btn-primary, .pill-primary').forEach(b => {
      b.classList.add('magnetic');
      b.addEventListener('pointermove', e => {
        const r = b.getBoundingClientRect();
        b.style.transform = `translate(${((e.clientX - r.left) / r.width - 0.5) * 3}px, ${((e.clientY - r.top) / r.height - 0.5) * 2}px)`;   // subtle: never feels like the pointer is pulled
      });
      b.addEventListener('pointerleave', () => { b.style.transform = ''; });
    });
    const SPOT = '[data-spot], .tz, .tm, .mc-card, .res-card, .stat-row li, .bench-card, .mis';
    document.addEventListener('pointermove', e => {
      const c = e.target.closest && e.target.closest(SPOT);
      if (!c) return;
      const r = c.getBoundingClientRect();
      c.style.setProperty('--mx', `${e.clientX - r.left}px`); c.style.setProperty('--my', `${e.clientY - r.top}px`);
      c.classList.add('spot-on');
    }, { passive: true });
    document.addEventListener('pointerout', e => { const c = e.target.closest && e.target.closest(SPOT); if (c && !c.contains(e.relatedTarget)) c.classList.remove('spot-on'); });
  }
}

// Small page effects (number tickers, pointer tilt); see js/fx.js
import('./fx.js').catch(() => {});
// Motion pass (staggered reveals, extra number tickers, custom cursor, SVG path drawing); see js/motion-fx.js
import('./motion-fx.js').catch(() => {});
if (document.querySelector('form[data-waitlist]')) import('./waitlist.js').catch(() => {});
