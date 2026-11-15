// Company home (index.html): ONE requestAnimationFrame loop for the whole page.
//  - Scroll is lerped (k 0.12); cursor is lerped too.
//  - Hero: the 3D card (js/hero3d.js, external mode: this loop calls its tick()) opens from an inset rounded card to
//    full-bleed (clip-path) over the first ~0.7 viewport of scroll; the 3D camera sinks toward the Moon with scroll, so the stars, the
//    Moon limb and the lander move at different speeds; the text layer moves fastest and follows the cursor slightly.
//  - [data-parallax] images drift -8 % .. +8 % of their box as they cross the viewport (M9).
//  Reduced motion / ?static: nothing moves; the hero is a static rounded card with the poster image.
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));

export function mountHome() {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches || new URLSearchParams(location.search).has('static');
  const root = document.documentElement;
  const heroSec = document.querySelector('[data-hero]'), stage = document.getElementById('hero3d');
  const text = document.querySelector('.ch-hero-text'), cue = document.querySelector('.ch-scroll');
  if (reduced) { root.classList.add('ch-still'); return; }
  root.classList.add('ch-motion');

  let hero = null;
  import('./hero3d.js').then(m => { hero = m.mountHero3D(stage, { external: true }); }).catch(() => {});

  // the looping preview card plays only while on screen
  const reel = document.querySelector('.ch-reel');
  if (reel) new IntersectionObserver(es => reel.classList.toggle('ch-reel-paused', !es[0].isIntersecting)).observe(reel);
  const par = [...document.querySelectorAll('[data-parallax]')].map(el => ({ el, top: 0, h: 0, box: el.parentElement, last: '' }));
  const reelEl = document.querySelector('.ch-reel'), big = document.querySelector('.ch-big');
  const bigH = big && big.querySelector('.ch-big-h'), wl = document.querySelector('.waitlist-band');
  let wlTop = 1e9;
  const over = document.querySelector('.ch-over'), overFill = over && over.querySelector('.ch-over-fill');
  let reelTop = 0, reelH = 1, bigTop = 0, bigHt = 1, docH = 1, mainTop = 0, mainH = 1;
  let vh = innerHeight, vw = innerWidth, heroH = 1, dirty = true;
  const segEls = { reel: document.getElementById('reel'), products: document.getElementById('products'), big: document.getElementById('bigtype'),
    learn: document.getElementById('learn'), research: document.getElementById('research') };
  const segRect = new Map();
  const measure = () => {
    vh = innerHeight; vw = innerWidth; const y = scrollY;
    heroH = heroSec ? heroSec.offsetHeight : vh;
    par.forEach(p => { const r = p.box.getBoundingClientRect(); p.top = r.top + y; p.h = r.height; });
    if (reelEl) { const r = reelEl.parentElement.getBoundingClientRect(); reelTop = r.top + y; reelH = r.height; }
    if (big) { const r = big.getBoundingClientRect(); bigTop = r.top + y; bigHt = r.height; }
    docH = document.documentElement.scrollHeight;
    if (main) { const r = main.getBoundingClientRect(); mainTop = r.top + y; mainH = r.height; }
    if (wl) wlTop = wl.getBoundingClientRect().top + y;
    for (const el of Object.values(segEls)) if (el) { const r = el.getBoundingClientRect(); segRect.set(el, { top: r.top + y, h: r.height }); }
    dirty = false;
  };
  addEventListener('resize', () => { dirty = true; });
  new ResizeObserver(() => { dirty = true; }).observe(document.body);

  let sy = scrollY, mx = 0, my = 0, cx = 0, cy = 0, lastHero = '';
  let fill = 0, lastPush = 0, leaving = false;
  const push = d => {   // only real input (not automation) can carry you into the landing page
    if (navigator.webdriver || !over || scrollY < docH - vh - 4 || d <= 0) return;
    fill = Math.min(1, fill + d / 1400); lastPush = performance.now();
  };
  addEventListener('wheel', e => push(e.deltaY), { passive: true });
  let ty0 = null;
  addEventListener('touchstart', e => { ty0 = e.touches[0].clientY; }, { passive: true });
  addEventListener('touchmove', e => { if (ty0 != null) { push((ty0 - e.touches[0].clientY) * 2); ty0 = e.touches[0].clientY; } }, { passive: true });
  addEventListener('pointermove', e => { mx = e.clientX / vw * 2 - 1; my = e.clientY / vh * 2 - 1; }, { passive: true });

  const loop = () => {
    requestAnimationFrame(loop);
    if (document.hidden) return;
    if (dirty) measure();
    const y = scrollY;
    sy += (y - sy) * 0.12; if (Math.abs(y - sy) < 0.3) sy = y;
    cx += (mx - cx) * 0.06; cy += (my - cy) * 0.06;

    // hero: inset card -> full-bleed, depth layers
    const hp = clamp(sy / Math.max(1, Math.min(heroH - vh, vh * 0.7)));
    const e = 1 - Math.pow(1 - hp, 3);
    const inset = vw < 760 ? 12 : 24, topI = vw < 760 ? 76 : 88;
    const clip = `inset(${(topI * (1 - e)).toFixed(1)}px ${(inset * (1 - e)).toFixed(1)}px ${(inset * (1 - e)).toFixed(1)}px round ${(24 * (1 - e)).toFixed(1)}px)`;
    const tx = `translate3d(${(cx * -10).toFixed(1)}px,${(cy * -6 - sy * 0.35).toFixed(1)}px,0)`;
    const key = clip + tx;
    if (key !== lastHero) {
      lastHero = key;
      if (stage) stage.style.clipPath = clip;
      if (text) { text.style.transform = tx; text.style.opacity = String(clamp(1 - (sy / vh - 0.35) * 1.6)); }
      if (cue) cue.style.opacity = String(clamp(1 - sy / 120));
    }
    if (hero && hero.setSegments) {   // one Moon effect per segment, each a different kind of motion
      const segP = el => { const r = segRect.get(el); return r ? clamp((sy + vh - r.top) / (r.h + vh)) : 0; };
      hero.setSegments({
        relay: clamp((sy - vh * 0.15) / (vh * 1.1)),   // hero tail: a relay satellite crosses the sky
        terminator: segP(segEls.reel),                  // preview card: the sunlight terminator sweeps
        pins: segP(segEls.products),                    // products: landing-site pins pop up one by one
        rover: segP(segEls.learn),                      // watch it learn: the rover drives its route
        packets: segP(segEls.research),                 // research: data packets rover -> lander -> Earth
      });
    }
    if (hero) {   // the fixed Moon scene follows the whole page
      hero.setState({ hero: clamp(sy / vh), f: clamp(sy / Math.max(1, docH - vh)), big: big ? clamp((sy - bigTop + vh * 0.3) / (vh * 0.5)) * (1 - clamp((sy - bigTop - (bigHt - vh) * 0.8) / (vh * 0.4))) : 0, end: clamp((sy + vh - wlTop) / vh) });
      hero.tick();
    }

    // bent reel: enters curled (rotateX 26deg, slightly smaller) and flattens as it scrolls in
    if (reelEl) {
      const k = clamp((sy + vh - reelTop) / (vh * 0.85)), e = 1 - Math.pow(1 - k, 3);
      const v = `rotateX(${(26 * (1 - e)).toFixed(2)}deg) scale(${(0.9 + 0.1 * e).toFixed(4)})`;
      if (v !== reelEl.__t) { reelEl.__t = v; reelEl.style.transform = v; }
    }
    // big type over 3D: the words appear ONCE, only while the section is pinned (products have left), above the lander
    let bigPin = 0;
    if (big) {
      bigPin = clamp((sy - bigTop) / Math.max(1, bigHt - vh));
      const inn = clamp((sy - bigTop + vh * 0.15) / (vh * 0.35)), out = clamp((bigPin - 0.85) / 0.15);
      const e = 1 - Math.pow(1 - inn, 3), v = `scale(${(0.9 + 0.1 * e).toFixed(4)})`, o = String((e * (1 - out)).toFixed(3));
      if (v + o !== big.__t) { big.__t = v + o; bigH.style.transform = v; bigH.style.opacity = o; }
    }
    // overscroll band: visible at the very bottom; wheel / touch beyond the end fills it; full -> landing.html
    if (over) {
      const atEnd = y >= docH - vh - 4;
      over.classList.toggle('on', atEnd);
      if (!atEnd || performance.now() - lastPush > 450) fill *= 0.94;
      if (fill < 0.002) fill = 0;
      overFill.style.transform = `scaleX(${fill.toFixed(3)})`;
      if (fill >= 1 && !leaving) { leaving = true; location.href = over.getAttribute('href'); }
    }
    // section image parallax
    for (const p of par) {
      const t = (sy + vh - p.top) / (vh + p.h);            // 0 entering at the bottom, 1 leaving at the top
      if (t < -0.1 || t > 1.1) continue;
      const v = `translate3d(0,${((clamp(t) - 0.5) * 16).toFixed(2)}%,0)`;
      if (v !== p.last) { p.last = v; p.el.style.transform = v; }
    }
  };
  requestAnimationFrame(loop);
}
