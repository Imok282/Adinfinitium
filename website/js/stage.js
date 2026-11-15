// Scroll-scrubbed "stage" helpers (Poly vocabulary) + the hero, mission and results beats.
// Everything is a pure function of a progress value, so it is reversible and exact in screenshots.
// Reuse from any beat's update(p, vis, t):
//
//   clamp(x, a=0, b=1)          lerp(a, b, t)          seg(s, a, b) -> 0..1 position of s inside [a, b]
//   ease(t) cubic-out           easeIO(t) cubic in-out
//   damper(k = 0.12)            -> d(target, snap) returns a damped value (s += (target - s) * k; snap = jump)
//   set(el, prop, value)        writes el.style[prop] (or a --custom-prop) only when the string changed
//   words(h)                    -> word spans of a heading (reuses js/nav.js .w spans, splits otherwise);
//                                  marks the heading .scrub so CSS transitions do not fight the scrub
//   revealWords(ws, k)          k 0..1: words rise 10 px, fade in and go blur -> sharp, one after another
//   fly(el, k, from)            k 0 = at `from` {x, y (px), s (scale), r (deg), blur (px)}, transparent; 1 = settled.
//                                  Writes --sh (0..1) so CSS can fade a soft drop shadow in as it lands.
//   deck(cards, f, opt)         f = -1..n-1 (fractional index of the front card). Card i rises from below while
//                                  f goes i-1 -> i, then sits behind: scale 1 - shrink*d, moves up lift*d px,
//                                  --dk (0..1) darkens it (CSS overlay). Returns the index of the front card.
//   dim(el, k)                  opacity of a black layer (dim-to-black showcase)
//   stageMode(ctx)              true when scrubbed animation should run (motion OK, WebGL on, viewport > 760 px).
//                                  When false the beats clear their inline styles: the static CSS layout shows.
//
// CSS contract (css/stage.css): a tall section with class .stage-sec holds a `position: sticky` .stage. The tall
// height and the sticky stage only apply under html.stage-on, which stageMode() sets; otherwise normal flow.

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const seg = (s, a, b) => clamp((s - a) / (b - a));
export const ease = t => 1 - Math.pow(1 - t, 3);
export const easeIO = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const r2 = x => Math.round(x * 100) / 100;

export function damper(k = 0.12) {
  let v = null;
  return (target, snap) => {
    if (v === null || snap) v = target;
    else { v += (target - v) * k; if (Math.abs(target - v) < 1e-4) v = target; }
    return v;
  };
}

export function set(el, prop, value) {
  const c = el.__st || (el.__st = {});
  if (c[prop] === value) return;
  c[prop] = value;
  if (prop.startsWith('--')) el.style.setProperty(prop, value); else el.style[prop] = value;
}
export function clear(el) {
  if (!el.__st) return;
  for (const k of Object.keys(el.__st)) { if (k.startsWith('--')) el.style.removeProperty(k); else el.style[k] = ''; }
  el.__st = {};
}

export function words(h) {
  if (!h) return [];
  let ws = [...h.querySelectorAll('.w')];
  if (!ws.length) {
    const split = node => [...node.childNodes].forEach(c => {
      if (c.nodeType === 3) {
        const frag = document.createDocumentFragment();
        c.textContent.split(/(\s+)/).forEach(t => {
          if (!t) return;
          if (/^\s+$/.test(t)) { frag.append(t); return; }
          const w = document.createElement('span'); w.className = 'w'; w.textContent = t; frag.append(w);
        });
        c.replaceWith(frag);
      } else if (c.nodeType === 1 && !c.matches('svg')) split(c);
    });
    split(h); ws = [...h.querySelectorAll('.w')];
  }
  h.classList.add('scrub');
  return ws;
}

export function revealWords(ws, k) {
  const n = ws.length;
  for (let i = 0; i < n; i++) {
    const e = ease(seg(k * (n + 2) / 3 - i / 3, 0, 1));   // each word takes ~1/3 of the span, staggered
    const w = ws[i];
    set(w, 'opacity', String(r2(e)));
    set(w, 'transform', e >= 1 ? 'none' : `translate3d(0,${r2((1 - e) * 10)}px,0)`);
    set(w, 'filter', e >= 1 ? 'none' : `blur(${r2((1 - e) * 6)}px)`);
  }
}

export function fly(el, k, from = {}) {
  const e = ease(clamp(k)), u = 1 - e;
  const x = (from.x || 0) * u, y = (from.y || 0) * u, s = 1 + ((from.s ?? 1.6) - 1) * u, r = (from.r || 0) * u;
  set(el, 'transform', e >= 1 ? 'none' : `translate3d(${r2(x)}px,${r2(y)}px,0) rotate(${r2(r)}deg) scale(${r2(s * 1000) / 1000})`);
  set(el, 'opacity', String(r2(clamp(k * 2.2))));
  const b = (from.blur ?? 8) * u;
  set(el, 'filter', b < 0.2 ? 'none' : `blur(${r2(b)}px)`);
  set(el, '--sh', String(r2(e)));
}

export function deck(cards, f, { lift = 18, shrink = 0.04, dark = 0.28, rise = innerHeight * 0.9, keep = 3 } = {}) {
  let front = -1;
  cards.forEach((c, i) => {
    const d = f - i;
    if (d <= -1) { set(c, 'transform', `translate3d(0,${Math.round(rise)}px,0)`); set(c, 'opacity', '0'); set(c, '--dk', '0'); return; }
    if (d < 0) {
      const e = ease(1 + d);
      set(c, 'transform', `translate3d(0,${r2((1 - e) * rise)}px,0) scale(${r2((1.04 - 0.04 * e) * 1000) / 1000})`);
      set(c, 'opacity', String(r2(clamp((1 + d) * 3))));
      set(c, '--dk', '0');
      return;
    }
    front = i;
    const q = Math.min(d, keep);
    set(c, 'transform', `translate3d(0,${r2(-q * lift)}px,0) scale(${r2((1 - shrink * q) * 1000) / 1000})`);
    set(c, 'opacity', d > keep ? String(r2(clamp(keep + 1 - d))) : '1');
    set(c, '--dk', String(r2(clamp(q * dark))));
  });
  return front;
}

export const dim = (el, k) => el && set(el, 'opacity', String(r2(clamp(k))));

export function stageMode(ctx) {
  const on = !ctx.reducedMotion && !document.body.classList.contains('no-webgl') && innerWidth > 760;
  document.documentElement.classList.toggle('stage-on', on);
  return on;
}

// ---------------------------------------------------------------------------------------------------------------
// Hero: centred headline (js/nav.js reveals it word by word on load); it settles back to ~30 % while the Moon
// takes over, like Poly's headline during its transition.
export function mountHero(sec, ctx) {
  const inner = sec.querySelector('.hero-inner');
  const d = damper();
  return {
    update(p) {
      if (!inner) return;
      if (!stageMode(ctx)) { clear(inner); return; }
      const s = d(p, ctx.shot);
      const k = seg(s, 0.15, 1);
      set(inner, 'opacity', String(r2(1 - 0.75 * k)));
      set(inner, 'transform', k <= 0 ? 'none' : `translate3d(0,${r2(-40 * k)}px,0)`);
    },
  };
}

// Mission: four constraints fly in from depth, land as a scattered collage of big cards, then snap to a grid.
const COLLAGE = [   // stage fractions (centre x, centre y), scale, rotation: one overlapping collage in the centre
  { x: 0.42, y: 0.6, s: 1.12, r: -6 }, { x: 0.58, y: 0.57, s: 1.12, r: 5 },
  { x: 0.45, y: 0.74, s: 1.08, r: 4 }, { x: 0.56, y: 0.76, s: 1.08, r: -4 },
];
const FROM = [   // fly-in offsets relative to the landing spot: one card from each edge (left, top, right, bottom)
  { x: -900, y: 40, s: 1.25, r: -18 }, { x: 60, y: -700, s: 1.25, r: 14 },
  { x: 900, y: -30, s: 1.25, r: 16 }, { x: -40, y: 650, s: 1.25, r: -14 },
];
export function mountMission(sec, ctx) {
  const stage = sec.querySelector('.stage'), grid = sec.querySelector('.features');
  const cards = [...sec.querySelectorAll('.features > li')];
  const head = sec.querySelector('.stage-head'), lede = head && head.querySelector('.lede');
  const caps = cards.map(c => c.querySelector('.cap'));
  const src = sec.querySelector('.stage .src');
  const ws = words(head && head.querySelector('h2'));
  const d = damper(), dv = damper();
  let W = 0, H = 0, centres = [], on = null;
  const measure = () => {
    W = stage.offsetWidth - (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lane')) || 0); H = stage.offsetHeight;   // collage centred left of the lander lane
    centres = cards.map(c => ({ x: grid.offsetLeft + c.offsetLeft + c.offsetWidth / 2, y: grid.offsetTop + c.offsetTop + c.offsetHeight / 2 }));
  };
  addEventListener('resize', () => { W = 0; });
  const ro = new ResizeObserver(() => { W = 0; }); ro.observe(stage); ro.observe(grid);   // fonts / layout (transforms do not fire it)
  return {
    update(p, vis) {
      const mode = stageMode(ctx);
      if (mode !== on) { on = mode; W = 0; if (!mode) [...cards, ...caps, ...ws, lede, src].forEach(e => e && clear(e)); }
      if (!mode) return;
      if (!W) measure();
      const s = d(p, ctx.shot), v = dv(vis, ctx.shot);
      const hk = Math.max(seg(v, 0.25, 0.95), s > 0 ? 1 : 0);
      revealWords(ws, hk);
      if (lede) set(lede, 'opacity', String(r2(seg(hk, 0.4, 1))));
      const g = easeIO(seg(s, 0.56, 0.74));           // snap to grid
      cards.forEach((c, i) => {
        const k = seg(s, 0.02 + i * 0.1, 0.2 + i * 0.1);   // fly-in, one after another
        const L = COLLAGE[i], F = FROM[i], C = centres[i];
        const cx = (L.x * W - C.x) * (1 - g), cy = (L.y * H - C.y) * (1 - g);
        const cs = 1 + (L.s - 1) * (1 - g), cr = L.r * (1 - g);
        const e = ease(k), u = 1 - e;
        const x = cx + F.x * u, y = cy + F.y * u, sc = cs * (1 + (F.s - 1) * u), r = cr + F.r * u;
        set(c, 'transform', `translate3d(${r2(x)}px,${r2(y)}px,0) rotate(${r2(r)}deg) scale(${r2(sc * 1000) / 1000})`);
        set(c, 'opacity', String(r2(clamp(k * 2.5))));
        const b = 9 * u;
        set(c, 'filter', b < 0.2 ? 'none' : `blur(${r2(b)}px)`);
        set(c, '--sh', String(r2(e)));
        set(c, 'zIndex', String(i + 1));
        if (caps[i]) { const ck = seg(s, 0.7 + i * 0.025, 0.8 + i * 0.025); set(caps[i], 'opacity', String(r2(ck))); set(caps[i], 'transform', `translate3d(0,${r2((1 - ease(ck)) * 8)}px,0)`); }
      });
      if (src) set(src, 'opacity', String(r2(seg(s, 0.78, 0.88))));
    },
  };
}

// Results: a pinned horizontal strip of real landings slides sideways, one panel at a time; each big number
// counts up as its panel reaches the centre. Our own tiles (results.json) follow below in normal flow.
export function mountResultsStage(sec, ctx) {
  const stage = sec.querySelector('.strip-stage'), strip = sec.querySelector('.strip');
  const head = sec.querySelector('.stage-head'), ws = words(head && head.querySelector('h2'));
  const panels = strip ? [...strip.children] : [];
  const nums = panels.map(p => p.querySelector('[data-count]'));
  const d = damper(), dv = damper();
  let span = -1, on = null, shown = [];
  const ro = strip && new ResizeObserver(() => { span = -1; }); ro && (ro.observe(strip), ro.observe(stage));
  const fmt = (el, k) => {   // count up toward data-count, keeping the decimals of the target
    const t = el.dataset.count; if (!t) return;
    const dec = (t.split('.')[1] || '').length, v = (parseFloat(t) * k).toFixed(dec);
    return v.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (el.dataset.unit || '');
  };
  return {
    update(p, vis) {
      if (!strip) return;
      const mode = stageMode(ctx);
      if (mode !== on) {
        on = mode; span = -1;
        if (!mode) { [strip, ...panels, ...ws].forEach(e => e && clear(e)); nums.forEach(n => { if (n && n.dataset.count) n.textContent = n.dataset.final || n.textContent; }); return; }
      }
      if (!mode) return;
      if (span < 0) span = Math.max(0, strip.scrollWidth - stage.offsetWidth);
      const s = d(p, ctx.shot), v = dv(vis, ctx.shot);
      revealWords(ws, Math.max(seg(v, 0.25, 0.95), s > 0 ? 1 : 0));
      const x = easeIO(seg(s, 0.05, 0.95));
      set(strip, 'transform', `translate3d(${r2(-x * span)}px,0,0)`);
      const n = panels.length, at = x * (n - 1);   // fractional index of the centred panel
      panels.forEach((el, i) => {
        const dd = Math.abs(at - i);
        set(el, 'opacity', String(r2(1 - 0.55 * clamp(dd))));
        set(el, 'transform', `scale(${r2((1 - 0.06 * clamp(dd)) * 1000) / 1000})`);
        const nb = nums[i];
        if (nb && nb.dataset.count) {
          if (!nb.dataset.final) nb.dataset.final = nb.textContent;
          const k = ease(clamp(1.6 - dd * 1.6));
          const txt = k >= 0.999 ? nb.dataset.final : fmt(nb, k);
          if (shown[i] !== txt) { shown[i] = txt; nb.textContent = txt; }
        }
      });
    },
  };
}
