// landing.html #mc-teaser: "screen-in-screen" hand-off. A full-viewport tile showing the mission-control dashboard
// shrinks into a centred card, then the headline and the button appear. Pure function of section progress p
// (main.js addBeat via the 'adi:beat' event); transforms / opacity / clip-path only; sizes cached on resize.
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (s, a, b) => clamp((s - a) / (b - a));
const ease = t => 1 - Math.pow(1 - t, 3);
const easeIO = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const r2 = x => Math.round(x * 100) / 100;
function set(el, prop, v) { const c = el.__m || (el.__m = {}); if (c[prop] === v) return; c[prop] = v; el.style[prop] = v; }

export function mount(sec, ctx = {}) {
  const live = !ctx.reducedMotion && !document.body.classList.contains('no-webgl') && !matchMedia('(prefers-reduced-motion: reduce)').matches;
  sec.classList.add(live ? 'mct-live' : 'mct-static');
  const stage = sec.querySelector('.mct-stage'), tile = sec.querySelector('.mct-tile'), copy = sec.querySelector('.mct-copy');
  let box = null;                                  // tile's settled rect relative to the stage (layout, no transforms)
  const measure = () => {
    if (!tile || !stage) return;
    const s = stage.getBoundingClientRect(), t = tile.getBoundingClientRect();
    const sc = parseFloat(tile.__m?.sc || 1);
    box = { sw: s.width, sh: s.height, cx: t.left - s.left + t.width / 2, cy: t.top - s.top + t.height / 2, w: t.width, h: t.height };
  };
  if ('ResizeObserver' in window && stage) new ResizeObserver(() => { set(tile, 'transform', 'none'); box = null; }).observe(stage);
  let sv = live ? 0 : 1;
  function update(p) {
    if (!live || !tile) return;
    if (!box) { set(tile, 'transform', 'none'); measure(); if (!box) return; }
    sv = ctx.shot ? p : sv + (p - sv) * 0.2;
    if (Math.abs(p - sv) < 1e-4) sv = p;
    const k = easeIO(seg(sv, 0.12, 0.62));                        // 0 = full viewport, 1 = settled card
    const s0 = Math.max(box.sw / box.w, box.sh / box.h);           // scale that covers the viewport
    const s = s0 + (1 - s0) * k;
    const dx = (box.sw / 2 - box.cx) * (1 - k), dy = (box.sh / 2 - box.cy) * (1 - k);
    set(tile, 'transform', k >= 1 ? 'none' : `translate3d(${r2(dx)}px,${r2(dy)}px,0) scale(${r2(s * 1000) / 1000})`);
    set(tile, 'opacity', String(r2(0.25 + 0.75 * ease(seg(sv, 0, 0.14)))));
    set(tile, 'clipPath', `inset(0 round ${r2(24 * k / s)}px)`);
    const c = ease(seg(sv, 0.58, 0.8));
    if (copy) { set(copy, 'opacity', String(r2(c))); set(copy, 'transform', c >= 1 ? 'none' : `translate3d(0,${r2((1 - c) * 18)}px,0)`); }
  }
  return { update };
}
