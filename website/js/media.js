// Cinematic media slots: hero backdrop, descent clip (Mission), landing clip (Touchdown).
// Each slot stays hidden unless its file exists in assets/video/ (listed in assets/video/manifest.json),
// so the 3D scene is always the fallback. Files: assets/video/{hero,descent,landing}.mp4 (+ optional .webm)
// and a poster assets/video/{slot}.jpg. Make them with tools/process_video.sh <input> <slot>.
// Reduced motion: poster only, never autoplay. These clips are decorative (AI-generated, credited in the footer).

const DIR = 'assets/video/';
const exists = url => fetch(url, { method: 'HEAD', cache: 'no-cache' })
  .then(r => r.ok && !/text\/html/i.test(r.headers.get('content-type') || ''))
  .catch(() => false);

// assets/video/manifest.json lists the files that exist (no 404 probing): { "files": ["hero.mp4", "hero.jpg", ...] }.
// tools/process_video.sh adds its outputs there; edit it by hand if you add or delete a clip.
let manifest = null;
const loadManifest = () => manifest || (manifest = fetch(`${DIR}manifest.json`, { cache: 'no-cache' })
  .then(r => (r.ok ? r.json() : { files: [] })).then(d => new Set(d.files || [])).catch(() => new Set()));
async function probe(slot) {
  const have = await loadManifest();
  const [mp4, webm, jpg] = ['mp4', 'webm', 'jpg'].map(ext => have.has(`${slot}.${ext}`));
  return { mp4, webm, jpg, any: mp4 || webm };
}

function fill(video, slot, have) {
  if (have.jpg) video.poster = `${DIR}${slot}.jpg`;
  if (have.webm) { const s = document.createElement('source'); s.src = `${DIR}${slot}.webm`; s.type = 'video/webm'; video.append(s); }
  if (have.mp4) { const s = document.createElement('source'); s.src = `${DIR}${slot}.mp4`; s.type = 'video/mp4'; video.append(s); }
  video.muted = true; video.loop = true; video.playsInline = true;
}

const tryPlay = v => { const p = v.play(); p && p.catch(() => {}); };

async function mountHero(reducedMotion) {
  const video = document.getElementById('hero-video');
  if (!video) return;
  const have = await probe('hero');
  if (!have.any && !have.jpg) return;
  if (reducedMotion) {             // poster only
    if (!have.jpg) return;
    video.poster = `${DIR}hero.jpg`;
  } else {
    if (!have.any) return;
    fill(video, 'hero', have);
    video.preload = 'auto';
    video.load();
    tryPlay(video);
  }
  video.hidden = false;
  document.body.classList.add('has-hero-video');
  window.dispatchEvent(new CustomEvent('adi:backdrop', { detail: { on: true } }));

  // fade out as the page scrolls past the hero; pause once invisible
  const hero = document.getElementById('top');
  let raf = 0;
  const update = () => {
    raf = 0;
    const r = hero.getBoundingClientRect();
    const o = Math.max(0, Math.min(1, 1 - (-r.top) / Math.max(1, r.height * 0.75)));
    video.style.opacity = o.toFixed(3);
    video.style.visibility = o < 0.01 ? 'hidden' : 'visible';
    if (!reducedMotion) { if (o < 0.01 && !video.paused) video.pause(); else if (o >= 0.01 && video.paused) tryPlay(video); }
    window.dispatchEvent(new CustomEvent('adi:backdrop', { detail: { on: o >= 0.01, opacity: o } }));
  };
  // capture: catches window scrolling and the body scroller used in screenshot mode (?scroll=)
  document.addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(update); }, { passive: true, capture: true });
  addEventListener('resize', update);
  update();
}

async function mountClip(fig, reducedMotion) {
  const slot = fig.dataset.slot;
  const video = fig.querySelector('video');
  const have = await probe(slot);
  if (reducedMotion) {
    if (!have.jpg) return;
    video.poster = `${DIR}${slot}.jpg`;      // still frame only, no sources, no autoplay
    fig.hidden = false;
    return;
  }
  if (!have.any) return;
  fill(video, slot, have);
  video.preload = 'none';
  fig.hidden = false;
  if (!('IntersectionObserver' in window)) { video.controls = true; return; }
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting) tryPlay(video); else if (!video.paused) video.pause();
  }, { threshold: 0.25 }).observe(fig);
}

export function mountMedia({ reducedMotion = false } = {}) {
  mountHero(reducedMotion).catch(e => console.warn('hero video', e));
  document.querySelectorAll('figure.media-slot[data-slot]').forEach(f => mountClip(f, reducedMotion).catch(e => console.warn('clip', e)));
  return {};
}
