/* Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in index.html. */
/* Full-screen option for every screen: a button (inside the company banner when the page has one, else floating in the corner),
   the F key, and Esc to leave (the browser's own). Pages add: <script src="/fullscreen.js"></script> after the banner script.
   window.ADFS.toggle(element?) toggles the page, or one element such as a pane. */
(() => {
  const root = document.documentElement;
  const request = (el) => (el.requestFullscreen || el.webkitRequestFullscreen || (() => Promise.reject(new Error('unsupported')))).call(el);
  const exit = () => (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  const current = () => document.fullscreenElement || document.webkitFullscreenElement || null;
  const supported = !!(root.requestFullscreen || root.webkitRequestFullscreen);
  const toggle = (el = root) => { if (!supported) return; if (current()) { exit(); } else { const r = request(el); if (r && r.catch) r.catch(() => {}); } };
  window.ADFS = { toggle, supported, isFull: () => !!current() };
  if (!supported || window.top !== window.self) return; // embedded views (the 3D pane of the 3D + Vizard screen) use their parent's button

  const EXPAND = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const SHRINK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const style = document.createElement('style');
  style.textContent = '.adfs { display: inline-flex; align-items: center; gap: 8px; height: 38px; padding: 0 11px; border-radius: 10px; border: 1px solid #235347; background: #000000; color: #FFFFFF; cursor: pointer;'
    + ' font: 600 13px "Instrument Sans", -apple-system, sans-serif; letter-spacing: .04em; transition: border-color .25s, background .25s; flex: none; }'
    + '.adfs svg { width: 18px; height: 18px; flex: none; } .adfs:hover { border-color: #8EB69B; background: #0B2B26; } .adfs .adfs-t { display: none; }'
    + '.adfs.float { position: fixed; right: 14px; bottom: 14px; z-index: 2000; opacity: .72; } .adfs.float:hover { opacity: 1; } .adfs.float .adfs-t { display: inline; }'
    + '@container (min-width: 1700px) { .adb .adfs .adfs-t { display: inline; } }'
    + 'body.adfs-on .adfs.float { opacity: .35; }';
  document.head.appendChild(style);

  function mount() {
    const btn = document.createElement('button'); btn.className = 'adfs'; btn.type = 'button';
    const paint = () => { const on = !!current(); btn.innerHTML = (on ? SHRINK : EXPAND) + `<span class="adfs-t">${on ? 'Exit full screen' : 'Full screen'}</span>`;
      btn.title = on ? 'Exit full screen (F or Esc)' : 'Full screen (F)'; btn.setAttribute('aria-label', btn.title); document.body.classList.toggle('adfs-on', on); };
    btn.onclick = () => toggle(); paint();
    document.addEventListener('fullscreenchange', paint); document.addEventListener('webkitfullscreenchange', paint);
    const inBanner = document.querySelector('#adb .adb-in');
    const navBtn = document.getElementById('adb-nav'); // next to the Screens menu, so the status lights keep the right edge
    if (inBanner) { if (navBtn) navBtn.after(btn); else inBanner.appendChild(btn); } else { btn.classList.add('float'); document.body.appendChild(btn); }
    addEventListener('keydown', (e) => { if ((e.key === 'f' || e.key === 'F') && !e.metaKey && !e.ctrlKey && !e.altKey && !/^(input|textarea|select)$/i.test(e.target.tagName || '') && !e.target.isContentEditable) { e.preventDefault(); toggle(); } });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount); else mount();
})();
