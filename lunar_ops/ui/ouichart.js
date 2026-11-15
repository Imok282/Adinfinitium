/* Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in index.html. */
/* Render OpenUI charts (vendor/openui, MIT) from OpenUI Lang into any element: OUI.render(el, lang). One React root per
   element, re-rendered in place so OpenUI's own transitions animate the change. Needs vendor/openui/openui-bundle.min.js. */
(() => {
  const roots = new WeakMap();
  window.OUI = {
    q: (s) => JSON.stringify(String(s)),
    arr: (a) => '[' + a.map((v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : 0)).join(',') + ']',
    strs: (a) => '[' + a.map((s) => JSON.stringify(String(s))).join(',') + ']',
    render(el, lang) {
      const O = window.__OpenUI; if (!O || !el) return false; el.style.setProperty('--oui-round', `${Math.max(60, el.clientHeight - 58)}px`);
      el.classList.add('ouic'); let r = roots.get(el); if (!r) { el.innerHTML = ''; r = O.createRoot(el); roots.set(el, r); }
      r.render(O.React.createElement(O.Renderer, { response: lang, library: /\bStack\(/.test(lang) && O.openuiLibrary ? O.openuiLibrary : O.openuiChatLibrary })); return true;
    },
  };
})();
