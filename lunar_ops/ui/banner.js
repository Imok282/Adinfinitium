/* Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in index.html. */
/* Company banner on every screen: brand, UTC and mission clocks, phase, time warp and the subsystem lights, all from the
   live feed (/ws/ui). Lights are computed, never decorative: GNC (pointing error), PROP (propellant), ENGINES (healthy),
   COMMS / SUN / TERRAIN (Earth link, sunlight, safety at the target, from the simulator), AI PILOT (on, trained).
   Usage: <div id="adb" data-screen="Mission wall"></div> <link rel="stylesheet" href="/banner.css"> <script src="/banner.js"></script> */
(() => {
  const host = document.getElementById('adb'); if (!host) return;
  host.classList.add('adb');
  const screen = host.dataset.screen || 'Mission';
  // our own mark: the infinity symbol (no agency logos); replaced by /assets/brand/logo.svg if present
  const MARK = '<svg viewBox="0 8 48 32" aria-hidden="true"><defs><path id="adb-inf" d="M4 24 C4 11 20 11 24 24 C28 37 44 37 44 24 C44 11 28 11 24 24 C20 37 4 37 4 24 Z"/></defs>'
    + '<use href="#adb-inf" fill="none" stroke="#235347" stroke-width="5.5" stroke-linejoin="round"/>'
    + '<use href="#adb-inf" fill="none" stroke="#8EB69B" stroke-width="2.6" stroke-linejoin="round"/>'
    + '<circle r="2.8" fill="#DAF1DE"><animateMotion dur="7s" repeatCount="indefinite"><mpath href="#adb-inf"/></animateMotion></circle></svg>'; // our mark: the infinity symbol, a light running along it
  host.innerHTML = `<div class="adb-in"><div class="mark"><span id="adb-logo">${MARK}</span><div class="word"><b>ADINFINITIUM</b><small>Lunar south pole · ${screen}</small></div></div>
    <button class="nav" id="adb-nav" aria-haspopup="true" aria-expanded="false">Screens<i></i></button>
    <div class="sep"></div><div class="clk"><span class="k">UTC</span><span class="v n" id="adb-utc">—</span></div>
    <div class="clk"><span class="k">Mission time</span><span class="v n" id="adb-met">—</span></div><div class="sep"></div>
    <div class="clk"><span class="k"><span class="live" id="adb-live"></span>Phase</span><span class="v txt" id="adb-phase">Waiting for the simulator</span></div>
    <div class="clk"><span class="k">Time warp</span><span class="v n" id="adb-warp">—</span></div>
    <div class="lights" id="adb-lights"></div></div>`;
  // Every screen is served by the one Mission Control server; this menu switches between them (same tab) or opens one in a new window.
  const SCREENS = [['Mission control', '/', 'Telemetry, maps, charts, logs'], ['Mission wall', '/showcase/wall.html', 'Terrain, AI layers, flight data'],
    ['3D + Vizard', '/showcase/duo.html', 'Our render next to Basilisk Vizard'], ['3D view', '/showcase/?live=1', 'Our 3D render on its own'],
    ['Pilot brain', '/ai.html', 'What the AI sees, thinks and does'], ['Vehicles', '/showcase/vehicles.html', 'Lander, rover and propulsion module'], ['All screens', '/hub.html', 'Hub: open the wall, 3D and console screens']];
  const here = location.pathname.replace(/index\.html$/, ''), menu = document.createElement('div'); menu.className = 'adb-menu'; menu.hidden = true;
  menu.innerHTML = SCREENS.map(([n, u, d]) => `<a href="${u}" class="${u.split('?')[0] === here ? 'on' : ''}"><b>${n}</b><span>${d}</span></a>`).join('');
  document.body.appendChild(menu);
  const nav = document.getElementById('adb-nav'), toggle = (open) => { menu.hidden = !open; nav.setAttribute('aria-expanded', String(open));
    if (open) { const r = nav.getBoundingClientRect(); menu.style.left = `${r.left}px`; menu.style.top = `${r.bottom + 8}px`; } };
  nav.onclick = (e) => { e.stopPropagation(); toggle(menu.hidden); }; addEventListener('click', (e) => { if (!menu.contains(e.target)) toggle(false); }); addEventListener('keydown', (e) => { if (e.key === 'Escape') toggle(false); });
  // the same mark as the browser tab icon
  if (!document.querySelector('link[rel~="icon"]')) { const ic = document.createElement('link'); ic.rel = 'icon'; ic.type = 'image/svg+xml';
    ic.href = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="10" fill="#000"/><path d="M6 24 C6 12 20 12 24 24 C28 36 42 36 42 24 C42 12 28 12 24 24 C20 36 6 36 6 24 Z" fill="none" stroke="#8EB69B" stroke-width="4" stroke-linejoin="round"/></svg>'); document.head.appendChild(ic); }
  fetch('/assets/brand/logo.svg', { method: 'HEAD' }).then((r) => { if (r.ok) document.getElementById('adb-logo').innerHTML = '<img src="/assets/brand/logo.svg" alt="">'; }).catch(() => {});
  const $ = (id) => document.getElementById(id);
  const PH = { DOCKED: 'Orbit · docked', SEPARATED: 'Orbit · separated', DEORBIT: 'Deorbit burn', COAST: 'Coast to 15 km', BRAKING: 'Braking', TERMINAL: 'Terminal descent', LANDED: 'Landed', ENDED: 'Run ended' };
  function lights(f) {
    const out = [], add = (k, st, why) => out.push(`<div class="lt ${st}" title="${why}">${k}<b>${st === 'go' ? 'GO' : st === 'caution' ? 'CAUTION' : st === 'nogo' ? 'NO-GO' : '—'}</b></div>`);
    const pe = f?.attitude?.pointing_error_deg; add('GNC', pe == null ? '' : pe < 8 ? 'go' : pe < 20 ? 'caution' : 'nogo', pe == null ? 'no pointing data' : `pointing error ${pe.toFixed(1)}°`);
    const fuel = f?.vehicle?.fuel_kg; add('PROP', fuel == null ? '' : fuel > 30 ? 'go' : fuel > 10 ? 'caution' : 'nogo', fuel == null ? '' : `${fuel.toFixed(1)} kg propellant`);
    const ok = f?.vehicle?.engines_ok; add('ENGINES', ok == null ? '' : ok === 5 ? 'go' : ok >= 3 ? 'caution' : 'nogo', ok == null ? '' : `${ok} of 5 engines`);
    const sl = f?.site_light; add('COMMS', !sl ? '' : sl.earth ? 'go' : 'nogo', sl ? (sl.earth ? 'Earth above the horizon at the target' : 'Earth below the horizon at the target') : 'no site data');
    add('SUN', !sl ? '' : sl.sun ? 'go' : 'caution', sl ? (sl.sun ? 'target sunlit' : 'target in shadow') : '');
    add('TERRAIN', !sl ? '' : sl.safe ? 'go' : 'nogo', sl ? (sl.safe ? 'target cell safe' : 'target cell unsafe') : '');
    const s = f?.sim; add('AI PILOT', !s ? '' : !s.ai ? 'caution' : s.pilot_trained ? 'go' : 'caution', !s ? '' : !s.ai ? 'off: classic guidance' : s.pilot_trained ? 'trained network' : 'untrained network, flies like classic');
    $('adb-lights').innerHTML = out.join('');
  }
  function onFrame(m) {
    if (!m.utc) return; const t = m.t_s, h = Math.floor(t / 3600), mi = Math.floor(t % 3600 / 60), se = Math.floor(t % 60);
    const [d, tm] = m.utc.replace('Z', '').split('T'); $('adb-utc').innerHTML = `<span class="d">${d} </span>${tm}`; /* the date hides first on narrow windows */ $('adb-met').textContent = `T+ ${String(h).padStart(2, '0')}:${String(mi).padStart(2, '0')}:${String(se).padStart(2, '0')}`;
    $('adb-phase').textContent = PH[m.phase] || m.phase; $('adb-warp').textContent = m.sim ? `${m.sim.playing ? '' : '❚❚ '}${m.sim.warp}×` : '—';
    $('adb-live').classList.add('on'); clearTimeout(onFrame.t); onFrame.t = setTimeout(() => $('adb-live').classList.remove('on'), 3000); lights(m);
  }
  lights(null);
  (function connect() { const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ui`);
    ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === 'frame' && m.utc) onFrame(m); }; ws.onclose = () => setTimeout(connect, 1500); })();
  window.ADB = { onFrame };
})();
