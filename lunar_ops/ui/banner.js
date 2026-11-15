/* Copyright (c) 2026, Adinfinitium team. ISC-style permission notice as in index.html. */
/* Company banner on every screen: brand, UTC and mission clocks, phase, time warp and the subsystem lights, all from the
   live feed (/ws/ui). Lights are computed, never decorative: GNC (pointing error), PROP (propellant), ENGINES (healthy),
   COMMS / SUN / TERRAIN (Earth link, sunlight, safety at the target, from the simulator), AI PILOT (on, trained).
   Usage: <div id="adb" data-screen="Mission wall"></div> <link rel="stylesheet" href="/banner.css"> <script src="/banner.js"></script> */
(() => {
  const host = document.getElementById('adb'); if (!host) return;
  host.classList.add('adb');
  const screen = host.dataset.screen || 'Mission';
  // our own mark: the Moon, an orbit and the lander on it (no agency logos); replaced by /assets/brand/logo.svg if present
  const MARK = '<svg viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="13" fill="#000000" stroke="#8EB69B" stroke-width="1.6"/>'
    + '<circle cx="19" cy="20" r="2.6" fill="#235347"/><circle cx="28" cy="29" r="3.4" fill="#235347"/><circle cx="29" cy="18" r="1.6" fill="#235347"/>'
    + '<ellipse cx="24" cy="24" rx="22" ry="8" fill="none" stroke="#DAF1DE" stroke-width="1.2" transform="rotate(-24 24 24)" stroke-dasharray="2 3"/>'
    + '<circle cx="43" cy="16" r="2.6" fill="#DAF1DE"><animateTransform attributeName="transform" type="rotate" from="0 24 24" to="360 24 24" dur="18s" repeatCount="indefinite"/></circle></svg>';
  host.innerHTML = `<div class="mark"><span id="adb-logo">${MARK}</span><div class="word"><b>ADINFINITIUM</b><small>Lunar south pole · ${screen}</small></div></div>
    <div class="sep"></div><div class="clk"><span class="k">UTC</span><span class="v n" id="adb-utc">—</span></div>
    <div class="clk"><span class="k">Mission time</span><span class="v n" id="adb-met">—</span></div><div class="sep"></div>
    <div class="clk"><span class="k"><span class="live" id="adb-live"></span>Phase</span><span class="v txt" id="adb-phase">Waiting for the simulator</span></div>
    <div class="clk"><span class="k">Time warp</span><span class="v n" id="adb-warp">—</span></div>
    <div class="lights" id="adb-lights"></div>`;
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
    $('adb-utc').textContent = m.utc.replace('T', ' ').replace('Z', ''); $('adb-met').textContent = `T+ ${String(h).padStart(2, '0')}:${String(mi).padStart(2, '0')}:${String(se).padStart(2, '0')}`;
    $('adb-phase').textContent = PH[m.phase] || m.phase; $('adb-warp').textContent = m.sim ? `${m.sim.playing ? '' : '❚❚ '}${m.sim.warp}×` : '—';
    $('adb-live').classList.add('on'); clearTimeout(onFrame.t); onFrame.t = setTimeout(() => $('adb-live').classList.remove('on'), 3000); lights(m);
  }
  lights(null);
  (function connect() { const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ui`);
    ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === 'frame' && m.utc) onFrame(m); }; ws.onclose = () => setTimeout(connect, 1500); })();
  window.ADB = { onFrame };
})();
