// Loader for the compact ghost-trajectory files made by tools/export_ghosts_web.py
//   assets/data/ghosts_<kind>_<site>.json  header (counts, frame, provenance, run table)
//   assets/data/ghosts_<kind>_<site>.bin   uint16 LE, 4 per point (x, y, z, t), value = offset + q * scale
// Returns { head, pts: Float32Array (x, y, z, t per point, tile ENU metres / seconds), runs: [{policy, outcome, ok, best, start, n, miss}] }
const cache = {};
export function loadGhosts(kind, site) {
  const tag = `ghosts_${kind}_${site}`;
  return (cache[tag] ||= Promise.all([
    fetch(`assets/data/${tag}.json`).then(r => { if (!r.ok) throw new Error(tag + '.json ' + r.status); return r.json(); }),
    fetch(`assets/data/${tag}.bin`).then(r => { if (!r.ok) throw new Error(tag + '.bin ' + r.status); return r.arrayBuffer(); }),
  ]).then(([head, buf]) => {
    const q = new Uint16Array(buf), { offset: o, scale: s } = head.encoding;
    const pts = new Float32Array(q.length);
    for (let i = 0; i < q.length; i += 4) {
      pts[i] = o[0] + q[i] * s[0]; pts[i + 1] = o[1] + q[i + 1] * s[1];
      pts[i + 2] = o[2] + q[i + 2] * s[2]; pts[i + 3] = o[3] + q[i + 3] * s[3];
    }
    const okSet = new Set(head.ok_outcomes);
    const runs = head.runs.map(([pi, oi, flags, start, n, miss]) => ({
      policy: head.policies[pi], outcome: head.outcomes[oi], ok: okSet.has(head.outcomes[oi]),
      best: !!(flags & 1), flags, start, n, miss,
    }));
    return { head, pts, runs };
  }));
}

// header only (small), for texts
const hcache = {};
export function loadGhostHead(kind, site) {
  const tag = `ghosts_${kind}_${site}`;
  return (hcache[tag] ||= fetch(`assets/data/${tag}.json`).then(r => { if (!r.ok) throw new Error(tag + '.json ' + r.status); return r.json(); }));
}

// Fill [data-g] elements inside root from the header, so the page never disagrees with the data file:
//   runs | okof:<policy> | n:<policy> | ok:<policy> | pct:<policy> | rate:<failure> | out:<outcome> | src
export function fillCounts(root, head) {
  const fmt = n => Number(n).toLocaleString('en');
  root.querySelectorAll('[data-g]').forEach(el => {
    const [k, a] = el.dataset.g.split(':');
    let v = null;
    if (k === 'runs') v = fmt(head.n_runs);
    else if (k === 'okof') { const c = countOf(head, a); v = `${fmt(c.ok)} / ${fmt(c.total)}`; }
    else if (k === 'n') v = fmt(countOf(head, a).total);
    else if (k === 'ok') v = fmt(countOf(head, a === '*' ? null : a).ok);
    else if (k === 'bad') { const c = countOf(head, a === '*' ? null : a); v = fmt(c.total - c.ok); }
    else if (k === 'pct') { const c = countOf(head, a); v = c.total ? `${Math.round(100 * c.ok / c.total)} %` : null; }
    else if (k === 'rate' && head.failure_rates) v = `${Math.round(100 * head.failure_rates[a])} %`;
    else if (k === 'out') v = fmt(head.counts_total[a] || 0);
    else if (k === 'src') v = head.meta.source;
    if (v != null) el.textContent = v;
  });
}

// counts helper: { total, ok } for one policy (or all when policy is null), straight from the header
export function countOf(head, policy = null) {
  let total = 0, ok = 0;
  for (const [p, oc] of Object.entries(head.counts)) {
    if (policy && p !== policy) continue;
    for (const [o, n] of Object.entries(oc)) { total += n; if (head.ok_outcomes.includes(o)) ok += n; }
  }
  return { total, ok };
}
