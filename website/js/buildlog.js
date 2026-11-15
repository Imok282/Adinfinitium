// Build log ticker: one line of real counts from the project's git history (assets/build_log.json,
// provenance "observed", made by tools/export_build_log.py from `git log --all`), counted up with scroll.
//
//   import { mountBuildLog } from './buildlog.js';
//   const bl = mountBuildLog(el, data);   // returns null when el is missing
//   bl.setProgress(p);                    // p in [0,1]: share of the history replayed (1 = every commit)
//
// No dependencies; text only through textContent.

const LANE_NAME = { imok: 'Imok', nimansh: 'Nimansh', team: 'AI coding agents alone' };

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

export function mountBuildLog(root, data, { reducedMotion = false } = {}) {
  if (!root || !data || !Array.isArray(data.commits)) return null;
  const commits = data.commits;
  const N = commits.length;
  const keys = ['imok', 'nimansh', 'team'];
  const cum = { imok: [0], nimansh: [0], team: [0], agent: [0] };
  commits.forEach((c, i) => {
    for (const k of keys) cum[k].push(cum[k][i] + (c.lane === k ? 1 : 0));
    cum.agent.push(cum.agent[i] + (c.agent ? 1 : 0));
  });

  root.textContent = '';
  root.classList.add('bl-tick');
  const line = el('p', 'bl-line');
  const b = {};
  const num = k => (b[k] = el('b', null, '0'));
  line.append(num('all'), ` / ${N} commits`);
  for (const k of keys) line.append(el('span', 'bl-sep', ' · '), `${LANE_NAME[k]} `, num(k));
  line.append(el('span', 'bl-sep', ' · '), num('agent'), ' with an AI coding agent');
  root.appendChild(line);

  let shown = -1;
  function setProgress(p) {
    const n = Math.max(0, Math.min(N, Math.round((reducedMotion ? 1 : p) * N)));
    if (n === shown) return;
    shown = n;
    b.all.textContent = String(n);
    for (const k of keys) b[k].textContent = String(cum[k][n]);
    b.agent.textContent = String(cum.agent[n]);
  }
  setProgress(0);
  return { setProgress, total: N, counts: { imok: cum.imok[N], nimansh: cum.nimansh[N], team: cum.team[N], agent: cum.agent[N] } };
}
