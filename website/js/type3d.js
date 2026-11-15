// 3D type for the home page (agent D). Real extruded headlines (three.js TextGeometry, small bevel) in Instrument Sans 500,
// with the accent phrase in Newsreader Italic 300. Fonts: assets/fonts3d/*.typeface.json (ASCII subset converted from the
// Google Fonts TTFs, SIL OFL). Lit by the scene's own lights (use the same sun DirectionalLight).
//
// API
//   import { loadType3DFonts, createType3D, type3DSupported } from './type3d.js';
//   if (type3DSupported()) {
//     const fonts = await loadType3DFonts();                       // { font, accentFont }
//     const T = createType3D(THREE, fonts);                        // optional 3rd arg: { color, accentColor }
//     const g = T.makeLine('Autonomy that lands where *missions survive.*', { size: 0.6, depth: 0.08 });
//     scene.add(g);                                                // position / scale the group like any Object3D
//     // every frame: p = 0..1 progress of this line's beat, pointer = { x, y } in -1..1 (or null)
//     T.update(g, p, pointer, { instant: false });
//   }
// Accent: wrap words in *...* in the text, or pass { accentFrom: n } (index of the first accent word, to the end).
// '\n' breaks lines. Timeline of p: 0 -> 0.32 words rise + rotate in from depth (staggered), hold, 0.72 -> 1 words fly past
// the camera and dissolve (staggered). Group tilts slightly with pointer and with p. Keep the real headline in the HTML,
// visually hidden, for SEO + screen readers; the 3D group is decoration (no text for assistive tech).

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const easeOut = x => 1 - Math.pow(1 - x, 3);
const easeIn = x => x * x * x;

export function type3DSupported() {
  try {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
    if (innerWidth <= 760) return false;
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch { return false; }
}

export async function loadType3DFonts(base = new URL('../assets/fonts3d/', import.meta.url).href) {
  const { FontLoader } = await import('three/addons/loaders/FontLoader.js');
  const L = new FontLoader();
  const load = f => new Promise((res, rej) => L.load(base + f, res, undefined, rej));
  const [font, accentFont] = await Promise.all([load('instrument_sans_500.typeface.json'), load('newsreader_italic_300.typeface.json')]);
  return { font, accentFont };
}

export function createType3D(THREE, { font, accentFont }, { color = '#f0eee6', accentColor = '#e8b9a0' } = {}) {
  let TextGeometry = null;
  const ready = import('three/addons/geometries/TextGeometry.js').then(m => { TextGeometry = m.TextGeometry; });

  function spaceW(f, size) {
    const g = f.data.glyphs[' '];
    return ((g ? g.ha : f.data.resolution * 0.25) / f.data.resolution) * size;
  }
  function parse(text, accentFrom) {
    const lines = String(text).split('\n');
    let inAcc = false, k = 0;
    return lines.map(line => line.trim().split(/\s+/).filter(Boolean).map(w => {
      let word = w, acc = inAcc;
      if (word.startsWith('*')) { word = word.slice(1); acc = true; inAcc = true; }
      if (word.endsWith('*')) { word = word.slice(0, -1); inAcc = false; }
      if (accentFrom != null && k >= accentFrom) acc = true;
      k++;
      return { word, acc };
    }));
  }

  function build(group, text, { accentFrom = null, size = 1, depth = 0.1, lineHeight = 1.18, bevel = true, align = 'center' } = {}) {
    const lines = parse(text, accentFrom);
    const words = [];
    const rows = lines.map(ws => {
      let x = 0;
      const items = ws.map(({ word, acc }) => {
        const f = acc ? accentFont : font;
        const s = acc ? size * 1.06 : size;              // the serif reads a touch smaller: nudge it up
        const geo = new TextGeometry(word, {
          font: f, size: s, height: depth * size, depth: depth * size, curveSegments: 7,
          bevelEnabled: bevel, bevelThickness: size * 0.018, bevelSize: size * 0.011, bevelSegments: 2,
        });
        geo.computeBoundingBox();
        const bb = geo.boundingBox, w = bb.max.x - bb.min.x;
        geo.translate(-bb.min.x - w / 2, 0, -(depth * size) / 2);     // pivot at the word's centre (baseline height)
        const mat = new THREE.MeshStandardMaterial({ color: acc ? accentColor : color, roughness: acc ? 0.42 : 0.55, metalness: 0.05, transparent: true, opacity: 1 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = false; mesh.receiveShadow = false;
        const pivot = new THREE.Group();
        pivot.add(mesh);
        const item = { pivot, mesh, w, x: x + w / 2 };
        x += w + spaceW(f, s) * 1.0;
        return item;
      });
      const width = Math.max(0, x - (items.length ? spaceW(font, size) : 0));
      return { items, width };
    });
    rows.forEach((r, li) => {
      const off = align === 'center' ? -r.width / 2 : 0;
      const y = -li * size * lineHeight + ((rows.length - 1) * size * lineHeight) / 2 - size * 0.36;
      r.items.forEach(it => {
        it.home = new THREE.Vector3(it.x + off, y, 0);
        it.pivot.position.copy(it.home);
        group.add(it.pivot);
        words.push(it);
      });
    });
    group.userData.type3d = { words, size, tilt: { x: 0, y: 0 } };
    group.userData.width = Math.max(...rows.map(r => r.width), 0);
    group.userData.height = rows.length * size * lineHeight;
  }

  return {
    ready,
    // Group of per-word meshes; geometry is built as soon as TextGeometry has loaded (await T.ready to be sure).
    makeLine(text, opts = {}) {
      const group = new THREE.Group();
      group.name = 'type3d:' + String(text).replace(/\*/g, '');
      if (TextGeometry) build(group, text, opts); else ready.then(() => build(group, text, opts));
      return group;
    },
    update(group, p, pointer = null, { instant = false } = {}) {
      const st = group.userData.type3d;
      if (!st) return;
      const n = st.words.length, S = st.size;
      st.words.forEach((it, i) => {
        const f = n > 1 ? i / (n - 1) : 0;
        const qi = easeOut(clamp((p - f * 0.14) / 0.18));             // entrance, staggered left -> right
        const qo = easeIn(clamp((p - 0.72 - f * 0.1) / 0.18));         // exit: fly past + dissolve
        const pv = it.pivot;
        pv.position.set(it.home.x * (1 + qo * 0.35), it.home.y - (1 - qi) * S * 0.9 + qo * S * 0.25, -(1 - qi) * S * 3 + qo * S * 7);
        pv.rotation.set(-(1 - qi) * 1.05 + qo * 0.5, (1 - qi) * (f - 0.5) * 0.5, 0);
        const sc = 1 + qo * 0.25;
        pv.scale.set(sc, sc, sc);
        it.mesh.material.opacity = clamp(qi * 1.4) * (1 - qo);
        pv.visible = it.mesh.material.opacity > 0.003;
      });
      // slight perspective tilt: pointer + scroll
      const tx = (pointer ? -pointer.y * 0.09 : 0) + (p - 0.5) * 0.16, ty = pointer ? pointer.x * 0.14 : 0;
      const k = instant ? 1 : 0.08;
      st.tilt.x += (tx - st.tilt.x) * k; st.tilt.y += (ty - st.tilt.y) * k;
      group.rotation.x = st.tilt.x; group.rotation.y = st.tilt.y;
    },
    dispose(group) {
      const st = group.userData.type3d;
      if (st) st.words.forEach(it => { it.mesh.geometry.dispose(); it.mesh.material.dispose(); });
      group.clear();
    },
  };
}
