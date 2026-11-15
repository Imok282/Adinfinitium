"""Copy the shared nav / waitlist band / footer from tools/shell.html into every page (between their SHELL markers,
or replacing <header class="topnav">…</header> and <footer class="footer">…</footer> if markers are missing).
Pages in research/ carry <base href="../"> so every relative URL (links, CSS, JS, fetch) resolves from website/;
their in-page "#anchor" links in the shell are rewritten to research/<page>.html#anchor. Usage: python3 website/tools/sync_shell.py"""
import re, pathlib
W = pathlib.Path(__file__).resolve().parent.parent
shell = (W / 'tools/shell.html').read_text()
def block(name):
    a = shell.index(f'<!-- BEGIN:SHELL-{name} -->'); b = shell.index(f'<!-- END:SHELL-{name} -->') + len(f'<!-- END:SHELL-{name} -->')
    return shell[a:b]
def relink(html, self_path):
    if not self_path: return html
    return re.sub(r'href="#([^"]*)"', lambda m: f'href="{self_path}#{m.group(1)}"', html)
def put(s, name, fallback_re):
    blk = block(name)
    a, b = f'<!-- BEGIN:SHELL-{name} -->', f'<!-- END:SHELL-{name} -->'
    if a in s: i = s.index(a); j = s.index(b) + len(b); return s[:i] + blk + s[j:]
    m = re.search(fallback_re, s, re.S)
    return s[:m.start()] + blk + s[m.end():] if m else s
pages = [p for p in W.glob('*.html')] + [p for p in (W / 'research').glob('*.html')] if (W / 'research').exists() else list(W.glob('*.html'))
for p in pages:
    if p.name.startswith('_'): continue
    s = p.read_text(); pre = f'research/{p.name}' if p.parent.name == 'research' else ''
    s2 = put(s, 'nav', r'<header class="topnav".*?</header>')
    s2 = put(s2, 'footer', r'<footer class="footer">.*?</footer>')
    if '<!-- WAITLIST -->' in s2: s2 = s2.replace('<!-- WAITLIST -->', block('waitlist'))
    elif '<!-- BEGIN:SHELL-waitlist -->' in s2: s2 = put(s2, 'waitlist', r'$^')
    elif 'NO-WAITLIST' not in s2: s2 = s2.replace('<!-- BEGIN:SHELL-footer -->', block('waitlist') + '\n<!-- BEGIN:SHELL-footer -->', 1)   # every page ends with the waitlist band
    # re-prefix the shell blocks only
    for name in ('nav', 'footer', 'waitlist'):
        blk = block(name)
        if blk in s2: s2 = s2.replace(blk, relink(blk, pre))
    if s2 != s: p.write_text(s2); print('synced', p.relative_to(W))
