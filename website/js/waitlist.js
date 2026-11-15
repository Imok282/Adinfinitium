// Waitlist form: VISUAL ONLY for now. It validates the email in the browser and shows "Thanks — we'll be in touch."
// It stores and sends NOTHING: no fetch, no beacon, no localStorage.
// TODO: wire to a form service (e.g. a hosted form endpoint) when the company has one.
export function mountWaitlist(root = document) {
  root.querySelectorAll('form[data-waitlist]').forEach(f => {
    f.addEventListener('submit', e => {
      e.preventDefault();
      const input = f.querySelector('input[type=email]'), ok = f.querySelector('.waitlist-ok');
      if (!input.value || !input.checkValidity()) { input.focus(); input.setAttribute('aria-invalid', 'true'); return; }
      input.removeAttribute('aria-invalid');
      f.classList.add('done'); input.value = ''; input.disabled = true; f.querySelector('button').disabled = true;
      ok.hidden = false;
    });
  });
}
mountWaitlist();
