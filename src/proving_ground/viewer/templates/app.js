(function () {
  'use strict';
  var root = document.documentElement;
  root.classList.add('js');

  /* ---- tabs with hash routing ---- */
  var tabs = [].slice.call(document.querySelectorAll('[role="tab"]'));
  var panels = [].slice.call(document.querySelectorAll('section.view'));
  function show(id, focus) {
    var found = false;
    panels.forEach(function (p) {
      var on = p.id === id;
      p.classList.toggle('active', on);
      found = found || on;
    });
    if (!found) { id = panels.length ? panels[0].id : ''; panels.forEach(function (p) { p.classList.toggle('active', p.id === id); }); }
    tabs.forEach(function (t) {
      var on = t.getAttribute('aria-controls') === id;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.setAttribute('tabindex', on ? '0' : '-1');
      if (on && focus) { t.focus(); }
    });
  }
  function fromHash() { show((location.hash || '').replace(/^#/, ''), false); }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function (e) { e.preventDefault(); location.hash = t.getAttribute('aria-controls'); });
    t.addEventListener('keydown', function (e) {
      var n = -1;
      if (e.key === 'ArrowRight') { n = (i + 1) % tabs.length; }
      else if (e.key === 'ArrowLeft') { n = (i - 1 + tabs.length) % tabs.length; }
      else if (e.key === 'Home') { n = 0; }
      else if (e.key === 'End') { n = tabs.length - 1; }
      if (n >= 0) { e.preventDefault(); location.hash = tabs[n].getAttribute('aria-controls'); tabs[n].focus(); }
    });
  });
  window.addEventListener('hashchange', fromHash);
  fromHash();

  /* ---- sortable tables (textContent only) ---- */
  [].slice.call(document.querySelectorAll('table.sortable')).forEach(function (tbl) {
    var heads = [].slice.call(tbl.querySelectorAll('thead th'));
    heads.forEach(function (th, col) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = th.textContent;
      b.setAttribute('aria-label', 'Sort by ' + th.textContent);
      th.textContent = '';
      th.appendChild(b);
      var asc = true;
      b.addEventListener('click', function () {
        var body = tbl.tBodies[0];
        var rows = [].slice.call(body.rows);
        rows.sort(function (r1, r2) {
          var a = r1.cells[col].textContent.trim(), c = r2.cells[col].textContent.trim();
          var na = parseFloat(a), nc = parseFloat(c);
          var r = (!isNaN(na) && !isNaN(nc)) ? na - nc : a.localeCompare(c);
          return asc ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        asc = !asc;
      });
    });
  });

  /* ---- optional in-browser audit-chain verification (Web Crypto) ---- */
  function cmp(a, b) {
    var x = Array.from(a), y = Array.from(b);
    for (var i = 0; i < Math.min(x.length, y.length); i++) {
      var d = x[i].codePointAt(0) - y[i].codePointAt(0);
      if (d !== 0) { return d; }
    }
    return x.length - y.length;
  }
  function canon(v) {
    if (v === null) { return 'null'; }
    if (Array.isArray(v)) { return '[' + v.map(canon).join(',') + ']'; }
    if (typeof v === 'object') {
      return '{' + Object.keys(v).sort(cmp).map(function (k) { return JSON.stringify(k) + ':' + canon(v[k]); }).join(',') + '}';
    }
    return JSON.stringify(v);
  }
  function sha256hex(s) {
    return crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)).then(function (buf) {
      return Array.from(new Uint8Array(buf)).map(function (b) { return ('0' + b.toString(16)).slice(-2); }).join('');
    });
  }
  var btn = document.getElementById('verify-btn');
  var out = document.getElementById('verify-result');
  var dataEl = document.getElementById('audit-data');
  if (btn) {
    if (!(window.crypto && window.crypto.subtle && window.TextEncoder) || !dataEl) {
      btn.classList.add('hidden');
      var note = document.getElementById('verify-unavailable');
      if (note) { note.classList.remove('hidden'); }
    } else {
      btn.addEventListener('click', function () {
        var entries = JSON.parse(dataEl.textContent);
        var prev = '0'.repeat(64);
        var chain = Promise.resolve(null);
        entries.forEach(function (e, i) {
          chain = chain.then(function (bad) {
            if (bad) { return bad; }
            var body = {};
            Object.keys(e).forEach(function (k) { if (k !== 'entry_hash') { body[k] = e[k]; } });
            return sha256hex(canon(body)).then(function (h) {
              if (e.seq !== i + 1) { return 'sequence gap at entry ' + (i + 1); }
              if (e.prev_entry_hash !== prev) { return 'broken link at seq ' + e.seq; }
              if (h !== e.entry_hash) { return 'hash mismatch at seq ' + e.seq; }
              prev = e.entry_hash;
              return null;
            });
          });
        });
        chain.then(function (bad) {
          out.textContent = bad ? ('✖ chain BROKEN in this browser: ' + bad) : ('✔ chain verified in this browser (' + entries.length + ' entries, head ' + prev.slice(0, 12) + ')');
        }, function (err) { out.textContent = 'verification error: ' + err; });
      });
    }
  }
})();
