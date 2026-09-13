/* feature_extract GUI: two views, Library (the slice: prompts | features) and Ingest (corpora, import, profiles, jobs), plus settings. Everything reads the store through /api. */
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => (n == null ? '' : Number(n).toLocaleString('en-US'));
const pct = x => (x == null ? '' : Math.round(100 * x) + '%');
const api = async (path, opts) => { const r = await fetch(path, opts); if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r.json(); };
const post = (path, body) => api(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
const href = (path, q = {}) => { const p = new URLSearchParams(); for (const [k, v] of Object.entries(q)) if (v) p.set(k, v); const s = p.toString(); return `#${path}${s ? '?' + s : ''}`; };
const unknown = (v, what = 'unknown') => (v == null || v === '' ? `<span class="muted"><i>${what}</i></span>` : esc(v));
let ES = null;
const pol = p => `<span class="pol ${p === 'forbid' ? 'forbid' : ''}">${p === 'forbid' ? 'must not' : 'must'}</span>`;

function parseHash() { const h = location.hash.slice(1) || '/library'; const [path, qs] = h.split('?'); return { parts: path.split('/').filter(Boolean), q: Object.fromEntries(new URLSearchParams(qs || '')) }; }
const GEAR = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"></path></svg>';
async function route() {
  const { parts, q } = parseHash(); const view = parts[0] || 'library';
  const lib = ['library', 'node', 'prompt', 'feature'].includes(view), ing = ['ingest', 'job'].includes(view);
  $('#nav').innerHTML = `<a href="#/library" class="${lib ? 'on' : ''}">Library</a><a href="#/ingest" class="${ing ? 'on' : ''}">Ingest</a>`;
  $('#gear').innerHTML = `<a href="#/settings" class="${view === 'settings' ? 'on' : ''}" title="settings">${GEAR}</a>`;
  if (ES) { ES.close(); ES = null; }
  const main = $('#main'); main.innerHTML = '<div class="loading">loading</div>'; window.scrollTo(0, 0);
  try {
    api('/api/spend').then(s => { $('#spend').textContent = `spent $${s.total.toFixed(2)}`; }).catch(() => {});
    if (view === 'library') await viewLibrary(main, q);
    else if (view === 'node') await viewNode(main, +parts[1], q);
    else if (view === 'prompt') await viewPrompt(main, decodeURIComponent(parts.slice(1).join('/')));
    else if (view === 'feature') await viewFeature(main, +parts[1]);
    else if (view === 'ingest') await viewIngest(main, parts[1] || 'corpora', q);
    else if (view === 'job') await viewJob(main, +parts[1]);
    else if (view === 'settings') await viewSettings(main, q);
    else main.innerHTML = '<p>No such page.</p>';
  } catch (e) { main.innerHTML = `<p class="err">${esc(e.message)}</p>`; console.error(e); }
}
const NOT_FIELDS = new Set(['kind', 'view', 'limit', 'profile']);
const fieldsOf = q => Object.keys(q).filter(f => !NOT_FIELDS.has(f) && q[f]);
const fieldQuery = q => Object.fromEntries(fieldsOf(q).map(f => [f, q[f]]));
const filterChips = (q, kind, view) => fieldsOf(q).flatMap(f => q[f].split(',').filter(Boolean).map(v => `<a class="chip on" href="${href('/library', { ...q, kind, view, [f]: q[f].split(',').filter(x => x !== v).join(',') })}" title="remove"><span>${esc(f)} ${esc(v)}</span><span class="n">×</span></a>`)).join(' ');

/* ---------- Library: one slice, two projections ---------- */
async function viewLibrary(main, q) {
  const kind = q.kind || 'guidance', view = q.view === 'prompts' ? 'prompts' : 'features';
  const s = await api('/api/cube?' + new URLSearchParams({ kind, view, ...fieldQuery(q) }));
  const nodeHref = id => href('/node/' + id, { kind, ...fieldQuery(q) });
  const toggle = (field, value, on) => { const cur = (q[field] || '').split(',').filter(Boolean); return href('/library', { ...q, kind, view, [field]: (on ? cur.filter(v => v !== value) : [...cur, value]).join(',') }); };
  const chip = (f, v) => `<a class="chip ${v.on ? 'on' : ''}" href="${toggle(f.field, v.value, v.on)}" data-v="${esc(v.value.toLowerCase())}"><span>${esc(v.value)}</span>${v.prompts == null ? '' : `<span class="n">${fmt(v.prompts)}</span>`}</a>`;
  const SHOW = 12;
  const fieldHtml = f => { const big = f.values.length > SHOW; return `<div class="field" data-field="${f.field}"><div class="t"><span>${esc(f.field)}${big ? ` <span class="muted">${f.values.length}</span>` : ''}</span>${big ? `<a href="#" class="more">all</a>` : ''}</div>${big ? `<input class="find" placeholder="find">` : ''}<div class="chips">${f.values.map((v, i) => `<span ${i >= SHOW && !v.on ? 'hidden' : ''} class="cw">${chip(f, v)}</span>`).join('')}</div></div>`; };
  const facets = view === 'features' ? s.facets : (window._facets || []);
  if (view === 'features') window._facets = s.facets;
  const polar = { field: 'polarity', values: ['require', 'forbid'].map(v => ({ value: v, prompts: null, on: (q.polarity || '').split(',').includes(v) })) };
  const left = `<div class="fields">
      <div class="search"><input type="search" id="lsearch" placeholder="search prompts and wordings, then Enter" value="${esc(q.text || '')}"></div>
      <div class="field"><div class="t"><span>kind</span></div><div class="chips">${['guidance', 'material'].map(k => `<a class="chip ${k === kind ? 'on' : ''}" href="${href('/library', { ...q, kind: k })}"><span>${k}</span></a>`).join('')}</div></div>
      ${facets.map(fieldHtml).join('')}${fieldHtml(polar)}</div>`;
  const seg = `<span class="seg"><a href="${href('/library', { ...q, kind, view: 'prompts' })}" class="${view === 'prompts' ? 'on' : ''}">prompts</a><a href="${href('/library', { ...q, kind, view: 'features' })}" class="${view === 'features' ? 'on' : ''}">features</a></span>`;
  let right, m = null, ccolor = {}, RX = 600, RY = 420;
  if (view === 'features') {
    const tr = (await Promise.all([api('/api/cube/map?' + new URLSearchParams({ kind, ring: q.ring || 'name', ...fieldQuery(q) })).then(x => { m = x; }), api('/api/cube/trees?' + new URLSearchParams({ kind, ...fieldQuery(q), ...(q.tree ? { tree: q.tree } : {}) }))]))[1];
    const sel = q.tree || '';
    const CPAL = ['#3B4FB8', '#B8741F', '#2E7D4F', '#7E3F8F', '#B8452B', '#2B8A9A', '#8A6D1F', '#C2418F', '#4F7F2B', '#6B6B6B'];
    ccolor = Object.fromEntries(m.corpora.map((c, i) => [c.name, CPAL[i % CPAL.length]]));
    const W = 1300, H = 1000, ax = c => (RX * 1.13 * c.x / m.R).toFixed(1), ay = c => (RY * 1.13 * c.y / m.R).toFixed(1);
    const autoThr = () => { const mx = m.globals.map(g => Math.max(...Object.values(g.prev))).sort((a, b) => b - a); return Math.max(2, Math.min(40, Math.ceil(100 * (mx[Math.min(44, mx.length - 1)] || 0.12)))); };   // the threshold that shows about 45 features
    const thr0 = q.thr ? +q.thr : autoThr();
    const mapHtml = m.corpora.length ? `<div class="mapwrap"><div class="maptools"><span class="muted" style="font-size:12px">show a feature present in at least</span><input type="range" id="mthr" min="2" max="40" value="${thr0}" style="width:130px"><span id="mthrv" class="muted" style="font-size:12px">${thr0}%</span><span class="muted" style="font-size:12px">of some corpus's prompts</span>
        <label class="muted" style="font-size:12px;margin-left:10px"><input type="checkbox" id="mlocal" ${sel ? 'checked' : ''}> features under no global${sel ? ` (${esc(sel)})` : ''}</label>
        <span class="muted" style="font-size:12px;margin-left:10px">ring <a href="${href('/library', { ...q, kind, view, ring: '' })}" class="${(q.ring || 'name') === 'name' ? 'on' : ''}" style="${(q.ring || 'name') === 'name' ? 'font-weight:600;color:var(--ink)' : ''}">by name</a> <a href="${href('/library', { ...q, kind, view, ring: 'similarity' })}" style="${q.ring === 'similarity' ? 'font-weight:600;color:var(--ink)' : ''}">by similarity</a></span>
        <span style="margin-left:auto"></span><a href="#" id="mzin" class="btn quiet small">+</a><a href="#" id="mzout" class="btn quiet small">−</a><a href="#" id="mzreset" class="btn quiet small">reset</a></div>
      <div class="mapgrid"><svg viewBox="${-W / 2} ${-H / 2} ${W} ${H}" class="map" id="map"><ellipse cx="0" cy="0" rx="${RX}" ry="${RY}" class="mring"></ellipse><g id="mlinks"></g><g id="mfeat"></g>
        ${m.corpora.map(c => `<a href="${href('/library', { ...q, kind, view, tree: sel === c.name ? '' : c.name })}" class="corp-a" data-c="${esc(c.name)}"><circle cx="${ax(c)}" cy="${ay(c)}" r="7" style="fill:${ccolor[c.name]}"></circle><text x="${ax(c)}" y="${+ay(c) + (c.y > 0 ? 24 : -14)}" class="manchor ${sel === c.name ? 'on' : ''}" text-anchor="middle" style="fill:${ccolor[c.name]}">${esc(c.name)} <tspan class="muted">${fmt(c.prompts)}</tspan></text></a>`).join('')}
      </svg><aside class="mapside" id="mapside"></aside></div>
      <div class="legend" style="flex-wrap:wrap">${m.corpora.map(c => `<span><i style="background:${ccolor[c.name]};border-radius:50%"></i>${esc(c.name)}</span>`).join('')}<span><i style="background:var(--muted);border-radius:50%"></i>three or more corpora</span><span class="muted">colour is the corpus that carries the feature most; size its largest share; hover or click a feature or a corpus</span></div></div>` : '<div class="muted">no codebook yet</div>';
    const leaf = f => `<div class="tnode leaf plain"><span class="read"><a href="${nodeHref(f.id)}">${esc(f.name)}</a>${f.global ? ` <span class="muted" style="font-size:12px">→ <a href="${nodeHref(f.global)}">${esc(f.global_name)}</a></span>` : ''}<div class="def">${esc(f.definition)}</div>${f.variants.length ? `<div class="mem">${f.variants.map(v => `<div><a href="${nodeHref(v.id)}">${esc(v.name)}</a> <span class="muted">${fmt(v.prompts)}</span></div>`).join('')}</div>` : ''}</span><span class="cnt">${fmt(f.prompts)}</span></div>`;
    const treeOf = t => `<details class="tnode section" ${q.tree ? 'open' : ''}><summary><b>${esc(t.corpus)}</b> <span class="muted">${fmt(t.in_slice)} of ${fmt(t.features)} features in the slice, ${fmt(t.aligned)} under a global</span></summary><div class="kids">${t.groups.map(g => `<details class="tnode section" open><summary>${esc(g.name)} <span class="muted">${g.features.length}${g.hidden ? ` (+${g.hidden} not in the slice)` : ''}</span></summary><div class="kids">${g.features.map(leaf).join('')}</div></details>`).join('') || '<span class="muted">nothing in the slice</span>'}</div></details>`;
    const corpusChips = tr.trees.map(t => `<a class="chip ${q.tree === t.corpus ? 'on' : ''}" href="${href('/library', { ...q, kind, view, tree: q.tree === t.corpus ? '' : t.corpus })}"><span>${esc(t.corpus)}</span><span class="n">${fmt(t.in_slice)}</span></a>`).join('');
    right = `<div class="muted" style="margin-bottom:10px">${fmt(s.prompts)} prompts, ${fmt(m.globals.length)} global features over ${m.corpora.length} corpora with a codebook${m.local.length ? `, ${fmt(m.local.length)} features under no global` : ''}${sel ? `; showing ${esc(sel)}: its globals in colour, its own features labelled` : ''}</div>
      <div class="block">${mapHtml}</div>
      <div class="block"><div class="t">each corpus's own codebook, with the prompts of the slice</div><div class="chips" style="margin-bottom:8px">${corpusChips}</div><div class="tree" style="max-height:none">${tr.trees.map(treeOf).join('') || '<span class="muted">none</span>'}</div></div>`;
  } else {
    right = `<div class="muted" style="margin-bottom:14px">${fmt(s.prompts)} prompts${s.listed < s.prompts ? `, first ${fmt(s.listed)} listed` : ''}</div>
      <table class="list"><tr><th>prompt</th><th class="n">chars</th><th class="n">readings</th><th class="n"></th></tr>
      ${s.list.map(p => `<tr><td><a href="${href('/prompt/' + encodeURIComponent(p.id))}" class="mono" style="font-size:11.5px">${esc(p.id)}</a> <span class="corp">${esc(p.corpus)}</span>${['role', 'family', 'stage'].filter(k => p.fields[k]).map(k => `<span class="tag">${esc(p.fields[k])}</span>`).join('')}<div style="font-size:12.5px;color:var(--ink2)">${esc(p.head)}…</div><div class="fchips">${p.features.map(f => `<a href="${nodeHref(f.id)}" class="corp">${esc(f.name)}</a>`).join('')}</div></td><td class="n">${fmt(p.chars)}</td><td class="n">${p.readings}</td><td class="n"><a href="#" class="muted rm" data-id="${esc(p.id)}">remove</a></td></tr>`).join('')}</table>`;
  }
  main.innerHTML = `<div class="cube">${left}<div><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px"><h1>Library</h1>${seg}</div>
    <div style="margin-bottom:10px">${filterChips(q, kind, view)}</div>${right}</div></div>`;
  if (view === 'features' && $('#map')) wireMap($('#map'), m, q.tree || '', nodeHref, ccolor, q, { RX, RY });
  for (const el of main.querySelectorAll('.field')) {
    const more = el.querySelector('.more'), find = el.querySelector('.find');
    if (more) more.onclick = e => { e.preventDefault(); const all = more.textContent === 'all'; el.querySelectorAll('.cw').forEach((w, i) => { w.hidden = !all && i >= SHOW && !w.querySelector('.on'); }); more.textContent = all ? 'fewer' : 'all'; };
    if (find) find.oninput = () => { const t = find.value.trim().toLowerCase(); el.querySelectorAll('.cw').forEach((w, i) => { w.hidden = t ? !w.querySelector('.chip').dataset.v.includes(t) : (i >= SHOW && !w.querySelector('.on')); }); };
  }
  $('#lsearch').onkeydown = e => { if (e.key === 'Enter') { location.hash = href('/library', { ...q, kind, view, text: e.target.value.trim() }); } };
  for (const a of main.querySelectorAll('a.rm')) a.onclick = async e => { e.preventDefault(); if (a.textContent !== 'sure?') { a.textContent = 'sure?'; return; } await post('/api/prompts/delete', { ids: [a.dataset.id] }); route(); };
}

/* the map, after FACET's vocabulary map: a feature is its label, sized by its largest share, placed at the share-weighted
   mean of its corpora's anchors pulled toward the centre by how many corpora carry it, then relaxed so no two labels
   overlap; links to its corpora and a side panel on hover or click; a threshold on the share keeps it readable; zoom and pan */
function wireMap(svg, m, sel, nodeHref, ccolor, q, { RX, RY }) {
  const box0 = svg.getAttribute('viewBox').split(' ').map(Number); let box = box0.slice(); let zoom = 1;
  const sx = x => RX * 1.13 * x / m.R, sy = y => RY * 1.13 * y / m.R;                       // the server's circle onto the ellipse
  const anchors = Object.fromEntries(m.corpora.map(c => [c.name, { ...c, x: sx(c.x), y: sy(c.y) }]));
  const gF = svg.querySelector('#mfeat'), gL = svg.querySelector('#mlinks'), side = $('#mapside'), thr = $('#mthr'), thrv = $('#mthrv'), locBox = $('#mlocal');
  let feats = [], pin = null, tierAt = 0;
  const effThr = () => (+thr.value) / 100 / zoom;                                            // semantic zoom: more features as you zoom in
  const prep = () => {
    const t = effThr();
    const gs = m.globals.filter(g => Math.max(...Object.values(g.prev)) >= t && (!sel || sel in g.share)).map(g => {
      const links = Object.entries(g.prev).filter(([c, v]) => v >= t * 0.5);
      const n = g.n_corpora, mx = Math.max(...Object.values(g.prev));                        // the server's x, y: angle from the anchors, radius from the entropy of the share
      const top = Object.entries(g.prev).sort((a, b) => b[1] - a[1])[0][0];
      return { ...g, links, mx, size: 7 + 17 * Math.sqrt(mx), color: n >= 3 ? 'var(--muted)' : ccolor[top], shared: n >= 3, tx: sx(g.x), ty: sy(g.y), x: sx(g.x), y: sy(g.y), glob: true };
    });
    const ls = (locBox.checked ? m.local.filter(f => (!sel || f.corpus === sel) && f.prev >= t) : []).map(f => ({ ...f, links: [[f.corpus, f.prev]], mx: f.prev, size: 6 + 12 * Math.sqrt(f.prev), color: ccolor[f.corpus], shared: false, tx: sx(f.x), ty: sy(f.y), x: sx(f.x), y: sy(f.y), glob: false, share: { [f.corpus]: 1 } }));
    return gs.concat(ls);
  };
  const boxOf = d => [d.size * 0.5 * d.name.length * 0.5 + 10, d.size * 0.75 * 0.5 + 6];    // half sizes of the drawn text (the svd session's measured factors)
  const collide = (steps, strength) => { for (let it = 0; it < steps; it++) for (let i = 0; i < feats.length; i++) for (let j = i + 1; j < feats.length; j++) { const a = feats[i], b = feats[j];
    const dx = b.x - a.x, dy = b.y - a.y, ox = a.hw + b.hw - Math.abs(dx), oy = a.hh + b.hh - Math.abs(dy);
    if (ox > 0 && oy > 0) { if (ox * 0.6 < oy) { const s = (ox / 2) * strength * (dx < 0 ? -1 : 1); a.x -= s; b.x += s; } else { const s = (oy / 2) * strength * (dy < 0 ? -1 : 1); a.y -= s; b.y += s; } } } };
  const clamp = () => { for (const d of feats) { if (!d.glob) continue; const r = Math.hypot(d.x / RX, d.y / RY); if (r > 0.92) { d.x *= 0.92 / r; d.y *= 0.92 / r; } } };
  const relax = () => {                                                                       // weak springs, a strong collider, many ticks; a last collision pass after the rim clamp
    for (const d of feats) { [d.hw, d.hh] = boxOf(d); }
    for (let it = 0; it < 600; it++) { const k = 0.05 * (1 - it / 700); for (const d of feats) { d.x += (d.tx - d.x) * k; d.y += (d.ty - d.y) * k; } collide(1, 0.5); }
    clamp(); collide(40, 0.3);
  };
  const render = () => {
    feats = prep(); relax();
    gL.innerHTML = feats.flatMap(d => d.links.map(([c, v]) => `<line class="mlink" data-f="${d.id}" data-c="${esc(c)}" x1="${d.x.toFixed(1)}" y1="${d.y.toFixed(1)}" x2="${anchors[c].x}" y2="${anchors[c].y}" stroke-width="${(0.6 + 4 * v).toFixed(1)}"></line>`)).join('');
    gF.innerHTML = feats.map(d => `<a href="${nodeHref(d.id)}" class="mf" data-id="${d.id}"><text x="${d.x.toFixed(1)}" y="${d.y.toFixed(1)}" text-anchor="middle" dominant-baseline="middle" font-size="${d.size.toFixed(1)}" style="fill:${d.color};${d.shared ? 'font-style:italic' : ''}${d.glob ? '' : ';opacity:.8'}">${esc(d.name)}</text></a>`).join('');
    gF.querySelectorAll('a.mf').forEach(a => { const d = feats.find(x => x.id === +a.dataset.id); a.onmouseenter = () => focus({ f: d }); a.onmouseleave = () => focus(null); a.onclick = e => { if (e.shiftKey || e.ctrlKey) return; e.preventDefault(); pin = pin && pin.f === d ? null : { f: d }; focus(null); }; });
    svg.querySelectorAll('a.corp-a').forEach(a => { const c = a.dataset.c; a.onmouseenter = () => focus({ c }); a.onmouseleave = () => focus(null); });
    focus(null);
  };
  const focus = h => {
    const s = h || pin;
    const hot = l => s && ((s.f && +l.dataset.f === s.f.id) || (s.c && l.dataset.c === s.c));
    gL.querySelectorAll('line').forEach(l => { l.style.opacity = s ? (hot(l) ? 0.9 : 0.03) : 0.18; });
    gF.querySelectorAll('a.mf').forEach(a => { const d = feats.find(x => x.id === +a.dataset.id); a.classList.toggle('dim', !!s && !((s.f && d === s.f) || (s.c && d.links.some(([c]) => c === s.c)))); });
    svg.querySelectorAll('a.corp-a').forEach(a => a.classList.toggle('dim', !!s && !((s.c && a.dataset.c === s.c) || (s.f && s.f.links.some(([c]) => c === a.dataset.c)))));
    sidePanel(s);
  };
  const bar = (lab, v, color, h) => `<div class="mrow"><span class="lab">${h ? `<a href="${h}">${lab}</a>` : lab}</span><span class="trk" style="width:${Math.max(4, 100 * v)}%;background:${color}"></span><span class="v muted">${Math.round(100 * v)}%</span></div>`;
  const sidePanel = s => {
    if (!s) { const shared = feats.filter(f => f.shared).sort((a, b) => b.mx - a.mx);
      side.innerHTML = `<b>${feats.length} features shown</b><div class="muted" style="font-size:12px;margin:4px 0 10px">of ${m.globals.length} globals${sel ? ` carried by ${esc(sel)}` : ''}; a feature is drawn when some corpus carries it in at least ${Math.round(100 * effThr())}% of its prompts in the slice${zoom > 1 ? ' (lowered by the zoom)' : ''}. A feature sits toward the corpora that carry it, nearer the centre the more evenly they share it. Colour is the corpus that carries it most; grey italic, three or more.</div>
        <div class="t muted" style="font-size:12px">shared by three or more corpora</div>${shared.slice(0, 30).map(f => bar(esc(f.name), f.mx, 'var(--muted)', nodeHref(f.id))).join('') || '<div class="muted">none at this threshold</div>'}`; return; }
    if (s.c) { const rows = feats.filter(f => f.links.some(([c]) => c === s.c)).map(f => [f, f.prev[s.c] ?? f.prev]).sort((a, b) => b[1] - a[1]);
      side.innerHTML = `<b style="color:${ccolor[s.c]}">${esc(s.c)}</b><div class="muted" style="font-size:12px;margin:4px 0 10px">${fmt(anchors[s.c].prompts)} prompts in the slice. <a href="${href('/library', { ...q, tree: s.c })}">view this codebook</a></div>${rows.slice(0, 24).map(([f, v]) => bar(esc(f.name), v, f.color, nodeHref(f.id))).join('')}`; return; }
    const f = s.f;
    side.innerHTML = `<b style="color:${f.color}"><a href="${nodeHref(f.id)}" style="color:inherit">${esc(f.name)}</a></b><div class="muted" style="font-size:12px;margin:4px 0 10px">${esc(f.definition || '')}${f.glob ? ` <span>${fmt(f.prompts)} prompts, ${f.n_corpora} corpora${f.group ? ', ' + esc(f.group) : ''}</span>` : ` ${esc(f.corpus)}, under no global, ${fmt(f.prompts)} prompts`}</div>
      <div class="t muted" style="font-size:12px">share of each corpus's prompts</div>${(f.glob ? Object.entries(f.prev) : [[f.corpus, f.prev]]).map(([c, v]) => bar(esc(c), v, ccolor[c], href('/library', { ...q, corpus: c, view: 'prompts' }))).join('')}`;
  };
  const apply = () => { svg.setAttribute('viewBox', box.join(' ')); const tier = Math.round(Math.log2(zoom) * 2); if (tier !== tierAt) { tierAt = tier; render(); } };
  const zoomAt = (f, px, py) => { const nz = Math.min(8, Math.max(1, zoom * f)); f = nz / zoom; box = [px - (px - box[0]) / f, py - (py - box[1]) / f, box[2] / f, box[3] / f]; zoom = nz; if (zoom === 1) box = box0.slice(); apply(); };
  const pt = e => { const r = svg.getBoundingClientRect(); return [box[0] + (e.clientX - r.left) / r.width * box[2], box[1] + (e.clientY - r.top) / r.height * box[3]]; };
  svg.addEventListener('wheel', e => { e.preventDefault(); const [px, py] = pt(e); zoomAt(e.deltaY < 0 ? 1.25 : 0.8, px, py); }, { passive: false });
  let drag = null;
  svg.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, box: box.slice(), moved: false }; });
  window.addEventListener('mousemove', e => { if (!drag) return; const r = svg.getBoundingClientRect(); const dx = (e.clientX - drag.x) / r.width * box[2], dy = (e.clientY - drag.y) / r.height * box[3]; if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) drag.moved = true; box = [drag.box[0] - dx, drag.box[1] - dy, box[2], box[3]]; apply(); });
  window.addEventListener('mouseup', () => { drag = null; });
  svg.addEventListener('click', e => { if (drag && drag.moved) { e.preventDefault(); e.stopPropagation(); } }, true);
  $('#mzin').onclick = e => { e.preventDefault(); zoomAt(1.5, box[0] + box[2] / 2, box[1] + box[3] / 2); };
  $('#mzout').onclick = e => { e.preventDefault(); zoomAt(1 / 1.5, box[0] + box[2] / 2, box[1] + box[3] / 2); };
  $('#mzreset').onclick = e => { e.preventDefault(); zoom = 1; box = box0.slice(); apply(); };
  thr.oninput = () => { thrv.textContent = thr.value + '%'; render(); };
  locBox.onchange = render;
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { pin = null; focus(null); } });
  render();
}

async function viewNode(main, id, q) {
  const kind = q.kind || 'guidance';
  const d = await api(`/api/cube/node/${id}?` + new URLSearchParams({ kind, ...fieldQuery(q) }));
  const nodeHref = i => href('/node/' + i, { kind, ...fieldQuery(q) });
  const wrow = w => `<details><summary><span>${esc(w.declaration)}</span><span class="n">${fmt(w.prompts)}</span><span class="n muted">${fmt(w.corpus_prompts)}</span></summary>
      <div class="quotes">${w.quotes.map(x => `<div class="quote"><a href="${href('/prompt/' + encodeURIComponent(x.prompt))}" class="mono" style="font-size:11.5px">${esc(x.prompt)}</a><div>${esc(x.text)}${x.chars > 320 ? '…' : ''}</div></div>`).join('') || '<span class="muted" style="font-size:12.5px">no quote in this slice</span>'}${w.prompts > w.quotes.length ? `<div class="muted" style="font-size:12px">${fmt(w.prompts - w.quotes.length)} more prompts carry it</div>` : ''}</div></details>`;
  const member = m => `<div class="block">${d.global ? `<div class="t"><span class="corp">${esc(m.corpus)}</span><a href="${nodeHref(m.id)}">${esc(m.name)}</a>, ${fmt(m.prompts)} prompts, ${m.wordings.length} wordings</div>` : ''}
    <div class="members"><div class="mh"><span>wording</span><span class="n">prompts here</span><span class="n">in its corpus</span></div>${m.wordings.slice(0, 80).map(wrow).join('')}${m.wordings.length > 80 ? `<details><summary class="muted" style="display:block;cursor:pointer;padding:6px 0">${m.wordings.length - 80} more wordings</summary>${m.wordings.slice(80).map(wrow).join('')}</details>` : ''}</div></div>`;
  main.innerHTML = `<div style="font-size:12.5px;color:var(--muted);margin-bottom:8px"><a href="${href('/library', { ...q, kind })}">Library</a> › ${d.group ? esc(d.group.name) + ' › ' : ''}${d.global ? 'global feature' : `<span class="corp">${esc(d.corpus)}</span>`}</div>
    <h1>${esc(d.name)}</h1><p class="lede">${esc(d.definition)}</p>
    <div class="muted" style="margin-bottom:14px">${fmt(d.prompts)} of ${fmt(d.selected)} prompts in the slice, ${fmt(d.readings)} readings${d.global ? `, ${d.members.length} per-corpus features` : d.global_of ? `, aligned to <a href="${nodeHref(d.global_of)}">${esc(d.global_name)}</a>` : ', not under a global yet'}${d.global ? '' : `, <a href="${href('/feature/' + d.id)}" class="muted">its codebook page</a>`}${filterChips(q, kind, 'features') ? ' ' + filterChips(q, kind, 'features') : ''}</div>
    <div class="node">${d.members.map(member).join('') || '<span class="muted">no readings in this slice</span>'}</div>`;
}

const fmtSec = s => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
/* ---------- one prompt ---------- */
function paint(text, spans) {
  const items = [];
  for (const a of spans) {
    if (a.kind === 'section') continue;
    const cls = a.kind === 'material' ? 'material' : (a.kind === 'unrefined' || a.kind === 'gap') ? 'gap' : ((a.readings[0] || {}).polarity === 'forbid' ? 'forbid' : '');
    items.push({ s: a.lo, e: a.hi, cls, path: a.path, title: a.note || a.path, sup: a.kind === 'atom' || a.kind === 'unrefined' });
  }
  items.sort((x, y) => x.s - y.s);
  let out = '', pos = 0;
  for (const it of items) { if (it.s < pos) continue; out += esc(text.slice(pos, it.s)); out += `<mark class="${it.cls}" data-path="${esc(it.path)}" title="${esc(it.title)}">${esc(text.slice(it.s, it.e))}</mark>${it.sup ? `<sup>${esc(it.path)}</sup>` : ''}`; pos = it.e; }
  return out + esc(text.slice(pos));
}
function buildTree(spans) {
  const nodes = new Map(spans.filter(a => a.kind !== 'gap').map(a => [a.path, { ...a, children: [] }]));
  const roots = [];
  for (const n of nodes.values()) { const i = n.path.lastIndexOf('.'); const parent = i < 0 ? null : nodes.get(n.path.slice(0, i)); (parent ? parent.children : roots).push(n); }
  const sortRec = ns => { ns.sort((a, b) => a.lo - b.lo); ns.forEach(n => sortRec(n.children)); };
  sortRec(roots);
  return roots;
}
function renderTree(nodes, text) {
  return nodes.map(n => {
    if (n.kind === 'section') return `<details class="tnode section" open><summary><span class="path">${esc(n.path)}</span><span class="muted">section · ${n.children.length} parts · ${fmt(n.hi - n.lo)} chars</span></summary><div class="kids">${renderTree(n.children, text)}</div></details>`;
    if (n.kind === 'material') return `<div class="tnode leaf material" data-path="${esc(n.path)}"><span class="path">${esc(n.path)}</span><span class="read">${(n.readings || []).map(r => `<span class="verb">${esc(r.verb)}</span> ${esc(r.object)}${r.qualifier ? ' <span class="muted">' + esc(r.qualifier) + '</span>' : ''}`).join('; ') || '<i>material</i>'} <span class="tag">${esc(n.note)}</span><br><span class="muted" style="font-size:12px">${esc(text.slice(n.lo, n.hi)).slice(0, 90)}</span></span></div>`;
    if (n.kind === 'unrefined') return `<div class="tnode leaf unrefined" data-path="${esc(n.path)}"><span class="path">${esc(n.path)}</span><span class="read"><i>unrefined</i> <span class="muted">${esc(text.slice(n.lo, n.hi)).slice(0, 90)}</span></span></div>`;
    const forbid = (n.readings[0] || {}).polarity === 'forbid';
    return `<div class="tnode leaf atom ${forbid ? 'forbid' : ''}" data-path="${esc(n.path)}"><span class="path">${esc(n.path)}</span><span>${n.readings.map(r => `<div class="read"><span class="verb">${esc(r.verb)}</span> ${esc(r.object)}${r.qualifier ? ` <span class="muted">${esc(r.qualifier)}</span>` : ''}${r.condition && r.condition !== 'always' ? ` <span class="muted">if ${esc(r.condition)}</span>` : ''}${r.polarity === 'forbid' ? ' ' + pol('forbid') : ''}</div>`).join('') || '<span class="muted"><i>no reading</i></span>'}${n.flags.length ? `<span class="tag">${esc(n.flags.join(' '))}</span>` : ''}</span></div>`;
  }).join('');
}
async function viewPrompt(main, pid) {
  const p = await api('/api/prompt/' + encodeURIComponent(pid));
  const d = p.decomp, sp = p.spans;
  const nAtoms = sp.filter(a => a.kind === 'atom').length, nReadings = sp.reduce((n, a) => n + a.readings.length, 0);
  const tree = buildTree(sp);
  main.innerHTML = `<h1 class="mono" style="font-size:20px">${esc(p.id)}</h1>
    <div class="facts" style="margin-bottom:16px"><span class="k">corpus</span><span>${esc(p.corpus)}</span><span class="k">source, recorded</span><span>${unknown(p.system)} ${p.domain ? '· ' + esc(p.domain) : ''} ${p.source_id ? '· <span class="mono" style="font-size:12px">' + esc(p.source_id) + '</span>' : ''}</span>
      <span class="k">task label</span><span>${unknown(p.task, 'unknown')}</span>
      ${p.meta && p.meta.harvest ? `<span class="k">unwrapped from</span><span>${esc(p.meta.harvest.wrapped)} <details style="display:inline"><summary class="muted" style="display:inline;cursor:pointer">show original</summary><pre class="mono" style="font-size:11.5px;white-space:pre-wrap;max-height:30vh;overflow:auto">${esc(p.meta.harvest.original)}</pre></details></span>` : ''}
      <span class="k">decomposition</span><span>${d ? (d.status === 'done' ? `coverage <b>${pct(d.coverage)}</b> of the instruction text · material ${pct(d.material_share)} · ${nAtoms} atoms · ${nReadings} readings · ${d.calls} calls, ${d.reasks} re-asks, ${d.seconds} s · ${esc(d.model)}` : `<span class="err">${esc(d.status)}: ${esc(d.error || '')}</span>`) : '<span class="muted"><i>not decomposed</i></span>'}</span></div>
    <div class="legend"><span><i style="background:var(--req-soft);border-bottom:1.5px solid var(--req)"></i>atom, must</span><span><i style="background:var(--for-soft);border-bottom:3px double var(--for)"></i>atom, must not</span><span><i style="background:var(--mat-soft)"></i>material</span><span><i style="background:var(--gap-soft);border-bottom:1.5px dashed var(--gap)"></i>unrefined, or a gap the model declined</span></div>
    <div class="cols2"><div class="rawtext" id="raw">${paint(p.text, sp)}</div>
      <div class="tree" id="tree">${renderTree(tree, p.text)}
      ${sp.filter(a => a.kind === 'gap').map(g => `<div class="tnode leaf gap" data-path="${esc(g.path)}"><span class="path">gap</span><span class="read"><i>declined</i> <span class="muted">${esc(p.text.slice(g.lo, g.hi)).slice(0, 120)}</span></span></div>`).join('')}</div></div>
    ${d && (d.failures.length || d.flags.length) ? `<details style="margin-top:18px;font-size:13px"><summary class="muted">calls and failures</summary><div class="mono" style="font-size:12px;margin-top:8px">${esc(d.flags.join(' '))}</div>${d.failures.map(f => `<div class="mono muted" style="font-size:12px">${esc(f)}</div>`).join('')}</details>` : ''}`;
  const hot = (path, on) => document.querySelectorAll(`[data-path="${CSS.escape(path)}"]`).forEach(el => el.classList.toggle('hot', on));
  document.querySelectorAll('[data-path]').forEach(el => { el.onmouseenter = () => hot(el.dataset.path, true); el.onmouseleave = () => hot(el.dataset.path, false); });
  document.querySelectorAll('#raw mark').forEach(el => el.onclick = () => { const n = document.querySelector(`#tree [data-path="${CSS.escape(el.dataset.path)}"]`); if (n) n.scrollIntoView({ behavior: 'smooth', block: 'center' }); });
  document.querySelectorAll('#tree .leaf').forEach(el => el.onclick = () => { const m = document.querySelector(`#raw mark[data-path="${CSS.escape(el.dataset.path)}"]`); if (m) m.scrollIntoView({ behavior: 'smooth', block: 'center' }); });
}

async function viewFeature(main, fid) {
  const r = await api('/api/feature/' + fid); const f = r.feature, cb = r.codebook;
  const byR = {}; for (const m of r.members) byR[m.id] = m;
  const quotes = {}; for (const x of r.readings) (quotes[x.realization] = quotes[x.realization] || []).push(x);
  const prompts = r.members.reduce((s, m) => s + m.prompts, 0), readings = r.members.reduce((s, m) => s + m.n, 0);
  const anchors = f.examples.map(id => byR[id] ? esc(byR[id].declaration) : `<span class="err">R${id} left this feature</span>`);
  main.innerHTML = `<div style="font-size:12.5px;color:var(--muted);margin-bottom:8px"><a href="${href('/library', { corpus: cb.corpus_name, kind: cb.kind })}">Library</a> › <span class="corp">${esc(cb.corpus_name)}</span> ${esc(cb.kind)} codebook${r.group ? ' › ' + (f.level === 'variant' ? `<a href="${href('/feature/' + r.group.id)}">${esc(r.group.name)}</a>` : esc(r.group.name)) : ''}</div>
    <h1>${esc(f.name)}</h1><p class="lede">${esc(f.definition)}</p>
    <div class="muted" style="margin-bottom:6px">${f.level === 'variant' ? 'a variant, ' : ''}${fmt(prompts)} prompts, ${fmt(readings)} readings, ${r.members.length} wordings${f.round ? `, added in round ${f.round}` : ''}${r.aligned ? (r.aligned.global ? `, aligned to <a href="${href('/node/' + r.aligned.global)}">${esc(r.aligned.global_name)}</a>` : r.aligned.note === 'specific' ? ', specific to this corpus at the seed' : ', open at the seed') : ''}</div>
    ${anchors.length ? `<div class="muted" style="font-size:12.5px;margin-bottom:6px">anchors: ${anchors.join('; ')}</div>` : ''}
    ${r.flags.length ? `<div style="font-size:12.5px;margin-bottom:6px">${r.flags.map(x => `<div><span class="err">${esc(x.verdict)}${x.standing ? ', standing' : ''}</span> ${x.verdict === 'indistinct' ? 'with <a href="' + href('/feature/' + (x.feature === f.id ? x.other : x.feature)) + '">' + esc(x.feature === f.id ? x.other_name : x.feature_name) + '</a>' : esc(x.declaration || '')} <span class="muted">${esc(x.verdict === 'split' ? JSON.parse(x.note || '{}').why || '' : x.note || '')}</span></div>`).join('')}</div>` : ''}
    <div class="node" style="margin-top:14px"><div class="members"><div class="mh"><span>wording</span><span class="n">prompts</span><span class="n">readings</span></div>
      ${r.members.map(m => `<details><summary><span>${esc(m.declaration)}${m.conditions.filter(c => c !== 'always').length ? ` <span class="muted" style="font-size:12px">when: ${esc(m.conditions.filter(c => c !== 'always').slice(0, 2).join('; '))}</span>` : ''}${m.confidence && m.confidence !== 'high' ? ` <span class="muted" style="font-size:12px">${esc(m.confidence)}</span>` : ''}</span><span class="n">${m.prompts}</span><span class="n">${m.n}</span></summary>
        <div class="quotes">${(quotes[m.id] || []).map(x => `<div class="quote"><a href="${href('/prompt/' + encodeURIComponent(x.prompt))}" class="mono" style="font-size:11.5px">${esc(x.prompt)}</a><div>${esc(x.text)}</div></div>`).join('') || '<span class="muted" style="font-size:12.5px">quotes beyond the first 300 readings are on the prompt pages</span>'}</div></details>`).join('')}</div></div>`;
}

/* ---------- Ingest: corpora, import, profiles, jobs ---------- */
const STAGE = { done: 'done', partial: 'part', none: '', running: 'run' };
const strip = st => `<span class="strip">${st.map(x => `<i class="${STAGE[x] || ''}" title="${x}"></i>`).join('')}</span>`;
const jstatus = s => `<span class="status ${s === 'running' ? 'run' : s === 'done' ? 'done' : s === 'failed' || s === 'stale' || s === 'lost' ? 'fail' : 'look'}">${esc(s)}</span>`;
async function viewIngest(main, sub, q) {
  const d = await api('/api/ingest' + (sub === 'corpora' && q.kind ? `?kind=${encodeURIComponent(q.kind)}` : ''));
  const tabs = ['corpora', 'import', 'profiles', 'jobs', 'history'].map(t => `<a href="#/ingest/${t}" class="${t === sub ? 'on' : ''}">${t[0].toUpperCase() + t.slice(1)}</a>`).join('');
  let body = '';
  if (sub === 'corpora') {
    const prof0 = d.profiles.find(p => p.name === 'default') || d.profiles[0], alignOn = prof0 && prof0.params.align === 'on';
    const pending = d.corpora.filter(c => (alignOn ? c.pending : c.pending_core) && !c.running).length, kind = q.kind || 'guidance';
    body = `<h1>Corpora</h1><p class="lede">A corpus is kept: edit it, add or remove prompts, and run it to bring its codebook and its alignment current.</p>
      <div class="chips" style="margin-bottom:12px">${['guidance', 'material'].map(k => `<a class="chip ${k === kind ? 'on' : ''}" href="${href('/ingest/corpora', { kind: k })}"><span>${k}</span></a>`).join('')}</div>
      <div class="block"><div class="t">stages: decomposed, codebook, aligned &nbsp;${strip(['done'])} done &nbsp;${strip(['partial'])} partial &nbsp;${strip(['running'])} running &nbsp;${strip(['none'])} not run</div>
      <table class="list"><tr><th>corpus</th><th class="n">prompts</th><th>stages</th><th class="n">decomposed</th><th class="n">features</th><th class="n">aligned</th><th>last job</th><th class="n">estimate</th><th></th><th></th></tr>
      ${d.corpora.map(c => `<tr data-c="${esc(c.name)}"><td><b>${esc(c.name)}</b><br><span class="muted" style="font-size:12px">${esc(c.domain || '')}${c.source ? ', ' + esc(c.source) : ''}</span></td><td class="n">${fmt(c.prompts)}</td><td>${strip(c.stages)}${c.alone && c.codebook ? ' <span class="muted" style="font-size:11.5px" title="alignment needs a second corpus with a codebook">alone</span>' : ''}</td><td class="n">${fmt(c.decomposed)}${c.coverage != null ? ` <span class="muted">${pct(c.coverage)}</span>` : ''}</td><td class="n">${c.codebook ? fmt(c.features) + (c.unassigned ? ` <span class="muted">+${fmt(c.unassigned)} to place</span>` : '') : '<span class="muted">none</span>'}</td><td class="n">${c.codebook ? `${fmt(c.aligned)} of ${fmt(c.features)}` : '<span class="muted">–</span>'}</td><td style="font-size:12.5px">${c.last_job ? `<a href="${href('/job/' + c.last_job.id)}">#${c.last_job.id}</a> ${esc(c.last_job.kind)} ${jstatus(c.last_job.status)} <span class="muted">$${(c.last_job.spent || 0).toFixed(2)}</span>` : '<span class="muted">none</span>'}</td><td class="n est" data-c="${esc(c.name)}"></td><td>${c.running ? '<span class="muted">running</span>' : `<a href="#" class="btn quiet small run">run</a>`}</td><td><a href="#" class="muted edit" style="font-size:12px">edit</a></td></tr>
        <tr class="editor" hidden><td colspan="9"><form class="form cedit" style="grid-template-columns:110px minmax(0,1fr)"><label>name</label><input type="text" name="name" value="${esc(c.name)}"><label>domain</label><input type="text" name="domain" value="${esc(c.domain || '')}"><label>tags</label><span><input type="text" name="tags" placeholder="field=value, field=value; an empty value removes the field" style="width:100%"><span class="muted" style="font-size:12px">on every prompt of the corpus; fields now: ${(c.tags || []).map(t => `${esc(t.field)} (${t.values})`).join(', ') || 'none'}</span></span><span></span><span><button class="btn quiet small" type="submit">save</button> <a href="#" class="muted del" style="margin-left:14px;font-size:12px">delete the corpus</a> <span class="muted cstat" style="font-size:12px"></span></span></form></td></tr>`).join('')}</table>
      <div style="margin-top:12px;display:flex;gap:12px;align-items:baseline"><a href="#" class="btn" id="runall">run all pending</a><span class="muted" style="font-size:12.5px">${pending} corpora with a stage to do, profile <select id="prof" style="padding:2px 6px">${d.profiles.map(p => `<option>${esc(p.name)}</option>`).join('')}</select>, estimate <b id="esttotal" style="color:var(--ink)">…</b></span><span id="rstat" class="muted"></span></div></div>`;
  } else if (sub === 'import') {
    body = `<h1>Import</h1><p class="lede">A file, a folder or a pasted prompt lands in a corpus, new or existing.</p>
      <div style="max-width:720px"><div class="drop" id="drop">drop a FACET prompts file, a zipped folder or a text file here, or click to choose<input type="file" id="file" hidden></div>
      <form id="iform" class="form" style="margin-top:12px"><label>or a path here</label><input type="text" name="path" placeholder="e.g. data/corpora/facet/text2sql.jsonl, a folder, or a zip" list="paths"><datalist id="paths"><option value="data/corpora/facet/text2sql.jsonl"><option value="data/corpora/facet/math.jsonl"><option value="data/corpora/facet/table-qa.jsonl"><option value="data/corpora/facet/science-quantitative.jsonl"><option value="data/corpora/facet/text2cypher.jsonl"><option value="data/corpora/facet/code-generation.jsonl"><option value="data/corpora/plain"></datalist>
        <label>into corpus</label><select name="into"><option value="">a new corpus</option>${d.corpora.map(c => `<option>${esc(c.name)}</option>`).join('')}</select>
        <label>name</label><input type="text" name="name" placeholder="for a new corpus">
        <label>domain</label><input type="text" name="domain" placeholder="tag every prompt with this domain; for a FACET jsonl also keeps only this domain">
        <label>or paste a prompt</label><textarea name="text"></textarea>
        <span></span><span><button class="btn" type="submit">import</button> <span id="istatus" class="muted"></span></span></form>
      ${d.imports && d.imports.length ? `<div class="block" style="margin-top:22px"><div class="t">imports</div><table class="list"><tr><th>when</th><th>corpus</th><th>from</th><th class="n">added</th><th class="n">skipped</th></tr>${d.imports.map(i => `<tr><td class="muted" style="font-size:12.5px">${esc(i.at)}</td><td>${esc(i.corpus_name)}</td><td style="font-size:12.5px">${esc(i.kind)}${i.path ? ` <span class="mono muted">${esc(i.path)}</span>` : ''}${i.domain ? ` <span class="muted">domain ${esc(i.domain)}</span>` : ''}</td><td class="n">${fmt(i.added)}</td><td class="n">${fmt(i.skipped)}</td></tr>`).join('')}</table></div>` : ''}</div>`;
  } else if (sub === 'profiles') {
    const cur = d.profiles.find(p => p.name === (q.profile || 'default')) || d.profiles[0];
    const row = (k, label, hint) => `<tr><td>${label}</td><td><input type="text" name="${k}" value="${esc(cur.params[k] ?? '')}" list="models" style="width:100%"></td><td class="muted" style="font-size:12px">${hint}</td></tr>`;
    body = `<h1>Profiles</h1><p class="lede">The settings a run uses. A corpus picks one when it runs.</p>
      <div style="max-width:800px"><div class="chips" style="margin-bottom:12px">${d.profiles.map(p => `<a class="chip ${p.name === cur.name ? 'on' : ''}" href="${href('/ingest/profiles', { profile: p.name })}"><span>${esc(p.name)}</span></a>`).join('')}<a class="chip" href="${href('/ingest/profiles', { profile: '__new' })}"><span>new</span></a></div>
      <form id="pform" class="panel"><div class="h"><b><input type="text" name="name" value="${esc(q.profile === '__new' ? '' : cur.name)}" placeholder="name" style="font-weight:600"></b></div>
      <table class="list"><tr><th>role</th><th>model</th><th></th></tr>
        ${row('decompose_model', 'decompose', 'stage 1')}${row('codebook_model', 'codebook', 'cold start and naming')}${row('batch_model', 'batch', 'assign and judge')}${row('judge_model', 'judge', 'empty: same as batch')}${row('embed_model', 'embed', 'local, sentence-transformers')}
        <tr><td>budget</td><td><input type="number" name="budget" value="${esc(cur.params.budget ?? '')}" step="1" placeholder="none"></td><td class="muted" style="font-size:12px">dollars per run; stops the run, keeps what it wrote</td></tr>
        <tr><td>workers</td><td><input type="number" name="workers" value="${esc(cur.params.workers ?? 128)}" min="1" max="512"></td><td class="muted" style="font-size:12px">calls in flight</td></tr>
        <tr><td>effort</td><td><select name="effort"><option ${cur.params.effort === 'low' ? 'selected' : ''}>low</option><option ${cur.params.effort === 'default' ? 'selected' : ''}>default</option></select></td><td class="muted" style="font-size:12px">reasoning effort for assign and judge; decomposition always reasons</td></tr>
        <tr><td>stage 3</td><td><select name="align"><option value="off" ${cur.params.align !== 'on' ? 'selected' : ''}>off</option><option value="on" ${cur.params.align === 'on' ? 'selected' : ''}>on</option></select></td><td class="muted" style="font-size:12px">align this corpus's features into the global library after its codebook; off by default, and needs a second corpus with a codebook</td></tr>
        <tr><td>kind</td><td><select name="kind">${['guidance', 'material', 'both'].map(k => `<option ${(cur.params.kind || 'guidance') === k ? 'selected' : ''}>${k}</option>`).join('')}</select></td><td class="muted" style="font-size:12px">which library a run builds and aligns: the guidance, the material, or both</td></tr>
        <tr><td>base url</td><td><input type="text" name="base_url" value="${esc(cur.params.base_url ?? '')}" placeholder="optional: another OpenAI-compatible server" style="width:100%"></td><td class="muted" style="font-size:12px">for every role</td></tr></table>
      <datalist id="models">${d.models.map(m => `<option value="${esc(m.model)}">${esc(m.endpoint)}</option>`).join('')}</datalist>
      <div style="margin-top:12px"><button class="btn quiet" type="submit">save</button> ${cur.name !== 'default' && q.profile !== '__new' ? `<a href="#" id="pdel" class="muted" style="margin-left:14px;font-size:12px">delete</a>` : ''} <span id="pstat" class="muted" style="font-size:12px"></span></div></form></div>`;
  } else if (sub === 'history') {
    const h = await api('/api/history');
    const kb = n => n == null ? '' : n < 1e6 ? `${Math.round(n / 1024)} kB` : `${(n / 1e6).toFixed(1)} MB`;
    body = `<h1>History</h1><p class="lede">A checkpoint is one corpus as it was before a job (the seed is its own line), kept as blobs shared by every checkpoint and branch. Restore brings that corpus back and drops the seed rows that pointed at its replaced features; branch makes a new workspace restored to it.</p>
      <div style="display:flex;gap:10px;align-items:baseline;margin-bottom:14px"><select id="ckc">${d.corpora.map(c => `<option>${esc(c.name)}</option>`).join('')}<option>seed</option></select><a href="#" class="btn quiet small" id="cknow">checkpoint now</a><span id="ckstat" class="muted" style="font-size:12.5px"></span></div>
      <table class="list" style="max-width:1000px"><tr><th>checkpoint</th><th>corpus</th><th>when</th><th>before job</th><th>note</th><th class="n">prompts</th><th class="n">features</th><th class="n">placed</th><th class="n">size</th><th></th></tr>
      ${h.checkpoints.map(c => `<tr data-id="${c.id}"><td>#${c.id}</td><td>${esc(c.corpus)}</td><td class="muted" style="font-size:12.5px">${esc(c.at)}</td><td>${c.job ? `<a href="${href('/job/' + c.job)}">#${c.job}</a> <span class="muted">${esc(c.job_kind || '')}</span>` : ''}</td><td class="muted" style="font-size:12.5px">${esc(c.note || '')}</td><td class="n">${fmt(c.counts.prompts)}</td><td class="n">${fmt(c.counts.features)}</td><td class="n">${fmt(c.counts.placed)}</td><td class="n">${kb(c.bytes)}</td><td style="white-space:nowrap"><a href="#" class="muted hdiff" style="font-size:12px">diff</a> &nbsp;<a href="#" class="muted hrestore" style="font-size:12px">restore</a> &nbsp;<a href="#" class="muted hbranch" style="font-size:12px">branch</a></td></tr><tr class="editor" hidden><td colspan="10" class="hout" style="font-size:12.5px"></td></tr>`).join('') || '<tr><td class="muted" colspan="10">none yet: a run leaves one per corpus, or take one now</td></tr>'}</table>
      <p class="muted" style="font-size:12px">objects in ${esc(h.objects)}</p>`;
  } else {
    body = `<h1>Jobs</h1><table class="list" style="max-width:860px"><tr><th>job</th><th>corpus</th><th>stage</th><th class="n">progress</th><th class="n">status</th><th class="n">spent</th></tr>
      ${d.jobs.map(j => `<tr><td><a href="${href('/job/' + j.id)}">#${j.id}</a> ${esc(j.kind)}</td><td>${esc(j.corpus || '')}</td><td class="muted">${esc(j.params.stage || j.params.kind || '')}</td><td class="n">${j.total ? `${fmt(j.done)} / ${fmt(j.total)}` : ''}</td><td class="n">${jstatus(j.status)}${j.status === 'stale' ? ` <a href="#" class="muted close" data-id="${j.id}" title="the process that ran it is gone; mark it stopped">close</a>` : ''}</td><td class="n">$${(j.spent || 0).toFixed(2)}</td></tr>`).join('') || '<tr><td class="muted">none yet</td></tr>'}</table>
      <p class="muted" style="font-size:12.5px">stale: marked running, but its log has not been written for ${20} minutes, so the process that ran it is gone.</p>`;
  }
  main.innerHTML = `<div class="sub">${tabs}</div>${body}`;
  if (sub === 'corpora') {
    const loadEstimate = async () => { main.querySelectorAll('.est').forEach(td => td.textContent = '…'); try { const e = await api(`/api/ingest/estimate?profile=${encodeURIComponent($('#prof').value)}`); for (const [n, x] of Object.entries(e.corpora)) { const td = main.querySelector(`.est[data-c="${CSS.escape(n)}"]`); if (td) { td.textContent = '$' + x.total.toFixed(2); td.title = x.steps.map(st => `${st.stage}${st.step ? ' ' + st.step : ''}: ${st.error || (st.calls + ' calls, $' + st.dollars)}`).join('\n'); } } main.querySelectorAll('.est').forEach(td => { if (td.textContent === '…') td.textContent = ''; }); $('#esttotal').textContent = '$' + e.total.toFixed(2) + (e.kind !== 'guidance' ? ` (${e.kind})` : ''); } catch (err) { $('#esttotal').textContent = err.message; } };
    loadEstimate(); $('#prof').onchange = loadEstimate;
    const runOne = async names => { $('#rstat').textContent = 'starting'; try { const r = await post('/api/ingest/run', { corpora: names, profile: $('#prof').value }); location.hash = href('/job/' + r.ids[0]); } catch (e) { $('#rstat').textContent = e.message; } };
    main.querySelectorAll('a.run').forEach(a => a.onclick = e => { e.preventDefault(); runOne([a.closest('tr').dataset.c]); });
    $('#runall').onclick = async e => { e.preventDefault(); $('#rstat').textContent = 'starting'; try { const r = await post('/api/ingest/run', { pending: true, profile: $('#prof').value }); location.hash = href('/job/' + r.ids[0]); } catch (err) { $('#rstat').textContent = err.message; } };
    main.querySelectorAll('a.edit').forEach(a => a.onclick = e => { e.preventDefault(); const ed = a.closest('tr').nextElementSibling; ed.hidden = !ed.hidden; });
    main.querySelectorAll('form.cedit').forEach(f => { const name = f.closest('tr').previousElementSibling.dataset.c, st = f.querySelector('.cstat');
      f.onsubmit = async e => { e.preventDefault(); const v = Object.fromEntries(new FormData(f)); const tags = Object.fromEntries((v.tags || '').split(/[,;]/).map(x => x.split('=')).filter(x => x[0].trim()).map(([k, ...r]) => [k.trim(), r.join('=').trim()])); try { await post(`/api/corpus/${encodeURIComponent(name)}/retag`, { domain: v.domain, tags }); if (v.name !== name) await post(`/api/corpus/${encodeURIComponent(name)}/rename`, { name: v.name }); route(); } catch (err) { st.textContent = err.message; } };
      f.querySelector('.del').onclick = async e => { e.preventDefault(); const a = e.target; if (a.textContent !== 'sure? this removes its prompts, codebook and alignment') { a.textContent = 'sure? this removes its prompts, codebook and alignment'; return; } try { await api(`/api/corpus/${encodeURIComponent(name)}`, { method: 'DELETE' }); route(); } catch (err) { st.textContent = err.message; } }; });
  } else if (sub === 'history') {
    $('#cknow').onclick = async e => { e.preventDefault(); $('#ckstat').textContent = 'writing'; try { const r = await post('/api/history/checkpoint', { corpus: $('#ckc').value }); $('#ckstat').textContent = r.repeated ? `unchanged since #${r.id}` : `#${r.id} written`; route(); } catch (err) { $('#ckstat').textContent = err.message; } };
    const outOf = a => { const tr = a.closest('tr'); return [tr.dataset.id, tr.nextElementSibling, tr.nextElementSibling.querySelector('.hout')]; };
    main.querySelectorAll('a.hdiff').forEach(a => a.onclick = async e => { e.preventDefault(); const [id, ed, out] = outOf(a); ed.hidden = false; out.textContent = 'reading'; try { const r = await api(`/api/history/${id}/diff`);
      out.innerHTML = r.gone ? 'the corpus is gone' : `<div>since then: ${Object.entries(r.delta).filter(([k, v]) => v).map(([k, v]) => `${k} ${v > 0 ? '+' : ''}${fmt(v)}`).join(', ') || 'nothing changed'}</div>${r.features_added && r.features_added.length ? `<div style="margin-top:4px">features added: ${r.features_added.slice(0, 40).map(esc).join('; ')}${r.features_added.length > 40 ? ` and ${r.features_added.length - 40} more` : ''}</div>` : ''}${r.features_gone && r.features_gone.length ? `<div style="margin-top:4px">features gone: ${r.features_gone.slice(0, 40).map(esc).join('; ')}</div>` : ''}`; } catch (err) { out.textContent = err.message; } });
    main.querySelectorAll('a.hrestore').forEach(a => a.onclick = async e => { e.preventDefault(); const [id, ed, out] = outOf(a); if (a.textContent !== 'sure? everything after it goes') { a.textContent = 'sure? everything after it goes'; return; } ed.hidden = false; out.textContent = 'restoring'; try { const r = await post(`/api/history/${id}/restore`, {}); out.textContent = `restored ${r.corpus}: ${fmt(r.restored.prompts)} prompts, ${fmt(r.restored.features)} features, ${fmt(r.restored.placed)} placed; ${r.seed_rows_pruned} seed rows pruned`; a.textContent = 'restore'; } catch (err) { out.textContent = err.message; a.textContent = 'restore'; } });
    main.querySelectorAll('a.hbranch').forEach(a => a.onclick = async e => { e.preventDefault(); const [id, ed, out] = outOf(a); ed.hidden = false; out.innerHTML = `<form class="hb" style="display:flex;gap:8px;align-items:baseline"><input type="text" name="name" placeholder="workspace name, e.g. try2" required><button class="btn quiet small" type="submit">branch</button><span class="muted hbs"></span></form>`;
      out.querySelector('form').onsubmit = async ev => { ev.preventDefault(); const st = out.querySelector('.hbs'); st.textContent = 'copying'; try { const r = await post(`/api/history/${id}/branch`, { name: out.querySelector('input').value }); st.innerHTML = `made ${esc(r.workspace)}; serve it with <span class="mono">${esc(r.serve)}</span>`; } catch (err) { st.textContent = err.message; } }; });
  } else if (sub === 'jobs') {
    main.querySelectorAll('a.close').forEach(a => a.onclick = async e => { e.preventDefault(); await post(`/api/jobs/${a.dataset.id}/close`, {}); route(); });
  } else if (sub === 'import') {
    const drop = $('#drop'), file = $('#file'), nameIn = $('#iform input[name=name]');
    drop.onclick = () => file.click();
    drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
    drop.ondragleave = () => drop.classList.remove('over');
    drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) { file.files = e.dataTransfer.files; drop.textContent = e.dataTransfer.files[0].name; if (!nameIn.value) nameIn.value = e.dataTransfer.files[0].name.replace(/\.[^.]+$/, ''); } };
    $('#iform input[name=path]').onchange = e => { const v = e.target.value.trim(); if (v && !nameIn.value) nameIn.value = v.replace(/\/+$/, '').split('/').pop().replace(/\.[^.]+$/, ''); };
    file.onchange = () => { drop.textContent = file.files[0] ? file.files[0].name : 'drop a file here, or click to choose'; if (!nameIn.value && file.files[0]) nameIn.value = file.files[0].name.replace(/\.[^.]+$/, ''); };
    $('#iform').onsubmit = async e => { e.preventDefault(); const fd = new FormData($('#iform')); const into = fd.get('into'); fd.set('name', into || fd.get('name')); fd.delete('into'); if (file.files[0]) fd.append('file', file.files[0]); if (!fd.get('name')) { $('#istatus').textContent = 'a name is needed for a new corpus'; return; } $('#istatus').textContent = 'importing';
      try { const r = await api('/api/import', { method: 'POST', body: fd }); $('#istatus').textContent = `added ${r.added}, skipped ${r.skipped}${r.unwrapped ? `, ${r.unwrapped} unwrapped from harvest residue` : ''}`; } catch (err) { $('#istatus').textContent = err.message; } };
  } else if (sub === 'profiles') {
    $('#pform').onsubmit = async e => { e.preventDefault(); const v = Object.fromEntries(new FormData($('#pform'))); const name = v.name; delete v.name; try { await post('/api/profiles', { name, params: v }); location.hash = href('/ingest/profiles', { profile: name }); route(); } catch (err) { $('#pstat').textContent = err.message; } };
    const del = $('#pdel'); if (del) del.onclick = async e => { e.preventDefault(); await api('/api/profiles/' + encodeURIComponent($('#pform input[name=name]').value), { method: 'DELETE' }); location.hash = '#/ingest/profiles'; route(); };
  }
}

/* ---------- a job: the stages one below another, progress over SSE ---------- */
const codebookTree = (groups, flags) => { const flagsBy = {}; for (const f of flags || []) (flagsBy[f.feature] = flagsBy[f.feature] || []).push(f);
  return groups.map(g => `<details class="tnode section"><summary><b>${esc(g.name)}</b> <span class="muted">${g.features.length} features, ${fmt(g.support)} prompts</span></summary><div class="kids">
    ${g.features.map(f => `<div class="tnode leaf plain"><span class="read"><a href="${href('/feature/' + f.id)}">${esc(f.name)}</a>${(flagsBy[f.id] || []).length ? ` <span class="err" style="font-size:12px">${(flagsBy[f.id] || []).length} flags</span>` : ''}<div class="def">${esc(f.definition)}</div>${(f.variants || []).map(v => `<div style="margin:3px 0 0 14px;padding-left:10px;border-left:2px solid var(--rule)"><a href="${href('/feature/' + v.id)}">${esc(v.name)}</a> <span class="muted">variant, ${fmt(v.support)}</span></div>`).join('')}</span><span class="cnt">${fmt(f.support)}</span></div>`).join('')}</div></details>`).join(''); };
async function viewJob(main, jid) {
  const [j, st] = await Promise.all([api('/api/jobs/' + jid), api(`/api/jobs/${jid}/stages`)]);
  const stage = j.params.stage || (j.kind.startsWith('decompose') ? 'decompose' : j.kind.startsWith('library') ? 'codebook' : j.kind.startsWith('align') ? 'align' : '');
  const st1 = st.decomposition, st2 = st.codebook, st3 = st.alignment;
  const badge = (name) => { if (j.status === 'running' && stage === name) return jstatus('running'); if (name === 'decompose') return st1 && st1.done >= st1.prompts && st1.prompts ? jstatus('done') : st1 && st1.done ? jstatus('partial') : jstatus('not run'); if (name === 'codebook') return st2 && st2.current ? jstatus('done') : jstatus('not run'); return st3 && st3.corpus && (st3.corpus.aligned || st3.corpus.domain_specific) ? jstatus('done') : jstatus('not run'); };
  const cur = st2 && st2.versions && st2.versions.length ? st2.versions[st2.versions.length - 1] : null;
  main.innerHTML = `<div style="font-size:12.5px;color:var(--muted);margin-bottom:8px"><a href="#/ingest/jobs">Jobs</a> › #${jid}</div>
    <div style="display:flex;justify-content:space-between;align-items:baseline;max-width:960px"><h1>#${jid} ${esc(j.corpus || 'seed')}${j.params.profile ? `, profile ${esc(j.params.profile)}` : `, ${esc(j.kind)}`}</h1><span><span id="jst">${jstatus(j.status)}</span> &nbsp;<button class="btn quiet" id="stop" ${j.status !== 'running' ? 'disabled' : ''}>stop</button></span></div>
    <div id="jbody" style="max-width:960px"></div>
    <div style="max-width:960px">
    ${j.corpus && j.corpus !== 'seed' ? `<div class="stage"><div class="h"><b>1 &nbsp; decomposition</b>${badge('decompose')}<span class="muted">${st1 ? `${fmt(st1.done)} of ${fmt(st1.prompts)} prompts${st1.coverage != null ? `, mean coverage ${pct(st1.coverage)}` : ''}${st1.failed ? `, ${st1.failed} failed` : ''}` : ''}</span></div>
      ${st1 ? `<table class="list" style="max-width:720px"><tr><th>needs a look</th><th class="n">prompts</th><th></th></tr>
        <tr><td>coverage below 90%</td><td class="n">${st1.low_coverage.length}</td><td>${st1.low_coverage.length ? `<a href="#" class="btn quiet small redo" data-ids="${esc(st1.low_coverage.map(x => x.id).join(','))}">redo</a>` : ''}</td></tr>
        <tr><td>gaps the model declined</td><td class="n">${st1.gaps.length}</td><td>${st1.gaps.length ? `<a href="#" class="btn quiet small redo" data-ids="${esc([...new Set(st1.gaps.map(x => x.id))].join(','))}">redo</a>` : ''}</td></tr>
        <tr><td>failed</td><td class="n">${st1.failures.length}</td><td>${st1.failures.length ? `<a href="#" class="btn quiet small redo" data-ids="${esc(st1.failures.map(x => x.id).join(','))}">redo</a>` : ''}</td></tr></table>
        <details style="margin-top:8px;font-size:13px"><summary class="muted" style="cursor:pointer">the prompts</summary>${['low_coverage', 'gaps', 'failures'].map(k => st1[k].length ? `<div class="block" style="margin-top:8px"><div class="t">${k.replace('_', ' ')}</div><table class="list">${st1[k].map(x => `<tr><td><a href="${href('/prompt/' + encodeURIComponent(x.id))}" class="mono" style="font-size:11.5px">${esc(x.id)}</a></td><td style="font-size:12.5px">${esc(x.head || x.text || x.error || '')}</td><td class="n">${x.coverage != null ? pct(x.coverage) : esc(x.note || '')}</td></tr>`).join('')}</table></div>` : '').join('')}</details>` : ''}</div>` : ''}
    <div class="stage"><div class="h"><b>${j.corpus && j.corpus !== 'seed' ? '2' : ''} &nbsp; codebook</b>${badge('codebook')}<span class="muted">${cur ? `${fmt(cur.features)} features, ${fmt(cur.variants)} variants, ${pct(cur.reading_coverage)} of readings placed, anchors ${cur.anchor_agreement == null ? '' : pct(cur.anchor_agreement)}, ${cur.flags} flags (${cur.standing ?? 0} standing), ${fmt(cur.leftover)} open wordings` : 'none yet'}</span></div>
      ${st2 && st2.groups.length ? `<div class="tree" style="max-height:none">${codebookTree(st2.groups, st2.flags)}</div>${st2.leftover.length ? `<details style="margin-top:8px;font-size:13px"><summary class="muted" style="cursor:pointer">open wordings, ${fmt(cur ? cur.leftover : st2.leftover.length)}</summary><table class="list">${st2.leftover.slice(0, 100).map(r => `<tr><td>${esc(r.declaration)}</td><td class="n">${r.prompts} prompts</td><td class="n">${esc(r.note || '')}</td></tr>`).join('')}</table></details>` : ''}` : ''}</div>
    <div class="stage"><div class="h"><b>${j.corpus && j.corpus !== 'seed' ? '3' : ''} &nbsp; alignment</b>${badge('align')}<span class="muted">${st3 && st3.corpus && st3.corpus.features != null ? `${fmt(st3.corpus.aligned)} of ${fmt(st3.corpus.features)} features under ${fmt(st3.seed.globals)} globals, ${fmt(st3.corpus.domain_specific)} specific to this corpus, ${fmt(st3.open.length)} open` : st3 && st3.error ? esc(st3.error) : 'no seed yet'}</span></div>
      ${st3 && st3.under && st3.under.length ? `<div class="tree" style="max-height:none">${st3.under.map(g => `<details class="tnode section"><summary><b>${esc(g.name)}</b> <span class="muted">${g.features.length}</span></summary><div class="kids">${g.features.map(f => `<div class="tnode leaf plain"><span class="read"><a href="${href('/node/' + f.id)}">${esc(f.name)}</a><div class="def">${f.members.map(m => `<a href="${href('/feature/' + m.id)}">${esc(m.name)}</a>`).join(', ')}</div></span><span class="cnt">${fmt(f.support)}</span></div>`).join('')}</div></details>`).join('')}</div>` : ''}
      ${st3 && st3.open && st3.open.length ? `<details style="margin-top:8px;font-size:13px"><summary class="muted" style="cursor:pointer">open and specific features, ${st3.open.length}</summary><table class="list">${st3.open.slice(0, 120).map(c => `<tr><td><a href="${href('/feature/' + c.id)}">${esc(c.name)}</a></td><td class="muted" style="font-size:12.5px">${esc(c.definition.slice(0, 120))}</td><td class="n">${c.support}</td><td class="n">${esc(c.note || '')}</td></tr>`).join('')}</table></details>` : ''}</div>
    <div class="stage"><div class="h"><b>log</b> <span class="mono muted" style="font-size:11.5px">${esc(j.log || '')}</span></div><pre class="log" id="jlog"></pre></div></div>`;
  const render = d => { const share = d.total ? d.done / d.total : 0; const rate = d.elapsed && d.done ? d.elapsed / d.done : null;
    $('#jst').innerHTML = jstatus(d.status);
    $('#jbody').innerHTML = `<div class="muted" style="margin-bottom:6px">${stage && d.status === 'running' ? `stage ${esc(d.params && d.params.stage || stage)}, ` : ''}${d.total ? `${fmt(d.done)} of ${fmt(d.total)}${d.status === 'running' && rate && d.total > d.done ? `, about ${fmtSec(rate * (d.total - d.done))} left` : ''}, ` : ''}${d.elapsed == null ? '' : fmtSec(d.elapsed) + ', '}$${(d.spent || 0).toFixed(2)} spent, ${fmt(d.calls)} calls${d.error ? ` <span class="err">${esc(d.error)}</span>` : ''}</div>
      <div class="bar"><i style="width:${(100 * share).toFixed(1)}%"></i></div>
      <div class="recent" style="margin-top:8px">${(d.recent || []).slice(-3).map(r => `<div>${Object.entries(r).filter(([k, v]) => k !== 'error' && v != null).map(([k, v]) => `${esc(k)} ${esc(String(v))}`).join(', ')}${r.error ? ` <span class="err">${esc(r.error)}</span>` : ''}</div>`).join('')}</div>`; };
  render(j);
  const loadLog = async () => { const l = await api(`/api/jobs/${jid}/log?tail=60`); $('#jlog').textContent = l.lines.join('\n'); };
  loadLog();
  $('#stop').onclick = () => post(`/api/jobs/${jid}/stop`, {});
  main.querySelectorAll('a.redo').forEach(a => a.onclick = async e => { e.preventDefault(); const r = await post('/api/jobs', { corpus: j.corpus, ids: a.dataset.ids.split(','), redo: true, model: j.params.decompose_model || j.model }); location.hash = href('/job/' + r.id); });
  if (j.status === 'running') { ES = new EventSource(`/api/jobs/${jid}/events`); let n = 0; ES.onmessage = e => { const d = JSON.parse(e.data); d.params = j.params; render(d); if (++n % 10 === 0) loadLog(); }; ES.addEventListener('end', () => { ES.close(); ES = null; route(); }); }
}

/* ---------- settings: endpoints and keys by reference; a key is written blind and never read back ---------- */
async function viewSettings(main) {
  const s = await api('/api/settings');
  const src = k => k.set ? `<span class="ok">set</span> <span class="muted">from the ${k.source === 'file' ? 'key file' : 'shell environment'}${k.secret ? `, ${k.length} characters` : ''}</span>` : '<span class="muted">not set</span>';
  const row = k => `<tr><td class="mono">${esc(k.name)}</td><td>${src(k)}</td><td><form class="kf" data-name="${esc(k.name)}" style="display:flex;gap:6px"><input type="${k.secret ? 'password' : 'text'}" name="value" placeholder="${k.secret ? 'paste a key to save it' : 'value'}" value="${k.secret ? '' : esc(k.value || '')}" autocomplete="off" style="flex:1"><button class="btn quiet" type="submit">save</button>${k.set && k.source === 'file' ? '<button class="btn quiet" type="button" data-clear>remove</button>' : ''}</form></td></tr>`;
  main.innerHTML = `<h1>Settings</h1>
    <p class="lede">Keys are references: a job names the endpoint, the endpoint names the variable, the variable's value lives in your shell or in the key file below. The site writes the file and never shows a value.</p>
    <div class="kv" style="margin-bottom:22px"><span class="k">key file</span><span class="mono">${esc(s.file)} <span class="muted" style="font-family:var(--sans)">${s.exists ? `· mode ${s.mode}${s.loose ? ' <span class="err">(should be 600)</span>' : ''}` : '· not written yet'}</span></span>
      <span class="k">shell alternative</span><span class="muted">export the variables before <span class="mono">fx serve</span>; the shell's value wins over the file's</span></div>
    <div class="block"><div class="t">keys</div><table class="list"><tr><th>variable</th><th>status</th><th>set it</th></tr>${s.keys.map(row).join('')}</table></div>
    <div class="block"><div class="t">endpoints · a probe lists the endpoint's models with its key, which costs nothing</div><table class="list"><tr><th>endpoint</th><th>base url</th><th>key</th><th></th><th>probe</th></tr>
      ${s.endpoints.map(e => `<tr><td>${esc(e.name)}</td><td class="mono" style="font-size:12px">${esc(e.base_url)}</td><td class="mono" style="font-size:12px">${esc(e.key_env || 'none needed')}</td><td><button class="btn quiet" data-probe="${esc(e.name)}">probe</button></td><td id="probe-${esc(e.name)}" class="muted" style="font-size:12.5px"></td></tr>`).join('')}
      <tr><td>any other</td><td colspan="2"><input type="text" id="purl" placeholder="an OpenAI-compatible base url, e.g. http://localhost:8002/v1; key from FX_API_KEY" style="width:100%"></td><td><button class="btn quiet" data-probe="" id="pcustom">probe</button></td><td id="probe-custom" class="muted" style="font-size:12.5px"></td></tr></table></div>
    <div class="block"><div class="t">settings the same file may hold</div><table class="list"><tr><th>variable</th><th>status</th><th>value</th></tr>${s.settings.map(row).join('')}</table>
      <p class="muted" style="font-size:12.5px">FX_LOCAL_URL: the local vLLM server (default http://localhost:8000/v1). FX_PROVIDER: the OpenRouter upstream to pin (default Wafer; empty allows any but the known loopers). FX_MODEL: the default batch model, read at start, so a change applies after a restart.</p></div>
    <div class="block"><div class="t">models the registry knows · default ${esc(s.default_model)}</div><table class="list"><tr><th>model</th><th>endpoint</th><th class="n">$ / M in</th><th class="n">$ / M out</th></tr>${s.models.map(m => `<tr><td class="mono" style="font-size:12px">${esc(m.model)}</td><td>${esc(m.endpoint)}</td><td class="n">${m.price_in}</td><td class="n">${m.price_out}</td></tr>`).join('')}</table></div>`;
  for (const f of main.querySelectorAll('form.kf')) {
    f.onsubmit = async e => { e.preventDefault(); const v = f.value.value; if (!v.trim()) return; try { await post('/api/settings/key', { name: f.dataset.name, value: v }); f.value.value = ''; await viewSettings(main); } catch (err) { f.querySelector('button').textContent = err.message.slice(0, 80); } };
    const c = f.querySelector('[data-clear]'); if (c) c.onclick = async () => { await post('/api/settings/key', { name: f.dataset.name, value: '' }); await viewSettings(main); };
  }
  for (const b of main.querySelectorAll('[data-probe]')) b.onclick = async () => { const name = b.dataset.probe, out = $('#probe-' + (name || 'custom')); out.textContent = 'probing'; try { const r = await post('/api/settings/probe', name ? { endpoint: name } : { base_url: $('#purl').value }); out.innerHTML = r.ok ? `<span class="ok">ok</span> · ${r.models} models · ${r.ms} ms${r.sample.length ? ' · e.g. ' + esc(r.sample.slice(0, 4).join(', ')) : ''}` : `<span class="err">failed</span> · ${r.status || ''} ${esc(r.error || '')} · ${r.ms} ms`; } catch (e) { out.textContent = e.message; } };
}

window.addEventListener('hashchange', route);
route();                                   // last: every const above is initialised before the first view runs
