/* feature_extract GUI: corpora (import, run), prompts, one prompt, queues. Everything reads the store through /api. */
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

function parseHash() { const h = location.hash.slice(1) || '/corpora'; const [path, qs] = h.split('?'); return { parts: path.split('/').filter(Boolean), q: Object.fromEntries(new URLSearchParams(qs || '')) }; }
async function route() {
  const { parts, q } = parseHash(); const view = parts[0] || 'corpora';
  $('#nav').innerHTML = [['corpora', 'Corpora'], ['prompts', 'Prompts'], ['queues', 'Queues'], ['library', 'Library'], ['seed', 'Seed'], ['cube', 'Cube'], ['settings', 'Settings']].map(([k, l]) => `<a href="${href('/' + k, k === 'cube' || k === 'settings' ? {} : { corpus: q.corpus })}" class="${view === k || (k === 'prompts' && view === 'prompt') || (k === 'library' && view === 'feature') || (k === 'cube' && view === 'node') ? 'on' : ''}" ${k === 'cube' ? 'style="margin-left:18px"' : ''}>${l}</a>`).join('');
  if (ES) { ES.close(); ES = null; }
  const main = $('#main'); main.innerHTML = '<div class="loading">loading</div>'; window.scrollTo(0, 0);
  try {
    api('/api/spend').then(s => { $('#spend').textContent = `spent $${s.total.toFixed(2)}`; }).catch(() => {});
    if (view === 'corpora') await viewCorpora(main, q);
    else if (view === 'prompts') await viewPrompts(main, q);
    else if (view === 'prompt') await viewPrompt(main, decodeURIComponent(parts.slice(1).join('/')));
    else if (view === 'queues') await viewQueues(main, q);
    else if (view === 'job') await viewJob(main, +parts[1]);
    else if (view === 'library') await viewLibrary(main, q);
    else if (view === 'seed') await viewSeed(main, q);
    else if (view === 'feature') await viewFeature(main, +parts[1]);
    else if (view === 'cube') await viewCube(main, q);
    else if (view === 'node') await viewNode(main, +parts[1], q);
    else if (view === 'settings') await viewSettings(main, q);
    else main.innerHTML = '<p>No such page.</p>';
  } catch (e) { main.innerHTML = `<p class="err">${esc(e.message)}</p>`; console.error(e); }
}

/* ---------- corpora: import and run ---------- */
async function viewCorpora(main, q) {
  const [cs, jobs, ms] = await Promise.all([api('/api/corpora'), api('/api/jobs'), api('/api/models')]);
  const sel = q.corpus || (cs[0] && cs[0].name) || '';
  main.innerHTML = `<div class="split"><div>
    <h1>Corpora</h1>
    <p class="lede">Drop a FACET prompts file, a folder zipped, or a text file; or name a path on this machine; or paste one prompt. Then decompose.</p>
    <div class="drop" id="drop">drop a file here, or click to choose<input type="file" id="file" hidden></div>
    <form id="iform" class="form" style="margin-top:14px">
      <label>or a path here</label><input type="text" name="path" placeholder="e.g. data/corpora/facet/text2sql.jsonl, a folder, or a zip" list="paths"><datalist id="paths"><option value="data/corpora/facet/text2sql.jsonl"><option value="data/corpora/facet/math.jsonl"><option value="data/corpora/facet/table-qa.jsonl"><option value="data/corpora/facet/science-quantitative.jsonl"><option value="data/corpora/facet/text2cypher.jsonl"><option value="data/corpora/facet/code-generation.jsonl"><option value="data/corpora/plain"></datalist>
      <label>corpus name</label><input type="text" name="name" placeholder="e.g. text2sql" required>
      <label>domain filter</label><input type="text" name="domain" placeholder="for a FACET jsonl: keep only this domain">
      <label>or paste a prompt</label><textarea name="text"></textarea>
      <span></span><span><button class="btn" type="submit">import</button> <span id="istatus" class="muted"></span></span></form>
    <div class="block" style="margin-top:28px"><div class="t">corpora in the store</div>
      <table class="list"><tr><th>corpus</th><th class="n">prompts</th><th class="n">characters</th><th class="n">decomposed</th><th class="n">mean coverage</th></tr>
      ${cs.map(c => `<tr><td><a href="${href('/prompts', { corpus: c.name })}">${esc(c.name)}</a> <span class="muted" style="font-size:12px">${esc(c.source || '')}</span></td><td class="n">${fmt(c.n_prompts)}</td><td class="n">${fmt(c.chars)}</td><td class="n">${fmt(c.decomposed)}</td><td class="n">${pct(c.coverage)}</td></tr>`).join('')}</table></div>
  </div><div>
    <h1 style="font-size:22px">Decompose</h1>
    <form id="rform" class="form" style="grid-template-columns:90px minmax(0,1fr)">
      <label>corpus</label><select name="corpus">${cs.map(c => `<option value="${esc(c.name)}" ${c.name === sel ? 'selected' : ''}>${esc(c.name)}</option>`).join('')}</select>
      <label>model</label><span><input type="text" name="model" value="${esc(ms.default)}" list="models" style="width:100%"><datalist id="models">${ms.models.map(m => `<option value="${esc(m.model)}">${esc(m.endpoint)} · $${m.price_in}/M in, $${m.price_out}/M out</option>`).join('')}</datalist><span class="muted" style="font-size:12px">any model name works: gpt-* goes to OpenAI, vendor/model to OpenRouter, anything else to the local server at ${esc(ms.local_url)}; FX_MODEL sets the default, FX_MODELS adds models with prices</span></span>
      <label>workers</label><input type="number" name="workers" value="512" min="1" max="4096"> <span class="muted" style="font-size:12px">calls in flight</span>
      <label>first N only</label><input type="number" name="limit" value="30" min="0" placeholder="0 for all">
      <label>budget $</label><input type="number" name="budget" value="" placeholder="none" step="0.5">
      <label>options</label><span><label><input type="checkbox" name="redo"> redo finished prompts</label></span>
      <span></span><span><button class="btn quiet" type="button" id="prevbtn">preview</button> <button class="btn" type="button" id="runbtn">run first N</button> <button class="btn quiet" type="button" id="allbtn">run all</button></span></form>
    <div id="preview" class="block" style="margin-top:16px"></div>
    <div class="block"><div class="t">jobs</div>${jobs.length ? `<table class="list">${jobs.slice(0, 8).map(j => `<tr><td><a href="${href('/job/' + j.id)}">#${j.id}</a> ${esc(j.corpus || 'all')} · ${esc(j.model)}</td><td class="n">${j.done}/${j.total}</td><td class="n">${esc(j.status)}</td></tr>`).join('')}</table>` : '<span class="muted">none yet</span>'}</div>
  </div></div>`;
  const drop = $('#drop'), file = $('#file');
  drop.onclick = () => file.click();
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) file.files = e.dataTransfer.files; drop.textContent = e.dataTransfer.files[0] ? e.dataTransfer.files[0].name : drop.textContent; if (!$('#iform input[name=name]').value) $('#iform input[name=name]').value = (e.dataTransfer.files[0]?.name || '').replace(/\.[^.]+$/, ''); };
  $('#iform input[name=path]').onchange = e => { const v = e.target.value.trim(); if (v && !$('#iform input[name=name]').value) $('#iform input[name=name]').value = v.replace(/\/+$/, '').split('/').pop().replace(/\.[^.]+$/, ''); };
  file.onchange = () => { drop.textContent = file.files[0] ? file.files[0].name : 'drop a file here, or click to choose'; if (!$('#iform input[name=name]').value && file.files[0]) $('#iform input[name=name]').value = file.files[0].name.replace(/\.[^.]+$/, ''); };
  $('#iform').onsubmit = async e => { e.preventDefault(); const fd = new FormData($('#iform')); if (file.files[0]) fd.append('file', file.files[0]); $('#istatus').textContent = 'importing';
    try { const r = await api('/api/import', { method: 'POST', body: fd }); $('#istatus').textContent = `added ${r.added}, skipped ${r.skipped}, ${r.duplicates_elsewhere} also in another corpus${r.unwrapped ? `, ${r.unwrapped} unwrapped from harvest residue` : ''}`; location.hash = href('/corpora', { corpus: r.corpus }); route(); } catch (err) { $('#istatus').textContent = err.message; } };
  const rf = $('#rform');
  const params = () => { const d = Object.fromEntries(new FormData(rf)); return { corpus: d.corpus, model: d.model, workers: +d.workers, limit: +d.limit || 0, budget: d.budget || null, redo: !!d.redo }; };
  const showPreview = async () => { const p = params(); $('#preview').innerHTML = '<span class="muted">estimating</span>';
    const r = await api(`/api/preview?corpus=${encodeURIComponent(p.corpus)}&model=${encodeURIComponent(p.model)}&workers=${p.workers}&redo=${p.redo}&limit=${p.limit}`);
    if (!r.prompts) { $('#preview').innerHTML = '<span class="muted">nothing to do: every prompt is decomposed</span>'; return; }
    $('#preview').innerHTML = `<div class="prev"><span><b>${fmt(r.prompts)}</b>prompts</span><span><b>${fmt(r.calls)}</b>calls</span><span><b>${fmt(Math.round(r.tokens_in / 1000))}k</b>tokens in</span><span><b>$${r.dollars.toFixed(2)}</b>${esc(r.endpoint)}</span><span><b>${r.seconds == null ? '?' : fmtSec(r.seconds)}</b>at ${r.workers} workers</span></div>
      <div class="muted" style="font-size:12px;margin-top:6px">${esc(r.basis)}${r.note ? ' · ' + esc(r.note) : ''}</div>`; };
  $('#prevbtn').onclick = showPreview;
  rf.querySelectorAll('select,input').forEach(el => el.onchange = showPreview);
  const start = async limit => { const p = params(); const r = await post('/api/jobs', { ...p, limit }); location.hash = href('/job/' + r.id); };
  $('#runbtn').onclick = () => start(params().limit);
  $('#allbtn').onclick = () => start(0);
  showPreview();
}
const fmtSec = s => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;

/* ---------- a job: progress over SSE ---------- */
async function viewJob(main, jid) {
  const j = await api('/api/jobs/' + jid);
  main.innerHTML = `<h1>Job #${jid} · ${esc(j.corpus || 'all corpora')}</h1><p class="lede">${esc(j.model)} · ${j.params.workers} workers${j.params.limit ? ` · pilot of ${j.params.limit}` : ''}</p>
    <div id="jbody"></div><details style="margin-top:12px;font-size:12.5px"><summary class="muted">log · <span class="mono">${esc(j.log || '')}</span></summary><pre class="mono" id="jlog" style="font-size:11.5px;white-space:pre-wrap;max-height:40vh;overflow:auto"></pre></details><p style="margin-top:14px"><button class="btn quiet" id="stop">stop</button> ${j.kind.startsWith('align') ? `<a href="${href('/seed', { kind: j.params.kind })}" style="margin-left:14px">seed →</a>` : j.kind.startsWith('library') ? `<a href="${href('/library', { corpus: j.corpus, kind: j.params.kind })}" style="margin-left:14px">library →</a>` : `<a href="${href('/prompts', { corpus: j.corpus, status: 'done' })}" style="margin-left:14px">decomposed prompts →</a>`} <a href="${href('/queues', { corpus: j.corpus })}" style="margin-left:14px">queues →</a></p>`;
  const render = d => { const share = d.total ? d.done / d.total : 0; const rate = d.elapsed && d.done ? d.elapsed / d.done : null;
    $('#jbody').innerHTML = `<div class="bar" style="max-width:720px"><i style="width:${(100 * share).toFixed(1)}%"></i></div>
      <div class="prev" style="margin-top:12px"><span><b>${fmt(d.done)}${d.total ? ' / ' + fmt(d.total) : ''}</b>${j.kind.startsWith('library') || j.kind.startsWith('align') ? 'batches' : 'prompts'}</span><span><b>${fmt(d.calls)}</b>calls</span><span><b>$${(d.spent || 0).toFixed(2)}</b>spent</span><span><b>${d.elapsed == null ? '' : fmtSec(d.elapsed)}</b>elapsed</span><span><b>${d.status !== 'running' ? d.status : (rate && d.total > d.done ? fmtSec(rate * (d.total - d.done)) : 'running')}</b>${d.status === 'running' && rate && d.total > d.done ? 'remaining, projected' : 'status'}</span></div>
      <div class="recent" style="margin-top:12px">${(d.recent || []).map(r => `<div>${r.id != null && !('batch' in r || 'cluster' in r || 'what' in r) ? `<a href="${href('/prompt/' + encodeURIComponent(r.id))}">${esc(r.id)}</a> · coverage ${pct(r.coverage)} · ${r.n_atoms} atoms · ${r.calls} calls` : Object.entries(r).filter(([k, v]) => k !== 'error' && v != null).map(([k, v]) => `${esc(k)} ${esc(String(v))}`).join(' · ')}${r.error ? ` · <span class="err">${esc(r.error)}</span>` : ''}</div>`).join('')}</div>${d.error ? `<p class="err">${esc(d.error)}</p>` : ''}`; };
  render(j);
  document.querySelector('details').addEventListener('toggle', async e => { if (e.target.open) { const l = await api(`/api/jobs/${jid}/log`); $('#jlog').textContent = l.lines.join('\n'); } });
  $('#stop').onclick = () => post(`/api/jobs/${jid}/stop`, {});
  if (j.status === 'running') { ES = new EventSource(`/api/jobs/${jid}/events`); ES.onmessage = e => render(JSON.parse(e.data)); ES.addEventListener('end', () => { ES.close(); ES = null; api('/api/jobs/' + jid).then(render); }); }
}

/* ---------- prompts ---------- */
async function viewPrompts(main, q) {
  const r = await api(`/api/prompts?corpus=${encodeURIComponent(q.corpus || '')}&status=${q.status || ''}`);
  main.innerHTML = `<h1>Prompts${q.corpus ? ' · ' + esc(q.corpus) : ''}</h1>
    <p class="lede">${fmt(r.total)} in ${q.corpus ? 'the corpus' : 'the store'}. <a href="${href('/prompts', { corpus: q.corpus, status: 'done' })}">decomposed</a> · <a href="${href('/prompts', { corpus: q.corpus, status: 'todo' })}">not yet</a> · <a href="${href('/prompts', { corpus: q.corpus })}">all</a></p>
    <table class="list"><tr><th>prompt</th><th>starts with</th><th class="n">chars</th><th class="n">coverage</th><th class="n">material</th><th class="n">atoms</th><th class="n">calls</th></tr>
    ${r.prompts.map(p => `<tr><td><a href="${href('/prompt/' + encodeURIComponent(p.id))}" class="mono" style="font-size:12px">${esc(p.system || p.id)}</a><div class="muted" style="font-size:11.5px">${esc(p.domain || '')} ${esc(p.task || '')}</div></td><td class="serif" style="max-width:520px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">${esc(p.head)}</td><td class="n">${fmt(p.chars)}</td><td class="n">${p.status === 'done' ? pct(p.coverage) : p.status ? esc(p.status) : ''}</td><td class="n">${p.status === 'done' ? pct(p.material_share) : ''}</td><td class="n">${p.status === 'done' ? p.n_atoms : ''}</td><td class="n">${p.calls ?? ''}</td></tr>`).join('')}</table>
    ${r.total > r.prompts.length ? `<p class="muted">${fmt(r.total - r.prompts.length)} more not listed.</p>` : ''}`;
}

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

/* ---------- queues ---------- */
async function viewQueues(main, q) {
  const r = await api(`/api/queues?corpus=${encodeURIComponent(q.corpus || '')}`);
  const row = (x, extra) => `<tr><td><a href="${href('/prompt/' + encodeURIComponent(x.id))}" class="mono" style="font-size:12px">${esc(x.id)}</a></td><td class="serif">${esc(x.text || x.head || '')}</td>${extra || ''}</tr>`;
  main.innerHTML = `<h1>Queues${q.corpus ? ' · ' + esc(q.corpus) : ''}</h1><p class="lede">What to read: prompts the model covered poorly, stretches it declined twice, leaves it could not refine, and failures.</p>
    <div class="block"><div class="t">low coverage, under 90%</div>${r.low_coverage.length ? `<table class="list">${r.low_coverage.map(x => row(x, `<td class="n">${pct(x.coverage)}</td><td class="n">${x.n_atoms} atoms</td>`)).join('')}</table>` : '<span class="muted">none</span>'}</div>
    <div class="block"><div class="t">gaps the model declined</div>${r.gaps.length ? `<table class="list">${r.gaps.map(x => row(x, `<td class="n">${esc(x.note || '')}</td>`)).join('')}</table>` : '<span class="muted">none</span>'}</div>
    <div class="block"><div class="t">unrefined leaves</div>${r.unrefined.length ? `<table class="list">${r.unrefined.map(x => row(x)).join('')}</table>` : '<span class="muted">none</span>'}</div>
    <div class="block"><div class="t">unwrapped at import (read that the prompt is the whole prompt)</div>${r.unwrapped.length ? `<table class="list">${r.unwrapped.map(x => row(x, `<td class="n">${esc(x.note || '')}</td>`)).join('')}</table>` : '<span class="muted">none</span>'}</div>
    <div class="block"><div class="t">failed</div>${r.failed.length ? `<table class="list">${r.failed.map(x => row({ id: x.id, text: x.error })).join('')}</table>` : '<span class="muted">none</span>'}</div>`;
}

window.addEventListener('hashchange', route);
route();

/* ---------- library: the codebook of a corpus, one per kind ---------- */
async function viewLibrary(main, q) {
  const cs = await api('/api/corpora');
  const corpus = q.corpus || (cs[0] && cs[0].name) || '', kind = q.kind || 'guidance';
  if (!corpus) { main.innerHTML = '<p class="muted">import a corpus first</p>'; return; }
  const [lib, jobs] = await Promise.all([api(`/api/library?corpus=${encodeURIComponent(corpus)}&kind=${kind}${q.version ? '&version=' + q.version : ''}`), api('/api/jobs')]);
  const cb = lib.codebook, vs = lib.versions;
  const sel = `<select id="lcorpus">${cs.map(c => `<option ${c.name === corpus ? 'selected' : ''}>${esc(c.name)}</option>`).join('')}</select>
    <span style="margin-left:14px">${['guidance', 'material'].map(k => `<a href="${href('/library', { corpus, kind: k })}" class="${k === kind ? 'on' : ''}" style="margin-right:10px;${k === kind ? 'font-weight:600;color:var(--ink)' : ''}">${k}</a>`).join('')}</span>`;
  const vtable = vs.length ? `<table class="list"><tr><th>version</th><th>round</th><th>model</th><th class="n">groups</th><th class="n">features</th><th class="n">variants</th><th class="n">rounds</th><th class="n">assigned</th><th class="n">open</th><th class="n">of which specific</th><th class="n">reading coverage</th><th class="n">anchors</th><th class="n">flags</th><th class="n">standing</th></tr>
    ${vs.map(v => `<tr><td><a href="${href('/library', { corpus, kind, version: v.version })}">v${v.version}</a>${cb && v.id === cb.id ? ' <span class="muted">shown</span>' : ''}</td><td class="n">${v.round}</td><td>${esc(v.model)}</td><td class="n">${v.groups}</td><td class="n">${v.features}</td><td class="n">${v.variants}</td><td class="n">${v.rounds}</td><td class="n">${fmt(v.assigned)}${v.unassigned ? ` <span class="muted">(+${fmt(v.unassigned)} to do)</span>` : ''}</td><td class="n">${fmt(v.leftover)}</td><td class="n">${fmt(v.specific)}</td><td class="n">${pct(v.reading_coverage)}</td><td class="n">${v.anchor_agreement == null ? '' : pct(v.anchor_agreement)}</td><td class="n">${v.flags}</td><td class="n">${v.standing ?? ''}</td></tr>`).join('')}</table>` : '<span class="muted">no codebook yet: run cold start</span>';
  const flagsBy = {}; for (const f of lib.flags) (flagsBy[f.feature] = flagsBy[f.feature] || []).push(f);
  const tree = lib.groups.map(g => `<details class="tnode section" open><summary><b>${esc(g.name)}</b> <span class="tag">${esc(g.aspect)} · ${fmt(g.support)} prompts · ${g.features.length} features</span> <span class="muted" style="font-size:12.5px">${esc(g.definition)}</span></summary><div class="kids">
      ${g.features.map(f => `<div class="tnode leaf ${f.polarity === 'forbid' ? 'forbid' : ''}"><span class="path">${fmt(f.support)}</span><span class="read"><a href="${href('/feature/' + f.id)}">${pol(f.polarity)} ${esc(f.name)}</a><span class="tag">${f.realizations} wordings${f.round ? ' · round ' + f.round : ''}${(flagsBy[f.id] || []).length ? ` · <span class="err">${(flagsBy[f.id] || []).length} flags</span>` : ''}</span><br><span class="muted" style="font-size:12.5px">${esc(f.definition)}</span>${(f.variants || []).map(v => `<div style="margin:4px 0 0 14px;padding-left:10px;border-left:2px solid var(--rule)"><a href="${href('/feature/' + v.id)}">${esc(v.name)}</a> <span class="tag">variant · ${fmt(v.support)} prompts · round ${v.round}</span><br><span class="muted" style="font-size:12px">${esc(v.definition)}</span></div>`).join('')}</span></div>`).join('')}</div></details>`).join('');
  main.innerHTML = `<div class="split"><div>
    <h1>Library</h1><p class="lede">${sel}</p>
    <div class="block"><div class="t">versions · ${fmt(lib.realizations)} distinct wordings over ${fmt(lib.readings)} readings</div>${vtable}</div>
    <div class="block"><div class="t">codebook${cb ? ' v' + cb.version : ''}${cb && cb.notes ? ` · <span class="muted">${esc(cb.notes.slice(0, 300))}</span>` : ''}</div><div class="tree" style="max-height:none">${tree || '<span class="muted">empty</span>'}</div></div>
    ${cb ? `<div class="block"><div class="t">open wordings (on no node yet; "specific" ones have no echo elsewhere in the corpus)</div>${lib.leftover.length ? `<table class="list">${lib.leftover.slice(0, 80).map(r => `<tr><td class="serif">${r.polarity === 'forbid' ? pol('forbid') + ' ' : ''}${esc(r.declaration)}</td><td class="n">${r.prompts} prompts</td><td class="n">${esc(r.note || r.confidence || '')}</td></tr>`).join('')}</table>${lib.leftover.length > 80 ? `<div class="muted">and ${lib.leftover.length - 80} more</div>` : ''}` : '<span class="muted">none</span>'}</div>` : ''}
  </div><div>
    <h1 style="font-size:22px">Run</h1>
    <form id="lform" class="form" style="grid-template-columns:90px minmax(0,1fr)">
      <label>step</label><select name="step"><option value="round" selected>round loop: cold start if needed, assign, judge, then cluster → name → assign until no candidates</option><option value="coldstart">cold start</option><option value="assign">assign (new and open wordings)</option><option value="judge">judge (read-only)</option><option value="reopen">reopen the flagged members (no calls)</option><option value="cluster">cluster the open wordings (no calls)</option><option value="name">name the candidate clusters</option></select>
      <label>batch model</label><input type="text" name="model" placeholder="assign and judge; default: the decomposition model" style="width:100%">
      <label>codebook model</label><input type="text" name="codebook_model" placeholder="cold start and naming; default gpt-5.6-sol" style="width:100%">
      <label>rounds</label><input type="number" name="rounds" value="5" min="1" max="20">
      <label>base url</label><input type="text" name="base_url" placeholder="optional: another OpenAI-compatible server, e.g. http://localhost:8002/v1" style="width:100%">
      <label>workers</label><input type="number" name="workers" value="128" min="1" max="512">
      <label>budget $</label><input type="number" name="budget" value="" placeholder="none" step="1">
      <label>effort</label><select name="effort"><option value="low" selected>low reasoning effort for assign and judge</option><option value="default">provider default</option></select>
      <span></span><span><button class="btn quiet" type="button" id="lprev">preview</button> <button class="btn" type="button" id="lrun">run</button> <span id="lstatus" class="muted"></span></span></form>
    <div id="lpreview" class="block" style="margin-top:16px"></div>
    <p class="muted" style="font-size:12.5px">The tree only grows. A round clusters the open wordings by retrieval, names each candidate cluster (a variant under a feature, a new feature, or a rejection), and assigns the open wordings against the tree again; a flag raised for the first time sends its member back to open, barred from the node it left; a flag raised again after that is standing (the member stays, the flag is the report). It ends when every flag is standing and no candidate is left. A wording no other prompt echoes is marked specific and waits. Each step is one line in the job log; a re-run resumes.</p>
    <div class="block"><div class="t">library jobs</div>${jobs.filter(j => j.kind.startsWith('library')).length ? `<table class="list">${jobs.filter(j => j.kind.startsWith('library')).slice(0, 10).map(j => `<tr><td><a href="${href('/job/' + j.id)}">#${j.id}</a> ${esc(j.kind.slice(8))} ${esc(j.params.kind)} · ${esc(j.corpus)}</td><td class="n">${esc(j.status)}</td></tr>`).join('')}</table>` : '<span class="muted">none yet</span>'}</div>
  </div></div>`;
  $('#lcorpus').onchange = e => { location.hash = href('/library', { corpus: e.target.value, kind }); };
  const params = () => { const d = Object.fromEntries(new FormData($('#lform'))); return { corpus, kind, step: d.step, model: d.model, codebook_model: d.codebook_model, rounds: +d.rounds, base_url: d.base_url, workers: +d.workers, effort: d.effort, budget: d.budget || null }; };
  $('#lprev').onclick = async () => { const p = params(); const r = await api(`/api/library/preview?corpus=${encodeURIComponent(corpus)}&kind=${kind}&step=${p.step}&model=${encodeURIComponent(p.model)}`);
    $('#lpreview').innerHTML = `<div class="prev"><span><b>${fmt(r.calls)}</b>calls</span><span><b>${fmt(r.tokens_in)}</b>tokens in</span><span><b>${fmt(r.tokens_out)}</b>tokens out</span><span><b>$${r.dollars.toFixed(2)}</b>${esc(r.model)} · ${esc(r.endpoint)}</span></div>`; };
  $('#lrun').onclick = async () => { const p = params(); $('#lstatus').textContent = 'starting'; try { const r = await post('/api/library/jobs', p); location.hash = href('/job/' + r.id); } catch (e) { $('#lstatus').textContent = e.message; } };
}

async function viewFeature(main, fid) {
  const r = await api('/api/feature/' + fid); const f = r.feature, cb = r.codebook;
  const byR = {}; for (const m of r.members) byR[m.id] = m;
  const quotes = {}; for (const x of r.readings) (quotes[x.realization] = quotes[x.realization] || []).push(x);
  main.innerHTML = `<h1>${pol(f.polarity)} ${esc(f.name)}</h1>
    <p class="lede">${esc(f.definition)}</p>
    <div class="facts" style="margin-bottom:16px"><span class="k">library</span><span><a href="${href('/library', { corpus: cb.corpus_name, kind: cb.kind, version: cb.version })}">${esc(cb.corpus_name)} · ${esc(cb.kind)}</a>${f.level === 'variant' ? ' · <span class="muted">a variant</span>' : ''}${f.round ? ` · <span class="muted">added in round ${f.round}</span>` : ''}</span>
      <span class="k">${f.level === 'variant' ? 'feature' : 'group'}</span><span>${r.group ? (f.level === 'variant' ? `<a href="${href('/feature/' + r.group.id)}">${esc(r.group.name)}</a>` : esc(r.group.name)) + ' <span class="muted">· ' + esc(r.group.aspect || r.group.polarity || '') + ' · ' + esc(r.group.definition) + '</span>' : ''}</span>
      <span class="k">support</span><span>${fmt(r.members.reduce((s, m) => s + m.prompts, 0))} prompts · ${fmt(r.members.reduce((s, m) => s + m.n, 0))} readings · ${r.members.length} distinct wordings</span>
      <span class="k">anchors</span><span>${f.examples.map(id => byR[id] ? `<span style="color:var(--req)">✓ ${esc(byR[id].declaration)}</span>` : `<span class="err">✗ R${id} not on this feature</span>`).join(' · ') || '<span class="muted">none</span>'}</span>
      ${r.aligned ? `<span class="k">seed</span><span>${r.aligned.global ? 'aligned to <b>' + esc(r.aligned.global_name) + '</b>' : (r.aligned.note === 'domain-specific' ? '<span class="muted">domain-specific: no other corpus says this yet</span>' : '<span class="muted">open: ' + esc(r.aligned.note || 'not aligned yet') + '</span>')}</span>` : ''}
      ${r.lineage.length ? `<span class="k">lineage</span><span>${r.lineage.map(l => `<a href="${href('/feature/' + l.id)}">${esc(l.name)}</a>`).join(' ← ')}</span>` : ''}
      ${r.flags.length ? `<span class="k">flags</span><span>${r.flags.map(x => `<div><span class="err">${esc(x.verdict)}${x.standing ? ' · standing' : ''}</span> ${x.verdict === 'indistinct' ? 'with <a href="' + href('/feature/' + (x.feature === f.id ? x.other : x.feature)) + '">' + esc(x.feature === f.id ? x.other_name : x.feature_name) + '</a>' : esc(x.declaration || '')} <span class="muted">${esc(x.verdict === 'split' ? JSON.parse(x.note || '{}').why || '' : x.note || '')}</span></div>`).join('')}</span>` : ''}</div>
    <div class="block"><div class="t">wordings on this ${f.level}, by support · open one for the prompts it comes from</div>
    <div class="members"><div style="display:grid;grid-template-columns:minmax(0,1fr) 70px 70px 90px;gap:10px;font-size:12px;color:var(--muted);padding:4px 0"><span>wording</span><span class="n">prompts</span><span class="n">readings</span><span class="n">confidence</span></div>
      ${r.members.map(m => `<details><summary><span class="serif">${esc(m.declaration)}${m.conditions.filter(c => c !== 'always').length ? ` <span class="muted" style="font-size:12px">when: ${esc(m.conditions.filter(c => c !== 'always').slice(0, 2).join('; '))}</span>` : ''}</span><span class="n">${m.prompts}</span><span class="n">${m.n}</span><span class="n">${esc(m.confidence)}</span></summary>
        <div class="quotes">${(quotes[m.id] || []).map(x => `<div class="quote"><a href="${href('/prompt/' + encodeURIComponent(x.prompt))}" class="mono" style="font-size:11.5px">${esc(x.prompt)}</a><div class="serif">${esc(x.text)}</div></div>`).join('') || '<span class="muted" style="font-size:12.5px">quotes beyond the first 300 readings are on the prompt pages</span>'}</div></details>`).join('')}</div></div>`;
}

/* ---------- seed: the per-corpus libraries aligned into global features ---------- */
async function viewSeed(main, q) {
  const kind = q.kind || 'guidance';
  const [s, jobs] = await Promise.all([api(`/api/seed?kind=${kind}`), api('/api/jobs')]);
  const st = s.status, corpora = Object.entries(st.per_corpus);
  const flagsBy = {}; for (const f of s.flags) (flagsBy[f.feature] = flagsBy[f.feature] || []).push(f);
  const tree = s.groups.map(g => `<details class="tnode section" open><summary><b>${esc(g.name)}</b> <span class="tag">${esc(g.aspect)} · ${g.features.length} global features · ${fmt(g.support)} prompts</span> <span class="muted" style="font-size:12.5px">${esc(g.definition)}</span></summary><div class="kids">
      ${g.features.map(f => `<div class="tnode leaf ${f.polarity === 'forbid' ? 'forbid' : ''}"><span class="path">${f.corpora}</span><span class="read"><span class="verb">${esc(f.polarity)}</span> ${esc(f.name)}<span class="tag">${f.corpora} corpora · ${fmt(f.support)} prompts · round ${f.round}${(flagsBy[f.id] || []).length ? ` · <span class="err">${(flagsBy[f.id] || []).length} flags</span>` : ''}</span><br><span class="muted" style="font-size:12.5px">${esc(f.definition)}</span>
        <div style="margin-top:4px;font-size:12.5px">${f.members.map(m => `<div><span class="mono" style="font-size:11px;color:var(--muted)">${esc(m.corpus)}</span> <a href="${href('/feature/' + m.id)}">${esc(m.name)}</a> <span class="muted">· ${m.support} prompts${m.note === 'named' ? '' : ' · ' + esc(m.confidence || '')}</span>${(flagsBy[f.id] || []).some(x => x.other === m.id) ? ' <span class="err">flagged</span>' : ''}</div>`).join('')}</div></span></div>`).join('')}</div></details>`).join('');
  main.innerHTML = `<div class="split"><div>
    <h1>Seed library</h1>
    <p class="lede">${['guidance', 'material'].map(k => `<a href="${href('/seed', { kind: k })}" style="margin-right:10px;${k === kind ? 'font-weight:600;color:var(--ink)' : ''}">${k}</a>`).join('')}</p>
    <div class="block"><div class="t">${st.globals} global features in ${st.groups} groups · ${st.aligned} of ${st.cards} per-corpus features aligned · ${st.domain_specific} domain-specific · ${st.open} open · ${st.flags} flags (${st.standing} standing) · ${st.rounds} rounds</div>
      <table class="list"><tr><th>corpus</th><th class="n">features</th><th class="n">aligned</th><th class="n">domain-specific</th><th class="n">prompts under aligned features</th></tr>
      ${corpora.map(([c, d]) => `<tr><td><a href="${href('/library', { corpus: c, kind })}">${esc(c)}</a></td><td class="n">${d.features}</td><td class="n">${d.aligned} <span class="muted">(${pct(d.aligned / Math.max(d.features, 1))})</span></td><td class="n">${d.domain_specific}</td><td class="n">${pct(d.aligned_support / Math.max(d.support, 1))}</td></tr>`).join('')}</table></div>
    <div class="block"><div class="t">global features, with their members per corpus</div><div class="tree" style="max-height:none">${tree || '<span class="muted">none yet: run a round</span>'}</div></div>
    <div class="block"><div class="t">open and domain-specific features</div>${s.open_cards_note || ''}${s.open.length ? `<table class="list">${s.open.slice(0, 120).map(c => `<tr><td><span class="mono" style="font-size:11px;color:var(--muted)">${esc(c.corpus)}</span> <a href="${href('/feature/' + c.id)}">${esc(c.name)}</a></td><td class="muted" style="font-size:12.5px">${esc(c.definition.slice(0, 120))}</td><td class="n">${c.support}</td><td class="n">${esc(c.note || '')}</td></tr>`).join('')}</table>${s.open.length > 120 ? `<div class="muted">and ${s.open.length - 120} more</div>` : ''}` : '<span class="muted">none</span>'}</div>
  </div><div>
    <h1 style="font-size:22px">Run</h1>
    <form id="aform" class="form" style="grid-template-columns:90px minmax(0,1fr)">
      <label>step</label><select name="step"><option value="round" selected>round loop: embed, assign, judge, then reopen → cluster → name → assign until settled</option><option value="embed">embed the cards (no calls)</option><option value="assign">assign (open cards onto the globals)</option><option value="judge">judge (read-only)</option><option value="reopen">reopen the flagged members (no calls)</option><option value="cluster">cluster the open cards (no calls)</option><option value="name">name the candidate clusters</option></select>
      <label>batch model</label><input type="text" name="model" placeholder="assign and judge; default: the decomposition model" style="width:100%">
      <label>naming model</label><input type="text" name="codebook_model" placeholder="default gpt-5.6-sol" style="width:100%">
      <label>rounds</label><input type="number" name="rounds" value="5" min="1" max="20">
      <label>workers</label><input type="number" name="workers" value="64" min="1" max="512">
      <label>budget $</label><input type="number" name="budget" value="" placeholder="none" step="1">
      <span></span><span><button class="btn" type="button" id="arun">run</button> <span id="astatus" class="muted"></span></span></form>
    <p class="muted" style="font-size:12.5px">Units are the per-corpus features of every library of this kind. A global feature is one instruction several corpora give under their own domain nouns; a feature no other corpus echoes is domain-specific until a corpus arrives that does. Variants stay under their feature, so the hierarchy is global → per-corpus feature → variant.</p>
    <div class="block"><div class="t">align jobs</div>${jobs.filter(j => j.kind.startsWith('align')).length ? `<table class="list">${jobs.filter(j => j.kind.startsWith('align')).slice(0, 10).map(j => `<tr><td><a href="${href('/job/' + j.id)}">#${j.id}</a> ${esc(j.kind.slice(6))} ${esc(j.params.kind)}</td><td class="n">${esc(j.status)}</td></tr>`).join('')}</table>` : '<span class="muted">none yet</span>'}</div>
  </div></div>`;
  $('#arun').onclick = async () => { const d = Object.fromEntries(new FormData($('#aform'))); $('#astatus').textContent = 'starting';
    try { const r = await post('/api/align/jobs', { kind, step: d.step, model: d.model, codebook_model: d.codebook_model, rounds: +d.rounds, workers: +d.workers, budget: d.budget || null }); location.hash = href('/job/' + r.id); } catch (e) { $('#astatus').textContent = e.message; } };
}

/* ---------- the cube: the Library side. Fields on the left; the seed's hierarchy present in the slice on the right ---------- */
const FIELDS = ['corpus', 'domain', 'system', 'task', 'collection', 'bank_source', 'role', 'family', 'stage', 'subtask', 'polarity'];
const fieldQuery = q => Object.fromEntries(FIELDS.filter(f => q[f]).map(f => [f, q[f]]));
const filterChips = (q, kind) => FIELDS.filter(f => q[f]).flatMap(f => q[f].split(',').filter(Boolean).map(v => `<a class="chip on" href="${href('/cube', { ...q, kind, [f]: q[f].split(',').filter(x => x !== v).join(',') })}" title="remove"><span>${esc(f)} = ${esc(v)}</span><span class="n">×</span></a>`)).join(' ');

async function viewCube(main, q) {
  const kind = q.kind || 'guidance';
  const s = await api('/api/cube?' + new URLSearchParams({ kind, ...fieldQuery(q) }));
  const nodeHref = id => href('/node/' + id, { kind, ...fieldQuery(q) });
  const toggle = (field, value, on) => { const cur = (q[field] || '').split(',').filter(Boolean); return href('/cube', { ...q, kind, [field]: (on ? cur.filter(v => v !== value) : [...cur, value]).join(',') }); };
  const chip = (f, v) => `<a class="chip ${v.on ? 'on' : ''}" href="${toggle(f.field, v.value, v.on)}" data-v="${esc(v.value.toLowerCase())}"><span>${esc(v.value)}</span><span class="n">${fmt(v.prompts)}</span></a>`;
  const SHOW = 14;
  const fieldHtml = f => { const big = f.values.length > SHOW; return `<div class="field" data-field="${f.field}"><div class="t"><span>${esc(f.field)} <span class="muted">· ${f.values.length}</span></span>${big ? `<a href="#" class="more" data-n="${f.values.length}">all</a>` : ''}</div>${big ? `<input class="find" placeholder="find a value">` : ''}<div class="chips">${f.values.map((v, i) => `<span ${i >= SHOW && !v.on ? 'hidden' : ''} class="cw">${chip(f, v)}</span>`).join('')}</div></div>`; };
  const polar = { field: 'polarity', values: ['require', 'forbid'].map(v => ({ value: v, prompts: null, on: (q.polarity || '').split(',').includes(v) })) };
  const bar = (share, pol) => `<div class="gbar"><i class="${pol === 'forbid' ? 'forbid' : ''}" style="width:${Math.max(1, Math.round(100 * share))}%"></i></div>`;
  const leaf = f => `<div class="tnode leaf ${f.polarity === 'forbid' ? 'forbid' : ''}"><span class="path">${fmt(f.prompts)}<small>${pct(f.share)}</small></span><span class="read"><a href="${nodeHref(f.id)}">${pol(f.polarity)} ${esc(f.name)}</a><span class="tag">${f.corpora ? f.corpora.map(c => `<span class="corp">${esc(c)}</span>`).join('') : ''}${f.readings} readings${f.corpus && !f.corpora ? '' : ''}</span>${bar(f.share, f.polarity)}<span class="muted" style="font-size:12.5px">${esc(f.definition)}</span>${f.members ? `<div style="margin-top:4px;font-size:12.5px">${f.members.slice(0, 4).map(m => `<span class="corp">${esc(m.corpus)}</span><a href="${nodeHref(m.id)}">${esc(m.name)}</a> <span class="muted">${fmt(m.prompts)}</span>&nbsp;&nbsp; `).join('')}${f.members.length > 4 ? `<details style="display:inline"><summary class="muted" style="cursor:pointer;display:inline">+${f.members.length - 4} more</summary>${f.members.slice(4).map(m => `<div><span class="corp">${esc(m.corpus)}</span><a href="${nodeHref(m.id)}">${esc(m.name)}</a> <span class="muted">${fmt(m.prompts)}</span></div>`).join('')}</details>` : ''}</div>` : ''}</span></div>`;
  const tree = s.groups.map(g => `<details class="tnode section" open><summary><b>${esc(g.name)}</b> <span class="tag">${esc(g.aspect || '')} · ${g.features.length} global feature${g.features.length === 1 ? '' : 's'} · ${fmt(g.prompts)} prompts</span> <span class="muted" style="font-size:12.5px">${esc(g.definition)}</span></summary><div class="kids">${g.features.map(leaf).join('')}</div></details>`).join('');
  const local = s.unaligned.map(u => `<details class="tnode section" ${s.seed ? '' : 'open'}><summary><b>${esc(u.corpus)}</b> <span class="tag">${u.features} features of its own · ${fmt(u.prompts)} prompts</span></summary><div class="kids">${u.groups.map(g => `<details class="tnode section" open><summary>${esc(g.name)} <span class="tag">${esc(g.aspect || '')} · ${g.features.length} · ${fmt(g.prompts)} prompts</span></summary><div class="kids">${g.features.map(leaf).join('')}</div></details>`).join('')}</div></details>`).join('');
  main.innerHTML = `<div class="cube"><div class="fields">
      <div class="field"><div class="t"><span>kind</span></div><div class="chips">${['guidance', 'material'].map(k => `<a class="chip ${k === kind ? 'on' : ''}" href="${href('/cube', { ...q, kind: k })}"><span>${k}</span></a>`).join('')}</div></div>
      ${s.facets.map(fieldHtml).join('')}${fieldHtml(polar)}
    </div><div>
    <h1>Cube</h1>
    <p class="lede">Pick values on the left; the slice is every prompt matching them. The right side is what those prompts ask for: the seed's global features with their support in the slice, then each library's own features the seed has not absorbed.${s.seed ? '' : ' There is no seed yet, so each library stands for itself.'}</p>
    <div style="margin-bottom:10px">${filterChips(q, kind) || '<span class="muted" style="font-size:12.5px">the whole corpus</span>'}</div>
    <div class="measures"><span><b>${fmt(s.prompts)}</b><span class="k">prompts in the slice</span></span><span><b>${fmt(s.decomposed)}</b><span class="k">decomposed</span></span><span><b>${pct(s.covered / Math.max(s.decomposed, 1))}</b><span class="k">carry a feature</span></span><span><b>${pct(s.on_global / Math.max(s.decomposed, 1))}</b><span class="k">carry a global feature</span></span><span><b>${fmt(s.globals)}</b><span class="k">global features present</span></span><span><b>${s.libraries}</b><span class="k">libraries</span></span></div>
    <div class="block"><div class="t">global features in the slice, by prompts (share of the decomposed prompts)</div><div class="tree" style="max-height:none">${tree || '<span class="muted">none: no seed, or nothing aligned yet</span>'}</div></div>
    <div class="block"><div class="t">each library's own features in the slice (not under a global: open or domain-specific at the seed)</div><div class="tree" style="max-height:none">${local || '<span class="muted">none</span>'}</div></div>
  </div></div>`;
  for (const el of main.querySelectorAll('.field')) {
    const more = el.querySelector('.more'), find = el.querySelector('.find');
    if (more) more.onclick = e => { e.preventDefault(); const all = more.textContent === 'all'; el.querySelectorAll('.cw').forEach((w, i) => { w.hidden = !all && i >= SHOW && !w.querySelector('.on'); }); more.textContent = all ? 'fewer' : 'all'; };
    if (find) find.oninput = () => { const t = find.value.trim().toLowerCase(); el.querySelectorAll('.cw').forEach((w, i) => { w.hidden = t ? !w.querySelector('.chip').dataset.v.includes(t) : (i >= SHOW && !w.querySelector('.on')); }); };
  }
}

async function viewNode(main, id, q) {
  const kind = q.kind || 'guidance';
  const d = await api(`/api/cube/node/${id}?` + new URLSearchParams({ kind, ...fieldQuery(q) }));
  const nodeHref = i => href('/node/' + i, { kind, ...fieldQuery(q) });
  const wrow = w => `<tr><td class="serif">${w.polarity === 'forbid' ? pol('forbid') + ' ' : ''}${esc(w.declaration)}</td><td class="n">${fmt(w.prompts)}</td><td class="n muted">${fmt(w.corpus_prompts)}</td><td style="font-size:11.5px">${w.sample.map(p => `<a href="${href('/prompt/' + encodeURIComponent(p))}" class="mono">${esc(p)}</a>`).join(' ')}</td></tr>`;
  const member = m => `<div class="block"><div class="t">${d.global ? `<span class="corp">${esc(m.corpus)}</span><a href="${nodeHref(m.id)}">${esc(m.name)}</a> · <a href="${href('/feature/' + m.id)}" class="muted">its library page</a> · ` : ''}${fmt(m.prompts)} prompts in the slice · ${m.wordings.length} wordings${m.group ? ` · <span class="muted">${esc(m.group)}</span>` : ''}</div>
    <table class="list"><tr><th>wording</th><th class="n">prompts here</th><th class="n">in its corpus</th><th>e.g.</th></tr>${m.wordings.slice(0, 60).map(wrow).join('')}</table>${m.wordings.length > 60 ? `<details><summary class="muted" style="cursor:pointer">and ${m.wordings.length - 60} more wordings</summary><table class="list">${m.wordings.slice(60).map(wrow).join('')}</table></details>` : ''}</div>`;
  main.innerHTML = `<div class="crumb"><a href="${href('/cube', { ...q, kind })}">Cube</a> › ${d.group ? esc(d.group.name) + ' › ' : ''}${d.global ? 'global feature' : `<span class="corp">${esc(d.corpus)}</span> feature`}</div>
    <h1>${pol(d.polarity)} ${esc(d.name)}</h1><p class="lede">${esc(d.definition)}</p>
    <div class="facts" style="margin-bottom:16px">
      <span class="k">${d.global ? 'seed group' : 'library group'}</span><span>${d.group ? esc(d.group.name) + ' <span class="muted">· ' + esc(d.group.aspect || '') + ' · ' + esc(d.group.definition) + '</span>' : ''}</span>
      ${d.global ? `<span class="k">members</span><span>${d.members.length} per-corpus features from ${[...new Set(d.members.map(m => m.corpus))].length} corpora</span>` : `<span class="k">global feature</span><span>${d.global_of ? `aligned to <a href="${nodeHref(d.global_of)}">${esc(d.global_name)}</a>` : '<span class="muted">none yet: open or domain-specific at the seed</span>'} · <a href="${href('/feature/' + d.id)}" class="muted">its library page</a></span>`}
      <span class="k">in the slice</span><span>${fmt(d.prompts)} of ${fmt(d.selected)} selected prompts · ${fmt(d.readings)} readings</span>
      <span class="k">slice</span><span>${filterChips(q, kind) || '<span class="muted">the whole corpus</span>'}</span></div>
    <div class="cols2"><div>${d.members.map(member).join('') || '<span class="muted">no readings in this slice</span>'}</div>
    <div><div class="block"><div class="t">prompts in the slice carrying it, by readings${d.prompt_list.length < d.prompts ? ` (first ${d.prompt_list.length})` : ''}</div>
      <table class="list">${d.prompt_list.map(p => `<tr><td><a href="${href('/prompt/' + encodeURIComponent(p.id))}" class="mono" style="font-size:11.5px">${esc(p.id)}</a> <span class="corp">${esc(p.corpus)}</span> ${['role', 'family', 'stage'].filter(k => p.fields[k]).map(k => `<span class="tag">${esc(p.fields[k])}</span>`).join('')}<div class="serif" style="font-size:12.5px;color:var(--ink2)">${esc(p.head)}…</div></td><td class="n">${p.readings}</td></tr>`).join('')}</table></div></div></div>`;
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
