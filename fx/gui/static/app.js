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

function parseHash() { const h = location.hash.slice(1) || '/corpora'; const [path, qs] = h.split('?'); return { parts: path.split('/').filter(Boolean), q: Object.fromEntries(new URLSearchParams(qs || '')) }; }
async function route() {
  const { parts, q } = parseHash(); const view = parts[0] || 'corpora';
  $('#nav').innerHTML = [['corpora', 'Corpora'], ['prompts', 'Prompts'], ['queues', 'Queues']].map(([k, l]) => `<a href="${href('/' + k, { corpus: q.corpus })}" class="${view.startsWith(k.slice(0, 6)) ? 'on' : ''}">${l}</a>`).join('');
  if (ES) { ES.close(); ES = null; }
  const main = $('#main'); main.innerHTML = '<div class="loading">loading</div>'; window.scrollTo(0, 0);
  try {
    api('/api/spend').then(s => { $('#spend').textContent = `spent $${s.total.toFixed(2)}`; }).catch(() => {});
    if (view === 'corpora') await viewCorpora(main, q);
    else if (view === 'prompts') await viewPrompts(main, q);
    else if (view === 'prompt') await viewPrompt(main, decodeURIComponent(parts.slice(1).join('/')));
    else if (view === 'queues') await viewQueues(main, q);
    else if (view === 'job') await viewJob(main, +parts[1]);
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
      <label>workers</label><input type="number" name="workers" value="128" min="1" max="512">
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
    try { const r = await api('/api/import', { method: 'POST', body: fd }); $('#istatus').textContent = `added ${r.added}, skipped ${r.skipped}, ${r.duplicates_elsewhere} also in another corpus`; location.hash = href('/corpora', { corpus: r.corpus }); route(); } catch (err) { $('#istatus').textContent = err.message; } };
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
    <div id="jbody"></div><details style="margin-top:12px;font-size:12.5px"><summary class="muted">log · <span class="mono">${esc(j.log || '')}</span></summary><pre class="mono" id="jlog" style="font-size:11.5px;white-space:pre-wrap;max-height:40vh;overflow:auto"></pre></details><p style="margin-top:14px"><button class="btn quiet" id="stop">stop</button> <a href="${href('/prompts', { corpus: j.corpus, status: 'done' })}" style="margin-left:14px">decomposed prompts →</a> <a href="${href('/queues', { corpus: j.corpus })}" style="margin-left:14px">queues →</a></p>`;
  const render = d => { const share = d.total ? d.done / d.total : 0; const rate = d.elapsed && d.done ? d.elapsed / d.done : null;
    $('#jbody').innerHTML = `<div class="bar" style="max-width:720px"><i style="width:${(100 * share).toFixed(1)}%"></i></div>
      <div class="prev" style="margin-top:12px"><span><b>${fmt(d.done)} / ${fmt(d.total)}</b>prompts</span><span><b>${fmt(d.calls)}</b>calls</span><span><b>$${(d.spent || 0).toFixed(2)}</b>spent</span><span><b>${d.elapsed == null ? '' : fmtSec(d.elapsed)}</b>elapsed</span><span><b>${rate && d.status === 'running' ? fmtSec(rate * (d.total - d.done)) : d.status}</b>${d.status === 'running' ? 'remaining, projected' : 'status'}</span></div>
      <div class="recent" style="margin-top:12px">${(d.recent || []).map(r => `<div><a href="${href('/prompt/' + encodeURIComponent(r.id))}">${esc(r.id)}</a> · coverage ${pct(r.coverage)} · ${r.n_atoms} atoms · ${r.calls} calls${r.error ? ` · <span class="err">${esc(r.error)}</span>` : ''}</div>`).join('')}</div>${d.error ? `<p class="err">${esc(d.error)}</p>` : ''}`; };
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
    return `<div class="tnode leaf atom ${forbid ? 'forbid' : ''}" data-path="${esc(n.path)}"><span class="path">${esc(n.path)}</span><span>${n.readings.map(r => `<div class="read"><span class="verb">${esc(r.verb)}</span> ${esc(r.object)}${r.qualifier ? ` <span class="muted">${esc(r.qualifier)}</span>` : ''}${r.condition && r.condition !== 'always' ? ` <span class="muted">if ${esc(r.condition)}</span>` : ''}${r.polarity === 'forbid' ? ' <span class="tag">forbid</span>' : ''}</div>`).join('') || '<span class="muted"><i>no reading</i></span>'}${n.flags.length ? `<span class="tag">${esc(n.flags.join(' '))}</span>` : ''}</span></div>`;
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
      <span class="k">decomposition</span><span>${d ? (d.status === 'done' ? `coverage <b>${pct(d.coverage)}</b> of the instruction text · material ${pct(d.material_share)} · ${nAtoms} atoms · ${nReadings} readings · ${d.calls} calls, ${d.reasks} re-asks, ${d.seconds} s · ${esc(d.model)}` : `<span class="err">${esc(d.status)}: ${esc(d.error || '')}</span>`) : '<span class="muted"><i>not decomposed</i></span>'}</span></div>
    <div class="legend"><span><i style="background:var(--req-soft);border-bottom:1.5px solid var(--req)"></i>atom</span><span><i style="background:var(--for-soft);border-bottom:1.5px solid var(--for)"></i>forbid</span><span><i style="background:var(--mat-soft)"></i>material</span><span><i style="background:var(--gap-soft);border-bottom:1.5px dashed var(--gap)"></i>unrefined, or a gap the model declined</span></div>
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
    <div class="block"><div class="t">failed</div>${r.failed.length ? `<table class="list">${r.failed.map(x => row({ id: x.id, text: x.error })).join('')}</table>` : '<span class="muted">none</span>'}</div>`;
}

window.addEventListener('hashchange', route);
route();
