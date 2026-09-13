// node tools/smoke_gui.js http://127.0.0.1:8782 fx/gui/static/app.js  : every page routed once against a running server, with a stub DOM
// Headless smoke test of the site's script: a stub DOM, the real API, every view routed once; reports thrown errors.
const fs = require('fs');
const BASE = process.argv[2] || 'http://127.0.0.1:8782';
const src = fs.readFileSync(process.argv[3], 'utf8');
function stub(tag = 'div') {
  const store = {};
  const el = new Proxy({}, {
    get(_, p) {
      if (p in store) return store[p];
      if (p === 'dataset') return store.dataset = {};
      if (p === 'classList') return { toggle() {}, contains() { return false; }, add() {}, remove() {} };
      if (p === 'style') return store.style = {};
      if (p === 'value') return '12';
      if (p === 'checked' || p === 'hidden' || p === 'open') return false;
      if (p === 'files') return [];
      if (p === 'querySelectorAll') return () => [];
      if (p === 'querySelector') return () => stub();
      if (p === 'getAttribute') return a => a === 'viewBox' ? '-650 -500 1300 1000' : null;
      if (p === 'closest') return () => stub();
      if (p === 'nextElementSibling' || p === 'previousElementSibling') return stub();
      if (p === 'getBoundingClientRect') return () => ({ left: 0, top: 0, width: 1, height: 1 });
      if (p === 'innerHTML' || p === 'textContent') return store[p] || '';
      if (p === 'then') return undefined;
      return () => stub();
    },
    set(_, p, v) { store[p] = v; return true; },
  });
  return el;
}
const main = stub('main');
const els = { '#main': main, '#nav': stub(), '#gear': stub(), '#spend': stub() };
global.document = { querySelector: s => els[s] || stub(), querySelectorAll: () => [], addEventListener() {}, body: stub() };
global.window = { addEventListener() {}, scrollTo() {}, location: { hash: '' }, _facets: [] };
global.location = window.location;
global.CSS = { escape: s => s };
global.EventSource = class { constructor() {} close() {} addEventListener() {} };
global.FormData = class { constructor() {} get() { return ''; } set() {} delete() {} append() {} [Symbol.iterator]() { return [][Symbol.iterator](); } };
global.fetch = (path, opts) => fetchHttp(path, opts);
const http = require('node:http');
function fetchHttp(path, opts) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE);
    const req = http.request(url, { method: (opts && opts.method) || 'GET', headers: (opts && opts.headers) || {} }, res => {
      let body = ''; res.on('data', d => body += d); res.on('end', () => resolve({ ok: res.statusCode < 300, status: res.statusCode, text: async () => body, json: async () => JSON.parse(body) }));
    });
    req.on('error', reject); if (opts && opts.body && typeof opts.body === 'string') req.write(opts.body); req.end();
  });
}
const errors = [];
process.on('unhandledRejection', e => errors.push('unhandled: ' + (e && e.stack || e)));
const orig = console.error; console.error = (...a) => { errors.push('console.error: ' + a.map(x => x && x.stack || x).join(' ')); };
(async () => {
  const script = src.replace(/window\.addEventListener\('hashchange', route\);\s*route\(\);.*$/s, '');
  eval(script + '\nglobal.__route = route;');
  const ids = {};
  try { const c = await (await fetchHttp('/api/cube?kind=guidance')).json(); ids.glob = c.groups[0] && c.groups[0].features[0] && c.groups[0].features[0].id; ids.local = c.unaligned[0] && c.unaligned[0].groups[0].features[0].id; } catch (e) { errors.push('api cube: ' + e); }
  try { const p = await (await fetchHttp('/api/cube?view=prompts&limit=1')).json(); ids.prompt = p.list[0] && p.list[0].id; } catch (e) { errors.push('api prompts: ' + e); }
  try { const i = await (await fetchHttp('/api/ingest')).json(); ids.job = i.jobs[0] && i.jobs[0].id; ids.corpus = i.corpora[0] && i.corpora[0].name; } catch (e) { errors.push('api ingest: ' + e); }
  const views = ['/library', '/library?view=prompts', `/library?tree=${ids.corpus}`, '/library?view=features&role=staged&ring=similarity', `/node/${ids.glob}`, `/node/${ids.local}?corpus=${ids.corpus}`, `/feature/${ids.local}`, `/prompt/${encodeURIComponent(ids.prompt)}`,
                 '/ingest/corpora', '/ingest/import', '/ingest/profiles', '/ingest/jobs', '/ingest/history', `/job/${ids.job}`, '/settings'];
  for (const v of views) {
    window.location.hash = '#' + v; main.innerHTML = '';
    const before = errors.length;
    try { await global.__route(); } catch (e) { errors.push(v + ': threw ' + (e.stack || e)); }
    const html = String(main.innerHTML);
    const m = html.match(/<p class="err">([^<]*)<\/p>/);
    const status = m ? 'ERR ' + m[1] : errors.length > before ? 'ERR (console)' : `ok ${html.length} chars`;
    console.log(v.padEnd(60), status);
  }
  if (errors.length) { console.log('\nerrors:'); for (const e of errors) console.log(' -', String(e).split('\n').slice(0, 3).join(' | ')); process.exit(1); }
})();
