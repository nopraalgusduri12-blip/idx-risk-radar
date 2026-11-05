#!/usr/bin/env node
// IDX Risk Radar web backend: one process that serves
//   /            the dashboard (behind a token login, works on phones and any browser)
//   /api/...     JSON API (market, stocks, assessments, portfolio risk, validation, status)
//   /mcp         MCP over Streamable HTTP, for claude.ai custom connectors and other MCP clients
//   /api/ingest  uploads from the PC collector (raw daily bars -> the server retrains), and
//   /api/publish uploads of results computed on the PC (model.json + risk.json)
// No TradingView needed here: the PC with TradingView only collects and uploads.
//
// Environment:
//   RADAR_TOKEN          required: access token for the dashboard, API and MCP
//   RADAR_INGEST_TOKEN   optional: separate token for uploads (defaults to RADAR_TOKEN)
//   RADAR_DATA_DIR       where data lives (use a persistent disk on a cloud host)
//   PORT / HOST          listen address (default 8080 / 0.0.0.0)
//   RADAR_SECURE_COOKIE  1 to force Secure cookies (automatic behind an HTTPS proxy)
import { createServer } from 'http';
import { createHmac, timingSafeEqual, randomBytes } from 'crypto';
import { writeFileSync, renameSync, mkdirSync, existsSync, statSync, readFileSync } from 'fs';
import { gunzipSync } from 'zlib';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { renderDashboard } from '../quant/dashboard.mjs';
import * as C from './core.js';

const PORT = +(process.env.PORT || 8080), HOST = process.env.HOST || '0.0.0.0';
let TOKEN = process.env.RADAR_TOKEN;
if (!TOKEN) {
  TOKEN = randomBytes(24).toString('base64url');
  console.error(`RADAR_TOKEN is not set. Generated a temporary one for this run:\n  ${TOKEN}\nSet RADAR_TOKEN to keep it stable.`);
}
if (TOKEN.length < 16) { console.error('RADAR_TOKEN must be at least 16 characters.'); process.exit(1); }
const INGEST_TOKEN = process.env.RADAR_INGEST_TOKEN || TOKEN;
const SESSION = createHmac('sha256', TOKEN).update('radar-session-v1').digest('base64url');
mkdirSync(C.DATA_DIR, { recursive: true });

// ---------- helpers ----------
const same = (a, b) => { const x = Buffer.from(String(a)), y = Buffer.from(String(b)); return x.length === y.length && timingSafeEqual(x, y); };
const cookies = req => Object.fromEntries((req.headers.cookie || '').split(';').map(c => c.trim().split('=')).filter(p => p.length === 2));
const bearer = req => (req.headers.authorization || '').replace(/^Bearer\s+/i, '');
const authed = req => same(cookies(req).radar_session || '', SESSION) || same(bearer(req), TOKEN);
const isHttps = req => process.env.RADAR_SECURE_COOKIE === '1' || req.headers['x-forwarded-proto'] === 'https';
const SEC = {
  'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
  'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
};
function send(res, code, body, type = 'application/json; charset=utf-8', extra = {}) {
  const data = typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body);
  res.writeHead(code, { 'Content-Type': type, 'Cache-Control': 'no-store', ...SEC, ...extra }); res.end(data);
}
const json = (res, code, obj) => send(res, code, obj);
function readBody(req, limit) {
  return new Promise((ok, fail) => {
    const chunks = []; let n = 0;
    req.on('data', c => { n += c.length; if (n > limit) { fail(Object.assign(new Error('Body too large'), { code: 413 })); req.destroy(); } else chunks.push(c); });
    req.on('end', () => { try { let b = Buffer.concat(chunks); if (/gzip/i.test(req.headers['content-encoding'] || '')) b = gunzipSync(b); ok(b); } catch (e) { fail(e); } });
    req.on('error', fail);
  });
}
const parseJson = buf => { try { return buf.length ? JSON.parse(buf.toString('utf8')) : undefined; } catch { throw Object.assign(new Error('Invalid JSON'), { code: 400 }); } };
// Simple per-IP limiter for the login form.
const attempts = new Map();
function limited(ip) { const now = Date.now(), a = (attempts.get(ip) || []).filter(t => now - t < 60_000); a.push(now); attempts.set(ip, a); return a.length > 10; }
const writeAtomic = (path, data) => { const tmp = path + '.tmp'; writeFileSync(tmp, data); renameSync(tmp, path); };
const updateMeta = patch => writeAtomic(C.PATHS.meta, JSON.stringify({ ...C.loadMeta(), ...patch }));

// ---------- background pipeline job ----------
const job = { state: 'idle', step: null, started: null, finished: null, error: null };
async function startJob(source) {
  if (job.state === 'running') return false;
  Object.assign(job, { state: 'running', step: 'starting', started: new Date().toISOString(), finished: null, error: null, source });
  C.runModels(s => { job.step = s; })
    .then(() => { Object.assign(job, { state: 'done', finished: new Date().toISOString() }); updateMeta({ models_updated_at: job.finished }); })
    .catch(e => { Object.assign(job, { state: 'failed', finished: new Date().toISOString(), error: e.message.slice(-500) }); });
  return true;
}

// ---------- dashboard rendering (cached on file changes) ----------
let dashCache = { key: '', html: '' };
function dashboard() {
  const mt = p => existsSync(p) ? statSync(p).mtimeMs : 0;
  const key = [C.PATHS.model, C.PATHS.risk, C.PATHS.daytrade, C.PATHS.meta].map(mt).join('|');
  if (dashCache.key === key) return dashCache.html;
  const meta = C.loadMeta();
  let { html } = renderDashboard(C.loadModel(), C.loadRiskOrNull(), { drawn: meta.drawn || [], daytrade: C.loadDaytradeOrNull() });
  const updated = meta.models_updated_at || meta.published_at || meta.uploaded_at;
  const bar = `<div style="display:flex;flex-wrap:wrap;gap:6px 14px;justify-content:space-between;align-items:center;font-size:13px;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:10px">
    <span>Server copy${updated ? ` · updated ${new Date(updated).toLocaleString('en-GB', { timeZone: 'Asia/Jakarta' })} WIB` : ''}${meta.source ? ` · from ${meta.source}` : ''}</span>
    <span><a href="/api/status" style="color:var(--accent)">Status</a> · <a href="/logout" style="color:var(--accent)">Log out</a></span></div>`;
  html = html.replace('<div class="wrap">', () => `<div class="wrap">${bar}`);
  dashCache = { key, html }; return html;
}
const page = (title, inner) => `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${title}</title>
<style>:root{--bg:#f3f5f4;--fg:#14201d;--muted:#586763;--line:#d6dedb;--accent:#0d6b6e;--surface:#fff;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#0f1514;--fg:#e3ebe8;--muted:#93a39f;--line:#2a3735;--accent:#4fc1c0;--surface:#161e1d;--bad:#f07b6a;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 "Segoe UI",system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;padding:0 16px}
main{background:var(--surface);border:1px solid var(--line);padding:28px;max-width:380px;width:100%;display:grid;gap:14px}
h1{margin:0;font-size:24px}p{margin:0;color:var(--muted);font-size:14px}label{display:grid;gap:6px;font-size:14px;color:var(--muted)}
input{font:16px ui-monospace,Consolas,monospace;padding:10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{font:600 16px system-ui,sans-serif;padding:10px;border:0;border-radius:6px;background:var(--accent);color:#fff;cursor:pointer}.err{color:var(--bad)}</style></head><body><main>${inner}</main></body></html>`;
const loginPage = err => page('IDX Risk Radar', `<h1>IDX Risk Radar</h1><p>Enter the access token set on this server (RADAR_TOKEN).</p>
  ${err ? `<p class="err">${err}</p>` : ''}<form method="post" action="/login"><label for="token">Access token<input id="token" name="token" type="password" autocomplete="current-password" required autofocus></label><br><button type="submit">Open dashboard</button></form>`);

// ---------- MCP over HTTP (stateless) ----------
const statusTool = { name: 'radar_status', config: { title: 'Data freshness and pipeline status',
  description: 'When the data was last uploaded and the models last retrained, and whether a retrain is running.', inputSchema: {} },
  handler: async () => ({ job, data: C.loadMeta() }) };
async function handleMcp(req, res) {
  const body = parseJson(await readBody(req, 1_000_000));
  const server = C.buildMcp({ extra: [statusTool] });
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on('close', () => { transport.close(); server.close(); });
  await server.connect(transport);
  await transport.handleRequest(req, res, body);
}

// ---------- routes ----------
async function route(req, res) {
  const url = new URL(req.url, 'http://x'), p = url.pathname, m = req.method;

  if (p === '/healthz') { let as_of = null; try { as_of = C.loadModel().as_of; } catch {} return json(res, 200, { ok: true, as_of }); }
  if (p === '/login' && m === 'GET') return send(res, 200, loginPage(), 'text/html; charset=utf-8');
  if (p === '/login' && m === 'POST') {
    if (limited(req.socket.remoteAddress)) return send(res, 429, loginPage('Too many attempts. Wait a minute and try again.'), 'text/html; charset=utf-8');
    const token = new URLSearchParams((await readBody(req, 10_000)).toString()).get('token') || '';
    if (!same(token, TOKEN)) return send(res, 401, loginPage('That token is not right.'), 'text/html; charset=utf-8');
    const cookie = `radar_session=${SESSION}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${60 * 60 * 24 * 30}${isHttps(req) ? '; Secure' : ''}`;
    return send(res, 303, '', 'text/plain', { Location: '/', 'Set-Cookie': cookie });
  }
  if (p === '/logout') return send(res, 303, '', 'text/plain', { Location: '/login', 'Set-Cookie': 'radar_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0' });

  // MCP: Authorization: Bearer <token>, or the token in the path for clients that cannot send headers.
  if (p === '/mcp' || p.startsWith('/mcp/')) {
    const pathTok = p.startsWith('/mcp/') ? decodeURIComponent(p.slice(5)) : '';
    if (!(same(bearer(req), TOKEN) || (pathTok && same(pathTok, TOKEN)))) return json(res, 401, { error: 'unauthorized' });
    return handleMcp(req, res);
  }

  // Uploads from the PC collector use the ingest token.
  if (p === '/api/ingest' || p === '/api/publish') {
    if (m !== 'POST') return json(res, 405, { error: 'POST only' });
    if (!same(bearer(req), INGEST_TOKEN)) return json(res, 401, { error: 'unauthorized' });
    const body = parseJson(await readBody(req, 80_000_000));
    if (p === '/api/ingest') {
      if (!body?.bars || !body.index || !body.bars[body.index]) return json(res, 400, { error: 'Expected daily.json: { index, bars: { symbol: [bars] } }' });
      writeAtomic(C.PATHS.daily, JSON.stringify(body));
      updateMeta({ uploaded_at: new Date().toISOString(), source: body.source || 'collector', symbols: Object.keys(body.bars).length, drawn: body.drawn || [] });
      const started = await startJob('ingest');
      return json(res, 202, { ok: true, retraining: started, note: started ? 'Models are retraining; check /api/status.' : 'A retrain is already running; this upload will be used next time.' });
    }
    if (!body?.model?.stocks || !body?.risk?.stocks) return json(res, 400, { error: 'Expected { model, risk } from model.json and risk.json' });
    writeAtomic(C.PATHS.model, JSON.stringify(body.model)); writeAtomic(C.PATHS.risk, JSON.stringify(body.risk));
    if (body.daytrade?.plan) writeAtomic(C.PATHS.daytrade, JSON.stringify(body.daytrade));
    updateMeta({ published_at: new Date().toISOString(), models_updated_at: new Date().toISOString(), source: body.source || 'collector', drawn: body.drawn || [] });
    return json(res, 200, { ok: true, as_of: body.model.as_of });
  }

  // Everything below needs a login (cookie) or the bearer token.
  if (!authed(req)) return p.startsWith('/api/') ? json(res, 401, { error: 'unauthorized' }) : send(res, 303, '', 'text/plain', { Location: '/login' });

  if (p === '/' && m === 'GET') {
    try { return send(res, 200, dashboard(), 'text/html; charset=utf-8'); }
    catch (e) { return send(res, 200, page('IDX Risk Radar', `<h1>Waiting for data</h1><p>${e.message}</p><p>Run the collector on the PC with TradingView: <code>npm run collect</code>.</p><p><a href="/logout">Log out</a></p>`), 'text/html; charset=utf-8'); }
  }
  if (p === '/api/status') return json(res, 200, { job, data: C.loadMeta(), files: Object.fromEntries(Object.entries(C.PATHS).map(([k, v]) => [k, existsSync(v)])) });
  if (p === '/api/retrain' && m === 'POST') { if (!existsSync(C.PATHS.daily)) return json(res, 409, { error: 'No daily data uploaded yet.' }); return json(res, 202, { started: await startJob('manual') }); }
  if (p === '/api/market') return json(res, 200, C.market());
  if (p === '/api/stocks') {
    const q = Object.fromEntries(url.searchParams); if (q.limit) q.limit = +q.limit;
    return json(res, 200, C.table(q));
  }
  let mm;
  if ((mm = p.match(/^\/api\/stocks\/([A-Za-z0-9:_]+)$/))) return json(res, 200, C.stock(mm[1]));
  if ((mm = p.match(/^\/api\/assess\/([A-Za-z0-9:_]+)$/))) {
    const R = C.loadRisk(), s = R.stocks.find(x => x.tk === C.normTicker(mm[1]));
    return s ? json(res, 200, { as_of: R.as_of, regime: { p_high_vol: R.regime.p_high_vol, high: R.regime.high }, ...s }) : json(res, 404, { error: 'No assessment for that ticker' });
  }
  if (p === '/api/portfolio' && m === 'POST') { const b = parseJson(await readBody(req, 100_000)); return json(res, 200, C.portfolioRisk(C.loadRisk(), b?.holdings || [])); }
  if (p === '/api/validation') return json(res, 200, C.validation());
  if (p === '/api/daytrade') return json(res, 200, C.daytrade({ all: url.searchParams.get('all') === '1' }));
  if (p === '/api/model.json') return send(res, 200, readFileSync(C.PATHS.model));
  if (p === '/api/risk.json') return send(res, 200, readFileSync(C.PATHS.risk));
  return json(res, 404, { error: 'not found' });
}

createServer((req, res) => {
  route(req, res).catch(e => {
    const code = e.code === 413 || e.code === 400 ? e.code : /No (model|integrated)/.test(e.message) ? 503 : /not in the radar|Not in the universe|Give at least/.test(e.message) ? 400 : 500;
    if (!res.headersSent) json(res, code, { error: e.message }); else res.end();
  });
}).listen(PORT, HOST, () => {
  console.error(`IDX Risk Radar web backend on http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}  (data: ${C.DATA_DIR})`);
});
