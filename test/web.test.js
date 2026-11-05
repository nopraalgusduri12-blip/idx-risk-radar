// End-to-end test of the web backend against the latest model output in quant/data (skipped if absent).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'child_process';
import { mkdtempSync, copyFileSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { gzipSync } from 'zlib';

const HAVE = existsSync('quant/data/model.json') && existsSync('quant/data/risk.json');
const TOKEN = 'test-token-0123456789abcdef', PORT = 18000 + Math.floor(Math.random() * 2000), BASE = `http://127.0.0.1:${PORT}`;

test('web backend: auth, dashboard, API, MCP and uploads', { skip: !HAVE && 'no model output in quant/data' }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'radar-web-'));
  copyFileSync('quant/data/model.json', join(dir, 'model.json')); copyFileSync('quant/data/risk.json', join(dir, 'risk.json'));
  const srv = spawn(process.execPath, ['src/web.js'], { env: { ...process.env, PORT: String(PORT), HOST: '127.0.0.1', RADAR_TOKEN: TOKEN, RADAR_DATA_DIR: dir }, stdio: 'ignore' });
  try {
    for (let k = 0; k < 50; k++) { try { if ((await fetch(BASE + '/healthz')).ok) break; } catch {} await new Promise(r => setTimeout(r, 100)); }
    const health = await (await fetch(BASE + '/healthz')).json();
    assert.equal(health.ok, true); assert.ok(health.as_of);

    // Without credentials: API refuses, pages redirect to the login form.
    assert.equal((await fetch(BASE + '/api/market')).status, 401);
    const home = await fetch(BASE + '/', { redirect: 'manual' });
    assert.equal(home.status, 303); assert.equal(home.headers.get('location'), '/login');

    // Login: wrong token refused, right token sets an HttpOnly session cookie that opens the dashboard.
    const form = t => ({ method: 'POST', redirect: 'manual', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'token=' + encodeURIComponent(t) });
    assert.equal((await fetch(BASE + '/login', form('wrong-token-wrong-token'))).status, 401);
    const login = await fetch(BASE + '/login', form(TOKEN));
    assert.equal(login.status, 303);
    const cookie = login.headers.get('set-cookie').split(';')[0];
    assert.match(login.headers.get('set-cookie'), /HttpOnly/);
    const dash = await fetch(BASE + '/', { headers: { Cookie: cookie } });
    assert.equal(dash.status, 200);
    const html = await dash.text();
    assert.match(html, /IDX Risk Radar/); assert.match(html, /Log out/);
    assert.ok(dash.headers.get('content-security-policy'));

    // API with the bearer token.
    const auth = { Authorization: `Bearer ${TOKEN}` };
    const mk = await (await fetch(BASE + '/api/market', { headers: auth })).json();
    assert.ok(mk.market && Array.isArray(mk.shortlist));
    const tbl = await (await fetch(BASE + '/api/stocks?label=all&limit=5', { headers: auth })).json();
    assert.equal(tbl.rows.length, 5); assert.ok('bep' in tbl.rows[0] && 'integrated_verdict' in tbl.rows[0]);
    const one = await (await fetch(BASE + '/api/stocks/IDX:BBRI', { headers: auth })).json();
    assert.equal(one.tk, 'BBRI'); assert.ok(one.integrated_risk);
    assert.equal((await fetch(BASE + '/api/stocks/NOPE', { headers: auth })).status, 400);
    const port = await (await fetch(BASE + '/api/portfolio', { method: 'POST', headers: { ...auth, 'Content-Type': 'application/json' },
      body: JSON.stringify({ holdings: [{ ticker: 'BBCA', amount: 5e7 }, { ticker: 'TLKM', amount: 5e7 }] }) })).json();
    assert.ok(port.var95_1d > 0 && port.diversification_benefit > 0);

    // MCP over HTTP: header token and path token both work; no token is refused.
    const rpc = (path, h, b) => fetch(BASE + path, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', ...h }, body: JSON.stringify(b) });
    const init = { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '0' } } };
    assert.equal((await rpc('/mcp', {}, init)).status, 401);
    assert.equal((await rpc('/mcp', auth, init)).status, 200);
    const list = await (await rpc(`/mcp/${TOKEN}`, {}, { jsonrpc: '2.0', id: 2, method: 'tools/list' })).text();
    for (const t of ['radar_market', 'radar_assess', 'radar_portfolio', 'radar_status']) assert.match(list, new RegExp(t));
    assert.doesNotMatch(list, /radar_draw/);

    // Uploads: wrong token refused, bad body rejected, a gzipped publish accepted.
    assert.equal((await fetch(BASE + '/api/publish', { method: 'POST', headers: { Authorization: 'Bearer nope' }, body: '{}' })).status, 401);
    assert.equal((await fetch(BASE + '/api/ingest', { method: 'POST', headers: auth, body: '{"x":1}' })).status, 400);
    const { readFileSync } = await import('fs');
    const body = gzipSync(JSON.stringify({ model: JSON.parse(readFileSync('quant/data/model.json')), risk: JSON.parse(readFileSync('quant/data/risk.json')), source: 'test' }));
    const pub = await fetch(BASE + '/api/publish', { method: 'POST', headers: { ...auth, 'Content-Encoding': 'gzip', 'Content-Type': 'application/json' }, body });
    assert.equal(pub.status, 200);
    const st = await (await fetch(BASE + '/api/status', { headers: auth })).json();
    assert.equal(st.data.source, 'test');
  } finally { srv.kill(); }
});
