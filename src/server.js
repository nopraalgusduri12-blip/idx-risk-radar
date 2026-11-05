#!/usr/bin/env node
// IDX Risk Radar MCP server.
//   stdio (default):  for Claude Desktop / Claude Code as a local MCP server.
//   --http:           Streamable HTTP for a claude.ai custom connector (through a tunnel).
//                     Serves at /mcp/<RADAR_TOKEN>. Tools that control TradingView are only
//                     exposed over HTTP when RADAR_ALLOW_CONTROL=1.
import { readFileSync, existsSync } from 'fs';
import { spawn } from 'child_process';
import { createServer } from 'http';
import { randomBytes } from 'crypto';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { z } from 'zod';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODEL = join(ROOT, 'quant/data/model.json');
const DASH = join(ROOT, 'quant/idx-risk-radar.html');
const HTTP = process.argv.includes('--http');
const ALLOW_CONTROL = !HTTP || process.env.RADAR_ALLOW_CONTROL === '1';

const ok = obj => ({ content: [{ type: 'text', text: JSON.stringify(obj, null, 2) }] });
const fail = msg => ({ isError: true, content: [{ type: 'text', text: msg }] });
function loadModel() {
  if (!existsSync(MODEL)) throw new Error('No model output yet. Run the radar_refresh tool (or `npm run radar`) first.');
  return JSON.parse(readFileSync(MODEL, 'utf8'));
}
// Run one pipeline step as a child process from the repo root; resolves with its output.
function run(script, args = [], timeoutMs = 15 * 60 * 1000) {
  return new Promise((resolve, reject) => {
    const p = spawn(process.execPath, [join(ROOT, script), ...args], { cwd: ROOT });
    let out = '', err = '';
    const t = setTimeout(() => { p.kill(); reject(new Error(`${script} timed out`)); }, timeoutMs);
    p.stdout.on('data', d => { out += d; }); p.stderr.on('data', d => { err += d; });
    p.on('close', code => { clearTimeout(t); code === 0 ? resolve(out.trim()) : reject(new Error(`${script} exited ${code}: ${err || out}`.slice(0, 2000))); });
  });
}
const plain = s => ({ tk: s.tk, sector: s.sector, label: s.label, score: s.score, risk_grade: s.risk_grade, trend: s.trend, price: s.price,
  entry: s.plan.entry, bep: s.plan.bep, stop_loss: s.plan.stop, take_profit_1: s.plan.tp1, take_profit_2: s.plan.tp2,
  loss_to_stop_pct: -s.plan.risk_pct, risk_per_lot_rp: s.plan.lot_risk, tp1_before_stop_rate: s.bracket.p_tp_first, traded_value_bn: s.liq_bn });

function buildServer() {
  const server = new McpServer({ name: 'idx-risk-radar', version: '0.1.0' }, {
    instructions: 'Data-only risk screen for 48 liquid IDX stocks built from TradingView daily bars. Not investment advice. ' +
      'Start with radar_market (market conditions and how far to trust the model ranking), then radar_table or radar_stock. ' +
      'Always tell the user that the ranking has low predictive reliability when radar_market says reliable=false.',
  });

  server.registerTool('radar_market', {
    title: 'Market weather and model reliability',
    description: 'IHSG regime (caution level, drawdown, breadth, volatility), the shortlist, the min-variance split, and out-of-sample validation of the ranking models.',
    inputSchema: {},
  }, async () => {
    const m = loadModel();
    return ok({ as_of: m.as_of, universe: m.universe, market: m.market, reliable: m.reliable, shortlist: m.shortlist, portfolio: m.portfolio,
      validation: m.validation, ensemble_weights: m.weights, fees: m.fees, disclaimer: 'Price-data screen only; not investment advice.' });
  });

  server.registerTool('radar_table', {
    title: 'Trade parameter table',
    description: 'Entry, break-even (BEP), stop loss and take-profit levels plus risk grade and label for stocks, filtered and sorted.',
    inputSchema: {
      label: z.enum(['all', 'shortlist', 'watch', 'avoid']).default('all').describe('Filter by label'),
      sector: z.string().optional().describe('Exact sector name, e.g. "Banks"'),
      max_risk: z.enum(['A', 'B', 'C', 'D', 'E']).default('E').describe('Highest risk grade to include'),
      sort: z.enum(['score', 'risk', 'liquidity']).default('score'),
      limit: z.number().int().min(1).max(48).default(48),
    },
  }, async ({ label, sector, max_risk, sort, limit }) => {
    const m = loadModel();
    const key = { score: s => -s.score, risk: s => s.risk_pct, liquidity: s => -s.liq_bn }[sort];
    const rows = m.stocks.filter(s => (label === 'all' || s.label === label) && (!sector || s.sector === sector) && s.risk_grade <= max_risk)
      .sort((a, b) => key(a) - key(b)).slice(0, limit).map(plain);
    return ok({ as_of: m.as_of, count: rows.length, rows });
  });

  server.registerTool('radar_stock', {
    title: 'One stock in detail',
    description: 'Full radar record for one IDX ticker: plan levels, BEP, risk metrics, why it got its label, each model\'s rank, support/resistance.',
    inputSchema: { ticker: z.string().describe('IDX ticker without prefix, e.g. BBRI') },
  }, async ({ ticker }) => {
    const m = loadModel(), s = m.stocks.find(x => x.tk === ticker.toUpperCase().replace(/^IDX[_A-Z]*:/, ''));
    if (!s) return fail(`${ticker} is not in the radar universe (${m.stocks.map(x => x.tk).join(', ')}).`);
    const { spark, raw, ...rest } = s;
    return ok({ as_of: m.as_of, ...rest, features: raw });
  });

  if (ALLOW_CONTROL) {
    server.registerTool('radar_refresh', {
      title: 'Refresh data and models',
      description: 'Pulls fresh daily bars for all 48 stocks and IHSG from TradingView Desktop (must be running with CDP on port 9222), retrains and re-validates the models, and rebuilds the dashboard HTML. Takes a few minutes and switches the chart symbol temporarily.',
      inputSchema: {},
    }, async () => {
      try {
        const f = await run('quant/fetch.mjs');
        await run('quant/model.mjs', ['quant/data/daily.json', '--out', 'quant/data/model.json']);
        await run('quant/build-radar.mjs');
        const m = loadModel();
        return ok({ fetched: f.split('\n').slice(-1)[0], as_of: m.as_of, shortlist: m.shortlist, reliable: m.reliable, dashboard: DASH });
      } catch (e) { return fail(e.message); }
    });

    server.registerTool('radar_draw', {
      title: 'Draw levels in TradingView',
      description: 'Draws entry, BEP, stop loss, take profits and support/resistance as labelled horizontal lines on each stock\'s TradingView chart. Replaces earlier radar lines; leaves your own drawings alone.',
      inputSchema: { tickers: z.string().default('all').describe('"all" or comma-separated tickers, e.g. "BBRI,TLKM"') },
    }, async ({ tickers }) => {
      try { return ok({ log: await run('quant/draw.mjs', [tickers]) }); } catch (e) { return fail(e.message); }
    });
  }
  return server;
}

if (!HTTP) {
  await buildServer().connect(new StdioServerTransport());
} else {
  const port = +(process.env.PORT || 8787);
  const token = process.env.RADAR_TOKEN || randomBytes(18).toString('base64url');
  const path = `/mcp/${token}`;
  createServer(async (req, res) => {
    if (!req.url || req.url.split('?')[0] !== path) { res.writeHead(404).end(); return; }
    // Stateless: a fresh server and transport per request.
    const server = buildServer();
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    res.on('close', () => { transport.close(); server.close(); });
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', async () => {
      try { await server.connect(transport); await transport.handleRequest(req, res, body ? JSON.parse(body) : undefined); }
      catch (e) { if (!res.headersSent) res.writeHead(500).end(String(e.message)); }
    });
  }).listen(port, '127.0.0.1', () => {
    console.error(`IDX Risk Radar MCP (HTTP) on http://127.0.0.1:${port}${path}`);
    console.error(ALLOW_CONTROL ? 'Control tools ENABLED (refresh, draw).' : 'Read-only tools (set RADAR_ALLOW_CONTROL=1 to enable refresh/draw).');
    if (!process.env.RADAR_TOKEN) console.error('Set RADAR_TOKEN to keep this URL stable across restarts.');
  });
}
