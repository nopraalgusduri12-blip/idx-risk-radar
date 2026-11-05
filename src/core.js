// Shared by the local MCP server (src/server.js) and the web backend (src/web.js):
// data access, portfolio risk, the refresh pipeline, and the MCP tool definitions.
import { readFileSync, existsSync, statSync } from 'fs';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';
import { z } from 'zod';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
// Where daily.json, model.json and risk.json live. Point it at a persistent disk on a server.
export const DATA_DIR = resolve(process.env.RADAR_DATA_DIR || join(ROOT, 'quant/data'));
export const PATHS = { daily: join(DATA_DIR, 'daily.json'), model: join(DATA_DIR, 'model.json'), risk: join(DATA_DIR, 'risk.json'), meta: join(DATA_DIR, 'meta.json') };

// JSON files cached by modification time, so a refresh is picked up without a restart.
const cache = new Map();
function readJson(path) {
  if (!existsSync(path)) return null;
  const m = statSync(path).mtimeMs, c = cache.get(path);
  if (c && c.m === m) return c.v;
  const v = JSON.parse(readFileSync(path, 'utf8')); cache.set(path, { m, v }); return v;
}
export function loadModel() { const m = readJson(PATHS.model); if (!m) throw new Error('No model output yet. Upload data from the collector or run `npm run radar`.'); return m; }
export function loadRisk() { const r = readJson(PATHS.risk); if (!r) throw new Error('No integrated risk output yet. Run the pipeline (`npm run risk`).'); return r; }
export const loadRiskOrNull = () => readJson(PATHS.risk);
export const loadMeta = () => readJson(PATHS.meta) || {};
export const normTicker = t => String(t).toUpperCase().trim().replace(/^IDX[_A-Z]*:/, '');

// Portfolio risk for holdings [{ticker, amount}] from the shrunk covariance and per-stock stress results.
export function portfolioRisk(R, holdings) {
  const P = R.portfolio, idx = Object.fromEntries(P.tickers.map((t, k) => [t, k])), by = Object.fromEntries(R.stocks.map(s => [s.tk, s]));
  const h = holdings.map(x => ({ tk: normTicker(x.ticker), amt: +x.amount })).filter(x => x.amt > 0);
  if (!h.length) throw new Error('Give at least one holding with a positive amount.');
  const missing = h.filter(x => !(x.tk in idx)).map(x => x.tk); if (missing.length) throw new Error(`Not in the universe: ${missing.join(', ')}`);
  const V = h.reduce((a, x) => a + x.amt, 0), cov = (a, b) => P.cov_daily[idx[a]][idx[b]];
  const sw = h.map(a => h.reduce((s0, b) => s0 + cov(a.tk, b.tk) * b.amt, 0)), varP = h.reduce((s0, x, i) => s0 + x.amt * sw[i], 0), sdP = Math.sqrt(varP);
  const alone = h.reduce((s0, x) => s0 + x.amt * Math.sqrt(cov(x.tk, x.tk)), 0);
  const stress = f => h.reduce((s0, x) => s0 + x.amt * f(by[x.tk]), 0);
  return {
    as_of: R.as_of, total: V, vol_ann: sdP / V * Math.sqrt(252), var95_1d: 1.645 * sdP, var99_1d: 2.326 * sdP, var95_20d: 1.645 * sdP * Math.sqrt(20), es95_1d: 2.063 * sdP,
    diversification_benefit: 1 - sdP / alone, effective_n: 1 / h.reduce((s0, x) => s0 + (x.amt / V) ** 2, 0),
    risk_contribution: Object.fromEntries(h.map((x, i) => [x.tk, x.amt * sw[i] / varP])),
    stress_rp: { ihsg_minus10_bad: stress(s => s.stress.ihsg10.bad), ihsg_minus20_bad: stress(s => s.stress.ihsg20.bad),
      worst_month_replay: stress(s => s.stress.hist ?? s.stress.ihsg20.expected), sum_es95_20d: stress(s => s.sim.es95_20d) },
    weighted_p_drop10: stress(s => s.integrated.p_crash10) / V,
    verdicts: Object.fromEntries(h.map(x => [x.tk, by[x.tk].assessment.verdict])), market_regime_high_vol: R.regime.high,
    note: 'Parametric normal VaR on a Ledoit-Wolf shrunk covariance; stress uses downside beta and each stock\'s own tail. Not investment advice.',
  };
}

// Run one pipeline step as a child process from the repo root; resolves with its output.
export function run(script, args = [], timeoutMs = 15 * 60 * 1000) {
  return new Promise((ok, fail) => {
    const p = spawn(process.execPath, [join(ROOT, script), ...args], { cwd: ROOT, env: process.env });
    let out = '', err = '';
    const t = setTimeout(() => { p.kill(); fail(new Error(`${script} timed out`)); }, timeoutMs);
    p.stdout.on('data', d => { out += d; }); p.stderr.on('data', d => { err += d; });
    p.on('close', code => { clearTimeout(t); code === 0 ? ok(out.trim()) : fail(new Error(`${script} exited ${code}: ${err || out}`.slice(-2000))); });
  });
}
// Retrain and re-assess from the daily.json already in DATA_DIR (no TradingView needed).
export async function runModels(log = () => {}) {
  log('training ranking and risk-forecast models');
  await run('quant/model.mjs', [PATHS.daily, '--out', PATHS.model], 40 * 60 * 1000);
  log('running the integrated risk engine');
  await run('quant/risk.mjs', [PATHS.daily, PATHS.model, '--out', PATHS.risk], 60 * 60 * 1000);
  log('done');
}

// Compact one-row view of a stock for tables.
export const plain = (s, k) => ({ tk: s.tk, sector: s.sector, label: s.label, score: s.score, risk_grade: s.risk_grade, trend: s.trend, price: s.price,
  entry: s.plan.entry, bep: s.plan.bep, stop_loss: s.plan.stop, take_profit_1: s.plan.tp1, take_profit_2: s.plan.tp2,
  loss_to_stop_pct: -s.plan.risk_pct, risk_per_lot_rp: s.plan.lot_risk, tp1_before_stop_rate: s.bracket.p_tp_first, traded_value_bn: s.liq_bn,
  integrated_verdict: k?.assessment.verdict ?? null, integrated_p_drop10_20d: k?.integrated.p_crash10 ?? s.forecast?.p_crash10,
  simulated_p_stop_20d: k?.sim.p_stop ?? null, size_factor: k?.assessment.size_factor ?? s.forecast?.size_mult,
  noise_aware_stop: k && k.assessment.noise_stop < s.plan.stop ? k.assessment.noise_stop : null, realistic_target_20d: k?.assessment.realistic_target ?? null });

// Stocks table with the integrated assessment merged in (labels can only become more cautious).
export function table({ label = 'all', sector, max_risk = 'E', sort = 'score', limit = 48 } = {}) {
  const m = loadModel(), R = loadRiskOrNull(), byTk = R ? Object.fromEntries(R.stocks.map(s => [s.tk, s])) : {};
  const adj = s => { const k = byTk[s.tk]; let l = s.label; if (k?.assessment.verdict === 'Severe') l = 'avoid'; else if (k?.assessment.verdict === 'High' && l === 'shortlist') l = 'watch'; return { ...s, label: l }; };
  const key = { score: s => -s.score, risk: s => s.risk_pct, liquidity: s => -s.liq_bn }[sort] || (s => -s.score);
  const rows = m.stocks.map(adj).filter(s => (label === 'all' || s.label === label) && (!sector || s.sector === sector) && s.risk_grade <= max_risk)
    .sort((a, b) => key(a) - key(b)).slice(0, limit).map(s => plain(s, byTk[s.tk]));
  return { as_of: m.as_of, count: rows.length, rows };
}
export function market() {
  const m = loadModel(), R = loadRiskOrNull();
  return { as_of: m.as_of, universe: m.universe, market: m.market, regime: R ? { ...R.regime, history: undefined } : null, reliable_return_edge: m.reliable,
    risk_reliable: m.risk_reliable, shortlist: table({ label: 'shortlist' }).rows.map(r => r.tk), portfolio: m.portfolio, fees: m.fees,
    data: loadMeta(), disclaimer: 'Price-data screen only; not investment advice.' };
}
export function stock(ticker) {
  const m = loadModel(), tk = normTicker(ticker), s = m.stocks.find(x => x.tk === tk);
  if (!s) throw new Error(`${ticker} is not in the radar universe (${m.stocks.map(x => x.tk).join(', ')}).`);
  const { spark, raw, ...rest } = s;
  const k = loadRiskOrNull()?.stocks.find(x => x.tk === tk) ?? null;
  return { as_of: m.as_of, ...rest, features: raw, integrated_risk: k };
}
export function validation() {
  const m = loadModel(), R = loadRiskOrNull();
  return { ml_models: m.validation, reliable_return_edge: m.reliable, risk_reliable: m.risk_reliable, integrated_engine: R?.validation ?? null };
}

const ok = obj => ({ content: [{ type: 'text', text: JSON.stringify(obj, null, 2) }] });
const fail = msg => ({ isError: true, content: [{ type: 'text', text: msg }] });
const safe = f => async args => { try { return ok(await f(args)); } catch (e) { return fail(e.message); } };

// Build an MCP server with the read tools, plus any extra tools the host provides (refresh, draw, status).
export function buildMcp({ extra = [] } = {}) {
  const server = new McpServer({ name: 'idx-risk-radar', version: '0.2.0' }, {
    instructions: 'Data-only risk screen for 48 liquid IDX stocks built from TradingView daily bars. Not investment advice. ' +
      'Start with radar_market (market conditions, regime and how far to trust the models), then radar_table, radar_stock, or radar_assess for the integrated risk verdict and mitigation actions; ' +
      'radar_portfolio for a set of holdings; radar_validation for out-of-sample evidence. ' +
      'Always tell the user that the return ranking has low predictive reliability when reliable_return_edge is false. Risk forecasts are separate; check risk_reliable before leaning on them.',
  });
  server.registerTool('radar_market', { title: 'Market weather and model reliability',
    description: 'IHSG caution level, high-volatility regime probability, breadth, the shortlist, the suggested split, model reliability flags, and when the data was last updated.', inputSchema: {} }, safe(market));
  server.registerTool('radar_table', { title: 'Trade parameter table',
    description: 'Entry, break-even (BEP), stop loss, take profits, integrated risk verdict, 10%-drop chance, size factor, noise-aware stop and realistic target for stocks, filtered and sorted.',
    inputSchema: {
      label: z.enum(['all', 'shortlist', 'watch', 'avoid']).default('all').describe('Filter by label'),
      sector: z.string().optional().describe('Exact sector name, e.g. "Banks"'),
      max_risk: z.enum(['A', 'B', 'C', 'D', 'E']).default('E').describe('Highest risk grade to include'),
      sort: z.enum(['score', 'risk', 'liquidity']).default('score'),
      limit: z.number().int().min(1).max(48).default(48),
    } }, safe(table));
  server.registerTool('radar_stock', { title: 'One stock in detail',
    description: 'Full record for one IDX ticker: plan levels, BEP, risk metrics, why it got its label, each model\'s rank, support/resistance, and the integrated risk assessment.',
    inputSchema: { ticker: z.string().describe('IDX ticker, e.g. BBRI') } }, safe(({ ticker }) => stock(ticker)));
  server.registerTool('radar_assess', { title: 'Integrated risk assessment for one stock',
    description: 'Integrated risk verdict (Low to Severe), score, mitigation actions, Monte Carlo 20-day outcome range, GARCH volatility, EVT tail risk, stress tests in rupiah per lot, liquidity, and what drives the risk.',
    inputSchema: { ticker: z.string().describe('IDX ticker, e.g. BBRI') } }, safe(({ ticker }) => {
      const R = loadRisk(), s = R.stocks.find(x => x.tk === normTicker(ticker));
      if (!s) throw new Error(`${ticker} has no risk assessment.`);
      return { as_of: R.as_of, regime: { p_high_vol: R.regime.p_high_vol, high: R.regime.high }, ...s };
    }));
  server.registerTool('radar_portfolio', { title: 'Portfolio risk',
    description: 'Risk of a set of holdings: yearly volatility, 1-day and 20-day VaR/ES in rupiah, diversification benefit, each stock\'s share of total risk, stress-test losses (IHSG -10%/-20%, worst-month replay), and integrated verdicts.',
    inputSchema: { holdings: z.array(z.object({ ticker: z.string(), amount: z.number().positive().describe('Rupiah amount') })).min(1) } },
    safe(({ holdings }) => portfolioRisk(loadRisk(), holdings)));
  server.registerTool('radar_validation', { title: 'How well the models performed on unseen data',
    description: 'Walk-forward validation of every forecast: return ranking IC, risk forecasts AUC/Brier/calibration, VaR Kupiec test, deep network train vs test AUC.', inputSchema: {} }, safe(validation));
  for (const t of extra) server.registerTool(t.name, t.config, safe(t.handler));
  return server;
}
