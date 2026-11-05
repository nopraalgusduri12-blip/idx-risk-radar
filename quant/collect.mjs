// Collector for the PC that runs TradingView Desktop: fetch fresh data, compute, upload to the web backend.
//
//   node quant/collect.mjs                  fetch -> compute models here -> upload results (default; fastest)
//   node quant/collect.mjs --server-compute fetch -> upload raw bars; the server retrains (slower on small hosts)
//   node quant/collect.mjs --no-fetch       skip the TradingView fetch and upload what is already in quant/data
//   node quant/collect.mjs --draw           also redraw the levels on the TradingView charts
//   node quant/collect.mjs --morning        quick morning run for day trading: update recent bars (incl. today's
//                                           open), rebuild the day-trade plan, upload (~3 minutes; run after 09:05 WIB)
//
// Environment: RADAR_SERVER (e.g. https://radar.example.com), RADAR_INGEST_TOKEN or RADAR_TOKEN.
import { readFileSync, existsSync } from 'fs';
import { gzipSync } from 'zlib';
import { spawnSync } from 'child_process';
import { hostname } from 'os';

const has = f => process.argv.includes(f);
const SERVER = (process.env.RADAR_SERVER || '').replace(/\/+$/, '');
const TOKEN = process.env.RADAR_INGEST_TOKEN || process.env.RADAR_TOKEN;
if (!SERVER || !TOKEN) { console.error('Set RADAR_SERVER and RADAR_INGEST_TOKEN (or RADAR_TOKEN). See DEPLOY.md.'); process.exit(1); }

const step = (label, args) => {
  console.log(`\n== ${label}`);
  const r = spawnSync(process.execPath, args, { stdio: 'inherit' });
  if (r.status !== 0) { console.error(`${label} failed (exit ${r.status}).`); process.exit(1); }
};
async function post(path, body) {
  const data = gzipSync(Buffer.from(JSON.stringify(body)));
  const res = await fetch(SERVER + path, { method: 'POST', body: data,
    headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' } });
  const text = await res.text();
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}: ${text.slice(0, 300)}`);
  return JSON.parse(text);
}

const MORNING = has('--morning');
if (!has('--no-fetch')) step(MORNING ? 'Updating recent bars from TradingView' : 'Fetching daily bars from TradingView', MORNING ? ['quant/fetch.mjs', '--update'] : ['quant/fetch.mjs']);
const daily = JSON.parse(readFileSync('quant/data/daily.json', 'utf8'));
const source = `collector@${hostname()}`;

if (has('--server-compute')) {
  console.log(`\n== Uploading raw bars (${Object.keys(daily.bars).length} symbols) to ${SERVER}`);
  console.log(await post('/api/ingest', { ...daily, source }));
} else {
  if (!MORNING && (!has('--no-fetch') || !existsSync('quant/data/model.json'))) {
    step('Training and validating models', ['quant/model.mjs', 'quant/data/daily.json', '--out', 'quant/data/model.json']);
    step('Running the integrated risk engine', ['quant/risk.mjs', 'quant/data/daily.json', 'quant/data/model.json', '--out', 'quant/data/risk.json']);
  }
  const model = JSON.parse(readFileSync('quant/data/model.json', 'utf8'));
  const risk = existsSync('quant/data/risk.json') ? JSON.parse(readFileSync('quant/data/risk.json', 'utf8')) : null;
  if (!risk) { console.error('quant/data/risk.json is missing; run `npm run risk` first.'); process.exit(1); }
  step('Building the day-trade plan', ['quant/daytrade.mjs', 'quant/data/daily.json', 'quant/data/risk.json', '--out', 'quant/data/daytrade.json']);
  const daytrade = JSON.parse(readFileSync('quant/data/daytrade.json', 'utf8'));
  if (has('--draw')) step('Drawing levels on TradingView charts', ['quant/draw.mjs', 'all']);
  console.log(`\n== Publishing results (as of ${model.as_of}) to ${SERVER}`);
  console.log(await post('/api/publish', { model, risk, daytrade, source, drawn: has('--draw') ? model.stocks.map(s => s.tk) : [] }));
}
step('Rebuilding the local dashboard file', ['quant/build-radar.mjs']);
console.log('\nDone. Open the server dashboard from any device.');
