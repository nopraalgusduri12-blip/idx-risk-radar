// Build the static dashboard file from model output and, when present, the integrated risk assessment.
// Usage: node quant/build-radar.mjs [drawnTickers,comma,separated]
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { renderDashboard } from './dashboard.mjs';

const dir = new URL('.', import.meta.url);
const model = JSON.parse(readFileSync(new URL('data/model.json', dir), 'utf8'));
const riskPath = new URL('data/risk.json', dir);
const risk = existsSync(riskPath) ? JSON.parse(readFileSync(riskPath, 'utf8')) : null;
// Levels are drawn in TradingView for every stock by draw.mjs, unless a list is given.
const drawn = process.argv[2] ? process.argv[2].split(',').filter(Boolean) : model.stocks.map(s => s.tk);
const { html, data } = renderDashboard(model, risk, { drawn });
writeFileSync(new URL('idx-risk-radar.html', dir), html);
console.log('idx-risk-radar.html', html.length, 'bytes; shortlist:', data.shortlist.join(' ') || '(none)');
