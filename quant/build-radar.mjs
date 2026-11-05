// Fill the radar template with model output and, when present, the integrated risk assessment.
// Usage: node quant/build-radar.mjs [drawnTickers,comma,separated]
import { readFileSync, writeFileSync, existsSync } from 'fs';
const dir = new URL('.', import.meta.url);
const D = JSON.parse(readFileSync(new URL('data/model.json', dir), 'utf8'));
const riskPath = new URL('data/risk.json', dir);
if (existsSync(riskPath)) {
  const R = JSON.parse(readFileSync(riskPath, 'utf8'));
  const byTk = Object.fromEntries(R.stocks.map(s => [s.tk, s]));
  for (const s of D.stocks) {
    const k = byTk[s.tk]; if (!k) continue;
    s.risk = k;
    // The integrated assessment can only make a label more cautious, never less.
    if (k.assessment.verdict === 'Severe' && s.label !== 'avoid') { s.label = 'avoid'; s.why.push('integrated risk assessment is Severe (top fifth of the list)'); }
    else if (k.assessment.verdict === 'High' && s.label === 'shortlist') { s.label = 'watch'; s.why.push('integrated risk assessment is High, so it drops from the shortlist'); }
  }
  D.shortlist = D.stocks.filter(s => s.label === 'shortlist').map(s => s.tk);
  const { stocks, ...rest } = R;
  D.riskmeta = rest;
}
// Tickers whose levels are drawn in TradingView (draw.mjs draws all by default).
D.drawn = process.argv[2] ? process.argv[2].split(',').filter(Boolean) : D.stocks.map(s => s.tk);
const html = readFileSync(new URL('radar.template.html', dir), 'utf8').replace('__DATA__', JSON.stringify(D));
writeFileSync(new URL('idx-risk-radar.html', dir), html);
console.log('idx-risk-radar.html', html.length, 'bytes; shortlist:', D.shortlist.join(' ') || '(none)');
