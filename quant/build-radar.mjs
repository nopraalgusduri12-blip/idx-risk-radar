// Fill the radar template with model output. Usage: node quant/build-radar.mjs [drawnTickers,comma,separated]
import { readFileSync, writeFileSync } from 'fs';
const dir = new URL('.', import.meta.url);
const D = JSON.parse(readFileSync(new URL('data/model.json', dir), 'utf8'));
// Tickers whose levels are drawn in TradingView (draw.mjs draws all by default).
D.drawn = process.argv[2] ? process.argv[2].split(',').filter(Boolean) : D.stocks.map(s => s.tk);
const html = readFileSync(new URL('radar.template.html', dir), 'utf8').replace('__DATA__', JSON.stringify(D));
writeFileSync(new URL('idx-risk-radar.html', dir), html);
console.log('idx-risk-radar.html', html.length, 'bytes; drawn:', D.drawn.join(' '));
