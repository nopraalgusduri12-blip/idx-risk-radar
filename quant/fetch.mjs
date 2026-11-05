// Pull daily OHLCV for a universe of liquid IDX stocks (plus the IHSG index) from the live
// TradingView chart, one symbol at a time, then restore the chart's original symbol/timeframe.
// Usage: node quant/fetch.mjs
import { writeFileSync, mkdirSync } from 'fs';
import { chart, data } from 'tradingview-mcp/core';

const UNIVERSE = [
  'BBCA', 'BBRI', 'BMRI', 'BBNI', 'BRIS', 'BBTN', 'ARTO', 'TLKM', 'ISAT', 'EXCL', 'TOWR', 'MTEL',
  'ASII', 'UNTR', 'UNVR', 'ICBP', 'INDF', 'MYOR', 'KLBF', 'SIDO', 'CPIN', 'JPFA', 'AMRT', 'MAPI',
  'ACES', 'ERAA', 'ADRO', 'PTBA', 'ITMG', 'HRUM', 'MEDC', 'PGAS', 'AKRA', 'ANTM', 'INCO', 'MDKA',
  'AMMN', 'BRPT', 'TPIA', 'INKP', 'TKIM', 'SMGR', 'INTP', 'CTRA', 'BSDE', 'PWON', 'GOTO', 'ESSA',
];
const INDEX = 'IDX:COMPOSITE';
const sleep = ms => new Promise(r => setTimeout(r, ms));

const state = await chart.getState();
const origSymbol = state.symbol, origTf = state.resolution;
await chart.setTimeframe({ timeframe: 'D' });
await sleep(1500);

mkdirSync('quant/data', { recursive: true });
const out = {}, failed = [];
let prevSig = '';
for (const sym of [INDEX, ...UNIVERSE.map(s => 'IDX:' + s)]) {
  try {
    await chart.setSymbol({ symbol: sym });
    let res = null;
    // Wait until the bar series actually changed from the previous symbol.
    for (let k = 0; k < 10; k++) {
      await sleep(700);
      try { res = await data.getOhlcv({ count: 500 }); } catch { res = null; }
      const sig = res ? `${res.bars.length}:${res.bars.at(-1).close}:${res.bars[0].close}` : '';
      if (res && res.bars.length > 50 && sig !== prevSig) { prevSig = sig; break; }
      res = null;
    }
    if (!res) throw new Error('no fresh bars');
    out[sym] = res.bars;
    console.log(sym.padEnd(16), res.bars.length, 'bars, last close', res.bars.at(-1).close);
  } catch (e) {
    failed.push(sym); console.log(sym.padEnd(16), 'FAILED', e.message);
  }
}
writeFileSync('quant/data/daily.json', JSON.stringify({ fetched_at: new Date().toISOString(), index: INDEX, bars: out, failed }));

await chart.setSymbol({ symbol: origSymbol });
await chart.setTimeframe({ timeframe: origTf });
console.log('restored chart to', origSymbol, origTf, '| ok', Object.keys(out).length, 'failed', failed.length);
process.exit(0);
