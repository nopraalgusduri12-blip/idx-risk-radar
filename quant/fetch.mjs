// Pull daily OHLCV for a universe of liquid IDX stocks (plus the IHSG index) from the live
// TradingView chart, one symbol at a time, then restore the chart's original symbol/timeframe.
// Usage: node quant/fetch.mjs
import { writeFileSync, mkdirSync } from 'fs';
import { chart, data, ui } from 'tradingview-mcp/core';

// The chart loads ~300 daily bars by default. Ask the series for more history, chunk by chunk,
// until it reaches TARGET_BARS or stops growing, then read every loaded bar.
const TARGET_BARS = +(process.env.RADAR_BARS || 1500);
const SERIES = 'window.TradingViewApi._activeChartWidgetWV.value()._chartWidget.model().mainSeries()';
const evalPage = async expression => (await ui.uiEvaluate({ expression })).result;
async function loadHistory() {
  let size = await evalPage(`${SERIES}.bars().size()`), stall = 0;
  while (size < TARGET_BARS && stall < 3) {
    await evalPage(`(function(){ try { ${SERIES}.requestMoreData(300); } catch (e) {} return 1; })()`);
    let grown = false;
    for (let k = 0; k < 12 && !grown; k++) { await sleep(500); const s = await evalPage(`${SERIES}.bars().size()`); if (s > size) { size = s; grown = true; } }
    stall = grown ? 0 : stall + 1;
  }
  return evalPage(`(function(){ var b = ${SERIES}.bars(), out = [];
    for (var i = b.firstIndex(); i <= b.lastIndex(); i++) { var v = b.valueAt(i); if (v) out.push([v[0], v[1], v[2], v[3], v[4], v[5] || 0]); }
    return out; })()`);
}

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
    const all = await loadHistory();
    out[sym] = all.map(([time, open, high, low, close, volume]) => ({ time, open, high, low, close, volume }));
    console.log(sym.padEnd(16), out[sym].length, 'bars from', new Date(all[0][0] * 1000).toISOString().slice(0, 10), 'last close', all.at(-1)[4]);
  } catch (e) {
    failed.push(sym); console.log(sym.padEnd(16), 'FAILED', e.message);
  }
}
writeFileSync('quant/data/daily.json', JSON.stringify({ fetched_at: new Date().toISOString(), index: INDEX, bars: out, failed }));

await chart.setSymbol({ symbol: origSymbol });
await chart.setTimeframe({ timeframe: origTf });
console.log('restored chart to', origSymbol, origTf, '| ok', Object.keys(out).length, 'failed', failed.length);
process.exit(0);
