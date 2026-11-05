// Draw the radar's plan levels (entry, break-even, stop, targets, swing levels) on each stock's chart,
// then restore the original symbol and timeframe. Safe to rerun: earlier radar lines on a symbol
// are removed first, so each chart always shows the latest levels.
// Usage: node quant/draw.mjs [all | TICKER,TICKER,...]   (defaults to all stocks in the model)
import { readFileSync } from 'fs';
import { chart, drawing } from 'tradingview-mcp/core';

const M = JSON.parse(readFileSync('quant/data/model.json', 'utf8'));
const RAW = JSON.parse(readFileSync('quant/data/daily.json', 'utf8'));
const arg = process.argv[2] || 'all';
const list = arg === 'all' ? M.stocks.map(s => s.tk) : arg.split(',').filter(Boolean);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const C = { entry: '#0d6b6e', bep: '#6b5bd2', stop: '#c0392b', tp: '#1a7f4e', lvl: '#7a8a86' };
const LABEL = { shortlist: 'SHORTLIST', watch: 'WATCH', avoid: 'AVOID' };
// Labels this tool has written (current format, plus the older manual-analysis format).
const OURS = /^(RADAR\b|Resistance \d+ \(\d+(×| touch)|Support \d+ \((\d+(×| touch)|52-week)|LONG |SHORT |Entry \d)/;

async function clearOurs() {
  const { shapes } = await drawing.listDrawings();
  let removed = 0;
  for (const sh of shapes.filter(x => x.name === 'horizontal_line')) {
    try {
      const p = await drawing.getProperties({ entity_id: sh.id });
      const text = p.properties?.text ?? '';
      if (OURS.test(text)) { await drawing.removeOne({ entity_id: sh.id }); removed++; }
    } catch { /* shape vanished or unreadable: leave it */ }
  }
  return removed;
}

const st = await chart.getState();
await chart.setTimeframe({ timeframe: 'D' });
for (const tk of list) {
  const s = M.stocks.find(x => x.tk === tk);
  if (!s) { console.log(tk.padEnd(5), 'not in model output'); continue; }
  try {
    const time = RAW.bars['IDX:' + tk].at(-1).time;
    await chart.setSymbol({ symbol: 'IDX:' + tk });
    await sleep(1200);
    const removed = await clearOurs();
    const p = s.plan, lines = [
      [p.entry, `RADAR ${tk} ${LABEL[s.label]} · entry ${p.entry} · score ${s.score} · risk ${s.risk_grade}`, C.entry, 2, 0, 'right'],
      [p.bep, `RADAR BEP ${p.bep} (after fees)`, C.bep, 1, 2, 'right'],
      [p.stop, `RADAR stop loss ${p.stop} (-${(p.risk_pct * 100).toFixed(1)}% · Rp${p.lot_risk.toLocaleString('id-ID')}/lot)`, C.stop, 2, 0, 'right'],
      [p.tp1, `RADAR take profit 1 ${p.tp1} (1.5R)`, C.tp, 2, 0, 'right'],
      [p.tp2, `RADAR take profit 2 ${p.tp2} (2.5R)`, C.tp, 1, 2, 'right'],
      ...s.levels.res.map(l => [l.price, `RADAR resistance ${l.price} (${l.touches}×)`, C.lvl, 1, 1, 'left']),
      ...s.levels.sup.map(l => [l.price, `RADAR support ${l.price} (${l.touches}×)`, C.lvl, 1, 1, 'left']),
    ];
    let ok = 0;
    for (const [price, text, color, width, style, align] of lines) {
      const r = await drawing.drawShape({ shape: 'horizontal_line', point: { time, price }, text,
        overrides: { linecolor: color, linewidth: width, linestyle: style, showLabel: true, textcolor: color, horzLabelsAlign: align, vertLabelsAlign: 'bottom', fontsize: 11 } });
      if (r.success) ok++;
    }
    console.log(tk.padEnd(5), `removed ${removed}, drew ${ok}/${lines.length}`);
  } catch (e) {
    console.log(tk.padEnd(5), 'FAILED', e.message);
  }
}
await chart.setSymbol({ symbol: st.symbol });
await chart.setTimeframe({ timeframe: st.resolution });
console.log('restored chart to', st.symbol, st.resolution);
process.exit(0);
