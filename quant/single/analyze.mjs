// Statistical / risk analysis of daily OHLCV pulled from the live TradingView chart.
// Usage: node analysis/analyze.mjs analysis/ohlcv_D.json [SYMBOL] > analysis/result.json
import { readFileSync } from 'fs';

const src = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const bars = src.bars;
const C = bars.map(b => b.close), H = bars.map(b => b.high), L = bars.map(b => b.low), V = bars.map(b => b.volume);
const n = bars.length, last = bars[n - 1], px = last.close;

const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const quantile = (a, q) => { const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * q; const lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); };
const sma = (a, p, i = a.length - 1) => i + 1 < p ? null : mean(a.slice(i + 1 - p, i + 1));
const r2 = x => x == null ? null : Math.round(x * 100) / 100;
const r4 = x => x == null ? null : Math.round(x * 10000) / 10000;

// Returns
const ret = []; for (let i = 1; i < n; i++) ret.push(Math.log(C[i] / C[i - 1]));
const mu = mean(ret), sig = sd(ret);
const skew = mean(ret.map(r => ((r - mu) / sig) ** 3));
const kurt = mean(ret.map(r => ((r - mu) / sig) ** 4)) - 3;
const simple = ret.map(r => Math.exp(r) - 1);
const var95 = -quantile(simple, 0.05), var99 = -quantile(simple, 0.01);
const tail95 = simple.filter(r => r <= -var95), cvar95 = -mean(tail95);
const upDays = simple.filter(r => r > 0).length / simple.length;
const sig20 = sd(ret.slice(-20)), sig60 = sd(ret.slice(-60));

// Drawdown
let peak = C[0], mdd = 0, mddPeak = 0, mddTrough = 0, pk = 0;
for (let i = 0; i < n; i++) { if (C[i] > peak) { peak = C[i]; pk = i; } const dd = C[i] / peak - 1; if (dd < mdd) { mdd = dd; mddPeak = pk; mddTrough = i; } }
const curDD = px / Math.max(...C) - 1;

// ATR(14), Wilder
const tr = bars.map((b, i) => i === 0 ? b.high - b.low : Math.max(b.high - b.low, Math.abs(b.high - C[i - 1]), Math.abs(b.low - C[i - 1])));
let atr = mean(tr.slice(1, 15)); for (let i = 15; i < n; i++) atr = (atr * 13 + tr[i]) / 14;

// RSI(14), Wilder
let ag = 0, al = 0; for (let i = 1; i <= 14; i++) { const d = C[i] - C[i - 1]; ag += Math.max(d, 0); al += Math.max(-d, 0); } ag /= 14; al /= 14;
for (let i = 15; i < n; i++) { const d = C[i] - C[i - 1]; ag = (ag * 13 + Math.max(d, 0)) / 14; al = (al * 13 + Math.max(-d, 0)) / 14; }
const rsi = 100 - 100 / (1 + ag / al);

// Moving averages, Bollinger
const sma20 = sma(C, 20), sma50 = sma(C, 50), sma200 = sma(C, 200);
const bbSd = sd(C.slice(-20)), bbU = sma20 + 2 * bbSd, bbL = sma20 - 2 * bbSd;
const ema = (a, p) => { const k = 2 / (p + 1); let e = mean(a.slice(0, p)); for (let i = p; i < a.length; i++) e = a[i] * k + e * (1 - k); return e; };
const macd = ema(C, 12) - ema(C, 26);

// Swing pivots (5 bars each side) -> support/resistance clusters
const piv = [];
for (let i = 5; i < n - 5; i++) {
  const wH = H.slice(i - 5, i + 6), wL = L.slice(i - 5, i + 6);
  if (H[i] === Math.max(...wH)) piv.push({ price: H[i], kind: 'high', t: bars[i].time });
  if (L[i] === Math.min(...wL)) piv.push({ price: L[i], kind: 'low', t: bars[i].time });
}
const tol = atr * 0.5, clusters = [];
for (const p of [...piv].sort((a, b) => a.price - b.price)) {
  const c = clusters.find(c => Math.abs(c.price - p.price) <= tol);
  if (c) { c.members.push(p); c.price = mean(c.members.map(m => m.price)); } else clusters.push({ price: p.price, members: [p] });
}
const levels = clusters.map(c => ({ price: r2(c.price), touches: c.members.length, last: Math.max(...c.members.map(m => m.t)) }));
const resist = levels.filter(l => l.price > px * 1.003).sort((a, b) => a.price - b.price).slice(0, 4);
const support = levels.filter(l => l.price < px * 0.997).sort((a, b) => b.price - a.price).slice(0, 4);
const hi52 = Math.max(...H.slice(-252)), lo52 = Math.min(...L.slice(-252));

// Empirical TP-before-SL test: enter at each historical close, ATR-scaled stop and target,
// walk forward up to `horizon` bars; if a bar touches both, count as a loss (conservative).
function bracket(side, slMult, rr, horizon = 20) {
  let win = 0, loss = 0, open = 0, bars2 = [];
  let a = mean(tr.slice(1, 15));
  for (let i = 15; i < n - 1; i++) {
    a = (a * 13 + tr[i]) / 14;
    const e = C[i], risk = slMult * a;
    const sl = side === 'long' ? e - risk : e + risk, tp = side === 'long' ? e + rr * risk : e - rr * risk;
    let res = null;
    for (let j = i + 1; j < Math.min(n, i + 1 + horizon); j++) {
      const hitSL = side === 'long' ? L[j] <= sl : H[j] >= sl;
      const hitTP = side === 'long' ? H[j] >= tp : L[j] <= tp;
      if (hitSL) { res = 'loss'; bars2.push(j - i); break; }
      if (hitTP) { res = 'win'; bars2.push(j - i); break; }
    }
    if (res === 'win') win++; else if (res === 'loss') loss++; else open++;
  }
  const decided = win + loss, p = decided ? win / decided : 0;
  const expR = p * rr - (1 - p);
  const kelly = p - (1 - p) / rr;
  return { side, sl_atr: slMult, rr, horizon, trials: win + loss + open, wins: win, losses: loss, unresolved: open,
    win_rate: r4(p), breakeven_win_rate: r4(1 / (1 + rr)), expectancy_R: r4(expR), kelly: r4(kelly), median_bars: bars2.length ? quantile(bars2, 0.5) : null };
}
const grid = [];
for (const side of ['long', 'short']) for (const sl of [1, 1.5, 2]) for (const rr of [1, 1.5, 2, 3]) grid.push(bracket(side, sl, rr));

// Concrete reference brackets from the current close
const plan = side => {
  const s = side === 'long' ? 1 : -1;
  const stop = px - s * 1.5 * atr, risk = Math.abs(px - stop);
  return { side, entry: r2(px), stop: r2(stop), risk_per_share: r2(risk), risk_pct: r4(risk / px),
    tp1: r2(px + s * 1.5 * risk), tp2: r2(px + s * 2 * risk), tp3: r2(px + s * 3 * risk) };
};

const trend = sma50 && sma200 ? (px > sma50 && sma50 > sma200 ? 'up' : px < sma50 && sma50 < sma200 ? 'down' : 'mixed') : 'n/a';

console.log(JSON.stringify({
  symbol: process.argv[3] || 'BATS:NVDA', timeframe: 'D', bars: n, from: bars[0].time, to: last.time,
  last: { close: px, open: last.open, high: last.high, low: last.low, volume: last.volume },
  returns: { mean_daily: r4(Math.exp(mu) - 1), sd_daily: r4(sig), ann_vol: r4(sig * Math.sqrt(252)), vol_20d_ann: r4(sig20 * Math.sqrt(252)), vol_60d_ann: r4(sig60 * Math.sqrt(252)),
    ann_return: r4(Math.exp(mu * 252) - 1), sharpe_rf0: r2(mu / sig * Math.sqrt(252)), skew: r2(skew), excess_kurtosis: r2(kurt), up_day_pct: r4(upDays),
    var95_1d: r4(var95), var99_1d: r4(var99), cvar95_1d: r4(cvar95), best_day: r4(Math.max(...simple)), worst_day: r4(Math.min(...simple)) },
  drawdown: { max: r4(mdd), peak_time: bars[mddPeak].time, trough_time: bars[mddTrough].time, current_from_high: r4(curDD) },
  indicators: { atr14: r2(atr), atr_pct: r4(atr / px), rsi14: r2(rsi), sma20: r2(sma20), sma50: r2(sma50), sma200: r2(sma200),
    bb_upper: r2(bbU), bb_lower: r2(bbL), bb_pctb: r4((px - bbL) / (bbU - bbL)), macd: r2(macd), trend,
    vol_vs_20d: r4(last.volume / mean(V.slice(-21, -1))) },
  levels: { resistance: resist, support, hi_52w: hi52, lo_52w: lo52 },
  plans: [plan('long'), plan('short')],
  close_series: bars.map(b => [b.time, b.close]),
  grid,
}, null, 1));
