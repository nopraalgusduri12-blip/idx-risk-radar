// Day-trade planner: buy in the morning, sell in the afternoon of the same day.
//
// For every stock and trading day it asks: with what is known at the open (yesterday's bars and today's
// opening price), how did "buy at the open, sell by the close" work out after costs? Trade rule:
//   entry  = opening price
//   stop   = entry - 0.5 x ATR(14)      (if the day's low touches it, the trade loses there; when a day
//                                        touches both stop and target, the stop is assumed to come first)
//   target = entry + 0.75 x ATR(14)     (if the high reaches it, the trade wins there)
//   else   = sell at the close
//   costs  = buy fee + sell fee (default 0.15% + 0.25%, the sell side includes the 0.1% tax)
// Models (logistic, ridge, deep network) rank stocks each morning; everything is validated walk-forward,
// and the strategy "buy the top 3 each morning" is compared with buying every liquid stock.
//
// Usage: node quant/daytrade.mjs [quant/data/daily.json] [quant/data/risk.json] [--out quant/data/daytrade.json]
import { readFileSync, writeFileSync, existsSync } from 'fs';
import * as Q from './lib.mjs';
const { mean, sd, quantile, rankPct, r, day, snap, logit } = Q;

const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
const SRC = JSON.parse(readFileSync(args[0] || 'quant/data/daily.json', 'utf8'));
const RISK = existsSync(args[1] || 'quant/data/risk.json') ? JSON.parse(readFileSync(args[1] || 'quant/data/risk.json', 'utf8')) : null;
const FEES = { buy: 0.0015, sell: 0.0025 }, COST = FEES.buy + FEES.sell;
const STOP_ATR = 0.5, TARGET_ATR = 0.75, TOP_K = 3, LIQ_MIN_BN = 10;
const START = 500, FOLD = 60;
const log = s => process.stderr.write(s + '\n');

const { T, N: N0, IX, tickers, S } = Q.alignData(SRC);
const ixBars = SRC.bars[SRC.index], IXO = ixBars.map(b => b.open);
// A bar dated today (Jakarta time) before 16:15 WIB is still forming: its open is known, its close is not.
const WIB = 7 * 3600, dayNo = t => Math.floor((t + WIB) / 86400);
const nowSec = Date.now() / 1000, nowWIBh = ((nowSec + WIB) % 86400) / 3600;
const LIVE = dayNo(T[N0 - 1]) === dayNo(nowSec) && nowWIBh < 16.25;
const N = N0; // index N-1 is the plan day when LIVE, otherwise the last completed day

// ---------- per-stock series ----------
const ixr = IX.map((x, i) => i ? Math.log(x / IX[i - 1]) : 0);
const ew = (ret, from) => { const out = new Float64Array(N).fill(NaN); if (from + 31 >= N) return out; let v = sd(ret.slice(from + 1, from + 30)) ** 2;
  for (let i = from + 30; i < N; i++) { v = 0.94 * v + 0.06 * ret[i] ** 2; out[i] = Math.sqrt(v); } return out; };
const ixEw = ew(ixr, 0);
const D = {};
for (const tk of tickers) {
  const { o, h, l, c, v, first: f } = S[tk];
  const ret = c.map((x, i) => i > f ? Math.log(x / c[i - 1]) : 0);
  const tr = c.map((x, i) => i > f ? Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1])) : 0);
  const atr = new Float64Array(N).fill(NaN), rsi = new Float64Array(N).fill(NaN);
  if (f >= 0 && f + 20 < N) {
    let a = mean(tr.slice(f + 1, f + 15)); atr[f + 14] = a; for (let i = f + 15; i < N; i++) { a = (a * 13 + tr[i]) / 14; atr[i] = a; }
    let ag = 0, al = 0; for (let i = f + 1; i <= f + 14; i++) { const d = c[i] - c[i - 1]; ag += Math.max(d, 0) / 14; al += Math.max(-d, 0) / 14; }
    for (let i = f + 15; i < N; i++) { const d = c[i] - c[i - 1]; ag = (ag * 13 + Math.max(d, 0)) / 14; al = (al * 13 + Math.max(-d, 0)) / 14; rsi[i] = 100 - 100 / (1 + ag / (al || 1e-9)); }
  }
  D[tk] = { ret, atr, rsi, ew: ew(ret, f), tv: c.map((x, i) => x == null ? 0 : x * v[i]), first: f };
}
let regime = new Float64Array(N).fill(0.5);
try { const h = Q.hmmFit(ixr.slice(1, N - 1)); const f = Q.hmmFilter(h, ixr.slice(1)); for (let i = 1; i < N; i++) regime[i] = f[i - 1]; } catch { /* keep neutral */ }

// ---------- features known at the open of day t ----------
const FEAT = ['Opening gap (in daily swings)', 'Opening gap (%)', 'IHSG opening gap', 'Yesterday\'s return', '5-day return', '20-day return', 'Where yesterday closed in its range',
  'Yesterday\'s range', 'Yesterday open-to-close', 'Yesterday\'s volume vs 20-day average', 'RSI 14', 'Distance from 20-day average', 'Volatility (log)', 'ATR % (log)',
  'IHSG yesterday', 'IHSG volatility (log)', 'High-volatility regime', 'Traded value (log)', 'Monday', 'Friday'];
// `pre` describes a day that has not opened yet: { y: last completed index, open, ixOpen, time }.
function feat(tk, t, pre) {
  const { o, h, l, c, v } = S[tk], d = D[tk], y = pre ? pre.y : t - 1, open = pre ? pre.open : o[t], ixOpen = pre ? pre.ixOpen : IXO[t], time = pre ? pre.time : T[t];
  if (d.first < 0 || y - d.first < 60 || !isFinite(d.ew[y]) || !isFinite(d.atr[y]) || !(open > 0) || !(c[y] > 0)) return null;
  const s = d.ew[y], gap = Math.log(open / c[y]), rng = h[y] - l[y];
  const tv20 = mean(d.tv.slice(y - 19, y + 1)), v20 = mean(v.slice(y - 19, y + 1));
  const wd = new Date((time + WIB) * 1000).getUTCDay();
  return [gap / s, gap, Math.log(ixOpen / IX[y]), d.ret[y] / s, Math.log(c[y] / c[y - 5]) / (s * Math.sqrt(5)), Math.log(c[y] / c[y - 20]) / (s * Math.sqrt(20)),
    rng > 0 ? (c[y] - l[y]) / rng : 0.5, Math.log(h[y] / l[y]) / s, Math.log(c[y] / o[y]) / s, v20 ? v[y] / v20 : 1, d.rsi[y] / 100,
    c[y] / mean(c.slice(y - 19, y + 1)) - 1, Math.log(s), Math.log(d.atr[y] / c[y]), ixr[y] / (ixEw[y] || 0.01), Math.log(ixEw[y] || 0.01), regime[y],
    Math.log(tv20 + 1), wd === 1 ? 1 : 0, wd === 5 ? 1 : 0];
}
// Outcome of the trade rule on day t (log returns, before costs).
function outcome(tk, t) {
  const { o, h, l, c } = S[tk], a = D[tk].atr[t - 1], e = o[t];
  const stop = e - STOP_ATR * a, tgt = e + TARGET_ATR * a;
  if (l[t] <= stop) return { plan: Math.log(stop / e), exit: 'stop' };
  if (h[t] >= tgt) return { plan: Math.log(tgt / e), exit: 'target' };
  return { plan: Math.log(c[t] / e), exit: 'close' };
}
const liquidAt = (tk, t) => mean(D[tk].tv.slice(t - 20, t)) / 1e9 >= LIQ_MIN_BN;

// ---------- panel ----------
const rows = [];
const lastTrain = LIVE ? N - 1 : N; // a forming bar has no outcome yet
for (let t = 80; t < lastTrain; t++) for (const tk of tickers) {
  if (S[tk].c[t] == null || S[tk].v[t] === 0) continue; // suspended or forward-filled day
  const x = feat(tk, t); if (!x) continue;
  const oc = outcome(tk, t), net = Math.exp(oc.plan) * (1 - FEES.sell) / (1 + FEES.buy) - 1, closeOnly = S[tk].c[t] / S[tk].o[t] * (1 - FEES.sell) / (1 + FEES.buy) - 1;
  rows.push({ tk, t, x, net, gross: Math.exp(oc.plan) - 1, closeOnly, exit: oc.exit, win: net > 0 ? 1 : 0, liquid: liquidAt(tk, t) });
}
log(`panel: ${rows.length} stock-days, live=${LIVE}`);

// ---------- learners ----------
function ridge(X, y, lam) {
  const P = X[0].length, A = Array.from({ length: P + 1 }, () => new Float64Array(P + 1)), b = new Float64Array(P + 1);
  for (let n = 0; n < X.length; n++) { const x = [1, ...X[n]]; for (let j = 0; j <= P; j++) { b[j] += x[j] * y[n]; for (let k = 0; k <= P; k++) A[j][k] += x[j] * x[k]; } }
  for (let j = 1; j <= P; j++) A[j][j] += lam;
  for (let c = 0; c <= P; c++) { let p = c; for (let q = c + 1; q <= P; q++) if (Math.abs(A[q][c]) > Math.abs(A[p][c])) p = q; [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let q = 0; q <= P; q++) if (q !== c) { const f = A[q][c] / A[c][c]; for (let k = c; k <= P; k++) A[q][k] -= f * A[c][k]; b[q] -= f * b[c]; } }
  const w = Array.from(b, (x, j) => x / A[j][j]); return { w, predict: x => w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0) };
}
// Daily cross-sectional rank IC of predictions against realised net returns.
function dailyIC(rs, p) { const by = new Map(); rs.forEach((rw, k) => { (by.get(rw.t) || by.set(rw.t, []).get(rw.t)).push(k); });
  return mean([...by.values()].filter(ks => ks.length >= 8).map(ks => Q.pearson(rankPct(ks.map(k => p[k])), rankPct(ks.map(k => rs[k].net))))); }
function fitAll(train, seedv) {
  const days = [...new Set(train.map(rw => rw.t))].sort((a, b) => a - b), cut = days[Math.floor(days.length * 0.85)];
  const inner = train.filter(rw => rw.t < cut - 1), val = train.filter(rw => rw.t >= cut);
  const st = Q.standardiser(inner.map(rw => rw.x)), Xi = inner.map(rw => st(rw.x)), Xv = val.map(rw => st(rw.x));
  const mods = {};
  let best = null; for (const lam of [10, 100, 1000, 10000]) { const m = ridge(Xi, inner.map(rw => rw.net), lam); const ic = dailyIC(val, Xv.map(m.predict)); if (!best || ic > best.ic) best = { lam, ic }; }
  const rm = ridge(Xi, inner.map(rw => rw.net), best.lam); mods.ridge = { predict: x => rm.predict(st(x)), val: best.ic, lam: best.lam };
  const lg = Q.logistic(Xi, inner.map(rw => rw.win), 0.01, 150, 0.5); mods.logistic = { predict: x => lg.predict(st(x)), val: dailyIC(val, Xv.map(x => lg.predict(x))) };
  Q.reseed(seedv);
  const dn = Q.deepNet(Xi, inner.map(rw => rw.win), { val: { X: Xv, y: val.map(rw => rw.win) }, hidden: [32, 16], maxEpochs: 12, patience: 2, batch: 256 });
  mods.deep = { predict: x => dn.predict(st(x)), val: dailyIC(val, Xv.map(x => dn.predict(x))) };
  const raw = Object.fromEntries(Object.entries(mods).map(([k, m]) => [k, Math.max(0, m.val)])), tot = Object.values(raw).reduce((a, b) => a + b, 0);
  const weights = Object.fromEntries(Object.keys(mods).map(k => [k, tot > 0 ? raw[k] / tot : 1 / 3]));
  // P(win) for the plan: calibrated on validation with Platt scaling of the logistic output.
  const pm = Q.logistic(Xv.map(x => [logit(lg.predict(x))]), val.map(rw => rw.win), 0, 200, 0.5);
  const pwin = x => pm.predict([logit(lg.predict(st(x)))]);
  // Ensemble score: weighted average of each model's within-day rank (computed by the caller per day).
  return { mods, weights, pwin, trainIC: dailyIC(inner.filter((_, k) => k % 3 === 0), inner.filter((_, k) => k % 3 === 0).map(rw => rm.predict(st(rw.x)))) };
}
function scoreDay(fit, rs) {
  const out = new Array(rs.length).fill(0);
  for (const [k, w] of Object.entries(fit.weights)) { if (!w) continue; const rk = rankPct(rs.map(rw => fit.mods[k].predict(rw.x))); rk.forEach((v, j) => { out[j] += w * v; }); }
  return out;
}

// ---------- walk-forward backtest ----------
const byDay = new Map(); rows.forEach(rw => { (byDay.get(rw.t) || byDay.set(rw.t, []).get(rw.t)).push(rw); });
const allDays = [...byDay.keys()].sort((a, b) => a - b);
const daily = [], tested = [], foldLog = [];
let fold = 0;
for (let s = START; s < allDays.length; s += FOLD, fold++) {
  const startT = allDays[s], endT = allDays[Math.min(allDays.length - 1, s + FOLD - 1)];
  const fit = fitAll(rows.filter(rw => rw.t < startT - 1), 900 + fold);
  const ics = [];
  for (const t of allDays.filter(t => t >= startT && t <= endT)) {
    const rs = byDay.get(t), sc = scoreDay(fit, rs);
    rs.forEach((rw, j) => { rw.score = sc[j]; rw.pwin = fit.pwin(rw.x); rw.pred = fit.mods.ridge.predict(rw.x); rw.fold = fold; tested.push(rw); });
    ics.push(Q.pearson(rankPct(sc), rankPct(rs.map(rw => rw.net))));
    const liquid = rs.filter(rw => rw.liquid);
    if (liquid.length < TOP_K) continue;
    const top = [...liquid].sort((a, b) => b.score - a.score).slice(0, TOP_K);
    // Selective: only stocks whose expected return after fees (ridge) is positive; otherwise stay in cash (0).
    const sel = liquid.filter(rw => rw.pred > 0).sort((a, b) => b.score - a.score).slice(0, TOP_K);
    daily.push({ t, top: mean(top.map(rw => rw.net)), topGross: mean(top.map(rw => rw.gross)), all: mean(liquid.map(rw => rw.net)), allClose: mean(liquid.map(rw => rw.closeOnly)),
      bottom: mean([...liquid].sort((a, b) => a.score - b.score).slice(0, TOP_K).map(rw => rw.net)), regime: regime[t - 1],
      sel: sel.length ? mean(sel.map(rw => rw.net)) : 0, selN: sel.length, selNets: sel.map(rw => rw.net) });
  }
  foldLog.push({ fold, from: day(T[startT]), to: day(T[endT]), test_ic: r(mean(ics), 4), train_ic: r(fit.trainIC, 4), weights: Object.fromEntries(Object.entries(fit.weights).map(([k, v]) => [k, r(v, 2)])) });
  log(`fold ${fold}: ${day(T[startT])}..${day(T[endT])} ic ${mean(ics).toFixed(4)}`);
}

// ---------- strategy statistics ----------
function stats(series) {
  const m = mean(series), s = sd(series), n = series.length;
  let eq = 1, pk = 1, mdd = 0; for (const x of series) { eq *= 1 + x; pk = Math.max(pk, eq); mdd = Math.min(mdd, eq / pk - 1); }
  return { days: n, avg_per_day: r(m, 5), win_rate: r(mean(series.map(x => x > 0 ? 1 : 0)), 3), t_stat: r(m / (s / Math.sqrt(n)), 2),
    sharpe_ann: r(m / s * Math.sqrt(250), 2), total_return: r(eq - 1, 4), max_drawdown: r(mdd, 4) };
}
const strategy = {
  top3: stats(daily.map(d => d.top)), top3_before_costs: stats(daily.map(d => d.topGross)),
  all_liquid: stats(daily.map(d => d.all)), all_liquid_hold_to_close: stats(daily.map(d => d.allClose)), bottom3: stats(daily.map(d => d.bottom)),
  selective: stats(daily.map(d => d.sel)),
  selective_trade_days: { days_traded: daily.filter(d => d.selN).length, share_of_days: r(mean(daily.map(d => d.selN ? 1 : 0)), 3), trades: daily.reduce((a, d) => a + d.selN, 0),
    avg_per_trade: r(mean(daily.flatMap(d => d.selNets)), 5), win_rate_per_trade: r(mean(daily.flatMap(d => d.selNets).map(x => x > 0 ? 1 : 0)), 3) },
  top3_calm_regime: stats(daily.filter(d => d.regime < 0.5).map(d => d.top)), top3_volatile_regime: stats(daily.filter(d => d.regime >= 0.5).map(d => d.top)),
};
const icAll = mean(foldLog.map(f => f.test_ic)), icT = icAll / (sd(foldLog.map(f => f.test_ic)) / Math.sqrt(foldLog.length));
const edge = strategy.top3.t_stat >= 2 && strategy.top3.avg_per_day > 0;
const edgeSelective = strategy.selective.t_stat >= 2 && strategy.selective.avg_per_day > 0;
// Morning rule from the opening gap (in units of the stock's daily volatility), on tested rows of liquid stocks.
const GAPS = [[-Infinity, -1, 'Opens well below yesterday (more than 1 daily swing down)'], [-1, -0.25, 'Opens a little lower'], [-0.25, 0.25, 'Opens near yesterday\'s close'],
  [0.25, 1, 'Opens a little higher'], [1, Infinity, 'Opens well above yesterday (more than 1 daily swing up)']];
const liqTested = tested.filter(rw => rw.liquid);
const gapTable = GAPS.map(([lo, hi, label]) => { const g = liqTested.filter(rw => rw.x[0] >= lo && rw.x[0] < hi);
  return { label, lo, hi, n: g.length, avg_net: r(mean(g.map(rw => rw.net)), 5), win_rate: r(mean(g.map(rw => rw.win)), 3),
    hit_stop: r(mean(g.map(rw => rw.exit === 'stop' ? 1 : 0)), 3), hit_target: r(mean(g.map(rw => rw.exit === 'target' ? 1 : 0)), 3) }; });
const exits = { stop: r(mean(liqTested.map(rw => rw.exit === 'stop' ? 1 : 0)), 3), target: r(mean(liqTested.map(rw => rw.exit === 'target' ? 1 : 0)), 3), close: r(mean(liqTested.map(rw => rw.exit === 'close' ? 1 : 0)), 3) };
const calib = Q.calibration(liqTested.map(rw => rw.pwin), liqTested.map(rw => rw.win), [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 1.01]);
const bestGaps = gapTable.filter(g => g.avg_net >= Math.max(...gapTable.map(x => x.avg_net)) - 0.0005).map(g => g.label);

// ---------- plan for the next session (or today, when the open is already known) ----------
log('final fit');
const fitF = fitAll(rows, 4242);
const planT = N - 1, yIdx = LIVE ? N - 2 : N - 1;
const riskBy = RISK ? Object.fromEntries(RISK.stocks.map(s => [s.tk, s])) : {};
const planRows = [];
for (const tk of tickers) {
  // Before the open, assume a flat open (gap 0) and give the morning rule; once the open is known, use it.
  const c = S[tk].c, y = yIdx; if (!(c[y] > 0)) continue;
  // Next session: next weekday after the last bar (holidays are not known here).
  const nextT = (() => { let t = T[y] + 86400; while ([0, 6].includes(new Date((t + WIB) * 1000).getUTCDay())) t += 86400; return t; })();
  const x = LIVE ? feat(tk, planT) : feat(tk, null, { y, open: c[y], ixOpen: IX[y], time: nextT });
  if (!x) continue;
  const ref = LIVE ? S[tk].o[planT] : c[y], a = D[tk].atr[y];
  const k = riskBy[tk];
  planRows.push({ tk, ref, x, atr: a, vol: D[tk].ew[y], liquid: mean(D[tk].tv.slice(y - 19, y + 1)) / 1e9 >= LIQ_MIN_BN, tv_bn: r(mean(D[tk].tv.slice(y - 19, y + 1)) / 1e9, 1),
    verdict: k?.assessment.verdict ?? null, pwin: fitF.pwin(x), pred: fitF.mods.ridge.predict(x) });
}
const sc = scoreDay(fitF, planRows);
planRows.forEach((p, j) => { p.score = Math.round(sc[j] * 100); });
const ranked = [...planRows].sort((a, b) => b.score - a.score);
const plan = ranked.map((p, rank) => {
  const stop = snap(p.ref - STOP_ATR * p.atr, 'down'), tgt = snap(p.ref + TARGET_ATR * p.atr, 'up'), bep = snap(p.ref * (1 + FEES.buy) / (1 - FEES.sell), 'up');
  const skip = [];
  if (!p.liquid) skip.push(`thin trading (Rp${p.tv_bn} bn a day)`);
  if (p.verdict === 'Severe') skip.push('integrated risk is Severe');
  if (tgt <= bep) skip.push('target does not clear fees');
  if (!(p.pred > 0)) skip.push('expected result after fees is negative');
  const eligible = !skip.length;
  return { tk: p.tk, rank: rank + 1, score: p.score, ref_price: p.ref, entry_rule: LIVE ? `Opened at Rp${p.ref.toLocaleString('id-ID')}` : `Buy only if it opens near Rp${p.ref.toLocaleString('id-ID')} (see morning rule)`,
    atr: r(p.atr, 2), vol_daily: r(p.vol, 5), stop, target: tgt, bep, stop_pct: r(stop / p.ref - 1, 4), target_pct: r(tgt / p.ref - 1, 4), lot_risk: Math.round((p.ref - stop) * 100),
    p_win: r(p.pwin, 3), expected_net: r(p.pred, 5), verdict: p.verdict, liquid: p.liquid, traded_bn: p.tv_bn, eligible, skip };
});
const picks = plan.filter(p => p.eligible).slice(0, TOP_K).map(p => p.tk);

const SESSION = {
  mon_thu: [['09:00', 'Market opens. Wait for the opening auction price and the first minutes to settle.'], ['09:05–09:30', 'Check the morning rule against the opening price, then buy the picks that pass.'],
    ['After buying', 'Enter the stop and target as orders (or alerts) right away.'], ['12:00–13:30', 'Lunch break: no trading. Orders stay in the book.'],
    ['15:30–15:49', 'Sell any position still open, or use the closing auction (15:50–16:00).']],
  friday: [['09:00–11:30', 'Session 1 (shorter on Friday).'], ['11:30–14:00', 'Friday break.'], ['14:00–15:49', 'Session 2; sell by 15:49 or in the closing auction (15:50–16:00).']],
};

const out = {
  version: 1, generated: new Date().toISOString(), live: LIVE, plan_for: LIVE ? day(T[planT]) : 'next trading day', based_on: day(T[yIdx]),
  rules: { stop_atr: STOP_ATR, target_atr: TARGET_ATR, top_k: TOP_K, fees: FEES, round_trip_cost: COST, liquidity_min_bn: LIQ_MIN_BN },
  validation: { from: foldLog[0]?.from, to: foldLog.at(-1)?.to, folds: fold, test_days: daily.length, ranking_ic: r(icAll, 4), ranking_ic_t: r(icT, 2),
    train_ic_avg: r(mean(foldLog.map(f => f.train_ic)), 4), edge_after_costs: edge, edge_selective: edgeSelective, strategy, exits, calibration_pwin: calib, folds_detail: foldLog },
  morning_rule: { table: gapTable, best: bestGaps },
  features: FEAT, picks, plan, session: SESSION, regime_now: r(regime[N - 1], 3),
};
const oi = process.argv.indexOf('--out'), txt = JSON.stringify(out);
oi > 0 ? writeFileSync(process.argv[oi + 1], txt) : console.log(txt);
log(`done: picks ${picks.join(' ')} | top3 avg/day ${strategy.top3.avg_per_day} t ${strategy.top3.t_stat} | all ${strategy.all_liquid.avg_per_day}`);
