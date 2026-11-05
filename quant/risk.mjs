// Integrated risk engine: quantitative finance models + machine learning, validated walk-forward,
// compiled into a per-stock risk assessment and mitigation plan and a portfolio risk dataset.
//
// Components (all fitted only on data before each test block):
//   GARCH(1,1)        conditional volatility and its 20-day term structure
//   FHS Monte Carlo   filtered historical simulation: GARCH volatility path x bootstrapped real
//                     (close, low, high) shocks -> P(stop), P(10% drop), P(target before stop), 20-day ranges
//   EVT               generalised Pareto tail of standardised losses -> conditional 1-day VaR / ES 99%
//   HMM               2-state regime model on IHSG -> probability of the high-volatility regime
//   Liquidity         Corwin-Schultz spread, traded value
//   Logistic baseline small volatility/market-stress model (the winner in model.mjs)
//   Deep network      3 hidden layers with dropout on the last 20 days of normalised price/market moves
//   Meta-model        stacked logistic regression on all component outputs -> integrated risk probability
//
// Usage: node quant/risk.mjs [quant/data/daily.json] [quant/data/model.json] [--out quant/data/risk.json]
import { readFileSync, writeFileSync } from 'fs';
import * as Q from './lib.mjs';
const { mean, sd, quantile, rankPct, r, day, auc, brier, logit, sigmoid, snap } = Q;

const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
const SRC = JSON.parse(readFileSync(args[0] || 'quant/data/daily.json', 'utf8'));
const MODEL = JSON.parse(readFileSync(args[1] || 'quant/data/model.json', 'utf8'));
const H = 20, STOP_ATR = 1.5, CRASH = Math.log(0.9);
const START = 500, FOLD = 120, EVAL_STEP = 5, WIN = 1000;
const PATHS_VAL = 300, PATHS_FINAL = 4000;
const log = s => process.stderr.write(s + '\n');

const { T, N, IX, tickers, S } = Q.alignData(SRC);
const ixr = IX.map((x, i) => i ? Math.log(x / IX[i - 1]) : 0);
const ewma = (ret, from) => { const out = new Float64Array(N).fill(NaN); if (from + 31 >= N) return out; let v = sd(ret.slice(from + 1, from + 30)) ** 2;
  for (let i = from + 30; i < N; i++) { v = 0.94 * v + 0.06 * ret[i] ** 2; out[i] = Math.sqrt(v); } return out; };
const mktEw = ewma(ixr, 0);
const mktDD = IX.map((x, i) => x / Math.max(...IX.slice(Math.max(0, i - 249), i + 1)) - 1);

// ---------- per-stock series ----------
const D = {};
for (const tk of tickers) {
  const { h, l, c, first: f } = S[tk];
  const ret = c.map((x, i) => i > f ? Math.log(x / c[i - 1]) : 0);
  const lz = c.map((x, i) => i > f ? Math.log(l[i] / c[i - 1]) : 0);
  const hz = c.map((x, i) => i > f ? Math.log(h[i] / c[i - 1]) : 0);
  const tr = c.map((x, i) => i > f ? Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1])) : 0);
  const atr = new Float64Array(N).fill(NaN);
  if (f >= 0 && f + 16 < N) { let a = mean(tr.slice(f + 1, f + 15)); atr[f + 14] = a; for (let i = f + 15; i < N; i++) { a = (a * 13 + tr[i]) / 14; atr[i] = a; } }
  D[tk] = { ret, lz, hz, atr, ew: ewma(ret, f), first: f };
}
const usable = (tk, i) => { const d = D[tk]; return d.first >= 0 && i - d.first >= 150 && isFinite(d.ew[i]) && isFinite(d.atr[i]) && S[tk].c[i] != null; };

function labels(tk, i) {
  const { h, l, c } = S[tk], e = c[i], stop = e - STOP_ATR * D[tk].atr[i];
  let minL = Infinity; for (let j = i + 1; j <= i + H; j++) minL = Math.min(minL, l[j]);
  return { crash: Math.log(minL / e) <= CRASH ? 1 : 0, stop: minL <= stop ? 1 : 0,
    vol: Math.log(sd(D[tk].ret.slice(i + 1, i + H + 1)) + 1e-5), ret20: Math.log(c[i + H] / e) };
}
function downBeta(tk, i, n = 250) {
  const from = Math.max(D[tk].first + 1, i - n + 1), xs = [], ys = [];
  for (let t = from; t <= i; t++) if (ixr[t] < 0) { xs.push(ixr[t]); ys.push(D[tk].ret[t]); }
  if (xs.length < 30) return 1;
  const mx = mean(xs), my = mean(ys); let num = 0, den = 0; xs.forEach((x, k) => { num += (x - mx) * (ys[k] - my); den += (x - mx) ** 2; });
  return num / (den || 1e-12);
}
function betaIdio(tk, i, n = 250) {
  const from = Math.max(D[tk].first + 1, i - n + 1), xs = ixr.slice(from, i + 1), ys = D[tk].ret.slice(from, i + 1);
  const mx = mean(xs), my = mean(ys); let num = 0, den = 0; xs.forEach((x, k) => { num += (x - mx) * (ys[k] - my); den += (x - mx) ** 2; });
  const b = num / (den || 1e-12); return { beta: b, idio: sd(ys.map((y, k) => y - b * xs[k])) };
}
// Inputs for the logistic baseline and the meta-model's market context.
const staticX = (tk, i) => [Math.log(D[tk].ew[i]), Math.log(D[tk].atr[i] / S[tk].c[i]), Math.log(mktEw[i] || 1e-3), mktDD[i]];
// Sequence input for the deep network: last 20 days of close/low/high moves (scaled by the stock's
// current EWMA volatility) and IHSG moves (scaled by IHSG volatility), plus the static inputs.
function seqX(tk, i) {
  const d = D[tk], s = d.ew[i], sm = mktEw[i] || 1e-3, x = [];
  for (let k = i - 19; k <= i; k++) x.push(d.ret[k] / s, d.lz[k] / s, d.hz[k] / s, ixr[k] / sm);
  const c = S[tk].c, hi60 = Math.max(...S[tk].h.slice(i - 59, i + 1));
  return [...x, ...staticX(tk, i), c[i] / hi60 - 1, Math.log(c[i] / c[i - 60])];
}

// ---------- GARCH fit + standardised shock pool + EVT tail for one stock up to `end` ----------
function fitStock(tk, end) {
  const d = D[tk], from = Math.max(d.first + 1, end - WIN);
  if (end - from < 250) return null;
  const ret = d.ret.slice(from, end + 1), g = Q.garchFit(ret);
  // Filter through to the end of the data with fixed parameters (only uses returns up to each day).
  const all = d.ret.slice(from), { sig2 } = Q.garchFilter(all.map(x => x - g.mu), g);
  const pool = [];
  for (let k = 0; k < ret.length; k++) { const s = Math.sqrt(sig2[k]); pool.push([(ret[k] - g.mu) / s, d.lz[from + k] / s, d.hz[from + k] / s]); }
  const evt = Q.gpdFit(pool.map(z => -z[0]));
  const sig2At = i => { const k = i - from; return k + 1 < sig2.length ? sig2[k + 1] : g.omega + g.alpha * (d.ret[i] - g.mu) ** 2 + g.beta * sig2[k]; }; // variance for day i+1
  return { g, from, pool, evt, sig2At, sig2Of: t => sig2[t - from] };
}

// ---------- filtered historical simulation over the next 20 days ----------
function fhs(fit, s2next, entry, stop, tp, paths) {
  const { g, pool } = fit, ls = Math.log(stop / entry), lt = Math.log(tp / entry), P = pool.length;
  let hitStop = 0, crash = 0, tpFirst = 0; const end = new Float64Array(paths), minL = new Float64Array(paths), maxH = new Float64Array(paths);
  for (let p = 0; p < paths; p++) {
    let s2 = s2next, lp = 0, mn = 0, mx = 0, decided = false;
    for (let dd = 0; dd < H; dd++) {
      const z = pool[Math.floor(Q.rand() * P)], s = Math.sqrt(s2), lo = lp + s * z[1], hi = lp + s * z[2];
      if (!decided) { if (lo <= ls) { hitStop++; decided = true; } else if (hi >= lt) { tpFirst++; decided = true; } }
      else if (lo <= ls && mn > ls) hitStop++; // stop touched after the target (still counts as touched)
      mn = Math.min(mn, lo); mx = Math.max(mx, hi);
      const e = s * z[0]; lp += g.mu + e; s2 = g.omega + g.alpha * e * e + g.beta * s2;
    }
    if (mn <= CRASH) crash++; end[p] = lp; minL[p] = mn; maxH[p] = mx;
  }
  const q = (a, x) => quantile(a, x);
  return { p_stop: Math.min(1, hitStop / paths), p_crash: crash / paths, p_tp_first: tpFirst / paths,
    ret_q: [0.05, 0.25, 0.5, 0.75, 0.95].map(x => q(end, x)), es95: mean(Array.from(end).filter(v => v <= q(end, 0.05))),
    minlow_q35: q(minL, 0.35), minlow_q10: q(minL, 0.10), minlow_med: q(minL, 0.5), maxhigh_med: q(maxH, 0.5), maxhigh_q75: q(maxH, 0.75) };
}
const evtRisk = (fit, s2next, qq) => { if (!fit.evt) return null; const z = Q.gpdRisk(fit.evt, qq), s = Math.sqrt(s2next); return { var: s * z.var - fit.g.mu, es: s * z.es - fit.g.mu }; };

// ---------- training samples for the learned components ----------
function trainSet(end) {
  const rows = [];
  for (let i = 160; i + H < end; i += 5) for (const tk of tickers) if (usable(tk, i)) rows.push({ tk, i, y: labels(tk, i).crash });
  return rows;
}
function fitBaseline(rows) {
  const X = rows.map(rw => staticX(rw.tk, rw.i)), st = Q.standardiser(X), y = rows.map(rw => rw.y);
  const m = Q.logistic(X.map(st), y, 0.001, 200, 0.5);
  return (tk, i) => m.predict(st(staticX(tk, i)));
}
function fitDeep(rows, seedv) {
  const dates = [...new Set(rows.map(rw => rw.i))].sort((a, b) => a - b), cut = dates[Math.floor(dates.length * 0.85)];
  const tr = rows.filter(rw => rw.i < cut - H), va = rows.filter(rw => rw.i >= cut);
  const Xtr = tr.map(rw => seqX(rw.tk, rw.i)), st = Q.standardiser(Xtr);
  Q.reseed(seedv);
  const net = Q.deepNet(Xtr.map(st), tr.map(rw => rw.y), { val: { X: va.map(rw => st(seqX(rw.tk, rw.i))), y: va.map(rw => rw.y) } });
  const trAuc = auc(tr.filter((_, k) => k % 4 === 0).map(rw => net.predict(st(seqX(rw.tk, rw.i)))), tr.filter((_, k) => k % 4 === 0).map(rw => rw.y));
  return { predict: (tk, i) => net.predict(st(seqX(tk, i))), epochs: net.epochs, train_auc: trAuc };
}

// ---------- walk-forward evaluation ----------
const rows = [], kup = { n: 0, x: 0 }, kupStocks = {}, deepLog = [], kupRatio = [];
let fold = 0, deep = null;
for (let start = START; start < N - H; start += FOLD, fold++) {
  const cut = start - H, end = Math.min(start + FOLD, N - H);
  const fits = {}; for (const tk of tickers) { const ft = fitStock(tk, cut); if (ft) fits[tk] = ft; }
  const hmm = Q.hmmFit(ixr.slice(1, cut + 1)), regime = Q.hmmFilter(hmm, ixr.slice(1));
  const tset = trainSet(cut), base = fitBaseline(tset);
  if (fold % 2 === 0) { deep = fitDeep(tset, 100 + fold); deepLog.push({ fold, rows: tset.length, epochs: deep.epochs, train_auc: r(deep.train_auc, 3) }); }
  Q.reseed(1000 + fold);
  for (const tk of tickers) { const ft = fits[tk]; if (!ft || !ft.evt) continue; kupStocks[tk] ||= { n: 0, x: 0 };
    for (let t = start; t < end; t++) { if (!usable(tk, t - 1)) continue; const v = evtRisk(ft, ft.sig2Of(t), 0.99); if (!v) continue;
      kup.n++; kupStocks[tk].n++; kupRatio.push([fold, -D[tk].ret[t] / v.var]); if (D[tk].ret[t] < -v.var) { kup.x++; kupStocks[tk].x++; } } }
  for (let i = start; i < end; i += EVAL_STEP) for (const tk of tickers) {
    const ft = fits[tk]; if (!ft || !usable(tk, i)) continue;
    const e = S[tk].c[i], atr = D[tk].atr[i], stop = e - STOP_ATR * atr, tp = e + 1.5 * STOP_ATR * atr, s2 = ft.sig2At(i);
    const sim = fhs(ft, s2, e, stop, tp, PATHS_VAL), ev = evtRisk(ft, s2, 0.99) || { es: NaN }, lb = labels(tk, i);
    rows.push({ fold, tk, i, ...lb,
      fhs_crash: sim.p_crash, fhs_stop: sim.p_stop, base_crash: base(tk, i), deep_crash: deep.predict(tk, i),
      es99: ev.es, spread: Q.corwinSchultz(S[tk].h, S[tk].l, i - 20, i), regime: regime[i - 1], dbeta: downBeta(tk, i),
      garch_vol20: Math.log(Math.sqrt(Q.garchTerm(ft.g, s2, H) / H)), ewma_vol: Math.log(D[tk].ew[i]) });
  }
  log(`fold ${fold}: days ${start}-${end - 1}, ${Object.keys(fits).length} stocks, ${rows.length} rows so far`);
}

// ---------- meta-model (stacking) ----------
const metaX = rw => [logit(rw.fhs_crash * 0.98 + 0.01), logit(rw.base_crash), logit(rw.deep_crash), Math.log(Math.max(1e-4, rw.es99 || 0.05)),
  Math.log(rw.spread + 1e-4), rw.regime, rw.dbeta];
const metaXstop = rw => [logit(rw.fhs_stop * 0.98 + 0.01), logit(rw.base_crash), logit(rw.deep_crash), Math.log(Math.max(1e-4, rw.es99 || 0.05)), Math.log(rw.spread + 1e-4), rw.regime, rw.dbeta];
const META_FROM = 2;
for (const [target, fx, key] of [['crash', metaX, 'meta_crash'], ['stop', metaXstop, 'meta_stop']]) {
  for (let k = META_FROM; k < fold; k++) {
    const tr = rows.filter(rw => rw.fold < k && isFinite(rw.es99)), te = rows.filter(rw => rw.fold === k);
    const X = tr.map(fx), st = Q.standardiser(X), m = Q.logistic(X.map(st), tr.map(rw => rw[target]), 0.01, 300, 0.5);
    te.forEach(rw => { rw[key] = m.predict(st(fx({ ...rw, es99: isFinite(rw.es99) ? rw.es99 : 0.05 }))); });
  }
}

// ---------- validation summaries ----------
const ev = rows.filter(rw => rw.fold >= META_FROM && rw.meta_crash != null);
const yC = ev.map(rw => rw.crash), yS = ev.map(rw => rw.stop);
const comp = (key, y) => { const p = ev.map(rw => rw[key]); return { auc: r(auc(p, y), 3), brier: r(brier(p, y), 4) }; };
const constC = ev.map(() => mean(rows.filter(rw => rw.fold < META_FROM).map(rw => rw.crash)));
const constS = ev.map(() => mean(rows.filter(rw => rw.fold < META_FROM).map(rw => rw.stop)));
const deepTest = auc(rows.map(rw => rw.deep_crash), rows.map(rw => rw.crash));
// Volatility forecasts: correlation and raw R^2 of the log forecast against log realised 20-day volatility.
const volR2 = key => { const p = rows.map(rw => rw[key]), y = rows.map(rw => rw.vol), my = mean(y);
  return { corr: r(Q.pearson(p, y), 3), r2: r(1 - mean(p.map((v, k) => (v - y[k]) ** 2)) / mean(y.map(v => (v - my) ** 2)), 3) }; };
// VaR backtest correction: scale factor so the 99% limit is broken on 1% of days. Estimated on the first
// half of the test blocks and checked on the second half, then re-estimated on all blocks for use.
const halfF = Math.floor(fold / 2), scaleEarly = Math.max(1, quantile(kupRatio.filter(x => x[0] < halfF).map(x => x[1]), 0.99));
const lateR = kupRatio.filter(x => x[0] >= halfF).map(x => x[1]), varScale = Math.max(1, quantile(kupRatio.map(x => x[1]), 0.99));
const validation = {
  period: { from: day(T[START]), to: day(T[N - H - 1]), folds: fold, eval_rows: rows.length, meta_rows: ev.length },
  crash: { label: 'Drop of 10%+ within 20 days', base_rate: r(mean(yC), 3),
    components: { fhs: comp('fhs_crash', yC), baseline: comp('base_crash', yC), deep: comp('deep_crash', yC), es99_rank: { auc: r(auc(ev.map(rw => rw.es99 || 0), yC), 3) },
      regime_rank: { auc: r(auc(ev.map(rw => rw.regime), yC), 3) }, integrated: comp('meta_crash', yC) },
    brier_const: r(brier(constC, yC), 4), calibration_integrated: Q.calibration(ev.map(rw => rw.meta_crash), yC), calibration_fhs: Q.calibration(ev.map(rw => rw.fhs_crash), yC) },
  stop: { label: 'Stop loss (1.5 ATR) touched within 20 days', base_rate: r(mean(yS), 3),
    components: { fhs: comp('fhs_stop', yS), integrated: comp('meta_stop', yS) }, brier_const: r(brier(constS, yS), 4),
    calibration_fhs: Q.calibration(ev.map(rw => rw.fhs_stop), yS), calibration_integrated: Q.calibration(ev.map(rw => rw.meta_stop), yS) },
  volatility: { label: '20-day realised volatility (log)', garch: volR2('garch_vol20'), ewma: volR2('ewma_vol') },
  var99_1d: { label: 'GARCH-EVT 1-day VaR 99% (Kupiec test)', ...Q.kupiec(kup.x, kup.n, 0.01),
    scale_factor: r(varScale, 3), adjusted_check: { scale_from_first_half: r(scaleEarly, 3), ...Q.kupiec(lateR.filter(x => x > scaleEarly).length, lateR.length, 0.01) },
    stocks_rejected_5pct: Object.values(kupStocks).filter(s => s.n > 100 && Q.kupiec(s.x, s.n, 0.01).p_value < 0.05).length, stocks_tested: Object.values(kupStocks).filter(s => s.n > 100).length },
  deep_network: { architecture: `${seqX(tickers[0], N - 1).length}-32-16-8-1 ReLU, dropout 0.2, L2 1e-4, Adam, early stopping`, refits: deepLog,
    train_auc_avg: r(mean(deepLog.map(x => x.train_auc)), 3), test_auc: r(deepTest, 3) },
};
validation.crash.brier_skill_integrated = r(1 - validation.crash.components.integrated.brier / validation.crash.brier_const, 3);
validation.stop.brier_skill_integrated = r(1 - validation.stop.components.integrated.brier / validation.stop.brier_const, 3);
validation.crash.best = Object.entries(validation.crash.components).filter(([, v]) => v.brier != null).sort((a, b) => b[1].auc - a[1].auc)[0][0];

// ---------- final fit on all data ----------
log('final fit');
const LAST = N - 1;
const fitsF = {}; for (const tk of tickers) { const ft = fitStock(tk, LAST); if (ft) fitsF[tk] = ft; }
const hmmF = Q.hmmFit(ixr.slice(1)), regF = Q.hmmFilter(hmmF, ixr.slice(1)), regimeNow = regF[LAST - 1];
const tsetF = trainSet(N), baseF = fitBaseline(tsetF), deepF = fitDeep(tsetF, 777);
const metaFit = (fx, target) => { const tr = rows.filter(rw => isFinite(rw.es99)), X = tr.map(fx), st = Q.standardiser(X), m = Q.logistic(X.map(st), tr.map(rw => rw[target]), 0.01, 300, 0.5);
  return { predict: rw => m.predict(st(fx(rw))), w: m.w }; };
const metaC = metaFit(metaX, 'crash'), metaS = metaFit(metaXstop, 'stop');
const META_NAMES = ['Monte Carlo 10%-drop chance', 'Volatility baseline', 'Deep network', 'EVT expected shortfall 99%', 'Bid-ask spread', 'High-volatility regime', 'Downside beta'];

// Worst 20-day IHSG window in the data, for the historical stress test.
let worst = { i: 0, ret: 0 }; for (let i = 0; i + H < N; i++) { const rr = IX[i + H] / IX[i] - 1; if (rr < worst.ret) worst = { i, ret: rr }; }

const byTk = Object.fromEntries(MODEL.stocks.map(s => [s.tk, s]));
const stocks = [];
Q.reseed(4242);
for (const tk of tickers) {
  const ft = fitsF[tk], m = byTk[tk]; if (!ft || !m || !usable(tk, LAST)) continue;
  const e = S[tk].c[LAST], plan = m.plan, s2 = ft.sig2At(LAST);
  const sim = fhs(ft, s2, e, plan.stop, plan.tp1, PATHS_FINAL), ev1raw = evtRisk(ft, s2, 0.99), ev95 = evtRisk(ft, s2, 0.95);
  const ev1 = ev1raw && { var: ev1raw.var * varScale, es: ev1raw.es * varScale }; // backtest-corrected
  const g = ft.g, vol20 = Math.sqrt(Q.garchTerm(g, s2, H) * 252 / H), term = [1, 5, 10, 20].map(h => r(Math.sqrt(Q.garchTerm(g, s2, h) * 252 / h), 4));
  const rets = D[tk].ret.slice(Math.max(D[tk].first + 1, LAST - 249), LAST + 1), sk = mean(rets.map(x => ((x - mean(rets)) / sd(rets)) ** 3)), ku = mean(rets.map(x => ((x - mean(rets)) / sd(rets)) ** 4)) - 3;
  const cf99 = -(mean(rets) + sd(rets) * Q.cornishFisher(-2.326, sk, ku));
  const spread = Q.corwinSchultz(S[tk].h, S[tk].l, LAST - 20, LAST), bi = betaIdio(tk, LAST), db = downBeta(tk, LAST);
  const row = { fhs_crash: sim.p_crash, fhs_stop: sim.p_stop, base_crash: baseF(tk, LAST), deep_crash: deepF.predict(tk, LAST), es99: ev1 ? ev1.es : 0.05, spread, regime: regimeNow, dbeta: db };
  const hist = S[tk].c[worst.i] != null && S[tk].c[worst.i + H] != null ? S[tk].c[worst.i + H] / S[tk].c[worst.i] - 1 : null;
  const shock = s => ({ expected: Math.max(-0.95, db * s), bad: Math.max(-0.95, db * s - 1.645 * bi.idio * Math.sqrt(H)) });
  stocks.push({
    tk, price: e, plan,
    garch: { omega: g.omega, alpha: r(g.alpha, 4), beta: r(g.beta, 4), persistence: r(g.persistence, 4), half_life_days: r(Math.log(0.5) / Math.log(g.persistence), 1),
      vol_now_ann: r(Math.sqrt(s2 * 252), 4), vol_longrun_ann: r(Math.sqrt(g.uncond * 252), 4), vol20_ann: r(vol20, 4), term_ann: term },
    sim: { p_stop: r(sim.p_stop, 3), p_crash10: r(sim.p_crash, 3), p_tp1_first: r(sim.p_tp_first, 3), ret20_q: sim.ret_q.map(x => r(Math.exp(x) - 1, 4)),
      es95_20d: r(Math.exp(sim.es95) - 1, 4), typical_low: r(Math.exp(sim.minlow_med) - 1, 4), bad_low: r(Math.exp(sim.minlow_q10) - 1, 4),
      low_q35: r(Math.exp(sim.minlow_q35) - 1, 4), typical_high: r(Math.exp(sim.maxhigh_med) - 1, 4), high_q75: r(Math.exp(sim.maxhigh_q75) - 1, 4) },
    tail: { evt_var99_1d: ev1 ? r(1 - Math.exp(-ev1.var), 4) : null, evt_es99_1d: ev1 ? r(1 - Math.exp(-ev1.es), 4) : null, evt_var95_1d: ev95 ? r(1 - Math.exp(-ev95.var), 4) : null,
      cf_var99_1d: r(1 - Math.exp(-cf99), 4), xi: ft.evt ? r(ft.evt.xi, 3) : null, skew: r(sk, 2), exkurt: r(ku, 2) },
    liquidity: { spread_cs: r(spread, 5), adv_value_bn: m.liq_bn, adv_shares: Math.round(mean(S[tk].v.slice(LAST - 19, LAST + 1))) },
    market: { beta: r(bi.beta, 2), down_beta: r(db, 2), idio_ann: r(bi.idio * Math.sqrt(252), 4) },
    stress: { hist: hist == null ? null : r(hist, 4), ihsg10: { expected: r(shock(-0.10).expected, 4), bad: r(shock(-0.10).bad, 4) }, ihsg20: { expected: r(shock(-0.20).expected, 4), bad: r(shock(-0.20).bad, 4) } },
    components: { fhs_crash: r(row.fhs_crash, 3), baseline_crash: r(row.base_crash, 3), deep_crash: r(row.deep_crash, 3) },
    integrated: { p_crash10: r(metaC.predict(row), 3), p_stop: r(metaS.predict(row), 3) },
    _row: row,
  });
}

// Driver attribution: standardised input x meta coefficient, relative to the universe average.
{
  const X = stocks.map(s => metaX(s._row)), mu = X[0].map((_, j) => mean(X.map(x => x[j]))), sg = X[0].map((_, j) => sd(X.map(x => x[j])) || 1);
  const trX = rows.filter(rw => isFinite(rw.es99)).map(metaX), tsg = trX[0].map((_, j) => sd(trX.map(x => x[j])) || 1);
  stocks.forEach((s, k) => { s.drivers = META_NAMES.map((name, j) => ({ name, effect: r(metaC.w[j + 1] * (X[k][j] - mu[j]) / tsg[j], 3) }))
    .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).slice(0, 4); delete s._row; });
}

// ---------- assessment and mitigation plan ----------
const pc = rankPct(stocks.map(s => s.integrated.p_crash10));
const VERDICT = [['Low', 'Size normally'], ['Moderate', 'Size slightly smaller'], ['Elevated', 'Reduce size'], ['High', 'Small size only'], ['Severe', 'Avoid new positions']];
const SIZE = [1, 0.8, 0.6, 0.4, 0];
const regimeHigh = regimeNow >= 0.7;
stocks.forEach((s, k) => {
  const score = Math.round(pc[k] * 100), band = Math.min(4, Math.floor(pc[k] * 5)), m = byTk[s.tk], p = s.plan, e = s.price;
  // Noise-aware stop: the price with about a 35% simulated chance of being touched within 20 days.
  const noiseStop = Math.min(p.stop, snap(e * (1 + s.sim.low_q35), 'down'));
  const target = snap(e * (1 + s.sim.typical_high));
  const actions = [];
  let sizeF = SIZE[band] * (regimeHigh ? 0.5 : 1);
  actions.push({ kind: 'size', text: band === 4 ? 'Avoid opening a new position: integrated risk is in the top fifth of the list.' :
    `Use ${Math.round(sizeF * 100)}% of your normal position size${regimeHigh ? ' (halved because the market is in its high-volatility regime)' : ''}.` });
  if (s.sim.p_stop > 0.5) actions.push({ kind: 'stop', text: `The planned stop at Rp${p.stop.toLocaleString('id-ID')} has a ${Math.round(s.sim.p_stop * 100)}% simulated chance of being touched within 20 days. A noise-aware stop near Rp${noiseStop.toLocaleString('id-ID')} (about 35% chance) with fewer lots keeps the same rupiah risk.` });
  else actions.push({ kind: 'stop', text: `Keep the stop at Rp${p.stop.toLocaleString('id-ID')}: simulated chance of touching it within 20 days is ${Math.round(s.sim.p_stop * 100)}%.` });
  actions.push({ kind: 'target', text: s.sim.p_tp1_first < 0.3 ? `Take profit 1 (Rp${p.tp1.toLocaleString('id-ID')}) is ambitious: only ${Math.round(s.sim.p_tp1_first * 100)}% of simulations reach it before the stop. A typical 20-day high is near Rp${target.toLocaleString('id-ID')}; consider taking partial profit there.`
    : `Take profit 1 (Rp${p.tp1.toLocaleString('id-ID')}) is reached before the stop in ${Math.round(s.sim.p_tp1_first * 100)}% of simulations.` });
  if (s.liquidity.spread_cs > 0.01) actions.push({ kind: 'liquidity', text: `Estimated bid-ask spread is ${(s.liquidity.spread_cs * 100).toFixed(1)}%: use limit orders, and expect to lose some of that when exiting quickly.` });
  if (s.market.down_beta > 1.2) actions.push({ kind: 'market', text: `Falls about ${s.market.down_beta.toFixed(1)}× as much as IHSG on down days. Pair it with lower-beta stocks or keep more cash.` });
  if (s.tail.xi > 0.25) actions.push({ kind: 'tail', text: `Heavy left tail (EVT shape ${s.tail.xi}): single-day losses beyond ${(s.tail.evt_es99_1d * 100).toFixed(1)}% happen more often than a normal curve suggests. Avoid holding through known event dates with a full position.` });
  actions.push({ kind: 'monitor', text: `Reassess if volatility climbs past ${(s.garch.vol20_ann * 150).toFixed(0)}% a year, if price closes below ${m.levels.sup[0] ? 'support at Rp' + m.levels.sup[0].price.toLocaleString('id-ID') : 'the stop'}, or if IHSG's regime changes.` });
  const actionTxt = band === 4 ? VERDICT[4][1] : sizeF >= 1 ? VERDICT[0][1] : `Use ${Math.round(sizeF * 100)}% of normal size`;
  s.assessment = { score, verdict: VERDICT[band][0], action: actionTxt, size_factor: r(sizeF, 2), noise_stop: noiseStop, noise_stop_lot_risk: (e - noiseStop) * 100,
    realistic_target: target, lot_value: e * 100, es95_20d_lot: r(-s.sim.es95_20d * e * 100, 0), actions };
});

// ---------- portfolio dataset: Ledoit-Wolf shrunk covariance of daily returns (last 250 days) ----------
const PT = stocks.map(s => s.tk), W = 250;
const R = PT.map(tk => D[tk].ret.slice(LAST - W + 1, LAST + 1));
const mus = R.map(mean), Xc = R.map((a, k) => a.map(v => v - mus[k]));
const Sm = PT.map((_, a) => PT.map((_, b) => Xc[a].reduce((s0, v, t) => s0 + v * Xc[b][t], 0) / W));
const muI = mean(PT.map((_, a) => Sm[a][a]));
let num = 0, den = 0;
for (let t = 0; t < W; t++) for (let a = 0; a < PT.length; a++) for (let b = 0; b < PT.length; b++) num += (Xc[a][t] * Xc[b][t] - Sm[a][b]) ** 2;
num /= W * W; for (let a = 0; a < PT.length; a++) for (let b = 0; b < PT.length; b++) den += (Sm[a][b] - (a === b ? muI : 0)) ** 2;
const shrink = Math.min(1, num / (den || 1e-12));
const cov = PT.map((_, a) => PT.map((_, b) => r((1 - shrink) * Sm[a][b] + (a === b ? shrink * muI : 0), 10)));

const out = {
  version: 1, generated: new Date().toISOString(), as_of: day(T[LAST]), horizon: H,
  regime: { p_high_vol: r(regimeNow, 3), high: regimeHigh, vol_low_ann: r(hmmF.sg[0] * Math.sqrt(252), 3), vol_high_ann: r(hmmF.sg[1] * Math.sqrt(252), 3),
    stay_low: r(hmmF.A[0][0], 4), stay_high: r(hmmF.A[1][1], 4), expected_days_high: r(1 / (1 - hmmF.A[1][1]), 1), history: Array.from(regF.slice(-250)).map(v => r(v, 3)) },
  worst_window: { from: day(T[worst.i]), to: day(T[worst.i + H]), ihsg: r(worst.ret, 4) },
  validation, meta_weights: Object.fromEntries(META_NAMES.map((n, j) => [n, r(metaC.w[j + 1], 3)])),
  portfolio: { tickers: PT, cov_daily: cov, shrinkage: r(shrink, 3), window_days: W },
  stocks,
};
const outIdx = process.argv.indexOf('--out');
const txt = JSON.stringify(out);
outIdx > 0 ? writeFileSync(process.argv[outIdx + 1], txt) : console.log(txt);
log(`done: ${stocks.length} stocks`);
