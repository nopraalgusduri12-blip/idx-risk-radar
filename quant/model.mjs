// IDX Risk Radar model, v2.
//
//   Return ranking  : which stocks may beat the others over the next 20 days (rank target).
//   Risk forecasts  : for mitigation. Per stock, over the next 20 trading days:
//                       stop     P(price touches the plan's stop loss)          classification
//                       win      P(take profit 1 is reached before the stop)   classification
//                       vol      realised volatility                           regression (log)
//                       mae      worst dip below today's close                 regression
//
// Anti-overfitting:
//   * ~6 years of history, training rows thinned to every 5th day (overlapping 20-day labels
//     are near-duplicates; thinning stops the models from memorising them)
//   * nested walk-forward: outer folds test on unseen future blocks; inside each fold the last
//     15% of the training window (after a 20-day purge) picks hyperparameters and early-stops
//     the trees and the neural net
//   * L2 penalties everywhere, minimum leaf sizes, two-seed bagging for the neural net
//   * ensemble weights come from validation skill inside each fold, so reported test scores
//     are not tuned on the test data
//   * train-vs-test gap is reported for every target
//
// Usage: node quant/model.mjs quant/data/daily.json [--out quant/data/model.json]
import { readFileSync, writeFileSync } from 'fs';

const SRC = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const H = 20;              // forecast horizon (trading days)
const LOOK = 100;          // longest feature lookback
const THIN = 5;            // use every 5th day for training rows
const FOLD = 60;           // outer test block length (days)
const MIN_TRAIN_DAYS = 400;
const STOP_ATR = 1.5, TP_R = 1.5;
const FEES = { buy: 0.0015, sell: 0.0025 }; // typical IDX broker fees; sell side includes the 0.1% final tax
const SECTORS = {
  BBCA: 'Banks', BBRI: 'Banks', BMRI: 'Banks', BBNI: 'Banks', BRIS: 'Banks', BBTN: 'Banks', ARTO: 'Banks',
  TLKM: 'Telco & towers', ISAT: 'Telco & towers', EXCL: 'Telco & towers', TOWR: 'Telco & towers', MTEL: 'Telco & towers',
  ASII: 'Industrials', UNTR: 'Industrials',
  UNVR: 'Consumer staples', ICBP: 'Consumer staples', INDF: 'Consumer staples', MYOR: 'Consumer staples',
  KLBF: 'Healthcare', SIDO: 'Healthcare', CPIN: 'Poultry & feed', JPFA: 'Poultry & feed',
  AMRT: 'Retail', MAPI: 'Retail', ACES: 'Retail', ERAA: 'Retail',
  ADRO: 'Coal & energy', PTBA: 'Coal & energy', ITMG: 'Coal & energy', HRUM: 'Coal & energy', MEDC: 'Oil & gas', PGAS: 'Oil & gas', AKRA: 'Oil & gas',
  ANTM: 'Metals & mining', INCO: 'Metals & mining', MDKA: 'Metals & mining', AMMN: 'Metals & mining',
  BRPT: 'Chemicals', TPIA: 'Chemicals', ESSA: 'Chemicals', INKP: 'Pulp & paper', TKIM: 'Pulp & paper',
  SMGR: 'Cement', INTP: 'Cement', CTRA: 'Property', BSDE: 'Property', PWON: 'Property', GOTO: 'Technology',
};

// ---------- helpers ----------
let seed = 42;
const reseed = s => { seed = s; };
const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1)); };
const quantile = (a, q) => { const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + ((s[Math.ceil(i)] ?? s[lo]) - s[lo]) * (i - lo); };
const rankPct = a => { const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = new Array(a.length); idx.forEach(([, i], k) => { r[i] = a.length > 1 ? k / (a.length - 1) : 0.5; }); return r; };
const pearson = (a, b) => { const ma = mean(a), mb = mean(b); let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; } return n / Math.sqrt(da * db || 1e-12); };
const spearman = (a, b) => pearson(rankPct(a), rankPct(b));
const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
const logit = p => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
const r = (x, d = 4) => x == null || !isFinite(x) ? null : Math.round(x * 10 ** d) / 10 ** d;
const day = t => new Date(t * 1000).toISOString().slice(0, 10);
const idxTick = p => p < 200 ? 1 : p < 500 ? 2 : p < 2000 ? 5 : p < 5000 ? 10 : 25;
const snap = (p, mode) => { const t = idxTick(p); return (mode === 'down' ? Math.floor(p / t) : mode === 'up' ? Math.ceil(p / t) : Math.round(p / t)) * t; };
function auc(p, y) { const pos = y.filter(v => v === 1).length, neg = y.length - pos; if (!pos || !neg) return 0.5; const rk = rankPct(p).map(v => v * (y.length - 1) + 1); let s = 0; y.forEach((v, i) => { if (v === 1) s += rk[i]; }); return (s - pos * (pos + 1) / 2) / (pos * neg); }
const mse = (p, y) => mean(p.map((v, i) => (v - y[i]) ** 2));
const brier = mse;
const logloss = (p, y) => -mean(p.map((v, i) => y[i] ? Math.log(Math.max(1e-9, v)) : Math.log(Math.max(1e-9, 1 - v))));

// ---------- align on the IHSG calendar; forward-fill suspensions after listing ----------
const ixBars = SRC.bars[SRC.index];
const T = ixBars.map(b => b.time), N = T.length;
const IX = ixBars.map(b => b.close);
const tickers = Object.keys(SRC.bars).filter(s => s !== SRC.index).map(s => s.replace('IDX:', ''));
const S = {};
for (const tk of tickers) {
  const m = new Map(SRC.bars['IDX:' + tk].map(b => [b.time, b]));
  const o = [], h = [], l = [], c = [], v = []; let prev = null;
  for (const t of T) { const b = m.get(t);
    if (b) { o.push(b.open); h.push(b.high); l.push(b.low); c.push(b.close); v.push(b.volume); prev = b.close; }
    else if (prev != null) { o.push(prev); h.push(prev); l.push(prev); c.push(prev); v.push(0); }
    else { o.push(null); h.push(null); l.push(null); c.push(null); v.push(null); } }
  S[tk] = { o, h, l, c, v, first: c.findIndex(x => x != null) };
}
const ixRet = IX.map((x, i) => i ? Math.log(x / IX[i - 1]) : 0);
const ixEw = (() => { const out = new Array(N).fill(null); let ev = sd(ixRet.slice(1, 30)) ** 2; for (let i = 30; i < N; i++) { ev = 0.94 * ev + 0.06 * ixRet[i] ** 2; out[i] = Math.sqrt(ev); } return out; })();

function series(tk) {
  const { h, l, c, v, first: f } = S[tk];
  const ret = c.map((x, i) => i > f ? Math.log(x / c[i - 1]) : 0);
  const tr = c.map((x, i) => i > f ? Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1])) : i === f ? h[i] - l[i] : 0);
  const atr = new Array(N).fill(null), rsi = new Array(N).fill(null), ewv = new Array(N).fill(null);
  if (f >= 0 && f + 31 < N) {
    let a = mean(tr.slice(f + 1, f + 15)); atr[f + 14] = a; for (let i = f + 15; i < N; i++) { a = (a * 13 + tr[i]) / 14; atr[i] = a; }
    let ag = 0, al = 0; for (let i = f + 1; i <= f + 14; i++) { const d = c[i] - c[i - 1]; ag += Math.max(d, 0) / 14; al += Math.max(-d, 0) / 14; }
    for (let i = f + 15; i < N; i++) { const d = c[i] - c[i - 1]; ag = (ag * 13 + Math.max(d, 0)) / 14; al = (al * 13 + Math.max(-d, 0)) / 14; rsi[i] = 100 - 100 / (1 + ag / (al || 1e-9)); }
    let ev = sd(ret.slice(f + 1, f + 30)) ** 2; for (let i = f + 30; i < N; i++) { ev = 0.94 * ev + 0.06 * ret[i] ** 2; ewv[i] = Math.sqrt(ev); }
  }
  return { ret, tr, atr, rsi, ewv, tv: c.map((x, i) => x == null ? null : x * v[i]) };
}
const SER = Object.fromEntries(tickers.map(tk => [tk, series(tk)]));

const FEATURES = [
  ['mom5', 'Return last 5 days'], ['mom20', 'Return last 20 days'], ['mom60', 'Return last 60 days'], ['mom100x5', 'Return 100 days, skipping last 5'],
  ['rs20', 'Return vs IHSG, 20 days'], ['rs60', 'Return vs IHSG, 60 days'],
  ['sma20d', 'Distance from 20-day average'], ['sma50d', 'Distance from 50-day average'], ['sma100d', 'Distance from 100-day average'],
  ['rsi', 'RSI 14'], ['bbpb', 'Bollinger %B'], ['atrp', 'ATR as % of price'],
  ['vol20', 'Volatility 20 days'], ['vol60', 'Volatility 60 days'], ['volratio', 'Vol 20d / Vol 60d'], ['ewvol', 'EWMA volatility (λ 0.94)'],
  ['dd60', 'Drawdown from 60-day high'], ['up60', 'Rise from 60-day low'],
  ['beta60', 'Beta to IHSG, 60 days'], ['corr60', 'Correlation to IHSG'], ['idio60', 'Stock-specific volatility'],
  ['skew60', 'Skewness 60 days'], ['downdev60', 'Downside deviation 60 days'], ['vr5', 'Variance ratio (5-day)'],
  ['liq', 'Average traded value (log)'], ['volsurge', 'Volume 5d / 20d'],
];
const FN = FEATURES.map(f => f[0]);
// Market-state features, used by the risk models (risk depends on the level of market stress, not just ranks).
const MKT = [['mkt_ewvol', 'IHSG EWMA volatility'], ['mkt_mom20', 'IHSG return, 20 days'], ['mkt_dd250', 'IHSG below its 1-year high'], ['breadth', 'Share of stocks above their 50-day average']];
const RISKF = [...FEATURES, ...MKT];
const LOGF = new Set(['atrp', 'vol20', 'vol60', 'ewvol', 'idio60', 'downdev60']);

function feat(tk, i) {
  const { c, h, l, v, first: f } = S[tk], s = SER[tk];
  if (f < 0 || i - f < LOOK || s.atr[i] == null || s.ewv[i] == null) return null;
  const R = s.ret.slice(i - 59, i + 1), RI = ixRet.slice(i - 59, i + 1);
  const mR = mean(R), mI = mean(RI);
  let cov = 0, vI = 0, vS = 0; for (let k = 0; k < 60; k++) { cov += (R[k] - mR) * (RI[k] - mI); vI += (RI[k] - mI) ** 2; vS += (R[k] - mR) ** 2; }
  const beta = cov / (vI || 1e-12), resid = R.map((x, k) => x - beta * RI[k]);
  const sma = p => mean(c.slice(i - p + 1, i + 1));
  const c20 = c.slice(i - 19, i + 1), m20 = mean(c20), s20 = sd(c20);
  const hi60 = Math.max(...h.slice(i - 59, i + 1)), lo60 = Math.min(...l.slice(i - 59, i + 1));
  const r5 = []; for (let k = i - 55; k <= i; k += 5) r5.push(Math.log(c[k] / c[k - 5]));
  const v1 = sd(R) ** 2, v5 = sd(r5) ** 2, neg = R.filter(x => x < 0);
  return {
    mom5: Math.log(c[i] / c[i - 5]), mom20: Math.log(c[i] / c[i - 20]), mom60: Math.log(c[i] / c[i - 60]), mom100x5: Math.log(c[i - 5] / c[i - 100]),
    rs20: Math.log(c[i] / c[i - 20]) - Math.log(IX[i] / IX[i - 20]), rs60: Math.log(c[i] / c[i - 60]) - Math.log(IX[i] / IX[i - 60]),
    sma20d: c[i] / sma(20) - 1, sma50d: c[i] / sma(50) - 1, sma100d: c[i] / sma(100) - 1,
    rsi: s.rsi[i], bbpb: s20 ? (c[i] - (m20 - 2 * s20)) / (4 * s20) : 0.5, atrp: s.atr[i] / c[i],
    vol20: sd(s.ret.slice(i - 19, i + 1)), vol60: sd(R), volratio: sd(s.ret.slice(i - 19, i + 1)) / (sd(R) || 1e-9), ewvol: s.ewv[i],
    dd60: c[i] / hi60 - 1, up60: c[i] / lo60 - 1,
    beta60: beta, corr60: cov / Math.sqrt(vI * vS || 1e-12), idio60: sd(resid),
    skew60: mean(R.map(x => ((x - mR) / (Math.sqrt(vS / 59) || 1)) ** 3)), downdev60: Math.sqrt(mean(neg.map(x => x * x)) || 0), vr5: v5 / (5 * v1 || 1e-12),
    liq: Math.log(mean(s.tv.slice(i - 19, i + 1)) + 1), volsurge: mean(v.slice(i - 4, i + 1)) / (mean(v.slice(i - 19, i + 1)) || 1),
  };
}

// ---------- panel ----------
const panel = []; // per date: { i, rows: [{ tk, raw, x (ranked), z (risk raw), targets }] }
for (let i = LOOK; i < N; i++) {
  const rows = [];
  for (const tk of tickers) { const f = feat(tk, i); if (f) rows.push({ tk, raw: f }); }
  if (rows.length < 8) continue;
  for (let j = 0; j < FN.length; j++) { const rk = rankPct(rows.map(rw => rw.raw[FN[j]])); rows.forEach((rw, q) => { (rw.x ||= [])[j] = rk[q] - 0.5; }); }
  const breadth = mean(rows.map(rw => rw.raw.sma50d > 0 ? 1 : 0));
  const mkt = [ixEw[i] ?? 0, Math.log(IX[i] / IX[Math.max(0, i - 20)]), IX[i] / Math.max(...IX.slice(Math.max(0, i - 249), i + 1)) - 1, breadth];
  for (const rw of rows) {
    // Volatility-type inputs enter the risk models as logs: risk scales multiplicatively, and logs keep
    // crash-level values from sitting far outside the training range.
    rw.z = [...FN.map(k => LOGF.has(k) ? Math.log(rw.raw[k] + 1e-4) : rw.raw[k]), Math.log(mkt[0] + 1e-4), mkt[1], mkt[2], mkt[3]];
    if (i + H < N) {
      const { h, l, c } = S[rw.tk], atr = SER[rw.tk].atr[i], e = c[i], stop = e - STOP_ATR * atr, tp = e + TP_R * STOP_ATR * atr;
      let minL = Infinity, win = 0, decided = false;
      for (let j = i + 1; j <= i + H; j++) { minL = Math.min(minL, l[j]); if (!decided) { if (l[j] <= stop) decided = true; else if (h[j] >= tp) { win = 1; decided = true; } } }
      rw.fwd = Math.log(c[i + H] / e);
      rw.stop = minL <= stop ? 1 : 0;
      rw.win = win;
      rw.vol = Math.log(sd(SER[rw.tk].ret.slice(i + 1, i + H + 1)) + 1e-4);
      rw.mae = minL / e - 1;
      rw.crash = rw.mae <= -0.10 ? 1 : 0;
    }
  }
  const lab = rows.filter(rw => rw.fwd != null);
  if (lab.length) { const ry = rankPct(lab.map(rw => rw.fwd)); lab.forEach((rw, q) => { rw.ret = ry[q] - 0.5; }); }
  panel.push({ i, rows });
}
const labelled = panel.filter(p => p.rows.some(rw => rw.ret != null));
const DIM = { ret: FN.length, risk: RISKF.length };

// ---------- feature preparation ----------
// Risk models see z-scored raw features (fitted on training rows only, clipped at ±4).
function zScaler(rows) { const k = rows[0].z.length, mu = [], sg = [];
  for (let j = 0; j < k; j++) { const col = rows.map(rw => rw.z[j]); mu.push(mean(col)); sg.push(sd(col) || 1); }
  return z => z.map((v, j) => Math.max(-4, Math.min(4, (v - mu[j]) / sg[j]))); }
// Trees work on quantile bins (9 bins per feature) fitted on training rows.
function binner(X) { const k = X[0].length, cuts = [];
  for (let j = 0; j < k; j++) { const col = X.map(x => x[j]); cuts.push([1, 2, 3, 4, 5, 6, 7, 8].map(q => quantile(col, q / 9))); }
  return x => { const b = new Uint8Array(k); for (let j = 0; j < k; j++) { let n = 0; for (const ct of cuts[j]) if (x[j] > ct) n++; b[j] = n; } return b; }; }

// ---------- learners ----------
function ridge(X, y, lam) {
  const P = X[0].length, A = Array.from({ length: P + 1 }, () => new Array(P + 1).fill(0)), b = new Array(P + 1).fill(0);
  for (let n = 0; n < X.length; n++) { const x = [1, ...X[n]]; for (let j = 0; j <= P; j++) { b[j] += x[j] * y[n]; for (let k = 0; k <= P; k++) A[j][k] += x[j] * x[k]; } }
  for (let j = 1; j <= P; j++) A[j][j] += lam;
  for (let c = 0; c <= P; c++) { let p = c; for (let q = c + 1; q <= P; q++) if (Math.abs(A[q][c]) > Math.abs(A[p][c])) p = q; [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let q = 0; q <= P; q++) if (q !== c) { const f = A[q][c] / A[c][c]; for (let k = c; k <= P; k++) A[q][k] -= f * A[c][k]; b[q] -= f * b[c]; } }
  const w = b.map((x, j) => x / A[j][j]);
  return { w, predict: x => w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0) };
}
// Logistic regression; optional per-row offsets (logits of a baseline) so it learns corrections to it.
function logistic(X, y, lam, epochs = 200, lr = 0.5, off = null) {
  const P = X[0].length, w = new Array(P + 1).fill(0); w[0] = off ? 0 : logit(mean(y));
  for (let e = 0; e < epochs; e++) { const g = new Array(P + 1).fill(0);
    for (let n = 0; n < X.length; n++) { let z = w[0] + (off ? off[n] : 0); for (let j = 0; j < P; j++) z += w[j + 1] * X[n][j]; const d = sigmoid(z) - y[n]; g[0] += d; for (let j = 0; j < P; j++) g[j + 1] += d * X[n][j]; }
    for (let j = 0; j <= P; j++) w[j] -= lr * (g[j] / X.length + (j ? lam * w[j] : 0)); }
  return { w, predict: (x, o = 0) => sigmoid(o + w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0)) };
}
// Histogram gradient boosting with Newton leaves, L2 on leaf values and early stopping on a validation set.
// Regularised: shallow trees, large leaves, row and column subsampling, small learning rate.
// `init` / `val.init` are per-row starting logits (a baseline) for classification.
function gbt(Xb, y, task, { val, init, maxTrees = 400, depth = 2, lr = 0.03, minLeaf = 200, lam = 5, sub = 0.5, colsample = 0.6, patience = 40 } = {}) {
  const P = Xb[0].length, NB = 9, cls = task === 'cls';
  const base = init ? 0 : cls ? logit(mean(y)) : mean(y), gain = new Array(P).fill(0), trees = [];
  const F = Float64Array.from({ length: y.length }, (_, n) => init ? init[n] : base);
  const Fv = val ? Float64Array.from({ length: val.y.length }, (_, n) => val.init ? val.init[n] : base) : null;
  let cols = [];
  const loss = (Fa, ya) => cls ? logloss(Array.from(Fa, sigmoid), ya) : mse(Array.from(Fa), ya);
  let best = val ? loss(Fv, val.y) : 0, bestN = 0, since = 0;
  const pt = (nd, b) => nd.val !== undefined ? nd.val : pt(b[nd.j] <= nd.cut ? nd.L : nd.R, b);
  function build(idx, g, hs, d) {
    let G = 0, Hh = 0; for (const n of idx) { G += g[n]; Hh += hs[n]; }
    const leaf = { val: G / (Hh + lam) };
    if (d === 0 || idx.length < 2 * minLeaf) return leaf;
    let bestS = null;
    for (const j of cols) {
      const gs = new Float64Array(NB), hh = new Float64Array(NB), cn = new Int32Array(NB);
      for (const n of idx) { const b = Xb[n][j]; gs[b] += g[n]; hh[b] += hs[n]; cn[b]++; }
      let gl = 0, hl = 0, nl = 0;
      for (let b = 0; b < NB - 1; b++) { gl += gs[b]; hl += hh[b]; nl += cn[b]; const nr = idx.length - nl;
        if (nl < minLeaf || nr < minLeaf) continue;
        const sc = gl * gl / (hl + lam) + (G - gl) ** 2 / (Hh - hl + lam) - G * G / (Hh + lam);
        if (!bestS || sc > bestS.sc) bestS = { j, cut: b, sc }; }
    }
    if (!bestS || bestS.sc <= 0) return leaf;
    gain[bestS.j] += bestS.sc;
    const L = [], R = []; for (const n of idx) (Xb[n][bestS.j] <= bestS.cut ? L : R).push(n);
    return { j: bestS.j, cut: bestS.cut, L: build(L, g, hs, d - 1), R: build(R, g, hs, d - 1) };
  }
  for (let t = 0; t < maxTrees; t++) {
    const g = new Float64Array(y.length), hs = new Float64Array(y.length);
    for (let n = 0; n < y.length; n++) { if (cls) { const p = sigmoid(F[n]); g[n] = y[n] - p; hs[n] = Math.max(1e-6, p * (1 - p)); } else { g[n] = y[n] - F[n]; hs[n] = 1; } }
    const idx = []; for (let n = 0; n < y.length; n++) if (rand() < sub) idx.push(n);
    cols = []; for (let j = 0; j < P; j++) if (rand() < colsample) cols.push(j); if (!cols.length) cols.push(Math.floor(rand() * P));
    const tree = build(idx, g, hs, depth); trees.push(tree);
    for (let n = 0; n < y.length; n++) F[n] += lr * pt(tree, Xb[n]);
    if (val) { for (let n = 0; n < val.y.length; n++) Fv[n] += lr * pt(tree, val.Xb[n]);
      const L = loss(Fv, val.y); if (L < best - 1e-7) { best = L; bestN = t + 1; since = 0; } else if (++since >= patience) break; }
  }
  const used = val ? trees.slice(0, Math.max(1, bestN)) : trees;
  return { gain, trees: used.length, predict: (b, o) => { const f = (o ?? base) + lr * used.reduce((s, tr) => s + pt(tr, b), 0); return cls ? sigmoid(f) : f; } };
}
// Small MLP (k-16-8-1), ReLU, Adam, L2, early stopping on validation loss; bagged over seeds by the caller.
// `off` / `val.off`: per-row baseline logits added to the output for classification.
function mlp(X, y, task, { val, off, hid = [8], maxEpochs = 20, lr = 0.003, l2 = 1e-3, batch = 128, patience = 3 } = {}) {
  const cls = task === 'cls', sizes = [X[0].length, ...hid, 1], L = sizes.length - 1;
  const ym = cls ? 0 : mean(y), ys = cls ? 1 : (sd(y) || 1), yt = y.map(v => cls ? v : (v - ym) / ys);
  const W = [], B = [], mW = [], vW = [], mB = [], vB = []; let step = 0;
  for (let l = 0; l < L; l++) { const s = Math.sqrt(2 / sizes[l]); W.push(Array.from({ length: sizes[l + 1] }, () => Float64Array.from({ length: sizes[l] }, () => (rand() * 2 - 1) * s)));
    B.push(new Float64Array(sizes[l + 1])); mW.push(W[l].map(rw => new Float64Array(rw.length))); vW.push(W[l].map(rw => new Float64Array(rw.length))); mB.push(new Float64Array(sizes[l + 1])); vB.push(new Float64Array(sizes[l + 1])); }
  if (cls && !off) B[L - 1][0] = logit(mean(y));
  const fwd = x => { const acts = [x]; for (let l = 0; l < L; l++) { const a = new Float64Array(sizes[l + 1]); for (let o = 0; o < a.length; o++) { let s = B[l][o]; const w = W[l][o], p = acts[l]; for (let k = 0; k < w.length; k++) s += w[k] * p[k]; a[o] = l < L - 1 ? Math.max(0, s) : s; } acts.push(a); } return acts; };
  const out = (x, o = 0) => { const z = fwd(x)[L][0]; return cls ? sigmoid(z + o) : z * ys + ym; };
  const vloss = () => { const p = val.X.map((x, n) => out(x, val.off ? val.off[n] : 0)); return cls ? logloss(p, val.y) : mse(p, val.y); };
  const snapW = () => ({ W: W.map(m => m.map(rw => Float64Array.from(rw))), B: B.map(b => Float64Array.from(b)) });
  let best = val ? vloss() : 0, bestW = snapW(), since = 0;
  const order = y.map((_, n) => n);
  for (let e = 0; e < maxEpochs; e++) {
    for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(rand() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
    for (let s0 = 0; s0 < order.length; s0 += batch) {
      const gW = W.map(m => m.map(rw => new Float64Array(rw.length))), gB = B.map(b => new Float64Array(b.length)), bs = Math.min(batch, order.length - s0);
      for (let q = s0; q < s0 + bs; q++) { const n = order[q], acts = fwd(X[n]); const z = acts[L][0]; let delta = [cls ? sigmoid(z + (off ? off[n] : 0)) - yt[n] : z - yt[n]];
        for (let l = L - 1; l >= 0; l--) { for (let o = 0; o < delta.length; o++) { gB[l][o] += delta[o]; const gw = gW[l][o], p = acts[l]; for (let k = 0; k < p.length; k++) gw[k] += delta[o] * p[k]; }
          if (l) { const nd = new Array(acts[l].length).fill(0); for (let k = 0; k < nd.length; k++) if (acts[l][k] > 0) { let s = 0; for (let o = 0; o < delta.length; o++) s += delta[o] * W[l][o][k]; nd[k] = s; } delta = nd; } } }
      step++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** step, c2 = 1 - b2 ** step;
      for (let l = 0; l < L; l++) for (let o = 0; o < W[l].length; o++) {
        for (let k = 0; k < W[l][o].length; k++) { const g = gW[l][o][k] / bs + l2 * W[l][o][k]; mW[l][o][k] = b1 * mW[l][o][k] + (1 - b1) * g; vW[l][o][k] = b2 * vW[l][o][k] + (1 - b2) * g * g; W[l][o][k] -= lr * (mW[l][o][k] / c1) / (Math.sqrt(vW[l][o][k] / c2) + 1e-8); }
        const g = gB[l][o] / bs; mB[l][o] = b1 * mB[l][o] + (1 - b1) * g; vB[l][o] = b2 * vB[l][o] + (1 - b2) * g * g; B[l][o] -= lr * (mB[l][o] / c1) / (Math.sqrt(vB[l][o] / c2) + 1e-8); }
    }
    if (val) { const vl = vloss(); if (vl < best - 1e-7) { best = vl; bestW = snapW(); since = 0; } else if (++since >= patience) break; }
  }
  if (val) { bestW.W.forEach((m, l) => m.forEach((rw, o) => W[l][o].set(rw))); bestW.B.forEach((b, l) => B[l].set(b)); }
  return { predict: out };
}

// ---------- targets ----------
const TARGETS = {
  ret:   { label: 'Next-20-day return rank', task: 'reg', feats: 'ret' },
  stop:  { label: 'Stop loss hit within 20 days', task: 'cls', feats: 'risk' },
  crash: { label: 'Drop of 10% or more within 20 days', task: 'cls', feats: 'risk' },
  vol:   { label: 'Volatility over next 20 days', task: 'reg', feats: 'risk' },
  mae:   { label: 'Worst dip over next 20 days', task: 'reg', feats: 'risk' },
};
const MODELS = { ridge: 'Ridge / logistic regression', gbt: 'Gradient-boosted trees', nn: 'Neural network (bagged ×2)' };
const ZI = { ewvol: FN.indexOf('ewvol'), atrp: FN.indexOf('atrp'), mktvol: FN.length, mktdd: FN.length + 2 };

// Skill of predictions on a set of rows (higher is better; 0 = no better than the baseline).
function skill(target, pred, rows, base) {
  const y = rows.map(rw => rw[target]);
  if (target === 'ret') { // mean cross-sectional IC per date
    const by = new Map(); rows.forEach((rw, k) => { (by.get(rw.d) || by.set(rw.d, []).get(rw.d)).push(k); });
    return mean([...by.values()].filter(ks => ks.length >= 8).map(ks => spearman(ks.map(k => pred[k]), ks.map(k => rows[k].fwd))));
  }
  if (TARGETS[target].task === 'cls') return auc(pred, y) - 0.5;
  return 1 - mse(pred, y) / mse(base, y); // vs baseline
}
// Baselines: small, hard-to-overfit models on the stock's own log volatility, its ATR and market stress.
// Every learner below fits corrections on top of this, so the ensemble starts from a sound prior.
function baseline(target, train) {
  if (target === 'ret') return () => 0;
  const pick = rw => [rw.z[ZI.ewvol], rw.z[ZI.atrp], rw.z[ZI.mktvol], rw.z[ZI.mktdd]];
  const X = train.map(pick), mu = [0, 1, 2, 3].map(j => mean(X.map(x => x[j]))), sg = [0, 1, 2, 3].map(j => sd(X.map(x => x[j])) || 1);
  const st = x => x.map((v, j) => (v - mu[j]) / sg[j]);
  const y = train.map(rw => rw[target]);
  const m = TARGETS[target].task === 'cls' ? logistic(X.map(st), y, 0.001, 150, 0.5) : ridge(X.map(st), y, 1);
  return rw => m.predict(st(pick(rw)));
}

// Fit three learners for one target with nested validation. Risk targets learn residuals over the
// baseline (regression) or start from its logits (classification). Returns full-prediction functions.
function fitTarget(target, trainRows) {
  const tk = TARGETS[target], cls = tk.task === 'cls', risk = tk.feats === 'risk';
  const dates = [...new Set(trainRows.map(rw => rw.d))].sort((a, b) => a - b);
  const cutAt = dates[Math.floor(dates.length * 0.85)];
  const inner = trainRows.filter(rw => rw.d < cutAt - H), val = trainRows.filter(rw => rw.d >= cutAt);
  const scal = risk ? zScaler(inner) : null, fx = rw => risk ? scal(rw.z) : rw.x;
  const bI = baseline(target, inner);
  const bInner = inner.map(bI), bVal = val.map(bI);
  const Xi = inner.map(fx), Xv = val.map(fx);
  const yRaw = inner.map(rw => rw[target]), yvRaw = val.map(rw => rw[target]);
  // What the learners are fitted on.
  const resid = !cls && risk;
  const yi = resid ? yRaw.map((v, n) => v - bInner[n]) : yRaw, yv = resid ? yvRaw.map((v, n) => v - bVal[n]) : yvRaw;
  const offI = cls ? bInner.map(logit) : null, offV = cls ? bVal.map(logit) : null;
  const full = (raw, b) => cls ? raw : resid ? b + raw : raw; // turn a learner output into a full prediction
  const out = {};

  // Linear: penalty chosen on validation, then refit on the whole training window.
  const grid = cls ? [0.01, 0.1, 1] : [10, 100, 1000, 10000];
  let bestLam = grid[0], bestSk = -Infinity;
  for (const lam of grid) {
    const m = cls ? logistic(Xi, yi, lam, 200, 0.5, offI) : ridge(Xi, yi, lam);
    const p = Xv.map((x, n) => cls ? m.predict(x, offV[n]) : full(m.predict(x), bVal[n]));
    const sk = skill(target, p, val, bVal); if (sk > bestSk) { bestSk = sk; bestLam = lam; }
  }
  const scalA = risk ? zScaler(trainRows) : null, fxA = rw => risk ? scalA(rw.z) : rw.x;
  const bA = baseline(target, trainRows), bAll = trainRows.map(bA);
  const Xa = trainRows.map(fxA), yaRaw = trainRows.map(rw => rw[target]);
  const ya = resid ? yaRaw.map((v, n) => v - bAll[n]) : yaRaw, offA = cls ? bAll.map(logit) : null;
  const lin = cls ? logistic(Xa, ya, bestLam, 200, 0.5, offA) : ridge(Xa, ya, bestLam);
  out.ridge = { val: bestSk, hp: { lambda: bestLam }, w: lin.w,
    predict: rw => { const b = bA(rw); return cls ? lin.predict(fxA(rw), logit(b)) : full(lin.predict(fxA(rw)), b); } };

  // Trees: early stopping on validation picks the tree count; refit on the full window with that count.
  const bin = binner(Xi);
  reseed(7); const gEs = gbt(Xi.map(bin), yi, tk.task, { val: { Xb: Xv.map(bin), y: yv, init: offV }, init: offI });
  const gValP = Xv.map((x, n) => cls ? gEs.predict(bin(x), offV[n]) : full(gEs.predict(bin(x)), bVal[n]));
  const binA = binner(Xa); reseed(7); const gA = gbt(Xa.map(binA), ya, tk.task, { maxTrees: gEs.trees, init: offA });
  out.gbt = { val: skill(target, gValP, val, bVal), hp: { trees: gEs.trees }, gain: gA.gain,
    predict: rw => { const b = bA(rw); return cls ? gA.predict(binA(fxA(rw)), logit(b)) : full(gA.predict(binA(fxA(rw))), b); } };

  // Neural net: two seeds, each early-stopped on validation, averaged. Kept as trained on the inner window.
  const nets = [11, 23].map(s0 => { reseed(s0); return mlp(Xi, yi, tk.task, { val: { X: Xv, y: yv, off: offV }, off: offI }); });
  const nnRaw = (x, o) => mean(nets.map(n => n.predict(x, o)));
  const nnValP = Xv.map((x, n) => cls ? nnRaw(x, offV[n]) : full(nnRaw(x), bVal[n]));
  out.nn = { val: skill(target, nnValP, val, bVal), hp: {},
    predict: rw => { const b = bI(rw); return cls ? nnRaw(fx(rw), logit(b)) : full(nnRaw(fx(rw)), b); } };

  // Ensemble weights from validation skill (positive part only).
  const raw = Object.fromEntries(Object.entries(out).map(([k, m]) => [k, Math.max(0, m.val)]));
  const tot = Object.values(raw).reduce((a, b) => a + b, 0);
  const weights = Object.fromEntries(Object.keys(out).map(k => [k, tot > 0 ? raw[k] / tot : 0]));
  const fit = { models: out, weights, base: bA, anySkill: tot > 0, calib: null };
  // Calibrate yes/no probabilities on the validation window (Platt scaling), inside the fold.
  if (cls && fit.anySkill) {
    const pv = { gbt: gValP, nn: nnValP, ridge: Xv.map((x, n) => out.ridge.predict(val[n])) };
    const ens = val.map((_, n) => Object.entries(weights).reduce((s, [k, w]) => s + w * pv[k][n], 0));
    const pm = logistic(ens.map(p => [logit(p)]), yvRaw, 0, 300, 0.5);
    fit.calib = p => pm.predict([logit(p)]);
  }
  return fit;
}
function ensemblePredict(fit, target, rows) {
  if (target === 'ret') { // weighted average of ranks within each date
    const ws = Object.values(fit.weights).some(w => w > 0) ? fit.weights : { ridge: 1 / 3, gbt: 1 / 3, nn: 1 / 3 };
    const byD = new Map(); rows.forEach((rw, k) => { (byD.get(rw.d) || byD.set(rw.d, []).get(rw.d)).push(k); });
    const out = new Array(rows.length).fill(0);
    for (const [m, w] of Object.entries(ws)) { if (!w) continue; const p = rows.map(fit.models[m].predict);
      for (const ks of byD.values()) { const rk = rankPct(ks.map(k => p[k])); ks.forEach((k, q) => { out[k] += w * rk[q]; }); } }
    return out;
  }
  if (!fit.anySkill) return rows.map(fit.base); // no learner beat the baseline: use the baseline
  const p = rows.map(rw => Object.entries(fit.weights).reduce((s, [m, w]) => s + (w ? w * fit.models[m].predict(rw) : 0), 0));
  return fit.calib ? p.map(fit.calib) : p;
}

// ---------- nested walk-forward ----------
const allRows = []; labelled.forEach((p, d) => p.rows.forEach(rw => { if (rw.ret != null) { rw.d = d; allRows.push(rw); } }));
const D = labelled.length;
const oos = Object.fromEntries(Object.keys(TARGETS).map(t => [t, { rows: [], pred: [], base: [], baseRate: [], perModel: { ridge: [], gbt: [], nn: [] }, trainSkill: [], testSkill: [], weights: [] }]));
let folds = 0;
for (let start = MIN_TRAIN_DAYS; start < D; start += FOLD) {
  const train = allRows.filter(rw => rw.d < start - H && rw.d % THIN === 0);
  const test = allRows.filter(rw => rw.d >= start && rw.d < start + FOLD);
  if (!test.length) break;
  folds++;
  for (const t of Object.keys(TARGETS)) {
    const fit = fitTarget(t, train), o = oos[t];
    const p = ensemblePredict(fit, t, test), b = test.map(fit.base);
    o.rows.push(...test); o.pred.push(...p); o.base.push(...b); const br = mean(train.map(rw => rw[t])); o.baseRate.push(...test.map(() => br));
    for (const m of Object.keys(MODELS)) o.perModel[m].push(...(t === 'ret' ? test.map(fit.models[m].predict) : test.map(fit.models[m].predict)));
    const trS = train.filter((_, k) => k % 3 === 0);
    o.trainSkill.push(skill(t, ensemblePredict(fit, t, trS), trS, trS.map(fit.base)));
    o.testSkill.push(skill(t, p, test, b));
    o.weights.push(fit.weights);
  }
  process.stderr.write(`fold ${folds}: test days ${start}-${Math.min(D, start + FOLD) - 1} of ${D}\n`);
}

// ---------- validation summaries ----------
function calib(p, y) { const bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]; return bins.slice(0, -1).map((lo, k) => { const ks = p.map((v, i) => [v, i]).filter(([v]) => v >= lo && v < bins[k + 1]).map(([, i]) => i);
  return { from: lo, to: Math.min(1, bins[k + 1]), n: ks.length, predicted: r(mean(ks.map(i => p[i])), 3), actual: r(mean(ks.map(i => y[i])), 3) }; }).filter(b => b.n > 0); }
const validation = {};
for (const [t, o] of Object.entries(oos)) {
  const y = o.rows.map(rw => rw[t]);
  const v = { label: TARGETS[t].label, task: TARGETS[t].task, folds, test_rows: o.rows.length,
    train_skill: r(mean(o.trainSkill)), test_skill: r(mean(o.testSkill)), overfit_gap: r(mean(o.trainSkill) - mean(o.testSkill)),
    avg_weights: Object.fromEntries(Object.keys(MODELS).map(m => [m, r(mean(o.weights.map(w => w[m])), 3)])),
    per_model: Object.fromEntries(Object.keys(MODELS).map(m => [m, r(skill(t, o.perModel[m], o.rows, o.base))])) };
  if (t === 'ret') {
    const byD = new Map(); o.rows.forEach((rw, k) => { (byD.get(rw.d) || byD.set(rw.d, []).get(rw.d)).push(k); });
    const ds = [...byD.keys()].sort((a, b) => a - b), ics = [], spreads = [];
    for (const d of ds) { const ks = byD.get(d); if (ks.length < 8) continue; const p = ks.map(k => o.pred[k]), f = ks.map(k => o.rows[k].fwd);
      ics.push({ d, ic: spearman(p, f) }); const rk = rankPct(p); spreads.push(mean(f.filter((_, q) => rk[q] >= 0.8)) - mean(f.filter((_, q) => rk[q] <= 0.2))); }
    const non = ics.filter((_, k) => k % H === 0).map(x => x.ic);
    Object.assign(v, { ic: r(mean(ics.map(x => x.ic))), ic_hit: r(mean(ics.map(x => x.ic > 0 ? 1 : 0))), ic_t_nonoverlap: r(mean(non) / (sd(non) / Math.sqrt(non.length)), 2),
      n_independent: non.length, spread20d: r(mean(spreads)), test_days: ics.length });
  } else if (TARGETS[t].task === 'cls') {
    // Compare with a constant forecast (the training base rate of each fold) and with the baseline model.
    const constP = o.rows.map((_, k) => o.baseRate[k]);
    Object.assign(v, { auc: r(auc(o.pred, y), 3), auc_baseline: r(auc(o.base, y), 3), brier: r(brier(o.pred, y)), brier_const: r(brier(constP, y)),
      brier_skill: r(1 - brier(o.pred, y) / brier(constP, y)), brier_skill_vs_baseline: r(1 - brier(o.pred, y) / brier(o.base, y)),
      base_rate: r(mean(constP), 3), actual_rate: r(mean(y), 3), calibration: calib(o.pred, y) });
  } else {
    Object.assign(v, { r2: r(1 - mse(o.pred, y) / mse(y.map(() => mean(y)), y)), skill_vs_baseline: r(1 - mse(o.pred, y) / mse(o.base, y)), corr: r(pearson(o.pred, y), 3) });
  }
  validation[t] = v;
}
const reliable = validation.ret.ic > 0 && validation.ret.ic_t_nonoverlap >= 2 && validation.ret.n_independent >= 8;
const riskReliable = {
  stop: validation.stop.auc >= 0.6 && validation.stop.brier_skill > 0.02,
  crash: validation.crash.auc >= 0.6 && validation.crash.brier_skill > 0.02,
  // The baseline is itself a forecast; "reliable" means the whole forecast explains a useful share of what happened.
  vol: validation.vol.r2 > 0.3,
  mae: validation.mae.r2 > 0.15,
};
// Bad-case dip: 10th percentile of (actual − predicted) on out-of-sample rows.
const maeResQ10 = quantile(oos.mae.rows.map((rw, k) => rw.mae - oos.mae.pred[k]), 0.10);

// ---------- final models on all data ----------
const finalTrain = allRows.filter(rw => rw.d % THIN === 0);
const FIN = Object.fromEntries(Object.keys(TARGETS).map(t => [t, fitTarget(t, finalTrain)]));
// Return ranking: average out-of-sample fold weights. Risk targets keep the final fit's own validation
// weights and calibration, falling back to the baseline when the learners showed no skill out of sample.
{
  const w = validation.ret.avg_weights, tot = Object.values(w).reduce((a, b) => a + b, 0);
  FIN.ret.weights = tot > 0 ? Object.fromEntries(Object.entries(w).map(([k, v]) => [k, v / tot])) : { ridge: 1 / 3, gbt: 1 / 3, nn: 1 / 3 };
  for (const t of ['vol', 'mae']) if (validation[t].test_skill <= 0) FIN[t].anySkill = false;
  for (const t of ['stop', 'crash']) if (validation[t].auc <= validation[t].auc_baseline) FIN[t].anySkill = false;
}
// Record which predictor won out of sample, and calibrate yes/no forecasts on the winner's pooled
// out-of-sample predictions (Platt scaling), so a 40% forecast means about 40%.
const finalCalib = {};
for (const t of ['stop', 'crash', 'vol', 'mae']) {
  const v = validation[t], o = oos[t], useL = FIN[t].anySkill, p = useL ? o.pred : o.base, y = o.rows.map(rw => rw[t]);
  v.chosen = useL ? 'learners' : 'baseline';
  if (TARGETS[t].task === 'cls') {
    v.chosen_auc = r(auc(p, y), 3);
    const pm = logistic(p.map(q => [logit(q)]), y, 0, 300, 0.5); finalCalib[t] = q => pm.predict([logit(q)]);
    const pc = p.map(finalCalib[t]); v.chosen_calibration = calib(pc, y); v.chosen_brier_skill = r(1 - brier(pc, y) / brier(o.rows.map((_, k) => o.baseRate[k]), y));
  } else v.chosen_r2 = r(1 - mse(p, y) / mse(y.map(() => mean(y)), y));
}
// What the forecasts can be used for: ranking stocks by risk (AUC / R² on unseen data), not exact odds.
riskReliable.stop = validation.stop.chosen_auc >= 0.6;
riskReliable.crash = validation.crash.chosen_auc >= 0.6;
riskReliable.vol = validation.vol.chosen_r2 >= 0.25;
riskReliable.mae = validation.mae.chosen_r2 >= 0.15;
const latest = panel.at(-1); latest.rows.forEach(rw => { rw.d = 1e9; });
const pred = Object.fromEntries(Object.keys(TARGETS).map(t => [t, ensemblePredict(FIN[t], t, latest.rows)]));
const modelRanks = Object.fromEntries(Object.keys(MODELS).map(m => [m, rankPct(latest.rows.map(FIN.ret.models[m].predict))]));
const ensRank = rankPct(pred.ret);
const voters = Object.keys(MODELS).filter(m => FIN.ret.weights[m] > 0);
const agreement = latest.rows.map((_, j) => voters.length < 2 ? 1 : 1 - sd(voters.map(m => modelRanks[m][j])) / 0.29);
const gGain = FIN.ret.models.gbt.gain, gTot = gGain.reduce((a, b) => a + b, 0) || 1;
const sGain = FIN.crash.models.gbt.gain, sTot = sGain.reduce((a, b) => a + b, 0) || 1;
const importance = FEATURES.map(([k, label], j) => ({ key: k, label, gbt: r(gGain[j] / gTot), ridge: r(FIN.ret.models.ridge.w[j + 1]), logit: r(FIN.ret.models.ridge.w[j + 1]) })).sort((a, b) => b.gbt - a.gbt);
const riskImportance = RISKF.map(([k, label], j) => ({ key: k, label, gbt: r(sGain[j] / sTot) })).sort((a, b) => b.gbt - a.gbt);

// ---------- per-stock output ----------
const last = N - 1;
function bracketStats(tk, slM = STOP_ATR, rr = TP_R) {
  const { h, l, c, first: f } = S[tk], atr = SER[tk].atr; let w = 0, ls = 0, op = 0;
  for (let i = Math.max(15, f + 15); i < N - 1; i++) { if (atr[i] == null) continue; const e = c[i], rk = slM * atr[i], sl = e - rk, tp = e + rr * rk; let res = null;
    for (let j = i + 1; j < Math.min(N, i + 1 + H); j++) { if (l[j] <= sl) { res = 0; break; } if (h[j] >= tp) { res = 1; break; } }
    if (res === 1) w++; else if (res === 0) ls++; else op++; }
  return { p_tp_first: r(w / (w + ls || 1)), breakeven: r(1 / (1 + rr)), expectancy_R: r((w / (w + ls || 1)) * rr - (ls / (w + ls || 1))), trials: w + ls + op };
}
function swingLevels(tk) {
  const { h, l, c } = S[tk], atr = SER[tk].atr[last], px = c[last], piv = [], from = Math.max(5, N - 300);
  for (let i = from; i < N - 5; i++) { const wh = h.slice(i - 5, i + 6), wl = l.slice(i - 5, i + 6); if (h[i] === Math.max(...wh)) piv.push(h[i]); if (l[i] === Math.min(...wl)) piv.push(l[i]); }
  const cl = []; for (const p of piv.sort((a, b) => a - b)) { const k = cl.find(q => Math.abs(q.p - p) <= atr * 0.5); if (k) { k.m.push(p); k.p = mean(k.m); } else cl.push({ p, m: [p] }); }
  const lv = cl.map(q => ({ price: snap(q.p), touches: q.m.length }));
  return { res: lv.filter(q => q.price > px * 1.005).sort((a, b) => a.price - b.price).slice(0, 3), sup: lv.filter(q => q.price < px * 0.995).sort((a, b) => b.price - a.price).slice(0, 3) };
}
function maxDD(c) { let pk = -Infinity, m = 0; for (const x of c) { if (x == null) continue; pk = Math.max(pk, x); m = Math.min(m, x / pk - 1); } return m; }
const W252 = Math.max(0, N - 300); // risk statistics use the last ~300 days so they describe the current regime

const stocks = latest.rows.map((rw, j) => {
  const tk = rw.tk, f = rw.raw, c = S[tk].c, px = c[last], s = SER[tk];
  const rets = s.ret.slice(Math.max(W252, S[tk].first + 1)).map(x => Math.exp(x) - 1);
  const var95 = -quantile(rets, 0.05), cvar95 = -mean(rets.filter(x => x <= -var95));
  const atr = s.atr[last], stop = snap(px - STOP_ATR * atr, 'down'), risk = px - stop;
  const pStop = finalCalib.stop(pred.stop[j]), pCrash = finalCalib.crash(pred.crash[j]);
  const vol20 = Math.exp(pred.vol[j]) * Math.sqrt(252), maeMed = Math.min(0, pred.mae[j]), maeBad = Math.min(0, pred.mae[j] + maeResQ10);
  return {
    tk, sector: SECTORS[tk] || 'Other', price: px, tick: idxTick(px),
    chg1d: r(c[last] / c[last - 1] - 1), chg20d: r(Math.exp(f.mom20) - 1), chg60d: r(Math.exp(f.mom60) - 1),
    spark: c.slice(-60),
    score: Math.round(ensRank[j] * 100), agreement: r(Math.max(0, agreement[j]), 2),
    model_ranks: Object.fromEntries(Object.keys(MODELS).map(m => [m, Math.round(modelRanks[m][j] * 100)])),
    trend: f.sma20d > 0 && f.sma50d > 0 && f.sma100d > 0 ? 'up' : f.sma20d < 0 && f.sma50d < 0 && f.sma100d < 0 ? 'down' : 'mixed',
    rsi: r(f.rsi, 1), atr: r(atr, 1), atr_pct: r(f.atrp), vol_ann: r(f.vol60 * Math.sqrt(252)), ewvol_ann: r(f.ewvol * Math.sqrt(252)),
    var95: r(var95), cvar95: r(cvar95), maxdd: r(maxDD(c.slice(W252))), dd_now: r(px / Math.max(...c.slice(W252).filter(x => x != null)) - 1),
    beta: r(f.beta60, 2), corr: r(f.corr60, 2), skew: r(f.skew60, 2), kurt: r(mean(rets.map(x => ((x - mean(rets)) / sd(rets)) ** 4)) - 3, 1),
    liq_bn: r(Math.exp(f.liq) / 1e9, 1), history_days: N - S[tk].first,
    plan: { entry: px, stop, risk, risk_pct: r(risk / px), tp1: snap(px + 1.5 * risk), tp2: snap(px + 2.5 * risk), lot_risk: risk * 100,
      bep: snap(px * (1 + FEES.buy) / (1 - FEES.sell), 'up') },
    forecast: { p_stop: r(pStop, 3), p_crash10: r(pCrash, 3), vol20_ann: r(vol20, 3), dip_typical: r(maeMed, 4), dip_bad: r(maeBad, 4) },
    bracket: bracketStats(tk),
    levels: swingLevels(tk),
    raw: Object.fromEntries(FN.map(k => [k, r(f[k])])),
  };
});

// Mitigation: size so every position carries similar forecast volatility; widen the stop if the bad-case dip reaches it.
const medVol = quantile(stocks.map(s => s.forecast.vol20_ann), 0.5);
for (const s of stocks) {
  const fc = s.forecast, p = s.plan;
  fc.size_mult = r(Math.max(0.25, Math.min(1.5, medVol / fc.vol20_ann)), 2);
  const badPx = p.entry * (1 + fc.dip_bad);
  fc.noise_reaches_stop = badPx <= p.stop;
  fc.wide_stop = fc.noise_reaches_stop ? snap(badPx * 0.995, 'down') : p.stop;
  fc.wide_stop_lot_risk = (p.entry - fc.wide_stop) * 100;
  fc.level = fc.p_stop >= 0.6 ? 'high' : fc.p_stop >= 0.4 ? 'medium' : 'low';
}

// Risk grade: average cross-sectional percentile of vol, CVaR, drawdown, beta, illiquidity, kurtosis, forecast stop-hit probability.
const riskCols = [[s => s.vol_ann, 1], [s => s.cvar95, 1], [s => s.maxdd, -1], [s => s.beta, 1], [s => s.liq_bn, -1], [s => s.kurt, 1],
  [s => s.forecast.p_crash10, riskReliable.crash ? 2 : 0], [s => s.forecast.vol20_ann, riskReliable.vol ? 1 : 0], [s => s.forecast.p_stop, riskReliable.stop ? 1 : 0]];
const riskPct = riskCols.filter(([, w]) => w).map(([g, w]) => ({ w: Math.abs(w), col: rankPct(stocks.map(s => Math.sign(w) * g(s))) }));
const wSum = riskPct.reduce((a, x) => a + x.w, 0);
stocks.forEach((s, j) => { const rp = riskPct.reduce((a, x) => a + x.w * x.col[j], 0) / wSum; s.risk_pct = r(rp, 2); s.risk_grade = 'ABCDE'[Math.min(4, Math.floor(rp * 5))]; });
const LIQ_MIN = 10, medCrash = quantile(stocks.map(s => s.forecast.p_crash10), 0.5), p90Crash = quantile(stocks.map(s => s.forecast.p_crash10), 0.9);
stocks.forEach(s => {
  s.liquid = s.liq_bn >= LIQ_MIN;
  // Mitigation filter: when the 10%-drop forecast ranks risk reliably, shortlist only the safer half
  // and mark the riskiest tenth as avoid.
  const crashOk = !riskReliable.crash || s.forecast.p_crash10 <= medCrash;
  const crashBad = riskReliable.crash && s.forecast.p_crash10 >= p90Crash;
  s.label = s.score >= 70 && 'ABC'.includes(s.risk_grade) && s.liquid && s.bracket.expectancy_R > -0.1 && crashOk ? 'shortlist'
    : s.score <= 30 || s.risk_grade === 'E' || !s.liquid || crashBad ? 'avoid' : 'watch';
  const why = [];
  if (s.score >= 70) why.push('ranked in the top 30% by the models'); else if (s.score <= 30) why.push('ranked in the bottom 30% by the models');
  if (s.trend === 'up') why.push('price above its 20, 50 and 100-day averages'); if (s.trend === 'down') why.push('price below its 20, 50 and 100-day averages');
  if ('AB'.includes(s.risk_grade)) why.push('lower risk than most of the list'); if ('DE'.includes(s.risk_grade)) why.push('higher risk than most of the list');
  if (riskReliable.crash) { if (s.forecast.p_crash10 >= p90Crash) why.push(`among the 10% most likely to drop 10% or more in the next 20 days (${Math.round(s.forecast.p_crash10 * 100)}% chance)`);
    else if (s.forecast.p_crash10 <= medCrash) why.push(`lower-than-median chance of a 10% drop in the next 20 days (${Math.round(s.forecast.p_crash10 * 100)}%)`);
    else why.push(`higher-than-median chance of a 10% drop in the next 20 days (${Math.round(s.forecast.p_crash10 * 100)}%)`); }
  if (!s.liquid) why.push(`thin trading (Rp${s.liq_bn} bn a day)`);
  if (s.bracket.expectancy_R > 0.05) why.push('its stop/target rule would have paid off historically');
  if (s.bracket.expectancy_R < -0.1) why.push('its stop/target rule lost money historically');
  s.why = why;
});

// ---------- minimum-variance allocation across the shortlist ----------
function minVar(names) {
  if (names.length < 2) return { weights: names.map(n => ({ tk: n, w: 1 })) };
  const R = names.map(n => SER[n].ret.slice(-120)), k = names.length;
  const C = R.map(a => R.map(b => { const ma = mean(a), mb = mean(b); return a.reduce((s, x, t) => s + (x - ma) * (b[t] - mb), 0) / (a.length - 1); }));
  let w = new Array(k).fill(1 / k);
  for (let it = 0; it < 3000; it++) { const g = C.map(row => 2 * row.reduce((s, c, q) => s + c * w[q], 0)); w = w.map((x, q) => Math.max(0, x - 0.5 * g[q] / (C[q][q] || 1) * 0.01));
    for (let rep = 0; rep < 20; rep++) { const sm = w.reduce((a, b) => a + b, 0); w = w.map(x => Math.min(0.35, x / sm)); } }
  const sm = w.reduce((a, b) => a + b, 0); w = w.map(x => x / sm);
  const pv = Math.sqrt(w.reduce((s, wi, a) => s + wi * C[a].reduce((t, c, b) => t + c * w[b], 0), 0)) * Math.sqrt(252);
  const avgVol = w.reduce((s, wi, a) => s + wi * Math.sqrt(C[a][a]), 0) * Math.sqrt(252);
  const corrs = []; for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) corrs.push(C[a][b] / Math.sqrt(C[a][a] * C[b][b]));
  return { weights: names.map((n, q) => ({ tk: n, w: r(w[q], 3) })), port_vol: r(pv), avg_stock_vol: r(avgVol), diversification: r(avgVol / pv, 2), avg_corr: r(mean(corrs), 2) };
}
const shortlist = stocks.filter(s => s.label === 'shortlist').sort((a, b) => b.score - a.score).map(s => s.tk);
const portfolio = minVar(shortlist);

// ---------- market regime ----------
const ixWin = IX.slice(W252), ixr = ixRet.slice(W252 + 1), ixv20 = sd(ixRet.slice(-20)) * Math.sqrt(252), ixvAll = sd(ixRet.slice(1)) * Math.sqrt(252);
const ixSma = p => mean(IX.slice(-p));
const breadthUp = mean(stocks.map(s => s.trend === 'up' ? 1 : 0)), breadthDown = mean(stocks.map(s => s.trend === 'down' ? 1 : 0));
const market = {
  last: r(IX[last], 2), chg20d: r(IX[last] / IX[last - 20] - 1), chg_window: r(IX[last] / ixWin[0] - 1), dd: r(IX[last] / Math.max(...ixWin) - 1),
  chg_full: r(IX[last] / IX[0] - 1), dd_full: r(IX[last] / Math.max(...IX) - 1),
  above50: IX[last] > ixSma(50), above100: IX[last] > ixSma(100), vol20: r(ixv20), vol_all: r(ixvAll),
  breadth_up: r(breadthUp, 2), breadth_down: r(breadthDown, 2), avg_p_stop: r(mean(stocks.map(s => s.forecast.p_stop)), 3),
};
const cautionPts = (market.above50 ? 0 : 1) + (market.above100 ? 0 : 1) + (ixv20 > ixvAll ? 1 : 0) + (breadthDown > 0.5 ? 1 : 0);
market.caution = ['Calm', 'Mild caution', 'Caution', 'High caution', 'High caution'][cautionPts];
market.caution_pts = cautionPts;

// Legacy fields kept for the dashboard and MCP tools.
const weights = FIN.ret.weights;
const outIdx = process.argv.indexOf('--out');
const emit = s => outIdx > 0 ? writeFileSync(process.argv[outIdx + 1], s) : console.log(s);
emit(JSON.stringify({
  version: 2, generated: new Date().toISOString(), as_of: day(T[last]), from: day(T[0]), bars: N, horizon: H, universe: tickers.length, fees: FEES,
  training: { thin: THIN, fold_days: FOLD, min_train_days: MIN_TRAIN_DAYS, folds, rows_total: allRows.length, rows_used_final: finalTrain.length },
  validation, reliable, risk_reliable: riskReliable, weights, edge_found: Object.values(weights).some(w => w > 0),
  importance, risk_importance: riskImportance, market, portfolio, shortlist, features: FEATURES, risk_features: RISKF,
  stocks: stocks.sort((a, b) => b.score - a.score),
}));
