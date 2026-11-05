// Cross-sectional risk + return model for a universe of IDX stocks.
//   1. Features per stock per day (momentum, trend, volatility, tail risk, beta, liquidity, ...)
//   2. Cross-sectional rank normalisation (removes market-wide moves from the signal)
//   3. Four learners (ridge, logistic, gradient-boosted trees, small neural net) plus a
//      transparent factor score, each validated walk-forward with a purge gap
//   4. Ensemble weighted by out-of-sample rank IC (data decides the weights)
//   5. Risk grade, regime, trade levels, bracket statistics and a min-variance allocation
// Usage: node quant/model.mjs quant/data/daily.json [--out quant/data/model.json]   (stdout when --out is omitted)
import { readFileSync, writeFileSync } from 'fs';

const SRC = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const H = 20;          // forecast horizon in trading days
const LOOK = 100;      // longest feature lookback
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

// ---------- small numeric helpers ----------
let seed = 42;
const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1)); };
const quantile = (a, q) => { const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + ((s[Math.ceil(i)] ?? s[lo]) - s[lo]) * (i - lo); };
const rankPct = a => { const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = new Array(a.length); idx.forEach(([, i], k) => { r[i] = a.length > 1 ? k / (a.length - 1) : 0.5; }); return r; };
const spearman = (a, b) => { const ra = rankPct(a), rb = rankPct(b), ma = mean(ra), mb = mean(rb); let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; } return n / Math.sqrt(da * db || 1); };
const r = (x, d = 4) => x == null || !isFinite(x) ? null : Math.round(x * 10 ** d) / 10 ** d;
const day = t => new Date(t * 1000).toISOString().slice(0, 10);
const idxTick = p => p < 200 ? 1 : p < 500 ? 2 : p < 2000 ? 5 : p < 5000 ? 10 : 25;
const snap = (p, mode) => { const t = idxTick(p); return (mode === 'down' ? Math.floor(p / t) : mode === 'up' ? Math.ceil(p / t) : Math.round(p / t)) * t; };

// ---------- align series on the index calendar ----------
const ixBars = SRC.bars[SRC.index];
const T = ixBars.map(b => b.time), N = T.length;
const IX = ixBars.map(b => b.close);
const tickers = Object.keys(SRC.bars).filter(s => s !== SRC.index).map(s => s.replace('IDX:', ''));
const S = {};
for (const tk of tickers) {
  const m = new Map(SRC.bars['IDX:' + tk].map(b => [b.time, b]));
  const get = k => T.map(t => m.get(t)?.[k] ?? null);
  S[tk] = { o: get('open'), h: get('high'), l: get('low'), c: get('close'), v: get('volume') };
}
const ixRet = IX.map((c, i) => i ? Math.log(c / IX[i - 1]) : 0);

// ---------- per-stock indicator series ----------
function series(tk) {
  const { h, l, c, v } = S[tk];
  const ret = c.map((x, i) => i ? Math.log(x / c[i - 1]) : 0);
  const tr = c.map((x, i) => i ? Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1])) : h[i] - l[i]);
  const atr = new Array(N).fill(null); let a = mean(tr.slice(1, 15)); atr[14] = a;
  for (let i = 15; i < N; i++) { a = (a * 13 + tr[i]) / 14; atr[i] = a; }
  const rsi = new Array(N).fill(null); let ag = 0, al = 0;
  for (let i = 1; i <= 14; i++) { const d = c[i] - c[i - 1]; ag += Math.max(d, 0) / 14; al += Math.max(-d, 0) / 14; }
  for (let i = 15; i < N; i++) { const d = c[i] - c[i - 1]; ag = (ag * 13 + Math.max(d, 0)) / 14; al = (al * 13 + Math.max(-d, 0)) / 14; rsi[i] = 100 - 100 / (1 + ag / (al || 1e-9)); }
  const ewv = new Array(N).fill(null); let ev = sd(ret.slice(1, 30)) ** 2;
  for (let i = 30; i < N; i++) { ev = 0.94 * ev + 0.06 * ret[i] ** 2; ewv[i] = Math.sqrt(ev); }
  return { ret, tr, atr, rsi, ewv, tv: c.map((x, i) => x * v[i]) };
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

function feat(tk, i) {
  const { c, h, l, v } = S[tk], s = SER[tk];
  if (i < LOOK || c[i] == null) return null;
  const R = s.ret.slice(i - 59, i + 1), RI = ixRet.slice(i - 59, i + 1);
  const mR = mean(R), mI = mean(RI);
  let cov = 0, vI = 0, vS = 0; for (let k = 0; k < 60; k++) { cov += (R[k] - mR) * (RI[k] - mI); vI += (RI[k] - mI) ** 2; vS += (R[k] - mR) ** 2; }
  const beta = cov / vI, resid = R.map((x, k) => x - beta * RI[k]);
  const sma = p => mean(c.slice(i - p + 1, i + 1));
  const c20 = c.slice(i - 19, i + 1), m20 = mean(c20), s20 = sd(c20);
  const hi60 = Math.max(...h.slice(i - 59, i + 1)), lo60 = Math.min(...l.slice(i - 59, i + 1));
  const r5 = []; for (let k = i - 55; k <= i; k += 5) r5.push(Math.log(c[k] / c[k - 5]));
  const v1 = sd(R) ** 2, v5 = sd(r5) ** 2;
  const neg = R.filter(x => x < 0);
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

// ---------- panel: dates x stocks, rank-normalised features, forward-return labels ----------
const FN = FEATURES.map(f => f[0]);
const panel = []; // { i, rows: [{tk, x[], fwd}] }
for (let i = LOOK; i < N; i++) {
  const rows = [];
  for (const tk of tickers) {
    const f = feat(tk, i); if (!f) continue;
    const fwd = i + H < N ? Math.log(S[tk].c[i + H] / S[tk].c[i]) : null;
    rows.push({ tk, raw: f, fwd });
  }
  for (const k of FN) { const rk = rankPct(rows.map(rw => rw.raw[k])); rows.forEach((rw, j) => { (rw.x ||= [])[FN.indexOf(k)] = rk[j] - 0.5; }); }
  const lab = rows.filter(rw => rw.fwd != null);
  if (lab.length) { const ry = rankPct(lab.map(rw => rw.fwd)); lab.forEach((rw, j) => { rw.y = ry[j] - 0.5; }); }
  panel.push({ i, rows });
}
const labelled = panel.filter(p => p.rows[0]?.y != null);
const P = FN.length;

// ---------- learners ----------
function ridge(X, y, lam = 5) {
  const A = Array.from({ length: P + 1 }, () => new Array(P + 1).fill(0)), b = new Array(P + 1).fill(0);
  for (let n = 0; n < X.length; n++) { const x = [1, ...X[n]]; for (let j = 0; j <= P; j++) { b[j] += x[j] * y[n]; for (let k = 0; k <= P; k++) A[j][k] += x[j] * x[k]; } }
  for (let j = 1; j <= P; j++) A[j][j] += lam;
  for (let c = 0; c <= P; c++) { let p = c; for (let rr = c + 1; rr <= P; rr++) if (Math.abs(A[rr][c]) > Math.abs(A[p][c])) p = rr; [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let rr = 0; rr <= P; rr++) if (rr !== c) { const f = A[rr][c] / A[c][c]; for (let k = c; k <= P; k++) A[rr][k] -= f * A[c][k]; b[rr] -= f * b[c]; } }
  const w = b.map((x, j) => x / A[j][j]);
  return { w, predict: x => w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0) };
}
function logistic(X, y, lam = 0.01, epochs = 300, lr = 0.5) {
  const w = new Array(P + 1).fill(0), t = y.map(v => v > 0 ? 1 : 0);
  for (let e = 0; e < epochs; e++) { const g = new Array(P + 1).fill(0);
    for (let n = 0; n < X.length; n++) { const z = w[0] + X[n].reduce((s, v, j) => s + v * w[j + 1], 0), p = 1 / (1 + Math.exp(-z)), d = p - t[n]; g[0] += d; for (let j = 0; j < P; j++) g[j + 1] += d * X[n][j]; }
    for (let j = 0; j <= P; j++) w[j] -= lr * (g[j] / X.length + (j ? lam * w[j] : 0)); }
  return { w, predict: x => 1 / (1 + Math.exp(-(w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0)))) - 0.5 };
}
function gbt(X, y, { trees = 120, depth = 3, lr = 0.05, sub = 0.7, minLeaf = 40 } = {}) {
  const cuts = [-0.3, -0.15, 0, 0.15, 0.3], base = mean(y), F = new Array(y.length).fill(base), ens = [], gain = new Array(P).fill(0);
  function build(idx, res, d) {
    const val = mean(idx.map(n => res[n]));
    if (d === 0 || idx.length < 2 * minLeaf) return { val };
    let best = null; const tot = idx.reduce((s, n) => s + res[n], 0);
    for (let j = 0; j < P; j++) for (const ct of cuts) { let sl = 0, nl = 0; for (const n of idx) if (X[n][j] <= ct) { sl += res[n]; nl++; }
      const nr = idx.length - nl; if (nl < minLeaf || nr < minLeaf) continue; const g = sl * sl / nl + (tot - sl) ** 2 / nr - tot * tot / idx.length;
      if (!best || g > best.g) best = { j, ct, g }; }
    if (!best) return { val };
    gain[best.j] += best.g;
    return { j: best.j, ct: best.ct, L: build(idx.filter(n => X[n][best.j] <= best.ct), res, d - 1), R: build(idx.filter(n => X[n][best.j] > best.ct), res, d - 1) };
  }
  const pt = (nd, x) => nd.val !== undefined ? nd.val : pt(x[nd.j] <= nd.ct ? nd.L : nd.R, x);
  for (let k = 0; k < trees; k++) { const res = y.map((v, n) => v - F[n]); const idx = y.map((_, n) => n).filter(() => rand() < sub); const tr = build(idx, res, depth); ens.push(tr); for (let n = 0; n < y.length; n++) F[n] += lr * pt(tr, X[n]); }
  return { gain, predict: x => base + lr * ens.reduce((s, tr) => s + pt(tr, x), 0) };
}
function mlp(X, y, { hid = [16, 8], epochs = 40, lr = 0.003, l2 = 1e-4, batch = 128 } = {}) {
  const sizes = [P, ...hid, 1], L = sizes.length - 1;
  const W = [], B = [], mW = [], vW = [], mB = [], vB = []; let step = 0;
  for (let l = 0; l < L; l++) { const s = Math.sqrt(2 / sizes[l]); W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => (rand() * 2 - 1) * s)));
    B.push(new Array(sizes[l + 1]).fill(0)); mW.push(W[l].map(rw => rw.map(() => 0))); vW.push(W[l].map(rw => rw.map(() => 0))); mB.push(B[l].map(() => 0)); vB.push(B[l].map(() => 0)); }
  const fwd = x => { const acts = [x]; for (let l = 0; l < L; l++) { const a = W[l].map((rw, o) => rw.reduce((s, w, k) => s + w * acts[l][k], B[l][o])); acts.push(l < L - 1 ? a.map(v => Math.max(0, v)) : a); } return acts; };
  const order = y.map((_, n) => n);
  for (let e = 0; e < epochs; e++) {
    for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(rand() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
    for (let s0 = 0; s0 < order.length; s0 += batch) {
      const gW = W.map(m => m.map(rw => rw.map(() => 0))), gB = B.map(bb => bb.map(() => 0)), bs = Math.min(batch, order.length - s0);
      for (let q = s0; q < s0 + bs; q++) { const n = order[q], acts = fwd(X[n]); let delta = [acts[L][0] - y[n]];
        for (let l = L - 1; l >= 0; l--) { for (let o = 0; o < delta.length; o++) { gB[l][o] += delta[o]; for (let k = 0; k < acts[l].length; k++) gW[l][o][k] += delta[o] * acts[l][k]; }
          if (l) delta = acts[l].map((a, k) => a > 0 ? delta.reduce((s, d, o) => s + d * W[l][o][k], 0) : 0); } }
      step++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** step, c2 = 1 - b2 ** step;
      for (let l = 0; l < L; l++) { for (let o = 0; o < W[l].length; o++) { for (let k = 0; k < W[l][o].length; k++) { const g = gW[l][o][k] / bs + l2 * W[l][o][k];
        mW[l][o][k] = b1 * mW[l][o][k] + (1 - b1) * g; vW[l][o][k] = b2 * vW[l][o][k] + (1 - b2) * g * g; W[l][o][k] -= lr * (mW[l][o][k] / c1) / (Math.sqrt(vW[l][o][k] / c2) + 1e-8); }
        const g = gB[l][o] / bs; mB[l][o] = b1 * mB[l][o] + (1 - b1) * g; vB[l][o] = b2 * vB[l][o] + (1 - b2) * g * g; B[l][o] -= lr * (mB[l][o] / c1) / (Math.sqrt(vB[l][o] / c2) + 1e-8); } }
    }
  }
  return { predict: x => fwd(x)[L][0] };
}
// Transparent rule-based factor score: momentum + relative strength + trend - volatility - tail risk.
const FACTOR = { mom60: 1, mom100x5: 1, rs60: 1, sma50d: 1, vol60: -1, downdev60: -1, dd60: 1 };
const factorModel = () => ({ predict: x => Object.entries(FACTOR).reduce((s, [k, wt]) => s + wt * x[FN.indexOf(k)], 0) / Object.keys(FACTOR).length });

const LEARNERS = {
  ridge: { label: 'Ridge regression', fit: (X, y) => ridge(X, y) },
  logit: { label: 'Logistic regression', fit: (X, y) => logistic(X, y) },
  gbt: { label: 'Gradient-boosted trees', fit: (X, y) => gbt(X, y) },
  nn: { label: 'Neural network (26-16-8-1)', fit: (X, y) => mlp(X, y) },
  factor: { label: 'Rule-based factor score', fit: () => factorModel() },
};

// ---------- walk-forward validation with a purge gap of H days ----------
const MIN_TRAIN = 60, STEP = H;
const oos = Object.fromEntries(Object.keys(LEARNERS).map(k => [k, []])); // per test date: {ic, spread, i}
const oosPred = []; // for ensemble evaluation
for (let t0 = MIN_TRAIN + H; t0 < labelled.length; t0 += STEP) {
  const train = labelled.slice(0, t0 - H), test = labelled.slice(t0, Math.min(t0 + STEP, labelled.length));
  const X = train.flatMap(p => p.rows.map(rw => rw.x)), y = train.flatMap(p => p.rows.map(rw => rw.y));
  const models = Object.fromEntries(Object.entries(LEARNERS).map(([k, L]) => [k, L.fit(X, y)]));
  for (const p of test) {
    const fw = p.rows.map(rw => rw.fwd), rec = { i: p.i, preds: {} };
    for (const [k, m] of Object.entries(models)) {
      const pr = p.rows.map(rw => m.predict(rw.x)); rec.preds[k] = pr;
      const rk = rankPct(pr), top = fw.filter((_, j) => rk[j] >= 0.8), bot = fw.filter((_, j) => rk[j] <= 0.2);
      oos[k].push({ i: p.i, ic: spearman(pr, fw), spread: mean(top) - mean(bot) });
    }
    rec.fwd = fw; oosPred.push(rec);
  }
}
function summarise(list) {
  const ics = list.map(o => o.ic), non = list.filter((_, k) => k % H === 0).map(o => o.ic);
  const m = mean(ics), tstat = non.length > 1 ? mean(non) / (sd(non) / Math.sqrt(non.length)) : null;
  return { ic: r(m), ic_hit: r(ics.filter(x => x > 0).length / ics.length), ic_t_nonoverlap: r(tstat, 2), n_independent: non.length,
    spread20d: r(mean(list.map(o => o.spread))), test_days: list.length };
}
const validation = Object.fromEntries(Object.entries(oos).map(([k, v]) => [k, { label: LEARNERS[k].label, ...summarise(v) }]));

// Ensemble weights from out-of-sample IC (only positive skill counts).
let wts = Object.fromEntries(Object.entries(validation).map(([k, v]) => [k, Math.max(0, v.ic)]));
const wsum = Object.values(wts).reduce((a, b) => a + b, 0);
const edgeFound = wsum > 0;
wts = Object.fromEntries(Object.entries(wts).map(([k, v]) => [k, edgeFound ? v / wsum : 1 / Object.keys(wts).length]));
// Note: weights are chosen on the same OOS window they are evaluated on, so the ensemble line is optimistic.
const ensList = oosPred.map(rec => {
  const z = rec.fwd.map((_, j) => Object.entries(wts).reduce((s, [k, w]) => s + w * rankPct(rec.preds[k])[j], 0));
  const rk = rankPct(z), fw = rec.fwd;
  return { i: rec.i, ic: spearman(z, fw), spread: mean(fw.filter((_, j) => rk[j] >= 0.8)) - mean(fw.filter((_, j) => rk[j] <= 0.2)) };
});
validation.ensemble = { label: 'Ensemble (IC-weighted)', ...summarise(ensList) };
// A "reliable" edge needs the ensemble to be positive with t >= 2 over at least 8 non-overlapping periods.
const reliable = validation.ensemble.ic > 0 && validation.ensemble.ic_t_nonoverlap >= 2 && validation.ensemble.n_independent >= 8;

// ---------- final fit on all labelled data, score the latest day ----------
const Xall = labelled.flatMap(p => p.rows.map(rw => rw.x)), yall = labelled.flatMap(p => p.rows.map(rw => rw.y));
const finals = Object.fromEntries(Object.entries(LEARNERS).map(([k, L]) => [k, L.fit(Xall, yall)]));
const latest = panel.at(-1);
const modelRanks = Object.fromEntries(Object.entries(finals).map(([k, m]) => [k, rankPct(latest.rows.map(rw => m.predict(rw.x)))]));
const ensScore = latest.rows.map((_, j) => Object.entries(wts).reduce((s, [k, w]) => s + w * modelRanks[k][j], 0));
const ensRank = rankPct(ensScore);
// Agreement among the models that earned weight: 1 = identical ranks, 0 = as spread as random ranks (sd of uniform = 0.29).
const voters = Object.keys(wts).filter(k => wts[k] > 0);
const agreement = latest.rows.map((_, j) => voters.length < 2 ? 1 : 1 - sd(voters.map(k => modelRanks[k][j])) / 0.29);

// Feature importance: GBT split gain + ridge coefficients (both on all data).
const gGain = finals.gbt.gain, gTot = gGain.reduce((a, b) => a + b, 0) || 1;
const importance = FEATURES.map(([k, label], j) => ({ key: k, label, gbt: r(gGain[j] / gTot), ridge: r(finals.ridge.w[j + 1]), logit: r(finals.logit.w[j + 1]) }))
  .sort((a, b) => b.gbt - a.gbt);

// ---------- per-stock risk, levels, brackets ----------
const last = N - 1;
function bracketStats(tk, slM = 1.5, rr = 1.5) {
  const { h, l, c } = S[tk], atr = SER[tk].atr; let w = 0, ls = 0, op = 0;
  for (let i = 15; i < N - 1; i++) { const e = c[i], rk = slM * atr[i], sl = e - rk, tp = e + rr * rk; let res = null;
    for (let j = i + 1; j < Math.min(N, i + 1 + H); j++) { if (l[j] <= sl) { res = 0; break; } if (h[j] >= tp) { res = 1; break; } }
    if (res === 1) w++; else if (res === 0) ls++; else op++; }
  return { p_tp_first: r(w / (w + ls || 1)), breakeven: r(1 / (1 + rr)), expectancy_R: r((w / (w + ls || 1)) * rr - (ls / (w + ls || 1))), trials: w + ls + op };
}
function swingLevels(tk) {
  const { h, l, c } = S[tk], atr = SER[tk].atr[last], px = c[last], piv = [];
  for (let i = 5; i < N - 5; i++) { const wh = h.slice(i - 5, i + 6), wl = l.slice(i - 5, i + 6); if (h[i] === Math.max(...wh)) piv.push(h[i]); if (l[i] === Math.min(...wl)) piv.push(l[i]); }
  const cl = []; for (const p of piv.sort((a, b) => a - b)) { const k = cl.find(q => Math.abs(q.p - p) <= atr * 0.5); if (k) { k.m.push(p); k.p = mean(k.m); } else cl.push({ p, m: [p] }); }
  const lv = cl.map(q => ({ price: snap(q.p), touches: q.m.length }));
  return { res: lv.filter(q => q.price > px * 1.005).sort((a, b) => a.price - b.price).slice(0, 3), sup: lv.filter(q => q.price < px * 0.995).sort((a, b) => b.price - a.price).slice(0, 3) };
}
function maxDD(c) { let pk = c[0], m = 0; for (const x of c) { pk = Math.max(pk, x); m = Math.min(m, x / pk - 1); } return m; }

const stocks = latest.rows.map((rw, j) => {
  const tk = rw.tk, f = rw.raw, c = S[tk].c, px = c[last], s = SER[tk];
  const rets = s.ret.slice(1).map(x => Math.exp(x) - 1);
  const var95 = -quantile(rets, 0.05), cvar95 = -mean(rets.filter(x => x <= -var95));
  const atr = s.atr[last], stop = snap(px - 1.5 * atr, 'down'), risk = px - stop;
  const lv = swingLevels(tk);
  return {
    tk, sector: SECTORS[tk] || 'Other', price: px, tick: idxTick(px),
    chg1d: r(c[last] / c[last - 1] - 1), chg20d: r(Math.exp(f.mom20) - 1), chg60d: r(Math.exp(f.mom60) - 1),
    spark: c.slice(-60),
    score: Math.round(ensRank[j] * 100), agreement: r(Math.max(0, agreement[j]), 2),
    model_ranks: Object.fromEntries(Object.keys(finals).map(k => [k, Math.round(modelRanks[k][j] * 100)])),
    trend: f.sma20d > 0 && f.sma50d > 0 && f.sma100d > 0 ? 'up' : f.sma20d < 0 && f.sma50d < 0 && f.sma100d < 0 ? 'down' : 'mixed',
    rsi: r(f.rsi, 1), atr: r(atr, 1), atr_pct: r(f.atrp), vol_ann: r(f.vol60 * Math.sqrt(252)), ewvol_ann: r(f.ewvol * Math.sqrt(252)),
    var95: r(var95), cvar95: r(cvar95), maxdd: r(maxDD(c)), dd_now: r(px / Math.max(...c) - 1),
    beta: r(f.beta60, 2), corr: r(f.corr60, 2), skew: r(f.skew60, 2), kurt: r(mean(rets.map(x => ((x - mean(rets)) / sd(rets)) ** 4)) - 3, 1),
    liq_bn: r(Math.exp(f.liq) / 1e9, 1),
    plan: { entry: px, stop, risk, risk_pct: r(risk / px), tp1: snap(px + 1.5 * risk), tp2: snap(px + 2.5 * risk), lot_risk: risk * 100,
      // Break-even price after typical IDX costs: buy fee 0.15%, sell fee 0.25% (includes 0.1% sales tax), rounded up to a valid tick.
      bep: snap(px * (1 + FEES.buy) / (1 - FEES.sell), 'up') },
    bracket: bracketStats(tk),
    levels: lv,
    raw: Object.fromEntries(FN.map(k => [k, r(f[k])])),
  };
});

// Risk grade: average cross-sectional percentile of vol, CVaR, drawdown, beta, illiquidity, kurtosis.
const riskCols = [['vol_ann', 1], ['cvar95', 1], ['maxdd', -1], ['beta', 1], ['liq_bn', -1], ['kurt', 1]];
const riskPct = riskCols.map(([k, sgn]) => rankPct(stocks.map(s => sgn * s[k])));
stocks.forEach((s, j) => { const rp = mean(riskPct.map(col => col[j])); s.risk_pct = r(rp, 2); s.risk_grade = 'ABCDE'[Math.min(4, Math.floor(rp * 5))]; });
// Minimum liquidity for a beginner: Rp 10 bn traded per day on average.
const LIQ_MIN = 10;
stocks.forEach(s => {
  s.liquid = s.liq_bn >= LIQ_MIN;
  s.label = s.score >= 70 && 'ABC'.includes(s.risk_grade) && s.liquid && s.bracket.expectancy_R > -0.1 ? 'shortlist'
    : s.score <= 30 || s.risk_grade === 'E' || !s.liquid ? 'avoid' : 'watch';
  const why = [];
  if (s.score >= 70) why.push('ranked in the top 30% by the models'); else if (s.score <= 30) why.push('ranked in the bottom 30% by the models');
  if (s.trend === 'up') why.push('price above its 20, 50 and 100-day averages'); if (s.trend === 'down') why.push('price below its 20, 50 and 100-day averages');
  if ('AB'.includes(s.risk_grade)) why.push('lower risk than most of the list'); if ('DE'.includes(s.risk_grade)) why.push('higher risk than most of the list');
  if (!s.liquid) why.push(`thin trading (Rp${s.liq_bn} bn a day)`);
  if (s.bracket.expectancy_R > 0.05) why.push('its stop/target bracket would have paid off over the past year');
  if (s.bracket.expectancy_R < -0.1) why.push('its stop/target bracket lost money over the past year');
  s.why = why;
});

// ---------- minimum-variance allocation across the shortlist (long only, max 35% each) ----------
function minVar(names) {
  if (names.length < 2) return names.map(n => ({ tk: n, w: 1 }));
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

// ---------- market regime (IHSG) ----------
const ixr = ixRet.slice(1), ixv20 = sd(ixr.slice(-20)) * Math.sqrt(252), ixv250 = sd(ixr) * Math.sqrt(252);
const ixSma = p => mean(IX.slice(-p));
const breadth = mean(stocks.map(s => s.trend === 'up' ? 1 : 0)), breadthDown = mean(stocks.map(s => s.trend === 'down' ? 1 : 0));
const market = {
  last: r(IX[last], 2), chg20d: r(IX[last] / IX[last - 20] - 1), chg_window: r(IX[last] / IX[0] - 1), dd: r(IX[last] / Math.max(...IX) - 1),
  above50: IX[last] > ixSma(50), above100: IX[last] > ixSma(100), vol20: r(ixv20), vol_all: r(ixv250),
  breadth_up: r(breadth, 2), breadth_down: r(breadthDown, 2),
};
const cautionPts = (market.above50 ? 0 : 1) + (market.above100 ? 0 : 1) + (ixv20 > ixv250 ? 1 : 0) + (breadthDown > 0.5 ? 1 : 0);
market.caution = ['Calm', 'Mild caution', 'Caution', 'High caution', 'High caution'][cautionPts];
market.caution_pts = cautionPts;

const outIdx = process.argv.indexOf('--out');
const emit = s => outIdx > 0 ? writeFileSync(process.argv[outIdx + 1], s) : console.log(s);
emit(JSON.stringify({
  generated: new Date().toISOString(), as_of: day(T[last]), from: day(T[0]), bars: N, horizon: H, universe: tickers.length, fees: FEES,
  validation, weights: Object.fromEntries(Object.entries(wts).map(([k, v]) => [k, r(v, 3)])), edge_found: edgeFound, reliable,
  importance, market, portfolio, shortlist, features: FEATURES,
  stocks: stocks.sort((a, b) => b.score - a.score),
}));
