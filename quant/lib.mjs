// Shared numerics for the risk engine: statistics, GARCH(1,1), Gaussian HMM, EVT (GPD), liquidity,
// logistic regression and a small deep network. Plain JavaScript, seeded, no dependencies.

// ---------- basic statistics ----------
let seed = 42;
export const reseed = s => { seed = s >>> 0; };
export const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
export const randn = () => { let u = 0, v = 0; while (u === 0) u = rand(); while (v === 0) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
export const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
export const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1)); };
export const quantile = (a, q) => { if (!a.length) return NaN; const s = Float64Array.from(a).sort(); const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + ((s[Math.ceil(i)] ?? s[lo]) - s[lo]) * (i - lo); };
export const rankPct = a => { const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = new Array(a.length); idx.forEach(([, i], k) => { r[i] = a.length > 1 ? k / (a.length - 1) : 0.5; }); return r; };
export const pearson = (a, b) => { const ma = mean(a), mb = mean(b); let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; } return n / Math.sqrt(da * db || 1e-12); };
export const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
export const logit = p => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
export const r = (x, d = 4) => x == null || !isFinite(x) ? null : Math.round(x * 10 ** d) / 10 ** d;
export const day = t => new Date(t * 1000).toISOString().slice(0, 10);
export const idxTick = p => p < 200 ? 1 : p < 500 ? 2 : p < 2000 ? 5 : p < 5000 ? 10 : 25;
export const snap = (p, mode) => { const t = idxTick(p); return (mode === 'down' ? Math.floor(p / t) : mode === 'up' ? Math.ceil(p / t) : Math.round(p / t)) * t; };
export function auc(p, y) { const pos = y.filter(v => v === 1).length, neg = y.length - pos; if (!pos || !neg) return 0.5; const rk = rankPct(p).map(v => v * (y.length - 1) + 1); let s = 0; y.forEach((v, i) => { if (v === 1) s += rk[i]; }); return (s - pos * (pos + 1) / 2) / (pos * neg); }
export const brier = (p, y) => mean(p.map((v, i) => (v - y[i]) ** 2));
export const logloss = (p, y) => -mean(p.map((v, i) => y[i] ? Math.log(Math.max(1e-9, v)) : Math.log(Math.max(1e-9, 1 - v))));
export function calibration(p, y, edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]) {
  return edges.slice(0, -1).map((lo, k) => { const ks = []; p.forEach((v, i) => { if (v >= lo && v < edges[k + 1]) ks.push(i); });
    return { from: lo, to: Math.min(1, edges[k + 1]), n: ks.length, predicted: r(mean(ks.map(i => p[i])), 3), actual: r(mean(ks.map(i => y[i])), 3) }; }).filter(b => b.n > 0);
}
// Complementary error function (Abramowitz-Stegun 7.1.26), for chi-square(1) p-values.
export function erfc(x) { const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const v = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? v : 2 - v; }
// Kupiec proportion-of-failures test for a VaR at coverage p.
export function kupiec(exceed, n, p) {
  const x = exceed, phat = x / n;
  const ll0 = (n - x) * Math.log(1 - p) + x * Math.log(p);
  const ll1 = (n - x) * Math.log(Math.max(1e-12, 1 - phat)) + (x ? x * Math.log(phat) : 0);
  const lr = -2 * (ll0 - ll1);
  return { exceedances: x, n, expected_rate: p, actual_rate: r(phat, 5), lr: r(lr, 3), p_value: r(erfc(Math.sqrt(Math.max(0, lr) / 2)), 4) };
}

// ---------- optimisation ----------
// Nelder-Mead minimiser for small unconstrained problems.
export function nelderMead(f, x0, { step = 0.5, iters = 300, tol = 1e-8 } = {}) {
  const n = x0.length; let pts = [x0.slice()];
  for (let i = 0; i < n; i++) { const p = x0.slice(); p[i] += step; pts.push(p); }
  let vals = pts.map(f);
  for (let it = 0; it < iters; it++) {
    const ord = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
    pts = ord.map(i => pts[i]); vals = ord.map(i => vals[i]);
    if (Math.abs(vals[n] - vals[0]) < tol) break;
    const c = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += pts[i][j] / n;
    const at = t => c.map((cj, j) => cj + t * (pts[n][j] - cj));
    const xr = at(-1), fr = f(xr);
    if (fr < vals[0]) { const xe = at(-2), fe = f(xe); if (fe < fr) { pts[n] = xe; vals[n] = fe; } else { pts[n] = xr; vals[n] = fr; } }
    else if (fr < vals[n - 1]) { pts[n] = xr; vals[n] = fr; }
    else { const xc = at(0.5), fc = f(xc);
      if (fc < vals[n]) { pts[n] = xc; vals[n] = fc; }
      else { for (let i = 1; i <= n; i++) { pts[i] = pts[i].map((v, j) => pts[0][j] + 0.5 * (v - pts[0][j])); vals[i] = f(pts[i]); } } }
  }
  const b = vals.indexOf(Math.min(...vals)); return { x: pts[b], f: vals[b] };
}

// ---------- GARCH(1,1) with variance targeting ----------
// Parameters: persistence P = alpha + beta in (0.5, 0.999), alpha share of P; omega = var * (1 - P).
const garchParams = (x, v) => { const P = 0.5 + 0.499 * sigmoid(x[0]), a = P * 0.5 * sigmoid(x[1]); return { omega: v * (1 - P), alpha: a, beta: P - a, persistence: P, uncond: v }; };
// sig2[t] is the conditional variance of ret[t] given information up to t-1; next = variance for t+1.
export function garchFilter(ret, g) { const sig2 = new Float64Array(ret.length); let s2 = g.uncond;
  for (let t = 0; t < ret.length; t++) { sig2[t] = s2; s2 = g.omega + g.alpha * ret[t] ** 2 + g.beta * s2; } return { sig2, next: s2 }; }
export function garchFit(ret) {
  const m = mean(ret), x = ret.map(v => v - m), v = mean(x.map(e => e * e));
  const nll = p => { const g = garchParams(p, v); let s2 = v, ll = 0; for (const e of x) { ll += Math.log(s2) + e * e / s2; s2 = g.omega + g.alpha * e * e + g.beta * s2; } return ll; };
  const best = nelderMead(nll, [2, -1], { step: 1, iters: 200 });
  return { ...garchParams(best.x, v), mu: m };
}
// Expected variance h days ahead and the h-day total variance.
export const garchTerm = (g, next, h) => { let tot = 0; for (let k = 0; k < h; k++) tot += g.uncond + g.persistence ** k * (next - g.uncond); return tot; };

// ---------- EVT: generalised Pareto tail by probability-weighted moments (Hosking & Wallis) ----------
export function gpdFit(losses, uq = 0.9) {
  const u = quantile(losses, uq), y = losses.filter(l => l > u).map(l => l - u).sort((a, b) => a - b), n = y.length;
  if (n < 20) return null;
  const a0 = mean(y); let a1 = 0; y.forEach((v, j) => { a1 += v * (n - 1 - j) / (n - 1); }); a1 /= n;
  let xi = 2 - a0 / (a0 - 2 * a1), beta = 2 * a0 * a1 / (a0 - 2 * a1);
  xi = Math.max(-0.5, Math.min(0.8, xi)); if (!(beta > 0)) beta = a0;
  return { u, xi, beta, nu: n, n: losses.length };
}
// VaR and expected shortfall of the loss at level q (e.g. 0.99) from a GPD tail fit.
export function gpdRisk(fit, q) {
  const { u, xi, beta, nu, n } = fit, tail = (n / nu) * (1 - q);
  const varq = Math.abs(xi) < 1e-6 ? u - beta * Math.log(tail) : u + (beta / xi) * (tail ** -xi - 1);
  return { var: varq, es: (varq + beta - xi * u) / (1 - xi) };
}
// Cornish-Fisher quantile adjustment for skewness and excess kurtosis.
export function cornishFisher(z, skew, exkurt) { return z + (z * z - 1) * skew / 6 + (z ** 3 - 3 * z) * exkurt / 24 - (2 * z ** 3 - 5 * z) * skew * skew / 36; }

// ---------- two-state Gaussian hidden Markov model (Baum-Welch with scaling) ----------
export function hmmFit(x, iters = 40) {
  const s = sd(x), K = 2, n = x.length;
  let mu = [mean(x), mean(x)], sg = [s * 0.7, s * 1.7], pi = [0.7, 0.3], A = [[0.98, 0.02], [0.05, 0.95]];
  const dens = (v, k) => Math.exp(-0.5 * ((v - mu[k]) / sg[k]) ** 2) / (sg[k] * 2.5066282746);
  let alpha, c;
  for (let it = 0; it < iters; it++) {
    alpha = Array.from({ length: n }, () => [0, 0]); c = new Float64Array(n);
    for (let k = 0; k < K; k++) alpha[0][k] = pi[k] * dens(x[0], k); c[0] = alpha[0][0] + alpha[0][1] || 1e-300; alpha[0] = alpha[0].map(v => v / c[0]);
    for (let t = 1; t < n; t++) { for (let k = 0; k < K; k++) alpha[t][k] = (alpha[t - 1][0] * A[0][k] + alpha[t - 1][1] * A[1][k]) * dens(x[t], k);
      c[t] = alpha[t][0] + alpha[t][1] || 1e-300; alpha[t] = alpha[t].map(v => v / c[t]); }
    const beta = Array.from({ length: n }, () => [1, 1]);
    for (let t = n - 2; t >= 0; t--) for (let k = 0; k < K; k++) beta[t][k] = (A[k][0] * dens(x[t + 1], 0) * beta[t + 1][0] + A[k][1] * dens(x[t + 1], 1) * beta[t + 1][1]) / c[t + 1];
    const gam = alpha.map((a, t) => { const g = [a[0] * beta[t][0], a[1] * beta[t][1]], s0 = g[0] + g[1] || 1e-300; return [g[0] / s0, g[1] / s0]; });
    const xi = [[0, 0], [0, 0]];
    for (let t = 0; t < n - 1; t++) { let tot = 0; const m = [[0, 0], [0, 0]];
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { m[i][j] = alpha[t][i] * A[i][j] * dens(x[t + 1], j) * beta[t + 1][j]; tot += m[i][j]; }
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) xi[i][j] += m[i][j] / (tot || 1e-300); }
    pi = gam[0];
    for (let i = 0; i < K; i++) { const row = xi[i][0] + xi[i][1] || 1; A[i] = [xi[i][0] / row, xi[i][1] / row]; }
    for (let k = 0; k < K; k++) { const w = gam.map(g => g[k]), ws = w.reduce((a, b) => a + b, 0) || 1;
      mu[k] = w.reduce((s0, wt, t) => s0 + wt * x[t], 0) / ws; sg[k] = Math.max(1e-5, Math.sqrt(w.reduce((s0, wt, t) => s0 + wt * (x[t] - mu[k]) ** 2, 0) / ws)); }
  }
  if (sg[0] > sg[1]) { mu = [mu[1], mu[0]]; sg = [sg[1], sg[0]]; A = [[A[1][1], A[1][0]], [A[0][1], A[0][0]]]; pi = [pi[1], pi[0]]; }
  return { mu, sg, A, pi };
}
// Filtered probability of the high-volatility state for each observation.
export function hmmFilter(h, x) {
  const dens = (v, k) => Math.exp(-0.5 * ((v - h.mu[k]) / h.sg[k]) ** 2) / (h.sg[k] * 2.5066282746);
  let p = h.pi.slice(); const out = new Float64Array(x.length);
  for (let t = 0; t < x.length; t++) { const pr = [p[0] * h.A[0][0] + p[1] * h.A[1][0], p[0] * h.A[0][1] + p[1] * h.A[1][1]];
    const a = [pr[0] * dens(x[t], 0), pr[1] * dens(x[t], 1)], s0 = a[0] + a[1] || 1e-300; p = [a[0] / s0, a[1] / s0]; out[t] = p[1]; }
  return out;
}

// ---------- liquidity ----------
// Corwin-Schultz (2012) bid-ask spread estimator from consecutive daily highs and lows.
export function corwinSchultz(h, l, from, to) {
  const k = 3 - 2 * Math.SQRT2, out = [];
  for (let t = from; t < to; t++) { if (!(h[t] > 0 && l[t] > 0 && h[t + 1] > 0 && l[t + 1] > 0)) continue;
    const b = Math.log(h[t] / l[t]) ** 2 + Math.log(h[t + 1] / l[t + 1]) ** 2, g = Math.log(Math.max(h[t], h[t + 1]) / Math.min(l[t], l[t + 1])) ** 2;
    const a = (Math.sqrt(2 * b) - Math.sqrt(b)) / k - Math.sqrt(g / k); out.push(Math.max(0, 2 * (Math.exp(a) - 1) / (1 + Math.exp(a)))); }
  return mean(out);
}

// ---------- logistic regression (L2, optional per-row offsets) ----------
export function logistic(X, y, lam = 0.01, epochs = 300, lr = 0.5, off = null) {
  const P = X[0].length, w = new Array(P + 1).fill(0); w[0] = off ? 0 : logit(mean(y));
  for (let e = 0; e < epochs; e++) { const g = new Array(P + 1).fill(0);
    for (let n = 0; n < X.length; n++) { let z = w[0] + (off ? off[n] : 0); for (let j = 0; j < P; j++) z += w[j + 1] * X[n][j]; const d = sigmoid(z) - y[n]; g[0] += d; for (let j = 0; j < P; j++) g[j + 1] += d * X[n][j]; }
    for (let j = 0; j <= P; j++) w[j] -= lr * (g[j] / X.length + (j ? lam * w[j] : 0)); }
  return { w, predict: (x, o = 0) => sigmoid(o + w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0)) };
}
export function standardiser(X) { const k = X[0].length, mu = [], sg = [];
  for (let j = 0; j < k; j++) { const col = X.map(x => x[j]); mu.push(mean(col)); sg.push(sd(col) || 1); }
  return x => x.map((v, j) => Math.max(-5, Math.min(5, (v - mu[j]) / sg[j]))); }

// ---------- deep network for yes/no targets ----------
// Fully connected ReLU layers with inverted dropout, L2, Adam, binary cross-entropy and early stopping.
export function deepNet(X, y, { val, hidden = [32, 16, 8], dropout = 0.2, l2 = 1e-4, lr = 1e-3, batch = 128, maxEpochs = 25, patience = 3 } = {}) {
  const sizes = [X[0].length, ...hidden, 1], L = sizes.length - 1;
  const W = [], B = [], mW = [], vW = [], mB = [], vB = []; let step = 0;
  for (let l = 0; l < L; l++) { const s = Math.sqrt(2 / sizes[l]);
    W.push(Array.from({ length: sizes[l + 1] }, () => Float64Array.from({ length: sizes[l] }, () => randn() * s)));
    B.push(new Float64Array(sizes[l + 1])); mW.push(W[l].map(rw => new Float64Array(rw.length))); vW.push(W[l].map(rw => new Float64Array(rw.length)));
    mB.push(new Float64Array(sizes[l + 1])); vB.push(new Float64Array(sizes[l + 1])); }
  B[L - 1][0] = logit(mean(y));
  const fwd = (x, train) => { const acts = [x], masks = [];
    for (let l = 0; l < L; l++) { const a = new Float64Array(sizes[l + 1]), mk = new Float64Array(sizes[l + 1]).fill(1);
      for (let o = 0; o < a.length; o++) { let s0 = B[l][o]; const w = W[l][o], p = acts[l]; for (let k = 0; k < w.length; k++) s0 += w[k] * p[k];
        if (l < L - 1) { s0 = Math.max(0, s0); if (train && dropout) { if (rand() < dropout) { mk[o] = 0; s0 = 0; } else { mk[o] = 1 / (1 - dropout); s0 *= mk[o]; } } }
        a[o] = s0; }
      acts.push(a); masks.push(mk); }
    return { acts, masks }; };
  const predict = x => sigmoid(fwd(x, false).acts[L][0]);
  const snapW = () => ({ W: W.map(m => m.map(rw => Float64Array.from(rw))), B: B.map(b => Float64Array.from(b)) });
  let best = val ? logloss(val.X.map(predict), val.y) : Infinity, bestW = snapW(), since = 0, epochs = 0;
  const order = y.map((_, n) => n);
  for (let e = 0; e < maxEpochs; e++) {
    epochs = e + 1;
    for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(rand() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
    for (let s0 = 0; s0 < order.length; s0 += batch) {
      const gW = W.map(m => m.map(rw => new Float64Array(rw.length))), gB = B.map(b => new Float64Array(b.length)), bs = Math.min(batch, order.length - s0);
      for (let q = s0; q < s0 + bs; q++) { const n = order[q], { acts, masks } = fwd(X[n], true); let delta = [sigmoid(acts[L][0]) - y[n]];
        for (let l = L - 1; l >= 0; l--) { for (let o = 0; o < delta.length; o++) { gB[l][o] += delta[o]; const gw = gW[l][o], p = acts[l]; for (let k = 0; k < p.length; k++) gw[k] += delta[o] * p[k]; }
          if (l) { const nd = new Float64Array(acts[l].length); for (let k = 0; k < nd.length; k++) if (acts[l][k] > 0) { let s1 = 0; for (let o = 0; o < delta.length; o++) s1 += delta[o] * W[l][o][k]; nd[k] = s1 * masks[l - 1][k]; } delta = nd; } } }
      step++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** step, c2 = 1 - b2 ** step;
      for (let l = 0; l < L; l++) for (let o = 0; o < W[l].length; o++) {
        for (let k = 0; k < W[l][o].length; k++) { const g = gW[l][o][k] / bs + l2 * W[l][o][k]; mW[l][o][k] = b1 * mW[l][o][k] + (1 - b1) * g; vW[l][o][k] = b2 * vW[l][o][k] + (1 - b2) * g * g; W[l][o][k] -= lr * (mW[l][o][k] / c1) / (Math.sqrt(vW[l][o][k] / c2) + 1e-8); }
        const g = gB[l][o] / bs; mB[l][o] = b1 * mB[l][o] + (1 - b1) * g; vB[l][o] = b2 * vB[l][o] + (1 - b2) * g * g; B[l][o] -= lr * (mB[l][o] / c1) / (Math.sqrt(vB[l][o] / c2) + 1e-8); }
    }
    if (val) { const vl = logloss(val.X.map(predict), val.y); if (vl < best - 1e-6) { best = vl; bestW = snapW(); since = 0; } else if (++since >= patience) break; }
  }
  if (val) { bestW.W.forEach((m, l) => m.forEach((rw, o) => W[l][o].set(rw))); bestW.B.forEach((b, l) => B[l].set(b)); }
  return { predict, epochs };
}

// ---------- data alignment on the index calendar ----------
export function alignData(SRC) {
  const ixBars = SRC.bars[SRC.index], T = ixBars.map(b => b.time), N = T.length, IX = ixBars.map(b => b.close);
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
  return { T, N, IX, tickers, S };
}
