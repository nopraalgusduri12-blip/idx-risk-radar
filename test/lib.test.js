// Checks the quant library against data with known properties.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as Q from '../quant/lib.mjs';

function garchSeries(n, a, b, v) {
  Q.reseed(1); const x = []; let s2 = v;
  for (let t = 0; t < n; t++) { const e = Math.sqrt(s2) * Q.randn(); x.push(e); s2 = v * (1 - a - b) + a * e * e + b * s2; }
  return x;
}

test('GARCH recovers persistence of a simulated process', () => {
  const g = Q.garchFit(garchSeries(3000, 0.08, 0.87, 1e-4));
  assert.ok(Math.abs(g.persistence - 0.95) < 0.04, `persistence ${g.persistence}`);
  assert.ok(g.alpha > 0.03 && g.alpha < 0.2, `alpha ${g.alpha}`);
});

test('Kupiec accepts the right failure rate and rejects double', () => {
  assert.ok(Q.kupiec(10, 1000, 0.01).p_value > 0.9);
  assert.ok(Q.kupiec(25, 1000, 0.01).p_value < 0.01);
});

test('GPD tail is heavier for Student-t-like data than for normal data', () => {
  Q.reseed(3);
  const normal = Array.from({ length: 5000 }, () => Q.randn());
  const heavy = Array.from({ length: 5000 }, () => { const z = Q.randn(), c = Q.randn() ** 2 + Q.randn() ** 2 + Q.randn() ** 2; return z / Math.sqrt(c / 3); });
  assert.ok(Q.gpdFit(heavy).xi > Q.gpdFit(normal).xi + 0.1);
  const risk = Q.gpdRisk(Q.gpdFit(normal), 0.99);
  assert.ok(Math.abs(risk.var - 2.326) < 0.25, `normal VaR99 ${risk.var}`);
  assert.ok(risk.es > risk.var);
});

test('HMM separates calm and volatile stretches', () => {
  Q.reseed(5);
  const x = [...Array.from({ length: 400 }, () => 0.005 * Q.randn()), ...Array.from({ length: 200 }, () => 0.02 * Q.randn())];
  const h = Q.hmmFit(x), p = Q.hmmFilter(h, x);
  assert.ok(h.sg[1] > 2 * h.sg[0]);
  assert.ok(Q.mean(Array.from(p.slice(450))) > 0.8 && Q.mean(Array.from(p.slice(50, 350))) < 0.2);
});

test('Corwin-Schultz reads a constant daily range with no drift as pure spread', () => {
  // Same high/low every day: the two-day range adds nothing, so the whole 2% range is attributed to the spread.
  const h = [101, 101, 101, 101], l = [99, 99, 99, 99];
  assert.ok(Math.abs(Q.corwinSchultz(h, l, 0, 3) - 0.02) < 0.001);
  // A trending path (range grows over two days) is read as volatility, giving a smaller spread.
  const h2 = [101, 103, 105, 107], l2 = [99, 101, 103, 105];
  assert.ok(Q.corwinSchultz(h2, l2, 0, 3) < 0.02);
});
