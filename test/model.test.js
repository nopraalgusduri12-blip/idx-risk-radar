// Runs the model on a small synthetic universe and checks the output contract.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { writeFileSync, mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { execFileSync } from 'child_process';

function synth() {
  let s = 7; const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
  const days = 650, t0 = 1735689600, bars = {};
  const series = (start, drift, vol) => { let c = start; return Array.from({ length: days }, (_, i) => {
    const o = c; c = Math.max(50, c * Math.exp(drift + vol * rnd())); const h = Math.max(o, c) * 1.01, l = Math.min(o, c) * 0.99;
    return { time: t0 + i * 86400, open: o, high: h, low: l, close: c, volume: 5e7 + 1e7 * rnd() }; }); };
  bars['IDX:COMPOSITE'] = series(7000, 0, 0.02);
  ['BBCA', 'BBRI', 'TLKM', 'ASII', 'UNVR', 'ADRO', 'ANTM', 'GOTO'].forEach((tk, k) => { bars['IDX:' + tk] = series(1000 + 500 * k, (k - 4) * 0.0005, 0.03 + k * 0.004); });
  return { index: 'IDX:COMPOSITE', bars, failed: [] };
}

test('model produces plans with valid ticks and BEP above entry', () => {
  const dir = mkdtempSync(join(tmpdir(), 'radar-'));
  const inp = join(dir, 'daily.json'), out = join(dir, 'model.json');
  writeFileSync(inp, JSON.stringify(synth()));
  execFileSync(process.execPath, ['quant/model.mjs', inp, '--out', out]);
  const m = JSON.parse(execFileSync(process.execPath, ['-e', `process.stdout.write(require('fs').readFileSync(${JSON.stringify(out)},'utf8'))`]).toString());
  assert.equal(m.universe, 8);
  assert.equal(typeof m.reliable, 'boolean');
  const tick = p => p < 200 ? 1 : p < 500 ? 2 : p < 2000 ? 5 : p < 5000 ? 10 : 25;
  for (const s of m.stocks) {
    assert.ok(['shortlist', 'watch', 'avoid'].includes(s.label));
    assert.ok('ABCDE'.includes(s.risk_grade));
    assert.ok(s.plan.stop < s.plan.entry && s.plan.entry < s.plan.tp1 && s.plan.tp1 < s.plan.tp2, s.tk);
    assert.ok(s.plan.bep > s.plan.entry, `${s.tk} BEP above entry`);
    assert.equal(s.plan.stop % tick(s.plan.stop), 0, `${s.tk} stop on tick`);
    assert.equal(s.plan.bep % tick(s.plan.bep), 0, `${s.tk} BEP on tick`);
  }
});
