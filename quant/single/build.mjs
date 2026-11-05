// Fill the report template with a result file and a market meta file.
// Usage: node build.mjs result.json meta.json out.html
import { readFileSync, writeFileSync } from 'fs';
const [resPath, metaPath, out] = process.argv.slice(2);
const R = JSON.parse(readFileSync(resPath, 'utf8'));
const M = JSON.parse(readFileSync(metaPath, 'utf8'));
// Snap plan levels to the exchange tick: stops away from entry, targets to nearest tick.
const t = M.tick, near = x => Math.round(x / t) * t;
for (const P of R.plans) {
  const long = P.side === 'long';
  P.stop = long ? Math.floor(P.stop / t) * t : Math.ceil(P.stop / t) * t;
  P.tp1 = near(P.tp1); P.tp2 = near(P.tp2); P.tp3 = near(P.tp3);
  P.risk_per_share = Math.abs(P.entry - P.stop); P.risk_pct = P.risk_per_share / P.entry;
}
const html = readFileSync(new URL('./report.template.html', import.meta.url), 'utf8')
  .replace('__TITLE__', M.title).replace('__DATA__', JSON.stringify(R)).replace('__META__', JSON.stringify(M));
writeFileSync(out, html);
console.log(out, html.length);
