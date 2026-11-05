// Render the radar dashboard HTML from model output and (optionally) the integrated risk assessment.
// Shared by the CLI builder (quant/build-radar.mjs) and the web backend (src/web.js).
import { readFileSync } from 'fs';

const TEMPLATE = new URL('radar.template.html', import.meta.url);

// Merge the risk assessment into the model's stocks. The integrated verdict can only make a label more cautious.
export function mergeRisk(model, risk) {
  const D = structuredClone(model);
  if (risk) {
    const byTk = Object.fromEntries(risk.stocks.map(s => [s.tk, s]));
    for (const s of D.stocks) {
      const k = byTk[s.tk]; if (!k) continue;
      s.risk = k;
      if (k.assessment.verdict === 'Severe' && s.label !== 'avoid') { s.label = 'avoid'; s.why.push('integrated risk assessment is Severe (top fifth of the list)'); }
      else if (k.assessment.verdict === 'High' && s.label === 'shortlist') { s.label = 'watch'; s.why.push('integrated risk assessment is High, so it drops from the shortlist'); }
    }
    D.shortlist = D.stocks.filter(s => s.label === 'shortlist').map(s => s.tk);
    const { stocks, ...rest } = risk;
    D.riskmeta = rest;
  }
  return D;
}

// `drawn`: tickers whose levels are drawn in TradingView (only known on the machine that drew them).
export function renderDashboard(model, risk, { drawn = [], daytrade = null } = {}) {
  const D = mergeRisk(model, risk);
  D.drawn = drawn;
  D.daytrade = daytrade;
  // Escape "</" so embedded data can never close the <script> element early.
  const json = JSON.stringify(D).replace(/<\//g, '<\\/');
  const body = readFileSync(TEMPLATE, 'utf8').replace('__DATA__', () => json);
  // A complete document: opens directly in a browser, and is what the web server and artifact hosting expect.
  const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}[hidden]{display:none!important}img{max-width:100%}</style></head><body>
${body}
</body></html>`;
  return { html, data: D };
}
