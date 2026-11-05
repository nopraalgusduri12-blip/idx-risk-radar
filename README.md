# IDX Risk Radar

Data-only risk screening for Indonesian stocks (IDX), built on live daily data from **TradingView Desktop**.

- Measures 48 liquid IDX stocks plus IHSG with 26 price-based features.
- Ranks them with four machine-learning models and a factor rule, **validated walk-forward**. Each model's weight comes from its out-of-sample skill, so models with no skill get zero weight.
- Grades risk A–E (volatility, CVaR, drawdown, beta, liquidity, tail fatness) and labels each stock **Shortlist / Watch / Avoid** by fixed rules.
- Gives every stock an entry, **break-even after fees (BEP)**, stop loss and two take-profit levels, rounded to valid IDX ticks, plus a lot calculator.
- Draws those levels as labelled lines on **each stock's own TradingView chart**, so they are there whenever you switch to that stock.
- Builds a beginner-friendly HTML dashboard.
- Ships as an **MCP server**, so Claude (Desktop, Code, or claude.ai as a custom connector) can query and refresh it.

> **Not investment advice.** This is a statistical screen of past prices. In testing so far the models have **not** shown a reliable edge for predicting returns; the dashboard and the `radar_market` tool say so whenever that is the case.

## Requirements

- Windows, macOS or Linux with **Node.js 18+**
- **TradingView Desktop**, started with the Chrome DevTools port open (see below)
- Market data for IDX symbols in your TradingView account

## Install

```bash
git clone https://github.com/nopraalgusduri12/idx-risk-radar.git
cd idx-risk-radar
npm install
```

`npm install` pulls [tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp) (MIT) and applies `patches/tradingview-mcp+1.0.0.patch`, which fixes launching the **Microsoft Store** build of TradingView on Windows (it otherwise fails with `spawn EPERM`).

## Run the pipeline

```bash
npm run launch      # start TradingView with the debug port (keep this terminal open on Windows)
npm run radar       # fetch data -> train/validate models -> build dashboard
npm run draw        # draw levels on every stock's chart
```

Outputs:

| File | What |
|---|---|
| `quant/data/daily.json` | Raw daily bars per symbol |
| `quant/data/model.json` | Every metric, plan level, label, validation result |
| `quant/idx-risk-radar.html` | The dashboard (open in a browser) |

The fetch and draw steps switch the chart symbol while they run and restore it at the end.

## Use it from Claude

### Claude Desktop or Claude Code (local, recommended)

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "idx-risk-radar": {
      "command": "node",
      "args": ["C:\\path\\to\\idx-risk-radar\\src\\server.js"]
    }
  }
}
```

Restart Claude fully. Tools:

| Tool | Does |
|---|---|
| `radar_market` | Market caution level, model reliability, shortlist, suggested split |
| `radar_table` | Entry / BEP / stop / take-profit table, filter by label, sector, risk |
| `radar_stock` | Everything about one ticker |
| `radar_refresh` | Re-fetch data from TradingView, retrain, rebuild the dashboard |
| `radar_draw` | Draw levels on TradingView charts (`all` or a ticker list) |

### claude.ai custom connector (remote)

claude.ai connects to MCP servers over HTTPS, so the server has to be reachable from the internet. Run it in HTTP mode and put a tunnel in front of it:

```bash
set RADAR_TOKEN=<long random string>          # PowerShell: $env:RADAR_TOKEN="..."
npm run start:http                            # serves http://127.0.0.1:8787/mcp/<RADAR_TOKEN>
cloudflared tunnel --url http://127.0.0.1:8787
```

In claude.ai: **Settings → Connectors → Add custom connector**, and paste `https://<tunnel-host>/mcp/<RADAR_TOKEN>`.

Security:
- The secret token in the path is the only protection, so treat the URL like a password and use a long random token.
- Over HTTP only the read-only tools are exposed. Set `RADAR_ALLOW_CONTROL=1` to also expose `radar_refresh` and `radar_draw`, which control your TradingView app.
- The server binds to `127.0.0.1` only; nothing is reachable without the tunnel.

## Single-stock report

`quant/single/` holds the earlier one-stock risk sheet (NVDA, BBRI): stop/target brackets, VaR/CVaR, a 24-combination bracket backtest and a position calculator.

```bash
node node_modules/tradingview-mcp/src/cli/index.js ohlcv -n 500 > quant/single/ohlcv.json
node quant/single/analyze.mjs quant/single/ohlcv.json IDX:BBRI > quant/single/result.json
node quant/single/build.mjs quant/single/result.json quant/single/meta_BBRI.json quant/single/bbri.html
```

## How it works

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Limits

- About 300 daily bars per stock (what TradingView loads by default), all in one market regime.
- Prices only: no earnings, valuation, dividends, news or foreign flows.
- Costs, IDX auto-rejection limits and gaps are not modelled in the backtests. A gap can skip past a stop.
- The universe is a fixed list of 48 liquid names in `quant/fetch.mjs`; edit it to change coverage.

## License

MIT. Uses tradingview-mcp (MIT, © 2026 tradesdontlie). Not affiliated with TradingView Inc., IDX or Anthropic.
