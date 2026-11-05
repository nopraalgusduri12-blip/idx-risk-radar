#!/usr/bin/env node
// IDX Risk Radar MCP server for the PC that runs TradingView Desktop (stdio, for Claude Desktop / Claude Code).
// It has the read tools plus the two tools that need TradingView: refresh from the chart, and draw levels.
// For access from other devices and claude.ai, run the web backend instead: `npm run web` (src/web.js).
import { z } from 'zod';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { buildMcp, run, runModels, loadModel, PATHS } from './core.js';

if (process.argv.includes('--http')) {
  console.error('The --http mode moved to the web backend: run `npm run web` (see DEPLOY.md).');
  process.exit(1);
}

const extra = [
  { name: 'radar_refresh', config: { title: 'Refresh data and models',
      description: 'Pulls fresh daily bars for all 48 stocks and IHSG from TradingView Desktop (must be running with CDP on port 9222), retrains and re-validates every model, and rebuilds the dashboard. Takes several minutes and switches the chart symbol temporarily.',
      inputSchema: {} },
    handler: async () => {
      const f = await run('quant/fetch.mjs', [], 30 * 60 * 1000);
      await runModels();
      await run('quant/build-radar.mjs');
      const m = loadModel();
      return { fetched: f.split('\n').slice(-1)[0], as_of: m.as_of, data_dir: PATHS.model };
    } },
  { name: 'radar_draw', config: { title: 'Draw levels in TradingView',
      description: 'Draws entry, BEP, stop loss, take profits, noise-aware stop, realistic target and support/resistance as labelled lines on each stock\'s TradingView chart. Replaces earlier radar lines; leaves your own drawings alone.',
      inputSchema: { tickers: z.string().default('all').describe('"all" or comma-separated tickers, e.g. "BBRI,TLKM"') } },
    handler: async ({ tickers }) => ({ log: await run('quant/draw.mjs', [tickers], 30 * 60 * 1000) }) },
];

await buildMcp({ extra }).connect(new StdioServerTransport());
