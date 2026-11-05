# IDX Risk Radar web backend (dashboard + API + MCP). TradingView is not needed in the container:
# the PC with TradingView runs `npm run collect` and uploads to this server.
FROM node:20-slim
# git is needed because npm installs the tradingview-mcp dependency from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json ./
COPY patches ./patches
RUN npm ci --omit=dev && npm cache clean --force
COPY . .
ENV NODE_ENV=production RADAR_DATA_DIR=/data PORT=8080 HOST=0.0.0.0
RUN mkdir -p /data && chown node:node /data
USER node
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s CMD node -e "fetch('http://127.0.0.1:'+process.env.PORT+'/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "src/web.js"]
