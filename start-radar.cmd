@echo off
rem Start the IDX Risk Radar web server and a Cloudflare quick tunnel on this Windows PC.
rem Double-click this file. Two windows open; keep both open while you want remote access.
rem The tunnel window prints the public https://....trycloudflare.com address (it changes every start).
rem Needs: Node.js 20.6+, a .env.local file with RADAR_TOKEN (see DEPLOY.md), and cloudflared.
cd /d "%~dp0"
if not exist ".env.local" (
  echo .env.local is missing. Create it with a line RADAR_TOKEN=^<long random string^>, HOST=127.0.0.1 and PORT=8787.
  pause
  exit /b 1
)
set "CF=C:\Program Files (x86)\cloudflared\cloudflared.exe"
if not exist "%CF%" set "CF=cloudflared"
start "Radar web server" cmd /k node --env-file=.env.local src/web.js
timeout /t 2 /nobreak >nul
start "Radar tunnel" cmd /k ""%CF%" tunnel --no-autoupdate --url http://127.0.0.1:8787"
