#!/bin/bash

##############################################
# Start ARC with Quick Cloudflare Tunnel
##############################################

PROJECT_DIR="/Users/merlin/projects/ARC"
ENV_FILE="$PROJECT_DIR/backend/.env"
cd "$PROJECT_DIR"

echo "Starting services..."

# Start backend and Qdrant
./restart_server.sh

# Kill any existing tunnel
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
sleep 2

# Start quick tunnel in background
echo ""
echo "Starting Cloudflare Tunnel..."
cloudflared tunnel --url http://localhost:8000 > ./backend/logs/cloudflare-tunnel.log 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
echo "Waiting for tunnel URL..."
TUNNEL_URL=""
for i in {1..20}; do
    TUNNEL_URL=$(grep -o "https://[^[:space:]]*.trycloudflare.com" ./backend/logs/cloudflare-tunnel.log 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "ERROR: Could not get tunnel URL after 20 seconds"
    echo "Check: tail -f backend/logs/cloudflare-tunnel.log"
    exit 1
fi

# Update CORS_ORIGINS in .env with the new tunnel URL
# Keep localhost origins and vercel, replace the old trycloudflare URL
if [ -f "$ENV_FILE" ]; then
    # Build new CORS value: keep everything except old trycloudflare URLs, add new one
    CURRENT_CORS=$(grep "^CORS_ORIGINS=" "$ENV_FILE" | cut -d'=' -f2-)
    # Remove any existing trycloudflare URLs
    NEW_CORS=$(echo "$CURRENT_CORS" | sed 's|,https://[^,]*\.trycloudflare\.com||g' | sed 's|https://[^,]*\.trycloudflare\.com,||g' | sed 's|https://[^,]*\.trycloudflare\.com||g')
    # Append the new tunnel URL
    NEW_CORS="${NEW_CORS},${TUNNEL_URL}"
    # Update .env
    sed -i '' "s|^CORS_ORIGINS=.*|CORS_ORIGINS=${NEW_CORS}|" "$ENV_FILE"
    echo "Updated CORS_ORIGINS with new tunnel URL"
fi

# Restart backend so it picks up the new CORS config
echo "Restarting backend with updated CORS..."
PID_FILE="$PROJECT_DIR/.backend.pid"
VENV_PATH="$PROJECT_DIR/venv"
BACKEND_DIR="$PROJECT_DIR/backend"
BACKEND_LOG="$PROJECT_DIR/backend/logs/backend.log"
BACKEND_ERROR_LOG="$PROJECT_DIR/backend/logs/backend-error.log"

# Kill existing backend
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$OLD_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
fi
pkill -f "uvicorn.*api.main:app" 2>/dev/null || true
sleep 1

# Start backend with updated .env
cd "$BACKEND_DIR"
nohup "$VENV_PATH/bin/python" -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    >> "$BACKEND_LOG" 2>> "$BACKEND_ERROR_LOG" &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$PID_FILE"

# Wait for backend health
for i in {1..30}; do
    if curl -f -s http://localhost:8000/health/quick > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

cd "$PROJECT_DIR"

echo ""
echo "======================================"
echo "✓ All services started!"
echo "======================================"
echo ""
echo "Backend URL: $TUNNEL_URL"
echo ""
echo "Update Vercel with this URL:"
echo "  VITE_API_BASE_URL=$TUNNEL_URL"
echo ""
echo "Tunnel logs: tail -f backend/logs/cloudflare-tunnel.log"
echo "Backend logs: tail -f backend/logs/backend.log"
echo ""
echo "Press Ctrl+C to stop (or close terminal)"
echo ""

# Keep script running
tail -f backend/logs/cloudflare-tunnel.log
