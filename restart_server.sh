#!/bin/bash

##############################################
# ARC Server Auto-Restart Script
# For Mac Mini after OS updates
##############################################

set -e  # Exit on error

PROJECT_DIR="/Users/merlin/projects/ARC"
VENV_PATH="$PROJECT_DIR/venv"
BACKEND_DIR="$PROJECT_DIR/backend"
LOG_DIR="$PROJECT_DIR/backend/logs"
BACKEND_LOG="$LOG_DIR/backend.log"
BACKEND_ERROR_LOG="$LOG_DIR/backend-error.log"
PID_FILE="$PROJECT_DIR/.backend.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

log_info "Starting ARC server restart sequence..."
cd "$PROJECT_DIR"

##############################################
# Step 1: Stop existing backend processes
##############################################
log_info "Stopping existing backend processes..."

# Kill process by PID file if it exists
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log_info "Killing backend process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE"
fi

# Kill any lingering uvicorn processes
UVICORN_PIDS=$(pgrep -f "uvicorn.*api.main:app" || true)
if [ ! -z "$UVICORN_PIDS" ]; then
    log_info "Killing lingering uvicorn processes: $UVICORN_PIDS"
    echo "$UVICORN_PIDS" | xargs kill -9 2>/dev/null || true
fi

# Check if using launchd service
if launchctl list | grep -q "com.researchpaper.backend"; then
    log_info "Stopping launchd service..."
    launchctl stop com.researchpaper.backend 2>/dev/null || true
fi

log_info "Backend processes stopped"

##############################################
# Step 2: Restart Docker (Qdrant)
##############################################
log_info "Restarting Qdrant database..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_warn "Docker is not running. Attempting to start Docker..."
    # Try to start Docker Desktop
    open -a Docker 2>/dev/null || log_warn "Could not start Docker Desktop automatically"

    # Wait for Docker to start
    log_info "Waiting for Docker to start..."
    for i in {1..30}; do
        if docker info > /dev/null 2>&1; then
            log_info "Docker is now running"
            break
        fi
        sleep 2
    done

    if ! docker info > /dev/null 2>&1; then
        log_error "Docker failed to start. Please start Docker manually and run this script again."
        exit 1
    fi
fi

# Stop and remove existing Qdrant container
log_info "Stopping existing Qdrant container..."
docker-compose down 2>/dev/null || true

# Start Qdrant
log_info "Starting Qdrant..."
docker-compose up -d qdrant

# Wait for Qdrant to be healthy
log_info "Waiting for Qdrant to become healthy..."
for i in {1..30}; do
    if curl -f -s http://localhost:6333/healthz > /dev/null 2>&1; then
        log_info "Qdrant is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Qdrant failed to start within 60 seconds"
        log_error "Check logs: docker logs qdrant"
        exit 1
    fi
    sleep 2
done

##############################################
# Step 3: Start Backend Server
##############################################
log_info "Starting backend server..."

# Activate virtual environment and start backend
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log_error "Virtual environment not found at $VENV_PATH"
    log_error "Please run: python3.11 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt"
    exit 1
fi

# Start backend server in background
log_info "Launching uvicorn server..."
cd "$BACKEND_DIR"
nohup "$VENV_PATH/bin/python" -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    >> "$BACKEND_LOG" 2>> "$BACKEND_ERROR_LOG" &

BACKEND_PID=$!
echo "$BACKEND_PID" > "$PID_FILE"

log_info "Backend started with PID: $BACKEND_PID"

# Wait for backend to be healthy
log_info "Waiting for backend to become healthy..."
for i in {1..30}; do
    if curl -f http://localhost:8001/health/quick > /dev/null 2>&1; then
        log_info "Backend is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Backend failed to start within 60 seconds"
        log_error "Check logs at: $BACKEND_ERROR_LOG"
        exit 1
    fi
    sleep 2
done

##############################################
# Step 4: Restart Cloudflare Tunnel (if configured)
##############################################
if command -v cloudflared > /dev/null 2>&1; then
    log_info "Checking Cloudflare Tunnel..."

    # Check if tunnel is configured
    if [ -d "$HOME/.cloudflared" ] && [ -f "$HOME/.cloudflared/config.yml" ]; then
        log_info "Restarting Cloudflare Tunnel service..."

        # Try to restart the service (check system-wide LaunchDaemon)
        if sudo launchctl list | grep -q "com.cloudflare.cloudflared"; then
            log_info "Found Cloudflare Tunnel system service, restarting..."
            sudo launchctl stop com.cloudflare.cloudflared 2>/dev/null || true
            sleep 2
            sudo launchctl start com.cloudflare.cloudflared
            log_info "Cloudflare Tunnel restarted"
        else
            log_warn "Cloudflare Tunnel service not installed"
            log_warn "Install as service: ./setup_cloudflare_service.sh"
            log_warn "Or start manually: cloudflared tunnel run researchpaper-api"
        fi
    else
        log_warn "Cloudflare Tunnel not configured. Skipping..."
    fi
else
    log_info "Cloudflare Tunnel not installed. Skipping..."
fi

##############################################
# Step 5: Verify Everything is Running
##############################################
log_info "Verifying all services..."

echo ""
echo "===== Service Status ====="

# Check Qdrant
if curl -f -s http://localhost:6333/healthz > /dev/null 2>&1; then
    log_info "✓ Qdrant: Running"
else
    log_error "✗ Qdrant: Not responding"
fi

# Check Backend
if curl -f http://localhost:8001/health/quick > /dev/null 2>&1; then
    log_info "✓ Backend: Running"
else
    log_error "✗ Backend: Not responding"
fi

# Check if process is running
if ps -p "$BACKEND_PID" > /dev/null 2>&1; then
    log_info "✓ Backend Process: Running (PID: $BACKEND_PID)"
else
    log_error "✗ Backend Process: Not running"
fi

echo ""
log_info "Server restart complete!"
echo ""
echo "Logs are available at:"
echo "  Backend:       $BACKEND_LOG"
echo "  Backend Error: $BACKEND_ERROR_LOG"
echo ""
echo "To view live logs, run:"
echo "  tail -f $BACKEND_LOG"
echo ""
echo "To stop the server, run:"
echo "  kill $BACKEND_PID"
echo "  # or"
echo "  kill \$(cat $PID_FILE)"
