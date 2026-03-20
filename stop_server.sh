#!/bin/bash

##############################################
# ARC Server Stop Script
##############################################

PROJECT_DIR="/Users/merlin/projects/ARC"
PID_FILE="$PROJECT_DIR/.backend.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_info "Stopping ARC server..."

# Stop backend by PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        log_info "Stopping backend (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            log_warn "Force killing backend..."
            kill -9 "$PID" 2>/dev/null || true
        fi
        log_info "Backend stopped"
    else
        log_warn "Backend process not running (PID: $PID)"
    fi
    rm -f "$PID_FILE"
else
    log_warn "No PID file found"
fi

# Kill any lingering uvicorn processes
UVICORN_PIDS=$(pgrep -f "uvicorn.*api.main:app" || true)
if [ ! -z "$UVICORN_PIDS" ]; then
    log_info "Killing lingering uvicorn processes: $UVICORN_PIDS"
    echo "$UVICORN_PIDS" | xargs kill -9 2>/dev/null || true
fi

# Stop Docker containers
if docker info > /dev/null 2>&1; then
    log_info "Stopping Docker containers..."
    cd "$PROJECT_DIR"
    docker-compose down 2>/dev/null || true
    log_info "Docker containers stopped"
else
    log_warn "Docker not running"
fi

log_info "Server stopped"
