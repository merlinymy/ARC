#!/bin/bash

##############################################
# ARC Server Status Checker
##############################################

PROJECT_DIR="/Users/merlin/projects/ARC"
PID_FILE="$PROJECT_DIR/.backend.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}     ARC Server Status Check${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check Docker
echo -e "${YELLOW}Docker Status:${NC}"
if docker info > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Docker is running"
    DOCKER_RUNNING=true
else
    echo -e "  ${RED}✗${NC} Docker is not running"
    DOCKER_RUNNING=false
fi
echo ""

# Check Qdrant
echo -e "${YELLOW}Qdrant Status:${NC}"
if [ "$DOCKER_RUNNING" = true ]; then
    if docker ps | grep -q qdrant; then
        echo -e "  ${GREEN}✓${NC} Qdrant container is running"
        if curl -f -s http://localhost:6333/healthz > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Qdrant is healthy (http://localhost:6333)"
            QDRANT_HEALTHY=true
        else
            echo -e "  ${RED}✗${NC} Qdrant is not responding"
            QDRANT_HEALTHY=false
        fi
    else
        echo -e "  ${RED}✗${NC} Qdrant container is not running"
        QDRANT_HEALTHY=false
    fi
else
    echo -e "  ${RED}✗${NC} Cannot check Qdrant (Docker not running)"
    QDRANT_HEALTHY=false
fi
echo ""

# Check Backend Process
echo -e "${YELLOW}Backend Process:${NC}"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Backend process is running (PID: $PID)"
        BACKEND_PROCESS=true
    else
        echo -e "  ${RED}✗${NC} Backend process not found (PID: $PID)"
        BACKEND_PROCESS=false
    fi
else
    # Check for any uvicorn process
    UVICORN_PID=$(pgrep -f "uvicorn.*api.main:app" | head -1)
    if [ ! -z "$UVICORN_PID" ]; then
        echo -e "  ${YELLOW}⚠${NC}  Backend process found but no PID file (PID: $UVICORN_PID)"
        BACKEND_PROCESS=true
    else
        echo -e "  ${RED}✗${NC} Backend process is not running"
        BACKEND_PROCESS=false
    fi
fi
echo ""

# Check Backend Health
echo -e "${YELLOW}Backend API:${NC}"
if curl -f -s http://localhost:8001/health/quick > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Backend API is responding (http://localhost:8001)"
    BACKEND_HEALTHY=true

    # Get detailed health info if available
    HEALTH_RESPONSE=$(curl -s http://localhost:8001/health/quick 2>/dev/null)
    if [ ! -z "$HEALTH_RESPONSE" ]; then
        echo -e "  ${GREEN}ℹ${NC}  Response: $HEALTH_RESPONSE"
    fi
else
    echo -e "  ${RED}✗${NC} Backend API is not responding"
    BACKEND_HEALTHY=false
fi
echo ""

# Check Cloudflare Tunnel
echo -e "${YELLOW}Cloudflare Tunnel:${NC}"
if command -v cloudflared > /dev/null 2>&1; then
    if launchctl list | grep -q "com.cloudflare.cloudflared"; then
        echo -e "  ${GREEN}✓${NC} Cloudflare Tunnel service is loaded"
    else
        echo -e "  ${YELLOW}⚠${NC}  Cloudflare Tunnel installed but service not loaded"
    fi
else
    echo -e "  ${YELLOW}ℹ${NC}  Cloudflare Tunnel not installed (optional)"
fi
echo ""

# Check Virtual Environment
echo -e "${YELLOW}Python Environment:${NC}"
if [ -d "$PROJECT_DIR/venv" ]; then
    echo -e "  ${GREEN}✓${NC} Virtual environment exists"
    VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
    if [ -f "$VENV_PYTHON" ]; then
        PYTHON_VERSION=$($VENV_PYTHON --version 2>&1)
        echo -e "  ${GREEN}ℹ${NC}  $PYTHON_VERSION"
    fi
else
    echo -e "  ${RED}✗${NC} Virtual environment not found"
fi
echo ""

# Check Environment File
echo -e "${YELLOW}Configuration:${NC}"
if [ -f "$PROJECT_DIR/.env" ]; then
    echo -e "  ${GREEN}✓${NC} .env file exists"
else
    echo -e "  ${RED}✗${NC} .env file not found"
fi
echo ""

# Overall Status
echo -e "${BLUE}======================================${NC}"
echo -e "${YELLOW}Overall Status:${NC}"

if [ "$DOCKER_RUNNING" = true ] && [ "$QDRANT_HEALTHY" = true ] && [ "$BACKEND_HEALTHY" = true ]; then
    echo -e "  ${GREEN}✓ All services are running normally${NC}"
    echo ""
    echo "Services are accessible at:"
    echo "  • Backend API:  http://localhost:8001"
    echo "  • Qdrant:       http://localhost:6333"
    echo "  • API Docs:     http://localhost:8001/docs"
    exit 0
else
    echo -e "  ${RED}✗ Some services are not running properly${NC}"
    echo ""
    echo "To restart all services, run:"
    echo "  ./restart_server.sh"
    echo ""
    echo "To view logs:"
    echo "  tail -f backend/logs/backend-error.log"
    exit 1
fi
