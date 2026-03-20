#!/bin/bash

##############################################
# Install ARC Auto-Start Service
# This will configure the server to auto-restart
# after Mac OS updates or reboots
##############################################

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

PLIST_SRC="/Users/merlin/projects/ARC/com.arc.autostart.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.arc.autostart.plist"

log_info "Installing ARC auto-start service..."

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$HOME/Library/LaunchAgents"

# Copy plist file
log_info "Copying plist file to LaunchAgents..."
cp "$PLIST_SRC" "$PLIST_DEST"

# Unload if already loaded
if launchctl list | grep -q "com.arc.autostart"; then
    log_info "Unloading existing service..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

# Load the service
log_info "Loading service..."
launchctl load "$PLIST_DEST"

# Verify it's loaded
if launchctl list | grep -q "com.arc.autostart"; then
    log_info "✓ Service installed successfully!"
    echo ""
    echo "The server will now automatically start:"
    echo "  • When you log in"
    echo "  • After OS updates/reboots"
    echo ""
    echo "To manually control the service:"
    echo "  Start:   launchctl start com.arc.autostart"
    echo "  Stop:    launchctl stop com.arc.autostart"
    echo "  Unload:  launchctl unload $PLIST_DEST"
    echo ""
    echo "Or use the convenience scripts:"
    echo "  Start:   ./restart_server.sh"
    echo "  Stop:    ./stop_server.sh"
else
    log_error "Failed to load service"
    exit 1
fi
