#!/bin/bash

##############################################
# Setup Cloudflare Tunnel as a Service
# This ensures the tunnel survives reboots
##############################################

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "Setting up Cloudflare Tunnel as a system service..."

# Check if cloudflared is installed
if ! command -v cloudflared > /dev/null 2>&1; then
    log_error "cloudflared is not installed"
    echo ""
    echo "To install cloudflared, run:"
    echo "  brew install cloudflare/cloudflare/cloudflared"
    exit 1
fi

# Check if tunnel is configured
if [ ! -d "$HOME/.cloudflared" ] || [ ! -f "$HOME/.cloudflared/config.yml" ]; then
    log_error "Cloudflare Tunnel is not configured"
    echo ""
    echo "To configure a tunnel, follow these steps:"
    echo "  1. cloudflared tunnel login"
    echo "  2. cloudflared tunnel create researchpaper-api"
    echo "  3. Create ~/.cloudflared/config.yml with your configuration"
    echo "  4. cloudflared tunnel route dns <tunnel-name> api.yourdomain.com"
    echo ""
    echo "See DEPLOYMENT_GUIDE.md for detailed instructions"
    exit 1
fi

log_info "Found Cloudflare Tunnel configuration"

# Stop any running manual instances
log_info "Stopping any manual cloudflared instances..."
pkill -f "cloudflared tunnel run" 2>/dev/null || true

# Check if service is already installed
if [ -f "/Library/LaunchDaemons/com.cloudflare.cloudflared.plist" ]; then
    log_warn "Service already installed. Reinstalling..."
    sudo launchctl unload /Library/LaunchDaemons/com.cloudflare.cloudflared.plist 2>/dev/null || true
    sudo cloudflared service uninstall 2>/dev/null || true
fi

# Install as service
log_info "Installing cloudflared as a system service..."
sudo cloudflared service install

# Load the service
log_info "Loading service..."
sudo launchctl load /Library/LaunchDaemons/com.cloudflare.cloudflared.plist

# Wait a moment for it to start
sleep 3

# Check if it's running
if sudo launchctl list | grep -q "com.cloudflare.cloudflared"; then
    log_info "✓ Cloudflare Tunnel service is running"
    echo ""
    echo "The tunnel will now:"
    echo "  • Start automatically on boot"
    echo "  • Restart automatically if it crashes"
    echo "  • Survive OS updates"
    echo ""
    echo "To check status:"
    echo "  sudo launchctl list | grep cloudflare"
    echo "  cloudflared tunnel info"
    echo ""
    echo "To view logs:"
    echo "  sudo tail -f /Library/Logs/cloudflared.log"
    echo ""
    echo "To manually control:"
    echo "  sudo launchctl stop com.cloudflare.cloudflared"
    echo "  sudo launchctl start com.cloudflare.cloudflared"
else
    log_error "Failed to start service"
    echo ""
    echo "Check logs with:"
    echo "  sudo tail -f /Library/Logs/cloudflared.log"
    exit 1
fi

log_info "Setup complete!"
