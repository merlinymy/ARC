# ARC Server Auto-Restart Scripts

This directory contains scripts to automatically restart your ARC server after macOS updates or reboots.

## Files

- `restart_server.sh` - Main script to restart all services
- `stop_server.sh` - Stop all services
- `check_status.sh` - Check status of all services
- `install_autostart.sh` - Install auto-start on boot
- `setup_cloudflare_service.sh` - Install Cloudflare Tunnel as a service
- `com.arc.autostart.plist` - macOS launchd configuration

## Quick Start

### 1. Manual Restart (Do this now!)

To restart the server manually after an OS update:

```bash
cd /Users/merlin/projects/ARC
./restart_server.sh
```

This will:
- Stop any running backend processes
- Restart Docker and Qdrant
- Start the FastAPI backend
- Restart Cloudflare Tunnel (if configured)
- Verify all services are healthy

### 2. Install Auto-Start (Recommended)

To automatically restart the server after future OS updates:

```bash
cd /Users/merlin/projects/ARC
./install_autostart.sh
```

This installs a macOS LaunchAgent that will:
- Start the server when you log in
- Start the server after OS updates/reboots
- Wait 30 seconds after boot to ensure Docker is ready

### 3. Setup Cloudflare Tunnel (Important!)

If you're using Cloudflare Tunnel to expose your backend, install it as a service so it survives reboots:

```bash
cd /Users/merlin/projects/ARC
./setup_cloudflare_service.sh
```

This ensures:
- Tunnel starts automatically on boot
- Tunnel restarts if it crashes
- Tunnel survives OS updates

**If you don't do this**, you'll need to manually run `cloudflared tunnel run researchpaper-api` in a terminal after each reboot.

### 4. Stop the Server

To stop all services:

```bash
cd /Users/merlin/projects/ARC
./stop_server.sh
```

## Manual Control

### Start/Stop Services

```bash
# Start
launchctl start com.arc.autostart

# Stop
launchctl stop com.arc.autostart

# Uninstall auto-start
launchctl unload ~/Library/LaunchAgents/com.arc.autostart.plist
rm ~/Library/LaunchAgents/com.arc.autostart.plist
```

### Check Service Status

```bash
# Check if services are running
launchctl list | grep arc

# Check backend process
ps aux | grep uvicorn

# Check Docker
docker ps

# Test endpoints
curl http://localhost:8000/health/quick
curl http://localhost:6333/health
```

### View Logs

```bash
# Backend logs
tail -f backend/logs/backend.log
tail -f backend/logs/backend-error.log

# Auto-start logs
tail -f backend/logs/autostart.log

# Docker logs
docker logs -f qdrant
```

## Troubleshooting

### Docker Not Starting

If Docker fails to start automatically:

1. Open Docker Desktop manually
2. Wait for it to fully start
3. Run `./restart_server.sh` again

Or install Colima for headless operation:

```bash
brew install colima
colima start
```

### Backend Not Starting

Check the error logs:

```bash
tail -f backend/logs/backend-error.log
```

Common issues:
- Virtual environment not found: Run `python3.11 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt`
- Port already in use: Kill the process using `lsof -ti:8000 | xargs kill -9`
- Missing dependencies: `source venv/bin/activate && pip install -r backend/requirements.txt`

### Cloudflare Tunnel Issues

**IMPORTANT: After OS updates, Cloudflare Tunnel won't restart automatically unless installed as a service!**

#### Option 1: Install as Service (Recommended)

```bash
./setup_cloudflare_service.sh
```

This installs it as a system service that survives reboots.

#### Option 2: Start Manually (Temporary)

```bash
# Find your tunnel name
cloudflared tunnel list

# Start it (replace with your tunnel name)
cloudflared tunnel run researchpaper-api
```

Keep this terminal open - closing it will stop the tunnel.

#### Check Tunnel Status

```bash
# Check if service is running (system-wide)
sudo launchctl list | grep cloudflare

# Check tunnel info
cloudflared tunnel info researchpaper-api

# Test external access
curl https://api.yourdomain.com/health/quick
```

#### Troubleshooting Tunnel

```bash
# View logs
sudo tail -f /Library/Logs/cloudflared.log

# Restart service
sudo launchctl stop com.cloudflare.cloudflared
sudo launchctl start com.cloudflare.cloudflared

# Uninstall and reinstall
sudo cloudflared service uninstall
./setup_cloudflare_service.sh
```

### Services Not Auto-Starting After Boot

Check if the LaunchAgent is loaded:

```bash
launchctl list | grep com.arc.autostart
```

If not found, reinstall:

```bash
./install_autostart.sh
```

Check the auto-start logs:

```bash
tail -f backend/logs/autostart.log
tail -f backend/logs/autostart-error.log
```

## What Each Script Does

### restart_server.sh

1. Stops any existing backend processes
2. Restarts Docker and Qdrant container
3. Waits for Qdrant to be healthy
4. Starts the FastAPI backend server
5. Restarts Cloudflare Tunnel (if installed)
6. Verifies all services are running

### stop_server.sh

1. Stops the backend process
2. Kills any lingering uvicorn processes
3. Stops Docker containers

### install_autostart.sh

1. Copies the plist file to ~/Library/LaunchAgents
2. Loads the LaunchAgent
3. Configures auto-start on boot

## Environment Variables

The scripts use settings from `.env` in the project root. Make sure it's properly configured:

- `QDRANT_HOST=localhost`
- `QDRANT_PORT=6333`
- `API_PORT=8000`
- All required API keys are set

## Security Notes

- The backend logs may contain sensitive information
- Keep your `.env` file secure
- Consider setting up log rotation for long-term operation
- The auto-start service runs as your user (not root)

## Maintenance

### Update After Code Changes

After pulling new code:

```bash
cd /Users/merlin/projects/ARC
git pull
source venv/bin/activate
pip install -r backend/requirements.txt
./restart_server.sh
```

### Clean Restart

For a completely clean restart:

```bash
./stop_server.sh
docker-compose down -v  # Warning: This removes Qdrant data!
./restart_server.sh
```

## Support

If you encounter issues:

1. Check the logs in `backend/logs/`
2. Verify Docker is running: `docker ps`
3. Test individual components:
   - Qdrant: `curl http://localhost:6333/health`
   - Backend: `curl http://localhost:8000/health/quick`
4. Review the deployment guide: `DEPLOYMENT_GUIDE.md`
