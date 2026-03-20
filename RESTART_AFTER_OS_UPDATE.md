# Quick Start: Restart After macOS Update

Your Mac Mini had an OS update, which killed all running services. Here's how to get everything back up.

## Step 1: Restart Everything (DO THIS NOW)

```bash
cd /Users/merlin/projects/ARC
./restart_server.sh
```

This will:
- ✓ Restart Docker and Qdrant
- ✓ Restart the FastAPI backend
- ✓ Attempt to restart Cloudflare Tunnel (if installed as service)

## Step 2: Fix Cloudflare Tunnel

Your Cloudflare Tunnel was running in a terminal, so it died with the update.

### Option A: Install as Service (Recommended - Auto-starts forever)

```bash
./setup_cloudflare_service.sh
```

Now it will **automatically restart** after future OS updates!

### Option B: Start Manually (Temporary - Only until next reboot)

```bash
cloudflared tunnel run researchpaper-api
```

Keep this terminal open. When you close it, the tunnel stops.

## Step 3: Verify Everything Works

```bash
./check_status.sh
```

Should show all green checkmarks.

## Step 4: Test Your Vercel Frontend

1. Open your Vercel-hosted app in a browser
2. Try logging in
3. Try making an API request

If it works, you're done! If not, check:

```bash
# Check if tunnel is running
sudo launchctl list | grep cloudflare

# Test backend locally
curl http://localhost:8000/health/quick

# Test backend externally (replace with your domain)
curl https://api.yourdomain.com/health/quick
```

## Step 5: Prevent This in the Future (Optional but Recommended)

Install auto-start for the backend:

```bash
./install_autostart.sh
```

Now after future OS updates:
- ✓ Backend auto-starts
- ✓ Qdrant auto-starts
- ✓ Cloudflare Tunnel auto-starts (if you did Step 2 Option A)

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./restart_server.sh` | Restart all services |
| `./stop_server.sh` | Stop all services |
| `./check_status.sh` | Check what's running |
| `./setup_cloudflare_service.sh` | Install Cloudflare as service |
| `./install_autostart.sh` | Auto-start on boot |

## Need Help?

See `SERVER_SCRIPTS_README.md` for detailed documentation.

---

**TLDR:** Run `./restart_server.sh` then `./setup_cloudflare_service.sh` then `./install_autostart.sh` and you're done!
