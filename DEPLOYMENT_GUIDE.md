# Deployment Guide: Mac Mini Backend + Vercel Frontend

This guide covers deploying:

- **Backend + Qdrant** on your local Mac Mini
- **Frontend** on Vercel

---

## Part 1: Mac Mini Setup (Backend + Qdrant)

### Step 1.1: Install Prerequisites

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+ and Docker
brew install python@3.11 docker docker-compose

# Start Docker Desktop (or install Colima for headless)
brew install --cask docker
# OR for headless Mac Mini:
brew install colima
colima start
```

### Step 1.2: Clone and Setup the Project

```bash
cd ~
git clone <your-repo-url> researchPaperAgent
cd researchPaperAgent

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Step 1.3: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example and edit
cp .env.example .env
nano .env
```

Set these values in `.env`:

```env
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=your_anthropic_api_key
VOYAGE_API_KEY=your_voyage_api_key
COHERE_API_KEY=your_cohere_api_key

# Authentication (CHANGE THESE!)
JWT_SECRET=generate-a-64-char-random-string-here
DEFAULT_USERNAME=admin
DEFAULT_PASSWORD=your-secure-password-here

# Paths
PDF_SOURCE_DIR=/Users/youruser/pdfs
PROCESSED_DATA_DIR=./processed_data
UPLOAD_DIR=./uploads

# Qdrant (Docker will run on localhost)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=research_papers

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# CORS - Add your Vercel domain here (update after deploying frontend)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://your-app.vercel.app

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

Generate a secure JWT secret:

```bash
openssl rand -hex 32
```

### Step 1.4: Start Qdrant with Docker

```bash
# Start Qdrant
docker-compose up -d qdrant

# Verify it's running
curl http://localhost:6333/health
```

### Step 1.5: Initialize the Database

```bash
# Create data directories
mkdir -p data processed_data uploads

# The database will be initialized on first run
```

### Step 1.6: Run the Backend

For testing:

```bash
source venv/bin/activate
 python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Verify it works:

```bash
curl http://localhost:8000/health/quick
```

### Step 1.7: Setup as a System Service (launchd)

Create a launch agent for auto-start:

```bash
mkdir -p ~/Library/LaunchAgents
nano ~/Library/LaunchAgents/com.researchpaper.backend.plist
```

Add this content (adjust paths to your setup):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.researchpaper.backend</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/youruser/researchPaperAgent/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>backend.api.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/youruser/researchPaperAgent</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/Users/youruser/researchPaperAgent/venv/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/youruser/researchPaperAgent/logs/backend.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/youruser/researchPaperAgent/logs/backend-error.log</string>
</dict>
</plist>
```

Load the service:

```bash
mkdir -p ~/researchPaperAgent/logs
launchctl load ~/Library/LaunchAgents/com.researchpaper.backend.plist

# Check status
launchctl list | grep researchpaper

# To stop/start manually:
launchctl stop com.researchpaper.backend
launchctl start com.researchpaper.backend
```

---

## Part 2: Expose Mac Mini to the Internet

You have several options to expose your backend:

### Option A: Cloudflare Tunnel (Recommended - Free & Secure)

1. Create a free Cloudflare account at https://cloudflare.com
2. Add a domain (or use a free subdomain)
3. Install cloudflared:

```bash
brew install cloudflare/cloudflare/cloudflared

# Login to Cloudflare
cloudflared tunnel login

# Create a tunnel
cloudflared tunnel create researchpaper-api

# Configure the tunnel
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Add to `config.yml`:

```yaml
tunnel: <YOUR_TUNNEL_ID>
credentials-file: /Users/youruser/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Route DNS:

```bash
cloudflared tunnel route dns researchpaper-api api.yourdomain.com
```

Run the tunnel:

```bash
# Test first
cloudflared tunnel run researchpaper-api

# Then install as service
sudo cloudflared service install
sudo launchctl start com.cloudflare.cloudflared
```

Your backend is now at: `https://api.yourdomain.com`

### Option B: Tailscale Funnel (Simple, but requires Tailscale)

```bash
brew install tailscale
# Sign in and enable Funnel in admin console
tailscale funnel 8000
```

### Option C: ngrok (Quick for Testing)

```bash
brew install ngrok
ngrok http 8000
# Note the HTTPS URL provided
```

### Option D: Port Forwarding (Advanced)

1. Set a static IP for your Mac Mini in router settings
2. Forward port 8000 (or 443) to your Mac Mini
3. Use a dynamic DNS service (like DuckDNS) if you don't have a static public IP
4. Set up SSL with Let's Encrypt (requires additional setup)

---

## Part 3: Frontend Configuration for Vercel

### Step 3.1: Update API Configuration

Create/update `frontend/.env.production`:

```bash
# Create production environment file
echo "VITE_API_BASE_URL=https://api.yourdomain.com" > frontend/.env.production
```

### Step 3.2: Update the API Service

Modify `frontend/src/services/api.ts` to use environment variables:

Change line 17 from:

```typescript
const API_BASE = "/api";
```

To:

```typescript
const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";
```

### Step 3.3: Update Vite Config for Production

The current `vite.config.ts` proxy only works in development. For production builds, the `VITE_API_BASE_URL` environment variable will be used instead.

---

## Part 4: Deploy Frontend to Vercel

### Step 4.1: Prepare for Deployment

```bash
cd frontend

# Test production build locally
npm run build
npm run preview
```

### Step 4.2: Deploy to Vercel

**Option A: Via Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel

# Follow prompts:
# - Link to existing project or create new
# - Set root directory to: frontend (if deploying from repo root)
# - Framework: Vite
# - Build command: npm run build
# - Output directory: dist
```

**Option B: Via Vercel Dashboard**

1. Go to https://vercel.com and sign in
2. Click "Add New Project"
3. Import your Git repository
4. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Add Environment Variable:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: `https://api.yourdomain.com` (your backend URL)
6. Click "Deploy"

### Step 4.3: Update CORS on Backend

After deployment, update your `.env` on the Mac Mini to include the Vercel URL:

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://your-app.vercel.app,https://your-custom-domain.com
```

Restart the backend:

```bash
launchctl stop com.researchpaper.backend
launchctl start com.researchpaper.backend
```

---

## Part 5: Verification Checklist

### Backend (Mac Mini)

- [ ] Qdrant is running: `curl http://localhost:6333/health`
- [ ] Backend is running: `curl http://localhost:8000/health/quick`
- [ ] Backend is accessible externally: `curl https://api.yourdomain.com/health/quick`
- [ ] CORS is configured for Vercel domain

### Frontend (Vercel)

- [ ] Deployment succeeded in Vercel dashboard
- [ ] Environment variable `VITE_API_BASE_URL` is set
- [ ] Site loads at your Vercel URL
- [ ] API calls work (check browser Network tab)
- [ ] Login works with your credentials

---

## Part 6: Troubleshooting

### CORS Errors

If you see CORS errors in browser console:

1. Verify `CORS_ORIGINS` in backend `.env` includes your Vercel URL
2. Restart the backend after changing `.env`
3. Check that the URL in `CORS_ORIGINS` matches exactly (no trailing slash)

### Connection Refused

1. Verify Mac Mini is online and backend is running
2. Check Cloudflare Tunnel status: `cloudflared tunnel info researchpaper-api`
3. Test locally on Mac Mini: `curl http://localhost:8000/health`

### Slow Responses

1. Mac Mini may be sleeping - disable sleep in System Settings > Energy
2. Cloudflare Tunnel may need warming up on first request

### Backend Logs

```bash
# View backend logs
tail -f ~/researchPaperAgent/logs/backend.log
tail -f ~/researchPaperAgent/logs/backend-error.log

# View Qdrant logs
docker logs -f researchpaperagent-qdrant-1
```

---

## Part 7: Security Recommendations

1. **Change default credentials** - Update `DEFAULT_USERNAME` and `DEFAULT_PASSWORD`
2. **Use strong JWT secret** - Generate with `openssl rand -hex 32`
3. **Keep Mac Mini updated** - Enable automatic security updates
4. **Use Cloudflare Tunnel** - Provides DDoS protection and hides your IP
5. **Enable Cloudflare Access** (optional) - Add additional authentication layer
6. **Backup Qdrant data** - Regularly backup `./qdrant_data` directory
7. **Backup SQLite database** - Regularly backup `./data/app.db`

---

## Quick Reference Commands

```bash
# Start everything on Mac Mini
docker-compose up -d qdrant
launchctl start com.researchpaper.backend
cloudflared tunnel run researchpaper-api

# Stop everything
launchctl stop com.researchpaper.backend
docker-compose down

# Check status
curl http://localhost:8000/health
curl http://localhost:6333/health
launchctl list | grep researchpaper

# View logs
tail -f ~/researchPaperAgent/logs/backend.log

# Redeploy frontend on Vercel
cd frontend && vercel --prod
```
