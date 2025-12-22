# Deployment Architecture - Local Mac Mini + Vercel

## Architecture Overview

```
Frontend (Vercel)
    ↓ HTTPS
Backend + Qdrant (Mac Mini M4)
    ↓ API calls
Claude API (Anthropic)
```

## Hardware Setup

**Mac Mini M4:**
- Internal SSD (256GB): OS, apps, project code, PDFs (25GB)
- External Drive (2TB): Qdrant vector storage (227GB-1.4TB)
- RAM (16GB): Qdrant (~4-6GB) + Backend (~2GB) = plenty of headroom

## Component Breakdown

### 1. Frontend - Vercel (Cloud)
- **Tech:** React + TypeScript
- **Hosting:** Vercel (free tier)
- **Domain:** yourapp.vercel.app or custom domain
- **Cost:** $0 (or $20/year for custom domain)

### 2. Backend - Mac Mini (Local)
- **Tech:** Python + FastAPI + LlamaIndex
- **Port:** 8000
- **Exposed via:** Cloudflare Tunnel (free) or Tailscale Funnel (free)
- **Public URL:** https://your-backend.trycloudflare.com
- **Location:** /Users/merlin/projects/researchPaperAgent/backend

### 3. Vector DB - Mac Mini (Local)
- **Tech:** Qdrant (Docker)
- **Port:** 6333 (only accessible to backend on localhost)
- **Storage:** /Volumes/ExternalDrive/qdrant_data
- **Data Size:** ~227GB (with optimizations)

### 4. APIs (Cloud)
- **Claude API:** For LLM responses
- **Voyage AI:** For embeddings
- **Cost:** ~$50-100/month (usage-based)

---

## Networking Setup

### Expose Backend to Internet (Pick One)

#### Option A: Cloudflare Tunnel (Recommended - Free)
```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared

# Start tunnel (auto-generates URL)
cloudflared tunnel --url http://localhost:8000

# Output: https://random-name.trycloudflare.com
```

**Pros:**
- Free forever
- HTTPS automatic
- No port forwarding needed
- No account required

**Cons:**
- URL changes each restart (or pay for static)

#### Option B: Tailscale Funnel (Easy - Free)
```bash
# Install Tailscale
brew install tailscale

# Expose port
tailscale funnel 8000
```

**Pros:**
- Free with static URL
- Very secure
- Easy setup

**Cons:**
- Requires Tailscale account

#### Option C: ngrok (Most Features - $8/month)
```bash
brew install ngrok
ngrok http 8000
```

**Pros:**
- Static domain
- More control
- Good dashboard

**Cons:**
- $8/month for static domain

---

## Updated docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "127.0.0.1:6333:6333"  # Only accessible from localhost
      - "127.0.0.1:6334:6334"  # Only accessible from localhost
    volumes:
      # Use external drive for vector storage
      - /Volumes/ExternalDrive/qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      # Enable optimizations
      - QDRANT__STORAGE__QUANTIZATION__ENABLED=true
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Storage Optimization Strategy

### Recommended: 1500-word chunks + quantization

```python
# backend/preprocessing/pdf_processor.py
def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 250):
```

**Result:**
- Total chunks: ~155K
- With quantization: ~227 GB
- Fits comfortably on 2TB external drive

**Storage Breakdown:**
```
External Drive (2TB):
├── qdrant_data/        227 GB  (vector database)
└── backup/            (optional)

Internal SSD (256GB):
├── macOS/              ~80 GB
├── Applications/       ~40 GB
├── User files/         ~60 GB
├── Project code/        ~1 GB
├── PDFs (resource)/    25 GB
└── Free space/         ~50 GB  ✓ Plenty
```

---

## Mac Mini Setup

### 1. Keep Mac Mini Running 24/7

```bash
# Prevent sleep when plugged in
sudo pmset -c sleep 0
sudo pmset -c disksleep 0

# Allow SSH access
sudo systemsetup -setremotelogin on
```

### 2. Auto-start Services on Boot

Create launch agent for FastAPI:

```bash
# ~/Library/LaunchAgents/com.researchpaper.backend.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.researchpaper.backend</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/merlin/projects/researchPaperAgent/start_backend.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Docker Qdrant already has `restart: unless-stopped`

### 3. Monitoring

```bash
# Check if services are running
docker ps  # Qdrant
curl http://localhost:8000/health  # Backend

# Monitor logs
docker logs -f qdrant
tail -f backend/logs/app.log
```

---

## Development vs Production

### Development (Current)
```bash
cd backend
source venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

### Production (Mac Mini 24/7)
```bash
# Use production ASGI server
cd backend
source venv/bin/activate
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Cost Breakdown

| Component | Provider | Cost |
|-----------|----------|------|
| **Frontend Hosting** | Vercel | $0 |
| **Backend Hosting** | Mac Mini (owned) | $0 |
| **Vector DB** | Mac Mini (owned) | $0 |
| **Tunnel** | Cloudflare Tunnel | $0 |
| **Claude API** | Anthropic | ~$50-100/month |
| **Embeddings** | Voyage AI | $132 one-time |
| **Electricity** | Mac Mini (8W idle) | ~$1/month |
| **Domain** (optional) | Namecheap | $10-20/year |
| **TOTAL** | | **~$50-100/month** |

Compare to cloud hosting everything: **$180-300/month**

**Savings: $130-200/month = $1,560-2,400/year!**

---

## Advantages of This Setup

1. **Cost-Effective:** Save $1,500+/year vs full cloud
2. **Fast:** Backend & DB on same machine (0 latency)
3. **Private:** PDFs never leave your possession
4. **Scalable:** Mac Mini M4 can handle thousands of queries/day
5. **Simple:** One machine to manage
6. **Professional Frontend:** Vercel provides excellent UX

---

## Disadvantages & Mitigations

| Disadvantage | Mitigation |
|--------------|------------|
| Mac must stay on 24/7 | Mac Mini uses only 8W, costs ~$1/month electricity |
| Internet outage = downtime | Most home internet is 99%+ uptime |
| No automatic scaling | Mac Mini M4 can handle your expected load |
| Need to manage updates | Set up auto-updates or SSH in monthly |

---

## Load Testing

Your Mac Mini M4 can handle:

**Conservative estimate:**
- Qdrant: ~500 queries/second (vector search)
- FastAPI: ~100 requests/second (with Claude API being bottleneck)
- Concurrent users: 20-50 easily

**For a personal research tool, this is massive overkill in a good way!**

---

## Security Considerations

1. **API Keys:** Store in `.env`, never commit
2. **Rate Limiting:** Add to FastAPI (prevent abuse)
3. **Authentication:** Add simple auth (email/password or OAuth)
4. **HTTPS:** Cloudflare Tunnel provides automatic HTTPS
5. **Firewall:** Only expose port 8000 via tunnel

---

## Backup Strategy

**What to backup:**
1. ❌ Vector DB (can be regenerated from PDFs)
2. ✅ PDFs (25GB - your source of truth)
3. ✅ Backend code (Git repository)
4. ✅ Environment configs (.env files)

**Recommendation:**
```bash
# Backup PDFs to cloud storage
rclone sync resource/ dropbox:research-papers/
```

---

## Next Steps

1. ✅ You have Mac Mini M4
2. ✅ You have 2TB external drive
3. ⬜ Update docker-compose.yml to use external drive
4. ⬜ Optimize chunk size to 1500 words
5. ⬜ Process PDFs and create vector DB
6. ⬜ Set up Cloudflare Tunnel
7. ⬜ Deploy frontend to Vercel
8. ⬜ Configure Mac Mini for 24/7 operation

---

**Status:** Ready to implement! Your hardware is perfect for this architecture.
