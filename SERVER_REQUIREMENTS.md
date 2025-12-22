# Server Requirements - Resource Calculation

## TL;DR - Recommended Specs

**For hosting everything on one server:**
- **Storage**: 200 GB SSD minimum (300 GB recommended for growth)
- **RAM**: 16 GB minimum (32 GB recommended)
- **CPU**: 4-8 cores (modern CPU)
- **Estimated cost**: $40-80/month depending on provider

---

## Detailed Storage Breakdown

### 1. Original PDF Files
```
22,401 papers × 3 MB average = 67 GB
```
- Research papers typically: 2-5 MB each
- Using 3 MB as conservative average

### 2. Vector Database (Qdrant)

**Calculating number of chunks:**
```
22,401 papers × 40 chunks/paper = 896,040 chunks
```
- Assuming average paper: 10-15 pages of content
- Chunk size: 512 tokens with 128 token overlap
- ~40 chunks per paper average

**Storage per chunk in Qdrant:**
```
Each chunk contains:
- Vector embedding (1536 dimensions): 1,536 floats × 4 bytes = 6,144 bytes
- Text content (~300 tokens): 300 × 4 bytes = 1,200 bytes
- Metadata (paper_id, title, page, etc.): ~500 bytes
- Total per chunk: ~7.8 KB
```

**Total vector database size:**
```
896,040 chunks × 7.8 KB = ~7 GB (raw data)
```

**With HNSW index overhead:**
```
7 GB × 2x (index structure) = ~14-15 GB
```

### 3. Extracted Images (Figures/Graphs)
```
22,401 papers × 7 images/paper = 156,807 images
156,807 images × 150 KB average = 23.5 GB
```
- Average 5-10 figures per paper
- PNG format: ~100-200 KB each

### 4. Application Code & Dependencies
```
Backend (Python + FastAPI + LlamaIndex): ~2 GB
Frontend (React build): ~200 MB
PostgreSQL database (optional): ~1-2 GB
System packages & Docker: ~3 GB
```

### 5. Operating System
```
Ubuntu/Debian: ~5 GB
```

### Total Storage Required

| Component | Size |
|-----------|------|
| PDFs | 67 GB |
| Vector DB (Qdrant) | 15 GB |
| Extracted images | 24 GB |
| Backend code | 2 GB |
| Frontend | 0.2 GB |
| PostgreSQL | 2 GB |
| System | 5 GB |
| **Subtotal** | **115 GB** |
| **Safety margin (30%)** | **+35 GB** |
| **TOTAL RECOMMENDED** | **150-200 GB** |

**For future growth (adding more papers): 300 GB recommended**

---

## RAM Requirements

### 1. Qdrant (Vector Database)

**Memory needs for fast search:**
```
HNSW index must be in RAM for performance
896,040 vectors with index structure: ~6-8 GB
```

Qdrant loads vectors into memory for:
- Fast similarity search
- HNSW graph traversal
- Query processing

### 2. Backend Application

```
FastAPI process: ~1-2 GB
LlamaIndex framework: ~1 GB
Request handling buffers: ~1 GB
Claude API client: ~500 MB
Total: ~3-4 GB
```

### 3. Self-Hosted Embedding Model (Optional)

**If using bge-large-en-v1.5 locally:**
```
Model size: 1.3 GB
Runtime memory: ~3-4 GB when processing
```

**Note**: Only needed during:
- Initial embedding (one-time)
- Adding new papers
- Embedding user questions (if not using API)

**If using Voyage/OpenAI API**: Skip this, saves 3-4 GB RAM!

### 4. PostgreSQL (Optional)
```
~1 GB for basic operations
```

### 5. System & Other Services
```
Ubuntu OS: ~1-2 GB
Nginx: ~100 MB
Docker overhead: ~500 MB
Total: ~2 GB
```

### Total RAM Required

**Scenario A: Using Embedding APIs (Voyage/OpenAI)**
```
Qdrant: 6-8 GB
Backend: 3-4 GB
PostgreSQL: 1 GB
System: 2 GB
-----------------
Total: 12-15 GB
Recommended: 16 GB RAM
```

**Scenario B: Self-Hosted Embeddings**
```
Above + Embedding model: 3-4 GB
-----------------
Total: 15-19 GB
Recommended: 32 GB RAM (to handle embedding + serving simultaneously)
```

**Note**: Embedding model only runs when:
- Processing new papers
- Embedding questions (can also use API just for questions)

---

## CPU Requirements

**Minimum: 4 cores**
**Recommended: 6-8 cores**

**Why:**
- Vector search is RAM-bound (not CPU-intensive)
- More cores help with:
  - Parallel user requests
  - Concurrent embedding jobs
  - Running multiple Docker containers
  - Background tasks (PDF processing)

**CPU type:**
- Modern x86_64 (Intel/AMD)
- 2.5+ GHz base clock
- AVX2 support (for faster vector operations)

---

## Network Requirements

**Bandwidth:**
- Upload: 100 Mbps minimum
- Download: 100 Mbps minimum
- Each response ~50-200 KB (text + citations)
- Images: up to 500 KB per figure

**For 100 concurrent users:**
- ~10-20 Mbps sustained

Most cloud servers include 1-5 TB/month bandwidth (more than enough).

---

## Specific Server Recommendations

### Option 1: DigitalOcean (Recommended)

**Droplet: "Performance" tier**
```yaml
CPU: 8 vCPUs (Premium Intel/AMD)
RAM: 16 GB
Storage: 320 GB SSD
Transfer: 5 TB/month
Cost: $96/month

OR (Budget option):

CPU: 4 vCPUs
RAM: 16 GB
Storage: 200 GB SSD
Transfer: 4 TB/month
Cost: $84/month
```

**For self-hosted embeddings:**
```yaml
CPU: 8 vCPUs
RAM: 32 GB
Storage: 320 GB SSD
Transfer: 6 TB/month
Cost: $168/month
```

### Option 2: AWS EC2

**Instance: t3.xlarge (Burstable)**
```yaml
vCPUs: 4
RAM: 16 GB
Storage: 200 GB EBS SSD (gp3)
Cost: ~$120/month (with EBS)
```

**Instance: c6a.2xlarge (Compute optimized, if embedding locally)**
```yaml
vCPUs: 8
RAM: 32 GB
Storage: 300 GB EBS SSD
Cost: ~$250/month
```

### Option 3: Hetzner (Best Price/Performance - Europe)

**CPX41**
```yaml
vCPUs: 8
RAM: 16 GB
Storage: 240 GB SSD
Transfer: 20 TB/month
Cost: €21.90/month (~$24/month) 🎉
```

**CPX51 (for self-hosted embeddings)**
```yaml
vCPUs: 16
RAM: 32 GB
Storage: 360 GB SSD
Transfer: 20 TB/month
Cost: €42.90/month (~$47/month)
```

### Option 4: Budget Option - Contabo

**VPS M SSD**
```yaml
vCPUs: 8
RAM: 16 GB
Storage: 400 GB SSD
Transfer: 32 TB/month
Cost: $14.50/month
```
⚠️ Warning: Contabo is cheaper but has mixed reviews on reliability

---

## Development vs Production

### Development (Your Local Computer)

**Minimum to test locally:**
```
RAM: 8 GB (can work, but tight)
Storage: 150 GB free space
CPU: 4 cores
```

**Comfortable development:**
```
RAM: 16 GB
Storage: 200 GB SSD
CPU: 6-8 cores
```

You can develop with a subset of papers (e.g., 1,000 papers) to save resources.

### Production (Cloud Server)

Use the full specs above for all 22,401 papers.

---

## Running on Local Machine (Home Computer)

### Can You Run Everything Locally? YES!

**Requirements for all 22,401 papers:**

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| **RAM** | 16 GB | 24 GB | 32 GB |
| **Storage** | 200 GB free | 300 GB free | 500 GB SSD |
| **CPU** | 4 cores (modern) | 6-8 cores | 8+ cores |
| **OS** | Windows/Mac/Linux | Linux/Mac | Linux |

### Local Machine Options

#### Option A: Desktop Computer (Best for this)

**Typical Gaming/Workstation PC:**
```yaml
CPU: AMD Ryzen 7 / Intel i7 (8 cores)
RAM: 32 GB DDR4
Storage: 1 TB NVMe SSD
Cost: $800-1500 (if buying new)
Monthly: $0 (just electricity ~$5-10/mo)
```

**Pros:**
- One-time cost, then free forever
- Upgradable (add more RAM/storage)
- Full control and privacy
- Can add GPU for local LLM later

**Cons:**
- Must keep computer running 24/7 for availability
- Uses electricity (~100-200W = $5-15/month)
- Limited by home internet upload speed
- If computer crashes, service is down

#### Option B: Laptop (Possible but not ideal)

**MacBook Pro / High-end Laptop:**
```yaml
RAM: 16-32 GB
Storage: 512 GB - 1 TB SSD
CPU: 8+ cores
```

**Pros:**
- Might already own one
- Portable
- Low power consumption

**Cons:**
- Not designed for 24/7 operation
- Gets hot when running continuously
- Wears out battery if always plugged in
- Can't easily upgrade

#### Option C: Mini PC / Home Server (Great middle ground)

**Intel NUC / Mac Mini / Beelink Mini PC:**
```yaml
CPU: 8 cores
RAM: 32 GB
Storage: 512 GB SSD
Power: 15-65W (very efficient)
Cost: $400-800
Monthly electricity: $2-5
```

**Pros:**
- Small, quiet, designed for continuous operation
- Very low power consumption
- Affordable
- Can hide in closet/shelf

**Cons:**
- Initial purchase cost
- Still need to keep it running

### Access from Outside Your Home

If running locally, A'Lester can access it:

**Option 1: Within Home Network Only (Easiest)**
```
- Access only when connected to home WiFi
- Zero setup needed
- Most secure
```

**Option 2: Tailscale VPN (Recommended)**
```
- Free for personal use
- Access from anywhere securely
- Works through firewalls
- Easy setup (~10 minutes)
```

**Option 3: Port Forwarding (Traditional)**
```
- Forward port 443 to your machine
- Need static IP or Dynamic DNS
- Security concerns (expose to internet)
- Router configuration needed
```

**Option 4: Cloudflare Tunnel (Free)**
```
- Free tunneling service
- No port forwarding needed
- Access from anywhere
- More complex setup
```

### Cost Comparison: Local vs Cloud

**Local Desktop/Mini PC:**
```
Initial: $400-1500 (hardware)
Monthly: $5-15 (electricity)
Year 1 total: $460-1680
Year 2+ total: $60-180/year
```

**Cloud Server (Hetzner):**
```
Initial: $0
Monthly: $24 (server) + $50-200 (Claude API)
Year 1 total: $888-2688
Year 2+ total: $888-2688/year
```

**Break-even point: ~6-18 months** depending on hardware cost

### Power Consumption

**Typical power usage:**
```
Desktop PC: 100-200W = $10-20/month (24/7)
Mini PC: 15-65W = $2-5/month (24/7)
Laptop: 45-85W = $5-10/month (24/7)
```

Based on $0.13/kWh average US electricity rate.

### Performance: Local vs Cloud

**Local advantages:**
- Usually FASTER (better single-machine specs)
- Lower latency (no network round-trip to cloud)
- No bandwidth costs
- Can add GPU for local LLM

**Cloud advantages:**
- Professional internet connection (faster upload)
- Redundancy (cloud providers have backups)
- Easy to scale up
- No hardware maintenance

### Hybrid Approach (Best of Both Worlds)

**My recommendation for your use case:**

```yaml
Development & Heavy Processing: Local machine
  - Process all 22K PDFs locally
  - Generate embeddings locally
  - Test everything
  - Cost: $0/month

Production Serving: Small cloud server
  - Just serves queries (lightweight)
  - Upload pre-processed database
  - Always available
  - Cost: $24/month for small server
```

This saves money because:
- Heavy one-time processing done locally (free)
- Only pay for 24/7 serving ($24/mo)
- Can use smaller cloud server since processing is done

### What Hardware Do You Currently Have?

**Check your current machine:**

```bash
# On Mac
sysctl hw.memsize hw.ncpu
df -h

# On Linux
free -h
lscpu
df -h

# On Windows (PowerShell)
Get-ComputerInfo | Select-Object CsProcessors,OsTotalVisibleMemorySize
Get-PSDrive C
```

**Can you tell me:**
1. What computer(s) do you have access to?
2. How much RAM do they have?
3. How much free storage?
4. Is it a desktop or laptop?

Then I can tell you if you can run everything locally!

### Realistic Scenarios

**Scenario 1: Already have good desktop**
```
Your desktop: 32 GB RAM, 1TB SSD
→ Run everything locally, use Tailscale for remote access
→ Cost: $5/month electricity
→ Best option!
```

**Scenario 2: Have decent laptop (16GB RAM)**
```
Your laptop: 16 GB RAM, 512 GB SSD
→ Develop locally, test with subset
→ Deploy full system to Hetzner ($24/mo)
→ Total: $24/month
```

**Scenario 3: Need to buy hardware**
```
→ Buy Mini PC ($600) + run locally
→ OR use cloud server ($24/mo)
→ Break-even: 25 months
→ Choose based on preference
```

**Scenario 4: Want best reliability**
```
→ Process locally (free)
→ Deploy to cloud ($24/mo)
→ Best of both worlds
```

---

## Local Machine Recommendations

### Budget Build (~$600)
```yaml
Option: Mini PC (Beelink SER6 Max)
CPU: AMD Ryzen 7 6800H (8 cores)
RAM: 32 GB DDR5
Storage: 500 GB NVMe SSD
Power: 54W
Cost: ~$500-600
Perfect for: 24/7 home server
```

### Mid-Range (~$1000)
```yaml
Option: Intel NUC 13 Pro
CPU: Intel i7-1360P (12 cores)
RAM: 32 GB DDR4
Storage: 1 TB NVMe SSD
Power: 28W
Cost: ~$900-1100
Perfect for: Quiet, efficient home server
```

### High-End (~$1500)
```yaml
Option: Custom Desktop / Mac Mini M2 Pro
CPU: 12+ cores
RAM: 32-64 GB
Storage: 1-2 TB SSD
GPU: Optional (for local LLM)
Cost: ~$1200-1800
Perfect for: Local LLM + embeddings + serving
```

### Budget Option (~$0 - Use What You Have)
```yaml
Your current computer (if specs allow)
Just needs: 16GB RAM, 200GB free space
Cost: $0
Limitation: Not ideal for 24/7
```

---

## Scaling Considerations

### If you get more papers:

**50,000 papers:**
- Storage: ~400 GB
- RAM: 24-32 GB
- Can upgrade server vertically

**100,000+ papers:**
- Consider splitting:
  - Vector DB on dedicated server
  - Backend on separate server
  - Scale horizontally

---

## Final Recommendations

### For Your Use Case (22,401 papers)

**Best Value Option:**
```yaml
Provider: Hetzner CPX41
CPU: 8 vCPUs
RAM: 16 GB
Storage: 240 GB SSD
Cost: ~$24/month
```

**If self-hosting embeddings too:**
```yaml
Provider: Hetzner CPX51
CPU: 16 vCPUs
RAM: 32 GB
Storage: 360 GB SSD
Cost: ~$47/month
```

**Conservative (DigitalOcean for reliability):**
```yaml
Provider: DigitalOcean
CPU: 8 vCPUs
RAM: 16 GB
Storage: 320 GB SSD
Cost: $96/month
```

---

## Cost Optimization Tips

1. **Start with embedding APIs** (Voyage/OpenAI)
   - Saves 16 GB RAM → can use smaller server
   - $132 one-time vs $50/month for bigger server
   - Pays for itself in 3 months

2. **Use subset for development**
   - Test with 1,000 papers locally
   - Only process all 22K on production server

3. **Separate embedding from serving**
   - Run embedding job on cheap spot instance
   - Serve queries on optimized server

4. **Use storage optimization**
   - Compress older images
   - Store PDFs separately (S3/B2) if needed
   - Keep only vectors + metadata in Qdrant

5. **Monitor and right-size**
   - Start with 16 GB
   - Monitor actual usage
   - Downgrade if using less

---

## Summary Table

| Scenario | Storage | RAM | CPU | Provider | Monthly Cost |
|----------|---------|-----|-----|----------|--------------|
| **Development (Local)** | 200 GB | 16 GB | 6 cores | Your computer | $0 |
| **Production (Recommended)** | 240 GB | 16 GB | 8 cores | Hetzner CPX41 | $24 |
| **Production (US-based)** | 320 GB | 16 GB | 8 cores | DigitalOcean | $96 |
| **With Self-Hosted Embeddings** | 360 GB | 32 GB | 16 cores | Hetzner CPX51 | $47 |
| **Budget Option** | 400 GB | 16 GB | 8 cores | Contabo | $15 |

**My recommendation: Start with Hetzner CPX41 ($24/mo) + Voyage API for embeddings**

Total monthly: $24 server + $1-2 embedding costs + $50-200 Claude API = **$75-225/month**

---

*Last updated: 2025-12-15*
