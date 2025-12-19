# MAIA Backend

**Motion Animation & Identity Acquisition**

Backend API for volumetric video to 3D human reconstruction. Converts video of humans into animated, world-grounded 3D meshes for VR/AR "Matrix replay".

## Features

- **SAM 3D Body** - Meta's state-of-the-art human mesh recovery
- **SAM3 Tracking** - Identity-consistent multi-person tracking
- **Temporal Smoothing** - One-Euro filter + foot contact detection
- **World Grounding** - Floor at Y=0, pelvis stabilization
- **Real-time Streaming** - SSE events for live progress

## Quick Start (RunPod)

```bash
# 1. Clone this repo
git clone https://github.com/Dexin-Huang/maia-backend.git
cd maia-backend

# 2. Set HuggingFace token (required for gated models)
export HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# 3. Run setup
chmod +x setup_runpod.sh
./setup_runpod.sh

# 4. Start server
uvicorn app.sam3d_api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/sam3d/body/process_stream` | POST | Original streaming endpoint |
| `/sam3d/body/process_video_v2` | POST | **V2 with world grounding** |
| `/sam3d/body/ground_sequence` | POST | Apply grounding to sequence |
| `/sam3d/body/template_mesh` | GET | Get SMPL mesh topology |

### V2 Endpoint (Recommended)

```bash
curl -X POST http://localhost:8000/sam3d/body/process_video_v2 \
  -F "file=@video.mp4" \
  -F "enable_grounding=true"
```

## Project Structure

```
maia-backend/
├── app/
│   ├── sam3d_api.py              # FastAPI server
│   ├── grounding/                # World grounding
│   ├── segmentation/             # SAM3 tracking
│   ├── pipeline/                 # Body4D pipeline
│   └── temporal_smoothing/       # Smoothing
├── Dockerfile
├── requirements.txt
└── setup_runpod.sh
```

## Requirements

- NVIDIA GPU with 12GB+ VRAM
- Python 3.10+, CUDA 12.1+
- HuggingFace access to facebook/sam-3d-body-dinov3

## Related

- [maia-frontend](https://github.com/Dexin-Huang/maia-frontend) - React Three Fiber 3D viewer
- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d) - Training-free 4D human mesh recovery
