# MAIA Backend - Development Plan

**Last Updated**: Dec 19, 2025
**Status**: Core pipeline implemented, needs testing & monitoring

---

## Project Overview

MAIA (Motion Animation & Identity Acquisition) converts video of humans into animated, world-grounded 3D SMPL meshes for VR/AR "Matrix replay".

**Goal**: Capture video → 3D mesh sequence → VR playback with slow-mo, rewind, etc.

---

## Current Implementation

### Completed Modules

| Module | File | Status | Description |
|--------|------|--------|-------------|
| SAM3D API | `app/sam3d_api.py` | ✅ Working | FastAPI server, SSE streaming |
| World Grounding | `app/grounding/world_grounding.py` | ✅ Implemented | Floor estimation, pelvis stabilization |
| SAM3 Tracking | `app/segmentation/sam3_tracker.py` | ✅ Implemented | Identity-consistent person tracking |
| Body4D Pipeline | `app/pipeline/body4d_pipeline.py` | ✅ Implemented | Unified async pipeline |
| Temporal Smoothing | `app/temporal_smoothing/` | ✅ Existing | One-Euro filter, physical constraints |
| RunPod Setup | `setup_runpod.sh` | ✅ Ready | One-command deployment script |

### API Endpoints

```
GET  /health                      - Health check
POST /sam3d/body/process_stream   - Original streaming (camera-space output)
POST /sam3d/body/process_video_v2 - V2 with world grounding (RECOMMENDED)
POST /sam3d/body/ground_sequence  - Apply grounding to existing sequence
GET  /sam3d/body/template_mesh    - Get SMPL mesh topology
```

### Key Files to Understand

1. **`app/sam3d_api.py`** - Main FastAPI app, all endpoints defined here
2. **`app/grounding/world_grounding.py`** - Core grounding logic:
   - `estimate_ground_plane()` - Find floor from foot positions
   - `detect_standing_frames()` - Detect when feet are stationary
   - `ground_sequence()` - Apply full grounding transform
3. **`app/pipeline/body4d_pipeline.py`** - Orchestrates the full pipeline:
   - SAM3 tracking → SAM3D Body → Smoothing → Grounding
4. **`setup_runpod.sh`** - Clones dependencies, downloads models

---

## Next Steps: Testing & Monitoring

### Priority 1: Validation Test Suite

Create `tests/test_pipeline.py`:

```python
# Test cases needed:
1. test_single_person_video()
   - Input: 3-5 second video with one person
   - Verify: Output has correct shape, no NaN, reasonable joint positions

2. test_grounding_applied()
   - Verify: Floor height near 0, feet touch ground in standing frames

3. test_temporal_smoothness()
   - Verify: No sudden jumps between frames (velocity < threshold)

4. test_multi_person()
   - Input: Video with 2+ people
   - Verify: Separate tracks maintained, no ID swaps
```

### Priority 2: Debug/Visual Validation Endpoint

Add to `sam3d_api.py`:

```python
@app.post("/sam3d/body/debug_video")
async def debug_video(file: UploadFile, render_overlay: bool = True):
    """
    Process video and return:
    - JSON with all SMPL params
    - Rendered video with mesh overlay (if render_overlay=True)
    - Per-stage timing metrics
    - Intermediate outputs (masks, raw poses, grounded poses)
    """
```

Implementation needs:
- PyRender or Open3D for mesh rendering
- OpenCV for video composition
- Save intermediate `.npy` files for debugging

### Priority 3: Structured Logging & Metrics

Add logging throughout pipeline:

```python
import logging
import time

logger = logging.getLogger("maia")

# In pipeline stages:
start = time.time()
# ... do work ...
logger.info(f"SAM3 tracking: {time.time()-start:.2f}s, {num_frames} frames, {num_people} people")
```

Metrics to track:
- Per-stage latency (tracking, body extraction, smoothing, grounding)
- GPU memory usage
- Frames per second
- Error rates by stage

### Priority 4: Health Dashboard

Create `app/dashboard.py`:

```python
@app.get("/status")
async def status():
    return {
        "gpu_memory_used": get_gpu_memory(),
        "gpu_memory_total": get_gpu_total(),
        "queue_depth": processing_queue.qsize(),
        "recent_errors": error_log[-10:],
        "avg_fps": compute_avg_fps(),
        "uptime": time.time() - start_time
    }
```

---

## Known Issues / TODOs

1. **SAM3 fallback**: `sam3_tracker.py` has YOLO fallback when SAM3 unavailable - needs testing
2. **Multi-person grounding**: Current grounding assumes single person - may need per-person ground planes
3. **Moving camera**: Current grounding assumes static camera - for moving camera, need SLAM or D4RT integration
4. **Model loading**: First request is slow (model loading) - consider preloading on startup

---

## Dependencies

**Required on RunPod:**
- CUDA 12.1+
- Python 3.10+
- HuggingFace token with access to `facebook/sam-3d-body-dinov3`

**Key packages:**
- `torch` - PyTorch with CUDA
- `fastapi` + `uvicorn` - API server
- `transformers` - HuggingFace model loading
- `smplx` - SMPL body model
- `numpy`, `scipy` - Numerical operations

---

## How to Continue Development

### Local Testing (no GPU)

1. Mock the SAM3D model responses
2. Test grounding/smoothing logic with synthetic data
3. Run unit tests: `pytest tests/`

### RunPod Testing (with GPU)

```bash
# 1. SSH into RunPod instance
# 2. Clone and setup
git clone https://github.com/Dexin-Huang/maia-backend.git
cd maia-backend
export HUGGINGFACE_HUB_TOKEN=hf_xxx
./setup_runpod.sh

# 3. Run server
uvicorn app.sam3d_api:app --host 0.0.0.0 --port 8000

# 4. Test with sample video
curl -X POST http://localhost:8000/sam3d/body/process_video_v2 \
  -F "file=@test_video.mp4" \
  -F "enable_grounding=true"
```

### Sample Test Videos

Need to add sample videos in `tests/fixtures/`:
- `single_person_walk.mp4` - One person walking
- `single_person_stand.mp4` - One person standing still
- `two_people.mp4` - Two people in frame
- `basketball_shot.mp4` - Athletic motion (fast movement)

---

## Architecture Notes

### Data Flow

```
Video Input (MP4)
    ↓
Frame Extraction (OpenCV)
    ↓
SAM3 Tracking (identity-consistent masks)
    ↓
SAM3D Body (per-frame SMPL extraction)
    ↓
Temporal Smoothing (One-Euro filter)
    ↓
World Grounding (floor=0, pelvis stable)
    ↓
Output: SMPL sequence {betas, body_pose, global_orient, transl, vertices}
```

### SMPL Output Format

```python
{
    "frames": [
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "people": [
                {
                    "person_id": 0,
                    "betas": [10 floats],        # Body shape
                    "body_pose": [69 floats],    # Joint rotations
                    "global_orient": [3 floats], # Root rotation
                    "transl": [3 floats],        # Root translation
                    "vertices": [6890 x 3]       # Optional mesh vertices
                }
            ]
        }
    ],
    "metadata": {
        "fps": 30.0,
        "grounding_applied": true,
        "floor_height": 0.0,
        "smoothing_preset": "balanced"
    }
}
```

---

## Related Repos

- **maia-frontend**: https://github.com/Dexin-Huang/maia-frontend (React Three Fiber viewer)
- **maia** (umbrella): https://github.com/Dexin-Huang/maia
- **SAM-Body4D** (reference): https://github.com/gaomingqi/sam-body4d

---

## Contact

Questions? Check the code comments or the original conversation context.
