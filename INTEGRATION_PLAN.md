# Nvwa Backend Integration Plan: SAM-Body4D + World Grounding

**Goal**: Matrix-style 3D video replay - capture video, convert to 3D, playback in VR/AR with slow-mo and free camera

---

## Current State

### What We Have (Nvwa Backend)

```
backend/
├── app/
│   ├── sam3d_api.py              # FastAPI server (SSE streaming)
│   ├── config.py                 # Environment config
│   ├── storage.py                # Local/S3 storage
│   ├── export_animation.py       # PLY/GLB export
│   └── temporal_smoothing/       # 3-phase smoothing pipeline
│       ├── __init__.py           # TemporalSmoother orchestrator
│       ├── identity_lock.py      # Phase 1: Beta locking
│       ├── motion_smooth.py      # Phase 2: One-Euro filter
│       ├── physical_constraints.py # Phase 3: Footskate removal
│       └── filters/one_euro.py   # Adaptive filter
├── Dockerfile                    # CUDA 12.1 + SAM3D Body
├── docker-compose.yml
├── entrypoint.sh
└── requirements.txt
```

**Strengths**:
- SAM3D Body integration working
- SSE streaming for real-time feedback
- Temporal smoothing (identity lock, motion smooth, foot contact)
- Docker + RunPod ready

**Gaps**:
- No identity tracking across frames (person IDs can swap)
- No occlusion handling (fails on partially hidden bodies)
- Camera-space output only (no world grounding)
- No ground plane estimation

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAM-Body4D Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Video Input                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │   SAM 3     │  Masklet Generator                             │
│  │  (Video)    │  - Identity-consistent person masks            │
│  └─────────────┘  - Handles disappearance/reappearance          │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │ Diffusion-  │  Occlusion-Aware Refinement                    │
│  │    VAS      │  - Detect occluded regions (IoU < 0.7)         │
│  └─────────────┘  - Inpaint missing body parts                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  SAM3D      │  Mask-Guided HMR                               │
│  │   Body      │  - Per-person mesh recovery                    │
│  └─────────────┘  - Batched multi-person inference              │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  Temporal   │  Existing Nvwa module                          │
│  │  Smoothing  │  - Beta lock (first frame)                     │
│  └─────────────┘  - One-Euro filter                             │
│       │           - Foot contact detection                      │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │   World     │  NEW: Ground plane + pelvis lock               │
│  │  Grounding  │  - Floor at Y=0                                │
│  └─────────────┘  - Gravity alignment                           │
│       │                                                         │
│       ▼                                                         │
│  SMPL Sequence (World Space)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Integration

### 1.1 Add SAM3 Video Segmentation

**Purpose**: Identity-consistent person tracking across frames

**New file**: `app/segmentation/sam3_tracker.py`

```python
class SAM3Tracker:
    """
    Video segmentation with SAM3 for identity tracking.

    Each person gets a consistent ID across all frames,
    even through occlusions and re-appearances.
    """

    def __init__(self, model_path: str = "sam3/sam3.pt"):
        self.model = load_sam3(model_path)

    def track_video(self, frames: List[np.ndarray]) -> List[PersonMasklet]:
        """
        Returns list of PersonMasklet, each containing:
        - person_id: int (consistent across frames)
        - masks: Dict[frame_idx, np.ndarray] (binary mask per frame)
        - bbox_sequence: Dict[frame_idx, BBox]
        """
        pass
```

**Dependencies to add**:
```
sam3>=1.0.0  # Meta's SAM3 video segmentation
```

### 1.2 Add Occlusion Handler (Optional)

**Purpose**: Recover body parts hidden by occlusion

**New file**: `app/segmentation/occlusion_handler.py`

```python
class OcclusionHandler:
    """
    Detect and handle occluded body parts using Diffusion-VAS.

    When IoU between original and completed mask < 0.7,
    person is considered occluded.
    """

    def __init__(self):
        self.amodal_model = load_diffusion_vas_amodal()
        self.completion_model = load_diffusion_vas_completion()

    def refine_masks(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool]]:
        """
        Returns:
        - refined_frames: with inpainted body parts
        - refined_masks: completed masks
        - occlusion_flags: True where occlusion detected
        """
        pass
```

**Dependencies to add**:
```
diffusers>=0.25.0  # For Diffusion-VAS
```

**Note**: This is optional for Phase 1. Can skip if occlusion is rare in target videos.

### 1.3 Add World Grounding Module

**Purpose**: Transform camera-space output to world coordinates

**New file**: `app/grounding/world_grounding.py`

```python
class WorldGrounding:
    """
    Transform SMPL sequences from camera space to world space.

    For static camera:
    - Estimate ground plane from foot positions
    - Lock pelvis height during standing
    - Align gravity to -Y axis
    """

    SMPL_LEFT_FOOT = 10
    SMPL_RIGHT_FOOT = 11
    SMPL_PELVIS = 0

    def __init__(
        self,
        ground_offset: float = 0.0,  # Additional offset from detected floor
        pelvis_lock_threshold: float = 0.02,  # Max foot velocity for standing
    ):
        self.ground_offset = ground_offset
        self.pelvis_lock_threshold = pelvis_lock_threshold

    def ground_sequence(
        self,
        smpl_sequence: List[Dict],
        vertices_sequence: List[np.ndarray],
        fps: float = 30.0,
    ) -> List[Dict]:
        """
        Transform sequence to world coordinates.

        Steps:
        1. Find minimum foot height across all frames
        2. Offset all translations so min_foot = ground_offset
        3. Detect standing frames (low foot velocity)
        4. Lock pelvis Y during standing phases
        """
        # Step 1: Find ground plane
        min_foot_y = float('inf')
        for vertices in vertices_sequence:
            left_foot_y = vertices[self.SMPL_LEFT_FOOT, 1]
            right_foot_y = vertices[self.SMPL_RIGHT_FOOT, 1]
            min_foot_y = min(min_foot_y, left_foot_y, right_foot_y)

        ground_y = min_foot_y - self.ground_offset

        # Step 2: Offset all frames
        grounded_sequence = []
        for params in smpl_sequence:
            new_params = params.copy()
            transl = np.array(params['transl']).copy()
            transl[1] -= ground_y  # Y is up
            new_params['transl'] = transl
            grounded_sequence.append(new_params)

        # Step 3-4: Pelvis lock during standing (TODO)
        # ...

        return grounded_sequence
```

---

## Phase 2: Pipeline Integration

### 2.1 New Unified Pipeline

**New file**: `app/pipeline/body4d_pipeline.py`

```python
class Body4DPipeline:
    """
    Complete video-to-3D pipeline combining:
    - SAM3 tracking
    - Optional occlusion handling
    - SAM3D Body mesh recovery
    - Temporal smoothing
    - World grounding
    """

    def __init__(
        self,
        enable_tracking: bool = True,
        enable_occlusion: bool = False,  # Optional, slower
        enable_grounding: bool = True,
        smoothing_preset: str = "balanced",
    ):
        # Load models
        self.sam3_tracker = SAM3Tracker() if enable_tracking else None
        self.occlusion_handler = OcclusionHandler() if enable_occlusion else None
        self.sam3d_body = load_sam3d_body()
        self.temporal_smoother = TemporalSmoother(preset=smoothing_preset)
        self.world_grounding = WorldGrounding() if enable_grounding else None

    async def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> AsyncGenerator[ProcessingEvent, None]:
        """
        Stream processing events as video is processed.

        Yields events:
        - init: faces, total_frames
        - tracking: person IDs detected
        - frame: per-frame SMPL results
        - smoothed: final smoothed sequence
        - grounded: world-space sequence
        - complete: done
        """
        # Extract frames
        frames, fps, timestamps = extract_video_frames(video_path, max_frames)

        # Step 1: Track people (if enabled)
        if self.sam3_tracker:
            yield ProcessingEvent(type="tracking", status="started")
            person_masklets = self.sam3_tracker.track_video(frames)
            yield ProcessingEvent(type="tracking", num_people=len(person_masklets))
        else:
            # Fallback: no tracking, process all detected people per frame
            person_masklets = None

        # Step 2: Process each person
        all_people_results = {}

        for person in person_masklets or [None]:
            person_id = person.person_id if person else 0

            # Get frames for this person (masked or full)
            person_frames = self._get_person_frames(frames, person)

            # Step 2a: Occlusion handling (optional)
            if self.occlusion_handler:
                person_frames, _, _ = self.occlusion_handler.refine_masks(
                    person_frames, person.masks if person else None
                )

            # Step 2b: SAM3D Body inference
            smpl_sequence = []
            vertices_sequence = []

            for i, frame in enumerate(person_frames):
                result = self.sam3d_body.process_one_image(frame)
                smpl_params = extract_smpl_params(result)
                smpl_sequence.append(smpl_params)
                vertices_sequence.append(result.get('pred_vertices'))

                yield ProcessingEvent(
                    type="frame",
                    person_id=person_id,
                    frame_index=i,
                    total_frames=len(person_frames),
                )

            # Step 3: Temporal smoothing
            smoothed = self.temporal_smoother.process_sequence(
                smpl_sequence,
                fps=fps,
            )

            # Step 4: World grounding
            if self.world_grounding:
                grounded = self.world_grounding.ground_sequence(
                    smoothed['final_sequence'],
                    vertices_sequence,
                    fps=fps,
                )
            else:
                grounded = smoothed['final_sequence']

            all_people_results[person_id] = {
                'smpl_sequence': grounded,
                'smoothing_summary': smoothed.get('summary'),
            }

        yield ProcessingEvent(
            type="complete",
            results=all_people_results,
        )
```

### 2.2 Update API Endpoints

**Modify**: `app/sam3d_api.py`

Add new endpoint that uses the unified pipeline:

```python
@app.post("/sam3d/body/process_video_v2")
async def process_video_v2(
    video: UploadFile = File(...),
    max_frames: Optional[int] = None,
    enable_tracking: bool = True,
    enable_occlusion: bool = False,
    enable_grounding: bool = True,
    smoothing_preset: str = "balanced",
):
    """
    V2 endpoint with full SAM-Body4D pipeline.

    Features:
    - Identity tracking (SAM3)
    - Occlusion handling (Diffusion-VAS, optional)
    - Temporal smoothing (One-Euro + foot contact)
    - World grounding (floor at Y=0)
    """
    pipeline = Body4DPipeline(
        enable_tracking=enable_tracking,
        enable_occlusion=enable_occlusion,
        enable_grounding=enable_grounding,
        smoothing_preset=smoothing_preset,
    )

    return StreamingResponse(
        pipeline.process_video(video_path, max_frames),
        media_type="text/event-stream",
    )
```

---

## Phase 3: RunPod Deployment

### 3.1 Directory Structure for Clean Clone

```
nvwa/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── sam3d_api.py
│   │   ├── config.py
│   │   ├── storage.py
│   │   ├── export_animation.py
│   │   ├── segmentation/           # NEW
│   │   │   ├── __init__.py
│   │   │   ├── sam3_tracker.py
│   │   │   └── occlusion_handler.py
│   │   ├── grounding/              # NEW
│   │   │   ├── __init__.py
│   │   │   └── world_grounding.py
│   │   ├── pipeline/               # NEW
│   │   │   ├── __init__.py
│   │   │   └── body4d_pipeline.py
│   │   └── temporal_smoothing/     # EXISTING
│   │       └── ...
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── setup_runpod.sh             # NEW: One-command setup
│   └── README.md
└── ...
```

### 3.2 RunPod Setup Script

**New file**: `backend/setup_runpod.sh`

```bash
#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Nvwa Backend - RunPod Setup"
echo "=========================================="

# Check for required environment variables
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: HUGGINGFACE_HUB_TOKEN not set"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Create workspace directories
mkdir -p /workspace/{checkpoints,data,logs}

# Clone SAM3D Body (if not already present)
if [[ ! -d "/workspace/sam-3d-body" ]]; then
    echo "Cloning SAM 3D Body..."
    git clone https://github.com/facebookresearch/sam-3d-body.git /workspace/sam-3d-body
    pip install -e /workspace/sam-3d-body
fi

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt

# Download model checkpoints
echo "Downloading model checkpoints..."
python3 << 'EOF'
import os
from huggingface_hub import login, snapshot_download

login(token=os.environ['HUGGINGFACE_HUB_TOKEN'])

# SAM 3D Body
print("Downloading SAM 3D Body...")
snapshot_download(
    "facebook/sam-3d-body-dinov3",
    cache_dir="/workspace/checkpoints",
    local_dir="/workspace/checkpoints/sam-3d-body-dinov3",
)

# SAM3 (if using tracking)
# print("Downloading SAM3...")
# snapshot_download("facebook/sam3", ...)

print("Done!")
EOF

# Set environment
export PYTHONPATH="/workspace/sam-3d-body:$PYTHONPATH"
export CHECKPOINTS_DIR="/workspace/checkpoints"
export DATA_DIR="/workspace/data"

echo "=========================================="
echo "Setup complete! Start server with:"
echo "  uvicorn app.sam3d_api:app --host 0.0.0.0 --port 8000"
echo "=========================================="
```

### 3.3 Updated Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git wget curl ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone and install SAM 3D Body
RUN git clone https://github.com/facebookresearch/sam-3d-body.git /opt/sam-3d-body && \
    pip install -e /opt/sam-3d-body

# Install backend dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Environment
ENV PYTHONPATH="/opt/sam-3d-body:$PYTHONPATH"
ENV HF_HOME="/models"
ENV CHECKPOINTS_DIR="/workspace/checkpoints"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "app.sam3d_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Dependencies Summary

### New Dependencies to Add

```
# requirements.txt additions

# SAM3 Video Segmentation (for tracking)
# sam3>=1.0.0  # Check exact package name when released

# Diffusion models (for occlusion handling)
diffusers>=0.25.0
accelerate>=0.25.0

# Already have these (keep)
torch>=2.5.0
torchvision>=0.20.0
transformers>=4.46.0
huggingface-hub>=0.26.0
```

### Model Checkpoints Required

| Model | Size | Source | Required |
|-------|------|--------|----------|
| SAM 3D Body | ~2GB | HuggingFace (gated) | Yes |
| SAM3 | ~2GB | HuggingFace | For tracking |
| Diffusion-VAS Amodal | ~2GB | HuggingFace | Optional |
| Diffusion-VAS Completion | ~2GB | HuggingFace | Optional |
| MoGe-2 | ~1GB | HuggingFace | Optional |
| Depth Anything V2 | ~1GB | HuggingFace | Optional |

**Minimum (no tracking/occlusion)**: ~2GB
**Full pipeline**: ~10GB

---

## Implementation Order

### Sprint 1: World Grounding (Highest Impact)
1. Create `app/grounding/world_grounding.py`
2. Add ground plane estimation
3. Add pelvis height lock
4. Integrate into existing API
5. Test with static camera videos

### Sprint 2: Identity Tracking (If Needed)
1. Integrate SAM3 video segmentation
2. Create `app/segmentation/sam3_tracker.py`
3. Update pipeline to use tracked masks
4. Test multi-person scenarios

### Sprint 3: Occlusion Handling (Optional)
1. Integrate Diffusion-VAS
2. Create `app/segmentation/occlusion_handler.py`
3. Add IoU-based occlusion detection
4. Test with occluded subjects

### Sprint 4: Unified Pipeline
1. Create `app/pipeline/body4d_pipeline.py`
2. Add V2 API endpoint
3. Update frontend to use new endpoint
4. End-to-end testing

---

## Testing Plan

### Unit Tests
- `test_world_grounding.py`: Ground plane detection, pelvis lock
- `test_sam3_tracker.py`: Identity consistency across frames
- `test_pipeline.py`: Full pipeline integration

### Integration Tests
- Static camera, single person
- Static camera, multiple people
- Moving subject (walking, sports)
- Occluded subject (behind object)

### Performance Benchmarks
- Target: <500ms per frame on RTX 4090
- Memory: <16GB VRAM for full pipeline
- Throughput: Real-time for 30fps video

---

## Open Questions

1. **SAM3 availability**: Is the video segmentation model publicly available yet?
2. **Diffusion-VAS**: What's the exact checkpoint/repo for this model?
3. **Moving camera**: Do we need D4RT integration for camera pose estimation?
4. **VR export**: What format does the frontend expect for WebXR playback?

---

## References

- [SAM-Body4D Paper](https://arxiv.org/html/2512.08406)
- [SAM-Body4D GitHub](https://github.com/gaomingqi/sam-body4d)
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
- [GVHMR (World Grounding)](https://github.com/zju3dv/GVHMR)
- [D4RT (4D Reconstruction)](https://arxiv.org/html/2512.08924)
