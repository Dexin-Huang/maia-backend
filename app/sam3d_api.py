"""
SAM 3D Body API Server
Provides REST endpoints for volumetric video to 3D human reconstruction
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, AsyncGenerator
import asyncio
import json
import imghdr

import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

# Use orjson for 3-10x faster JSON serialization (critical for large vertex arrays)
try:
    import orjson
    def json_dumps(obj) -> str:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
except ImportError:
    import json
    def json_dumps(obj) -> str:
        return json.dumps(obj)

from scipy.signal import butter, filtfilt

# Configuration via environment variables
SAM3D_BODY_PATH = os.environ.get('SAM3D_BODY_PATH', '/workspace/sam-3d-body')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', '/workspace/checkpoints')


def smooth_vertices_butterworth(
    all_vertices: np.ndarray,
    fps: float = 30.0,
    cutoff_hz: float = 5.0
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to vertex positions across time.

    This removes high-frequency jitter while preserving the natural motion.
    Sports motion typically has meaningful content below 5-6Hz.

    Args:
        all_vertices: [T, V, 3] array of vertex positions over time
        fps: Frame rate of the video
        cutoff_hz: Cutoff frequency (lower = smoother, 5Hz good for sports)

    Returns:
        Smoothed vertices with same shape [T, V, 3]
    """
    T, V, D = all_vertices.shape

    # Need at least ~10 frames for stable filtering
    if T < 10:
        return all_vertices

    # Design 2nd order Butterworth low-pass filter
    nyquist = fps / 2
    normalized_cutoff = cutoff_hz / nyquist

    # Clamp to valid range (0, 1)
    normalized_cutoff = min(max(normalized_cutoff, 0.01), 0.99)

    b, a = butter(N=2, Wn=normalized_cutoff, btype='low')

    # Apply filter to each vertex dimension
    smoothed = np.zeros_like(all_vertices)

    for v in range(V):
        for d in range(D):
            # filtfilt applies filter forward and backward (zero phase delay)
            smoothed[:, v, d] = filtfilt(b, a, all_vertices[:, v, d])

    return smoothed


def smooth_frame_sequence(
    frames_data: list,
    fps: float = 30.0,
    cutoff_hz: float = 5.0
) -> list:
    """
    Smooth a sequence of frames (handles multi-person).

    Args:
        frames_data: List of frames, each frame is list of people dicts with 'vertices'
        fps: Frame rate
        cutoff_hz: Smoothing cutoff frequency

    Returns:
        Smoothed frames_data with same structure
    """
    if len(frames_data) < 10:
        return frames_data

    # For now, assume consistent number of people across frames
    # (In production, would need tracking to match identities)
    num_people = len(frames_data[0]) if frames_data[0] else 0

    if num_people == 0:
        return frames_data

    # Smooth each person's trajectory independently
    for person_idx in range(num_people):
        # Collect vertices for this person across all frames
        person_vertices = []
        valid_frames = []

        for frame_idx, frame in enumerate(frames_data):
            if frame and len(frame) > person_idx and frame[person_idx].get('vertices') is not None:
                person_vertices.append(frame[person_idx]['vertices'])
                valid_frames.append(frame_idx)

        if len(person_vertices) < 10:
            continue

        # Stack into [T, V, 3]
        vertices_array = np.array(person_vertices)

        # Apply smoothing
        smoothed_vertices = smooth_vertices_butterworth(vertices_array, fps, cutoff_hz)

        # Write back
        for i, frame_idx in enumerate(valid_frames):
            frames_data[frame_idx][person_idx]['vertices'] = smoothed_vertices[i]

    return frames_data

# Add SAM 3D Body path (configurable via SAM3D_BODY_PATH env var)
sys.path.insert(0, SAM3D_BODY_PATH)

# Create FastAPI app
app = FastAPI(
    title="SAM 3D Body API",
    description="API for volumetric video to 3D human reconstruction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
sam3d_body_estimator = None
temporal_smoother = None


class HealthResponse(BaseModel):
    status: str
    models: dict


def load_sam3d_body():
    """Load SAM 3D Body model (lazy loading)"""
    global sam3d_body_estimator

    if sam3d_body_estimator is None:
        print("Loading SAM 3D Body model...")
        from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_cfg = load_sam_3d_body_hf(
            "facebook/sam-3d-body-dinov3",
            device=device,
            cache_dir=CHECKPOINTS_DIR
        )
        sam3d_body_estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg
        )
        print("SAM 3D Body model loaded!")

    return sam3d_body_estimator


def load_temporal_smoother(preset: str = "balanced"):
    """Load Temporal Smoother (lazy loading)"""
    global temporal_smoother

    if temporal_smoother is None:
        print(f"Loading Temporal Smoother (preset={preset})...")
        # Add temporal_smoothing to path
        import sys
        from pathlib import Path
        backend_path = Path(__file__).parent.parent
        sys.path.insert(0, str(backend_path / "app"))

        from temporal_smoothing import TemporalSmoother

        temporal_smoother = TemporalSmoother(
            preset=preset,
            enable_identity_lock=True,
            enable_motion_smoothing=True,
            enable_physical_constraints=True,
        )
        print("Temporal Smoother loaded!")

    return temporal_smoother


def is_image_file(file_data: bytes) -> bool:
    """Check if file data is an image"""
    try:
        img_type = imghdr.what(None, h=file_data[:32])
        return img_type in ('jpeg', 'jpg', 'png', 'gif', 'bmp')
    except:
        return False


def is_video_file(filename: str) -> bool:
    """Check if filename indicates a video file"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return Path(filename).suffix.lower() in video_extensions


def extract_smpl_params(result: dict) -> dict:
    """
    Extract SMPL parameters from SAM 3D Body output format.

    Note: We keep numpy arrays as-is - orjson serializes them directly,
    avoiding the overhead of .tolist() conversion.
    """
    if not result:
        return None

    betas = result.get('shape_params')
    body_pose = result.get('body_pose_params')

    # Only return if we have essential parameters
    if betas is None or body_pose is None:
        return None

    return {
        'betas': betas,                          # Shape [10]
        'body_pose': body_pose,                  # Body pose [69]
        'global_orient': result.get('global_rot'),    # Global orientation [3]
        'transl': result.get('pred_cam_t'),          # Translation [3]
        'vertices': result.get('pred_vertices'),      # Vertex positions [V, 3]
    }


def extract_video_frames(video_path: str, max_frames: Optional[int] = None) -> tuple:
    """
    Extract frames from video file

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None = all frames)

    Returns:
        Tuple of (frames, fps, timestamps) where:
        - frames: List of numpy arrays (H, W, 3)
        - fps: Frame rate of video
        - timestamps: List of timestamps for each frame
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        # Calculate timestamp
        timestamp = frame_idx / fps
        timestamps.append(timestamp)

        frame_idx += 1

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    return frames, fps, timestamps


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "sam3d_body": "loaded" if sam3d_body_estimator is not None else "not_loaded"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "sam3d_body": "loaded" if sam3d_body_estimator is not None else "not_loaded"
        }
    }


@app.get("/sam3d/body/template_mesh")
async def get_template_mesh():
    """
    Get SMPL template mesh topology (faces)

    Returns:
        JSON with faces array: shape (F, 3) where F is number of triangles
    """
    try:
        estimator = load_sam3d_body()

        # Extract faces from the model
        faces = estimator.faces  # numpy array of shape (F, 3)

        return {
            "faces": faces.tolist(),
            "num_faces": len(faces),
            "num_vertices": 18540,  # MHR model has 18540 vertices
            "format": "triangle_indices"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load template mesh: {str(e)}")


def process_batch_sync(estimator, frames_batch: list, start_idx: int) -> list:
    """
    Process a batch of frames synchronously (runs in thread pool).

    Args:
        estimator: SAM3D Body estimator
        frames_batch: List of frame numpy arrays
        start_idx: Starting frame index for this batch

    Returns:
        List of (frame_idx, people_list) tuples
    """
    results = []
    for i, frame_np in enumerate(frames_batch):
        frame_idx = start_idx + i
        result = estimator.process_one_image(frame_np)

        # Handle list return (multiple people detected)
        people = []
        if isinstance(result, list):
            for person_result in result:
                smpl_params = extract_smpl_params(person_result)
                if smpl_params is not None:
                    people.append(smpl_params)
        else:
            smpl_params = extract_smpl_params(result)
            if smpl_params is not None:
                people.append(smpl_params)

        results.append((frame_idx, people))

    return results


async def process_stream_generator(
    frames: list,
    fps: float,
    timestamps: list,
    preset: str,
    total_frames: int
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted events for each processed frame.

    Flow:
    1. Stream raw frames as they're processed (immediate feedback)
    2. After all frames complete, apply Butterworth smoothing
    3. Send smoothed frames in a single 'smoothed' event

    Performance optimizations:
    - Uses orjson for 3-10x faster serialization of large vertex arrays
    - Batch processing with configurable batch size
    - Numpy arrays passed directly to orjson (no .tolist() conversion)
    """
    # Load model
    estimator = await asyncio.to_thread(load_sam3d_body)

    # Get mesh faces (constant for all frames) - keep as numpy for orjson
    faces = estimator.faces if hasattr(estimator, 'faces') else None

    # Batch size - balance between latency and throughput
    BATCH_SIZE = 4

    # Send faces immediately so frontend can prepare
    if faces is not None:
        yield f"data: {json_dumps({'type': 'init', 'faces': faces, 'total_frames': total_frames})}\n\n"

    # Collect all frames for post-processing smoothing
    all_frames_data = [None] * total_frames

    # Process in batches
    processed_count = 0
    for batch_start in range(0, total_frames, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_frames)
        frames_batch = frames[batch_start:batch_end]

        # Process batch in thread pool
        batch_results = await asyncio.to_thread(
            process_batch_sync,
            estimator,
            frames_batch,
            batch_start
        )

        # Yield results for each frame in batch (raw, unsmoothed for live preview)
        for frame_idx, people in batch_results:
            if len(people) == 0:
                yield f"data: {json_dumps({'type': 'error', 'frame_index': frame_idx, 'message': 'No person detected'})}\n\n"
                all_frames_data[frame_idx] = []
                continue

            # Store for smoothing
            all_frames_data[frame_idx] = people

            frame_event = {
                'type': 'frame',
                'frame_index': frame_idx,
                'total_frames': total_frames,
                'people': people,
                'num_people': len(people),
            }

            yield f"data: {json_dumps(frame_event)}\n\n"
            processed_count += 1

        # Progress update after each batch
        yield f"data: {json_dumps({'type': 'progress', 'processed': processed_count, 'total': total_frames, 'percent': round((processed_count / total_frames) * 100, 1)})}\n\n"

    # Apply temporal smoothing after all frames collected
    valid_frames = [f for f in all_frames_data if f is not None and len(f) > 0]

    if len(valid_frames) >= 10:
        yield f"data: {json_dumps({'type': 'smoothing', 'status': 'started', 'num_frames': len(valid_frames)})}\n\n"

        # Run smoothing in thread pool (CPU-bound numpy operations)
        smoothed_frames = await asyncio.to_thread(
            smooth_frame_sequence,
            all_frames_data,
            fps,
            5.0  # 5Hz cutoff - good for sports motion
        )

        # Send smoothed data - frontend will replace raw frames with these
        yield f"data: {json_dumps({'type': 'smoothed', 'frames': smoothed_frames})}\n\n"

    # Completion event
    yield f"data: {json_dumps({'type': 'complete', 'success': True, 'total_frames': total_frames, 'smoothed': len(valid_frames) >= 10})}\n\n"


@app.post("/sam3d/body/process_stream")
async def process_stream(
    file: UploadFile = File(...),
    max_frames: Optional[int] = None,
    preset: str = "balanced"
):
    """
    Unified streaming endpoint for image or video processing

    Streams frame-by-frame results as they're processed (SSE format):
    - For images: Streams 1 frame
    - For videos: Streams N frames

    Args:
        file: Image (JPG/PNG) or Video (MP4/AVI/MOV) file
        max_frames: Maximum frames to process for videos (None = all)
        preset: (deprecated, kept for compatibility)

    Returns:
        Server-Sent Events stream with frame data and progress
    """
    try:
        # Read file data
        file_data = await file.read()

        # Detect file type
        is_image = is_image_file(file_data)
        is_video = is_video_file(file.filename or "")

        if is_image:
            # Single image - treat as 1-frame video
            pil_image = Image.open(io.BytesIO(file_data)).convert("RGB")
            frames = [np.array(pil_image)]
            fps = 30.0  # Dummy FPS for single image
            timestamps = [0.0]
            total_frames = 1

        elif is_video:
            # Video file - extract frames
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name

            try:
                frames, fps, timestamps = await asyncio.to_thread(
                    extract_video_frames,
                    temp_path,
                    max_frames
                )
                total_frames = len(frames)
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload an image (JPG/PNG) or video (MP4/AVI/MOV)."
            )

        # Return streaming response
        return StreamingResponse(
            process_stream_generator(frames, fps, timestamps, preset, total_frames),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


def _predict_body_mesh_sync(image_np: np.ndarray, save_mesh: bool) -> dict:
    """
    Synchronous GPU inference for SAM 3D Body (runs in threadpool)

    Args:
        image_np: Numpy array of the image
        save_mesh: Whether to save the mesh to a file

    Returns:
        Dictionary with results
    """
    # Load model
    estimator = load_sam3d_body()

    # Run inference (GPU work)
    result = estimator.process_one_image(image_np)

    # Handle different return formats (list of results or single result)
    if isinstance(result, list):
        # Multiple people detected, use first one
        if len(result) == 0:
            return {
                "success": False,
                "num_vertices": 0,
                "has_smpl_params": False,
                "mesh_path": None,
                "error": "No person detected in image"
            }
        result = result[0]

    # Save mesh if requested
    mesh_path = None
    if save_mesh and result.get('vertices') is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply', dir='/tmp') as f:
            mesh_path = f.name
            # TODO: Save mesh using trimesh or similar
            # For now, just return the path

    return {
        "success": True,
        "num_vertices": len(result.get('vertices', [])) if result.get('vertices') is not None else 0,
        "has_smpl_params": 'smpl_params' in result,
        "mesh_path": mesh_path
    }


@app.post("/sam3d/body/predict")
async def predict_body_mesh(
    image: UploadFile = File(...),
    save_mesh: bool = True
):
    """
    SAM 3D Body: Human mesh reconstruction from a single image

    Args:
        image: Input image file containing a person
        save_mesh: Whether to save the mesh to a file (default: True)

    Returns:
        JSON with SMPL parameters, vertices, and optionally mesh file path
    """
    try:
        # Read and process image (fast, can be async)
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(pil_image)

        # Offload GPU inference to thread pool to avoid blocking event loop
        result = await asyncio.to_thread(_predict_body_mesh_sync, image_np, save_mesh)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def _predict_video_mesh_sync(
    frames: list,
    fps: float,
    timestamps: list,
    preset: str,
    apply_temporal_smoothing: bool
) -> dict:
    """
    Synchronous GPU inference for video sequence (runs in threadpool)

    Args:
        frames: List of frame arrays
        fps: Frame rate
        timestamps: Frame timestamps
        preset: Temporal smoothing preset
        apply_temporal_smoothing: Whether to apply temporal smoothing

    Returns:
        Dictionary with results
    """
    # Load model
    estimator = load_sam3d_body()

    # Process all frames through SAM 3D Body
    smpl_params_sequence = []
    joints_3d_sequence = []

    for frame_idx, frame_np in enumerate(frames):
        result = estimator.process_one_image(frame_np)

        if 'smpl_params' in result:
            smpl_params_sequence.append(result['smpl_params'])

        if 'joints_3d' in result:
            joints_3d_sequence.append(result['joints_3d'])

    # Convert joints to numpy array if available
    joints_3d = None
    if joints_3d_sequence:
        joints_3d = np.stack(joints_3d_sequence, axis=0)  # (T, 24, 3)

    # Apply temporal smoothing if requested and have enough frames
    smoothed_sequence = smpl_params_sequence
    smoothing_result = None

    if apply_temporal_smoothing and len(smpl_params_sequence) > 1:
        smoother = load_temporal_smoother(preset=preset)
        smoothing_result = smoother.process_sequence(
            smpl_params_sequence,
            joints_3d=joints_3d,
            fps=fps,
            frame_timestamps=timestamps
        )
        smoothed_sequence = smoothing_result['final_sequence']

    return {
        "success": True,
        "num_frames": len(frames),
        "num_smpl_params": len(smoothed_sequence),
        "temporal_smoothing_applied": apply_temporal_smoothing and len(smpl_params_sequence) > 1,
        "smoothing_summary": smoothing_result['summary'] if smoothing_result else None,
        "smpl_sequence": smoothed_sequence,  # Include the full sequence
    }


@app.post("/sam3d/body/predict_video")
async def predict_video_mesh(
    video: UploadFile = File(...),
    max_frames: Optional[int] = None,
    preset: str = "balanced",
    apply_temporal_smoothing: bool = True
):
    """
    SAM 3D Body: Human mesh reconstruction from video with temporal smoothing

    Args:
        video: Input video file containing a person
        max_frames: Maximum number of frames to process (None = all frames)
        preset: Temporal smoothing preset ("conservative", "balanced", "responsive")
        apply_temporal_smoothing: Whether to apply temporal smoothing (default: True)

    Returns:
        JSON with smoothed SMPL parameters sequence and temporal smoothing diagnostics
    """
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_data = await video.read()
            temp_video.write(video_data)
            temp_video_path = temp_video.name

        # Extract frames from video (fast, can be async)
        frames, fps, timestamps = await asyncio.to_thread(
            extract_video_frames,
            temp_video_path,
            max_frames
        )

        # Clean up temp video file
        os.unlink(temp_video_path)

        # Offload GPU inference to thread pool to avoid blocking event loop
        result = await asyncio.to_thread(
            _predict_video_mesh_sync,
            frames,
            fps,
            timestamps,
            preset,
            apply_temporal_smoothing
        )

        return result

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_video_path' in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup (optional - can be lazy loaded)"""
    print("SAM 3D API Server starting...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/sam3d/body/export_animation")
async def export_animation(
    frames_data: list,
    format: str = "ply_sequence",
    fps: int = 30
):
    """
    Export animation in various formats.
    
    Args:
        frames_data: List of frame data (FrameData format with 'people' array)
        format: Export format ('ply_sequence', 'glb_single')
        fps: Frames per second (for timing metadata)
    
    Returns:
        File download (ZIP or GLB)
    """
    try:
        if not frames_data or len(frames_data) == 0:
            raise HTTPException(status_code=400, detail="No frames provided")
        
        # Load model to get faces
        estimator = await asyncio.to_thread(load_sam3d_body)
        faces = estimator.faces
        
        if format == "ply_sequence":
            # Export as ZIP of PLY files
            zip_bytes = await asyncio.to_thread(export_ply_sequence, frames_data, faces)
            
            return StreamingResponse(
                io.BytesIO(zip_bytes),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=animation_{len(frames_data)}frames.zip"
                }
            )
        
        elif format == "glb_single":
            # Export first frame as GLB
            glb_bytes = await asyncio.to_thread(export_single_glb, frames_data, faces, 0)
            
            return StreamingResponse(
                io.BytesIO(glb_bytes),
                media_type="model/gltf-binary",
                headers={
                    "Content-Disposition": f"attachment; filename=frame_0.glb"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ============================================
# V2 Endpoints: Body4D Pipeline with Grounding
# ============================================

@app.post("/sam3d/body/process_video_v2")
async def process_video_v2(
    file: UploadFile = File(...),
    max_frames: Optional[int] = None,
    enable_tracking: bool = False,
    enable_grounding: bool = True,
    smoothing_preset: str = "balanced",
):
    """
    V2 endpoint with full Body4D pipeline including world grounding.

    Features:
    - Identity tracking (SAM3, optional)
    - Temporal smoothing (One-Euro + foot contact)
    - World grounding (floor at Y=0)

    This is the recommended endpoint for VR/AR "Matrix replay" use cases.

    Args:
        file: Video file (MP4/AVI/MOV) or image (JPG/PNG)
        max_frames: Maximum frames to process (None = all)
        enable_tracking: Use SAM3 for multi-person tracking
        enable_grounding: Transform to world coordinates (floor at Y=0)
        smoothing_preset: "conservative", "balanced", or "responsive"

    Returns:
        Server-Sent Events stream with grounded SMPL sequences
    """
    try:
        from .pipeline import Body4DPipeline

        # Read file data
        file_data = await file.read()

        # Detect file type and extract frames
        is_img = is_image_file(file_data)
        is_vid = is_video_file(file.filename or "")

        if is_img:
            pil_image = Image.open(io.BytesIO(file_data)).convert("RGB")
            frames = [np.array(pil_image)]
            fps = 30.0
        elif is_vid:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name

            try:
                frames, fps, _ = await asyncio.to_thread(
                    extract_video_frames,
                    temp_path,
                    max_frames
                )
            finally:
                os.unlink(temp_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload image (JPG/PNG) or video (MP4/AVI/MOV)."
            )

        # Create pipeline
        pipeline = Body4DPipeline(
            enable_tracking=enable_tracking,
            enable_smoothing=True,
            enable_grounding=enable_grounding,
            smoothing_preset=smoothing_preset,
        )

        # Stream processing events
        async def event_generator():
            async for event in pipeline.process_video(frames, fps):
                yield f"data: {json_dumps(event.to_dict())}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/sam3d/body/ground_sequence")
async def ground_sequence_endpoint(
    smpl_sequence: list,
    vertices_sequence: Optional[list] = None,
    fps: float = 30.0,
    up_axis: str = "y",
    stabilize_pelvis: bool = True,
):
    """
    Apply world grounding to an existing SMPL sequence.

    Args:
        smpl_sequence: List of SMPL parameter dicts
        vertices_sequence: Optional list of vertex arrays
        fps: Frame rate for velocity calculations
        up_axis: "y" or "z" for up direction
        stabilize_pelvis: Lock pelvis height during standing

    Returns:
        Grounded sequence with floor at Y=0
    """
    try:
        from .grounding import WorldGrounding

        grounder = WorldGrounding(up_axis=up_axis)

        verts = None
        if vertices_sequence:
            verts = [np.array(v) if v is not None else None for v in vertices_sequence]

        result = await asyncio.to_thread(
            grounder.ground_sequence,
            smpl_sequence,
            verts,
            fps,
            stabilize_pelvis,
        )

        return {
            "success": True,
            "grounded_sequence": result['grounded_sequence'],
            "ground_height": result['ground_height'],
            "standing_frames": result['standing_frames'],
            "total_frames": result['total_frames'],
            "max_pelvis_adjustment": result['max_pelvis_adjustment'],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Grounding failed: {str(e)}")
