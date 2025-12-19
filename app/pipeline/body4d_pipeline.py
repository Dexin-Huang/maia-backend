"""
Body4D Pipeline

Unified video-to-3D human reconstruction pipeline.

Combines:
1. SAM3 video segmentation (identity tracking)
2. SAM3D Body mesh recovery (per-person SMPL)
3. Temporal smoothing (One-Euro + foot contact)
4. World grounding (floor at Y=0)

This provides the full SAM-Body4D-style pipeline with world grounding
for VR/AR "Matrix replay" use cases.

Author: Nvwa Team
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, AsyncGenerator, Any, Union
from enum import Enum
import time


class EventType(str, Enum):
    """Types of processing events streamed to client."""
    INIT = "init"
    TRACKING = "tracking"
    FRAME = "frame"
    PROGRESS = "progress"
    SMOOTHING = "smoothing"
    GROUNDING = "grounding"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProcessingEvent:
    """
    Event emitted during pipeline processing.

    Streamed to client via SSE for real-time progress updates.
    """
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            **self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class PersonResult:
    """Results for a single tracked person."""
    person_id: int
    smpl_sequence: List[Dict]  # SMPL params per frame
    vertices_sequence: List[np.ndarray]  # Vertices per frame (optional)
    frame_indices: List[int]  # Which frames this person appears in
    smoothing_summary: Optional[Dict] = None
    grounding_summary: Optional[Dict] = None


class Body4DPipeline:
    """
    Complete video-to-3D pipeline for human mesh recovery.

    Features:
    - Multi-person tracking (SAM3)
    - Per-person mesh recovery (SAM3D Body)
    - Temporal smoothing (One-Euro filter + foot contact)
    - World grounding (floor plane estimation)

    Usage:
        pipeline = Body4DPipeline()

        async for event in pipeline.process_video(frames, fps):
            if event.type == EventType.FRAME:
                # Render frame preview
            elif event.type == EventType.COMPLETE:
                results = event.data['results']
    """

    def __init__(
        self,
        enable_tracking: bool = True,
        enable_smoothing: bool = True,
        enable_grounding: bool = True,
        smoothing_preset: str = "balanced",
        up_axis: str = "y",
        device: str = "cuda",
    ):
        """
        Initialize Body4D Pipeline.

        Args:
            enable_tracking: Use SAM3 for multi-person tracking
            enable_smoothing: Apply temporal smoothing
            enable_grounding: Transform to world coordinates
            smoothing_preset: "conservative", "balanced", or "responsive"
            up_axis: "y" or "z" for world coordinate system
            device: "cuda" or "cpu"
        """
        self.enable_tracking = enable_tracking
        self.enable_smoothing = enable_smoothing
        self.enable_grounding = enable_grounding
        self.smoothing_preset = smoothing_preset
        self.up_axis = up_axis
        self.device = device

        # Lazy-loaded components
        self._tracker = None
        self._sam3d_body = None
        self._smoother = None
        self._grounder = None

    def _load_components(self):
        """Lazy load pipeline components."""
        if self.enable_tracking and self._tracker is None:
            from ..segmentation import SAM3Tracker
            self._tracker = SAM3Tracker(device=self.device)

        if self._sam3d_body is None:
            self._load_sam3d_body()

        if self.enable_smoothing and self._smoother is None:
            from ..temporal_smoothing import TemporalSmoother
            self._smoother = TemporalSmoother(preset=self.smoothing_preset)

        if self.enable_grounding and self._grounder is None:
            from ..grounding import WorldGrounding
            self._grounder = WorldGrounding(up_axis=self.up_axis)

    def _load_sam3d_body(self):
        """Load SAM3D Body model."""
        import sys
        import torch

        # Add SAM3D Body to path if needed
        sam3d_path = "/workspace/sam-3d-body"
        if sam3d_path not in sys.path:
            sys.path.insert(0, sam3d_path)

        try:
            from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator

            device = torch.device(self.device if torch.cuda.is_available() else "cpu")

            model, model_cfg = load_sam_3d_body_hf(
                "facebook/sam-3d-body-dinov3",
                device=device,
                cache_dir="/workspace/checkpoints"
            )

            self._sam3d_body = SAM3DBodyEstimator(
                sam_3d_body_model=model,
                model_cfg=model_cfg
            )
            self._faces = self._sam3d_body.faces

        except Exception as e:
            print(f"Failed to load SAM3D Body: {e}")
            raise

    def _extract_smpl_params(self, result: Dict) -> Optional[Dict]:
        """Extract SMPL parameters from SAM3D Body output."""
        if not result:
            return None

        betas = result.get('shape_params')
        body_pose = result.get('body_pose_params')

        if betas is None or body_pose is None:
            return None

        return {
            'betas': betas,
            'body_pose': body_pose,
            'global_orient': result.get('global_rot'),
            'transl': result.get('pred_cam_t'),
            'vertices': result.get('pred_vertices'),
        }

    async def process_video(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        max_people: Optional[int] = None,
    ) -> AsyncGenerator[ProcessingEvent, None]:
        """
        Process video frames through the full pipeline.

        Args:
            frames: List of video frames (H, W, 3) in RGB
            fps: Frame rate for temporal calculations
            max_people: Maximum number of people to track (None = all)

        Yields:
            ProcessingEvent objects for real-time progress updates
        """
        T = len(frames)

        # Load components
        yield ProcessingEvent(
            type=EventType.INIT,
            data={"status": "loading_models", "total_frames": T}
        )

        await asyncio.to_thread(self._load_components)

        # Send mesh topology
        if hasattr(self, '_faces') and self._faces is not None:
            yield ProcessingEvent(
                type=EventType.INIT,
                data={
                    "status": "ready",
                    "faces": self._faces.tolist() if isinstance(self._faces, np.ndarray) else self._faces,
                    "total_frames": T,
                }
            )

        # Step 1: Track people (if enabled)
        if self.enable_tracking and self._tracker is not None:
            yield ProcessingEvent(
                type=EventType.TRACKING,
                data={"status": "started"}
            )

            tracks = await asyncio.to_thread(self._tracker.track_video, frames)

            if max_people is not None:
                tracks = tracks[:max_people]

            yield ProcessingEvent(
                type=EventType.TRACKING,
                data={
                    "status": "complete",
                    "num_people": len(tracks),
                    "person_ids": [t.person_id for t in tracks],
                }
            )
        else:
            # No tracking - treat as single person, all frames
            from ..segmentation import PersonTrack
            single_track = PersonTrack(
                person_id=0,
                first_frame=0,
                last_frame=T - 1,
            )
            for t in range(T):
                H, W = frames[t].shape[:2]
                single_track.masks[t] = np.ones((H, W), dtype=bool)
                single_track.bboxes[t] = (0, 0, W, H)
            tracks = [single_track]

        # Step 2: Process each person
        all_results: Dict[int, PersonResult] = {}

        for track in tracks:
            person_id = track.person_id
            frame_indices = track.frame_indices

            smpl_sequence = []
            vertices_sequence = []

            # Process frames for this person
            for i, frame_idx in enumerate(frame_indices):
                frame = frames[frame_idx]

                # Get masked/cropped frame if tracking
                if self.enable_tracking:
                    # Use full frame but could use masked version
                    input_frame = frame
                else:
                    input_frame = frame

                # Run SAM3D Body inference
                result = await asyncio.to_thread(
                    self._sam3d_body.process_one_image,
                    input_frame
                )

                # Handle multi-person detection in single frame
                if isinstance(result, list):
                    result = result[0] if len(result) > 0 else None

                smpl_params = self._extract_smpl_params(result)

                if smpl_params is not None:
                    smpl_sequence.append(smpl_params)
                    vertices_sequence.append(
                        np.array(smpl_params['vertices']) if smpl_params.get('vertices') is not None else None
                    )
                else:
                    # Placeholder for failed frames
                    smpl_sequence.append({})
                    vertices_sequence.append(None)

                # Emit frame event
                yield ProcessingEvent(
                    type=EventType.FRAME,
                    data={
                        "person_id": person_id,
                        "frame_index": frame_idx,
                        "local_index": i,
                        "total_person_frames": len(frame_indices),
                        "smpl_params": smpl_params,
                    }
                )

                # Progress update
                yield ProcessingEvent(
                    type=EventType.PROGRESS,
                    data={
                        "person_id": person_id,
                        "processed": i + 1,
                        "total": len(frame_indices),
                        "percent": round((i + 1) / len(frame_indices) * 100, 1),
                    }
                )

            # Step 3: Temporal smoothing
            smoothing_summary = None
            if self.enable_smoothing and self._smoother is not None and len(smpl_sequence) >= 10:
                yield ProcessingEvent(
                    type=EventType.SMOOTHING,
                    data={"person_id": person_id, "status": "started"}
                )

                smooth_result = await asyncio.to_thread(
                    self._smoother.process_sequence,
                    smpl_sequence,
                    fps=fps,
                )

                smpl_sequence = smooth_result.get('final_sequence', smpl_sequence)
                smoothing_summary = smooth_result.get('summary')

                yield ProcessingEvent(
                    type=EventType.SMOOTHING,
                    data={
                        "person_id": person_id,
                        "status": "complete",
                        "summary": smoothing_summary,
                    }
                )

            # Step 4: World grounding
            grounding_summary = None
            if self.enable_grounding and self._grounder is not None:
                yield ProcessingEvent(
                    type=EventType.GROUNDING,
                    data={"person_id": person_id, "status": "started"}
                )

                ground_result = await asyncio.to_thread(
                    self._grounder.ground_sequence,
                    smpl_sequence,
                    vertices_sequence,
                    fps,
                )

                smpl_sequence = ground_result.get('grounded_sequence', smpl_sequence)
                grounding_summary = {
                    "ground_height": ground_result.get('ground_height'),
                    "standing_frames": ground_result.get('standing_frames'),
                    "total_frames": ground_result.get('total_frames'),
                    "max_pelvis_adjustment": ground_result.get('max_pelvis_adjustment'),
                }

                yield ProcessingEvent(
                    type=EventType.GROUNDING,
                    data={
                        "person_id": person_id,
                        "status": "complete",
                        "summary": grounding_summary,
                    }
                )

            # Store results
            all_results[person_id] = PersonResult(
                person_id=person_id,
                smpl_sequence=smpl_sequence,
                vertices_sequence=vertices_sequence,
                frame_indices=frame_indices,
                smoothing_summary=smoothing_summary,
                grounding_summary=grounding_summary,
            )

        # Final completion event
        yield ProcessingEvent(
            type=EventType.COMPLETE,
            data={
                "success": True,
                "num_people": len(all_results),
                "total_frames": T,
                "results": {
                    pid: {
                        "person_id": result.person_id,
                        "frame_count": len(result.smpl_sequence),
                        "frame_indices": result.frame_indices,
                        "smpl_sequence": result.smpl_sequence,
                        "smoothing_summary": result.smoothing_summary,
                        "grounding_summary": result.grounding_summary,
                    }
                    for pid, result in all_results.items()
                },
            }
        )

    async def process_single_frame(
        self,
        frame: np.ndarray,
    ) -> Dict:
        """
        Process a single frame (for real-time / image mode).

        Args:
            frame: Single frame (H, W, 3) in RGB

        Returns:
            Dict with SMPL params and vertices
        """
        await asyncio.to_thread(self._load_components)

        result = await asyncio.to_thread(
            self._sam3d_body.process_one_image,
            frame
        )

        if isinstance(result, list):
            return [self._extract_smpl_params(r) for r in result]
        else:
            return self._extract_smpl_params(result)


# Convenience function for simple usage
async def process_video_simple(
    frames: List[np.ndarray],
    fps: float = 30.0,
    enable_tracking: bool = False,
    enable_grounding: bool = True,
) -> Dict[int, PersonResult]:
    """
    Simple interface to process video and get results.

    Args:
        frames: Video frames
        fps: Frame rate
        enable_tracking: Use SAM3 tracking (slower but better for multi-person)
        enable_grounding: Transform to world coordinates

    Returns:
        Dict mapping person_id -> PersonResult
    """
    pipeline = Body4DPipeline(
        enable_tracking=enable_tracking,
        enable_grounding=enable_grounding,
    )

    results = {}
    async for event in pipeline.process_video(frames, fps):
        if event.type == EventType.COMPLETE:
            results = event.data.get('results', {})

    return results


if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("Body4D Pipeline Test")
    print("=" * 60)

    # Test with dummy data
    async def test():
        T = 10
        H, W = 480, 640

        frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(T)]

        pipeline = Body4DPipeline(
            enable_tracking=False,  # Skip SAM3 for test
            enable_smoothing=False,  # Skip smoothing for test
            enable_grounding=False,  # Skip grounding for test
        )

        print("\nPipeline created, events would stream here...")
        print("(Skipping actual processing in test mode)")

    asyncio.run(test())

    print("\n" + "=" * 60)
    print("Body4D Pipeline Test Complete!")
    print("=" * 60)
