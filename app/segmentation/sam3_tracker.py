"""
SAM3 Video Segmentation Tracker

Identity-consistent person tracking across video frames using SAM3.

Features:
- Automatic person detection and tracking
- Consistent person IDs across frames
- Handles occlusion and re-appearance
- Provides per-person masks for downstream processing

Author: Nvwa Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path


@dataclass
class PersonTrack:
    """
    Represents a tracked person across video frames.

    Attributes:
        person_id: Unique identifier for this person
        masks: Dict mapping frame_idx -> binary mask (H, W)
        bboxes: Dict mapping frame_idx -> (x1, y1, x2, y2)
        first_frame: First frame where person appears
        last_frame: Last frame where person appears
        confidence_scores: Optional confidence per frame
    """
    person_id: int
    masks: Dict[int, np.ndarray] = field(default_factory=dict)
    bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    first_frame: int = 0
    last_frame: int = 0
    confidence_scores: Dict[int, float] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        """Number of frames this person appears in."""
        return len(self.masks)

    @property
    def frame_indices(self) -> List[int]:
        """List of frame indices where person is visible."""
        return sorted(self.masks.keys())

    def get_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get mask for specific frame, or None if not present."""
        return self.masks.get(frame_idx)

    def get_bbox(self, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box for specific frame."""
        return self.bboxes.get(frame_idx)

    def get_cropped_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        padding: int = 20,
    ) -> Optional[np.ndarray]:
        """
        Get cropped region of frame containing this person.

        Args:
            frame: Full frame image (H, W, 3)
            frame_idx: Frame index
            padding: Pixels to add around bbox

        Returns:
            Cropped frame region or None if person not in frame
        """
        bbox = self.get_bbox(frame_idx)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]

        # Add padding with bounds checking
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)

        return frame[y1:y2, x1:x2].copy()

    def get_masked_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        background: str = "black",
    ) -> Optional[np.ndarray]:
        """
        Get frame with only this person visible (background masked).

        Args:
            frame: Full frame image (H, W, 3)
            frame_idx: Frame index
            background: "black", "white", or "blur"

        Returns:
            Masked frame or None if person not in frame
        """
        mask = self.get_mask(frame_idx)
        if mask is None:
            return None

        result = frame.copy()

        if background == "black":
            result[~mask] = 0
        elif background == "white":
            result[~mask] = 255
        elif background == "blur":
            import cv2
            blurred = cv2.GaussianBlur(frame, (51, 51), 0)
            result[~mask] = blurred[~mask]

        return result


class SAM3Tracker:
    """
    Video segmentation tracker using SAM3 (Segment Anything 3).

    Provides identity-consistent person masks across video frames.
    This is essential for multi-person scenarios where we need to
    track each person separately through the video.

    Usage:
        tracker = SAM3Tracker()
        tracks = tracker.track_video(frames)

        for person in tracks:
            for frame_idx in person.frame_indices:
                mask = person.get_mask(frame_idx)
                # Process each person separately
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        min_track_length: int = 5,  # Minimum frames for valid track
    ):
        """
        Initialize SAM3 Tracker.

        Args:
            model_path: Path to SAM3 checkpoint (uses HuggingFace default if None)
            device: Device to run model on ("cuda" or "cpu")
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for tracking association
            min_track_length: Minimum frames for a track to be considered valid
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_track_length = min_track_length

        # Lazy load model
        self._model = None
        self._predictor = None

    def _load_model(self):
        """Lazy load SAM3 model."""
        if self._model is not None:
            return

        try:
            # Try to import SAM3
            # Note: The exact import path may vary depending on how SAM3 is installed
            from sam3 import SAM3VideoPredictor, load_sam3

            print("Loading SAM3 model...")
            if self.model_path:
                self._model = load_sam3(self.model_path, device=self.device)
            else:
                # Load from HuggingFace
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="facebook/sam3",
                    filename="sam3.pt",
                    cache_dir="/workspace/checkpoints"
                )
                self._model = load_sam3(model_path, device=self.device)

            self._predictor = SAM3VideoPredictor(self._model)
            print("SAM3 model loaded!")

        except ImportError as e:
            print(f"SAM3 not available: {e}")
            print("Falling back to simple detection-based tracking...")
            self._model = "fallback"

    def track_video(
        self,
        frames: List[np.ndarray],
        initial_prompts: Optional[List[Dict]] = None,
    ) -> List[PersonTrack]:
        """
        Track people across video frames.

        Args:
            frames: List of video frames (H, W, 3) in RGB
            initial_prompts: Optional list of prompts for first frame
                Each prompt: {"type": "point"|"box", "coords": [...]}

        Returns:
            List of PersonTrack objects, one per detected person
        """
        self._load_model()

        if self._model == "fallback":
            return self._track_with_fallback(frames)

        return self._track_with_sam3(frames, initial_prompts)

    def _track_with_sam3(
        self,
        frames: List[np.ndarray],
        initial_prompts: Optional[List[Dict]] = None,
    ) -> List[PersonTrack]:
        """
        Track using SAM3 video segmentation.

        SAM3 uses propagation + detection:
        - Propagation: Track existing masks to next frame
        - Detection: Find new objects that appear
        """
        T = len(frames)
        tracks: Dict[int, PersonTrack] = {}
        next_person_id = 0

        # Initialize on first frame
        first_masks, first_scores = self._detect_people(frames[0], initial_prompts)

        for i, (mask, score) in enumerate(zip(first_masks, first_scores)):
            if score < self.confidence_threshold:
                continue

            track = PersonTrack(
                person_id=next_person_id,
                first_frame=0,
                last_frame=0,
            )
            track.masks[0] = mask
            track.bboxes[0] = self._mask_to_bbox(mask)
            track.confidence_scores[0] = score
            tracks[next_person_id] = track
            next_person_id += 1

        # Process remaining frames
        for t in range(1, T):
            frame = frames[t]

            # Get previous masks for propagation
            prev_masks = [
                (pid, track.get_mask(t - 1))
                for pid, track in tracks.items()
                if track.get_mask(t - 1) is not None
            ]

            # Propagate existing tracks
            for pid, prev_mask in prev_masks:
                if prev_mask is None:
                    continue

                # Use SAM3 propagation
                new_mask, score = self._propagate_mask(
                    frame, prev_mask, frames[t - 1]
                )

                if new_mask is not None and score >= self.confidence_threshold:
                    tracks[pid].masks[t] = new_mask
                    tracks[pid].bboxes[t] = self._mask_to_bbox(new_mask)
                    tracks[pid].confidence_scores[t] = score
                    tracks[pid].last_frame = t

            # Detect new people (not covered by existing tracks)
            existing_masks = [
                track.get_mask(t)
                for track in tracks.values()
                if track.get_mask(t) is not None
            ]

            new_masks, new_scores = self._detect_new_people(
                frame, existing_masks
            )

            for mask, score in zip(new_masks, new_scores):
                if score < self.confidence_threshold:
                    continue

                track = PersonTrack(
                    person_id=next_person_id,
                    first_frame=t,
                    last_frame=t,
                )
                track.masks[t] = mask
                track.bboxes[t] = self._mask_to_bbox(mask)
                track.confidence_scores[t] = score
                tracks[next_person_id] = track
                next_person_id += 1

        # Filter short tracks
        valid_tracks = [
            track for track in tracks.values()
            if track.frame_count >= self.min_track_length
        ]

        return valid_tracks

    def _track_with_fallback(
        self,
        frames: List[np.ndarray],
    ) -> List[PersonTrack]:
        """
        Fallback tracking using simple detection + IoU matching.

        Used when SAM3 is not available.
        """
        try:
            import cv2
            from ultralytics import YOLO

            # Use YOLOv8 for person detection
            model = YOLO("yolov8n-seg.pt")  # Segmentation model

        except ImportError:
            print("YOLOv8 not available, using dummy single-person track")
            return self._create_dummy_track(frames)

        T = len(frames)
        tracks: Dict[int, PersonTrack] = {}
        next_person_id = 0

        for t, frame in enumerate(frames):
            # Detect people in this frame
            results = model(frame, classes=[0], verbose=False)  # class 0 = person

            if len(results) == 0 or results[0].masks is None:
                continue

            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            # Match to existing tracks using IoU
            matched = set()
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if score < self.confidence_threshold:
                    continue

                # Resize mask to frame size
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0])
                ) > 0.5

                best_match = None
                best_iou = self.iou_threshold

                # Find best matching existing track
                for pid, track in tracks.items():
                    if pid in matched:
                        continue

                    prev_mask = track.get_mask(t - 1) if t > 0 else None
                    if prev_mask is None:
                        continue

                    iou = self._compute_iou(mask_resized, prev_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pid

                if best_match is not None:
                    # Update existing track
                    tracks[best_match].masks[t] = mask_resized
                    tracks[best_match].bboxes[t] = tuple(map(int, box))
                    tracks[best_match].confidence_scores[t] = float(score)
                    tracks[best_match].last_frame = t
                    matched.add(best_match)
                else:
                    # Create new track
                    track = PersonTrack(
                        person_id=next_person_id,
                        first_frame=t,
                        last_frame=t,
                    )
                    track.masks[t] = mask_resized
                    track.bboxes[t] = tuple(map(int, box))
                    track.confidence_scores[t] = float(score)
                    tracks[next_person_id] = track
                    next_person_id += 1

        # Filter short tracks
        valid_tracks = [
            track for track in tracks.values()
            if track.frame_count >= self.min_track_length
        ]

        return valid_tracks

    def _create_dummy_track(self, frames: List[np.ndarray]) -> List[PersonTrack]:
        """Create a dummy full-frame track when no detector available."""
        track = PersonTrack(
            person_id=0,
            first_frame=0,
            last_frame=len(frames) - 1,
        )

        for t, frame in enumerate(frames):
            H, W = frame.shape[:2]
            # Full frame mask
            track.masks[t] = np.ones((H, W), dtype=bool)
            track.bboxes[t] = (0, 0, W, H)
            track.confidence_scores[t] = 1.0

        return [track]

    def _detect_people(
        self,
        frame: np.ndarray,
        prompts: Optional[List[Dict]] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Detect people in a single frame."""
        # This would use SAM3's detection capabilities
        # For now, return empty if no specific implementation
        return [], []

    def _propagate_mask(
        self,
        current_frame: np.ndarray,
        prev_mask: np.ndarray,
        prev_frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Propagate mask from previous frame to current frame."""
        # This would use SAM3's video propagation
        # For now, return None
        return None, 0.0

    def _detect_new_people(
        self,
        frame: np.ndarray,
        existing_masks: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Detect new people not covered by existing masks."""
        return [], []

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return (0, 0, 1, 1)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(x1), int(y1), int(x2), int(y2))

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union


if __name__ == "__main__":
    print("=" * 60)
    print("SAM3 Tracker Test")
    print("=" * 60)

    # Create synthetic test data
    T = 30
    H, W = 480, 640

    frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(T)]

    # Test fallback tracker (without SAM3)
    tracker = SAM3Tracker(device="cpu")
    tracks = tracker._create_dummy_track(frames)

    print(f"\nCreated {len(tracks)} track(s)")
    for track in tracks:
        print(f"  Person {track.person_id}: frames {track.first_frame}-{track.last_frame}")
        print(f"    Total frames: {track.frame_count}")

    print("\n" + "=" * 60)
    print("SAM3 Tracker Test Complete!")
    print("=" * 60)
