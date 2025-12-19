"""
World Grounding Module

Transforms SMPL sequences from camera space to world coordinates for VR/AR playback.

Key features:
1. Ground plane estimation from minimum foot height
2. Pelvis height stabilization during standing
3. Configurable up-axis (Y or Z)

Author: Nvwa Team
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.signal import savgol_filter


class WorldGrounding:
    """
    Transform SMPL sequences from camera space to world space.

    For static camera scenarios:
    - Estimates ground plane from foot positions
    - Locks pelvis height during standing phases
    - Ensures person stands on floor (Y=0 or Z=0)

    For VR/AR "Matrix replay", this prevents:
    - Floating bodies
    - Sinking through floor
    - Inconsistent ground level
    """

    # SMPL joint indices
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    LEFT_TOE = 22
    RIGHT_TOE = 23

    def __init__(
        self,
        up_axis: str = "y",  # "y" or "z"
        ground_offset: float = 0.0,  # Additional offset from detected floor (meters)
        pelvis_lock_velocity_threshold: float = 0.05,  # Max foot velocity for standing (m/s)
        pelvis_lock_smoothing: bool = True,
        smoothing_window: int = 15,  # Frames for pelvis height smoothing
    ):
        """
        Initialize World Grounding.

        Args:
            up_axis: Which axis points up ("y" for Y-up, "z" for Z-up)
            ground_offset: Additional offset from detected floor (for shoe height, etc.)
            pelvis_lock_velocity_threshold: Max foot velocity to consider "standing"
            pelvis_lock_smoothing: Apply Savitzky-Golay smoothing to pelvis height
            smoothing_window: Window size for pelvis smoothing (must be odd)
        """
        self.up_axis = up_axis.lower()
        self.up_idx = 1 if self.up_axis == "y" else 2
        self.ground_offset = ground_offset
        self.pelvis_lock_velocity_threshold = pelvis_lock_velocity_threshold
        self.pelvis_lock_smoothing = pelvis_lock_smoothing
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1

    def estimate_ground_plane(
        self,
        vertices_sequence: List[np.ndarray],
        method: str = "min_foot",
    ) -> float:
        """
        Estimate the ground plane height from vertex data.

        Args:
            vertices_sequence: List of vertex arrays [V, 3] per frame
            method: Estimation method
                - "min_foot": Minimum foot vertex height (robust)
                - "percentile": 5th percentile of all heights (handles noise)
                - "median_min": Median of per-frame minimums

        Returns:
            Estimated ground height in original coordinate system
        """
        if not vertices_sequence or len(vertices_sequence) == 0:
            return 0.0

        if method == "min_foot":
            # Get minimum height from foot vertices across all frames
            min_heights = []
            for vertices in vertices_sequence:
                if vertices is None:
                    continue
                # Use ankle and foot vertices for ground estimation
                foot_indices = [
                    self.LEFT_ANKLE, self.RIGHT_ANKLE,
                    self.LEFT_FOOT, self.RIGHT_FOOT,
                ]
                # Handle both SMPL (6890 verts) and MHR (18540 verts)
                # by checking if indices are valid
                valid_indices = [i for i in foot_indices if i < len(vertices)]
                if valid_indices:
                    foot_heights = vertices[valid_indices, self.up_idx]
                    min_heights.append(np.min(foot_heights))
                else:
                    # Fallback: use minimum of all vertices
                    min_heights.append(np.min(vertices[:, self.up_idx]))

            if not min_heights:
                return 0.0
            # Use 10th percentile to be robust to noise
            return np.percentile(min_heights, 10)

        elif method == "percentile":
            all_heights = []
            for vertices in vertices_sequence:
                if vertices is not None:
                    all_heights.extend(vertices[:, self.up_idx].tolist())
            if not all_heights:
                return 0.0
            return np.percentile(all_heights, 5)

        elif method == "median_min":
            min_per_frame = []
            for vertices in vertices_sequence:
                if vertices is not None:
                    min_per_frame.append(np.min(vertices[:, self.up_idx]))
            if not min_per_frame:
                return 0.0
            return np.median(min_per_frame)

        else:
            raise ValueError(f"Unknown method: {method}")

    def detect_standing_frames(
        self,
        vertices_sequence: List[np.ndarray],
        fps: float = 30.0,
    ) -> np.ndarray:
        """
        Detect frames where the person is standing (feet stationary).

        Args:
            vertices_sequence: List of vertex arrays [V, 3] per frame
            fps: Frame rate for velocity calculation

        Returns:
            Boolean array [T] where True = standing
        """
        T = len(vertices_sequence)
        if T < 2:
            return np.ones(T, dtype=bool)

        dt = 1.0 / fps
        standing = np.zeros(T, dtype=bool)

        # Extract foot positions
        left_foot_pos = []
        right_foot_pos = []

        for vertices in vertices_sequence:
            if vertices is None:
                left_foot_pos.append(None)
                right_foot_pos.append(None)
            else:
                # Handle different vertex counts
                left_idx = self.LEFT_FOOT if self.LEFT_FOOT < len(vertices) else 0
                right_idx = self.RIGHT_FOOT if self.RIGHT_FOOT < len(vertices) else 0
                left_foot_pos.append(vertices[left_idx])
                right_foot_pos.append(vertices[right_idx])

        # Compute velocities
        for t in range(T):
            if t == 0:
                standing[t] = True  # Assume standing at start
                continue

            if left_foot_pos[t] is None or left_foot_pos[t-1] is None:
                standing[t] = standing[t-1]  # Carry forward
                continue

            left_vel = np.linalg.norm(left_foot_pos[t] - left_foot_pos[t-1]) / dt
            right_vel = np.linalg.norm(right_foot_pos[t] - right_foot_pos[t-1]) / dt

            # Standing if at least one foot is stationary
            standing[t] = min(left_vel, right_vel) < self.pelvis_lock_velocity_threshold

        return standing

    def compute_pelvis_heights(
        self,
        smpl_sequence: List[Dict],
        vertices_sequence: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract pelvis heights from sequence.

        Args:
            smpl_sequence: List of SMPL parameter dicts with 'transl'
            vertices_sequence: List of vertex arrays (for pelvis vertex position)

        Returns:
            Array of pelvis heights [T]
        """
        heights = []
        for i, params in enumerate(smpl_sequence):
            if 'transl' in params and params['transl'] is not None:
                transl = np.array(params['transl']).flatten()
                heights.append(transl[self.up_idx])
            elif vertices_sequence[i] is not None:
                # Fallback to pelvis vertex
                heights.append(vertices_sequence[i][self.PELVIS, self.up_idx])
            else:
                heights.append(0.0)
        return np.array(heights)

    def stabilize_pelvis_height(
        self,
        pelvis_heights: np.ndarray,
        standing_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Stabilize pelvis height during standing phases.

        During standing, pelvis height should be relatively constant.
        This removes jitter and bobbing.

        Args:
            pelvis_heights: Original pelvis heights [T]
            standing_mask: Boolean array [T] where True = standing

        Returns:
            Stabilized pelvis heights [T]
        """
        T = len(pelvis_heights)
        stabilized = pelvis_heights.copy()

        if T < 3:
            return stabilized

        # Find standing segments
        segments = []
        start = None
        for t in range(T):
            if standing_mask[t] and start is None:
                start = t
            elif not standing_mask[t] and start is not None:
                segments.append((start, t))
                start = None
        if start is not None:
            segments.append((start, T))

        # For each standing segment, use median height
        for start, end in segments:
            if end - start >= 3:  # Only stabilize segments of 3+ frames
                median_height = np.median(pelvis_heights[start:end])
                stabilized[start:end] = median_height

        # Optionally smooth transitions
        if self.pelvis_lock_smoothing and T >= self.smoothing_window:
            try:
                stabilized = savgol_filter(
                    stabilized,
                    self.smoothing_window,
                    polyorder=2
                )
            except ValueError:
                pass  # Fall back to unsmoothed if window too large

        return stabilized

    def ground_sequence(
        self,
        smpl_sequence: List[Dict],
        vertices_sequence: Optional[List[np.ndarray]] = None,
        fps: float = 30.0,
        stabilize_pelvis: bool = True,
    ) -> Dict:
        """
        Transform SMPL sequence to world coordinates with grounding.

        Args:
            smpl_sequence: List of SMPL parameter dicts with 'transl', 'vertices'
            vertices_sequence: Optional separate list of vertex arrays
            fps: Frame rate for velocity calculations
            stabilize_pelvis: Whether to stabilize pelvis height during standing

        Returns:
            Dict with:
            - 'grounded_sequence': Modified SMPL params with world-space transl
            - 'ground_height': Detected ground plane height
            - 'standing_mask': Boolean array of standing frames
            - 'pelvis_adjustment': How much pelvis was adjusted per frame
        """
        T = len(smpl_sequence)

        # Extract vertices if not provided separately
        if vertices_sequence is None:
            vertices_sequence = []
            for params in smpl_sequence:
                v = params.get('vertices')
                if v is not None:
                    vertices_sequence.append(np.array(v))
                else:
                    vertices_sequence.append(None)

        # Step 1: Estimate ground plane
        ground_height = self.estimate_ground_plane(vertices_sequence)

        # Step 2: Detect standing frames
        standing_mask = self.detect_standing_frames(vertices_sequence, fps)

        # Step 3: Get original pelvis heights
        pelvis_heights = self.compute_pelvis_heights(smpl_sequence, vertices_sequence)

        # Step 4: Stabilize pelvis during standing
        if stabilize_pelvis:
            stabilized_heights = self.stabilize_pelvis_height(pelvis_heights, standing_mask)
        else:
            stabilized_heights = pelvis_heights

        # Step 5: Apply grounding transformation
        grounded_sequence = []
        pelvis_adjustments = []

        for i, params in enumerate(smpl_sequence):
            new_params = {}
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    new_params[key] = value.copy()
                elif isinstance(value, list):
                    new_params[key] = value.copy()
                else:
                    new_params[key] = value

            # Adjust translation
            if 'transl' in new_params and new_params['transl'] is not None:
                transl = np.array(new_params['transl']).flatten()

                # Ground offset (move so floor = 0)
                transl[self.up_idx] -= ground_height
                transl[self.up_idx] -= self.ground_offset

                # Pelvis stabilization adjustment
                height_diff = stabilized_heights[i] - pelvis_heights[i]
                transl[self.up_idx] += height_diff
                pelvis_adjustments.append(height_diff)

                new_params['transl'] = transl.reshape(params['transl'].shape) if hasattr(params['transl'], 'shape') else transl.tolist()
            else:
                pelvis_adjustments.append(0.0)

            # Also adjust vertices if present
            if 'vertices' in new_params and new_params['vertices'] is not None:
                verts = np.array(new_params['vertices'])
                verts[:, self.up_idx] -= ground_height
                verts[:, self.up_idx] -= self.ground_offset
                if stabilize_pelvis:
                    verts[:, self.up_idx] += pelvis_adjustments[-1]
                new_params['vertices'] = verts

            grounded_sequence.append(new_params)

        return {
            'grounded_sequence': grounded_sequence,
            'ground_height': ground_height,
            'ground_offset_applied': self.ground_offset,
            'standing_mask': standing_mask,
            'standing_frames': int(np.sum(standing_mask)),
            'total_frames': T,
            'pelvis_adjustments': np.array(pelvis_adjustments),
            'max_pelvis_adjustment': float(np.max(np.abs(pelvis_adjustments))) if pelvis_adjustments else 0.0,
        }


class MovingCameraGrounding(WorldGrounding):
    """
    World grounding for moving camera scenarios.

    Requires camera poses from SLAM/D4RT to transform from camera space
    to world space before applying grounding.

    TODO: Implement when D4RT integration is added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ground_sequence_with_camera(
        self,
        smpl_sequence: List[Dict],
        camera_poses: List[np.ndarray],  # [T, 4, 4] camera-to-world transforms
        vertices_sequence: Optional[List[np.ndarray]] = None,
        fps: float = 30.0,
    ) -> Dict:
        """
        Transform to world coordinates using camera poses, then apply grounding.

        Args:
            smpl_sequence: SMPL params in camera space
            camera_poses: Camera-to-world transformation matrices [T, 4, 4]
            vertices_sequence: Optional vertex arrays
            fps: Frame rate

        Returns:
            Dict with grounded sequence in world coordinates
        """
        # Step 1: Transform from camera space to world space
        world_sequence = []
        world_vertices = []

        for i, (params, cam_pose) in enumerate(zip(smpl_sequence, camera_poses)):
            new_params = params.copy()

            if 'transl' in params:
                transl = np.array(params['transl']).flatten()
                # Transform translation: world_t = R @ cam_t + t
                transl_homogeneous = np.append(transl, 1.0)
                world_transl = cam_pose @ transl_homogeneous
                new_params['transl'] = world_transl[:3]

            if vertices_sequence and vertices_sequence[i] is not None:
                verts = vertices_sequence[i]
                # Transform vertices
                ones = np.ones((len(verts), 1))
                verts_homogeneous = np.hstack([verts, ones])
                world_verts = (cam_pose @ verts_homogeneous.T).T[:, :3]
                world_vertices.append(world_verts)
            else:
                world_vertices.append(None)

            world_sequence.append(new_params)

        # Step 2: Apply standard grounding to world-space sequence
        return self.ground_sequence(world_sequence, world_vertices, fps)


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("World Grounding Module Test")
    print("=" * 60)

    # Create synthetic data
    T = 60  # 2 seconds at 30fps
    np.random.seed(42)

    # Simulate person standing then walking
    smpl_sequence = []
    vertices_sequence = []

    for t in range(T):
        # Add some noise to pelvis height
        base_height = 0.9 + np.random.randn() * 0.02

        # Walking starts at frame 30
        if t >= 30:
            base_height += 0.05 * np.sin(2 * np.pi * t / 15)  # Bobbing

        transl = np.array([0.01 * t, base_height, 0.0])  # Moving forward

        # Fake vertices (just need foot positions)
        vertices = np.zeros((24, 3))
        vertices[10, :] = [0.0, 0.0 + np.random.randn() * 0.01, 0.1]  # Left foot
        vertices[11, :] = [0.0, 0.02 + np.random.randn() * 0.01, -0.1]  # Right foot
        vertices[0, :] = transl  # Pelvis

        smpl_sequence.append({'transl': transl, 'vertices': vertices})
        vertices_sequence.append(vertices)

    # Test grounding
    grounding = WorldGrounding(up_axis="y", ground_offset=0.0)
    result = grounding.ground_sequence(smpl_sequence, vertices_sequence, fps=30.0)

    print(f"\nGround height detected: {result['ground_height']:.3f}m")
    print(f"Standing frames: {result['standing_frames']}/{result['total_frames']}")
    print(f"Max pelvis adjustment: {result['max_pelvis_adjustment']:.3f}m")

    # Check grounding worked
    grounded = result['grounded_sequence']
    min_height = min(p['vertices'][:, 1].min() for p in grounded)
    print(f"Minimum height after grounding: {min_height:.3f}m")

    print("\n" + "=" * 60)
    print("World Grounding Test Complete!")
    print("=" * 60)
