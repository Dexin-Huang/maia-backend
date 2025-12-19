"""
Motion Smoothing Module (Phase 2)

Applies One-Euro filter to SMPL pose parameters (θ) to eliminate joint jitter
while preserving natural motion dynamics.

Author: Nvwa Team
"""

import numpy as np
from typing import List, Dict, Optional, Literal
from .filters.one_euro import OneEuroFilter, MultiChannelOneEuroFilter


class MotionSmoother:
    """
    Motion Smoothing: Apply temporal filtering to SMPL pose parameters

    SMPL pose parameters (θ) control joint rotations:
    - global_orient: Root orientation (3 params)
    - body_pose: Body joint rotations (23 joints × 3 = 69 params)
    - Total: 72 rotation parameters

    Problem: Per-frame estimation causes erratic limb movements
    Solution: Apply One-Euro filter to each rotation parameter independently
    """

    def __init__(
        self,
        filter_type: Literal["one_euro", "savgol"] = "one_euro",
        preset: Literal["conservative", "balanced", "responsive"] = "balanced",
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ):
        """
        Initialize Motion Smoother

        Args:
            filter_type: Type of filter to use
                - "one_euro": Adaptive One-Euro filter (recommended)
                - "savgol": Savitzky-Golay filter (for future implementation)

            preset: Tuning preset (for One-Euro filter)
                - "conservative": Min jitter, more lag (meditation, slow motion)
                - "balanced": Good balance (general use, default)
                - "responsive": Min lag, less smoothing (sports, dance)

            min_cutoff: Manual minimum cutoff frequency (overrides preset)
            beta: Manual speed coefficient (overrides preset)
            d_cutoff: Manual derivative cutoff (overrides preset)
        """
        self.filter_type = filter_type
        self.preset = preset

        # Define tuning presets (from one_euro.py documentation)
        presets = {
            "conservative": {"min_cutoff": 0.5, "beta": 0.005, "d_cutoff": 1.0},
            "balanced": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
            "responsive": {"min_cutoff": 2.0, "beta": 0.01, "d_cutoff": 1.0},
        }

        # Get preset params
        params = presets[preset]

        # Override with manual params if provided
        self.min_cutoff = min_cutoff if min_cutoff is not None else params["min_cutoff"]
        self.beta = beta if beta is not None else params["beta"]
        self.d_cutoff = d_cutoff if d_cutoff is not None else params["d_cutoff"]

        # Initialize filters (lazy)
        self.global_orient_filter: Optional[MultiChannelOneEuroFilter] = None
        self.body_pose_filter: Optional[MultiChannelOneEuroFilter] = None
        self.transl_filter: Optional[MultiChannelOneEuroFilter] = None

    def _ensure_filters_initialized(self):
        """Initialize filters on first use"""
        if self.filter_type == "one_euro":
            if self.global_orient_filter is None:
                # Global orientation: 3 channels
                self.global_orient_filter = MultiChannelOneEuroFilter(
                    num_channels=3,
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff,
                )

            if self.body_pose_filter is None:
                # Body pose: 69 channels (23 joints × 3)
                self.body_pose_filter = MultiChannelOneEuroFilter(
                    num_channels=69,
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff,
                )

            if self.transl_filter is None:
                # Translation: 3 channels (x, y, z)
                self.transl_filter = MultiChannelOneEuroFilter(
                    num_channels=3,
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff,
                )

    def smooth_sequence(
        self,
        smpl_params_sequence: List[Dict],
        frame_timestamps: Optional[List[float]] = None,
        fps: float = 30.0,
    ) -> List[Dict]:
        """
        Apply motion smoothing to SMPL parameter sequence

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries
                Each dict should contain: 'global_orient', 'body_pose', 'transl'
            frame_timestamps: Optional timestamps for each frame (seconds)
                If None, assumes constant fps
            fps: Frame rate (only used if frame_timestamps is None)

        Returns:
            Smoothed sequence with same format
        """
        self._ensure_filters_initialized()

        T = len(smpl_params_sequence)

        # Generate timestamps if not provided
        if frame_timestamps is None:
            frame_timestamps = [t / fps for t in range(T)]

        smoothed_sequence = []

        for t, (params, timestamp) in enumerate(zip(smpl_params_sequence, frame_timestamps)):
            smoothed_params = params.copy()

            # Smooth global orientation
            if "global_orient" in params:
                global_orient = np.array(params["global_orient"]).squeeze()[:3]
                smoothed_global_orient = self.global_orient_filter.filter(
                    global_orient.tolist(), timestamp
                )
                smoothed_params["global_orient"] = np.array(smoothed_global_orient).reshape(
                    params["global_orient"].shape
                )

            # Smooth body pose
            if "body_pose" in params:
                body_pose = np.array(params["body_pose"]).squeeze()[:69]
                smoothed_body_pose = self.body_pose_filter.filter(
                    body_pose.tolist(), timestamp
                )
                smoothed_params["body_pose"] = np.array(smoothed_body_pose).reshape(
                    params["body_pose"].shape
                )

            # Smooth translation
            if "transl" in params:
                transl = np.array(params["transl"]).squeeze()[:3]
                smoothed_transl = self.transl_filter.filter(transl.tolist(), timestamp)
                smoothed_params["transl"] = np.array(smoothed_transl).reshape(
                    params["transl"].shape
                )

            smoothed_sequence.append(smoothed_params)

        return smoothed_sequence

    def process_sequence(
        self,
        smpl_params_sequence: List[Dict],
        frame_timestamps: Optional[List[float]] = None,
        fps: float = 30.0,
    ) -> Dict:
        """
        Process sequence with diagnostics

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries
            frame_timestamps: Optional timestamps for each frame
            fps: Frame rate (if timestamps not provided)

        Returns:
            Dictionary with:
                - 'smoothed_sequence': Smoothed SMPL parameters
                - 'jitter_reduction': Jitter reduction statistics
        """
        # Compute jitter before smoothing
        jitter_before = self._compute_jitter(smpl_params_sequence)

        # Apply smoothing
        smoothed_sequence = self.smooth_sequence(
            smpl_params_sequence, frame_timestamps, fps
        )

        # Compute jitter after smoothing
        jitter_after = self._compute_jitter(smoothed_sequence)

        # Calculate reduction
        jitter_reduction = {
            "global_orient_before": jitter_before["global_orient"],
            "global_orient_after": jitter_after["global_orient"],
            "global_orient_reduction_pct": (
                (jitter_before["global_orient"] - jitter_after["global_orient"])
                / jitter_before["global_orient"]
                * 100
            ),
            "body_pose_before": jitter_before["body_pose"],
            "body_pose_after": jitter_after["body_pose"],
            "body_pose_reduction_pct": (
                (jitter_before["body_pose"] - jitter_after["body_pose"])
                / jitter_before["body_pose"]
                * 100
            ),
            "transl_before": jitter_before["transl"],
            "transl_after": jitter_after["transl"],
            "transl_reduction_pct": (
                (jitter_before["transl"] - jitter_after["transl"])
                / jitter_before["transl"]
                * 100
            ),
        }

        return {
            "smoothed_sequence": smoothed_sequence,
            "jitter_reduction": jitter_reduction,
        }

    def _compute_jitter(self, sequence: List[Dict]) -> Dict[str, float]:
        """
        Compute jitter metric (acceleration magnitude)

        Jitter is measured as the mean absolute second derivative:
        jitter = mean(|acceleration|) = mean(|x[t+1] - 2*x[t] + x[t-1]|)

        Args:
            sequence: List of SMPL parameter dictionaries

        Returns:
            Dictionary with jitter values for each parameter type
        """
        T = len(sequence)

        # Extract trajectories
        global_orients = np.stack(
            [np.array(p["global_orient"]).squeeze()[:3] for p in sequence]
        )  # (T, 3)
        body_poses = np.stack(
            [np.array(p["body_pose"]).squeeze()[:69] for p in sequence]
        )  # (T, 69)
        transls = np.stack(
            [np.array(p["transl"]).squeeze()[:3] for p in sequence]
        )  # (T, 3)

        # Compute acceleration (second derivative)
        def compute_acceleration_jitter(trajectory):
            """Compute mean absolute acceleration"""
            if len(trajectory) < 3:
                return 0.0
            acceleration = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
            return np.mean(np.abs(acceleration))

        jitter = {
            "global_orient": compute_acceleration_jitter(global_orients),
            "body_pose": compute_acceleration_jitter(body_poses),
            "transl": compute_acceleration_jitter(transls),
        }

        return jitter

    def reset(self):
        """Reset all filters"""
        if self.global_orient_filter is not None:
            self.global_orient_filter.reset()
        if self.body_pose_filter is not None:
            self.body_pose_filter.reset()
        if self.transl_filter is not None:
            self.transl_filter.reset()


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Motion Smoothing Module Test")
    print("=" * 60)

    # Generate synthetic SMPL sequence with jitter
    np.random.seed(42)
    T = 100  # 100 frames (3.33s @ 30fps)

    # Simulate smooth motion + noise
    t = np.linspace(0, 2 * np.pi, T)

    # Smooth ground truth trajectory
    smooth_global_orient = np.stack([
        0.5 * np.sin(t),  # Pitch
        0.3 * np.cos(t),  # Yaw
        0.2 * np.sin(2 * t),  # Roll
    ], axis=1)  # (T, 3)

    smooth_body_pose = np.random.randn(T, 69) * 0.1  # Small joint rotations
    smooth_transl = np.stack([
        np.sin(0.5 * t),  # Forward/backward
        np.zeros(T),  # Height (constant)
        np.cos(0.5 * t),  # Left/right
    ], axis=1)

    # Add realistic noise (measurement error)
    noise_scale = 0.05
    noisy_sequence = []
    for i in range(T):
        noisy_sequence.append({
            "global_orient": smooth_global_orient[i] + np.random.randn(3) * noise_scale,
            "body_pose": smooth_body_pose[i] + np.random.randn(69) * noise_scale,
            "transl": smooth_transl[i] + np.random.randn(3) * noise_scale,
            "betas": np.zeros(10),  # Shape (not smoothed here)
        })

    print(f"\nGenerated {T} frames with synthetic jitter")

    # Test different presets
    presets = ["conservative", "balanced", "responsive"]
    results = {}

    for preset in presets:
        print(f"\n--- Testing {preset.upper()} preset ---")
        smoother = MotionSmoother(preset=preset)
        result = smoother.process_sequence(noisy_sequence, fps=30.0)

        jitter_stats = result["jitter_reduction"]
        print(f"  Global Orient Jitter: {jitter_stats['global_orient_before']:.4f} → {jitter_stats['global_orient_after']:.4f} "
              f"({jitter_stats['global_orient_reduction_pct']:.1f}% reduction)")
        print(f"  Body Pose Jitter: {jitter_stats['body_pose_before']:.4f} → {jitter_stats['body_pose_after']:.4f} "
              f"({jitter_stats['body_pose_reduction_pct']:.1f}% reduction)")
        print(f"  Translation Jitter: {jitter_stats['transl_before']:.4f} → {jitter_stats['transl_after']:.4f} "
              f"({jitter_stats['transl_reduction_pct']:.1f}% reduction)")

        results[preset] = result

    # Visualize results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot global orientation (pitch)
    axes[0].plot(smooth_global_orient[:, 0], 'g-', label='Ground Truth', linewidth=2)
    axes[0].plot(
        [p["global_orient"][0] for p in noisy_sequence],
        'gray',
        alpha=0.3,
        label='Noisy Input',
    )
    for preset, color in zip(presets, ['b', 'r', 'm']):
        smoothed = results[preset]["smoothed_sequence"]
        axes[0].plot(
            [p["global_orient"][0] for p in smoothed],
            color=color,
            label=f'{preset.capitalize()}',
            linewidth=1.5,
        )
    axes[0].set_ylabel('Global Orient (Pitch)')
    axes[0].set_title('Motion Smoothing: Global Orientation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot body pose (first joint)
    axes[1].plot(smooth_body_pose[:, 0], 'g-', label='Ground Truth', linewidth=2)
    axes[1].plot(
        [p["body_pose"][0] for p in noisy_sequence],
        'gray',
        alpha=0.3,
        label='Noisy Input',
    )
    for preset, color in zip(presets, ['b', 'r', 'm']):
        smoothed = results[preset]["smoothed_sequence"]
        axes[1].plot(
            [p["body_pose"][0] for p in smoothed],
            color=color,
            label=f'{preset.capitalize()}',
            linewidth=1.5,
        )
    axes[1].set_ylabel('Body Pose (Joint 0)')
    axes[1].set_title('Motion Smoothing: Body Pose')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot translation (x)
    axes[2].plot(smooth_transl[:, 0], 'g-', label='Ground Truth', linewidth=2)
    axes[2].plot(
        [p["transl"][0] for p in noisy_sequence],
        'gray',
        alpha=0.3,
        label='Noisy Input',
    )
    for preset, color in zip(presets, ['b', 'r', 'm']):
        smoothed = results[preset]["smoothed_sequence"]
        axes[2].plot(
            [p["transl"][0] for p in smoothed],
            color=color,
            label=f'{preset.capitalize()}',
            linewidth=1.5,
        )
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Translation X')
    axes[2].set_title('Motion Smoothing: Translation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('motion_smoothing_test.png', dpi=150)
    print("\n✓ Test plot saved to: motion_smoothing_test.png")

    print("\n" + "=" * 60)
    print("✓ Motion Smoothing Test Complete!")
    print("=" * 60)
