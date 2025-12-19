"""
Temporal Smoothing Package

Complete temporal smoothing pipeline for 3D human reconstruction.
Eliminates jitter, flickering, and footskate artifacts for VR-grade quality.

Three-Phase Pipeline:
1. Identity Lock: Freeze shape parameters for consistent body proportions
2. Motion Smoothing: One-Euro filter for stable limb movements
3. Physical Constraints: Foot contact detection and footskate removal

Author: Nvwa Team
"""

import numpy as np
from typing import List, Dict, Optional, Literal

from .filters.one_euro import OneEuroFilter, MultiChannelOneEuroFilter
from .identity_lock import IdentityLocker, IdentityLockerWithOutlierRejection
from .motion_smooth import MotionSmoother
from .physical_constraints import PhysicalConstraints


class TemporalSmoother:
    """
    Complete Temporal Smoothing Pipeline

    Orchestrates all three phases to deliver production-ready temporal coherence:
    1. Identity Lock: β parameters (shape consistency)
    2. Motion Smoothing: θ parameters (motion stability)
    3. Physical Constraints: Footskate removal (VR quality)

    Usage:
        smoother = TemporalSmoother(preset="balanced")
        result = smoother.process_sequence(smpl_params, joints_3d, fps=30.0)
        smoothed_params = result['final_sequence']
    """

    def __init__(
        self,
        preset: Literal["conservative", "balanced", "responsive"] = "balanced",
        enable_identity_lock: bool = True,
        enable_motion_smoothing: bool = True,
        enable_physical_constraints: bool = True,
        identity_outlier_rejection: bool = True,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        foot_contact_threshold: float = 0.05,
    ):
        """
        Initialize Temporal Smoother

        Args:
            preset: Tuning preset for One-Euro filter
                - "conservative": Smoother, more lag
                - "balanced": Good balance (default)
                - "responsive": Less smoothing, minimal lag

            enable_identity_lock: Apply Phase 1 (shape consistency)
            enable_motion_smoothing: Apply Phase 2 (motion stability)
            enable_physical_constraints: Apply Phase 3 (footskate removal)

            identity_outlier_rejection: Use robust median for identity lock
            min_cutoff: Manual min cutoff (overrides preset)
            beta: Manual beta (overrides preset)
            foot_contact_threshold: Height threshold for foot contact (meters)
        """
        self.preset = preset
        self.enable_identity_lock = enable_identity_lock
        self.enable_motion_smoothing = enable_motion_smoothing
        self.enable_physical_constraints = enable_physical_constraints

        # Phase 1: Identity Lock
        if self.enable_identity_lock:
            if identity_outlier_rejection:
                self.identity_locker = IdentityLockerWithOutlierRejection(
                    num_shape_params=10
                )
            else:
                self.identity_locker = IdentityLocker(num_shape_params=10)
        else:
            self.identity_locker = None

        # Phase 2: Motion Smoothing
        if self.enable_motion_smoothing:
            self.motion_smoother = MotionSmoother(
                preset=preset, min_cutoff=min_cutoff, beta=beta
            )
        else:
            self.motion_smoother = None

        # Phase 3: Physical Constraints
        if self.enable_physical_constraints:
            self.physical_constraints = PhysicalConstraints(
                foot_contact_threshold=foot_contact_threshold
            )
        else:
            self.physical_constraints = None

    def process_sequence(
        self,
        smpl_params_sequence: List[Dict],
        joints_3d: Optional[np.ndarray] = None,
        fps: float = 30.0,
        frame_timestamps: Optional[List[float]] = None,
    ) -> Dict:
        """
        Apply complete temporal smoothing pipeline

        Args:
            smpl_params_sequence: List of SMPL parameter dicts
                Each dict should contain: 'betas', 'body_pose', 'global_orient', 'transl'

            joints_3d: 3D joint positions (T, 24, 3) - required for physical constraints
                If None, physical constraints will be skipped

            fps: Frame rate (for velocity calculations)
            frame_timestamps: Optional explicit timestamps

        Returns:
            Dictionary with:
                - 'final_sequence': Fully smoothed SMPL parameters
                - 'phase1_result': Identity lock diagnostics
                - 'phase2_result': Motion smoothing diagnostics
                - 'phase3_result': Physical constraints diagnostics
                - 'summary': Overall improvement statistics
        """
        results = {}
        current_sequence = smpl_params_sequence

        # Phase 1: Identity Lock
        if self.enable_identity_lock and self.identity_locker is not None:
            phase1_result = self.identity_locker.process_sequence(current_sequence)
            current_sequence = phase1_result['locked_sequence']
            results['phase1_result'] = {
                'beta_avg': phase1_result['beta_avg'],
                'beta_std': phase1_result['beta_std'],
                'max_deviation': phase1_result['max_deviation'],
                'mean_deviation': phase1_result['mean_deviation'],
            }
        else:
            results['phase1_result'] = {'skipped': True}

        # Phase 2: Motion Smoothing
        if self.enable_motion_smoothing and self.motion_smoother is not None:
            phase2_result = self.motion_smoother.process_sequence(
                current_sequence, frame_timestamps, fps
            )
            current_sequence = phase2_result['smoothed_sequence']
            results['phase2_result'] = phase2_result['jitter_reduction']
        else:
            results['phase2_result'] = {'skipped': True}

        # Phase 3: Physical Constraints
        if (
            self.enable_physical_constraints
            and self.physical_constraints is not None
            and joints_3d is not None
        ):
            phase3_result = self.physical_constraints.apply_constraints(
                current_sequence, joints_3d, fps
            )
            current_sequence = phase3_result['constrained_sequence']
            results['phase3_result'] = {
                'contact_phases': phase3_result['contact_phases'],
                'num_footskates_before': phase3_result['num_footskates_before'],
                'max_footskate_velocity': phase3_result['max_footskate_velocity_before'],
                'contact_mask': phase3_result['contact_mask'],
            }
        else:
            results['phase3_result'] = {'skipped': True}

        # Final result
        results['final_sequence'] = current_sequence

        # Summary statistics
        summary = {
            'total_frames': len(smpl_params_sequence),
            'phases_applied': [
                'Identity Lock' if self.enable_identity_lock else None,
                'Motion Smoothing' if self.enable_motion_smoothing else None,
                'Physical Constraints' if self.enable_physical_constraints else None,
            ],
            'preset': self.preset,
        }

        if self.enable_motion_smoothing and 'phase2_result' in results:
            phase2 = results['phase2_result']
            if 'body_pose_reduction_pct' in phase2:
                summary['jitter_reduction_pct'] = phase2['body_pose_reduction_pct']

        if self.enable_physical_constraints and 'phase3_result' in results:
            phase3 = results['phase3_result']
            if 'num_footskates_before' in phase3:
                summary['footskates_detected'] = phase3['num_footskates_before']

        results['summary'] = summary

        return results


__all__ = [
    "OneEuroFilter",
    "MultiChannelOneEuroFilter",
    "IdentityLocker",
    "IdentityLockerWithOutlierRejection",
    "MotionSmoother",
    "PhysicalConstraints",
    "TemporalSmoother",
]
