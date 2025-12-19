"""
Physical Constraints Module

Implements foot contact detection and footskate removal for VR-grade motion quality.
Based on SOTA research (2024): Temporal Consistency Loss for High Resolution Textured
and Clothed 3D Human Reconstruction.

Author: Nvwa Team
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class PhysicalConstraints:
    """
    Physical Constraints: Enforce biomechanical plausibility

    Key issues addressed:
    1. Footskate: Feet sliding on ground during contact (very noticeable in VR)
    2. Floating: Feet hovering above ground unnaturally
    3. Penetration: Feet sinking below ground plane

    Solution: Detect foot contacts and lock positions during contact phases
    """

    # SMPL joint indices for feet
    LEFT_FOOT_INDEX = 10
    RIGHT_FOOT_INDEX = 11
    LEFT_TOE_INDEX = 22  # Optional: for more precise tracking
    RIGHT_TOE_INDEX = 23

    def __init__(
        self,
        foot_contact_threshold: float = 0.05,  # meters
        foot_velocity_threshold: float = 0.1,  # m/s
        min_contact_duration: int = 3,  # frames
        enable_toe_tracking: bool = False,
    ):
        """
        Initialize Physical Constraints

        Args:
            foot_contact_threshold: Height threshold for ground contact (meters)
                Default: 0.05m (5cm) - typical shoe sole height
            foot_velocity_threshold: Max velocity for stationary foot (m/s)
                Default: 0.1 m/s - helps reject false positives
            min_contact_duration: Minimum frames for valid contact
                Default: 3 frames - reject jittery contacts
            enable_toe_tracking: Use toe joints in addition to heel
                Default: False - simpler, more robust
        """
        self.foot_contact_threshold = foot_contact_threshold
        self.foot_velocity_threshold = foot_velocity_threshold
        self.min_contact_duration = min_contact_duration
        self.enable_toe_tracking = enable_toe_tracking

    def detect_foot_contacts(
        self, joints_3d: np.ndarray, fps: float = 30.0
    ) -> np.ndarray:
        """
        Detect when feet are in contact with ground

        Args:
            joints_3d: 3D joint positions (T, 24, 3) where:
                - T: number of frames
                - 24: SMPL joints
                - 3: (x, y, z) coordinates
            fps: Frame rate for velocity calculation

        Returns:
            Contact mask (T, 2) with boolean values:
                - [:, 0]: left foot contact
                - [:, 1]: right foot contact
        """
        T = len(joints_3d)

        # Extract foot positions
        left_foot = joints_3d[:, self.LEFT_FOOT_INDEX, :]  # (T, 3)
        right_foot = joints_3d[:, self.RIGHT_FOOT_INDEX, :]  # (T, 3)

        # Height criterion: foot below threshold
        left_heights = left_foot[:, 2]  # Z coordinate (up axis)
        right_heights = right_foot[:, 2]

        left_low = left_heights < self.foot_contact_threshold
        right_low = right_heights < self.foot_contact_threshold

        # Velocity criterion: foot moving slowly
        dt = 1.0 / fps

        left_velocity = np.concatenate([
            [0.0],  # First frame has no velocity
            np.linalg.norm(np.diff(left_foot, axis=0), axis=1) / dt,
        ])
        right_velocity = np.concatenate([
            [0.0],
            np.linalg.norm(np.diff(right_foot, axis=0), axis=1) / dt,
        ])

        left_slow = left_velocity < self.foot_velocity_threshold
        right_slow = right_velocity < self.foot_velocity_threshold

        # Combine criteria: low AND slow
        left_contact_raw = left_low & left_slow
        right_contact_raw = right_low & right_slow

        # Filter short contacts (noise rejection)
        left_contact = self._filter_short_contacts(
            left_contact_raw, self.min_contact_duration
        )
        right_contact = self._filter_short_contacts(
            right_contact_raw, self.min_contact_duration
        )

        # Stack into (T, 2) array
        contact_mask = np.stack([left_contact, right_contact], axis=1)

        return contact_mask

    def _filter_short_contacts(
        self, contact_mask: np.ndarray, min_duration: int
    ) -> np.ndarray:
        """
        Remove contact phases shorter than min_duration

        Args:
            contact_mask: Boolean array (T,)
            min_duration: Minimum consecutive True values to keep

        Returns:
            Filtered boolean array
        """
        T = len(contact_mask)
        filtered = np.zeros(T, dtype=bool)

        # Find contact segments
        contact_start = None
        for t in range(T):
            if contact_mask[t] and contact_start is None:
                # Start of contact
                contact_start = t
            elif not contact_mask[t] and contact_start is not None:
                # End of contact
                duration = t - contact_start
                if duration >= min_duration:
                    filtered[contact_start:t] = True
                contact_start = None

        # Handle contact extending to end of sequence
        if contact_start is not None:
            duration = T - contact_start
            if duration >= min_duration:
                filtered[contact_start:] = True

        return filtered

    def remove_footskate(
        self,
        smpl_params_sequence: List[Dict],
        contact_mask: np.ndarray,
        joints_3d: np.ndarray,
    ) -> List[Dict]:
        """
        Lock foot positions during contact phases

        Strategy:
        1. When foot is in contact, record its 3D position
        2. Adjust global translation to keep foot stationary
        3. Preserve all other SMPL parameters (pose, shape)

        Args:
            smpl_params_sequence: List of SMPL parameter dicts
            contact_mask: Contact detection results (T, 2)
            joints_3d: 3D joint positions (T, 24, 3)

        Returns:
            Modified sequence with locked foot positions
        """
        T = len(smpl_params_sequence)
        constrained_sequence = [p.copy() for p in smpl_params_sequence]

        # Track locked positions for each foot
        left_locked_pos = None
        right_locked_pos = None

        for t in range(T):
            left_contact = contact_mask[t, 0]
            right_contact = contact_mask[t, 1]

            # Current foot positions
            left_foot_pos = joints_3d[t, self.LEFT_FOOT_INDEX, :]
            right_foot_pos = joints_3d[t, self.RIGHT_FOOT_INDEX, :]

            # Displacement to apply to root
            displacement = np.zeros(3)

            # Left foot
            if left_contact:
                if left_locked_pos is None:
                    # Start of contact - lock current position
                    left_locked_pos = left_foot_pos.copy()
                else:
                    # During contact - compute required displacement
                    # We want: left_foot_pos + displacement = left_locked_pos
                    displacement += (left_locked_pos - left_foot_pos)
            else:
                # Not in contact - unlock
                left_locked_pos = None

            # Right foot
            if right_contact:
                if right_locked_pos is None:
                    right_locked_pos = right_foot_pos.copy()
                else:
                    displacement += (right_locked_pos - right_foot_pos)
            else:
                right_locked_pos = None

            # Apply displacement to global translation
            # Note: If both feet are locked, we average the corrections
            if left_contact and right_contact and left_locked_pos is not None and right_locked_pos is not None:
                displacement /= 2.0

            if np.linalg.norm(displacement) > 1e-6:
                current_transl = np.array(constrained_sequence[t]["transl"]).squeeze()
                new_transl = current_transl + displacement
                constrained_sequence[t]["transl"] = new_transl.reshape(
                    constrained_sequence[t]["transl"].shape
                )

        return constrained_sequence

    def apply_constraints(
        self,
        smpl_params_sequence: List[Dict],
        joints_3d: np.ndarray,
        fps: float = 30.0,
    ) -> Dict:
        """
        Apply all physical constraints to sequence

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries
            joints_3d: 3D joint positions (T, 24, 3)
            fps: Frame rate

        Returns:
            Dictionary with:
                - 'constrained_sequence': Modified SMPL parameters
                - 'contact_mask': Detected foot contacts (T, 2)
                - 'num_footskates': Number of footskate violations detected
                - 'max_footskate_velocity': Worst footskate speed (m/s)
        """
        # Detect contacts
        contact_mask = self.detect_foot_contacts(joints_3d, fps=fps)

        # Measure footskate before correction
        footskate_stats_before = self._measure_footskate(joints_3d, contact_mask, fps)

        # Apply corrections
        constrained_sequence = self.remove_footskate(
            smpl_params_sequence, contact_mask, joints_3d
        )

        # Note: We can't measure footskate after without re-running SMPL forward pass
        # This would require SMPL model access, which we don't have in this module

        return {
            "constrained_sequence": constrained_sequence,
            "contact_mask": contact_mask,
            "num_footskates_before": footskate_stats_before["num_violations"],
            "max_footskate_velocity_before": footskate_stats_before["max_velocity"],
            "mean_footskate_velocity_before": footskate_stats_before["mean_velocity"],
            "contact_phases": footskate_stats_before["contact_phases"],
        }

    def _measure_footskate(
        self, joints_3d: np.ndarray, contact_mask: np.ndarray, fps: float
    ) -> Dict:
        """
        Measure footskate violations

        Footskate occurs when a foot moves while in contact with ground.

        Args:
            joints_3d: 3D joint positions (T, 24, 3)
            contact_mask: Contact mask (T, 2)
            fps: Frame rate

        Returns:
            Dictionary with violation statistics
        """
        T = len(joints_3d)
        dt = 1.0 / fps

        left_foot = joints_3d[:, self.LEFT_FOOT_INDEX, :]
        right_foot = joints_3d[:, self.RIGHT_FOOT_INDEX, :]

        # Compute velocities
        left_velocity = np.concatenate([
            [[0.0, 0.0, 0.0]],
            np.diff(left_foot, axis=0) / dt,
        ])
        right_velocity = np.concatenate([
            [[0.0, 0.0, 0.0]],
            np.diff(right_foot, axis=0) / dt,
        ])

        # Footskate: velocity magnitude during contact
        left_speed = np.linalg.norm(left_velocity, axis=1)
        right_speed = np.linalg.norm(right_velocity, axis=1)

        # Threshold for "acceptable" movement during contact (2 cm/s)
        # Any movement faster is considered footskate
        ACCEPTABLE_SPEED = 0.02  # m/s

        left_violations = contact_mask[:, 0] & (left_speed > ACCEPTABLE_SPEED)
        right_violations = contact_mask[:, 1] & (right_speed > ACCEPTABLE_SPEED)

        all_violations = left_violations | right_violations
        num_violations = np.sum(all_violations)

        # Compute violation velocities
        violation_speeds = []
        for t in range(T):
            if left_violations[t]:
                violation_speeds.append(left_speed[t])
            if right_violations[t]:
                violation_speeds.append(right_speed[t])

        max_velocity = np.max(violation_speeds) if violation_speeds else 0.0
        mean_velocity = np.mean(violation_speeds) if violation_speeds else 0.0

        # Count contact phases
        contact_phases = self._count_contact_phases(contact_mask)

        return {
            "num_violations": num_violations,
            "max_velocity": max_velocity,
            "mean_velocity": mean_velocity,
            "contact_phases": contact_phases,
        }

    def _count_contact_phases(self, contact_mask: np.ndarray) -> int:
        """Count number of distinct contact phases (for both feet combined)"""
        T = len(contact_mask)
        phases = 0

        # Check left foot
        in_contact = False
        for t in range(T):
            if contact_mask[t, 0] and not in_contact:
                phases += 1
                in_contact = True
            elif not contact_mask[t, 0]:
                in_contact = False

        # Check right foot
        in_contact = False
        for t in range(T):
            if contact_mask[t, 1] and not in_contact:
                phases += 1
                in_contact = True
            elif not contact_mask[t, 1]:
                in_contact = False

        return phases


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Physical Constraints Module Test")
    print("=" * 60)

    # Generate synthetic walking motion
    np.random.seed(42)
    T = 120  # 4 seconds @ 30fps
    fps = 30.0

    # Simulate walking gait with foot contacts
    t = np.linspace(0, 4, T)

    # Create synthetic SMPL joints (simplified)
    joints_3d = np.zeros((T, 24, 3))

    # Simulate left foot (steps at t=0-1s and t=2-3s)
    left_foot_height = np.where(
        ((t >= 0) & (t < 1)) | ((t >= 2) & (t < 3)),
        0.0,  # On ground
        0.3 * np.abs(np.sin(2 * np.pi * t)),  # Swing phase
    )
    left_foot_x = 0.5 * t  # Forward motion

    # Simulate right foot (steps at t=1-2s and t=3-4s)
    right_foot_height = np.where(
        ((t >= 1) & (t < 2)) | ((t >= 3) & (t < 4)),
        0.0,  # On ground
        0.3 * np.abs(np.sin(2 * np.pi * t + np.pi)),  # Swing phase (opposite phase)
    )
    right_foot_x = 0.5 * t + 0.25  # Forward motion (offset)

    # Add realistic noise to simulate estimation error
    noise_scale = 0.02
    left_foot_height += np.random.randn(T) * noise_scale
    right_foot_height += np.random.randn(T) * noise_scale

    # Assign to joints array
    joints_3d[:, PhysicalConstraints.LEFT_FOOT_INDEX, 0] = left_foot_x
    joints_3d[:, PhysicalConstraints.LEFT_FOOT_INDEX, 2] = left_foot_height

    joints_3d[:, PhysicalConstraints.RIGHT_FOOT_INDEX, 0] = right_foot_x
    joints_3d[:, PhysicalConstraints.RIGHT_FOOT_INDEX, 2] = right_foot_height

    # Create SMPL params (dummy)
    smpl_params_sequence = [
        {
            "betas": np.zeros(10),
            "body_pose": np.zeros(69),
            "global_orient": np.zeros(3),
            "transl": np.array([0, 0, 0]),
        }
        for _ in range(T)
    ]

    print(f"\nGenerated {T} frames of synthetic walking motion")

    # Apply physical constraints
    print("\n--- Applying Physical Constraints ---")
    constraints = PhysicalConstraints(
        foot_contact_threshold=0.05,
        foot_velocity_threshold=0.1,
        min_contact_duration=3,
    )

    result = constraints.apply_constraints(smpl_params_sequence, joints_3d, fps=fps)

    contact_mask = result["contact_mask"]
    num_footskates = result["num_footskates_before"]
    max_velocity = result["max_footskate_velocity_before"]
    contact_phases = result["contact_phases"]

    print(f"Detected {contact_phases} contact phases")
    print(f"Footskate violations (before correction): {num_footskates}")
    print(f"Max footskate velocity: {max_velocity:.3f} m/s")

    # Visualize results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot foot heights
    axes[0].plot(t, left_foot_height, 'b-', label='Left Foot Height', alpha=0.7)
    axes[0].plot(t, right_foot_height, 'r-', label='Right Foot Height', alpha=0.7)
    axes[0].axhline(y=constraints.foot_contact_threshold, color='g', linestyle='--',
                    label=f'Contact Threshold ({constraints.foot_contact_threshold}m)')
    axes[0].set_ylabel('Height (m)')
    axes[0].set_title('Foot Heights')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot contact detection
    axes[1].fill_between(t, 0, 1, where=contact_mask[:, 0], alpha=0.3, color='b', label='Left Contact')
    axes[1].fill_between(t, 0, 1, where=contact_mask[:, 1], alpha=0.3, color='r', label='Right Contact')
    axes[1].set_ylabel('Contact')
    axes[1].set_title('Detected Foot Contacts')
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['No Contact', 'Contact'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot foot velocities
    left_vel = np.concatenate([[0], np.linalg.norm(np.diff(joints_3d[:, PhysicalConstraints.LEFT_FOOT_INDEX, :], axis=0), axis=1) * fps])
    right_vel = np.concatenate([[0], np.linalg.norm(np.diff(joints_3d[:, PhysicalConstraints.RIGHT_FOOT_INDEX, :], axis=0), axis=1) * fps])

    axes[2].plot(t, left_vel, 'b-', label='Left Foot Velocity', alpha=0.7)
    axes[2].plot(t, right_vel, 'r-', label='Right Foot Velocity', alpha=0.7)
    axes[2].axhline(y=constraints.foot_velocity_threshold, color='g', linestyle='--',
                    label=f'Velocity Threshold ({constraints.foot_velocity_threshold} m/s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].set_title('Foot Velocities')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('physical_constraints_test.png', dpi=150)
    print("\n✓ Test plot saved to: physical_constraints_test.png")

    print("\n" + "=" * 60)
    print("✓ Physical Constraints Test Complete!")
    print("=" * 60)
