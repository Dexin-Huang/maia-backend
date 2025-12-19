"""
Identity Lock Module (Phase 1)

Averages shape parameters (β) across all frames to eliminate body proportion flickering.
This ensures consistent height, weight, and build throughout the entire video sequence.

Author: Nvwa Team
"""

import numpy as np
from typing import List, Dict, Optional


class IdentityLocker:
    """
    Identity Lock: Freeze body shape parameters across temporal sequence

    SMPL shape parameters (β) control body morphology:
    - Height, weight, muscle mass, bone structure
    - 10 parameters that define the "identity" of the person

    Problem: Per-frame SMPL estimation causes flickering body proportions
    Solution: Compute average shape (β_avg) and apply to all frames
    """

    def __init__(self, num_shape_params: int = 10):
        """
        Initialize Identity Locker

        Args:
            num_shape_params: Number of SMPL shape parameters (default: 10)
        """
        self.num_shape_params = num_shape_params
        self.beta_avg: Optional[np.ndarray] = None

    def compute_average_shape(
        self,
        smpl_params_sequence: List[Dict]
    ) -> np.ndarray:
        """
        Compute average shape parameters across all frames

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries
                Each dict should contain 'betas' key with shape (10,)

        Returns:
            Average shape parameters (10,)
        """
        # Extract all shape parameters
        all_betas = []
        for params in smpl_params_sequence:
            if 'betas' in params:
                betas = np.array(params['betas'])
                # Handle different input formats
                if betas.ndim == 2:
                    betas = betas.squeeze()
                all_betas.append(betas[:self.num_shape_params])

        if not all_betas:
            raise ValueError("No shape parameters found in sequence")

        # Compute mean
        betas_array = np.stack(all_betas, axis=0)  # (T, 10)
        beta_avg = np.mean(betas_array, axis=0)  # (10,)

        self.beta_avg = beta_avg
        return beta_avg

    def apply_average_shape(
        self,
        smpl_params_sequence: List[Dict],
        beta_avg: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Apply average shape to all frames

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries
            beta_avg: Optional pre-computed average shape (if None, computes it)

        Returns:
            Modified sequence with locked identity
        """
        # Compute average if not provided
        if beta_avg is None:
            if self.beta_avg is None:
                beta_avg = self.compute_average_shape(smpl_params_sequence)
            else:
                beta_avg = self.beta_avg

        # Apply average shape to all frames
        locked_sequence = []
        for params in smpl_params_sequence:
            locked_params = params.copy()

            # Replace shape parameters
            if 'betas' in locked_params:
                original_shape = np.array(locked_params['betas']).shape
                locked_params['betas'] = beta_avg.reshape(original_shape)
            else:
                # If no betas in params, add them
                locked_params['betas'] = beta_avg

            locked_sequence.append(locked_params)

        return locked_sequence

    def process_sequence(
        self,
        smpl_params_sequence: List[Dict]
    ) -> Dict:
        """
        One-shot processing: Compute average and apply

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries

        Returns:
            Dictionary with:
                - 'locked_sequence': Modified sequence with frozen identity
                - 'beta_avg': Average shape parameters used
                - 'beta_std': Standard deviation (for diagnostics)
        """
        # Compute average shape
        beta_avg = self.compute_average_shape(smpl_params_sequence)

        # Compute std for diagnostics
        all_betas = np.stack([
            np.array(params['betas']).squeeze()[:self.num_shape_params]
            for params in smpl_params_sequence
        ], axis=0)
        beta_std = np.std(all_betas, axis=0)

        # Apply average shape
        locked_sequence = self.apply_average_shape(smpl_params_sequence, beta_avg)

        return {
            'locked_sequence': locked_sequence,
            'beta_avg': beta_avg,
            'beta_std': beta_std,
            'max_deviation': np.max(beta_std),
            'mean_deviation': np.mean(beta_std)
        }

    def reset(self):
        """Reset stored average shape"""
        self.beta_avg = None


class IdentityLockerWithOutlierRejection(IdentityLocker):
    """
    Enhanced Identity Locker with outlier rejection

    Uses median instead of mean to be robust to outlier frames
    (e.g., frames where person is partially occluded)
    """

    def __init__(
        self,
        num_shape_params: int = 10,
        outlier_percentile: float = 95.0
    ):
        """
        Initialize with outlier rejection

        Args:
            num_shape_params: Number of SMPL shape parameters
            outlier_percentile: Percentile threshold for outlier detection
                (default: 95.0 means reject top 5% deviation)
        """
        super().__init__(num_shape_params)
        self.outlier_percentile = outlier_percentile

    def compute_average_shape(
        self,
        smpl_params_sequence: List[Dict]
    ) -> np.ndarray:
        """
        Compute robust average using median and outlier rejection

        Args:
            smpl_params_sequence: List of SMPL parameter dictionaries

        Returns:
            Robust average shape parameters (10,)
        """
        # Extract all shape parameters
        all_betas = []
        for params in smpl_params_sequence:
            if 'betas' in params:
                betas = np.array(params['betas'])
                if betas.ndim == 2:
                    betas = betas.squeeze()
                all_betas.append(betas[:self.num_shape_params])

        if not all_betas:
            raise ValueError("No shape parameters found in sequence")

        betas_array = np.stack(all_betas, axis=0)  # (T, 10)

        # Compute median instead of mean (robust to outliers)
        beta_median = np.median(betas_array, axis=0)  # (10,)

        # Calculate deviations from median
        deviations = np.abs(betas_array - beta_median[np.newaxis, :])  # (T, 10)
        max_deviations = np.max(deviations, axis=1)  # (T,) - max deviation per frame

        # Identify inliers (frames within percentile threshold)
        threshold = np.percentile(max_deviations, self.outlier_percentile)
        inlier_mask = max_deviations <= threshold

        # Compute average from inliers only
        inlier_betas = betas_array[inlier_mask]  # (T_inlier, 10)
        beta_avg = np.mean(inlier_betas, axis=0)  # (10,)

        self.beta_avg = beta_avg
        return beta_avg


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("Identity Lock Module Test")
    print("=" * 60)

    # Generate synthetic SMPL sequence with flickering shape
    np.random.seed(42)
    T = 100  # 100 frames

    # Ground truth shape
    true_beta = np.random.randn(10) * 0.5

    # Add noise to simulate per-frame estimation errors
    noisy_sequence = []
    for t in range(T):
        noise = np.random.randn(10) * 0.1  # ±0.1 std deviation
        noisy_beta = true_beta + noise

        noisy_sequence.append({
            'betas': noisy_beta,
            'body_pose': np.zeros(69),  # Placeholder
            'global_orient': np.zeros(3),
            'transl': np.zeros(3)
        })

    print(f"\nGenerated {T} frames with noisy shape parameters")
    print(f"True shape (first 3 params): {true_beta[:3]}")

    # Apply Identity Lock
    print("\n--- Applying Identity Lock ---")
    locker = IdentityLocker(num_shape_params=10)
    result = locker.process_sequence(noisy_sequence)

    locked_sequence = result['locked_sequence']
    beta_avg = result['beta_avg']
    beta_std = result['beta_std']

    print(f"Average shape (first 3 params): {beta_avg[:3]}")
    print(f"Std deviation before locking: {beta_std[:3]}")
    print(f"Max deviation: {result['max_deviation']:.4f}")
    print(f"Mean deviation: {result['mean_deviation']:.4f}")

    # Verify all frames have same shape
    all_same = all(
        np.allclose(locked_sequence[0]['betas'], locked_sequence[i]['betas'])
        for i in range(len(locked_sequence))
    )
    print(f"\n✓ All frames have identical shape: {all_same}")

    # Test outlier rejection variant
    print("\n--- Testing Outlier Rejection Variant ---")

    # Add 5 outlier frames
    outlier_sequence = noisy_sequence.copy()
    for i in [10, 30, 50, 70, 90]:
        outlier_sequence[i]['betas'] = true_beta + np.random.randn(10) * 2.0  # Large noise

    robust_locker = IdentityLockerWithOutlierRejection(
        num_shape_params=10,
        outlier_percentile=95.0
    )
    robust_result = robust_locker.process_sequence(outlier_sequence)

    print(f"Robust average (first 3 params): {robust_result['beta_avg'][:3]}")
    print(f"Standard locker would give: {np.mean([p['betas'][:3] for p in outlier_sequence], axis=0)}")
    print(f"Ground truth: {true_beta[:3]}")

    # Compare reconstruction error
    standard_error = np.linalg.norm(beta_avg - true_beta)
    robust_error = np.linalg.norm(robust_result['beta_avg'] - true_beta)

    print(f"\nReconstruction error:")
    print(f"  Standard locker: {standard_error:.4f}")
    print(f"  Robust locker: {robust_error:.4f}")
    print(f"  Improvement: {(standard_error - robust_error) / standard_error * 100:.1f}%")

    print("\n" + "=" * 60)
    print("✓ Identity Lock Test Complete!")
    print("=" * 60)
