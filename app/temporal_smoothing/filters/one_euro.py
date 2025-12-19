"""
One-Euro Filter Implementation

A low-pass filter with adaptive cutoff frequency based on signal velocity.
Used extensively in VR/AR hand tracking (Meta Quest, HTC Vive, Apple Vision Pro).

Paper: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
https://cristal.univ-lille.fr/~casiez/1euro/

Author: Georges Casiez, Nicolas Roussel, Daniel Vogel (2012)
"""

import time
from typing import Optional


class LowPassFilter:
    """First-order low-pass filter (exponential smoothing)"""

    def __init__(self, alpha: float, initial_value: float = 0.0):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing
            initial_value: Initial filter state
        """
        self.alpha = alpha
        self.y = initial_value
        self.is_initialized = False

    def filter(self, x: float) -> float:
        """
        Apply low-pass filter to new value

        Args:
            x: New input value

        Returns:
            Smoothed output value
        """
        if not self.is_initialized:
            self.y = x
            self.is_initialized = True
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y

        return self.y

    def reset(self):
        """Reset filter state"""
        self.is_initialized = False


class OneEuroFilter:
    """
    One-Euro Filter: Adaptive low-pass filter with velocity-based cutoff

    Key insight: Use higher cutoff (less smoothing) when signal changes quickly,
    lower cutoff (more smoothing) when signal is stable.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        initial_value: float = 0.0,
    ):
        """
        Initialize One-Euro Filter

        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother but more lag.
                Typical range: 0.5 - 2.0 Hz
                - 0.5 Hz: Very smooth, noticeable lag (meditation apps)
                - 1.0 Hz: Balanced (default, general use)
                - 2.0 Hz: Responsive, minimal lag (fast action)

            beta: Speed coefficient. Higher = more responsive to fast movements.
                Typical range: 0.001 - 0.01
                - 0.001: Minimal adaptation (mostly constant smoothing)
                - 0.007: Balanced (default)
                - 0.01: Aggressive adaptation (sports, dance)

            d_cutoff: Cutoff frequency for derivative filter (Hz).
                Usually kept at 1.0 Hz (default)

            initial_value: Initial value for the filter
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Low-pass filter for the signal
        # Initialize with default dt of 1/30 (30 fps)
        self.x_filter = LowPassFilter(self._alpha_from_dt(min_cutoff, 1.0 / 30.0), initial_value)

        # Low-pass filter for the derivative (velocity)
        self.dx_filter = LowPassFilter(self._alpha_from_dt(d_cutoff, 1.0 / 30.0), 0.0)

        self.last_time: Optional[float] = None

    def _alpha_from_dt(self, cutoff: float, dt: float) -> float:
        """
        Calculate smoothing factor alpha from cutoff frequency and actual time delta

        Based on Casiez et al. formula: α = 1 / (1 + τ/dt) where τ = 1/(2πfc)

        Args:
            cutoff: Cutoff frequency (Hz)
            dt: Actual time delta between samples (seconds)

        Returns:
            Alpha value for low-pass filter (0-1)
        """
        tau = 1.0 / (2.0 * 3.14159265359 * cutoff)  # Time constant
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Apply One-Euro filter to new value

        Args:
            x: New input value
            timestamp: Optional timestamp (seconds). If None, uses current time.

        Returns:
            Smoothed output value
        """
        # Handle timestamp
        if timestamp is None:
            timestamp = time.time()

        # Calculate time delta
        if self.last_time is None:
            self.last_time = timestamp
            return x  # First sample, no filtering

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt <= 0:
            # Invalid time delta, return last value
            return self.x_filter.y

        # Calculate derivative (velocity)
        dx = (x - self.x_filter.y) / dt

        # Smooth the derivative using actual dt
        self.dx_filter.alpha = self._alpha_from_dt(self.d_cutoff, dt)
        dx_smoothed = self.dx_filter.filter(dx)

        # Calculate adaptive cutoff frequency
        # Higher velocity → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)

        # Update alpha using actual time delta (dt)
        self.x_filter.alpha = self._alpha_from_dt(cutoff, dt)
        x_filtered = self.x_filter.filter(x)

        return x_filtered

    def filter_with_velocity(self, x: float, dx: float) -> float:
        """
        Apply One-Euro filter when velocity is already known

        Useful for sequential filtering where dt is constant (e.g., video frames)

        Args:
            x: New input value
            dx: Velocity (change from previous frame)

        Returns:
            Smoothed output value
        """
        # Smooth the derivative
        dx_smoothed = self.dx_filter.filter(dx)

        # Calculate adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)

        # Update alpha and apply filter
        self.x_filter.alpha = self._alpha(cutoff)
        x_filtered = self.x_filter.filter(x)

        return x_filtered

    def reset(self):
        """Reset filter state"""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class MultiChannelOneEuroFilter:
    """
    One-Euro Filter for multi-dimensional signals

    Applies independent filters to each dimension.
    Useful for joint rotations (3 angles) or positions (x, y, z).
    """

    def __init__(
        self,
        num_channels: int,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """
        Initialize multi-channel One-Euro Filter

        Args:
            num_channels: Number of independent channels to filter
            min_cutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative filter (Hz)
        """
        self.num_channels = num_channels
        self.filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(num_channels)
        ]

    def filter(self, x_values: list[float], timestamp: Optional[float] = None) -> list[float]:
        """
        Apply filters to multi-channel signal

        Args:
            x_values: List of input values (length = num_channels)
            timestamp: Optional timestamp (seconds)

        Returns:
            List of smoothed values
        """
        if len(x_values) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} values, got {len(x_values)}"
            )

        return [
            self.filters[i].filter(x_values[i], timestamp)
            for i in range(self.num_channels)
        ]

    def filter_with_velocity(
        self, x_values: list[float], dx_values: list[float]
    ) -> list[float]:
        """
        Apply filters when velocities are known

        Args:
            x_values: List of input values
            dx_values: List of velocities

        Returns:
            List of smoothed values
        """
        if len(x_values) != self.num_channels or len(dx_values) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} values for both x and dx"
            )

        return [
            self.filters[i].filter_with_velocity(x_values[i], dx_values[i])
            for i in range(self.num_channels)
        ]

    def reset(self):
        """Reset all filters"""
        for f in self.filters:
            f.reset()


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate noisy signal
    t = np.linspace(0, 10, 300)  # 10 seconds @ 30 Hz
    signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
    noise = np.random.normal(0, 0.1, len(t))  # Gaussian noise
    noisy_signal = signal + noise

    # Apply One-Euro filter
    filter_smooth = OneEuroFilter(min_cutoff=0.5, beta=0.005)  # Very smooth
    filter_balanced = OneEuroFilter(min_cutoff=1.0, beta=0.007)  # Balanced
    filter_responsive = OneEuroFilter(min_cutoff=2.0, beta=0.01)  # Responsive

    filtered_smooth = []
    filtered_balanced = []
    filtered_responsive = []

    for i, (time_val, noisy_val) in enumerate(zip(t, noisy_signal)):
        filtered_smooth.append(filter_smooth.filter(noisy_val, time_val))
        filtered_balanced.append(filter_balanced.filter(noisy_val, time_val))
        filtered_responsive.append(filter_responsive.filter(noisy_val, time_val))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'g-', label='Ground Truth', linewidth=2)
    plt.plot(t, noisy_signal, 'gray', alpha=0.3, label='Noisy Input')
    plt.plot(t, filtered_smooth, 'b-', label='Smooth (min_cutoff=0.5)', linewidth=1.5)
    plt.plot(t, filtered_balanced, 'r-', label='Balanced (min_cutoff=1.0)', linewidth=1.5)
    plt.plot(t, filtered_responsive, 'm-', label='Responsive (min_cutoff=2.0)', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('One-Euro Filter Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('one_euro_filter_test.png', dpi=150)
    print("Test plot saved to: one_euro_filter_test.png")
