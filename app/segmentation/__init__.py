"""
Segmentation Module

Video segmentation and identity tracking for multi-person scenarios.
- SAM3 video segmentation for identity-consistent masks
- Occlusion detection and handling (optional)
"""

from .sam3_tracker import SAM3Tracker, PersonTrack

__all__ = ['SAM3Tracker', 'PersonTrack']
