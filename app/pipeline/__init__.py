"""
Body4D Pipeline Module

Unified video-to-3D pipeline combining:
- SAM3 identity tracking
- SAM3D Body mesh recovery
- Temporal smoothing
- World grounding
"""

from .body4d_pipeline import Body4DPipeline, ProcessingEvent

__all__ = ['Body4DPipeline', 'ProcessingEvent']
