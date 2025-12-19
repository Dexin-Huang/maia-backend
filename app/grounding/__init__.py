"""
World Grounding Module

Transforms SMPL sequences from camera space to world coordinates.
- Ground plane estimation from foot positions
- Pelvis height lock during standing phases
- Gravity alignment
"""

from .world_grounding import WorldGrounding

__all__ = ['WorldGrounding']
