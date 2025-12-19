"""
Animation export utilities for SAM 3D Body
"""
import numpy as np
import trimesh
from io import BytesIO
import zipfile
import tempfile
import os
from typing import List, Dict


def export_ply_sequence(frames_data: List[Dict], faces: np.ndarray) -> bytes:
    """
    Export animation as a ZIP file containing PLY files for each frame.
    This can be imported into Blender as a mesh sequence.
    
    Parameters:
    -----------
    frames_data : List[Dict]  
        List of frames, each with 'people' array containing SMPL params
    faces : np.ndarray
        Triangle faces (F, 3)
    
    Returns:
    --------
    bytes : ZIP file containing PLY sequence
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add README
        readme = f"""SAM 3D Body Animation Export
================================

Frames: {len(frames_data)}
Format: PLY mesh sequence

To import into Blender:
1. Install the 'Stop Motion OBJ' or 'Mesh Sequence Cache' addon
2. Import the PLY files as a sequence
3. Set frame rate to 30 FPS

Each PLY file represents one frame of animation.
All meshes share the same topology (faces).
"""
        zip_file.writestr('README.txt', readme)
        
        # Export each frame
        for frame_idx, frame_data in enumerate(frames_data):
            if 'people' not in frame_data or len(frame_data['people']) == 0:
                continue
                
            # Export each person in the frame
            for person_idx, person in enumerate(frame_data['people']):
                if 'vertices' not in person:
                    continue
                    
                vertices = np.array(person['vertices'], dtype=np.float32)
                
                # Create trimesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                
                # Export to PLY
                ply_bytes = trimesh.exchange.ply.export_ply(mesh)
                
                # Add to ZIP
                filename = f'frame_{frame_idx:04d}_person_{person_idx}.ply'
                zip_file.writestr(filename, ply_bytes)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_single_glb(frames_data: List[Dict], faces: np.ndarray, frame_idx: int = 0) -> bytes:
    """
    Export a single frame as GLB (for testing/preview).
    
    Parameters:
    -----------
    frames_data : List[Dict]
        List of frames
    faces : np.ndarray
        Triangle faces
    frame_idx : int
        Which frame to export
    
    Returns:
    --------
    bytes : GLB file
    """
    if frame_idx >= len(frames_data):
        frame_idx = 0
    
    frame_data = frames_data[frame_idx]
    
    if 'people' not in frame_data or len(frame_data['people']) == 0:
        raise ValueError("No people in frame")
    
    # Create scene with all people in the frame
    meshes = []
    for person in frame_data['people']:
        if 'vertices' in person:
            vertices = np.array(person['vertices'], dtype=np.float32)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            meshes.append(mesh)
    
    if len(meshes) == 0:
        raise ValueError("No valid meshes in frame")
    
    # Create scene
    scene = trimesh.Scene(meshes)
    
    # Export to GLB
    glb_bytes = trimesh.exchange.gltf.export_glb(scene)
    
    return glb_bytes
