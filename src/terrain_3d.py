"""
3D Terrain Reconstruction Module
Generates interactive 3D visualizations from 2D satellite imagery
"""

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import torch
from PIL import Image


def estimate_height_from_image(image, method='brightness'):
    """
    Estimate terrain height from satellite image.
    
    Args:
        image: numpy array (H, W, C) or torch.Tensor
        method: 'brightness' or 'gradient'
    
    Returns:
        height_map: numpy array (H, W) with normalized heights
    """
    # Convert to numpy if tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img_array = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_array = image.cpu().numpy()
    else:
        img_array = np.array(image)
    
    # Ensure proper range [0, 1]
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    if method == 'brightness':
        # Use brightness as height proxy (darker = lower elevation)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Convert to grayscale using luminance formula
            height_map = 0.299 * img_array[:, :, 0] + \
                        0.587 * img_array[:, :, 1] + \
                        0.114 * img_array[:, :, 2]
        else:
            height_map = img_array.squeeze()
    
    elif method == 'gradient':
        # Use gradient magnitude as height indicator
        if len(img_array.shape) == 3:
            gray = 0.299 * img_array[:, :, 0] + \
                   0.587 * img_array[:, :, 1] + \
                   0.114 * img_array[:, :, 2]
        else:
            gray = img_array.squeeze()
        
        # Calculate gradients
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        height_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # Smooth the height map for realistic terrain
    height_map = gaussian_filter(height_map, sigma=2.0)
    
    # Normalize to [0, 1]
    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-8)
    
    return height_map


def create_3d_surface(image, height_map=None, exaggeration=2.0, resolution_factor=1.0):
    """
    Create 3D surface mesh from image and height map.
    
    Args:
        image: numpy array or tensor (H, W, C)
        height_map: optional pre-computed height map (H, W)
        exaggeration: height exaggeration factor (default 2.0)
        resolution_factor: downsample factor for performance (1.0 = full res)
    
    Returns:
        dict with 'x', 'y', 'z', 'colors' for 3D plotting
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img_array = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_array = image.cpu().numpy()
    else:
        img_array = np.array(image)
    
    # Ensure proper range
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    # Generate height map if not provided
    if height_map is None:
        height_map = estimate_height_from_image(img_array)
    
    # Downsample for performance if needed
    if resolution_factor < 1.0:
        from scipy.ndimage import zoom
        scale = resolution_factor
        img_array = zoom(img_array, (scale, scale, 1), order=1)
        height_map = zoom(height_map, scale, order=1)
    
    h, w = height_map.shape
    
    # Create coordinate meshgrid
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    X, Y = np.meshgrid(x, y)
    
    # Apply height exaggeration
    Z = height_map * exaggeration * max(h, w) * 0.1
    
    # Prepare colors (RGB values for surface)
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        colors = img_array[:, :, :3]
    else:
        # Grayscale to RGB
        gray = img_array.squeeze()
        colors = np.stack([gray, gray, gray], axis=2)
    
    return {
        'x': X,
        'y': Y,
        'z': Z,
        'colors': colors
    }


def create_interactive_3d_plot(surface_data, title="3D Terrain Visualization"):
    """
    Create interactive Plotly 3D surface plot.
    
    Args:
        surface_data: dict from create_3d_surface()
        title: plot title
    
    Returns:
        plotly Figure object
    """
    X = surface_data['x']
    Y = surface_data['y']
    Z = surface_data['z']
    colors = surface_data['colors']
    
    # Convert RGB array to color strings for Plotly
    h, w, _ = colors.shape
    color_array = (colors * 255).astype(np.uint8)
    
    # Create surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=colors[:, :, 0],  # Use red channel for coloring
            colorscale='earth',  # Earth-like color scheme
            showscale=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.2,
                roughness=0.8,
                fresnel=0.2
            ),
            lightposition=dict(
                x=1000,
                y=1000,
                z=1000
            )
        )
    ])
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        scene=dict(
            xaxis=dict(
                title='X (pixels)',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white',
                showbackground=True
            ),
            yaxis=dict(
                title='Y (pixels)',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white',
                showbackground=True
            ),
            zaxis=dict(
                title='Elevation',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white',
                showbackground=True
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def export_to_stl(surface_data, filename='terrain.stl'):
    """
    Export 3D terrain to STL file format.
    
    Args:
        surface_data: dict from create_3d_surface()
        filename: output STL filename
    
    Returns:
        bytes: STL file content
    """
    try:
        from stl import mesh
        import io
        
        X = surface_data['x']
        Y = surface_data['y']
        Z = surface_data['z']
        
        h, w = Z.shape
        
        # Create triangular mesh
        vertices = []
        faces = []
        
        # Generate vertices
        for i in range(h):
            for j in range(w):
                vertices.append([X[i, j], Y[i, j], Z[i, j]])
        
        # Generate faces (two triangles per grid cell)
        for i in range(h - 1):
            for j in range(w - 1):
                # Vertex indices
                v1 = i * w + j
                v2 = i * w + (j + 1)
                v3 = (i + 1) * w + j
                v4 = (i + 1) * w + (j + 1)
                
                # Two triangles
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Create mesh
        terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                terrain_mesh.vectors[i][j] = vertices[face[j], :]
        
        # Save to bytes
        buffer = io.BytesIO()
        terrain_mesh.save(filename, mode=mesh.Mode.BINARY)
        
        with open(filename, 'rb') as f:
            stl_bytes = f.read()
        
        return stl_bytes
    
    except ImportError:
        return None


# Quick test function
def generate_3d_visualization(image, exaggeration=2.0, method='brightness', resolution=1.0):
    """
    One-shot function to generate 3D visualization from image.
    
    Args:
        image: input image (numpy array or tensor)
        exaggeration: height exaggeration factor
        method: height estimation method
        resolution: resolution factor (0.5 = half res for performance)
    
    Returns:
        plotly Figure
    """
    height_map = estimate_height_from_image(image, method=method)
    surface_data = create_3d_surface(image, height_map, exaggeration, resolution)
    fig = create_interactive_3d_plot(surface_data)
    return fig
