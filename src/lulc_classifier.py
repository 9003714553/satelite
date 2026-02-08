"""
Land Use & Land Cover (LULC) Classification Module
Classifies satellite imagery into different land cover types
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image


# Land cover class definitions
LULC_CLASSES = {
    0: {'name': 'Water', 'color': [0, 119, 190], 'label': '💧 Water'},
    1: {'name': 'Forest', 'color': [0, 100, 0], 'label': '🌲 Forest'},
    2: {'name': 'Urban', 'color': [178, 34, 34], 'label': '🏙️ Urban'},
    3: {'name': 'Barren', 'color': [139, 90, 43], 'label': '🏜️ Barren'},
    4: {'name': 'Vegetation', 'color': [50, 205, 50], 'label': '🌾 Vegetation'}
}


def load_segmentation_model(device='cpu'):
    """
    Load pre-trained segmentation model.
    For now, we'll use a lightweight rule-based approach.
    Can be upgraded to DeepLabV3 later.
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        model or None (for rule-based)
    """
    # Placeholder for future deep learning model
    # try:
    #     import segmentation_models_pytorch as smp
    #     model = smp.DeepLabV3Plus(
    #         encoder_name="resnet50",
    #         encoder_weights="imagenet",
    #         classes=5
    #     )
    #     return model.to(device)
    # except:
    #     return None
    
    return None  # Use rule-based for now


def classify_land_cover_simple(image):
    """
    Simple rule-based land cover classification.
    Uses color and texture features to classify pixels.
    
    Args:
        image: numpy array (H, W, C) in range [0, 1] or torch.Tensor
    
    Returns:
        segmentation_mask: numpy array (H, W) with class indices
    """
    # Convert to numpy
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
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=2)
    
    h, w, c = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Extract color channels
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Calculate indices
    brightness = (R + G + B) / 3.0
    
    # NDVI-like index (Green - Red) / (Green + Red)
    vegetation_index = (G - R) / (G + R + 0.001)
    
    # Water index (Blue dominant)
    water_index = B - (R + G) / 2.0
    
    # Urban index (high brightness, low vegetation)
    urban_index = brightness - vegetation_index
    
    # Classification rules (priority order)
    
    # 1. Water: High blue, low red/green
    water_mask = (water_index > 0.1) & (B > 0.4)
    mask[water_mask] = 0
    
    # 2. Forest: High vegetation index, darker
    forest_mask = (vegetation_index > 0.15) & (brightness < 0.6) & (~water_mask)
    mask[forest_mask] = 1
    
    # 3. Urban: High brightness, low vegetation, reddish or gray
    urban_mask = ((brightness > 0.5) & (vegetation_index < 0.05)) | \
                 ((R > G) & (R > B) & (brightness > 0.4))
    urban_mask = urban_mask & (~water_mask) & (~forest_mask)
    mask[urban_mask] = 2
    
    # 4. Barren: Low vegetation, brownish
    barren_mask = (vegetation_index < 0.0) & (brightness > 0.3) & (brightness < 0.6)
    barren_mask = barren_mask & (~water_mask) & (~forest_mask) & (~urban_mask)
    mask[barren_mask] = 3
    
    # 5. Vegetation: Everything else with positive vegetation index
    vegetation_mask = (vegetation_index > 0.05) & (mask == 0)
    mask[vegetation_mask] = 4
    
    # Smooth the mask
    mask = cv2.medianBlur(mask, 5)
    
    return mask


def create_color_coded_map(segmentation_mask):
    """
    Convert segmentation mask to RGB color-coded image.
    
    Args:
        segmentation_mask: numpy array (H, W) with class indices
    
    Returns:
        color_map: numpy array (H, W, 3) with RGB colors
    """
    h, w = segmentation_mask.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, class_info in LULC_CLASSES.items():
        mask = segmentation_mask == class_id
        color_map[mask] = class_info['color']
    
    return color_map


def calculate_class_percentages(segmentation_mask):
    """
    Calculate percentage of each land cover class.
    
    Args:
        segmentation_mask: numpy array (H, W) with class indices
    
    Returns:
        dict: {class_name: percentage}
    """
    total_pixels = segmentation_mask.size
    percentages = {}
    
    for class_id, class_info in LULC_CLASSES.items():
        count = np.sum(segmentation_mask == class_id)
        percentage = (count / total_pixels) * 100
        percentages[class_info['name']] = percentage
    
    return percentages


def get_class_distribution_data(percentages):
    """
    Prepare data for pie chart visualization.
    
    Args:
        percentages: dict from calculate_class_percentages()
    
    Returns:
        labels, values, colors for plotting
    """
    labels = []
    values = []
    colors = []
    
    for class_id, class_info in LULC_CLASSES.items():
        class_name = class_info['name']
        if percentages[class_name] > 0.5:  # Only show if > 0.5%
            labels.append(class_info['label'])
            values.append(percentages[class_name])
            # Convert RGB to hex
            rgb = class_info['color']
            hex_color = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
            colors.append(hex_color)
    
    return labels, values, colors


def analyze_spatial_distribution(segmentation_mask, class_id):
    """
    Analyze where a specific class is located (North, South, East, West).
    
    Args:
        segmentation_mask: numpy array (H, W)
        class_id: int (0-4)
    
    Returns:
        str: description of location
    """
    h, w = segmentation_mask.shape
    mask = segmentation_mask == class_id
    
    if not mask.any():
        return "Not found"
    
    # Find center of mass
    y_coords, x_coords = np.where(mask)
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    # Determine quadrant
    mid_y, mid_x = h / 2, w / 2
    
    vertical = "North" if center_y < mid_y else "South"
    horizontal = "West" if center_x < mid_x else "East"
    
    # Calculate concentration
    coverage = mask.sum() / mask.size * 100
    
    if coverage > 50:
        return "Distributed across entire area"
    elif coverage > 25:
        return f"Mostly in {vertical}-{horizontal} region"
    else:
        return f"Concentrated in {vertical}-{horizontal} corner"


# Main classification function
def classify_and_visualize(image):
    """
    One-shot function to classify and create visualization.
    
    Args:
        image: input image (numpy array or tensor)
    
    Returns:
        dict with 'mask', 'color_map', 'percentages', 'labels', 'values', 'colors'
    """
    mask = classify_land_cover_simple(image)
    color_map = create_color_coded_map(mask)
    percentages = calculate_class_percentages(mask)
    labels, values, colors = get_class_distribution_data(percentages)
    
    return {
        'mask': mask,
        'color_map': color_map,
        'percentages': percentages,
        'labels': labels,
        'values': values,
        'colors': colors
    }
