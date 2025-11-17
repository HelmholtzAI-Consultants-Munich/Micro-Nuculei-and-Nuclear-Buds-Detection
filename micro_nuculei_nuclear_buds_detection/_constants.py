"""
Constants for annotation classes, colors, and class IDs.
"""

from typing import Dict, Tuple, Optional

# Color blind friendly colors (using RGB values 0-255)
# These colors are distinguishable for people with color vision deficiencies
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 114, 178),      # Blue - for micro-nuclei (color blind friendly)
    1: (213, 94, 0),       # Orange/Red - for nuclear-buds (color blind friendly)
}

# Class names mapping
CLASS_NAMES: Dict[int, str] = {
    0: "micro-nuclei",
    1: "nuclear-buds",
}

# Class IDs (for YOLO format)
CLASS_IDS: Dict[int, int] = {
    0: 0,  # micro-nuclei
    1: 1,  # nuclear-buds
}

# Reverse mapping: RGB color tuple to class ID
# This helps identify which class a color belongs to
COLOR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
    color: class_id for class_id, color in CLASS_COLORS.items()
}

# Napari color format (normalized 0-1 RGB)
def get_napari_color(class_id: int) -> Tuple[float, float, float]:
    """Get napari color format (normalized RGB) for a class ID."""
    r, g, b = CLASS_COLORS[class_id]
    return (r / 255.0, g / 255.0, b / 255.0)

# Get class ID from napari color (normalized RGB)
def get_class_from_napari_color(color: Tuple[float, float, float]) -> Optional[int]:
    """Get class ID from napari color (normalized RGB)."""
    # Convert normalized color to integer RGB
    r = int(round(color[0] * 255))
    g = int(round(color[1] * 255))
    b = int(round(color[2] * 255))
    
    # Find closest matching color
    rgb_tuple = (r, g, b)
    if rgb_tuple in COLOR_TO_CLASS:
        return COLOR_TO_CLASS[rgb_tuple]
    
    # If exact match not found, find closest color
    min_dist = float('inf')
    closest_class = None
    for class_id, class_color in CLASS_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb_tuple, class_color))
        if dist < min_dist:
            min_dist = dist
            closest_class = class_id
    
    return closest_class

