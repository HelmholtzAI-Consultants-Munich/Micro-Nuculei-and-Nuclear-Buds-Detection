"""
Micro-Nuculei and Nuclear Buds Detection - A napari plugin.

A napari plugin for detecting micro-nuclei and nuclear buds in biological images.
"""

from pathlib import Path
from ._widget import DetectionWidget
from ._data_management_widget import DataManagementWidget
from ._nuclei_segmentation_widget import NucleiSegmentationWidget

__version__ = "0.1.0"


def _get_napari_yaml_path():
    """Return the path to the napari.yaml manifest file."""
    return str(Path(__file__).parent / "napari.yaml")


