"""
Micro-Nuculei and Nuclear Buds Detection - A napari plugin.

A napari plugin for detecting micro-nuclei and nuclear buds in biological images.
"""

from pathlib import Path
from ._widget import DetectionWidget
from ._data_management_widget import DataManagementWidget

__version__ = "0.1.0"


def _get_napari_yaml_path():
    """Return the path to the napari.yaml manifest file."""
    return str(Path(__file__).parent / "napari.yaml")



def widget_factory(viewer):
    """
    Called by napari when the plugin loads.
    Automatically opens both dock widgets.
    """
    print("Opening widgets automatically...")

    # Create and add the main detection widget
    detection_widget = DetectionWidget(viewer)
    viewer.window.add_dock_widget(
        detection_widget, 
        area="right", 
        name="Micro-Nuclei and Nuclear Buds Detection"
    )

    # Create and add the data management widget
    data_widget = DataManagementWidget(viewer)
    viewer.window.add_dock_widget(
        data_widget, 
        area="right", 
        name="Data Management"
    )

    # Return one widget (napari expects a single return)
    return detection_widget


