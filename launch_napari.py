#!/usr/bin/env python
"""
Launch script for napari with Micro-Nuclei and Nuclear Buds Detection widgets.

This script automatically launches napari and loads both widgets:
- Micro-Nuclei Detection widget
- Data Management widget

Usage:
    python launch_napari.py
"""

import napari
from micro_nuculei_nuclear_buds_detection._widget import DetectionWidget
from micro_nuculei_nuclear_buds_detection._data_management_widget import DataManagementWidget


def main():
    """Launch napari and automatically load the widgets."""
    print("Launching napari with Micro-Nuclei and Nuclear Buds Detection widgets...")
    
    # Create napari viewer
    viewer = napari.Viewer(title="Micro-Nuclei and Nuclear Buds Detection")
    
    # Load widgets after a short delay to ensure window is ready
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication
    
    def load_widgets():
        """Load both widgets into the viewer."""
        try:
            # Load Micro-Nuclei Detection widget
            detection_widget = DetectionWidget(viewer)
            viewer.window.add_dock_widget(
                detection_widget,
                name="Micro-Nuclei Detection",
                area="right"
            )
            print("✓ Micro-Nuclei Detection widget loaded")
            
            # Load Data Management widget
            data_widget = DataManagementWidget(viewer)
            viewer.window.add_dock_widget(
                data_widget,
                name="Data Management",
                area="right"
            )
            print("✓ Data Management widget loaded")
            
            print("\nNapari is ready! Both widgets are loaded in the dock.")
        except Exception as e:
            print(f"Error loading widgets: {e}")
            import traceback
            traceback.print_exc()
    
    # Use a timer to ensure the window is ready before loading widgets
    app = QApplication.instance()
    if app is not None:
        timer = QTimer()
        timer.timeout.connect(load_widgets)
        timer.setSingleShot(True)
        timer.start(300)  # Wait 300ms for window to be ready
    else:
        # If QApplication doesn't exist, try loading immediately
        # (napari.run() will create it)
        load_widgets()
    
    # Run napari (this blocks until napari is closed)
    napari.run()


if __name__ == "__main__":
    main()

