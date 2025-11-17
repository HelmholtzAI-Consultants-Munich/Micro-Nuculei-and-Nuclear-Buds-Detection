"""
Main widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

import numpy as np
import napari
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette

from ._constants import CLASS_COLORS, CLASS_NAMES, get_napari_color


class DetectionWidget(QWidget):
    """Main widget for micro-nuclei and nuclear buds detection."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_class_id = 0  # Default to micro-nuclei

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Micro-Nuclei and Nuclear Buds Detection")
        title.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(title)

        # Annotation class selection
        class_label = QLabel("Annotation Class:")
        self.layout().addWidget(class_label)
        
        # Create color selection boxes
        color_layout = QHBoxLayout()
        self.color_buttons = {}
        
        for class_id in sorted(CLASS_COLORS.keys()):
            color = CLASS_COLORS[class_id]
            class_name = CLASS_NAMES[class_id]
            
            # Create button with colored background
            color_button = QPushButton(class_name.replace("-", " ").title())
            color_button.setCheckable(True)
            color_button.setAutoExclusive(True)
            
            # Set button color
            qcolor = QColor(color[0], color[1], color[2])
            palette = color_button.palette()
            palette.setColor(QPalette.Button, qcolor)
            palette.setColor(QPalette.ButtonText, Qt.white)
            color_button.setPalette(palette)
            color_button.setStyleSheet(
                f"QPushButton {{"
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"
                f"color: white;"
                f"font-weight: bold;"
                f"border: 2px solid black;"
                f"padding: 5px;"
                f"}}"
                f"QPushButton:checked {{"
                f"border: 3px solid white;"
                f"}}"
            )
            
            # Connect click handler
            color_button.clicked.connect(
                lambda checked, cid=class_id: self._on_class_selected(cid)
            )
            
            self.color_buttons[class_id] = color_button
            color_layout.addWidget(color_button)
        
        # Set first button as checked by default
        if self.color_buttons:
            self.color_buttons[0].setChecked(True)
        
        self.layout().addLayout(color_layout)

        # Detection button
        self.detect_button = QPushButton("Detect")
        self.detect_button.clicked.connect(self._on_detect_clicked)
        self.layout().addWidget(self.detect_button)

        # Add stretch to push everything to top
        self.layout().addStretch()

    def _on_class_selected(self, class_id: int):
        """Handle class selection - update drawing color and set rectangle tool."""
        self.current_class_id = class_id
        
        # Get the napari color for this class
        color = get_napari_color(class_id)
        
        # Find or create the single "annotations" layer
        shapes_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes):
                if layer.name == "annotations":
                    shapes_layer = layer
                    break
        
        # If no annotations layer exists, create one
        if shapes_layer is None:
            show_warning("No annotations layer found. Please load an image first.")
            return
        
        # Deselect any selected shapes to prevent color changes to existing shapes
        if hasattr(shapes_layer, 'selected_data') and len(shapes_layer.selected_data) > 0:
            shapes_layer.selected_data = set()
        
        # Store the number of shapes before any new ones are added
        # This helps us identify which shape is newly added
        self._previous_shape_count = len(shapes_layer.data)
        
        # Set the current drawing color for new shapes
        # This will be the default color napari uses when drawing new shapes
        color_rgba = np.array([color[0], color[1], color[2], 1.0])
        color_rgba_2d = np.array([[color[0], color[1], color[2], 1.0]])
        
        # Set the default color for new shapes
        # Napari uses the last color in the edge_color array as the default for new shapes
        if len(shapes_layer.data) == 0:
            # No shapes exist, just set the default color
            shapes_layer.edge_color = color_rgba_2d
        else:
            # There are existing shapes - append the new color to existing colors
            # Napari will use the last color as default for new shapes
            current_colors = shapes_layer.edge_color
            
            # Ensure we have a 2D array
            if not isinstance(current_colors, np.ndarray):
                current_colors = np.array(current_colors)
            if current_colors.ndim == 1:
                current_colors = current_colors.reshape(1, -1)
            
            # Clean up: remove any extra colors beyond the number of shapes
            # (from previous class changes where default was appended but no shape was drawn)
            n_shapes = len(shapes_layer.data)
            if current_colors.shape[0] > n_shapes:
                # Trim to match number of shapes
                current_colors = current_colors[:n_shapes]
            
            # Append the new color to the existing colors
            # This makes napari use it as the default for new shapes
            updated_colors = np.vstack([current_colors, color_rgba_2d])
            shapes_layer.edge_color = updated_colors
        
        # Set the shapes layer as active
        self.viewer.layers.selection.active = shapes_layer
        
        # Set rectangle tool
        try:
            if hasattr(shapes_layer, 'mode'):
                shapes_layer.mode = 'add_rectangle'
            self.viewer.cursor = 'cross'
        except Exception:
            pass
        
        # Connect to shape addition event to set color for new shapes
        # We'll use a callback to update the color of newly added shapes
        # Check if we've already connected to this layer
        if not hasattr(shapes_layer, '_annotation_color_connected'):
            shapes_layer.events.data.connect(self._on_shape_added)
            shapes_layer._annotation_color_connected = True
        
        show_info(f"Selected class: {CLASS_NAMES[class_id]} - new shapes will be {CLASS_NAMES[class_id]}")
    
    def _on_shape_added(self, event):
        """Handle when a new shape is added - set its color based on current class."""
        shapes_layer = event.source
        if shapes_layer.name != "annotations":
            return
        
        # Get current class color
        color = get_napari_color(self.current_class_id)
        color_rgba = np.array([color[0], color[1], color[2], 1.0])
        
        # Get current number of shapes
        n_shapes = len(shapes_layer.data)
        if n_shapes == 0:
            return
        
        # Get the previous shape count (stored when class was selected)
        previous_count = getattr(self, '_previous_shape_count', 0)
        
        # Only update colors if new shapes were actually added
        if n_shapes <= previous_count:
            return
        
        # Get current edge colors - this should already be per-shape if shapes exist
        current_colors = shapes_layer.edge_color
        
        # Convert to numpy array if needed
        if not isinstance(current_colors, np.ndarray):
            current_colors = np.array(current_colors)
        
        # Ensure we have a 2D array (n_shapes, 4) for RGBA
        if current_colors.ndim == 1:
            current_colors = current_colors.reshape(1, -1)
        
        # Now handle the color array
        # Note: We may have one extra color (the default color appended when class was selected)
        if current_colors.shape[0] == previous_count:
            # Perfect case: we have exactly the right number of colors for existing shapes
            # Just append the new color(s)
            new_colors = np.tile(color_rgba, (n_shapes - previous_count, 1))
            current_colors = np.vstack([current_colors, new_colors])
        elif current_colors.shape[0] == previous_count + 1:
            # We have one extra color (the default color we appended when class was selected)
            # The last color is the default for new shapes
            # Keep existing colors for previous shapes, use default for first new shape, add more if needed
            existing_colors = current_colors[:previous_count]  # Colors for existing shapes
            default_color = current_colors[-1]  # The default color we appended
            n_new_shapes = n_shapes - previous_count
            
            # Use default color for new shapes (it should match current class color)
            new_colors = np.tile(default_color, (n_new_shapes, 1))
            current_colors = np.vstack([existing_colors, new_colors])
            
            # Ensure new shapes use the current class color (in case default doesn't match)
            for i in range(previous_count, n_shapes):
                current_colors[i] = color_rgba
        elif current_colors.shape[0] == 1 and previous_count > 0:
            # Single color for all - we need to preserve existing shapes' colors
            # Since we don't know the actual colors, we'll keep the single color for existing
            # and add new color for new shapes
            existing_colors = np.tile(current_colors[0], (previous_count, 1))
            new_colors = np.tile(color_rgba, (n_shapes - previous_count, 1))
            current_colors = np.vstack([existing_colors, new_colors])
        elif current_colors.shape[0] == 1 and previous_count == 0:
            # Single color, no previous shapes - just expand with new color
            current_colors = np.tile(color_rgba, (n_shapes, 1))
        elif current_colors.shape[0] < previous_count:
            # Fewer colors than shapes - pad with last color for existing, add new
            last_color = current_colors[-1] if current_colors.shape[0] > 0 else color_rgba
            padding = np.tile(last_color, (previous_count - current_colors.shape[0], 1))
            existing_colors = np.vstack([current_colors, padding])
            new_colors = np.tile(color_rgba, (n_shapes - previous_count, 1))
            current_colors = np.vstack([existing_colors, new_colors])
        elif current_colors.shape[0] > previous_count + 1:
            # More colors than expected (more than one extra) - trim to previous, add new
            existing_colors = current_colors[:previous_count]
            new_colors = np.tile(color_rgba, (n_shapes - previous_count, 1))
            current_colors = np.vstack([existing_colors, new_colors])
        else:
            # Shouldn't happen, but handle it
            if current_colors.shape[0] < n_shapes:
                new_colors = np.tile(color_rgba, (n_shapes - current_colors.shape[0], 1))
                current_colors = np.vstack([current_colors, new_colors])
            elif current_colors.shape[0] > n_shapes:
                current_colors = current_colors[:n_shapes]
        
        # Ensure we have exactly n_shapes colors
        if current_colors.shape[0] != n_shapes:
            if current_colors.shape[0] > n_shapes:
                current_colors = current_colors[:n_shapes]
            else:
                padding = np.tile(color_rgba, (n_shapes - current_colors.shape[0], 1))
                current_colors = np.vstack([current_colors, padding])
        
        # Only update colors for newly added shapes (from previous_count onwards)
        # This ensures existing shapes keep their colors
        for i in range(previous_count, n_shapes):
            current_colors[i] = color_rgba
        
        # Update the layer's edge colors
        # Make sure no shapes are selected before updating to avoid affecting selected shapes
        if hasattr(shapes_layer, 'selected_data') and len(shapes_layer.selected_data) > 0:
            shapes_layer.selected_data = set()
        
        shapes_layer.edge_color = current_colors
        
        # Update the previous count for next time
        self._previous_shape_count = n_shapes
    
    def _on_detect_clicked(self):
        """Handle detect button click."""
        if len(self.viewer.layers) == 0:
            show_info("Please load an image first.")
            return

        # TODO: Implement detection logic
        show_info("Detection functionality to be implemented.")

