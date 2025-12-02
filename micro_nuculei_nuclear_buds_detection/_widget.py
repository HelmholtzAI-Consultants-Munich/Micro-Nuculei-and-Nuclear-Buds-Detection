"""
Main widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import napari
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from ._constants import CLASS_COLORS, CLASS_NAMES, get_napari_color, get_class_from_napari_color, ANNOTATION_LAYER_NAME, NUCLEI_SEGMENTATION_LAYER_NAME


class DetectionWidget(QWidget):
    """Main widget for micro-nuclei and nuclear buds detection."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_class_id = 0  # Default to micro-nuclei
        
        # Track shape count to detect new shapes
        self._shape_count_at_selection = 0
        
        # Store paths for saving/loading (set by data management widget)
        self.annotation_dir = None
        self.dataset_path = None

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Micro-Nuclei and Nuclear Buds Detection")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        self.layout().addWidget(title)

        # # Annotation class selection
        # class_label = QLabel("Select Annotation Class:")
        # class_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        # self.layout().addWidget(class_label)
        
        # Create class selection buttons
        button_layout = QHBoxLayout()
        self.class_buttons = {}
        
        for class_id in sorted(CLASS_COLORS.keys()):
            color = CLASS_COLORS[class_id]
            class_name = CLASS_NAMES[class_id]
            
            # Create button
            button = QPushButton(class_name.replace("-", " ").title())
            button.setCheckable(True)
            button.setAutoExclusive(True)
            
            # Style button with class color
            button.setStyleSheet(
                f"QPushButton {{"
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"
                f"color: white;"
                f"font-weight: bold;"
                f"border: 2px solid #333;"
                f"border-radius: 5px;"
                f"padding: 8px 15px;"
                f"min-height: 20px;"
                f"}}"
                f"QPushButton:hover {{"
                f"border: 2px solid white;"
                f"}}"
                f"QPushButton:checked {{"
                f"border: 3px solid white;"
                f"background-color: rgb({min(255, color[0] + 20)}, {min(255, color[1] + 20)}, {min(255, color[2] + 20)});"
                f"}}"
            )
            
            # Connect click handler
            button.clicked.connect(
                lambda checked, cid=class_id: self._on_class_selected(cid)
            )
            
            self.class_buttons[class_id] = button
            button_layout.addWidget(button)
        
        # Set first button as checked by default
        if self.class_buttons:
            self.class_buttons[0].setChecked(True)
        
        self.layout().addLayout(button_layout)

        # # Tool selection buttons
        # tool_label = QLabel("Tools:")
        # tool_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        # self.layout().addWidget(tool_label)
        
        tool_layout = QHBoxLayout()
        
        # Select shapes button
        self.select_button = QPushButton("Select Shapes")
        self.select_button.setStyleSheet(
            "QPushButton {"
            # "background-color: #7B1FA2;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "}"
            "QPushButton:hover {"
            # "background-color: #6A1B9A;"
            "border: 2px solid white;"
            "}"
        )
        self.select_button.clicked.connect(self._on_select_shapes_clicked)
        tool_layout.addWidget(self.select_button)
        
        # Move camera button
        self.pan_button = QPushButton("Move Camera")
        self.pan_button.setStyleSheet(
            "QPushButton {"
            # "background-color: #E91E63;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "}"
            "QPushButton:hover {"
            # "background-color: #C2185B;"
            "border: 2px solid white;"
            "}"
        )
        self.pan_button.clicked.connect(self._on_move_camera_clicked)
        tool_layout.addWidget(self.pan_button)
        
        self.layout().addLayout(tool_layout)

        # Detection button
        detect_layout = QHBoxLayout()
        detect_layout.addStretch()  # Add stretch before button to center it
        
        self.detect_button = QPushButton("Detect")
        self.detect_button.setStyleSheet(
            "QPushButton {"
            "background-color: #4CAF50;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "margin-top: 15px;"
            "}"
            "QPushButton:hover {"
            "background-color: #45a049;"
            "border: 2px solid white;"
            "}"
        )
        self.detect_button.clicked.connect(self._on_detect_clicked)
        detect_layout.addWidget(self.detect_button)
        
        detect_layout.addStretch()  # Add stretch after button to center it
        self.layout().addLayout(detect_layout)

        # Add stretch to push everything to top
        self.layout().addStretch()

    def _get_annotations_layer(self):
        """Get or create the annotations shapes layer."""
        # Find existing annotations layer
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == ANNOTATION_LAYER_NAME:
                return layer
        return None

    def _hide_nuclei_segmentation_layer(self):
        """Hide the nuclei segmentation layer if it exists."""
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                layer.visible = False
                break

    def _show_annotation_layer(self):
        """Show the annotation layer if it exists."""
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == ANNOTATION_LAYER_NAME:
                layer.visible = True
                break

    def _on_class_selected(self, class_id: int):
        """Handle class selection - update drawing color and set rectangle tool."""
        self.current_class_id = class_id
        
        # Hide nuclei segmentation layer and show annotation layer
        self._hide_nuclei_segmentation_layer()
        self._show_annotation_layer()
        
        # Get the annotations layer
        shapes_layer = self._get_annotations_layer()
        if shapes_layer is None:
            show_warning("No annotations layer found. Please load an image first.")
            return
        
        # Deselect any selected shapes
        shapes_layer.selected_data = set()
        
        # Store current shape count
        self._shape_count_at_selection = len(shapes_layer.data)
        
        # Get the napari color for this class
        color = get_napari_color(class_id)
        color_rgba = np.array([color[0], color[1], color[2], 1.0])
        
        # Set the default color for new shapes using current_edge_color if available
        # This is the cleanest way and doesn't affect existing shapes
        try:
            if hasattr(shapes_layer, 'current_edge_color'):
                shapes_layer.current_edge_color = color_rgba
            else:
                # Fallback: use edge_color array approach
                self._set_default_color_via_array(shapes_layer, color_rgba)
        except Exception as e:
            # If current_edge_color doesn't work, use array approach
            self._set_default_color_via_array(shapes_layer, color_rgba)
        
        # Set the shapes layer as active
        self.viewer.layers.selection.active = shapes_layer
        
        # Set rectangle tool
        try:
            shapes_layer.mode = 'add_rectangle'
        except Exception:
            pass
        
        # Connect to shape addition event if not already connected
        if not hasattr(shapes_layer, '_color_callback_connected'):
            shapes_layer.events.data.connect(self._on_shape_data_changed)
            shapes_layer._color_callback_connected = True
        
        show_info(f"Selected class: {CLASS_NAMES[class_id]}")

    def _set_default_color_via_array(self, shapes_layer, color_rgba):
        """Set default color using edge_color array (fallback method)."""
        n_shapes = len(shapes_layer.data)
        color_rgba_2d = color_rgba.reshape(1, -1) if color_rgba.ndim == 1 else color_rgba
        
        if n_shapes == 0:
            # No shapes - just set the default color
            shapes_layer.edge_color = color_rgba_2d
        else:
            # Get current colors
            current_colors = shapes_layer.edge_color
            
            # Convert to numpy array
            if not isinstance(current_colors, np.ndarray):
                current_colors = np.array(current_colors)
            if current_colors.ndim == 1:
                current_colors = current_colors.reshape(1, -1)
            
            # Ensure we have exactly n_shapes colors (remove any appended defaults)
            if current_colors.shape[0] > n_shapes:
                current_colors = current_colors[:n_shapes]
            elif current_colors.shape[0] < n_shapes:
                # Pad with last color
                if current_colors.shape[0] > 0:
                    last_color = current_colors[-1]
                    padding = np.tile(last_color, (n_shapes - current_colors.shape[0], 1))
                    current_colors = np.vstack([current_colors, padding])
                else:
                    current_colors = np.tile(color_rgba, (n_shapes, 1))
            
            # Append default color for new shapes
            # Napari uses the last color as default
            updated_colors = np.vstack([current_colors, color_rgba_2d])
            shapes_layer.edge_color = updated_colors

    def _on_shape_data_changed(self, event):
        """Handle when shape data changes - set color for new shapes."""
        shapes_layer = event.source
        if shapes_layer.name != ANNOTATION_LAYER_NAME:
            return
        
        n_shapes = len(shapes_layer.data)
        
        # Only process if new shapes were added
        if n_shapes <= self._shape_count_at_selection:
            return
        
        # Get current class color
        color = get_napari_color(self.current_class_id)
        color_rgba = np.array([color[0], color[1], color[2], 1.0])
        
        # Get current colors
        current_colors = shapes_layer.edge_color
        
        # Convert to numpy array
        if not isinstance(current_colors, np.ndarray):
            current_colors = np.array(current_colors)
        if current_colors.ndim == 1:
            current_colors = current_colors.reshape(1, -1)
        
        # Ensure we have the right number of colors
        n_existing = self._shape_count_at_selection
        n_new = n_shapes - n_existing
        
        if current_colors.shape[0] == n_existing:
            # Perfect - just append new colors
            new_colors = np.tile(color_rgba, (n_new, 1))
            updated_colors = np.vstack([current_colors, new_colors])
        elif current_colors.shape[0] > n_existing:
            # We have extra colors (might include a default)
            # Keep existing colors, add new ones
            existing_colors = current_colors[:n_existing]
            new_colors = np.tile(color_rgba, (n_new, 1))
            updated_colors = np.vstack([existing_colors, new_colors])
        else:
            # Fewer colors than expected - pad and add new
            if current_colors.shape[0] > 0:
                last_color = current_colors[-1]
                padding = np.tile(last_color, (n_existing - current_colors.shape[0], 1))
                existing_colors = np.vstack([current_colors, padding])
            else:
                existing_colors = np.tile(color_rgba, (n_existing, 1))
            new_colors = np.tile(color_rgba, (n_new, 1))
            updated_colors = np.vstack([existing_colors, new_colors])
        
        # Deselect shapes before updating colors
        shapes_layer.selected_data = set()
        
        # Update colors
        shapes_layer.edge_color = updated_colors
        
        # Update shape count
        self._shape_count_at_selection = n_shapes

    def _on_select_shapes_clicked(self):
        """Handle select shapes button click - switch to select mode."""
        # Hide nuclei segmentation layer and show annotation layer
        self._hide_nuclei_segmentation_layer()
        self._show_annotation_layer()
        
        shapes_layer = self._get_annotations_layer()
        if shapes_layer is None:
            show_warning("No annotations layer found. Please load an image first.")
            return
        
        # Set the shapes layer as active
        self.viewer.layers.selection.active = shapes_layer
        
        # Set to select mode
        try:
            shapes_layer.mode = 'select'
            show_info("Select mode: Click shapes to select, Delete key to remove")
        except Exception as e:
            show_warning(f"Could not switch to select mode: {str(e)}")

    def _on_move_camera_clicked(self):
        """Handle move camera button click - switch to pan_zoom mode."""
        # Hide nuclei segmentation layer and show annotation layer
        self._hide_nuclei_segmentation_layer()
        self._show_annotation_layer()
        
        shapes_layer = self._get_annotations_layer()
        if shapes_layer is None:
            show_warning("No annotations layer found. Please load an image first.")
            return
        
        # Set the shapes layer as active
        self.viewer.layers.selection.active = shapes_layer
        
        # Set to pan_zoom mode
        try:
            shapes_layer.mode = 'pan_zoom'
            show_info("Pan/Zoom mode: Click and drag to move, scroll to zoom")
        except Exception as e:
            show_warning(f"Could not switch to pan/zoom mode: {str(e)}")

    def _on_detect_clicked(self):
        """Handle detect button click."""
        # Hide nuclei segmentation layer and show annotation layer
        self._hide_nuclei_segmentation_layer()
        self._show_annotation_layer()
        
        if len(self.viewer.layers) == 0:
            show_info("Please load an image first.")
            return

        # TODO: Implement detection logic
        show_info("Detection functionality to be implemented.")
    
    def has_annotations(self, image_path: Path) -> bool:
        """Check if annotations exist for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if annotation file exists
        """
        if not self.annotation_dir:
            return False
        
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
        except (ValueError, AttributeError):
            annotation_file = self.annotation_dir / f"{image_path.stem}.txt"
        
        return annotation_file.exists()
    
    def save_annotations(self, image_path: Path) -> Optional[Path]:
        """Save current annotations to file in YOLO format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to saved annotation file or None if error
        """
        if not self.annotation_dir:
            return None
        
        # Get image dimensions from the first image layer
        image_width = None
        image_height = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                if layer.data.ndim >= 2:
                    if layer.data.ndim == 2:
                        image_height, image_width = layer.data.shape
                    elif layer.data.ndim == 3:
                        image_height, image_width = layer.data.shape[-2:]
                    break
        
        if image_width is None or image_height is None:
            show_warning("Could not determine image dimensions. Annotations not saved.")
            return None
        
        # Collect all annotations in YOLO format
        yolo_annotations = []
        
        # Only process the annotation layer with the unique name
        annotation_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == ANNOTATION_LAYER_NAME:
                annotation_layer = layer
                break
        
        if annotation_layer is None:
            show_warning(f"No annotation layer '{ANNOTATION_LAYER_NAME}' found. Nothing to save.")
            return None
        
        # Process only the annotation layer
        if annotation_layer.data:
            # Get edge colors for all shapes
            edge_colors = annotation_layer.edge_color
            
            # Handle different color formats
            if isinstance(edge_colors, np.ndarray):
                if edge_colors.ndim == 1:
                    edge_colors = np.array([edge_colors])
                elif edge_colors.ndim == 2:
                    pass
            else:
                edge_colors = np.array([edge_colors])
            
            # Process each shape
            for shape_idx, shape in enumerate(annotation_layer.data):
                if len(shape) == 0:
                    continue
                
                # Determine class ID from this shape's color
                class_id = 0  # Default
                
                if shape_idx < len(edge_colors):
                    shape_color = edge_colors[shape_idx]
                    if len(shape_color) >= 3:
                        color_tuple = tuple(shape_color[:3])
                        detected_class = get_class_from_napari_color(color_tuple)
                        if detected_class is not None:
                            class_id = detected_class
                
                # Get bounding box from shape vertices, napari swaps x and y coordinates
                x_coords = shape[:, 1]
                y_coords = shape[:, 0]
                
                x_min = float(np.min(x_coords))
                x_max = float(np.max(x_coords))
                y_min = float(np.min(y_coords))
                y_max = float(np.max(y_coords))
                
                # Calculate center and dimensions
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min
                
                # Normalize to 0-1 range
                center_x_norm = center_x / image_width
                center_y_norm = center_y / image_height
                width_norm = width / image_width
                height_norm = height / image_height
                
                # Clamp to [0, 1] range
                center_x_norm = max(0.0, min(1.0, center_x_norm))
                center_y_norm = max(0.0, min(1.0, center_y_norm))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))
                
                # YOLO format: class_id center_x center_y width height
                yolo_annotations.append(
                    f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                )
        
        # Save to YOLO format (.txt file)
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
            annotation_file.parent.mkdir(parents=True, exist_ok=True)
        except (ValueError, AttributeError):
            annotation_file = self.annotation_dir / f"{image_path.stem}.txt"
        
        with open(annotation_file, 'w') as f:
            f.writelines(yolo_annotations)
        
        # Show relative path in message
        try:
            display_path = str(image_path.relative_to(self.dataset_path))
        except (ValueError, AttributeError):
            display_path = image_path.name
        
        show_info(f"Annotations saved in YOLO format for {display_path}")
        return annotation_file
    
    def load_annotations(self, image_path: Path) -> bool:
        """Load annotations from YOLO format (.txt) file if it exists.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if annotations were loaded successfully
        """
        if not self.annotation_dir:
            return False
        
        # Get annotation file path preserving subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
        except (ValueError, AttributeError):
            annotation_file = self.annotation_dir / f"{image_path.stem}.txt"
        
        if not annotation_file.exists():
            return False
        
        try:
            # Get image dimensions from the first image layer
            image_width = None
            image_height = None
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    if layer.data.ndim >= 2:
                        if layer.data.ndim == 2:
                            image_height, image_width = layer.data.shape
                        elif layer.data.ndim == 3:
                            image_height, image_width = layer.data.shape[-2:]
                    break
            
            if image_width is None or image_height is None:
                show_warning("Could not determine image dimensions. Annotations not loaded.")
                return False
            
            # Collect all rectangles and their class IDs
            rectangles = []
            class_ids = []
            
            with open(annotation_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    try:
                        # YOLO format: class_id center_x center_y width height
                        class_id = int(parts[0])
                        center_x_norm = float(parts[1])
                        center_y_norm = float(parts[2])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])
                        
                        # Convert normalized coordinates back to pixel coordinates
                        center_x = center_x_norm * image_width
                        center_y = center_y_norm * image_height
                        width = width_norm * image_width
                        height = height_norm * image_height
                        
                        # Calculate bounding box corners
                        x_min = center_x - width / 2.0
                        x_max = center_x + width / 2.0
                        y_min = center_y - height / 2.0
                        y_max = center_y + height / 2.0
                        
                        # Create rectangle shape (4 corners)
                        # Note: napari uses (row, col) which is (y, x)
                        rectangle = np.array([
                            [y_min, x_min],
                            [y_max, x_min],
                            [y_max, x_max],
                            [y_min, x_max]
                        ])
                        
                        rectangles.append(rectangle)
                        class_ids.append(class_id)
                        
                    except (ValueError, IndexError) as e:
                        show_warning(f"Error parsing annotation line: {line} - {str(e)}")
                        continue
            
            annotations_loaded = False
            
            # Create a single "annotations" layer with all rectangles
            if rectangles:
                # Create color array - one color per rectangle based on its class
                edge_colors = []
                for class_id in class_ids:
                    edge_color = get_napari_color(class_id)
                    edge_colors.append([edge_color[0], edge_color[1], edge_color[2], 1.0])
                
                edge_color_array = np.array(edge_colors)
                
                shapes_layer = self.viewer.add_shapes(
                    rectangles,
                    name=ANNOTATION_LAYER_NAME,
                    edge_color=edge_color_array,
                    face_color="transparent",
                    shape_type="rectangle",
                )
                shapes_layer.edge_width = 2
                annotations_loaded = True
            
            # Get display path for messages
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except (ValueError, AttributeError):
                display_path = image_path.name
            
            if annotations_loaded:
                show_info(f"Loaded annotations for {display_path}")
            else:
                show_warning(f"No valid annotations found in file for {display_path}")
            
            return annotations_loaded
            
        except Exception as e:
            import traceback
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except (ValueError, AttributeError):
                display_path = image_path.name
            error_msg = f"Error loading annotations for {display_path}: {str(e)}\n{traceback.format_exc()}"
            show_warning(error_msg)
            return False
    
    def create_empty_annotation_layer(self):
        """Create an empty shapes layer for annotations."""
        default_class_id = 0
        default_color = get_napari_color(default_class_id)
        shapes_layer = self.viewer.add_shapes(
            data=[],
            name=ANNOTATION_LAYER_NAME,
            face_color="transparent",
        )
        shapes_layer.edge_color = np.array([[default_color[0], default_color[1], default_color[2], 1.0]])
        shapes_layer.edge_width = 2
        return shapes_layer

