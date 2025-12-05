"""
Nuclei Segmentation widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

import numpy as np
import colorsys
import json
import os
import hashlib
from pathlib import Path
from typing import Optional
from cellpose import models, core, utils
import napari
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QHBoxLayout
from qtpy.QtCore import Qt
from skimage import measure
from ._constants import NUCLEI_SEGMENTATION_LAYER_NAME, ANNOTATION_LAYER_NAME

class NucleiSegmentationWidget(QWidget):
    """Widget for nuclei segmentation."""

    def __init__(self, napari_viewer, nuclei_segmentation_params_path = None):
        super().__init__()
        self.viewer = napari_viewer
        
        # Default segmentation parameters
        self.nuclei_segmentation_params_path = nuclei_segmentation_params_path
        try:
            with open(self.nuclei_segmentation_params_path, 'r') as f:
                nuclei_segmentation_params = json.load(f)
            if 'min_size' in nuclei_segmentation_params:
                self.min_size = nuclei_segmentation_params['min_size']
            else:
                self.min_size = 150
            if 'cellprob_threshold' in nuclei_segmentation_params:
                self.cellprob_threshold = nuclei_segmentation_params['cellprob_threshold']
            else:
                self.cellprob_threshold = 0.0
            if 'diameter' in nuclei_segmentation_params:
                self.diameter = nuclei_segmentation_params['diameter']
                if self.diameter == 0:
                    self.diameter = None
            else:
                self.diameter = None
        except Exception as e:
            show_warning(f"Error loading nuclei segmentation parameters: {str(e)}")
            self.min_size = 150
            self.cellprob_threshold = 0.0
            self.diameter = None
        
        # Store segmentation state for interactive updates
        self.model = None
        self.image_data = None
        self.nuclei_layer = None
        
        # Store paths for saving/loading (set by data management widget)
        self.nuclei_segmentation_dir = None
        self.dataset_path = None

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Nuclei Segmentation")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        self.layout().addWidget(title)
        
        # Segmentation parameters section
        params_label = QLabel("Segmentation Parameters:")
        params_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.layout().addWidget(params_label)
        
        # Min size spinbox
        minsize_layout = QHBoxLayout()
        minsize_label = QLabel("Min Area:")
        minsize_label.setMinimumWidth(100)
        minsize_layout.addWidget(minsize_label)
        
        # Help icon with tooltip
        help_label = QLabel("?")
        help_label.setStyleSheet(
            "QLabel {"
            "color: #A0A0A0;"
            "font-weight: bold;"
            "font-size: 14px;"
            "background-color: transparent;"
            "border: 1px solid #A0A0A0;"
            "border-radius: 10px;"
            "min-width: 20px;"
            "max-width: 20px;"
            "min-height: 20px;"
            "max-height: 20px;"
            "}"
        )
        help_label.setAlignment(Qt.AlignCenter)

        # Optional: make cursor indicate it's informational
        help_label.setCursor(Qt.WhatsThisCursor)
        # Or: help_label.setCursor(Qt.PointingHandCursor)

        help_label.setToolTip(
            "Minimum area (in pixels) for detected nuclei.\n"
            "Nuclei smaller than this value will be filtered out.\n"
            "Use this to remove noise and small artifacts from the segmentation."
        )

        minsize_layout.addWidget(help_label)
        
        self.minsize_spinbox = QSpinBox()
        self.minsize_spinbox.setMinimum(0)
        self.minsize_spinbox.setMaximum(10000)
        self.minsize_spinbox.setSingleStep(1)
        self.minsize_spinbox.setValue(self.min_size)
        minsize_layout.addWidget(self.minsize_spinbox)
        
        self.layout().addLayout(minsize_layout)
        
        # Cell Probability parameter
        cellprob_layout = QHBoxLayout()
        cellprob_label = QLabel("Cell Score:")
        cellprob_label.setMinimumWidth(100)
        cellprob_layout.addWidget(cellprob_label)
        
        cellprob_help = QLabel("?")
        cellprob_help.setStyleSheet(
            "QLabel {"
            "color: #A0A0A0;"
            "font-weight: bold;"
            "font-size: 14px;"
            "background-color: transparent;"
            "border: 1px solid #A0A0A0;"
            "border-radius: 10px;"
            "min-width: 20px;"
            "max-width: 20px;"
            "min-height: 20px;"
            "max-height: 20px;"
            "}"
        )
        cellprob_help.setAlignment(Qt.AlignCenter)
        cellprob_help.setCursor(Qt.WhatsThisCursor)
        cellprob_help.setToolTip(
            "Cell score for Cellpose segmentation.\n"
            "Decrease if not as many nuceli as you’d expect are shown.\n Increase if too many are shown.\n"
            "Typical range: -6.0 to 6.0. Default: 0.0"
        )
        cellprob_layout.addWidget(cellprob_help)
        
        self.cellprob_threshold_spinbox = QDoubleSpinBox()
        self.cellprob_threshold_spinbox.setMinimum(-6.0)
        self.cellprob_threshold_spinbox.setMaximum(6.0)
        self.cellprob_threshold_spinbox.setSingleStep(0.1)
        self.cellprob_threshold_spinbox.setDecimals(2)
        self.cellprob_threshold_spinbox.setValue(self.cellprob_threshold)
        cellprob_layout.addWidget(self.cellprob_threshold_spinbox)
        self.layout().addLayout(cellprob_layout)
        
        # Diameter parameter
        diameter_layout = QHBoxLayout()
        diameter_label = QLabel("Diameter:")
        diameter_label.setMinimumWidth(100)
        diameter_layout.addWidget(diameter_label)
        
        diameter_help = QLabel("?")
        diameter_help.setStyleSheet(
            "QLabel {"
            "color: #A0A0A0;"
            "font-weight: bold;"
            "font-size: 14px;"
            "background-color: transparent;"
            "border: 1px solid #A0A0A0;"
            "border-radius: 10px;"
            "min-width: 20px;"
            "max-width: 20px;"
            "min-height: 20px;"
            "max-height: 20px;"
            "}"
        )
        diameter_help.setAlignment(Qt.AlignCenter)
        diameter_help.setCursor(Qt.WhatsThisCursor)
        diameter_help.setToolTip(
            "Expected cell diameter in pixels (0 = auto-detect).\n"
            "Set to 0 to let Cellpose automatically estimate the diameter.\n"
            "Default: 0 (auto-detect)"
        )
        diameter_layout.addWidget(diameter_help)
        
        self.diameter_spinbox = QSpinBox()
        self.diameter_spinbox.setMinimum(0)
        self.diameter_spinbox.setMaximum(1000)
        self.diameter_spinbox.setSingleStep(1)
        self.diameter_spinbox.setValue(0 if self.diameter is None else int(self.diameter))
        diameter_layout.addWidget(self.diameter_spinbox)
        self.layout().addLayout(diameter_layout)
        
        # Apply, Save as Default, and Show Segmentation buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.setStyleSheet(
            "QPushButton {"
            "background-color: #4CAF50;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 6px 10px;"
            "min-height: 20px;"
            "min-width: 70px;"
            "}"
            "QPushButton:hover {"
            "background-color: #45a049;"
            "border: 2px solid white;"
            "}"
        )
        self.apply_button.clicked.connect(self._on_apply_clicked)
        buttons_layout.addWidget(self.apply_button)
        
        # Save as default button
        self.save_default_button = QPushButton("Default")
        self.save_default_button.setStyleSheet(
            "QPushButton {"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 6px 10px;"
            "min-height: 20px;"
            "min-width: 70px;"
            "}"
            "QPushButton:hover {"
            "border: 2px solid white;"
            "}"
        )
        self.save_default_button.clicked.connect(self._on_save_default_clicked)
        buttons_layout.addWidget(self.save_default_button)

        # Show segmentation / Hide annotations button
        self.toggle_view_button = QPushButton("Show Seg.")
        self.toggle_view_button.setStyleSheet(
            "QPushButton {"
            "background-color: #00BCD4;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 6px 10px;"
            "min-height: 20px;"
            "min-width: 80px;"
            "}"
            "QPushButton:hover {"
            "background-color: #0097A7;"
            "border: 2px solid white;"
            "}"
        )
        self.toggle_view_button.clicked.connect(self._on_toggle_view_clicked)
        buttons_layout.addWidget(self.toggle_view_button)
        
        buttons_layout.addStretch()
        self.layout().addLayout(buttons_layout)

        # Add stretch to push everything to top
        self.layout().addStretch()

    def _get_active_image_layer(self):
        """Get the active image layer."""
        # First try to get the active layer if it's an image
        if len(self.viewer.layers) > 0:
            active_layer = self.viewer.layers.selection.active
            if isinstance(active_layer, napari.layers.Image):
                return active_layer
        
        # Otherwise, find the first image layer
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return layer
        return None
    

    def _masks_to_shapes(self, masks):
        """Convert segmentation masks to napari shapes (polygons).
        
        Args:
            masks: numpy array of shape (height, width) with integer labels for each instance
            
        Returns:
            List of polygon coordinates for each instance
        """
        shapes = []
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)
        
        for label_id in unique_labels:
            # Create binary mask for this instance
            binary_mask = (masks == label_id).astype(np.uint8)
            
            # Find contours
            contours = measure.find_contours(binary_mask, 0.5)
            
            if len(contours) > 0:
                # Use the largest contour (main shape)
                largest_contour = max(contours, key=len)
                # Convert to (N, 2) array: [[y1, x1], [y2, x2], ...]
                # Note: napari uses (row, col) which is (y, x)
                polygon = largest_contour
                shapes.append(polygon)
        
        return shapes

    def _generate_colors(self, n_instances):
        """Generate distinct colors for each segmentation instance.
        
        Args:
            n_instances: Number of instances to generate colors for
            
        Returns:
            Array of shape (n_instances, 4) with RGBA colors
        """
        colors = []
        # Use a colormap to generate distinct colors
        # Using a simple approach: cycle through hues
        if n_instances == 0:
            return np.empty((0, 4))
        for i in range(n_instances):
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.7, 0.9)
            # Add alpha (partial transparency)
            rgba = [rgb[0], rgb[1], rgb[2], 0.5]  # 50% transparency
            colors.append(rgba)
        
        return np.array(colors)


    def _on_save_default_clicked(self):
        """Handle save as default button click - save all current parameter values to JSON file."""
        try:
            # Get current values from spinboxes
            current_min_size = self.minsize_spinbox.value()
            current_cellprob_threshold = self.cellprob_threshold_spinbox.value()
            current_diameter_value = self.diameter_spinbox.value()
            
            # Update instance variables
            self.min_size = current_min_size
            self.cellprob_threshold = current_cellprob_threshold
            self.diameter = None if current_diameter_value == 0 else current_diameter_value
            
            # Load existing parameters or create new dict
            try:
                with open(self.nuclei_segmentation_params_path, 'r') as f:
                    params = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                params = {}
            
            # Update all parameters
            params['min_size'] = current_min_size
            params['cellprob_threshold'] = current_cellprob_threshold
            params['diameter'] = current_diameter_value  # Save 0 for None/auto-detect
            
            # Save to JSON file
            with open(self.nuclei_segmentation_params_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            show_info(f"Saved parameters as default: min_size={current_min_size}, cellprob_threshold={current_cellprob_threshold:.2f}, diameter={'auto' if current_diameter_value == 0 else current_diameter_value}")
        except Exception as e:
            show_warning(f"Error saving default values: {str(e)}")
            import traceback
            print(traceback.format_exc())


    
    def remove_small_masks(self, masks: np.ndarray, min_size: int) -> np.ndarray:
        """
        Remove labeled objects smaller than min_size and relabel remaining
        objects to 1..K (background stays 0). Works for 2D/3D integer masks.
        """
        if masks.size == 0:
            return masks

        max_label = int(masks.max())
        if max_label == 0:
            return np.zeros_like(masks)

        # Count pixels per label (0..max_label)
        counts = np.bincount(masks.ravel(), minlength=max_label + 1)

        # Keep labels with enough pixels (exclude background 0)
        keep = counts >= min_size
        keep[0] = False

        # Build mapping old_label -> new_label (1..K), removed -> 0
        mapping = np.zeros(max_label + 1, dtype=masks.dtype)
        mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=masks.dtype)

        # Remap entire mask in one vectorized step
        return mapping[masks]

    def run_cellpose(self) -> Optional[np.ndarray]:
        """Extract image from active layer, preprocess, run Cellpose, return masks.
        
        Returns:
            masks array or None if error
        """
        try:
            # Get the active image layer
            image_layer = self._get_active_image_layer()
            if image_layer is None:
                show_warning("Please load an image first.")
                return None
            
            # Get image data
            image_data = image_layer.data.copy()
            
            # Handle different image formats
            if image_data.ndim == 3:
                # Multi-channel or RGB image - use first channel or convert to grayscale
                if image_data.shape[2] <= 4:
                    # Assume it's RGB/RGBA - convert to grayscale
                    if image_data.dtype == np.uint8:
                        # Convert RGB to grayscale
                        if image_data.shape[2] == 3:
                            image_data = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
                        elif image_data.shape[2] == 4:
                            image_data = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
                    else:
                        # Use first channel
                        image_data = image_data[:, :, 0]
                else:
                    # Use first channel
                    image_data = image_data[:, :, 0]
            
            # Ensure image is 2D
            if image_data.ndim != 2:
                show_warning(f"Image must be 2D, got shape {image_data.shape}")
                return None
            
            # Normalize image to 0-255 uint8 if needed
            if image_data.dtype != np.uint8:
                image_min = image_data.min()
                image_max = image_data.max()
                if image_max > image_min:
                    image_data = ((image_data - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            
            # Check for GPU availability
            use_GPU = core.use_gpu()
            if use_GPU:
                show_info("Using GPU acceleration (CUDA or MPS)")
            else:
                show_info("Using CPU (GPU not available)")
            
            # Initialize or reuse Cellpose model
            if self.model is None:
                # Model type "nuclei" for version 3.1.1.1
                self.model = models.Cellpose(model_type='nuclei', gpu=use_GPU)
            
            # Store preprocessed image data
            self.image_data = image_data
            
            # Run segmentation
            show_info("Running segmentation...")
            
            masks, flows_tuple, styles, diams = self.model.eval(
                image_data,
                diameter=self.diameter,
                channels=[0, 0],  # Grayscale image
                flow_threshold=0.4,  # Default flow threshold
                cellprob_threshold=self.cellprob_threshold,
            )
            
            return masks
            
        except Exception as e:
            error_msg = f"Error during Cellpose segmentation: {str(e)}"
            show_warning(error_msg)
            import traceback
            print(traceback.format_exc())
            return None

    def add_to_layer(self, masks: np.ndarray) -> None:
        """Display masks in nuclei-segmentation layer.
        
        Args:
            masks: numpy array of segmentation masks to display
        """
        # Convert masks to shapes
        shapes_data = self._masks_to_shapes(masks)
        
        if len(shapes_data) == 0:
            # No shapes - clear the layer if it exists
            if self.nuclei_layer is not None:
                self.nuclei_layer.data = []
            return
        
        # Generate colors for each instance
        n_instances = len(shapes_data)
        colors = self._generate_colors(n_instances)
        
        # Update existing layer or create new one
        if self.nuclei_layer is not None:
            # Update existing layer
            self.nuclei_layer.data = shapes_data
            self.nuclei_layer.face_color = colors
        else:
            # Remove existing nuclei segmentation layer if it exists
            for layer in list(self.viewer.layers):
                if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                    self.viewer.layers.remove(layer)
            
            # Create new shapes layer
            self.nuclei_layer = self.viewer.add_shapes(
                shapes_data,
                name=NUCLEI_SEGMENTATION_LAYER_NAME,
                shape_type='polygon',
                edge_color='white',
                edge_width=1,
                face_color=colors,
                opacity=0.5,
            )

    def run_segmentation(self, image_data):
        """Run nuclei segmentation on image data.
        
        Public method for data management widget. Maintains backward compatibility.
        
        Args:
            image_data: numpy array of image data (can be 2D, 3D, or RGB)
            
        Returns:
            tuple: (masks, None) or (None, None) if error (flows no longer returned)
        """
        try:
            # Add image to viewer if not already present (for run_cellpose to work)
            # Check if image is already in viewer
            image_in_viewer = False
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    # Check if data matches
                    if np.array_equal(layer.data, image_data):
                        image_in_viewer = True
                        break
            
            if not image_in_viewer:
                # Temporarily add image to viewer
                temp_layer = self.viewer.add_image(image_data, name="temp_segmentation_image")
                temp_added = True
            else:
                temp_added = False
            
            # Run cellpose (extracts from active layer)
            masks = self.run_cellpose()
            
            if masks is None:
                if temp_added:
                    self.viewer.layers.remove(temp_layer)
                return None, None
            
            # Remove small masks
            filtered_masks = self.remove_small_masks(masks, self.min_size)
            
            # Remove temporary layer if we added it
            if temp_added:
                self.viewer.layers.remove(temp_layer)
            
            return filtered_masks, None
            
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            show_warning(error_msg)
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def create_segmentation_layer(self, masks, flows=None):
        """Create a nuclei segmentation layer from masks.
        
        Public method for data management widget. Maintains backward compatibility.
        
        Args:
            masks: numpy array of segmentation masks (already filtered)
            flows: optional flows data (ignored, kept for backward compatibility)
            
        Returns:
            shapes_layer or None if error
        """
        if masks is None:
            return None
        
        # Use add_to_layer to display masks
        self.add_to_layer(masks)
        
        # Store metadata in layer
        if self.nuclei_layer is not None:
            self.nuclei_layer._masks_data = masks
            # Store original masks (same as filtered in this case since masks are already filtered)
            # This allows future filtering with different min_size without rerunning the model
            self.nuclei_layer._original_masks_data = masks.copy()
            n_instances = len(self._masks_to_shapes(masks))
            show_info(f"Segmentation complete! Found {n_instances} nuclei.")
        
        return self.nuclei_layer
    
    def _on_apply_clicked(self):
        """Handle Apply button click - recompute segmentation with current parameters.
        
        Only reruns the model if:
        - cellprob_threshold changed
        - diameter changed
        - original masks are not stored in the layer
        
        Otherwise, just filters existing original masks with new min_size.
        """
        try:
            # Save old params to check for changes
            old_cellprob_threshold = self.cellprob_threshold
            old_diameter = self.diameter
            
            # Update current params to instance variables
            self.min_size = self.minsize_spinbox.value()
            self.cellprob_threshold = self.cellprob_threshold_spinbox.value()
            self.diameter = None if self.diameter_spinbox.value() == 0 else self.diameter_spinbox.value()
            
            # Check if we need to rerun the model
            cellprob_threshold_changed = old_cellprob_threshold != self.cellprob_threshold
            diameter_changed = old_diameter != self.diameter
            original_masks_not_stored = (self.nuclei_layer is None or 
                                       not hasattr(self.nuclei_layer, '_original_masks_data') or 
                                       self.nuclei_layer._original_masks_data is None)
            
            if cellprob_threshold_changed or diameter_changed or original_masks_not_stored:
                # Need to rerun the model
                masks = self.run_cellpose()
                
                if masks is None:
                    show_warning("Segmentation failed. Please check the image and parameters.")
                    return
                
                # Store original masks before filtering (will be stored in layer after add_to_layer)
                original_masks = masks.copy()
            else:
                # Reuse existing original masks
                original_masks = self.nuclei_layer._original_masks_data.copy()
                show_info("Reusing existing segmentation, applying new min_size filter...")
            
            # Filter small masks
            filtered_masks = self.remove_small_masks(original_masks, self.min_size)
            
            # Add to layer (creates layer if it doesn't exist)
            self.add_to_layer(filtered_masks)
            
            # Store metadata in layer
            if self.nuclei_layer is not None:
                self.nuclei_layer._masks_data = filtered_masks
                # Store original masks (before filtering) for future use
                self.nuclei_layer._original_masks_data = original_masks
                n_instances = len(self._masks_to_shapes(filtered_masks))
                show_info(f"Segmentation updated! Found {n_instances} nuclei.")
            
        except Exception as e:
            error_msg = f"Error during recomputation: {str(e)}"
            show_warning(error_msg)
            import traceback
            print(traceback.format_exc())

    def has_segmentation(self, image_path: Path) -> bool:
        """Check if nuclei segmentation exists for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if segmentation file exists
        """
        if not self.nuclei_segmentation_dir:
            return False
        
        # Get segmentation file path preserving subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            segmentation_file = self.nuclei_segmentation_dir / relative_path.with_suffix('.npy')
        except (ValueError, AttributeError):
            # Fallback: if path is not relative, use just the stem
            segmentation_file = self.nuclei_segmentation_dir / f"{image_path.stem}.npy"
        
        return segmentation_file.exists()
    
    def load_segmentation(self, image_path: Path) -> bool:
        """Load nuclei segmentation masks from .npy file if it exists.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if segmentation was loaded successfully
        """
        if not self.has_segmentation(image_path):
            return False
        
        # Reset nuclei_layer reference in case it points to a removed layer
        self.nuclei_layer = None
        
        # Get segmentation file path preserving subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            segmentation_file = self.nuclei_segmentation_dir / relative_path.with_suffix('.npy')
        except (ValueError, AttributeError):
            segmentation_file = self.nuclei_segmentation_dir / f"{image_path.stem}.npy"
        
        try:
            # Load masks
            masks = np.load(segmentation_file)
            
            # Convert masks to shapes
            shapes_data = self._masks_to_shapes(masks)
            
            if len(shapes_data) == 0:
                return False
            
            # Generate colors for each instance
            n_instances = len(shapes_data)
            colors = self._generate_colors(n_instances)
            
            # Remove any existing nuclei segmentation layer (shouldn't exist after clearing, but be safe)
            for layer in list(self.viewer.layers):
                if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                    self.viewer.layers.remove(layer)
            
            # Create shapes layer
            shapes_layer = self.viewer.add_shapes(
                shapes_data,
                name=NUCLEI_SEGMENTATION_LAYER_NAME,
                shape_type='polygon',
                edge_color='white',
                edge_width=1,
                face_color=colors,
                opacity=0.5,
            )
            
            # Store masks in layer metadata
            shapes_layer._masks_data = masks
            
            # Store reference
            self.nuclei_layer = shapes_layer
            
            # Get display path for messages
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except (ValueError, AttributeError):
                display_path = image_path.name
            
            show_info(f"Loaded nuclei segmentation for {display_path}")
            return True
            
        except Exception as e:
            import traceback
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except (ValueError, AttributeError):
                display_path = image_path.name
            error_msg = f"Error loading nuclei segmentation for {display_path}: {str(e)}\n{traceback.format_exc()}"
            show_warning(error_msg)
            return False
    
    def save_segmentation(self, image_path: Path) -> Optional[Path]:
        """Save current nuclei segmentation masks to file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to saved segmentation file or None if error
        """
        if not self.nuclei_segmentation_dir:
            return None
        
        # Find the nuclei segmentation layer
        nuclei_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                nuclei_layer = layer
                break
        
        if nuclei_layer is None or not hasattr(nuclei_layer, '_masks_data'):
            return None
        
        masks = nuclei_layer._masks_data
        
        # Save masks as .npy file
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            segmentation_file = self.nuclei_segmentation_dir / relative_path.with_suffix('.npy')
            segmentation_file.parent.mkdir(parents=True, exist_ok=True)
        except (ValueError, AttributeError):
            segmentation_file = self.nuclei_segmentation_dir / f"{image_path.stem}.npy"
        
        # Save masks
        np.save(segmentation_file, masks)
        
        # Show relative path in message
        try:
            display_path = str(image_path.relative_to(self.dataset_path))
        except (ValueError, AttributeError):
            display_path = image_path.name
        show_info(f"Nuclei segmentation saved for {display_path}")
        return segmentation_file

    def _on_toggle_view_clicked(self):
        """Handle toggle view button click - show nuclei segmentation, hide annotations."""
        # Show nuclei segmentation layer
        nuclei_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                nuclei_layer = layer
                break
        
        if nuclei_layer is not None:
            nuclei_layer.visible = True
            show_info("Nuclei segmentation layer shown.")
        else:
            show_warning("No nuclei segmentation layer found. Please segment nuclei first.")
        
        # Hide annotation layer
        annotation_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == ANNOTATION_LAYER_NAME:
                annotation_layer = layer
                break
        
        if annotation_layer is not None:
            annotation_layer.visible = False
            show_info("Bounding box annotation layer hidden.")

