"""
Nuclei Segmentation widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

import numpy as np
import colorsys
import json
import napari
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QHBoxLayout
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
        except Exception as e:
            show_warning(f"Error loading nuclei segmentation parameters: {str(e)}")
            self.min_size = 150

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Nuclei Segmentation")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        self.layout().addWidget(title)
        
        # Segmentation parameters section
        params_label = QLabel("Segmentation Parameters:")
        params_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.layout().addWidget(params_label)
        
        # Min size spinbox
        minsize_layout = QHBoxLayout()
        minsize_label = QLabel("Min Size:")
        minsize_label.setMinimumWidth(150)
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
            "Minimum size (in pixels) for detected nuclei.\n"
            "Nuclei smaller than this value will be filtered out.\n"
            "Use this to remove noise and small artifacts from the segmentation."
        )

        minsize_layout.addWidget(help_label)
        
        self.minsize_spinbox = QSpinBox()
        self.minsize_spinbox.setMinimum(0)
        self.minsize_spinbox.setMaximum(10000)
        self.minsize_spinbox.setSingleStep(1)
        self.minsize_spinbox.setValue(self.min_size)
        self.minsize_spinbox.editingFinished.connect(self._on_minsize_changed)
        minsize_layout.addWidget(self.minsize_spinbox)
        
        # Save as default button
        self.save_default_button = QPushButton("Save as Default")
        self.save_default_button.setStyleSheet(
            "QPushButton {"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 4px 10px;"
            "min-height: 20px;"
            "min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "border: 2px solid white;"
            "}"
        )
        self.save_default_button.clicked.connect(self._on_save_default_clicked)
        minsize_layout.addWidget(self.save_default_button)
        
        self.layout().addLayout(minsize_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()  # Add stretch before buttons to center them
        
        # Segment nuclei button
        self.segment_button = QPushButton("Segment Nuclei")
        self.segment_button.setStyleSheet(
            "QPushButton {"
            "background-color: #FF9800;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "}"
            "QPushButton:hover {"
            "background-color: #F57C00;"
            "border: 2px solid white;"
            "}"
        )
        self.segment_button.clicked.connect(self._on_segment_clicked)
        buttons_layout.addWidget(self.segment_button)

        # Show segmentation / Hide annotations button
        self.toggle_view_button = QPushButton("Show Segmentation")
        self.toggle_view_button.setStyleSheet(
            "QPushButton {"
            "background-color: #00BCD4;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "}"
            "QPushButton:hover {"
            "background-color: #0097A7;"
            "border: 2px solid white;"
            "}"
        )
        self.toggle_view_button.clicked.connect(self._on_toggle_view_clicked)
        buttons_layout.addWidget(self.toggle_view_button)
        
        buttons_layout.addStretch()  # Add stretch after buttons to center them
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

    def _on_minsize_changed(self):
        """Handle min size spinbox change."""
        self.min_size = self.minsize_spinbox.value()
        
        # Apply min_size filter to currently displayed masks if they exist
        self._apply_min_size_to_current_masks()

    def _on_save_default_clicked(self):
        """Handle save as default button click - save current min_size value to JSON file."""
        try:
            # Get current value from spinbox
            current_min_size = self.minsize_spinbox.value()
            self.min_size = current_min_size
            
            # Load existing parameters or create new dict
            try:
                with open(self.nuclei_segmentation_params_path, 'r') as f:
                    params = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                params = {}
            
            # Update min_size in parameters
            params['min_size'] = current_min_size
            
            # Save to JSON file
            with open(self.nuclei_segmentation_params_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            show_info(f"Saved min_size={current_min_size} as default")
        except Exception as e:
            show_warning(f"Error saving default value: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _apply_min_size_to_current_masks(self):
        """Apply min_size filter to the currently displayed masks layer."""
        # Find the nuclei segmentation layer
        nuclei_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                nuclei_layer = layer
                break
        
        if nuclei_layer is None:
            # No masks layer found
            return
        
        # Get the original masks (before any filtering)
        # Use _original_masks_data if available, otherwise use _masks_data
        if hasattr(nuclei_layer, '_original_masks_data'):
            original_masks = nuclei_layer._original_masks_data
        elif hasattr(nuclei_layer, '_masks_data'):
            original_masks = nuclei_layer._masks_data
        else:
            # No masks to filter
            return
        
        # Apply min_size filter
        if self.min_size > 0:
            try:
                from cellpose import utils
                # Try to use cellpose's remove_small_masks if available
                if hasattr(utils, 'remove_small_masks'):
                    filtered_masks = utils.remove_small_masks(original_masks.copy(), min_size=self.min_size)
                else:
                    # Use manual filtering if function doesn't exist
                    filtered_masks = self._manual_remove_small_masks(original_masks, self.min_size)
            except (ImportError, AttributeError):
                # Fallback: manual filtering if cellpose utils not available or function doesn't exist
                filtered_masks = self._manual_remove_small_masks(original_masks, self.min_size)
        else:
            filtered_masks = original_masks.copy()
        
        # Convert filtered masks to shapes
        shapes_data = self._masks_to_shapes(filtered_masks)
        
        if len(shapes_data) == 0:
            # No shapes after filtering - clear the layer but keep original masks
            nuclei_layer.data = []
            nuclei_layer.edge_color = np.array([[1.0, 1.0, 1.0, 1.0]])
            nuclei_layer._masks_data = filtered_masks
            return
        
        # Generate colors for each instance
        n_instances = len(shapes_data)
        colors = self._generate_colors(n_instances)
        
        # Update the layer with filtered shapes
        nuclei_layer.data = shapes_data
        nuclei_layer.face_color = colors
        
        # Update stored masks (keep original for future filtering)
        nuclei_layer._masks_data = filtered_masks
        if not hasattr(nuclei_layer, '_original_masks_data'):
            nuclei_layer._original_masks_data = original_masks

    def _manual_remove_small_masks_old(self, masks, min_size):
        """Manually remove small masks without cellpose utils."""
        filtered_masks = np.zeros_like(masks)
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)
        
        new_label = 1
        for label_id in unique_labels:
            mask = (masks == label_id).astype(bool)
            size = np.sum(mask)
            if size >= min_size:
                filtered_masks[mask] = new_label
                new_label += 1
        
        return filtered_masks

    
    def _manual_remove_small_masks(self, masks: np.ndarray, min_size: int) -> np.ndarray:
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

    def _on_segment_clicked(self):
        """Handle segment nuclei button click."""
        # Get the active image layer
        image_layer = self._get_active_image_layer()
        if image_layer is None:
            show_warning("Please load an image first.")
            return
        
        try:
            show_info("Starting nuclei segmentation with Cellpose...")
            
            # Import cellpose
            try:
                from cellpose import models, core
            except ImportError:
                show_warning("Cellpose is not installed. Please install it with: pip install cellpose==3.1.1.1")
                return
            
            # Get image data
            image_data = image_layer.data
            
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
                return
            
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
            
            # Initialize Cellpose model
            # Model type "nuclei" for version 3.1.1.1
            model = models.Cellpose(model_type='nuclei', gpu=use_GPU)
            
            # Run segmentation
            show_info("Running segmentation...")
            masks, flows, styles, diams = model.eval(
                image_data,
                diameter=None,  # Auto-detect diameter
                channels=[0, 0],  # Grayscale image
                flow_threshold=0.4,  # Default flow threshold
                cellprob_threshold=0.0,
            )
            
            # Store flows for interactive flow threshold adjustment
            # Store original masks before filtering (for real-time filtering)
            original_masks = masks.copy()
            
            # Apply min_size filter as post-processing
            masks = self._manual_remove_small_masks(masks, self.min_size)
            
            # Convert masks to shapes
            show_info("Converting masks to shapes...")
            shapes_data = self._masks_to_shapes(masks)
            
            if len(shapes_data) == 0:
                show_warning("No nuclei were segmented. Try adjusting the image or parameters.")
                return
            
            # Remove existing nuclei segmentation layer if it exists
            for layer in list(self.viewer.layers):
                if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                    self.viewer.layers.remove(layer)
            
            # Generate colors for each instance
            n_instances = len(shapes_data)
            colors = self._generate_colors(n_instances)
            
            # Create shapes layer
            shapes_layer = self.viewer.add_shapes(
                shapes_data,
                name=NUCLEI_SEGMENTATION_LAYER_NAME,
                shape_type='polygon',
                edge_color='white',
                edge_width=1,
                face_color=colors,
                opacity=0.5,  # Additional transparency control
            )
            
            # Store masks in layer metadata for later saving and interactive adjustment
            # Store both original (unfiltered) and current (filtered) masks
            # This allows the data management widget to save them when loading a new image
            # and allows real-time filtering via the min_size slider
            shapes_layer._masks_data = masks  # Current filtered masks
            shapes_layer._original_masks_data = original_masks  # Original unfiltered masks

            # Show nuclei segmentation layer and hide annotations layer
            self._on_toggle_view_clicked()
            
            show_info(f"Segmentation complete! Found {n_instances} nuclei.")
            
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            show_warning(error_msg)
            import traceback
            print(traceback.format_exc())

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

