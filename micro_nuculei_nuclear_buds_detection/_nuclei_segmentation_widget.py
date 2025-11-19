"""
Nuclei Segmentation widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

import numpy as np
import colorsys
import napari
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from qtpy.QtCore import Qt
from skimage import measure
from ._constants import NUCLEI_SEGMENTATION_LAYER_NAME, ANNOTATION_LAYER_NAME


class NucleiSegmentationWidget(QWidget):
    """Widget for nuclei segmentation."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Nuclei Segmentation")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        self.layout().addWidget(title)

        # Segment nuclei button
        self.segment_button = QPushButton("Segment Nuclei")
        self.segment_button.setStyleSheet(
            "QPushButton {"
            "background-color: #FF9800;"
            "color: white;"
            "font-weight: bold;"
            "border: none;"
            "border-radius: 5px;"
            "padding: 10px;"
            "margin-top: 15px;"
            "}"
            "QPushButton:hover {"
            "background-color: #F57C00;"
            "}"
        )
        self.segment_button.clicked.connect(self._on_segment_clicked)
        self.layout().addWidget(self.segment_button)

        # Show segmentation / Hide annotations button
        self.toggle_view_button = QPushButton("Show Nuceli Segmentation")
        self.toggle_view_button.setStyleSheet(
            "QPushButton {"
            "background-color: #00BCD4;"
            "color: white;"
            "font-weight: bold;"
            "border: none;"
            "border-radius: 5px;"
            "padding: 10px;"
            "margin-top: 15px;"
            "}"
            "QPushButton:hover {"
            "background-color: #0097A7;"
            "}"
        )
        self.toggle_view_button.clicked.connect(self._on_toggle_view_clicked)
        self.layout().addWidget(self.toggle_view_button)

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
        for i in range(n_instances):
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.7, 0.9)
            # Add alpha (partial transparency)
            rgba = [rgb[0], rgb[1], rgb[2], 0.5]  # 50% transparency
            colors.append(rgba)
        
        return np.array(colors)

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
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
            
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
            
            # Store masks in layer metadata for later saving
            # This allows the data management widget to save them when loading a new image
            shapes_layer._masks_data = masks

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

