"""
Data management widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import json
import napari
from napari.utils.notifications import show_info, show_warning
from ._constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    get_napari_color,
    get_class_from_napari_color,
    ANNOTATION_LAYER_NAME,
    NUCLEI_SEGMENTATION_LAYER_NAME,
    NUCLEI_SEGMENTATION_PARAMS_PATH,
    NUCLEI_SEGMENTATION_PARAMS_DEFAULT,
    ANNOTATIONS_SUBFOLDER,
    NUCLEI_SEGMENTATION_SUBFOLDER,
    POSTPROCESSING_SUBFOLDER,
)
from ._widget import DetectionWidget
from ._nuclei_segmentation_widget import NucleiSegmentationWidget
from ._postprocessing import postprocess_detections
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QFileDialog,
    QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from skimage import io, measure
import colorsys
import traceback

class DataManagementWidget(QWidget):
    """Widget for managing dataset and annotation workflow."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Data management state
        self.dataset_path: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_image_index: int = -1
        self.annotation_dir: Optional[Path] = None
        self.nuclei_segmentation_dir: Optional[Path] = None

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Data Management")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        self.layout().addWidget(title)

        # Setup UI
        self._setup_ui()
        
        # Don't add stretch - let the tree widget expand to fill space

    def _setup_ui(self):
        """Set up the user interface."""
        # Dataset folder selection
        folder_label = QLabel("Dataset Folder:")
        folder_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.layout().addWidget(folder_label)

        folder_layout = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Select dataset folder...")
        self.folder_path_edit.setReadOnly(True)
        folder_layout.addWidget(self.folder_path_edit)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_folder)
        folder_layout.addWidget(browse_button)
        self.layout().addLayout(folder_layout)

        # Image tree (hierarchical view)
        list_label = QLabel("Images:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.layout().addWidget(list_label)

        self.image_tree = QTreeWidget()
        self.image_tree.setHeaderLabel("Images")
        self.image_tree.setRootIsDecorated(True)  # Show expand/collapse icons
        self.image_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.image_tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        # Make tree expand to fill available vertical space
        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.image_tree.setSizePolicy(size_policy)
        self.layout().addWidget(self.image_tree)
        
        # We'll store path/index in the item's data instead of using a dict

        # Navigation button
        self.next_unannotated_button = QPushButton("Next Unannotated Image")
        self.next_unannotated_button.setStyleSheet(
            "QPushButton {"
            # "background-color: #00BCD4;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "}"
            "QPushButton:hover {"
            # "background-color: #0097A7;"
            "border: 2px solid white;"
            "}"
        )
        self.next_unannotated_button.clicked.connect(self._on_next_unannotated)
        self.next_unannotated_button.setEnabled(False)
        self.layout().addWidget(self.next_unannotated_button)

    def _on_browse_folder(self):
        """Handle browse folder button click."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", ""
        )
        if folder:
            # clear all widgets if they exist
            try:
                self.viewer.window.remove_dock_widget("Micro-Nuclei Detection")
            except Exception:
                pass
            try:
                self.viewer.window.remove_dock_widget("Nuclei Segmentation")
            except Exception:
                pass
            self.dataset_path = Path(folder)
            self.folder_path_edit.setText(str(self.dataset_path))
            self.annotation_dir = self.dataset_path / ANNOTATIONS_SUBFOLDER
            self.annotation_dir.mkdir(exist_ok=True)
            self.nuclei_segmentation_dir = self.dataset_path / NUCLEI_SEGMENTATION_SUBFOLDER
            self.nuclei_segmentation_dir.mkdir(exist_ok=True)
            self.postprocessing_dir = self.dataset_path / POSTPROCESSING_SUBFOLDER
            self.postprocessing_dir.mkdir(exist_ok=True)
            self.nuclei_segmentation_params_path = self.nuclei_segmentation_dir / NUCLEI_SEGMENTATION_PARAMS_PATH
            if not self.nuclei_segmentation_params_path.exists():
                with open(self.nuclei_segmentation_params_path, 'w') as f:
                    json.dump(NUCLEI_SEGMENTATION_PARAMS_DEFAULT, f)
            self._scan_images()
            self._update_image_list()
            # Load Micro-Nuclei Detection widget
            detection_widget = DetectionWidget(self.viewer)
            self.viewer.window.add_dock_widget(
                detection_widget,
                name="Micro-Nuclei Detection",
                area="right"
            )
            
            # Load Nuclei Segmentation widget
            segmentation_widget = NucleiSegmentationWidget(self.viewer, self.nuclei_segmentation_params_path)
            self.viewer.window.add_dock_widget(
                segmentation_widget,
                name="Nuclei Segmentation",
                area="right"
            )

    def _scan_images(self):
        """Scan the dataset folder recursively for .tif and .tiff images."""
        if not self.dataset_path or not self.dataset_path.exists():
            return

        self.image_files = []
        extensions = {".tif", ".tiff", ".TIF", ".TIFF"}
        
        # Recursively scan for images in all subdirectories
        for ext in extensions:
            # Use rglob to search recursively
            self.image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        self.image_files = [Path(image_file) for image_file in self.image_files]
        
        # Sort for consistent ordering (by full path)
        self.image_files.sort()
        
        if self.image_files:
            show_info(f"Found {len(self.image_files)} image(s) in dataset folder (including subdirectories).")
        else:
            show_warning("No .tif or .tiff images found in the selected folder.")

    def _is_annotated(self, image_path: Path) -> bool:
        """Check if an image has been annotated."""
        if not self.annotation_dir:
            return False
        
        # Get relative path from dataset folder to preserve subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            # Create annotation path preserving subdirectory structure
            annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
        except ValueError:
            # If image_path is not relative to dataset_path, use just the stem
            annotation_file = self.annotation_dir / f"{image_path.stem}.txt"
        
        return annotation_file.exists()

    def _update_image_list(self):
        """Update the image tree widget with hierarchical folder structure."""
        self.image_tree.clear()
        
        if not self.image_files:
            self.next_unannotated_button.setEnabled(False)
            return

        # Build a tree structure from image paths
        root_item = self.image_tree.invisibleRootItem()
        folder_items = {}  # Map folder paths to tree items
        
        for i, image_path in enumerate(self.image_files):
            # Get relative path from dataset folder
            try:
                relative_path = image_path.relative_to(self.dataset_path)
            except ValueError:
                relative_path = Path(image_path.name)
            
            # Get path parts (e.g., ["images_1", "image.tif"] or ["folder", "subfolder", "image.tif"])
            parts = list(relative_path.parts)
            
            # Build folder structure
            current_parent = root_item
            current_path = Path(self.dataset_path)
            
            # Create folder items for each directory in the path
            for part_idx, part in enumerate(parts[:-1]):  # All parts except the filename
                current_path = current_path / part
                folder_key = str(current_path)
                
                if folder_key not in folder_items:
                    # Create new folder item
                    folder_item = QTreeWidgetItem(current_parent)
                    folder_item.setText(0, part)
                    folder_item.setExpanded(True)  # Expand by default
                    folder_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                    # Set folder icon if available
                    try:
                        from qtpy.QtGui import QIcon
                        from qtpy.QtWidgets import QFileIconProvider
                        icon_provider = QFileIconProvider()
                        folder_item.setIcon(0, icon_provider.icon(QFileIconProvider.Folder))
                    except Exception:
                        pass
                    folder_items[folder_key] = folder_item
                    current_parent = folder_item
                else:
                    current_parent = folder_items[folder_key]
            
            # Create image item (leaf node)
            filename = parts[-1]
            image_item = QTreeWidgetItem(current_parent)
            image_item.setText(0, filename)
            # Set file icon if available
            try:
                from qtpy.QtGui import QIcon
                from qtpy.QtWidgets import QFileIconProvider
                icon_provider = QFileIconProvider()
                image_item.setIcon(0, icon_provider.icon(QFileIconProvider.File))
            except Exception:
                pass
            
            # Store image path and index in the item's data (using Qt.UserRole)
            # This allows us to retrieve it later without needing a hashable key
            image_item.setData(0, Qt.UserRole, (str(image_path), i))
            
            # Color code: green for annotated, red for not annotated
            if self._is_annotated(image_path):
                image_item.setForeground(0, QColor(0, 150, 0))  # Green
                image_item.setToolTip(0, f"{relative_path} - Annotated\nFull path: {image_path}")
            else:
                image_item.setForeground(0, QColor(200, 0, 0))  # Red
                image_item.setToolTip(0, f"{relative_path} - Not annotated\nFull path: {image_path}")
            
            # Highlight current image
            if i == self.current_image_index:
                image_item.setBackground(0, QColor(200, 200, 255))  # Light blue
        
        # Expand all folders by default
        self.image_tree.expandAll()
        
        # Enable next button if there are unannotated images
        has_unannotated = any(not self._is_annotated(img) for img in self.image_files)
        self.next_unannotated_button.setEnabled(has_unannotated)

    def _save_current_annotations(self):
        """Save current annotations to file in YOLO format."""
        if self.current_image_index < 0 or not self.image_files:
            return
        
        current_image = self.image_files[self.current_image_index]
        if not self.annotation_dir:
            return

        # Get image dimensions from the first image layer
        image_width = None
        image_height = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                # Get image dimensions (handle multi-dimensional images)
                if layer.data.ndim >= 2:
                    # For 2D: [height, width]
                    # For 3D: [depth, height, width] or [height, width, channels]
                    if layer.data.ndim == 2:
                        image_height, image_width = layer.data.shape
                    elif layer.data.ndim == 3:
                        # Assume last two dimensions are height and width
                        image_height, image_width = layer.data.shape[-2:]
                    break
        
        if image_width is None or image_height is None:
            show_warning("Could not determine image dimensions. Annotations not saved.")
            return

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
            return
        
        # Process only the annotation layer
        if annotation_layer.data:
            # Get edge colors for all shapes
            edge_colors = annotation_layer.edge_color
            
            # Handle different color formats
            if isinstance(edge_colors, np.ndarray):
                if edge_colors.ndim == 1:
                    # Single color for all shapes, convert to 2D
                    edge_colors = np.array([edge_colors])
                elif edge_colors.ndim == 2:
                    # Already in correct format (n_shapes, n_colors)
                    pass
            else:
                # Convert to array
                edge_colors = np.array([edge_colors])
            
            # Process each shape
            for shape_idx, shape in enumerate(annotation_layer.data):
                # Convert shape to bounding box
                # Shape is a numpy array of [x, y] coordinates
                if len(shape) == 0:
                    continue
                
                # Determine class ID from this shape's color
                class_id = 0  # Default
                
                # Get the color for this specific shape
                if shape_idx < len(edge_colors):
                    shape_color = edge_colors[shape_idx]
                    # Extract RGB (first 3 values, ignore alpha if present)
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
        # Preserve subdirectory structure in annotations folder
        relative_path = current_image.relative_to(self.dataset_path)
        # Create annotation path preserving subdirectory structure
        annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
        # Create parent directories if they don't exist
        annotation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_file, 'w') as f:
            f.writelines(yolo_annotations)
        
        # Show relative path in message
        display_path = str(current_image.relative_to(self.dataset_path))

        show_info(f"Annotations saved in YOLO format for {display_path}")
        return annotation_file

    def _save_current_nuclei_segmentation(self):
        """Save current nuclei segmentation masks to file."""
        if self.current_image_index < 0 or not self.image_files:
            return
        
        current_image = self.image_files[self.current_image_index]
        if not self.nuclei_segmentation_dir:
            return

        # Find the nuclei segmentation layer
        nuclei_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                nuclei_layer = layer
                break
        
        if nuclei_layer is None or not hasattr(nuclei_layer, '_masks_data'):
            # No nuclei segmentation to save
            return
        
        masks = nuclei_layer._masks_data
        
        # Save masks as .npy file
        # Preserve subdirectory structure in nuclei-segmentation folder
        try:
            relative_path = current_image.relative_to(self.dataset_path)
            # Create segmentation path preserving subdirectory structure
            segmentation_file = self.nuclei_segmentation_dir / relative_path.with_suffix('.npy')
            # Create parent directories if they don't exist
            segmentation_file.parent.mkdir(parents=True, exist_ok=True)
        except ValueError:
            # Fallback: if path is not relative, use just the stem
            segmentation_file = self.nuclei_segmentation_dir / f"{current_image.stem}.npy"
        
        # Save masks as numpy array
        np.save(segmentation_file, masks)
        
        # Show relative path in message
        try:
            display_path = str(current_image.relative_to(self.dataset_path))
        except ValueError:
            display_path = current_image.name
        show_info(f"Nuclei segmentation saved for {display_path}")
        return segmentation_file

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle click on a tree item (folder or image)."""
        # Check if this item has data (meaning it's an image, not a folder)
        item_data = item.data(0, Qt.UserRole)
        if item_data is not None:
            # This is an image item - load it
            image_path_str, image_index = item_data
            image_path = Path(image_path_str)
            
            # Save current annotations and nuclei segmentation if we have a current image
            if self.current_image_index >= 0 and self.current_image_index != image_index:
                annotation_file = self._save_current_annotations()
                nuclei_segmentation_file = self._save_current_nuclei_segmentation()
                postprocess_detections(self.dataset_path, self.image_files[self.current_image_index], annotation_file, nuclei_segmentation_file)
            
            self.current_image_index = image_index
            self._load_image(image_path)
            self._update_image_list()  # Refresh to update highlighting
        # For folder items, expansion/collapse is handled automatically by QTreeWidget
    
    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on a tree item."""
        # Double-click on image loads it (same as single click)
        item_data = item.data(0, Qt.UserRole)
        if item_data is not None:
            self._on_tree_item_clicked(item, column)

    def _load_annotations(self, image_path: Path):
        """Load annotations from YOLO format (.txt) file if it exists."""
        if not self.annotation_dir:
            show_warning("Annotation directory not set.")
            return False
        
        # Get annotation file path preserving subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            # Create annotation path preserving subdirectory structure
            annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
        except ValueError:
            # Fallback: if path is not relative, use just the stem
            annotation_file = self.annotation_dir / f"{image_path.stem}.txt"
        
        if not annotation_file.exists():
            show_warning(f"Annotation file not found: {annotation_file}")
            return False
        
        try:
            # Get image dimensions from the first image layer
            image_width = None
            image_height = None
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    # Get image dimensions (handle multi-dimensional images)
                    if layer.data.ndim >= 2:
                        if layer.data.ndim == 2:
                            image_height, image_width = layer.data.shape
                        elif layer.data.ndim == 3:
                            # Assume last two dimensions are height and width
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
                        # YOLO format: class_id center_y center_x width height, keeping in mind that napari swaps x and y coordinates
                        class_id = int(parts[0])
                        center_x_norm = float(parts[2])
                        center_y_norm = float(parts[1])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])
                        
                        # Convert normalized coordinates back to pixel coordinates
                        center_x = center_x_norm * image_height
                        center_y = center_y_norm * image_width
                        width = width_norm * image_width
                        height = height_norm * image_height
                        
                        # Calculate bounding box corners
                        x_min = center_x - height / 2.0
                        x_max = center_x + height / 2.0
                        y_min = center_y - width / 2.0
                        y_max = center_y + width / 2.0
                        
                        # Create rectangle shape (4 corners)
                        rectangle = np.array([
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max]
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
            except ValueError:
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
            except ValueError:
                display_path = image_path.name
            error_msg = f"Error loading annotations for {display_path}: {str(e)}\n{traceback.format_exc()}"
            show_warning(error_msg)
            return False

    def _has_nuclei_segmentation(self, image_path: Path) -> bool:
        """Check if nuclei segmentation exists for an image."""
        if not self.nuclei_segmentation_dir:
            return False
        
        # Get segmentation file path preserving subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            # Create segmentation path preserving subdirectory structure
            segmentation_file = self.nuclei_segmentation_dir / relative_path.with_suffix('.npy')
        except ValueError:
            # Fallback: if path is not relative, use just the stem
            segmentation_file = self.nuclei_segmentation_dir / f"{image_path.stem}.npy"
        
        return segmentation_file.exists()

    def _load_nuclei_segmentation(self, image_path: Path):
        """Load nuclei segmentation masks from .npy file if it exists."""
        if not self._has_nuclei_segmentation(image_path):
            return False
        
        # Get segmentation file path preserving subdirectory structure
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            # Create segmentation path preserving subdirectory structure
            segmentation_file = self.nuclei_segmentation_dir / relative_path.with_suffix('.npy')
        except ValueError:
            # Fallback: if path is not relative, use just the stem
            segmentation_file = self.nuclei_segmentation_dir / f"{image_path.stem}.npy"
        
        try:
            # Load masks
            masks = np.load(segmentation_file)
            
            # Convert masks to shapes (polygons)
            shapes_data = []
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
                    shapes_data.append(polygon)
            
            if len(shapes_data) == 0:
                return False
            
            # Generate colors for each instance
            n_instances = len(shapes_data)
            colors = []
            for i in range(n_instances):
                hue = (i * 137.508) % 360  # Golden angle for good distribution
                # Convert HSV to RGB
                rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.7, 0.9)
                # Add alpha (partial transparency)
                rgba = [rgb[0], rgb[1], rgb[2], 0.5]  # 50% transparency
                colors.append(rgba)
            
            color_array = np.array(colors)
            
            # Create shapes layer
            shapes_layer = self.viewer.add_shapes(
                shapes_data,
                name=NUCLEI_SEGMENTATION_LAYER_NAME,
                shape_type='polygon',
                edge_color='white',
                edge_width=1,
                face_color=color_array,
                opacity=0.5,
            )
            
            # Store masks in layer metadata
            shapes_layer._masks_data = masks
            
            # Get display path for messages
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except ValueError:
                display_path = image_path.name
            
            show_info(f"Loaded nuclei segmentation for {display_path}")
            return True
            
        except Exception as e:
            import traceback
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except ValueError:
                display_path = image_path.name
            error_msg = f"Error loading nuclei segmentation for {display_path}: {str(e)}\n{traceback.format_exc()}"
            show_warning(error_msg)
            return False

    def _get_segmentation_parameters(self):
        """Get segmentation parameters from nuclei segmentation widget if available, otherwise use defaults."""
        # Try to find the nuclei segmentation widget in the viewer's dock widgets
        min_size = 0
        
        try:
            # Try to find the widget in napari's dock widgets
            # Access dock widgets through the window
            if hasattr(self.viewer.window, '_dock_widgets'):
                for dock_widget in self.viewer.window._dock_widgets.values():
                    widget = dock_widget.widget()
                    if hasattr(widget, 'min_size'):
                        min_size = widget.min_size
                        break
            # Alternative: try accessing through window.qt_viewer
            elif hasattr(self.viewer.window, 'qt_viewer'):
                # Try to find widget in the dock area
                for widget in self.viewer.window.qt_viewer.findChildren(QWidget):
                    if hasattr(widget, 'min_size'):
                        min_size = widget.min_size
                        break
        except Exception:
            # If we can't find the widget, use defaults
            pass
        
        return min_size

    def _run_nuclei_segmentation(self, image_data):
        """Run nuclei segmentation on image data and return masks."""
        try:
            show_info("Starting automatic nuclei segmentation with Cellpose...")
            
            # Get parameters from segmentation widget or use defaults
            min_size = self._get_segmentation_parameters()
            
            # Import cellpose
            try:
                from cellpose import models, core
            except ImportError:
                show_warning("Cellpose is not installed. Please install it with: pip install cellpose==3.1.1.1")
                return None
            
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
            
            # Apply min_size filter as post-processing
            if min_size > 0:
                try:
                    from cellpose import utils
                    # Try to use cellpose's remove_small_masks if available
                    if hasattr(utils, 'remove_small_masks'):
                        masks = utils.remove_small_masks(masks, min_size=min_size)
                    else:
                        # Use manual filtering if function doesn't exist
                        masks = self._manual_remove_small_masks(masks, min_size)
                except (ImportError, AttributeError):
                    # Fallback: manual filtering if cellpose utils not available or function doesn't exist
                    masks = self._manual_remove_small_masks(masks, min_size)
            
            return masks
            
        except Exception as e:
            error_msg = f"Error during automatic segmentation: {str(e)}\n{traceback.format_exc()}"
            show_warning(error_msg)
            return None

    def _manual_remove_small_masks(self, masks, min_size):
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

    def _create_nuclei_segmentation_layer(self, masks):
        """Create a nuclei segmentation layer from masks."""
        if masks is None:
            return None
        
        # Convert masks to shapes (polygons)
        shapes_data = []
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
                shapes_data.append(polygon)
        
        if len(shapes_data) == 0:
            show_warning("No nuclei were segmented.")
            return None
        
        # Generate colors for each instance
        n_instances = len(shapes_data)
        colors = []
        for i in range(n_instances):
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.7, 0.9)
            # Add alpha (partial transparency)
            rgba = [rgb[0], rgb[1], rgb[2], 0.5]  # 50% transparency
            colors.append(rgba)
        
        color_array = np.array(colors)
        
        # Create shapes layer
        shapes_layer = self.viewer.add_shapes(
            shapes_data,
            name=NUCLEI_SEGMENTATION_LAYER_NAME,
            shape_type='polygon',
            edge_color='white',
            edge_width=1,
            face_color=color_array,
            opacity=0.5,
        )
        
        # Store masks in layer metadata
        shapes_layer._masks_data = masks
        
        show_info(f"Segmentation complete! Found {n_instances} nuclei.")
        return shapes_layer

    def _load_image(self, image_path: Path):
        """Load an image into napari viewer."""
        try:
            # Clear all existing layers (images and annotations from previous image)
            layers_to_remove = list(self.viewer.layers)
            for layer in layers_to_remove:
                self.viewer.layers.remove(layer)
            
            # Load new image
            image_data = io.imread(str(image_path))
            
            # Handle multi-channel images
            if image_data.ndim == 3 and image_data.shape[2] <= 4:
                # Assume it's a multi-channel image
                self.viewer.add_image(image_data, name=image_path.stem)
            else:
                self.viewer.add_image(image_data, name=image_path.stem)
            
            # Check if annotations exist and load them, otherwise create empty shapes layer
            annotations_loaded = False
            if self._is_annotated(image_path):
                annotations_loaded = self._load_annotations(image_path)
            
            # If annotations weren't loaded (either file doesn't exist or loading failed), create empty shapes layer
            if not annotations_loaded:
                # Create an empty shapes layer for annotations
                # Use default class (class 0 - micro-nuclei) color for initial drawing
                default_class_id = 0
                default_color = get_napari_color(default_class_id)
                # Create layer with empty data first, then set color
                shapes_layer = self.viewer.add_shapes(
                    data=[],
                    name=ANNOTATION_LAYER_NAME,
                    face_color="transparent",
                )
                # Set color after creation (as array format with RGBA)
                # Napari expects RGBA format: (r, g, b, alpha)
                shapes_layer.edge_color = np.array([[default_color[0], default_color[1], default_color[2], 1.0]])
                # Set edge width to make it more visible
                shapes_layer.edge_width = 2
            
            # Load nuclei segmentation if it exists, otherwise run automatic segmentation
            nuclei_segmentation_loaded = self._load_nuclei_segmentation(image_path)
            segmentation_just_created = False
            
            if not nuclei_segmentation_loaded:
                # No saved segmentation exists, run automatic segmentation
                show_info("No saved segmentation found. Running automatic nuclei segmentation...")
                masks = self._run_nuclei_segmentation(image_data)
                if masks is not None:
                    self._create_nuclei_segmentation_layer(masks)
                    # Save the automatically generated segmentation
                    self._save_current_nuclei_segmentation()
                    segmentation_just_created = True
            
            # If annotations were loaded, hide nuclei segmentation and show annotations
            if annotations_loaded:
                # Hide nuclei segmentation layer
                for layer in self.viewer.layers:
                    if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                        layer.visible = False
                        break
                
                # Ensure annotation layer is visible
                for layer in self.viewer.layers:
                    if isinstance(layer, napari.layers.Shapes) and layer.name == ANNOTATION_LAYER_NAME:
                        layer.visible = True
                        break
            elif segmentation_just_created:
                # If no annotations but segmentation was just created, show segmentation and hide annotations
                for layer in self.viewer.layers:
                    if isinstance(layer, napari.layers.Shapes) and layer.name == NUCLEI_SEGMENTATION_LAYER_NAME:
                        layer.visible = True
                    elif isinstance(layer, napari.layers.Shapes) and layer.name == ANNOTATION_LAYER_NAME:
                        layer.visible = False
            
            # Get display path for message
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except ValueError:
                display_path = image_path.name
            
            show_info(f"Loaded: {display_path}")
            
        except Exception as e:
            try:
                display_path = str(image_path.relative_to(self.dataset_path))
            except ValueError:
                display_path = image_path.name
            show_warning(f"Error loading image {display_path}: {str(e)}")

    def _on_next_unannotated(self):
        """Move to the next unannotated image."""
        if not self.image_files:
            show_warning("No images in dataset.")
            return

        # Save current annotations and nuclei segmentation if we have a current image
        if self.current_image_index >= 0:
            annotation_file = self._save_current_annotations()
            nuclei_segmentation_file = self._save_current_nuclei_segmentation()
            postprocess_detections(self.dataset_path, self.image_files[self.current_image_index], annotation_file, nuclei_segmentation_file)

        # Find next unannotated image
        # Start from current + 1, or from beginning if no current image
        start_index = (self.current_image_index + 1) if self.current_image_index >= 0 else 0
        
        # Search forward from start_index
        for i in range(len(self.image_files)):
            idx = (start_index + i) % len(self.image_files)
            if not self._is_annotated(self.image_files[idx]):
                self.current_image_index = idx
                self._load_image(self.image_files[idx])
                self._update_image_list()
                return
        
        # If we get here, all images are annotated
        show_info("All images have been annotated!")
        self.next_unannotated_button.setEnabled(False)

