"""
Data management widget for the Micro-Nuculei and Nuclear Buds Detection plugin.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import napari
from napari.utils.notifications import show_info, show_warning
from ._constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    get_napari_color,
    get_class_from_napari_color,
)
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
from skimage import io


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

        # Create layout
        self.setLayout(QVBoxLayout())

        # Add title
        title = QLabel("Data Management")
        title.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(title)

        # Setup UI
        self._setup_ui()
        
        # Don't add stretch - let the tree widget expand to fill space

    def _setup_ui(self):
        """Set up the user interface."""
        # Dataset folder selection
        folder_label = QLabel("Dataset Folder:")
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
        self.next_unannotated_button.clicked.connect(self._on_next_unannotated)
        self.next_unannotated_button.setEnabled(False)
        self.layout().addWidget(self.next_unannotated_button)

    def _on_browse_folder(self):
        """Handle browse folder button click."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", ""
        )
        if folder:
            self.dataset_path = Path(folder)
            self.folder_path_edit.setText(str(self.dataset_path))
            self.annotation_dir = self.dataset_path / "annotations"
            self.annotation_dir.mkdir(exist_ok=True)
            self._scan_images()
            self._update_image_list()

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
        
        for layer in self.viewer.layers:
            # Skip image layers
            if isinstance(layer, napari.layers.Image):
                continue
            
            # Handle shapes (rectangles, polygons, etc.)
            if isinstance(layer, napari.layers.Shapes):
                if layer.data:
                    # Get edge colors for all shapes
                    edge_colors = layer.edge_color
                    
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
                    for shape_idx, shape in enumerate(layer.data):
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
                        
                        # Get bounding box from shape vertices
                        x_coords = shape[:, 0]
                        y_coords = shape[:, 1]
                        
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
            
            # Handle points (convert to small bounding boxes)
            elif isinstance(layer, napari.layers.Points):
                if hasattr(layer, 'data') and layer.data.size > 0:
                    point_size = layer.size[0] if hasattr(layer.size, '__len__') else layer.size
                    # Use point size to create bounding box, or default to 10 pixels
                    box_size = float(point_size) if point_size > 0 else 10.0
                    half_size = box_size / 2.0
                    
                    for point in layer.data:
                        x, y = float(point[0]), float(point[1])
                        
                        # Create bounding box around point
                        x_min = x - half_size
                        x_max = x + half_size
                        y_min = y - half_size
                        y_max = y + half_size
                        
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
        try:
            relative_path = current_image.relative_to(self.dataset_path)
            # Create annotation path preserving subdirectory structure
            annotation_file = self.annotation_dir / relative_path.with_suffix('.txt')
            # Create parent directories if they don't exist
            annotation_file.parent.mkdir(parents=True, exist_ok=True)
        except ValueError:
            # Fallback: if path is not relative, use just the stem
            annotation_file = self.annotation_dir / f"{current_image.stem}.txt"
        
        with open(annotation_file, 'w') as f:
            f.writelines(yolo_annotations)
        
        # Show relative path in message
        try:
            display_path = str(current_image.relative_to(self.dataset_path))
        except ValueError:
            display_path = current_image.name
        show_info(f"Annotations saved in YOLO format for {display_path}")

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle click on a tree item (folder or image)."""
        # Check if this item has data (meaning it's an image, not a folder)
        item_data = item.data(0, Qt.UserRole)
        if item_data is not None:
            # This is an image item - load it
            image_path_str, image_index = item_data
            image_path = Path(image_path_str)
            
            # Save current annotations if we have a current image
            if self.current_image_index >= 0 and self.current_image_index != image_index:
                self._save_current_annotations()
            
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
                    name="annotations",
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
                    name="annotations",
                    face_color="transparent",
                )
                # Set color after creation (as array format with RGBA)
                # Napari expects RGBA format: (r, g, b, alpha)
                shapes_layer.edge_color = np.array([[default_color[0], default_color[1], default_color[2], 1.0]])
                # Set edge width to make it more visible
                shapes_layer.edge_width = 2
            
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

        # Save current annotations if we have a current image
        if self.current_image_index >= 0:
            self._save_current_annotations()

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

