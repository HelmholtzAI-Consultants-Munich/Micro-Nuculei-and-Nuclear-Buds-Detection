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
from qtpy.QtGui import QColor, QPixmap
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
        # folder_label = QLabel("Dataset Folder:")
        # folder_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        # self.layout().addWidget(folder_label)

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

        # Navigation and Save buttons layout
        buttons_layout = QHBoxLayout()
        
        # Next Unannotated button
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
        buttons_layout.addWidget(self.next_unannotated_button)
        
        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet(
            "QPushButton {"
            "background-color: #4CAF50;"
            "color: white;"
            "font-weight: bold;"
            "border: 2px solid #333;"
            "border-radius: 5px;"
            "padding: 8px 15px;"
            "min-height: 20px;"
            "min-width: 120px;"
            "}"
            "QPushButton:hover {"
            "background-color: #45a049;"
            "border: 2px solid white;"
            "}"
        )
        self.save_button.clicked.connect(self._save_current_image)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        self.layout().addLayout(buttons_layout)

    def _on_browse_folder(self):
        """Handle browse folder button click."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", ""
        )
        if folder:
            # Remove all existing dock widgets with these names
            widget_names = ["Micro-Nuclei Detection", "Nuclei Segmentation"]
            
            # Method 1: Try using remove_dock_widget directly
            for widget_name in widget_names:
                try:
                    dock = self.viewer.window._dock_widgets[widget_name]
                    self.viewer.window.remove_dock_widget(dock)
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
            # Update Save button state (disabled initially, no image loaded)
            if hasattr(self, 'save_button'):
                self.save_button.setEnabled(False)

            # Load Micro-Nuclei Detection widget
            self.detection_widget = DetectionWidget(self.viewer)
            # Set paths for saving/loading
            self.detection_widget.annotation_dir = self.annotation_dir
            self.detection_widget.dataset_path = self.dataset_path
            self.viewer.window.add_dock_widget(
                self.detection_widget,
                name="Micro-Nuclei Detection",
                area="right"
            )
            
            # Load Nuclei Segmentation widget
            self.segmentation_widget = NucleiSegmentationWidget(self.viewer, self.nuclei_segmentation_params_path)
            # Set paths for saving/loading
            self.segmentation_widget.nuclei_segmentation_dir = self.nuclei_segmentation_dir
            self.segmentation_widget.dataset_path = self.dataset_path
            self.viewer.window.add_dock_widget(
                self.segmentation_widget,
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
        if hasattr(self, 'detection_widget') and self.detection_widget:
            return self.detection_widget.has_annotations(image_path)
        return False

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
        
        # Use detection widget's save method
        if hasattr(self, 'detection_widget') and self.detection_widget:
            return self.detection_widget.save_annotations(current_image)
        return None

    def _save_current_nuclei_segmentation(self):
        """Save current nuclei segmentation masks to file."""
        if self.current_image_index < 0 or not self.image_files:
            return
        
        current_image = self.image_files[self.current_image_index]
        
        # Use segmentation widget's save method
        if hasattr(self, 'segmentation_widget') and self.segmentation_widget:
            return self.segmentation_widget.save_segmentation(current_image)
        return None

    def _save_current_image(self):
        """Save current annotations and nuclei segmentation, then run postprocessing.
        
        This unified method handles all saving operations for the current image.
        It saves annotations, nuclei segmentation, runs postprocessing, and updates the UI.
        
        Returns:
            bool: True if save was successful, False otherwise, None if no current image
        """
        # Check if there's a current image to save
        if self.current_image_index < 0 or not self.image_files:
            return None
        
        current_image = self.image_files[self.current_image_index]
        
        # Save annotations
        annotation_file = self._save_current_annotations()
        
        # Save nuclei segmentation
        nuclei_segmentation_file = self._save_current_nuclei_segmentation()
        
        # Run postprocessing
        postprocess_detections(self.dataset_path, current_image, annotation_file, nuclei_segmentation_file)
        
        # Update image list to refresh annotation status colors
        self._update_image_list()
        
        return True

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
                self._save_current_image()
            
            self.current_image_index = image_index
            self._load_image(image_path)
            self._update_image_list()  # Refresh to update highlighting
            # Update Save button state
            if hasattr(self, 'save_button'):
                self.save_button.setEnabled(self.current_image_index >= 0)
        # For folder items, expansion/collapse is handled automatically by QTreeWidget
    
    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on a tree item."""
        # Double-click on image loads it (same as single click)
        item_data = item.data(0, Qt.UserRole)
        if item_data is not None:
            self._on_tree_item_clicked(item, column)

    def _load_annotations(self, image_path: Path):
        """Load annotations from YOLO format (.txt) file if it exists."""
        if hasattr(self, 'detection_widget') and self.detection_widget:
            return self.detection_widget.load_annotations(image_path)
        return False

    def _has_nuclei_segmentation(self, image_path: Path) -> bool:
        """Check if nuclei segmentation exists for an image."""
        if hasattr(self, 'segmentation_widget') and self.segmentation_widget:
            return self.segmentation_widget.has_segmentation(image_path)
        return False
    
    def _load_nuclei_segmentation(self, image_path: Path):
        """Load nuclei segmentation masks from .npy file if it exists."""
        if hasattr(self, 'segmentation_widget') and self.segmentation_widget:
            return self.segmentation_widget.load_segmentation(image_path)
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
                # Use detection widget to create empty annotation layer
                if hasattr(self, 'detection_widget') and self.detection_widget:
                    self.detection_widget.create_empty_annotation_layer()
            
            # Load nuclei segmentation if it exists, otherwise run automatic segmentation
            nuclei_segmentation_loaded = self._load_nuclei_segmentation(image_path)
            segmentation_just_created = False
            
            if not nuclei_segmentation_loaded:
                # No saved segmentation exists, run automatic segmentation
                if hasattr(self, 'segmentation_widget') and self.segmentation_widget:
                    show_info("No saved segmentation found. Running automatic nuclei segmentation...")
                    # Run segmentation using segmentation widget (it handles image preprocessing)
                    masks, flows = self.segmentation_widget.run_segmentation(image_data)
                    if masks is not None:
                        # Create layer
                        self.segmentation_widget.create_segmentation_layer(masks, flows)
                        # Save the automatically generated segmentation
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
            self._save_current_image()

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
                # Update Save button state
                if hasattr(self, 'save_button'):
                    self.save_button.setEnabled(self.current_image_index >= 0)
                return
        
        # If we get here, all images are annotated
        show_info("All images have been annotated!")
        self.next_unannotated_button.setEnabled(False)

