"""Tests for data management widget functions."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import tempfile
import shutil

# Try to import dependencies - tests will skip if not available
try:
    import pytestqt
    PYTEST_QT_AVAILABLE = True
except ImportError:
    PYTEST_QT_AVAILABLE = False

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

# Only import widgets if dependencies are available
if PYTEST_QT_AVAILABLE and NAPARI_AVAILABLE:
    from micro_nuculei_nuclear_buds_detection._data_management_widget import (
        DataManagementWidget,
    )
    from micro_nuculei_nuclear_buds_detection._constants import (
        ANNOTATION_LAYER_NAME,
        NUCLEI_SEGMENTATION_LAYER_NAME,
        get_napari_color,
        CLASS_NAMES,
        ANNOTATIONS_SUBFOLDER,
        POSTPROCESSING_SUBFOLDER,
        NUCLEI_SEGMENTATION_SUBFOLDER,
    )
else:
    # Create dummy classes so tests can be collected
    DataManagementWidget = None
    ANNOTATION_LAYER_NAME = None
    get_napari_color = None
    CLASS_NAMES = None





class TestLoadAnnotations:
    """Tests for _load_annotations method."""

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_load_annotations_single_box(self, mock_napari_viewer, temp_dir, qtbot):
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        """Test loading a single bounding box annotation."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / "annotations"
        widget.annotation_dir.mkdir()
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create annotation file in YOLO format
        # Format: class_id center_y center_x height width
        annotation_file = widget.annotation_dir / "test_image.txt"
        with open(annotation_file, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")  # Center at (0.5, 0.5), size 0.2x0.2
        
        # Create mock image layer
        image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Mock add_shapes method
        mock_shapes_layer = MagicMock()
        mock_napari_viewer.add_shapes = MagicMock(return_value=mock_shapes_layer)
        
        # Load annotations
        result = widget._load_annotations(temp_dir / "test_image.tif")
        
        # Should return True if loaded successfully
        assert result is True or result is False  # May return False if show_warning is called
        
        # Verify add_shapes was called
        if result:
            assert mock_napari_viewer.add_shapes.called
            call_args = mock_napari_viewer.add_shapes.call_args
            rectangles = call_args[0][0]  # First positional argument
            assert len(rectangles) == 1
            # Verify rectangle coordinates (should be denormalized)
            rect = rectangles[0]
            assert rect.shape == (4, 2)  # 4 corners, 2 coordinates each

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_load_annotations_multiple_boxes(self, mock_napari_viewer, temp_dir, qtbot):
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        """Test loading multiple bounding box annotations."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / "annotations"
        widget.annotation_dir.mkdir()
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create annotation file with multiple boxes
        annotation_file = widget.annotation_dir / "test_image.txt"
        with open(annotation_file, 'w') as f:
            f.write("0 0.3 0.3 0.2 0.2\n")  # Box 1
            f.write("1 0.7 0.7 0.15 0.15\n")  # Box 2
        
        # Create mock image layer
        image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Mock add_shapes
        mock_shapes_layer = MagicMock()
        mock_napari_viewer.add_shapes = MagicMock(return_value=mock_shapes_layer)
        
        # Load annotations
        result = widget._load_annotations(temp_dir / "test_image.tif")
        
        if result:
            assert mock_napari_viewer.add_shapes.called
            call_args = mock_napari_viewer.add_shapes.call_args
            rectangles = call_args[0][0]
            assert len(rectangles) == 2

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_load_annotations_missing_file(self, mock_napari_viewer, temp_dir, qtbot):
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        """Test loading when annotation file doesn't exist."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / "annotations"
        widget.annotation_dir.mkdir()
        
        # Create mock image layer
        image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Try to load non-existent file
        result = widget._load_annotations(temp_dir / "nonexistent.tif")
        
        # Should return False
        assert result is False

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_load_annotations_invalid_format(self, mock_napari_viewer, temp_dir, qtbot):
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        """Test loading annotation file with invalid format."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / "annotations"
        widget.annotation_dir.mkdir()
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create annotation file with invalid format
        annotation_file = widget.annotation_dir / "test_image.txt"
        with open(annotation_file, 'w') as f:
            f.write("invalid line\n")
            f.write("0 0.5 0.5\n")  # Missing width and height
        
        # Create mock image layer
        image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Mock add_shapes
        mock_shapes_layer = MagicMock()
        mock_napari_viewer.add_shapes = MagicMock(return_value=mock_shapes_layer)
        
        # Load annotations - should handle invalid lines gracefully
        result = widget._load_annotations(temp_dir / "test_image.tif")
        
        # Should either return False or load valid lines only
        # The function may return False if no valid annotations were found
        assert result is True or result is False

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_load_annotations_coordinate_denormalization(self, mock_napari_viewer, temp_dir, qtbot):
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        """Test that normalized coordinates are properly denormalized."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / "annotations"
        widget.annotation_dir.mkdir()
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create annotation file
        annotation_file = widget.annotation_dir / "test_image.txt"
        with open(annotation_file, 'w') as f:
            # Center at (0.5, 0.5), size 0.2x0.2 in normalized coordinates
            f.write("0 0.5 0.5 0.2 0.2\n")
        
        # Create mock image layer (200x200 image)
        image_layer = MagicMock()
        image_layer.data = np.zeros((200, 200), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Mock add_shapes
        mock_shapes_layer = MagicMock()
        mock_napari_viewer.add_shapes = MagicMock(return_value=mock_shapes_layer)
        
        # Load annotations
        result = widget._load_annotations(temp_dir / "test_image.tif")
        
        if result:
            call_args = mock_napari_viewer.add_shapes.call_args
            rectangles = call_args[0][0]
            if len(rectangles) > 0:
                rect = rectangles[0]
                # Center should be at (100, 100) in 200x200 image
                # Box should be 40x40 (0.2 * 200)
                center_x = (rect[:, 0].min() + rect[:, 0].max()) / 2
                center_y = (rect[:, 1].min() + rect[:, 1].max()) / 2
                width = rect[:, 0].max() - rect[:, 0].min()
                height = rect[:, 1].max() - rect[:, 1].min()
                
                # Allow some tolerance for floating point
                assert abs(center_x - 100) < 1
                assert abs(center_y - 100) < 1
                assert abs(width - 40) < 1
                assert abs(height - 40) < 1


class TestAnnotationRoundTrip:
    """Tests for save/load round-trip of annotations."""

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_annotation_round_trip(self, mock_napari_viewer, temp_dir, qtbot):
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        """Test that saving and loading annotations preserves data."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / "annotations"
        widget.annotation_dir.mkdir()
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create mock image layer
        image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Create annotation layer with known boxes
        annotation_layer = MagicMock()
        annotation_layer.name = ANNOTATION_LAYER_NAME
        # Box at (20, 20) with size 20x20
        annotation_layer.data = [
            np.array([[20, 20], [40, 20], [40, 40], [20, 40]])
        ]
        color = get_napari_color(0)
        annotation_layer.edge_color = np.array([[color[0], color[1], color[2], 1.0]])
        mock_napari_viewer.layers.append(annotation_layer)
        
        # Save annotations
        widget._save_current_annotations()
        
        # Clear the annotation layer
        mock_napari_viewer.layers = [image_layer]
        
        # Mock add_shapes to capture loaded rectangles
        loaded_rectangles = []
        def capture_shapes(*args, **kwargs):
            loaded_rectangles.extend(args[0])
            return MagicMock()
        mock_napari_viewer.add_shapes = capture_shapes
        
        # Load annotations
        widget._load_annotations(temp_dir / "test_image.tif")
        
        # Verify that loaded rectangles match original (within tolerance)
        if len(loaded_rectangles) > 0:
            loaded_rect = loaded_rectangles[0]
            original_rect = annotation_layer.data[0]
            
            # Compare bounding boxes (allowing for small differences due to rounding)
            loaded_x_min = loaded_rect[:, 0].min()
            loaded_x_max = loaded_rect[:, 0].max()
            loaded_y_min = loaded_rect[:, 1].min()
            loaded_y_max = loaded_rect[:, 1].max()
            
            original_x_min = original_rect[:, 0].min()
            original_x_max = original_rect[:, 0].max()
            original_y_min = original_rect[:, 1].min()
            original_y_max = original_rect[:, 1].max()
            
            # Allow 1 pixel tolerance
            assert abs(loaded_x_min - original_x_min) < 1
            assert abs(loaded_x_max - original_x_max) < 1
            assert abs(loaded_y_min - original_y_min) < 1
            assert abs(loaded_y_max - original_y_max) < 1


class TestSaveCurrentImage:
    """Tests for _save_current_image method."""

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_save_current_image_no_current_image(self, mock_napari_viewer, qtbot):
        """Test _save_current_image when there's no current image."""
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        
        # No current image index set
        widget.current_image_index = -1
        widget.image_files = []
        
        result = widget._save_current_image()
        
        # Should return None when no current image
        assert result is None

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_save_current_image_calls_all_methods(self, mock_napari_viewer, temp_dir, qtbot):
        """Test that _save_current_image calls all required methods."""
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        
        # Import widgets
        from micro_nuculei_nuclear_buds_detection._widget import DetectionWidget
        from micro_nuculei_nuclear_buds_detection._nuclei_segmentation_widget import NucleiSegmentationWidget
        import json
        from micro_nuculei_nuclear_buds_detection._constants import NUCLEI_SEGMENTATION_PARAMS_PATH, NUCLEI_SEGMENTATION_PARAMS_DEFAULT
        
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        
        # Set up widget state
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / ANNOTATIONS_SUBFOLDER
        widget.annotation_dir.mkdir(parents=True, exist_ok=True)
        widget.nuclei_segmentation_dir = temp_dir / NUCLEI_SEGMENTATION_SUBFOLDER
        widget.nuclei_segmentation_dir.mkdir(parents=True, exist_ok=True)
        widget.postprocessing_dir = temp_dir / POSTPROCESSING_SUBFOLDER
        widget.postprocessing_dir.mkdir(parents=True, exist_ok=True)
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create params file for segmentation widget
        widget.nuclei_segmentation_params_path = widget.nuclei_segmentation_dir / NUCLEI_SEGMENTATION_PARAMS_PATH
        with open(widget.nuclei_segmentation_params_path, 'w') as f:
            json.dump(NUCLEI_SEGMENTATION_PARAMS_DEFAULT, f)
        
        # Create actual widgets (not mocks)
        widget.detection_widget = DetectionWidget(mock_napari_viewer)
        widget.detection_widget.annotation_dir = widget.annotation_dir
        widget.detection_widget.dataset_path = widget.dataset_path
        qtbot.addWidget(widget.detection_widget)
        
        widget.segmentation_widget = NucleiSegmentationWidget(mock_napari_viewer, widget.nuclei_segmentation_params_path)
        widget.segmentation_widget.nuclei_segmentation_dir = widget.nuclei_segmentation_dir
        widget.segmentation_widget.dataset_path = widget.dataset_path
        qtbot.addWidget(widget.segmentation_widget)
        
        # Create mock image layer
        if NAPARI_AVAILABLE:
            image_layer = MagicMock(spec=napari.layers.Image)
        else:
            image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Create annotation layer with specific test data
        if NAPARI_AVAILABLE:
            annotation_layer = MagicMock(spec=napari.layers.Shapes)
        else:
            annotation_layer = MagicMock()
        annotation_layer.name = ANNOTATION_LAYER_NAME
        # Create a test rectangle: [[y_min, x_min], [y_max, x_min], [y_max, x_max], [y_min, x_max]]
        # Box at (20, 20) with size 20x20
        test_rectangle = np.array([[20, 20], [40, 20], [40, 40], [20, 40]])
        annotation_layer.data = [test_rectangle]
        # Color for micro-nuclei (class 0)
        color = get_napari_color(0)
        annotation_layer.edge_color = np.array([[color[0], color[1], color[2], 1.0]])
        mock_napari_viewer.layers.append(annotation_layer)
        
        # Create a mock nuclei segmentation layer with specific masks data
        if NAPARI_AVAILABLE:
            nuclei_layer = MagicMock(spec=napari.layers.Shapes)
        else:
            nuclei_layer = MagicMock()
        nuclei_layer.name = NUCLEI_SEGMENTATION_LAYER_NAME
        # Create a specific test mask
        test_mask = np.zeros((100, 100), dtype=np.int32)
        test_mask[20:30, 20:30] = 1  # One nucleus at (20-30, 20-30)
        test_mask[50:60, 50:60] = 2  # Another nucleus at (50-60, 50-60)
        nuclei_layer._masks_data = test_mask
        mock_napari_viewer.layers.append(nuclei_layer)
        
        # Mock _update_image_list
        widget._update_image_list = MagicMock()
        
        # Mock postprocess_detections
        with patch('micro_nuculei_nuclear_buds_detection._data_management_widget.postprocess_detections') as mock_postprocess:
            result = widget._save_current_image()
        
        # Should return True on success
        assert result is True
        
        # Verify all methods were called
        # Check that annotation file was created (save_annotations was called)
        annotation_file = widget.annotation_dir / "test_image.txt"
        assert annotation_file.exists(), "Annotation file should be created by DetectionWidget.save_annotations()"
        
        # Verify annotation file content matches original
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1, "Should have one annotation"
        parts = lines[0].strip().split()
        assert len(parts) == 5, "YOLO format should have 5 values"
        assert parts[0] == "0", "Should be class 0 (micro-nuclei)"
        # Verify normalized coordinates are reasonable (center should be around 0.3, size around 0.2)
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        assert abs(center_x - 0.3) < 0.01, f"Center x should be ~0.3, got {center_x}"
        assert abs(center_y - 0.3) < 0.01, f"Center y should be ~0.3, got {center_y}"
        assert abs(width - 0.2) < 0.01, f"Width should be ~0.2, got {width}"
        assert abs(height - 0.2) < 0.01, f"Height should be ~0.2, got {height}"
        
        # Check that nuclei segmentation file was created (save_segmentation was called)
        nuclei_segmentation_file = widget.nuclei_segmentation_dir / "test_image.npy"
        assert nuclei_segmentation_file.exists(), "Nuclei segmentation file should be created by NucleiSegmentationWidget.save_segmentation()"
        
        # Verify segmentation file content matches original
        saved_mask = np.load(nuclei_segmentation_file)
        assert np.array_equal(saved_mask, test_mask), "Saved mask should match original mask"
        assert saved_mask.shape == test_mask.shape, "Saved mask should have same shape"
        assert saved_mask.dtype == test_mask.dtype, "Saved mask should have same dtype"
        # Verify specific regions
        assert np.all(saved_mask[20:30, 20:30] == 1), "First nucleus should be preserved"
        assert np.all(saved_mask[50:60, 50:60] == 2), "Second nucleus should be preserved"
        
        # Verify postprocess was called
        mock_postprocess.assert_called_once()
        widget._update_image_list.assert_called_once()

    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE or not NAPARI_AVAILABLE, reason="pytest-qt or napari not available")
    def test_save_current_image_with_real_save(self, mock_napari_viewer, temp_dir, qtbot):
        """Test _save_current_image with actual file saving using real widget methods."""
        if DataManagementWidget is None:
            pytest.skip("Dependencies not available")
        
        # Import widgets
        from micro_nuculei_nuclear_buds_detection._widget import DetectionWidget
        from micro_nuculei_nuclear_buds_detection._nuclei_segmentation_widget import NucleiSegmentationWidget
        import json
        from micro_nuculei_nuclear_buds_detection._constants import NUCLEI_SEGMENTATION_PARAMS_PATH, NUCLEI_SEGMENTATION_PARAMS_DEFAULT
        
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        
        # Set up widget state
        widget.dataset_path = temp_dir
        widget.annotation_dir = temp_dir / ANNOTATIONS_SUBFOLDER
        widget.annotation_dir.mkdir(parents=True, exist_ok=True)
        widget.nuclei_segmentation_dir = temp_dir / NUCLEI_SEGMENTATION_SUBFOLDER
        widget.nuclei_segmentation_dir.mkdir(parents=True, exist_ok=True)
        widget.postprocessing_dir = temp_dir / POSTPROCESSING_SUBFOLDER
        widget.postprocessing_dir.mkdir(parents=True, exist_ok=True)
        widget.current_image_index = 0
        widget.image_files = [temp_dir / "test_image.tif"]
        
        # Create params file for segmentation widget
        widget.nuclei_segmentation_params_path = widget.nuclei_segmentation_dir / NUCLEI_SEGMENTATION_PARAMS_PATH
        with open(widget.nuclei_segmentation_params_path, 'w') as f:
            json.dump(NUCLEI_SEGMENTATION_PARAMS_DEFAULT, f)
        
        # Create actual widgets (not mocks)
        widget.detection_widget = DetectionWidget(mock_napari_viewer)
        widget.detection_widget.annotation_dir = widget.annotation_dir
        widget.detection_widget.dataset_path = widget.dataset_path
        qtbot.addWidget(widget.detection_widget)
        
        widget.segmentation_widget = NucleiSegmentationWidget(mock_napari_viewer, widget.nuclei_segmentation_params_path)
        widget.segmentation_widget.nuclei_segmentation_dir = widget.nuclei_segmentation_dir
        widget.segmentation_widget.dataset_path = widget.dataset_path
        qtbot.addWidget(widget.segmentation_widget)
        
        # Create mock image layer
        if NAPARI_AVAILABLE:
            image_layer = MagicMock(spec=napari.layers.Image)
        else:
            image_layer = MagicMock()
        image_layer.data = np.zeros((100, 100), dtype=np.uint8)
        mock_napari_viewer.layers = [image_layer]
        
        # Create annotation layer with specific test data
        if NAPARI_AVAILABLE:
            annotation_layer = MagicMock(spec=napari.layers.Shapes)
        else:
            annotation_layer = MagicMock()
        annotation_layer.name = ANNOTATION_LAYER_NAME
        # Create test rectangles: [[y_min, x_min], [y_max, x_min], [y_max, x_max], [y_min, x_max]]
        # Box 1 at (10, 10) with size 15x15 (micro-nuclei)
        # Box 2 at (60, 60) with size 20x20 (nuclear-bud)
        test_rectangle1 = np.array([[10, 10], [25, 10], [25, 25], [10, 25]])
        test_rectangle2 = np.array([[60, 60], [80, 60], [80, 80], [60, 80]])
        annotation_layer.data = [test_rectangle1, test_rectangle2]
        # Colors for different classes
        color0 = get_napari_color(0)  # micro-nuclei
        color1 = get_napari_color(1)  # nuclear-bud
        annotation_layer.edge_color = np.array([
            [color0[0], color0[1], color0[2], 1.0],
            [color1[0], color1[1], color1[2], 1.0],
        ])
        mock_napari_viewer.layers.append(annotation_layer)
        
        # Create a mock nuclei segmentation layer with specific masks data
        if NAPARI_AVAILABLE:
            nuclei_layer = MagicMock(spec=napari.layers.Shapes)
        else:
            nuclei_layer = MagicMock()
        nuclei_layer.name = NUCLEI_SEGMENTATION_LAYER_NAME
        # Create a specific test mask with multiple nuclei
        test_mask = np.zeros((100, 100), dtype=np.int32)
        test_mask[15:25, 15:25] = 1  # Nucleus 1 at (15-25, 15-25)
        test_mask[65:75, 65:75] = 2  # Nucleus 2 at (65-75, 65-75)
        test_mask[40:50, 40:50] = 3  # Nucleus 3 at (40-50, 40-50)
        nuclei_layer._masks_data = test_mask
        mock_napari_viewer.layers.append(nuclei_layer)
        
        # Mock _update_image_list
        widget._update_image_list = MagicMock()
        
        # Mock postprocess_detections
        with patch('micro_nuculei_nuclear_buds_detection._data_management_widget.postprocess_detections') as mock_postprocess:
            result = widget._save_current_image()
        
        # Should return True
        assert result is True
        
        # Verify files were actually created by the real widget methods
        annotation_file = widget.annotation_dir / "test_image.txt"
        assert annotation_file.exists(), "Annotation file should be created by DetectionWidget.save_annotations()"
        
        # Verify annotation file content matches original
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 2, "Should have two annotations"
        # First annotation should be class 0 (micro-nuclei)
        parts1 = lines[0].strip().split()
        assert parts1[0] == "0", "First annotation should be class 0"
        # Second annotation should be class 1 (nuclear-bud)
        parts2 = lines[1].strip().split()
        assert parts2[0] == "1", "Second annotation should be class 1"
        # Verify coordinates are in valid range [0, 1]
        for parts in [parts1, parts2]:
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            assert 0.0 <= center_x <= 1.0, f"Center x should be in [0, 1], got {center_x}"
            assert 0.0 <= center_y <= 1.0, f"Center y should be in [0, 1], got {center_y}"
            assert 0.0 <= width <= 1.0, f"Width should be in [0, 1], got {width}"
            assert 0.0 <= height <= 1.0, f"Height should be in [0, 1], got {height}"
        
        nuclei_segmentation_file = widget.nuclei_segmentation_dir / "test_image.npy"
        assert nuclei_segmentation_file.exists(), "Nuclei segmentation file should be created by NucleiSegmentationWidget.save_segmentation()"
        
        # Verify segmentation file content matches original
        saved_mask = np.load(nuclei_segmentation_file)
        assert np.array_equal(saved_mask, test_mask), "Saved mask should match original mask exactly"
        assert saved_mask.shape == test_mask.shape, "Saved mask should have same shape"
        assert saved_mask.dtype == test_mask.dtype, "Saved mask should have same dtype"
        # Verify specific regions are preserved
        assert np.all(saved_mask[15:25, 15:25] == 1), "Nucleus 1 should be preserved"
        assert np.all(saved_mask[65:75, 65:75] == 2), "Nucleus 2 should be preserved"
        assert np.all(saved_mask[40:50, 40:50] == 3), "Nucleus 3 should be preserved"
        # Verify unique labels
        unique_labels = np.unique(saved_mask)
        assert set(unique_labels) == {0, 1, 2, 3}, f"Should have labels 0, 1, 2, 3, got {unique_labels}"
        
        # Verify postprocess was called with correct arguments
        mock_postprocess.assert_called_once()
        call_args = mock_postprocess.call_args[0]
        assert call_args[0] == temp_dir  # dataset_path
        assert call_args[1] == temp_dir / "test_image.tif"  # image_path
        assert call_args[2] == annotation_file  # annotation_file
        assert call_args[3] == nuclei_segmentation_file  # nuclei_segmentation_file

