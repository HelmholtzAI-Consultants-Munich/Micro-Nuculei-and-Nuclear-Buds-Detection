"""Shared pytest fixtures and mocks for testing."""

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def mock_napari_viewer():
    """Create a mock napari viewer for testing widgets."""
    viewer = MagicMock()
    
    # Mock layers
    viewer.layers = MagicMock()
    viewer.layers.selection = MagicMock()
    viewer.layers.selection.active = None
    
    # Mock window
    viewer.window = MagicMock()
    viewer.window.add_dock_widget = MagicMock()
    viewer.window.remove_dock_widget = MagicMock()
    
    return viewer


@pytest.fixture
def mock_shapes_layer():
    """Create a mock napari shapes layer."""
    layer = MagicMock()
    layer.name = "bounding_box_annotations"
    layer.data = []
    layer.edge_color = np.array([[1.0, 0.0, 0.0, 1.0]])
    layer.face_color = "transparent"
    layer.selected_data = set()
    layer.visible = True
    layer.mode = "select"
    return layer


@pytest.fixture
def mock_image_layer():
    """Create a mock napari image layer."""
    layer = MagicMock()
    layer.name = "test_image"
    layer.data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    layer.visible = True
    return layer


@pytest.fixture
def synthetic_mask_2d():
    """Create a synthetic 2D mask with multiple labeled regions."""
    mask = np.zeros((50, 50), dtype=np.int32)
    # Create 3 labeled regions
    mask[10:20, 10:20] = 1
    mask[25:35, 25:35] = 2
    mask[5:15, 30:40] = 3
    return mask


@pytest.fixture
def synthetic_mask_single():
    """Create a synthetic 2D mask with a single labeled region."""
    mask = np.zeros((30, 30), dtype=np.int32)
    mask[10:20, 10:20] = 1
    return mask


@pytest.fixture
def synthetic_mask_empty():
    """Create an empty mask (all zeros)."""
    return np.zeros((20, 20), dtype=np.int32)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_annotations():
    """Create sample annotation data in YOLO format."""
    # YOLO format: class_id center_y center_x height width
    annotations = [
        "0 0.5 0.5 0.2 0.2\n",  # micro-nuclei at center
        "1 0.3 0.3 0.15 0.15\n",  # nuclear-bud at top-left
        "0 0.7 0.7 0.1 0.1\n",  # micro-nuclei at bottom-right
    ]
    return annotations


@pytest.fixture
def sample_bounding_boxes():
    """Create sample bounding box coordinates for testing."""
    # Format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    boxes = [
        np.array([[40, 40], [60, 40], [60, 60], [40, 60]]),  # Box 1
        np.array([[20, 20], [35, 20], [35, 35], [20, 35]]),  # Box 2
        np.array([[70, 70], [80, 70], [80, 80], [70, 80]]),  # Box 3
    ]
    return boxes

