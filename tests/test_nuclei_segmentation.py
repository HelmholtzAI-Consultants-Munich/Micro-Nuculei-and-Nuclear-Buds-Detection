"""Tests for nuclei segmentation utility functions."""

import pytest
import numpy as np

# pytest-qt is required for Qt widget tests
pytest.importorskip("pytestqt")

from micro_nuculei_nuclear_buds_detection._nuclei_segmentation_widget import (
    NucleiSegmentationWidget,
)


class TestMasksToShapes:
    """Tests for _masks_to_shapes method."""

    def test_masks_to_shapes_single_nucleus(self, mock_napari_viewer, synthetic_mask_single, qtbot, temp_dir):
        """Test converting a single-nucleus mask to shapes."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        shapes = widget._masks_to_shapes(synthetic_mask_single)
        
        assert isinstance(shapes, list)
        assert len(shapes) == 1  # One nucleus
        assert isinstance(shapes[0], np.ndarray)
        assert shapes[0].ndim == 2
        assert shapes[0].shape[1] == 2  # (y, x) coordinates

    def test_masks_to_shapes_multiple_nuclei(self, mock_napari_viewer, synthetic_mask_2d, qtbot, temp_dir):
        """Test converting a multi-nucleus mask to shapes."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        shapes = widget._masks_to_shapes(synthetic_mask_2d)
        
        assert isinstance(shapes, list)
        # Should have 3 shapes (3 labeled regions)
        assert len(shapes) == 3
        for shape in shapes:
            assert isinstance(shape, np.ndarray)
            assert shape.ndim == 2
            assert shape.shape[1] == 2

    def test_masks_to_shapes_empty_mask(self, mock_napari_viewer, synthetic_mask_empty, qtbot, temp_dir):
        """Test converting an empty mask to shapes."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        shapes = widget._masks_to_shapes(synthetic_mask_empty)
        
        assert isinstance(shapes, list)
        assert len(shapes) == 0  # No nuclei

    def test_masks_to_shapes_shape_format(self, mock_napari_viewer, synthetic_mask_single, qtbot, temp_dir):
        """Test that shapes are in correct format for napari."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        shapes = widget._masks_to_shapes(synthetic_mask_single)
        
        if len(shapes) > 0:
            shape = shapes[0]
            # Should be (N, 2) array with (y, x) coordinates
            assert shape.ndim == 2
            assert shape.shape[1] == 2
            # Coordinates should be within mask bounds
            mask = synthetic_mask_single
            assert np.all(shape[:, 0] >= 0)  # y >= 0
            assert np.all(shape[:, 0] < mask.shape[0])  # y < height
            assert np.all(shape[:, 1] >= 0)  # x >= 0
            assert np.all(shape[:, 1] < mask.shape[1])  # x < width


class TestGenerateColors:
    """Tests for _generate_colors method."""

    def test_generate_colors_single_instance(self, mock_napari_viewer, qtbot, temp_dir):
        """Test color generation for a single instance."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        colors = widget._generate_colors(1)
        
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (1, 4)  # (n_instances, RGBA)
        assert np.all(colors[:, 3] == 0.5)  # Alpha should be 0.5
        assert np.all(colors[:, :3] >= 0.0)  # RGB >= 0
        assert np.all(colors[:, :3] <= 1.0)  # RGB <= 1

    def test_generate_colors_multiple_instances(self, mock_napari_viewer, qtbot, temp_dir):
        """Test color generation for multiple instances."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        n_instances = 10
        colors = widget._generate_colors(n_instances)
        
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (n_instances, 4)
        assert np.all(colors[:, 3] == 0.5)  # All have same alpha
        
        # Colors should be distinct (check that they're not all the same)
        unique_colors = np.unique(colors[:, :3], axis=0)
        assert len(unique_colors) == n_instances  # All should be unique

    def test_generate_colors_zero_instances(self, mock_napari_viewer, qtbot, temp_dir):
        """Test color generation for zero instances."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        colors = widget._generate_colors(n_instances=0)
        
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (0, 4)

    def test_generate_colors_rgba_format(self, mock_napari_viewer, qtbot, temp_dir):
        """Test that colors are in correct RGBA format."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        colors = widget._generate_colors(5)
        
        # Each color should be [R, G, B, A] with values in [0, 1]
        for color in colors:
            assert len(color) == 4
            assert 0.0 <= color[0] <= 1.0  # R
            assert 0.0 <= color[1] <= 1.0  # G
            assert 0.0 <= color[2] <= 1.0  # B
            assert color[3] == 0.5  # A (alpha)


class TestManualRemoveSmallMasks:
    """Tests for _manual_remove_small_masks method."""

    def test_manual_remove_small_masks_no_filtering(self, mock_napari_viewer, synthetic_mask_2d, qtbot, temp_dir):
        """Test mask filtering with min_size=0 (no filtering)."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        filtered = widget.remove_small_masks(synthetic_mask_2d, min_size=0)
        
        # Should keep all labels
        unique_original = np.unique(synthetic_mask_2d)
        unique_filtered = np.unique(filtered)
        # Background (0) + all labels should be present
        assert len(unique_filtered) == len(unique_original)

    def test_manual_remove_small_masks_filter_small(self, mock_napari_viewer, qtbot, temp_dir):
        """Test filtering out small masks."""
        # Create a mask with one large and one small region
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[10:30, 10:30] = 1  # Large region (20x20 = 400 pixels)
        mask[5:8, 5:8] = 2  # Small region (3x3 = 9 pixels)
        
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        filtered = widget.remove_small_masks(mask, min_size=100)
        
        # Small region (label 2) should be removed
        unique_filtered = np.unique(filtered)
        assert 2 not in unique_filtered  # Small region removed
        assert 1 in unique_filtered  # Large region kept
        # Should be relabeled to 1 (since 2 was removed)
        assert np.max(filtered) == 1

    def test_manual_remove_small_masks_all_removed(self, mock_napari_viewer, qtbot, temp_dir):
        """Test when all masks are filtered out."""
        # Create mask with only small regions
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:7, 5:7] = 1  # Small region (2x2 = 4 pixels)
        mask[10:12, 10:12] = 2  # Small region (2x2 = 4 pixels)
        
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        filtered = widget.remove_small_masks(mask, min_size=10)
        
        # All labels should be removed, only background (0) remains
        unique_filtered = np.unique(filtered)
        assert len(unique_filtered) == 1
        assert unique_filtered[0] == 0
        assert np.all(filtered == 0)

    def test_manual_remove_small_masks_relabeling(self, mock_napari_viewer, qtbot, temp_dir):
        """Test that remaining masks are properly relabeled."""
        # Create mask with labels 1, 2, 3, 4
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[5:15, 5:15] = 1  # Large (100 pixels)
        mask[20:22, 20:22] = 2  # Small (4 pixels) - will be removed
        mask[25:35, 25:35] = 3  # Large (100 pixels)
        mask[40:41, 40:41] = 4  # Small (1 pixel) - will be removed
        
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        filtered = widget.remove_small_masks(mask, min_size=50)
        
        # Labels 1 and 3 should remain, relabeled to 1 and 2
        assert filtered.shape == mask.shape
        assert filtered.dtype == mask.dtype
        assert np.all(filtered[mask == 1] == 1)
        assert np.all(filtered[mask == 2] == 0)
        assert np.all(filtered[mask == 3] == 2)
        assert np.all(filtered[mask == 4] == 0)

    def test_manual_remove_small_masks_empty_mask(self, mock_napari_viewer, synthetic_mask_empty, qtbot, temp_dir):
        """Test filtering on an empty mask."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        filtered = widget.remove_small_masks(synthetic_mask_empty, min_size=10)
        
        # Should remain empty
        assert np.all(filtered == 0)
        assert filtered.shape == synthetic_mask_empty.shape

    def test_manual_remove_small_masks_preserves_shape(self, mock_napari_viewer, synthetic_mask_2d, qtbot, temp_dir):
        """Test that filtering preserves mask shape."""
        params_file = temp_dir / "test_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        qtbot.addWidget(widget)
        filtered = widget.remove_small_masks(synthetic_mask_2d, min_size=0)
        
        assert filtered.shape == synthetic_mask_2d.shape
        assert filtered.dtype == synthetic_mask_2d.dtype

