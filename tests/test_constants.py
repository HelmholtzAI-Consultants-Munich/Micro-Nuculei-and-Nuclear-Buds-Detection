"""Tests for constants and color conversion functions."""

import pytest
import numpy as np
import importlib.util
from pathlib import Path

# Import directly from the module file to avoid triggering __init__.py imports
_constants_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "_constants.py"
spec = importlib.util.spec_from_file_location("_constants", _constants_path)
_constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_constants)

get_napari_color = _constants.get_napari_color
get_class_from_napari_color = _constants.get_class_from_napari_color
CLASS_COLORS = _constants.CLASS_COLORS
CLASS_NAMES = _constants.CLASS_NAMES


class TestGetNapariColor:
    """Tests for get_napari_color function."""

    def test_get_napari_color_micro_nuclei(self):
        """Test color conversion for micro-nuclei class (class 0)."""
        color = get_napari_color(0)
        assert len(color) == 3
        assert color == (0 / 255.0, 114 / 255.0, 178 / 255.0)
        assert all(0.0 <= c <= 1.0 for c in color)

    def test_get_napari_color_nuclear_buds(self):
        """Test color conversion for nuclear-buds class (class 1)."""
        color = get_napari_color(1)
        assert len(color) == 3
        assert color == (213 / 255.0, 94 / 255.0, 0 / 255.0)
        assert all(0.0 <= c <= 1.0 for c in color)

    def test_get_napari_color_all_classes(self):
        """Test color conversion for all defined classes."""
        for class_id in CLASS_COLORS.keys():
            color = get_napari_color(class_id)
            assert len(color) == 3
            assert all(0.0 <= c <= 1.0 for c in color)
            # Verify it matches expected normalized RGB
            expected_r, expected_g, expected_b = CLASS_COLORS[class_id]
            assert color == (expected_r / 255.0, expected_g / 255.0, expected_b / 255.0)

    def test_get_napari_color_invalid_class_id(self):
        """Test that invalid class ID raises KeyError."""
        with pytest.raises(KeyError):
            get_napari_color(999)


class TestGetClassFromNapariColor:
    """Tests for get_class_from_napari_color function."""

    def test_get_class_from_napari_color_exact_match_micro_nuclei(self):
        """Test exact color match for micro-nuclei."""
        # Micro-nuclei color: (0, 114, 178) normalized
        color = (0 / 255.0, 114 / 255.0, 178 / 255.0)
        class_id = get_class_from_napari_color(color)
        assert class_id == 0
        assert CLASS_NAMES[class_id] == "micro_nuclei"

    def test_get_class_from_napari_color_exact_match_nuclear_buds(self):
        """Test exact color match for nuclear-buds."""
        # Nuclear-buds color: (213, 94, 0) normalized
        color = (213 / 255.0, 94 / 255.0, 0 / 255.0)
        class_id = get_class_from_napari_color(color)
        assert class_id == 1
        assert CLASS_NAMES[class_id] == "nuclear_buds"

    def test_get_class_from_napari_color_rounding(self):
        """Test that slight rounding errors still match."""
        # Test with values that round to the correct integer RGB
        color = (0.0001, 114.4 / 255.0, 177.6 / 255.0)
        class_id = get_class_from_napari_color(color)
        assert class_id == 0

    def test_get_class_from_napari_color_closest_match(self):
        """Test closest color matching when exact match not found."""
        # Use a color close to micro-nuclei but not exact
        color = (1 / 255.0, 115 / 255.0, 179 / 255.0)
        class_id = get_class_from_napari_color(color)
        # Should return closest class (micro-nuclei)
        assert class_id == 0

    def test_get_class_from_napari_color_out_of_range(self):
        """Test with colors outside normal range."""
        # Very high values
        color = (1.5, 2.0, 0.5)
        class_id = get_class_from_napari_color(color)
        # Should still return a valid class ID (closest match)
        assert class_id in CLASS_NAMES.keys()

    def test_get_class_from_napari_color_negative_values(self):
        """Test with negative color values."""
        color = (-0.1, 0.0, 0.0)
        class_id = get_class_from_napari_color(color)
        # Should still return a valid class ID
        assert class_id in CLASS_NAMES.keys()

    def test_get_class_from_napari_color_roundtrip(self):
        """Test that converting class to color and back works."""
        for class_id in CLASS_COLORS.keys():
            color = get_napari_color(class_id)
            recovered_class_id = get_class_from_napari_color(color)
            assert recovered_class_id == class_id

