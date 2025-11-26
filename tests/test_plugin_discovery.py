"""Tests for napari plugin discovery and widget registration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

# pytest-qt is required for Qt widget tests
pytest.importorskip("pytestqt")

try:
    import yaml
except ImportError:
    yaml = None
    pytest.skip("PyYAML not installed", allow_module_level=True)

from micro_nuculei_nuclear_buds_detection import (
    DetectionWidget,
    DataManagementWidget,
    NucleiSegmentationWidget,
)
from micro_nuculei_nuclear_buds_detection._widget import DetectionWidget as DetectionWidgetClass
from micro_nuculei_nuclear_buds_detection._data_management_widget import (
    DataManagementWidget as DataManagementWidgetClass,
)
from micro_nuculei_nuclear_buds_detection._nuclei_segmentation_widget import (
    NucleiSegmentationWidget as NucleiSegmentationWidgetClass,
)


class TestPluginManifest:
    """Tests for napari.yaml plugin manifest."""

    def test_manifest_file_exists(self):
        """Test that napari.yaml manifest file exists."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        assert manifest_path.exists(), "napari.yaml manifest file not found"

    def test_manifest_valid_yaml(self):
        """Test that napari.yaml is valid YAML."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        assert manifest is not None
        assert isinstance(manifest, dict)

    def test_manifest_required_fields(self):
        """Test that manifest contains required fields."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Check required top-level fields
        assert "name" in manifest
        assert "display_name" in manifest
        assert "version" in manifest
        assert "contributions" in manifest
        
        # Check contributions structure
        contributions = manifest["contributions"]
        assert "commands" in contributions
        assert "widgets" in contributions

    def test_manifest_commands_defined(self):
        """Test that all widget commands are defined in manifest."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        commands = manifest["contributions"]["commands"]
        command_ids = [cmd["id"] for cmd in commands]
        
        # Check that expected commands exist
        assert "micro-nuculei-nuclear-buds-detection.detect" in command_ids
        assert "micro-nuculei-nuclear-buds-detection.data_management" in command_ids
        assert "micro-nuculei-nuclear-buds-detection.nuclei_segmentation" in command_ids

    def test_manifest_widgets_defined(self):
        """Test that all widgets are defined in manifest."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        widgets = manifest["contributions"]["widgets"]
        widget_commands = [widget["command"] for widget in widgets]
        
        # Check that expected widget commands exist
        assert "micro-nuculei-nuclear-buds-detection.detect" in widget_commands
        assert "micro-nuculei-nuclear-buds-detection.data_management" in widget_commands
        assert "micro-nuculei-nuclear-buds-detection.nuclei_segmentation" in widget_commands

    def test_manifest_python_names_valid(self):
        """Test that Python names in commands are valid."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        commands = manifest["contributions"]["commands"]
        
        for cmd in commands:
            python_name = cmd.get("python_name")
            if python_name:
                # Python name should be in format "module:Class"
                assert ":" in python_name
                module_path, class_name = python_name.split(":")
                
                # Try to import the class
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    assert hasattr(module, class_name), f"Class {class_name} not found in {module_path}"
                except ImportError as e:
                    pytest.fail(f"Could not import {python_name}: {e}")


class TestWidgetInstantiation:
    """Tests for widget instantiation."""

    def test_detection_widget_instantiation(self, mock_napari_viewer, qtbot):
        """Test that DetectionWidget can be instantiated."""
        widget = DetectionWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        
        assert widget is not None
        assert isinstance(widget, DetectionWidgetClass)
        assert widget.viewer == mock_napari_viewer

    def test_data_management_widget_instantiation(self, mock_napari_viewer, qtbot):
        """Test that DataManagementWidget can be instantiated."""
        widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(widget)
        
        assert widget is not None
        assert isinstance(widget, DataManagementWidgetClass)
        assert widget.viewer == mock_napari_viewer

    def test_nuclei_segmentation_widget_instantiation(self, mock_napari_viewer, temp_dir):
        """Test that NucleiSegmentationWidget can be instantiated."""
        # Create a temporary params file
        params_file = temp_dir / "nuclei_segmentation_params.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        
        widget = NucleiSegmentationWidget(mock_napari_viewer, params_file)
        
        assert widget is not None
        assert isinstance(widget, NucleiSegmentationWidgetClass)
        assert widget.viewer == mock_napari_viewer

    def test_widgets_have_required_attributes(self, mock_napari_viewer, qtbot):
        """Test that widgets have required attributes."""
        detection_widget = DetectionWidget(mock_napari_viewer)
        qtbot.addWidget(detection_widget)
        data_widget = DataManagementWidget(mock_napari_viewer)
        qtbot.addWidget(data_widget)
        
        # Check that widgets have viewer attribute
        assert hasattr(detection_widget, "viewer")
        assert hasattr(data_widget, "viewer")
        
        # Check that widgets have layout
        assert hasattr(detection_widget, "layout")
        assert hasattr(data_widget, "layout")


class TestPluginEntryPoints:
    """Tests for plugin entry points and registration."""

    def test_widget_classes_importable(self):
        """Test that widget classes can be imported from package."""
        from micro_nuculei_nuclear_buds_detection._widget import DetectionWidget
        from micro_nuculei_nuclear_buds_detection._data_management_widget import DataManagementWidget
        from micro_nuculei_nuclear_buds_detection._nuclei_segmentation_widget import NucleiSegmentationWidget
        
        assert DetectionWidget is not None
        assert DataManagementWidget is not None
        assert NucleiSegmentationWidget is not None

    def test_widget_classes_match_manifest(self):
        """Test that widget classes in manifest match actual classes."""
        manifest_path = Path(__file__).parent.parent / "micro_nuculei_nuclear_buds_detection" / "napari.yaml"
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        commands = manifest["contributions"]["commands"]
        
        for cmd in commands:
            python_name = cmd.get("python_name")
            if python_name:
                module_path, class_name = python_name.split(":")
                
                # Import the class
                module = __import__(module_path, fromlist=[class_name])
                widget_class = getattr(module, class_name)
                
                # Verify it's a class
                assert isinstance(widget_class, type), f"{python_name} is not a class"
                
                # Verify it can be instantiated with a mock viewer
                # Note: This test may need qtbot for Qt widgets, but we'll test import/class structure here
                # Full instantiation tests are in TestWidgetInstantiation
                pass  # Just verify the class exists and is importable

    def test_package_exports_widgets(self):
        """Test that package __init__ exports widget classes."""
        from micro_nuculei_nuclear_buds_detection import (
            DetectionWidget,
            DataManagementWidget,
            NucleiSegmentationWidget,
        )
        
        assert DetectionWidget is not None
        assert DataManagementWidget is not None
        assert NucleiSegmentationWidget is not None

