"""Comprehensive tests for metadata actions."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from photoflow.actions.metadata import (
    CopyMetadataAction,
    ExtractMetadataAction,
    RemoveMetadataAction,
    WriteMetadataAction,
)
from photoflow.core.actions import BaseAction
from photoflow.core.fields import BooleanField, ChoiceField, StringField
from photoflow.core.metadata import ExifData, IPTCData, MetadataError
from photoflow.core.models import ImageAsset


class TestWriteMetadataAction:
    """Test WriteMetadataAction."""

    def test_action_attributes(self):
        """Test action has correct attributes."""
        action = WriteMetadataAction()
        assert action.name == "write_metadata"
        assert action.label == "Write Metadata"
        assert "metadata" in action.tags
        assert "exif" in action.tags
        assert "iptc" in action.tags

    def test_setup_fields(self):
        """Test field setup."""
        action = WriteMetadataAction()
        action.setup_fields()
        expected_fields = ["title", "caption", "creator", "copyright", "keywords"]
        for field_name in expected_fields:
            assert field_name in action.fields

    def test_get_field_schema(self):
        """Test get_field_schema method."""
        action = WriteMetadataAction()
        schema = action.get_field_schema()
        assert isinstance(schema, dict)
        assert "title" in schema
        assert "caption" in schema


class TestRemoveMetadataAction:
    """Test RemoveMetadataAction."""

    def test_action_attributes(self):
        """Test action has correct attributes."""
        action = RemoveMetadataAction()
        assert action.name == "remove_metadata"
        assert action.label == "Remove Metadata"
        assert "metadata" in action.tags


class TestCopyMetadataAction:
    """Test CopyMetadataAction."""

    def test_action_attributes(self):
        """Test action has correct attributes."""
        action = CopyMetadataAction()
        assert action.name == "copy_metadata"
        assert action.label == "Copy Metadata"
        assert "metadata" in action.tags


class TestExtractMetadataAction:
    """Test ExtractMetadataAction."""

    def test_action_attributes(self):
        """Test action has correct attributes."""
        action = ExtractMetadataAction()
        assert action.name == "extract_metadata"
        assert action.label == "Extract Metadata"
        assert "metadata" in action.tags


class TestMetadataActionsIntegration:
    """Integration tests for metadata actions."""

    def test_all_actions_have_required_attributes(self):
        """Test that all metadata actions have required attributes."""
        actions = [
            WriteMetadataAction(),
            RemoveMetadataAction(),
            CopyMetadataAction(),
            ExtractMetadataAction(),
        ]
        for action in actions:
            assert hasattr(action, "name")
            assert hasattr(action, "label")
            assert hasattr(action, "description")
            assert hasattr(action, "tags")
            assert action.name
            assert action.label
            assert action.description
            assert action.tags
            assert "metadata" in action.tags

    def test_all_actions_setup_fields(self):
        """Test that all actions can setup their fields."""
        actions = [
            WriteMetadataAction(),
            RemoveMetadataAction(),
            CopyMetadataAction(),
            ExtractMetadataAction(),
        ]
        for action in actions:
            action.setup_fields()
            assert hasattr(action, "fields")
            assert isinstance(action.fields, dict)
            assert len(action.fields) > 0

    def test_all_actions_get_field_schema(self):
        """Test that all actions can generate field schemas."""
        actions = [
            WriteMetadataAction(),
            RemoveMetadataAction(),
            CopyMetadataAction(),
            ExtractMetadataAction(),
        ]
        for action in actions:
            schema = action.get_field_schema()
            assert isinstance(schema, dict)
            assert len(schema) > 0

    def test_action_inheritance(self):
        """Test that all metadata actions inherit from BaseAction."""
        actions = [
            WriteMetadataAction(),
            RemoveMetadataAction(),
            CopyMetadataAction(),
            ExtractMetadataAction(),
        ]
        for action in actions:
            assert isinstance(action, BaseAction)

    def test_action_versions(self):
        """Test that all actions have version information."""
        actions = [
            WriteMetadataAction(),
            RemoveMetadataAction(),
            CopyMetadataAction(),
            ExtractMetadataAction(),
        ]
        for action in actions:
            assert hasattr(action, "version")
            assert action.version is not None
            assert isinstance(action.version, str)

    def test_action_authors(self):
        """Test that all actions have author information."""
        actions = [
            WriteMetadataAction(),
            RemoveMetadataAction(),
            CopyMetadataAction(),
            ExtractMetadataAction(),
        ]
        for action in actions:
            assert hasattr(action, "author")
            assert action.author is not None
            assert isinstance(action.author, str)


class TestWriteMetadataActionApply:
    """Test WriteMetadataAction apply method."""

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_basic(self, mock_write_metadata, mock_read_metadata):
        """Test basic apply functionality."""
        action = WriteMetadataAction()
        exif_data = ExifData()
        iptc_data = IPTCData()
        mock_read_metadata.return_value = exif_data, iptc_data
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {
            "title": "Test Title",
            "creator": "Test Creator",
            "keywords": "test,photo,metadata",
        }
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        mock_write_metadata.assert_called_once()
        assert "metadata_written" in result.metadata
        assert result.metadata["metadata_written"] is True

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_with_overwrite_false(self, mock_write_metadata, mock_read_metadata):
        """Test apply with overwrite=False."""
        action = WriteMetadataAction()
        exif_data = ExifData(creator="Existing Creator")
        iptc_data = IPTCData(title="Existing Title")
        mock_read_metadata.return_value = exif_data, iptc_data
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"title": "New Title", "creator": "New Creator", "overwrite": False}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        mock_write_metadata.assert_called_once()
        assert "metadata_written" in result.metadata

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_metadata_error(self, mock_read_metadata):
        """Test apply with metadata error."""
        action = WriteMetadataAction()
        mock_read_metadata.side_effect = MetadataError("Test error")
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        with pytest.raises(ValueError, match="Failed to write metadata"):
            action.apply(image, title="Test")

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_keywords_list(self, mock_write_metadata, mock_read_metadata):
        """Test apply with keywords as list."""
        action = WriteMetadataAction()
        exif_data = ExifData()
        iptc_data = IPTCData()
        mock_read_metadata.return_value = exif_data, iptc_data
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"keywords": ["test", "photo", "metadata"]}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        mock_write_metadata.assert_called_once()
        assert "metadata_written" in result.metadata


class TestRemoveMetadataActionApply:
    """Test RemoveMetadataAction apply method."""

    @patch("photoflow.actions.metadata.PILImage")
    def test_apply_remove_all(self, mock_pil_image):
        """Test removing all metadata."""
        action = RemoveMetadataAction()
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = 1920, 1080
        mock_img.format = "JPEG"
        mock_img.info = {"dpi": (72, 72)}
        mock_img.getdata.return_value = [0] * (1920 * 1080)
        mock_clean_img = Mock()
        mock_pil_image.open.return_value.__enter__.return_value = mock_img
        mock_pil_image.new.return_value = mock_clean_img
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"metadata_type": "all"}
        result = action.apply(image, **params)
        mock_pil_image.open.assert_called_once_with(image.path)
        mock_clean_img.save.assert_called_once()
        assert "metadata_removed" in result.metadata
        assert result.metadata["metadata_removed"] == "all"

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_remove_exif(self, mock_write_metadata, mock_read_metadata):
        """Test removing EXIF metadata only."""
        action = RemoveMetadataAction()
        exif_data = ExifData(make="Canon", model="EOS R5")
        iptc_data = IPTCData(title="Test Image")
        mock_read_metadata.return_value = exif_data, iptc_data
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"metadata_type": "exif"}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        mock_write_metadata.assert_called_once()
        assert result.metadata["metadata_removed"] == "exif"

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_remove_gps(self, mock_write_metadata, mock_read_metadata):
        """Test removing GPS metadata only."""
        action = RemoveMetadataAction()
        exif_data = ExifData(gps_latitude=40.7128, gps_longitude=-74.006)
        iptc_data = IPTCData()
        mock_read_metadata.return_value = exif_data, iptc_data
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"metadata_type": "gps"}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        mock_write_metadata.assert_called_once()
        assert result.metadata["metadata_removed"] == "gps"

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_remove_personal(self, mock_write_metadata, mock_read_metadata):
        """Test removing personal information."""
        action = RemoveMetadataAction()
        exif_data = ExifData(creator="John Doe", copyright="© John Doe")
        iptc_data = IPTCData(byline="John Doe", location="New York")
        mock_read_metadata.return_value = exif_data, iptc_data
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"metadata_type": "personal"}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        mock_write_metadata.assert_called_once()
        assert result.metadata["metadata_removed"] == "personal"


class TestCopyMetadataActionApply:
    """Test CopyMetadataAction apply method."""

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_copy_all(self, mock_write_metadata, mock_read_metadata):
        """Test copying all metadata."""
        action = CopyMetadataAction()
        source_exif = ExifData(make="Canon", model="EOS R5")
        source_iptc = IPTCData(title="Source Image")
        target_exif = ExifData()
        target_iptc = IPTCData()
        mock_read_metadata.side_effect = [
            (source_exif, source_iptc),
            (target_exif, target_iptc),
        ]
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            source_path = tmp.name
        try:
            params = {"source_image": source_path, "metadata_type": "all"}
            result = action.apply(image, **params)
            assert mock_read_metadata.call_count == 2
            mock_write_metadata.assert_called_once()
            assert "metadata_copied_from" in result.metadata
        finally:
            Path(source_path).unlink(missing_ok=True)

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    def test_apply_copy_camera_settings(self, mock_write_metadata, mock_read_metadata):
        """Test copying camera settings only."""
        action = CopyMetadataAction()
        source_exif = ExifData(make="Canon", focal_length="85mm", f_number="f/2.8")
        source_iptc = IPTCData()
        target_exif = ExifData()
        target_iptc = IPTCData(title="Target Image")
        mock_read_metadata.side_effect = [
            (source_exif, source_iptc),
            (target_exif, target_iptc),
        ]
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            source_path = tmp.name
        try:
            params = {"source_image": source_path, "metadata_type": "camera"}
            result = action.apply(image, **params)
            mock_write_metadata.assert_called_once()
            assert result.metadata["metadata_copy_type"] == "camera"
        finally:
            Path(source_path).unlink(missing_ok=True)

    def test_apply_source_not_found(self):
        """Test error when source image not found."""
        action = CopyMetadataAction()
        image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"source_image": "/nonexistent/image.jpg"}
        with pytest.raises(ValueError, match="Source image not found"):
            action.apply(image, **params)


class TestExtractMetadataActionApply:
    """Test ExtractMetadataAction apply method."""

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_extract_json(self, mock_read_metadata):
        """Test extracting metadata to JSON."""
        action = ExtractMetadataAction()
        exif_data = ExifData(make="Canon", model="EOS R5")
        iptc_data = IPTCData(title="Test Image")
        mock_read_metadata.return_value = exif_data, iptc_data
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        image = ImageAsset(path=Path("/tmp/image.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"output_format": "json", "output_path": tmp_path}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once_with(image.path)
        assert "metadata_extracted_to" in result.metadata
        assert result.metadata["metadata_extract_format"] == "json"
        Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_extract_yaml(self, mock_read_metadata):
        """Test extracting metadata to YAML."""
        action = ExtractMetadataAction()
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        mock_read_metadata.return_value = exif_data, iptc_data
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        image = ImageAsset(path=Path("/tmp/image.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"output_format": "yaml", "output_path": tmp_path}
        result = action.apply(image, **params)
        assert result.metadata["metadata_extract_format"] == "yaml"
        Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_extract_csv(self, mock_read_metadata):
        """Test extracting metadata to CSV."""
        action = ExtractMetadataAction()
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        mock_read_metadata.return_value = exif_data, iptc_data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        image = ImageAsset(path=Path("/tmp/image.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"output_format": "csv", "output_path": tmp_path}
        result = action.apply(image, **params)
        assert result.metadata["metadata_extract_format"] == "csv"
        Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_extract_txt(self, mock_read_metadata):
        """Test extracting metadata to text format."""
        action = ExtractMetadataAction()
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        mock_read_metadata.return_value = exif_data, iptc_data
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        image = ImageAsset(path=Path("/tmp/image.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"output_format": "txt", "output_path": tmp_path}
        result = action.apply(image, **params)
        assert result.metadata["metadata_extract_format"] == "txt"
        Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_extract_with_raw_data(self, mock_read_metadata):
        """Test extracting metadata with raw data included."""
        action = ExtractMetadataAction()
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        mock_read_metadata.return_value = exif_data, iptc_data
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        image = ImageAsset(path=Path("/tmp/image.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        params = {"output_format": "json", "include_raw": True, "output_path": tmp_path}
        result = action.apply(image, **params)
        mock_read_metadata.assert_called_once()
        Path(tmp_path).unlink(missing_ok=True)
        assert "metadata_extracted_to" in result.metadata

    @patch("photoflow.actions.metadata.read_metadata")
    def test_apply_extract_error(self, mock_read_metadata):
        """Test error handling in metadata extraction."""
        action = ExtractMetadataAction()
        mock_read_metadata.side_effect = Exception("Test error")
        image = ImageAsset(path=Path("/test/image.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
        with pytest.raises(ValueError, match="Failed to extract metadata"):
            action.apply(image, output_format="json")


class TestMetadataActionsFieldSetup:
    """Test field setup for metadata actions."""

    def test_write_metadata_field_types(self):
        """Test WriteMetadataAction field types."""
        action = WriteMetadataAction()
        action.setup_fields()
        assert isinstance(action.fields["title"], StringField)
        assert isinstance(action.fields["overwrite"], BooleanField)
        assert action.fields["overwrite"].default is True

    def test_remove_metadata_field_types(self):
        """Test RemoveMetadataAction field types."""
        action = RemoveMetadataAction()
        action.setup_fields()
        assert isinstance(action.fields["metadata_type"], ChoiceField)
        assert isinstance(action.fields["preserve_basic"], BooleanField)
        assert action.fields["metadata_type"].default == "all"

    def test_copy_metadata_field_types(self):
        """Test CopyMetadataAction field types."""
        action = CopyMetadataAction()
        action.setup_fields()
        assert isinstance(action.fields["source_image"], StringField)
        assert isinstance(action.fields["metadata_type"], ChoiceField)
        assert isinstance(action.fields["overwrite"], BooleanField)
        assert action.fields["source_image"].required is True

    def test_extract_metadata_field_types(self):
        """Test ExtractMetadataAction field types."""
        action = ExtractMetadataAction()
        action.setup_fields()
        assert isinstance(action.fields["output_format"], ChoiceField)
        assert isinstance(action.fields["output_path"], StringField)
        assert isinstance(action.fields["include_raw"], BooleanField)
        assert action.fields["output_format"].default == "json"
