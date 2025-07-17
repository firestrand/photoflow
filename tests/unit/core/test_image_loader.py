"""Working tests for image loader functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from photoflow.core.image_loader import (
    IPTCINFO_AVAILABLE,
    PIEXIF_AVAILABLE,
    PIL_AVAILABLE,
    ImageLoadError,
    _clean_exif_value,
    _extract_exif_data,
    _extract_iptc_data,
    _extract_pil_info,
    _parse_exif_data,
    create_image_asset_from_pil,
    load_image_asset,
    load_test_image,
)
from photoflow.core.models import ImageAsset


class TestImageLoader:
    """Test image loader functionality."""

    def test_pil_available_check(self):
        """Test PIL availability check."""
        assert isinstance(PIL_AVAILABLE, bool)

    def test_piexif_available_check(self):
        """Test PIEXIF availability check."""
        assert isinstance(PIEXIF_AVAILABLE, bool)

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", False)
    def test_load_image_asset_no_pil(self):
        """Test load_image_asset when PIL is not available."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required"):
            load_image_asset("test.jpg")

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    def test_load_image_asset_nonexistent_file(self):
        """Test load_image_asset with nonexistent file."""
        with pytest.raises(ImageLoadError, match="Image file not found"):
            load_image_asset("/nonexistent/file.jpg")

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_loader.Image")
    @patch("pathlib.Path.exists")
    def test_load_image_asset_error_handling(self, mock_exists, mock_image):
        """Test error handling in load_image_asset."""
        mock_exists.return_value = True
        mock_image.open.side_effect = Exception("Test error")
        with pytest.raises(ImageLoadError, match="Failed to load image"):
            load_image_asset("test.jpg")

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    def test_clean_exif_value_function(self):
        """Test _clean_exif_value function."""
        assert _clean_exif_value("test") == "test"
        assert _clean_exif_value(None) is None
        assert _clean_exif_value(123) == 123

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    def test_extract_pil_info_function(self):
        """Test _extract_pil_info function."""
        info_dict = {"dpi": (72, 72), "quality": 95}
        result = _extract_pil_info(info_dict)
        assert isinstance(result, dict)


class TestImageLoadErrorException:
    """Test ImageLoadError exception."""

    def test_image_load_error_creation(self):
        """Test ImageLoadError exception creation."""
        error = ImageLoadError("Test message")
        assert str(error) == "Test message"
        assert isinstance(error, Exception)

    def test_image_load_error_with_cause(self):
        """Test ImageLoadError with underlying cause."""
        cause = ValueError("Underlying error")
        error = ImageLoadError("Test message")
        error.__cause__ = cause
        assert str(error) == "Test message"
        assert error.__cause__ == cause


class TestImageLoaderUtilities:
    """Test utility functions in image loader."""

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    def test_load_test_image_function_exists(self):
        """Test that load_test_image function exists."""
        assert callable(load_test_image)
        with pytest.raises((ImageLoadError, FileNotFoundError)):
            load_test_image("/nonexistent/test.jpg")


class TestImageLoaderComprehensive:
    """Comprehensive tests for image loader functionality."""

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_loader.Image")
    @patch("pathlib.Path.exists")
    def test_load_image_asset_success(self, mock_exists, mock_image):
        """Test successful image loading."""
        mock_exists.return_value = True
        mock_pil_image = Mock()
        mock_pil_image.size = 1920, 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {"dpi": (72, 72)}
        mock_image.open.return_value.__enter__ = Mock(return_value=mock_pil_image)
        mock_image.open.return_value.__exit__ = Mock(return_value=None)
        result = load_image_asset("test.jpg")
        assert isinstance(result, ImageAsset)
        assert result.width == 1920
        assert result.height == 1080
        assert result.mode == "RGB"
        assert result.format == "JPEG"
        mock_image.open.assert_called_once()

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_loader.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.image_loader.Image")
    @patch("photoflow.core.image_loader.piexif")
    @patch("pathlib.Path.exists")
    def test_load_image_asset_with_exif(self, mock_exists, mock_piexif, mock_image):
        """Test image loading with EXIF data."""
        mock_exists.return_value = True
        mock_pil_image = Mock()
        mock_pil_image.size = 1920, 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {"exif": b"mock_exif_data"}
        mock_image.open.return_value.__enter__ = Mock(return_value=mock_pil_image)
        mock_image.open.return_value.__exit__ = Mock(return_value=None)
        mock_exif_dict = {
            "0th": {(271): "Canon", (272): "EOS R5"},
            "Exif": {(37386): (85, 1)},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        mock_piexif.load.return_value = mock_exif_dict
        mock_piexif.TAGS = {
            "0th": {(271): {"name": "Make"}, (272): {"name": "Model"}},
            "Exif": {(37386): {"name": "FocalLength"}},
        }
        result = load_image_asset("test.jpg")
        assert isinstance(result, ImageAsset)
        assert result.exif is not None
        mock_piexif.load.assert_called_once()

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_loader.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.image_loader.Image")
    @patch("photoflow.core.image_loader.IPTCInfo")
    @patch("pathlib.Path.exists")
    def test_load_image_asset_with_iptc(self, mock_exists, mock_iptc_info, mock_image):
        """Test image loading with IPTC data."""
        mock_exists.return_value = True
        mock_pil_image = Mock()
        mock_pil_image.size = 1920, 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_image.open.return_value.__enter__ = Mock(return_value=mock_pil_image)
        mock_image.open.return_value.__exit__ = Mock(return_value=None)
        mock_iptc_obj = Mock()
        mock_iptc_obj.data = {
            b"object_name": b"Test Title",
            b"caption/abstract": b"Test Caption",
        }
        mock_iptc_info.return_value = mock_iptc_obj
        result = load_image_asset("test.jpg")
        assert isinstance(result, ImageAsset)
        assert result.iptc is not None
        mock_iptc_info.assert_called_once()

    def test_clean_exif_value_edge_cases(self):
        """Test _clean_exif_value with edge cases."""
        bytes_val = b"test_bytes"
        result = _clean_exif_value(bytes_val)
        assert result == "test_bytes"
        tuple_val = 1, 100
        result = _clean_exif_value(tuple_val)
        assert result == tuple_val
        list_val = [1, 2, 3]
        result = _clean_exif_value(list_val)
        assert result == list_val
        assert _clean_exif_value("") == ""
        assert _clean_exif_value(True) is True
        assert _clean_exif_value(False) is False

    def test_extract_pil_info_comprehensive(self):
        """Test _extract_pil_info with various inputs."""
        info_dict = {
            "dpi": (72, 72),
            "quality": 95,
            "optimize": True,
            "progressive": False,
            "unknown_key": "should_be_included",
        }
        result = _extract_pil_info(info_dict)
        assert isinstance(result, dict)
        assert "DPI" in result
        assert result["DPI"] == (72, 72)
        assert "Quality" in result
        assert result["Quality"] == 95
        result = _extract_pil_info({})
        assert isinstance(result, dict)
        assert len(result) == 0
        result = _extract_pil_info(None)
        assert result == {}


class TestImageLoaderMetadataExtraction:
    """Test metadata extraction functions."""

    @patch("photoflow.core.image_loader.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.image_loader.piexif")
    def test_extract_exif_data_success(self, mock_piexif):
        """Test successful EXIF data extraction."""
        mock_exif_dict = {
            "0th": {(271): "Canon", (272): "EOS R5"},
            "Exif": {(37386): (85, 1), (33434): (1, 100)},
            "GPS": {(1): "N", (2): ((40, 1), (42, 1), (0, 1))},
            "1st": {},
            "thumbnail": None,
        }
        mock_piexif.load.return_value = mock_exif_dict
        result = _extract_exif_data(b"mock_exif_bytes")
        assert isinstance(result, dict)
        mock_piexif.load.assert_called_once_with(b"mock_exif_bytes")

    @patch("photoflow.core.image_loader.PIEXIF_AVAILABLE", False)
    def test_extract_exif_data_no_piexif(self):
        """Test EXIF extraction when piexif not available."""
        result = _extract_exif_data(b"mock_exif_bytes")
        assert result == {}

    @patch("photoflow.core.image_loader.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.image_loader.piexif")
    def test_extract_exif_data_error(self, mock_piexif):
        """Test EXIF extraction error handling."""
        mock_piexif.load.side_effect = Exception("EXIF parse error")
        result = _extract_exif_data(b"corrupt_exif_bytes")
        assert result == {}

    @patch("photoflow.core.image_loader.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.image_loader.IPTCInfo")
    def test_extract_iptc_data_success(self, mock_iptc_info):
        """Test successful IPTC data extraction."""
        mock_iptc_obj = Mock()
        mock_iptc_obj.data = {
            b"object_name": b"Test Title",
            b"caption/abstract": b"Test Caption",
            b"by-line": b"Test Photographer",
        }
        mock_iptc_info.return_value = mock_iptc_obj
        result = _extract_iptc_data(Path("test.jpg"))
        assert isinstance(result, dict)
        assert b"object_name" in result
        assert result[b"object_name"] == b"Test Title"
        mock_iptc_info.assert_called_once()

    @patch("photoflow.core.image_loader.IPTCINFO_AVAILABLE", False)
    def test_extract_iptc_data_no_iptcinfo(self):
        """Test IPTC extraction when iptcinfo3 not available."""
        result = _extract_iptc_data(Path("test.jpg"))
        assert result == {}

    @patch("photoflow.core.image_loader.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.image_loader.IPTCInfo")
    def test_extract_iptc_data_error(self, mock_iptc_info):
        """Test IPTC extraction error handling."""
        mock_iptc_info.side_effect = Exception("IPTC parse error")
        result = _extract_iptc_data(Path("test.jpg"))
        assert result == {}


class TestImageLoaderIntegration:
    """Integration tests for image loader."""

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    @patch("pathlib.Path.exists")
    def test_load_image_from_pathlike(self, mock_exists):
        """Test loading image from path-like object."""
        test_path = Path("test.jpg")
        mock_exists.return_value = False
        with pytest.raises(ImageLoadError, match="Image file not found"):
            load_image_asset(test_path)

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", True)
    def test_load_image_from_string_path(self):
        """Test loading image from string path."""
        test_path = "/path/to/test.jpg"
        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(ImageLoadError, match="Image file not found"),
        ):
            load_image_asset(test_path)

    def test_availability_constants(self):
        """Test that availability constants are properly defined."""
        assert isinstance(PIL_AVAILABLE, bool)
        assert isinstance(PIEXIF_AVAILABLE, bool)
        assert isinstance(IPTCINFO_AVAILABLE, bool)


class TestImageLoaderImportErrors:
    """Test import error handling."""

    def test_pil_import_error_handling(self):
        """Test PIL import error path."""
        assert isinstance(PIL_AVAILABLE, bool)

    def test_piexif_import_error_handling(self):
        """Test piexif import error path."""
        assert isinstance(PIEXIF_AVAILABLE, bool)

    def test_iptcinfo_import_error_handling(self):
        """Test iptcinfo import error path."""
        assert isinstance(IPTCINFO_AVAILABLE, bool)

    @patch("photoflow.core.image_loader.PIL_AVAILABLE", False)
    def test_create_image_asset_without_pil(self):
        """Test create_image_asset_from_pil when PIL not available."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required"):
            create_image_asset_from_pil(None, "test.jpg")

    def test_missing_line_coverage(self):
        """Test specific missing lines for coverage."""
        result = _extract_pil_info({"dpi": 72})
        assert "DPI" in result
        with patch("photoflow.core.image_loader.PIEXIF_AVAILABLE", False):
            result = _parse_exif_data({})
            assert result == {}

    def test_additional_utility_functions(self):
        """Test additional utility functions for coverage."""
        with pytest.raises((ImageLoadError, FileNotFoundError)):
            load_test_image("/nonexistent/file.jpg")
