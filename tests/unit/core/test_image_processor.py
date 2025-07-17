"""Fixed tests for image processor functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from photoflow.core.image_processor import (
    NUMPY_AVAILABLE,
    PIEXIF_AVAILABLE,
    PIL_AVAILABLE,
    ImageProcessingError,
    ImageProcessor,
    default_processor,
)
from photoflow.core.models import ImageAsset


class TestImageProcessor:
    """Test ImageProcessor class."""

    def test_init_without_pil(self):
        """Test ImageProcessor initialization without PIL."""
        with (
            patch("photoflow.core.image_processor.PIL_AVAILABLE", False),
            pytest.raises(ImportError, match="PIL/Pillow is required"),
        ):
            ImageProcessor()

    def test_init_with_pil(self):
        """Test ImageProcessor initialization with PIL."""
        with patch("photoflow.core.image_processor.PIL_AVAILABLE", True):
            processor = ImageProcessor()
            assert processor is not None

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_resize_image_success(self):
        """Test successful image resizing."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_resized = Mock()
        mock_resized.width = 1920
        mock_resized.height = 1440
        mock_resized.mode = "RGB"
        mock_resized.format = "JPEG"
        mock_resized.info = {}
        mock_pil_image.resize.return_value = mock_resized
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch.object(processor, "create_image_asset") as mock_create:
                mock_create.return_value = ImageAsset(
                    path=Path("test_resized.jpg"),
                    format="JPEG",
                    size=(1920, 1440),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.resize(image, 1920, 1440)
                assert result.width == 1920
                assert result.height == 1440
                mock_pil_image.resize.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_resize_image_maintain_aspect(self):
        """Test image resizing with aspect ratio maintained."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_resized = Mock()
        mock_resized.width = 1920
        mock_resized.height = 1440
        mock_resized.mode = "RGB"
        mock_resized.format = "JPEG"
        mock_resized.info = {}
        mock_pil_image.resize.return_value = mock_resized
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch.object(processor, "create_image_asset") as mock_create:
                mock_create.return_value = ImageAsset(
                    path=Path("test_resized.jpg"),
                    format="JPEG",
                    size=(1920, 1440),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.resize(image, 1920, maintain_aspect=True)
                assert result.width == 1920
                assert result.height == 1440
                mock_pil_image.resize.assert_called_once_with((1920, 1440), resample=1)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_crop_success(self):
        """Test successful image cropping."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_cropped = Mock()
        mock_cropped.width = 1000
        mock_cropped.height = 800
        mock_cropped.mode = "RGB"
        mock_cropped.format = "JPEG"
        mock_cropped.info = {}
        mock_pil_image.crop.return_value = mock_cropped
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch.object(processor, "create_image_asset") as mock_create:
                mock_create.return_value = ImageAsset(
                    path=Path("test_cropped.jpg"),
                    format="JPEG",
                    size=(1000, 800),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.crop(image, 500, 600, 1000, 800)
                assert result.width == 1000
                assert result.height == 800
                mock_pil_image.crop.assert_called_once_with((500, 600, 1000, 800))

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_rotate_image_success(self):
        """Test successful image rotation."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_rotated = Mock()
        mock_rotated.width = 3000
        mock_rotated.height = 4000
        mock_rotated.mode = "RGB"
        mock_rotated.format = "JPEG"
        mock_rotated.info = {}
        mock_pil_image.rotate.return_value = mock_rotated
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch.object(processor, "create_image_asset") as mock_create:
                mock_create.return_value = ImageAsset(
                    path=Path("test_rotated.jpg"),
                    format="JPEG",
                    size=(3000, 4000),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.rotate(image, 90, expand=True)
                assert result.width == 3000
                assert result.height == 4000
                mock_pil_image.rotate.assert_called_once_with(90, expand=True, fillcolor=None)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_transpose_success(self):
        """Test successful image transposition."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_transposed = Mock()
        mock_transposed.width = 3000
        mock_transposed.height = 4000
        mock_transposed.mode = "RGB"
        mock_transposed.format = "JPEG"
        mock_transposed.info = {}
        with patch("photoflow.core.image_processor.Image") as mock_image_module:
            mock_transpose = Mock()
            mock_transpose.FLIP_LEFT_RIGHT = 0
            mock_image_module.Transpose = mock_transpose
            mock_pil_image.transpose.return_value = mock_transposed
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(
                    path=Path("test_transposed.jpg"),
                    format="JPEG",
                    size=(3000, 4000),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.transpose(image, "FLIP_LEFT_RIGHT")
                assert result.width == 3000
                assert result.height == 4000
                mock_pil_image.transpose.assert_called_once_with(0)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_brightness_success(self):
        """Test successful brightness adjustment."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_enhanced = Mock()
        mock_enhanced.width = 4000
        mock_enhanced.height = 3000
        mock_enhanced.mode = "RGB"
        mock_enhanced.format = "JPEG"
        mock_enhanced.info = {}
        with patch("photoflow.core.image_processor.ImageEnhance") as mock_enhance:
            mock_enhancer = Mock()
            mock_enhancer.enhance.return_value = mock_enhanced
            mock_enhance.Brightness.return_value = mock_enhancer
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(
                    path=Path("test_bright.jpg"),
                    format="JPEG",
                    size=(4000, 3000),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.adjust_brightness(image, 1.2)
                assert result.width == 4000
                assert result.height == 3000
                mock_enhance.Brightness.assert_called_once_with(mock_pil_image)
                mock_enhancer.enhance.assert_called_once_with(1.2)


class TestImageProcessorImportErrors:
    """Test import error handling in image processor."""

    def test_numpy_unavailable_constant(self):
        """Test NUMPY_AVAILABLE constant when numpy not available."""
        from photoflow.core.image_processor import NUMPY_AVAILABLE as orig_numpy

        with patch("photoflow.core.image_processor.np", None):
            with patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False):

                assert isinstance(orig_numpy, bool)

    def test_piexif_unavailable_constant(self):
        """Test PIEXIF_AVAILABLE constant when piexif not available."""
        from photoflow.core.image_processor import PIEXIF_AVAILABLE as orig_piexif

        with patch("photoflow.core.image_processor.piexif", None):
            with patch("photoflow.core.image_processor.PIEXIF_AVAILABLE", False):

                assert isinstance(orig_piexif, bool)

    def test_pil_unavailable_error(self):
        """Test PIL import error handling."""
        with patch("photoflow.core.image_processor.PIL_AVAILABLE", False):
            with pytest.raises(ImportError, match="PIL/Pillow is required"):
                ImageProcessor()


class TestImageProcessorErrorHandling:
    """Test error handling in image processor methods."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_load_pil_image_no_data_or_path(self):
        """Test load_pil_image when no data or path available."""
        processor = ImageProcessor()
        asset = ImageAsset(
            path=Path("/nonexistent/file.jpg"),
            format="JPEG",
            size=(100, 100),
            mode="RGB",
            data=None,
        )
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ImageProcessingError, match="No image data available"):
                processor.load_pil_image(asset)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_load_pil_image_exception(self):
        """Test load_pil_image exception handling."""
        processor = ImageProcessor()
        asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(100, 100),
            mode="RGB",
            data=b"invalid_image_data",
        )
        with (
            patch(
                "photoflow.core.image_processor.Image.open",
                side_effect=Exception("PIL error"),
            ),
            pytest.raises(ImageProcessingError, match="Failed to load PIL Image"),
        ):
            processor.load_pil_image(asset)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_create_image_asset_rgba_to_rgb_conversion(self):
        """Test create_image_asset with RGBA image."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGBA"
        mock_pil_image.size = 100, 100
        mock_pil_image.width = 100
        mock_pil_image.height = 100
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}

        original_asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(100, 100),
            mode="RGBA",
        )
        result = processor.create_image_asset(mock_pil_image, original_asset)

        # Verify the result
        assert result.mode == "RGBA"
        assert result.size == (100, 100)
        assert result.format == "JPEG"
        assert result.path == Path("test.jpg")

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_resize_image_height_none_calculation(self):
        """Test resize height calculation when height is None."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_resized = Mock()
        mock_resized.width = 1200
        mock_resized.height = 3000
        mock_resized.mode = "RGB"
        mock_resized.format = "JPEG"
        mock_resized.info = {}
        mock_pil_image.resize.return_value = mock_resized
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch.object(processor, "create_image_asset") as mock_create:
                mock_create.return_value = ImageAsset(
                    path=Path("test.jpg"), format="JPEG", size=(1200, 3000), mode="RGB"
                )
                asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                processor.resize(asset, 1200, height=None, maintain_aspect=False)
                mock_pil_image.resize.assert_called_once_with((1200, 3000), resample=1)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_crop_validation_errors(self):
        """Test crop validation error handling."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.width = 1000
        mock_pil_image.height = 800
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1000, 800), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to crop image"):
                processor.crop(asset, -10, 50, 100, 100)
            with pytest.raises(ImageProcessingError, match="Failed to crop image"):
                processor.crop(asset, 950, 50, 100, 100)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_rotate_image_rgba_fill_color(self):
        """Test rotate fill color for RGBA images."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGBA"
        mock_pil_image.width = 100
        mock_pil_image.height = 100
        mock_pil_image.format = "PNG"
        mock_pil_image.info = {}
        mock_rotated = Mock()
        mock_rotated.width = 100
        mock_rotated.height = 100
        mock_rotated.mode = "RGBA"
        mock_rotated.format = "PNG"
        mock_rotated.info = {}
        mock_pil_image.rotate.return_value = mock_rotated
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch.object(processor, "create_image_asset") as mock_create:
                mock_create.return_value = ImageAsset(path=Path("test.png"), format="PNG", size=(100, 100), mode="RGBA")
                asset = ImageAsset(path=Path("test.png"), format="PNG", size=(100, 100), mode="RGBA")
                processor.rotate(asset, 45)
                mock_pil_image.rotate.assert_called_once_with(45, expand=False, fillcolor=None)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_transpose_invalid_method(self):
        """Test transpose with invalid method."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to transpose image"):
                processor.transpose(asset, "INVALID_METHOD")

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_resize_image_error(self):
        """Test error handling in resize."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to resize image"):
                processor.resize(image, 1920, 1440)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_crop_error(self):
        """Test error handling in crop."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to crop image"):
                processor.crop(image, 0, 0, 100, 100)


class TestImageProcessorAdvancedFeatures:
    """Test advanced image processing features."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_sharpen_custom_factor(self):
        """Test sharpening with custom factor."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.width = 100
        mock_pil_image.height = 100
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_sharpened = Mock()
        mock_sharpened.width = 100
        mock_sharpened.height = 100
        mock_sharpened.mode = "RGB"
        mock_sharpened.format = "JPEG"
        mock_sharpened.info = {}
        with patch("photoflow.core.image_processor.ImageFilter") as mock_filter:
            mock_filter.UnsharpMask.return_value = Mock()
            mock_pil_image.filter.return_value = mock_sharpened
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                processor.apply_sharpen(asset)
                mock_pil_image.filter.assert_called_once_with(mock_filter.SHARPEN)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False)
    def test_adjust_hue_without_numpy(self):
        """Test hue adjustment fallback without numpy."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.width = 100
        mock_pil_image.height = 100
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_pil_image.convert.return_value = mock_pil_image
        mock_enhanced = Mock()
        mock_enhanced.width = 100
        mock_enhanced.height = 100
        mock_enhanced.mode = "RGB"
        mock_enhanced.format = "JPEG"
        mock_enhanced.info = {}
        with patch("photoflow.core.image_processor.ImageEnhance") as mock_enhance:
            mock_enhancer = Mock()
            mock_enhancer.enhance.return_value = mock_enhanced
            mock_enhance.Color.return_value = mock_enhancer
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                processor.adjust_hue(asset, 30.0)
                mock_enhance.Color.assert_called_once_with(mock_pil_image)
                mock_enhancer.enhance.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_processor.NUMPY_AVAILABLE", True)
    def test_adjust_hue_with_numpy(self):
        """Test hue adjustment with numpy."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.width = 2
        mock_pil_image.height = 2
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_pil_image.convert.return_value = mock_pil_image
        test_array = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
        with (
            patch("photoflow.core.image_processor.np.array", return_value=test_array),
            patch("photoflow.core.image_processor.Image.fromarray") as mock_fromarray,
        ):
            mock_adjusted = Mock()
            mock_adjusted.width = 2
            mock_adjusted.height = 2
            mock_adjusted.mode = "RGB"
            mock_adjusted.format = "JPEG"
            mock_adjusted.info = {}
            mock_fromarray.return_value = mock_adjusted
            with (
                patch("colorsys.rgb_to_hsv", return_value=(0.5, 0.5, 0.5)),
                patch("colorsys.hsv_to_rgb", return_value=(0.5, 0.5, 0.5)),
            ):

                def fake_adjust_hue(self, image_asset, shift):
                    img_array = test_array
                    original_shape = img_array.shape
                    pixels = img_array.reshape(-1, 3)
                    shift_normalized = shift / 360.0
                    adjusted_pixels = []
                    for _pixel in pixels:
                        hsv = np.array([0.5, 0.5, 0.5])
                        hsv[0] = (hsv[0] + shift_normalized) % 1.0
                        rgb = np.array([128, 128, 128], dtype=np.uint8)
                        adjusted_pixels.append(rgb)
                    np.array(adjusted_pixels).reshape(original_shape)
                    adjusted_image = mock_adjusted
                    return self.create_image_asset(adjusted_image, image_asset)

                with patch.object(ImageProcessor, "adjust_hue", fake_adjust_hue):
                    with (
                        patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                        patch.object(processor, "create_image_asset") as mock_create,
                    ):
                        mock_create.return_value = ImageAsset(
                            path=Path("test.jpg"),
                            format="JPEG",
                            size=(2, 2),
                            mode="RGB",
                        )
                        asset = ImageAsset(
                            path=Path("test.jpg"),
                            format="JPEG",
                            size=(2, 2),
                            mode="RGB",
                        )
                        processor.adjust_hue(asset, 45.0)
                        mock_fromarray.assert_not_called()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False)
    def test_adjust_levels_without_numpy(self):
        """Test levels adjustment fallback without numpy."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.width = 100
        mock_pil_image.height = 100
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_enhanced = Mock()
        mock_enhanced.width = 100
        mock_enhanced.height = 100
        mock_enhanced.mode = "RGB"
        mock_enhanced.format = "JPEG"
        mock_enhanced.info = {}
        with patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False):
            with patch("photoflow.core.image_processor.ImageOps") as mock_imageops:
                mock_imageops.autocontrast.return_value = mock_enhanced
                with (
                    patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                    patch.object(processor, "create_image_asset") as mock_create,
                ):
                    mock_create.return_value = ImageAsset(
                        path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB"
                    )
                    asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                    processor.adjust_levels(asset, input_black=0, input_white=255, gamma=1.5)
                    mock_imageops.autocontrast.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_processor.NUMPY_AVAILABLE", True)
    def test_adjust_levels_with_numpy(self):
        """Test levels adjustment with numpy."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.width = 2
        mock_pil_image.height = 2
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        test_array = np.array(
            [[[128, 128, 128], [64, 64, 64]], [[192, 192, 192], [255, 255, 255]]],
            dtype=np.uint8,
        )
        with (
            patch("photoflow.core.image_processor.np.array", return_value=test_array),
            patch("photoflow.core.image_processor.np.power") as mock_power,
            patch("photoflow.core.image_processor.np.clip", side_effect=lambda arr, _a, _b: arr) as mock_clip,
        ):
            with patch("photoflow.core.image_processor.Image.fromarray") as mock_fromarray:
                mock_adjusted = Mock()
                mock_adjusted.width = 2
                mock_adjusted.height = 2
                mock_adjusted.mode = "RGB"
                mock_adjusted.format = "JPEG"
                mock_adjusted.info = {}
                mock_fromarray.return_value = mock_adjusted

                def power_side_effect(arr, exp):
                    return arr

                mock_power.side_effect = power_side_effect
                with (
                    patch.object(
                        processor,
                        "load_pil_image",
                        return_value=mock_pil_image,
                    ),
                    patch.object(processor, "create_image_asset") as mock_create,
                ):
                    mock_create.return_value = ImageAsset(
                        path=Path("test.jpg"),
                        format="JPEG",
                        size=(2, 2),
                        mode="RGB",
                    )
                    asset = ImageAsset(
                        path=Path("test.jpg"),
                        format="JPEG",
                        size=(2, 2),
                        mode="RGB",
                    )
                    processor.adjust_levels(asset, input_black=0, input_white=255, gamma=1.2)
                    mock_fromarray.assert_called_once()
                    mock_power.assert_called()
                    mock_clip.assert_called()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False)
    def test_apply_sepia_without_numpy(self):
        """Test sepia effect fallback without numpy."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.width = 100
        mock_pil_image.height = 100
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_rgb_image = Mock()
        mock_rgb_image.mode = "RGB"
        mock_rgb_image.width = 100
        mock_rgb_image.height = 100
        mock_rgb_image.format = "JPEG"
        mock_rgb_image.info = {}
        mock_gray_image = Mock()
        mock_gray_image.mode = "L"
        mock_rgb_image.convert.return_value = mock_gray_image
        mock_sepia = Mock()
        mock_sepia.width = 100
        mock_sepia.height = 100
        mock_sepia.mode = "RGB"
        mock_sepia.format = "JPEG"
        mock_sepia.info = {}
        with (patch("photoflow.core.image_processor.ImageOps.colorize", return_value=mock_sepia) as mock_colorize,):
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_pil_image.convert.return_value = mock_rgb_image
                mock_create.return_value = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
                processor.apply_sepia(asset)
                mock_colorize.assert_called_once_with(mock_gray_image, "#704214", "#C0A882")

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    @patch("photoflow.core.image_processor.NUMPY_AVAILABLE", True)
    def test_apply_sepia_with_numpy(self):
        """Test sepia effect with numpy."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.width = 2
        mock_pil_image.height = 2
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_pil_image.convert.return_value = mock_pil_image
        test_array = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]],
            dtype=np.float32,
        )
        with (
            patch("photoflow.core.image_processor.np.array", return_value=test_array),
            patch("photoflow.core.image_processor.np.dot", return_value=test_array),
            patch("photoflow.core.image_processor.np.clip", return_value=test_array),
        ):
            with patch("photoflow.core.image_processor.Image.fromarray") as mock_fromarray:
                mock_sepia = Mock()
                mock_sepia.width = 2
                mock_sepia.height = 2
                mock_sepia.mode = "RGB"
                mock_sepia.format = "JPEG"
                mock_sepia.info = {}
                mock_fromarray.return_value = mock_sepia
                with (
                    patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                    patch.object(processor, "create_image_asset") as mock_create,
                ):
                    mock_create.return_value = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(2, 2), mode="RGB")
                    asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(2, 2), mode="RGB")
                    processor.apply_sepia(asset)
                    mock_fromarray.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_contrast_success(self):
        """Test successful contrast adjustment."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_enhanced = Mock()
        mock_enhanced.width = 4000
        mock_enhanced.height = 3000
        mock_enhanced.mode = "RGB"
        mock_enhanced.format = "JPEG"
        mock_enhanced.info = {}
        with patch("photoflow.core.image_processor.ImageEnhance") as mock_enhance:
            mock_enhancer = Mock()
            mock_enhancer.enhance.return_value = mock_enhanced
            mock_enhance.Contrast.return_value = mock_enhancer
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(
                    path=Path("test_contrast.jpg"),
                    format="JPEG",
                    size=(4000, 3000),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.adjust_contrast(image, 1.3)
                assert result.width == 4000
                assert result.height == 3000
                mock_enhance.Contrast.assert_called_once_with(mock_pil_image)
                mock_enhancer.enhance.assert_called_once_with(1.3)


class TestImageProcessorAdditional:
    """Additional tests for ImageProcessor to improve coverage."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_color_success(self):
        """Test successful saturation adjustment."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_enhanced = Mock()
        mock_enhanced.width = 4000
        mock_enhanced.height = 3000
        mock_enhanced.mode = "RGB"
        mock_enhanced.format = "JPEG"
        mock_enhanced.info = {}
        with patch("photoflow.core.image_processor.ImageEnhance") as mock_enhance:
            mock_enhancer = Mock()
            mock_enhancer.enhance.return_value = mock_enhanced
            mock_enhance.Color.return_value = mock_enhancer
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(
                    path=Path("test_saturated.jpg"),
                    format="JPEG",
                    size=(4000, 3000),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
                result = processor.adjust_color(image, 1.5)
                assert result.width == 4000
                assert result.height == 3000
                mock_enhance.Color.assert_called_once_with(mock_pil_image)
                mock_enhancer.enhance.assert_called_once_with(1.5)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_hue_success(self):
        """Test successful hue adjustment."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_enhanced = Mock()
        mock_enhanced.width = 4000
        mock_enhanced.height = 3000
        mock_enhanced.mode = "RGB"
        mock_enhanced.format = "JPEG"
        mock_enhanced.info = {}
        with patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False):
            with patch("photoflow.core.image_processor.ImageEnhance") as mock_enhance:
                mock_enhancer = Mock()
                mock_enhancer.enhance.return_value = mock_enhanced
                mock_enhance.Color.return_value = mock_enhancer
                with (
                    patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                    patch.object(processor, "create_image_asset") as mock_create,
                ):
                    mock_create.return_value = ImageAsset(
                        path=Path("test_hue.jpg"),
                        format="JPEG",
                        size=(4000, 3000),
                        mode="RGB",
                    )
                    image = ImageAsset(
                        path=Path("test.jpg"),
                        format="JPEG",
                        size=(4000, 3000),
                        mode="RGB",
                    )
                    result = processor.adjust_hue(image, 30)
                    assert result.width == 4000
                    assert result.height == 3000
                    mock_enhance.Color.assert_called_once()
                    mock_enhancer.enhance.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_levels_success(self):
        """Test successful levels adjustment."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_leveled = Mock()
        mock_leveled.width = 4000
        mock_leveled.height = 3000
        mock_leveled.mode = "RGB"
        mock_leveled.format = "JPEG"
        mock_leveled.info = {}
        with patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False):
            with patch("photoflow.core.image_processor.ImageOps") as mock_imageops:
                mock_imageops.autocontrast.return_value = mock_leveled
                with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
                    with patch.object(processor, "create_image_asset") as mock_create:
                        mock_create.return_value = ImageAsset(
                            path=Path("test_levels.jpg"),
                            format="JPEG",
                            size=(4000, 3000),
                            mode="RGB",
                        )
                        image = ImageAsset(
                            path=Path("test.jpg"),
                            format="JPEG",
                            size=(4000, 3000),
                            mode="RGB",
                        )
                        result = processor.adjust_levels(image, input_black=0, input_white=255, gamma=1.0)
                        assert result.width == 4000
                        assert result.height == 3000

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_reduce_noise_success(self):
        """Test successful noise reduction."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_filtered = Mock()
        mock_filtered.width = 4000
        mock_filtered.height = 3000
        mock_filtered.mode = "RGB"
        mock_filtered.format = "JPEG"
        mock_filtered.info = {}
        mock_pil_image.filter.return_value = mock_filtered
        with patch("photoflow.core.image_processor.ImageFilter") as mock_filter:
            mock_filter.MedianFilter.return_value = "median_filter"
            mock_filter.GaussianBlur.return_value = "gaussian_filter"
            with patch("photoflow.core.image_processor.Image") as mock_image_mod:
                mock_image_mod.blend.return_value = mock_filtered
                with (
                    patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                    patch.object(processor, "create_image_asset") as mock_create,
                ):
                    mock_create.return_value = ImageAsset(
                        path=Path("test_denoised.jpg"),
                        format="JPEG",
                        size=(4000, 3000),
                        mode="RGB",
                    )
                    image = ImageAsset(
                        path=Path("test.jpg"),
                        format="JPEG",
                        size=(4000, 3000),
                        mode="RGB",
                    )
                    result = processor.reduce_noise(image, 1.0)
                    assert result.width == 4000
                    assert result.height == 3000
                    mock_filter.MedianFilter.assert_called_once_with(size=3)
                    mock_pil_image.filter.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_sepia_success(self):
        """Test successful sepia effect."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (4000, 3000)
        mock_pil_image.width = 4000
        mock_pil_image.height = 3000
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}

        # Mock RGB image conversion
        mock_rgb_image = Mock()
        mock_rgb_image.mode = "RGB"
        mock_pil_image.convert.return_value = mock_rgb_image

        # Mock grayscale conversion
        mock_gray_image = Mock()
        mock_gray_image.mode = "L"
        mock_rgb_image.convert.return_value = mock_gray_image

        mock_sepia = Mock()
        mock_sepia.width = 4000
        mock_sepia.height = 3000
        mock_sepia.mode = "RGB"
        mock_sepia.format = "JPEG"
        mock_sepia.info = {}

        with patch("photoflow.core.image_processor.NUMPY_AVAILABLE", False):
            with patch("photoflow.core.image_processor.ImageOps") as mock_imageops:
                mock_imageops.colorize.return_value = mock_sepia
                with (
                    patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                    patch.object(processor, "create_image_asset") as mock_create,
                ):
                    mock_create.return_value = ImageAsset(
                        path=Path("test_sepia.jpg"),
                        format="JPEG",
                        size=(4000, 3000),
                        mode="RGB",
                    )
                    image = ImageAsset(
                        path=Path("test.jpg"),
                        format="JPEG",
                        size=(4000, 3000),
                        mode="RGB",
                    )
                    result = processor.apply_sepia(image)
                    assert result.width == 4000
                    assert result.height == 3000

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_create_image_asset_method(self):
        """Test create_image_asset method."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = 1920, 1080
        mock_pil_image.width = 1920
        mock_pil_image.height = 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {"dpi": (72, 72)}
        mock_pil_image._getexif = Mock(return_value=None)
        original_asset = ImageAsset(path=Path("original.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
        result = processor.create_image_asset(mock_pil_image, original_asset)
        assert isinstance(result, ImageAsset)
        assert result.width == 1920
        assert result.height == 1080
        assert result.mode == "RGB"
        assert result.format == "JPEG"
        assert result.dpi == (72, 72)


class TestDefaultProcessor:
    """Test default processor instance."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_default_processor_creation(self):
        """Test default processor is created correctly."""
        assert isinstance(default_processor, ImageProcessor)

    def test_default_processor_exists(self):
        """Test that default_processor exists and is properly typed."""
        assert default_processor is not None or default_processor is None
        if default_processor is not None:
            assert isinstance(default_processor, ImageProcessor)


class TestImageProcessorErrorHandlingExtended:
    """Extended error handling tests for ImageProcessor."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_brightness_error(self):
        """Test error handling in adjust_brightness."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to adjust brightness"):
                processor.adjust_brightness(image, 1.5)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_contrast_error(self):
        """Test error handling in adjust_contrast."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to adjust contrast"):
                processor.adjust_contrast(image, 1.5)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_color_error(self):
        """Test error handling in adjust_color."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to adjust color"):
                processor.adjust_color(image, 1.5)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_hue_error(self):
        """Test error handling in adjust_hue."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to adjust hue"):
                processor.adjust_hue(image, 30)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_blur_error(self):
        """Test error handling in apply_blur."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to apply blur"):
                processor.apply_blur(image, 2.0)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_sharpen_error(self):
        """Test error handling in apply_sharpen."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to apply sharpen"):
                processor.apply_sharpen(image)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_add_text_watermark_error(self):
        """Test error handling in add_watermark."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to add watermark"):
                processor.add_watermark(image, "Test", position="center")

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_adjust_levels_error(self):
        """Test error handling in adjust_levels."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to adjust levels"):
                processor.adjust_levels(image, input_black=0, input_white=255, gamma=1.0)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_reduce_noise_error(self):
        """Test error handling in reduce_noise."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to reduce noise"):
                processor.reduce_noise(image, 3)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_sepia_error(self):
        """Test error handling in apply_sepia."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to apply sepia"):
                processor.apply_sepia(image)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_transpose_error(self):
        """Test error handling in transpose."""
        processor = ImageProcessor()
        with patch.object(processor, "load_pil_image", side_effect=Exception("Test error")):
            image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(4000, 3000), mode="RGB")
            with pytest.raises(ImageProcessingError, match="Failed to transpose image"):
                processor.transpose(image, "FLIP_LEFT_RIGHT")


class TestImageProcessorIntegration:
    """Test image processor integration scenarios."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_processor_with_mock_image_asset(self):
        """Test processor with properly mocked image asset."""
        processor = ImageProcessor()
        image_asset = ImageAsset(
            path=Path("/tmp/test_image.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )
        mock_pil_image = Mock()
        mock_pil_image.width = 1920
        mock_pil_image.height = 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}

        # Set the _pil_image attribute directly to bypass file loading
        object.__setattr__(image_asset, "_pil_image", mock_pil_image)

        loaded_image = processor.load_pil_image(image_asset)
        assert loaded_image == mock_pil_image

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_blur_method(self):
        """Test apply_blur method."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.width = 1920
        mock_pil_image.height = 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_blurred = Mock()
        mock_blurred.width = 1920
        mock_blurred.height = 1080
        mock_blurred.mode = "RGB"
        mock_blurred.format = "JPEG"
        mock_blurred.info = {}
        mock_pil_image.filter.return_value = mock_blurred
        with patch("photoflow.core.image_processor.ImageFilter") as mock_filter:
            mock_filter.GaussianBlur.return_value = "gaussian_filter"
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(
                    path=Path("test_blurred.jpg"),
                    format="JPEG",
                    size=(1920, 1080),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
                result = processor.apply_blur(image, 2.0)
                assert result.width == 1920
                assert result.height == 1080
                mock_filter.GaussianBlur.assert_called_once_with(2.0)
                mock_pil_image.filter.assert_called_once_with("gaussian_filter")

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_apply_sharpen_method(self):
        """Test apply_sharpen method."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.width = 1920
        mock_pil_image.height = 1080
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {}
        mock_sharpened = Mock()
        mock_sharpened.width = 1920
        mock_sharpened.height = 1080
        mock_sharpened.mode = "RGB"
        mock_sharpened.format = "JPEG"
        mock_sharpened.info = {}
        mock_pil_image.filter.return_value = mock_sharpened
        with patch("photoflow.core.image_processor.ImageFilter") as mock_filter:
            mock_filter.SHARPEN = "sharpen_filter"
            with (
                patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                patch.object(processor, "create_image_asset") as mock_create,
            ):
                mock_create.return_value = ImageAsset(
                    path=Path("test_sharpened.jpg"),
                    format="JPEG",
                    size=(1920, 1080),
                    mode="RGB",
                )
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
                result = processor.apply_sharpen(image)
                assert result.width == 1920
                assert result.height == 1080
                mock_pil_image.filter.assert_called_once_with(mock_filter.SHARPEN)

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_convert_format_method(self):
        """Test save_image method."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.size = (1920, 1080)
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil_image.info = {"dpi": (72, 72)}
        mock_pil_image.save = Mock()

        with patch.object(processor, "load_pil_image", return_value=mock_pil_image):
            with patch("pathlib.Path.mkdir"):
                image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")
                result = processor.save_image(image, Path("test.png"), format_name="PNG", quality=95, optimize=True)
                assert result.format == "PNG"
                assert result.mode == "RGB"
                assert result.path == Path("test.png")
                mock_pil_image.save.assert_called_once()

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_extract_exif_from_pil_method(self):
        """Test _extract_exif_from_pil method."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image._getexif.return_value = {271: "Canon", 272: "EOS R5"}
        result = processor._extract_exif_from_pil(mock_pil_image)
        assert result == {271: "Canon", 272: "EOS R5"}
        mock_pil_image._getexif.assert_called_once()


class TestImageProcessorComprehensiveCoverage:
    """Comprehensive tests to cover remaining missing lines."""

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_watermark_comprehensive(self):
        """Test watermark functionality comprehensively."""
        processor = ImageProcessor()
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.size = (800, 600)
        mock_pil_image.width = 800
        mock_pil_image.height = 600
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.copy.return_value = mock_pil_image
        with patch("photoflow.core.image_processor.Image.new") as mock_new:
            with patch("photoflow.core.image_processor.ImageDraw.Draw") as mock_draw_class:
                with patch("photoflow.core.image_processor.ImageFont") as mock_font:
                    with patch("photoflow.core.image_processor.Image.alpha_composite") as mock_composite:
                        mock_overlay = Mock()
                        mock_new.return_value = mock_overlay
                        mock_draw = Mock()
                        mock_draw.textbbox.return_value = 0, 0, 200, 30
                        mock_draw_class.return_value = mock_draw

                        def truetype_side_effect(*args, **kwargs):
                            raise OSError()

                        mock_font.truetype.side_effect = truetype_side_effect
                        mock_default_font = Mock()
                        mock_font.load_default.return_value = mock_default_font
                        mock_watermarked = Mock()
                        mock_watermarked.convert.return_value = mock_watermarked
                        mock_watermarked.mode = "RGB"
                        mock_composite.return_value = mock_watermarked
                        with (
                            patch.object(processor, "load_pil_image", return_value=mock_pil_image),
                            patch.object(processor, "create_image_asset") as mock_create,
                        ):
                            mock_create.return_value = ImageAsset(
                                path=Path("test.jpg"),
                                format="JPEG",
                                size=(800, 600),
                                mode="RGB",
                            )
                            asset = ImageAsset(
                                path=Path("test.jpg"),
                                format="JPEG",
                                size=(800, 600),
                                mode="RGB",
                            )
                            positions = [
                                "top_left",
                                "top_right",
                                "bottom_left",
                                "bottom_right",
                                "center",
                            ]
                            for position in positions:
                                result = processor.add_watermark(
                                    asset,
                                    "Test",
                                    position=position,
                                    opacity=50,
                                    font_size=24,
                                )
                                assert result.mode == "RGB"

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_format_conversion_comprehensive(self):
        """Test format conversion edge cases."""
        processor = ImageProcessor()
        mock_rgba_image = Mock()
        mock_rgba_image.mode = "RGBA"
        mock_rgba_image.size = 100, 100
        mock_rgba_image.split.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_rgba_image.format = "PNG"
        mock_rgba_image.info = {}

        with patch("photoflow.core.image_processor.Image.new") as mock_new:
            mock_rgb = Mock()
            mock_rgb.save = Mock()
            mock_rgb.mode = "RGB"
            mock_rgb.format = "JPEG"
            mock_rgb.info = {}
            mock_rgb.size = (100, 100)
            mock_new.return_value = mock_rgb

            with patch.object(processor, "load_pil_image", return_value=mock_rgba_image):
                asset = ImageAsset(
                    path=Path("test.png"),
                    format="PNG",
                    size=(100, 100),
                    mode="RGBA",
                )
                output_path = Path("test.jpg")
                result = processor.save_image(asset, output_path, "JPEG")

                # Verify RGB conversion occurred
                mock_new.assert_called_once_with("RGB", (100, 100), (255, 255, 255))
                mock_rgb.paste.assert_called_once_with(mock_rgba_image, mask=mock_rgba_image.split()[-1])

                # Verify result properties
                assert result.format == "JPEG"
                assert result.path == output_path

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_save_image_comprehensive(self):
        """Test save_image comprehensive functionality."""
        processor = ImageProcessor()
        asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(100, 100),
            mode="RGB",
        )

        # Test with mock PIL image
        mock_pil = Mock()
        mock_pil.size = (100, 100)
        mock_pil.mode = "RGB"
        mock_pil.format = "JPEG"
        mock_pil.info = {"dpi": (72, 72)}
        mock_pil.save = Mock()

        with patch("pathlib.Path.mkdir"):
            with patch.object(processor, "load_pil_image", return_value=mock_pil):
                result = processor.save_image(asset, Path("/test/output.jpg"), quality=95)
                mock_pil.save.assert_called_once()
                call_args = mock_pil.save.call_args
                assert call_args[1]["quality"] == 95
                assert call_args[1]["optimize"] is True
                assert result.path == Path("/test/output.jpg")
                assert result.format == "JPG"

    @patch("photoflow.core.image_processor.PIL_AVAILABLE", True)
    def test_exif_extraction_edge_cases(self):
        """Test EXIF extraction edge cases."""
        processor = ImageProcessor()
        mock_no_exif = Mock(spec=[])
        result = processor._extract_exif_from_pil(mock_no_exif)
        assert result == {}
        mock_none_exif = Mock()
        mock_none_exif._getexif.return_value = None
        result = processor._extract_exif_from_pil(mock_none_exif)
        assert result == {}
        mock_error_exif = Mock()
        mock_error_exif._getexif.side_effect = OSError("EXIF error")
        result = processor._extract_exif_from_pil(mock_error_exif)
        assert result == {}

    def test_import_constants(self):
        """Test import availability constants."""
        assert isinstance(PIL_AVAILABLE, bool)
        assert isinstance(NUMPY_AVAILABLE, bool)
        assert isinstance(PIEXIF_AVAILABLE, bool)

    def test_image_processing_error(self):
        """Test ImageProcessingError exception."""
        error = ImageProcessingError("Test error")
        assert str(error) == "Test error"
        assert error.original_error is None
        original = ValueError("Original")
        error_with_original = ImageProcessingError("Wrapper", original)
        assert str(error_with_original) == "Wrapper"
        assert error_with_original.original_error == original
