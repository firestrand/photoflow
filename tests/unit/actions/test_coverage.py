"""Tests to improve coverage for action modules."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from photoflow.actions.advanced import (
    HueAction,
    LevelsAction,
    NoiseReductionAction,
    SepiaAction,
)
from photoflow.actions.basic import (
    CropAction,
    ResizeAction,
    SaveAction,
    WatermarkAction,
)
from photoflow.actions.enhanced import (
    BlurAction,
    BrightnessAction,
    ContrastAction,
    RotateAction,
    SaturationAction,
    SharpenAction,
    TransposeAction,
)
from photoflow.actions.metadata import (
    CopyMetadataAction,
    ExtractMetadataAction,
    RemoveMetadataAction,
    WriteMetadataAction,
)
from photoflow.core.fields import ValidationError
from photoflow.core.models import ImageAsset


class TestBasicActionsCoverage:
    """Test basic actions for coverage."""

    def create_test_image(self):
        """Create a test image asset."""
        return ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")

    def test_resize_action_coverage(self):
        """Test ResizeAction coverage."""
        action = ResizeAction()
        assert "width" in action.fields
        assert "height" in action.fields
        assert "maintain_aspect" in action.fields
        relevant_fields = action.get_relevant_fields({"maintain_aspect": True})
        assert "width" in relevant_fields
        assert "maintain_aspect" in relevant_fields
        relevant_fields = action.get_relevant_fields({"maintain_aspect": False})
        assert "height" in relevant_fields
        valid_params = action.validate_params({"width": 800, "maintain_aspect": True})
        assert valid_params["width"] == 800
        assert valid_params["maintain_aspect"] is True
        with pytest.raises(ValidationError):
            action.validate_params({"width": -100})

    @patch("photoflow.actions.basic.default_processor")
    def test_resize_action_apply(self, mock_processor):
        """Test ResizeAction apply method."""
        action = ResizeAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.width = 800
        mock_result.height = 600
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.resize.return_value = mock_result
        result = action.apply(image, width=800, height=600)
        mock_processor.resize.assert_called_once_with(image, width=800, height=600, maintain_aspect=True)
        assert result == mock_result

    def test_crop_action_coverage(self):
        """Test CropAction coverage."""
        action = CropAction()
        assert "width" in action.fields
        assert "height" in action.fields
        assert "x" in action.fields
        assert "y" in action.fields
        valid_params = action.validate_params({"x": 100, "y": 50, "width": 800, "height": 600})
        assert valid_params["x"] == 100
        assert valid_params["y"] == 50
        assert valid_params["width"] == 800
        assert valid_params["height"] == 600

    @patch("photoflow.actions.basic.default_processor")
    def test_crop_action_apply(self, mock_processor):
        """Test CropAction apply method."""
        action = CropAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.crop.return_value = mock_result
        result = action.apply(image, width=800, height=600, x=100, y=50)
        mock_processor.crop.assert_called_once_with(image, 100, 50, 900, 650)
        assert result == mock_result

    def test_save_action_coverage(self):
        """Test SaveAction coverage."""
        action = SaveAction()
        assert "format" in action.fields
        assert "quality" in action.fields
        assert "optimize" in action.fields
        relevant_fields = action.get_relevant_fields({"format": "JPEG"})
        assert "quality" in relevant_fields
        relevant_fields = action.get_relevant_fields({"format": "PNG"})
        assert "quality" not in relevant_fields

    @patch("photoflow.actions.basic.default_processor")
    def test_save_action_apply(self, mock_processor):
        """Test SaveAction apply method."""
        action = SaveAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.save_image.return_value = mock_result
        result = action.apply(image, format="PNG", quality=95)
        mock_processor.save_image.assert_called_once_with(image, image.path, "PNG", 95)
        assert result == mock_result

    def test_watermark_action_coverage(self):
        """Test WatermarkAction coverage."""
        action = WatermarkAction()
        assert "text" in action.fields
        assert "position" in action.fields
        assert "opacity" in action.fields
        valid_params = action.validate_params({"text": "Copyright", "position": "bottom_right", "opacity": 80})
        assert valid_params["text"] == "Copyright"
        assert valid_params["position"] == "bottom_right"
        assert valid_params["opacity"] == 80

    @patch("photoflow.actions.basic.default_processor")
    @patch("photoflow.core.templates.TemplateEvaluator")
    def test_watermark_action_apply(self, mock_template_evaluator, mock_processor):
        """Test WatermarkAction apply method."""
        action = WatermarkAction()
        image = self.create_test_image()
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = "Copyright"
        mock_template_evaluator.return_value = mock_evaluator
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.add_watermark.return_value = mock_result
        result = action.apply(image, text="Copyright", position="bottom_right", opacity=80)
        mock_processor.add_watermark.assert_called_once_with(image, "Copyright", "bottom_right", 80)
        assert result == mock_result


class TestEnhancedActionsCoverage:
    """Test enhanced actions for coverage."""

    def create_test_image(self):
        """Create a test image asset."""
        return ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")

    def test_rotate_action_coverage(self):
        """Test RotateAction coverage."""
        action = RotateAction()
        assert "angle" in action.fields
        assert "expand" in action.fields
        valid_params = action.validate_params({"angle": 90, "expand": True})
        assert valid_params["angle"] == 90
        assert valid_params["expand"] is True

    @patch("photoflow.actions.enhanced.default_processor")
    def test_rotate_action_apply(self, mock_processor):
        """Test RotateAction apply method."""
        action = RotateAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.rotate.return_value = mock_result
        result = action.apply(image, angle=90, expand=True)
        mock_processor.rotate.assert_called_once_with(image, 90, True)
        assert result == mock_result

    def test_transpose_action_coverage(self):
        """Test TransposeAction coverage."""
        action = TransposeAction()
        assert "method" in action.fields
        valid_params = action.validate_params({"method": "flip_horizontal"})
        assert valid_params["method"] == "flip_horizontal"

    @patch("photoflow.actions.enhanced.default_processor")
    def test_transpose_action_apply(self, mock_processor):
        """Test TransposeAction apply method."""
        action = TransposeAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.transpose.return_value = mock_result
        result = action.apply(image, method="flip_horizontal")
        mock_processor.transpose.assert_called_once_with(image, "flip_horizontal")
        assert result == mock_result

    def test_brightness_action_coverage(self):
        """Test BrightnessAction coverage."""
        action = BrightnessAction()
        assert "factor" in action.fields
        valid_params = action.validate_params({"factor": 1.2})
        assert valid_params["factor"] == 1.2

    @patch("photoflow.actions.enhanced.default_processor")
    def test_brightness_action_apply(self, mock_processor):
        """Test BrightnessAction apply method."""
        action = BrightnessAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.adjust_brightness.return_value = mock_result
        result = action.apply(image, factor=1.2)
        mock_processor.adjust_brightness.assert_called_once_with(image, 1.2)
        assert result == mock_result

    def test_contrast_action_coverage(self):
        """Test ContrastAction coverage."""
        action = ContrastAction()
        assert "factor" in action.fields
        valid_params = action.validate_params({"factor": 1.1})
        assert valid_params["factor"] == 1.1

    @patch("photoflow.actions.enhanced.default_processor")
    def test_contrast_action_apply(self, mock_processor):
        """Test ContrastAction apply method."""
        action = ContrastAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.adjust_contrast.return_value = mock_result
        result = action.apply(image, factor=1.1)
        mock_processor.adjust_contrast.assert_called_once_with(image, 1.1)
        assert result == mock_result

    def test_saturation_action_coverage(self):
        """Test SaturationAction coverage."""
        action = SaturationAction()
        assert "factor" in action.fields
        valid_params = action.validate_params({"factor": 1.3})
        assert valid_params["factor"] == 1.3

    @patch("photoflow.actions.enhanced.default_processor")
    def test_saturation_action_apply(self, mock_processor):
        """Test SaturationAction apply method."""
        action = SaturationAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.adjust_color.return_value = mock_result
        result = action.apply(image, factor=1.3)
        mock_processor.adjust_color.assert_called_once_with(image, 1.3)
        assert result == mock_result

    def test_blur_action_coverage(self):
        """Test BlurAction coverage."""
        action = BlurAction()
        assert "radius" in action.fields
        valid_params = action.validate_params({"radius": 2.0})
        assert valid_params["radius"] == 2.0

    @patch("photoflow.actions.enhanced.default_processor")
    def test_blur_action_apply(self, mock_processor):
        """Test BlurAction apply method."""
        action = BlurAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.apply_blur.return_value = mock_result
        result = action.apply(image, radius=2.0)
        mock_processor.apply_blur.assert_called_once_with(image, 2.0)
        assert result == mock_result

    def test_sharpen_action_coverage(self):
        """Test SharpenAction coverage."""
        action = SharpenAction()
        assert "factor" in action.fields
        valid_params = action.validate_params({"factor": 1.5})
        assert valid_params["factor"] == 1.5

    @patch("photoflow.actions.enhanced.default_processor")
    def test_sharpen_action_apply(self, mock_processor):
        """Test SharpenAction apply method."""
        action = SharpenAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.apply_sharpen.return_value = mock_result
        result = action.apply(image, factor=1.5)
        mock_processor.apply_sharpen.assert_called_once_with(image)
        assert result == mock_result


class TestAdvancedActionsCoverage:
    """Test advanced actions for coverage."""

    def create_test_image(self):
        """Create a test image asset."""
        return ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")

    def test_hue_action_coverage(self):
        """Test HueAction coverage."""
        action = HueAction()
        assert "shift" in action.fields
        valid_params = action.validate_params({"shift": 30})
        assert valid_params["shift"] == 30

    @patch("photoflow.actions.advanced.default_processor")
    def test_hue_action_apply(self, mock_processor):
        """Test HueAction apply method."""
        action = HueAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.adjust_hue.return_value = mock_result
        result = action.apply(image, shift=30)
        mock_processor.adjust_hue.assert_called_once_with(image, 30)
        assert result == mock_result

    def test_levels_action_coverage(self):
        """Test LevelsAction coverage."""
        action = LevelsAction()
        assert "shadows" in action.fields
        assert "midtones" in action.fields
        assert "highlights" in action.fields
        valid_params = action.validate_params({"shadows": 0.1, "midtones": 1.0, "highlights": 0.9})
        assert valid_params["shadows"] == 0.1
        assert valid_params["midtones"] == 1.0
        assert valid_params["highlights"] == 0.9

    @patch("photoflow.actions.advanced.default_processor")
    def test_levels_action_apply(self, mock_processor):
        """Test LevelsAction apply method."""
        action = LevelsAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.adjust_levels.return_value = mock_result
        result = action.apply(image, shadows=0.1, midtones=1.0, highlights=0.9)
        mock_processor.adjust_levels.assert_called_once_with(image, 0.1, 1.0, 0.9)
        assert result == mock_result

    def test_sepia_action_coverage(self):
        """Test SepiaAction coverage."""
        action = SepiaAction()
        assert "intensity" in action.fields
        valid_params = action.validate_params({"intensity": 0.8})
        assert valid_params["intensity"] == 0.8

    @patch("photoflow.actions.advanced.default_processor")
    def test_sepia_action_apply(self, mock_processor):
        """Test SepiaAction apply method."""
        action = SepiaAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.apply_sepia.return_value = mock_result
        result = action.apply(image, intensity=0.8)
        mock_processor.apply_sepia.assert_called_once_with(image)
        assert result == mock_result

    def test_noise_reduction_action_coverage(self):
        """Test NoiseReductionAction coverage."""
        action = NoiseReductionAction()
        assert "strength" in action.fields
        valid_params = action.validate_params({"strength": 0.5})
        assert valid_params["strength"] == 0.5

    @patch("photoflow.actions.advanced.default_processor")
    def test_noise_reduction_action_apply(self, mock_processor):
        """Test NoiseReductionAction apply method."""
        action = NoiseReductionAction()
        image = self.create_test_image()
        mock_result = Mock()
        mock_result.processing_metadata = {}
        mock_result.with_updates.return_value = mock_result
        mock_processor.reduce_noise.return_value = mock_result
        result = action.apply(image, strength=0.5)
        mock_processor.reduce_noise.assert_called_once_with(image, 0.5)
        assert result == mock_result


class TestActionErrorHandling:
    """Test action error handling."""

    def create_test_image(self):
        """Create a test image asset."""
        return ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")

    @patch("photoflow.actions.basic.default_processor")
    def test_resize_action_error(self, mock_processor):
        """Test ResizeAction error handling."""
        action = ResizeAction()
        image = self.create_test_image()
        mock_processor.resize.side_effect = Exception("Processing error")
        with pytest.raises(Exception):  # noqa: B017
            action.apply(image, width=800)

    @patch("photoflow.actions.enhanced.default_processor")
    def test_brightness_action_error(self, mock_processor):
        """Test BrightnessAction error handling."""
        action = BrightnessAction()
        image = self.create_test_image()
        mock_processor.adjust_brightness.side_effect = Exception("Processing error")
        with pytest.raises(Exception):  # noqa: B017
            action.apply(image, factor=1.5)

    @patch("photoflow.actions.advanced.default_processor")
    def test_sepia_action_error(self, mock_processor):
        """Test SepiaAction error handling."""
        action = SepiaAction()
        image = self.create_test_image()
        mock_processor.apply_sepia.side_effect = Exception("Processing error")
        with pytest.raises(Exception):  # noqa: B017
            action.apply(image, intensity=0.8)


class TestActionMetadata:
    """Test action metadata properties."""

    def test_basic_actions_metadata(self):
        """Test basic actions metadata."""
        resize = ResizeAction()
        assert resize.name == "resize"
        assert resize.label == "Resize Image"
        assert "geometry" in resize.tags
        crop = CropAction()
        assert crop.name == "crop"
        assert crop.label == "Crop Image"
        assert "geometry" in crop.tags
        save = SaveAction()
        assert save.name == "save"
        assert save.label == "Save Image"
        assert "output" in save.tags
        watermark = WatermarkAction()
        assert watermark.name == "watermark"
        assert watermark.label == "Add Watermark"
        assert "effect" in watermark.tags

    def test_enhanced_actions_metadata(self):
        """Test enhanced actions metadata."""
        rotate = RotateAction()
        assert rotate.name == "rotate"
        assert rotate.label == "Rotate Image"
        assert "geometry" in rotate.tags
        brightness = BrightnessAction()
        assert brightness.name == "brightness"
        assert brightness.label == "Adjust Brightness"
        assert "color" in brightness.tags
        blur = BlurAction()
        assert blur.name == "blur"
        assert blur.label == "Blur Image"
        assert "effect" in blur.tags

    def test_advanced_actions_metadata(self):
        """Test advanced actions metadata."""
        hue = HueAction()
        assert hue.name == "hue"
        assert hue.label == "Adjust Hue"
        assert "color" in hue.tags
        levels = LevelsAction()
        assert levels.name == "levels"
        assert levels.label == "Adjust Levels"
        assert "color" in levels.tags
        sepia = SepiaAction()
        assert sepia.name == "sepia"
        assert sepia.label == "Sepia Tone"
        assert "effect" in sepia.tags
        noise = NoiseReductionAction()
        assert noise.name == "noise_reduction"
        assert noise.label == "Noise Reduction"
        assert "enhancement" in noise.tags


class TestMetadataActionsCoverage:
    """Test metadata actions for coverage."""

    def create_test_image(self):
        """Create a test image asset."""
        return ImageAsset(path=Path("test.jpg"), format="JPEG", size=(1920, 1080), mode="RGB")

    def test_write_metadata_action_coverage(self):
        """Test WriteMetadataAction coverage."""
        action = WriteMetadataAction()
        assert "title" in action.fields
        assert "caption" in action.fields
        assert "creator" in action.fields
        assert "copyright" in action.fields
        valid_params = action.validate_params(
            {
                "title": "Test Image",
                "caption": "Test Caption",
                "creator": "Test Creator",
            }
        )
        assert valid_params["title"] == "Test Image"
        assert valid_params["caption"] == "Test Caption"
        assert valid_params["creator"] == "Test Creator"

    @patch("photoflow.actions.metadata.write_metadata")
    def test_write_metadata_action_apply(self, mock_write_metadata):
        """Test WriteMetadataAction apply method."""
        action = WriteMetadataAction()
        image = self.create_test_image()
        result = action.apply(image, title="Test Image", caption="Test Caption")
        mock_write_metadata.assert_called_once()
        assert result is not image
        assert result.path == image.path

    def test_remove_metadata_action_coverage(self):
        """Test RemoveMetadataAction coverage."""
        action = RemoveMetadataAction()
        assert "metadata_type" in action.fields
        assert "preserve_basic" in action.fields
        valid_params = action.validate_params({"metadata_type": "exif", "preserve_basic": True})
        assert valid_params["metadata_type"] == "exif"
        assert valid_params["preserve_basic"] is True

    @patch("photoflow.actions.metadata.write_metadata")
    def test_remove_metadata_action_apply(self, mock_write_metadata):
        """Test RemoveMetadataAction apply method."""
        action = RemoveMetadataAction()
        image = self.create_test_image()
        result = action.apply(image, metadata_type="exif", preserve_basic=True)
        mock_write_metadata.assert_called_once()
        assert result is not image

    def test_copy_metadata_action_coverage(self):
        """Test CopyMetadataAction coverage."""
        action = CopyMetadataAction()
        assert "source_image" in action.fields
        assert "metadata_type" in action.fields
        assert "overwrite" in action.fields
        valid_params = action.validate_params(
            {
                "source_image": "/path/to/source.jpg",
                "metadata_type": "exif",
                "overwrite": True,
            }
        )
        assert valid_params["source_image"] == "/path/to/source.jpg"
        assert valid_params["metadata_type"] == "exif"
        assert valid_params["overwrite"] is True

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("photoflow.actions.metadata.write_metadata")
    @patch("pathlib.Path.exists")
    def test_copy_metadata_action_apply(self, mock_exists, mock_writer_class, mock_reader_class):
        """Test CopyMetadataAction apply method."""
        action = CopyMetadataAction()
        image = self.create_test_image()
        mock_exists.return_value = True
        mock_reader_class.return_value = {"Make": "Canon"}, {"Caption": "Test"}
        result = action.apply(image, source_image="/path/to/source.jpg", metadata_type="exif")
        assert mock_reader_class.call_count == 2
        mock_writer_class.assert_called_once()
        assert result is not image

    def test_extract_metadata_action_coverage(self):
        """Test ExtractMetadataAction coverage."""
        action = ExtractMetadataAction()
        assert "output_format" in action.fields
        assert "output_path" in action.fields
        assert "include_raw" in action.fields
        valid_params = action.validate_params(
            {
                "output_format": "json",
                "output_path": "/path/to/output.json",
                "include_raw": True,
            }
        )
        assert valid_params["output_format"] == "json"
        assert valid_params["output_path"] == "/path/to/output.json"
        assert valid_params["include_raw"] is True

    @patch("photoflow.actions.metadata.read_metadata")
    @patch("builtins.open", new_callable=mock_open)
    def test_extract_metadata_action_apply(self, mock_file, mock_reader_class):
        """Test ExtractMetadataAction apply method."""
        action = ExtractMetadataAction()
        image = self.create_test_image()
        mock_exif = Mock()
        mock_exif.to_dict.return_value = {"Make": "Canon", "Model": "EOS"}
        mock_iptc = Mock()
        mock_iptc.to_dict.return_value = {"Caption": "Test image"}
        mock_reader_class.return_value = mock_exif, mock_iptc
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        result = action.apply(image, output_format="json", output_path=tmp_path)
        mock_reader_class.assert_called_once()
        assert result is not image
        Path(tmp_path).unlink(missing_ok=True)


class TestActionFieldSchemas:
    """Test action field schemas."""

    def test_resize_field_schema(self):
        """Test ResizeAction field schema."""
        action = ResizeAction()
        schema = action.get_field_schema()
        assert "width" in schema
        assert schema["width"]["type"] == "integer"
        assert schema["width"]["required"] is True
        assert "height" in schema
        assert schema["height"]["type"] == "integer"
        assert schema["height"]["required"] is False
        assert "maintain_aspect" in schema
        assert schema["maintain_aspect"]["type"] == "boolean"

    def test_crop_field_schema(self):
        """Test CropAction field schema."""
        action = CropAction()
        schema = action.get_field_schema()
        assert "width" in schema
        assert "height" in schema
        assert "x" in schema
        assert "y" in schema

    def test_watermark_field_schema(self):
        """Test WatermarkAction field schema."""
        action = WatermarkAction()
        schema = action.get_field_schema()
        assert "text" in schema
        assert schema["text"]["type"] == "string"
        assert schema["text"]["required"] is True
        assert "position" in schema
        assert schema["position"]["type"] == "choice"
        assert "opacity" in schema
        assert schema["opacity"]["type"] == "integer"
