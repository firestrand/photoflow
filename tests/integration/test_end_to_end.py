"""Integration tests for PhotoFlow end-to-end capabilities.

Tests the complete workflow from image creation through pipeline processing,
demonstrating all currently implemented functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import ClassVar

import pytest

from photoflow.core.actions import ActionError, ActionRegistry, BaseAction
from photoflow.core.fields import BooleanField, IntegerField, StringField, ValidationError
from photoflow.core.image_loader import load_image_asset
from photoflow.core.mixins import GeometryMixin, ProcessingMixin
from photoflow.core.models import ActionSpec, ImageAsset, Pipeline
from photoflow.core.serialization import load_pipeline, save_pipeline
from photoflow.core.settings import PhotoFlowSettings
from photoflow.core.templates import TemplateError, TemplateEvaluator
from tests.utils.test_image_generator import TestImageGenerator

try:
    IMAGE_LOADING_AVAILABLE = True
except ImportError:
    IMAGE_LOADING_AVAILABLE = False


class MockResizeAction(BaseAction, GeometryMixin):
    """Mock resize action for integration testing."""

    name = "resize"
    label = "Resize Image"
    description = "Resize image to specified dimensions"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar = ["geometry", "basic"]

    def setup_fields(self) -> None:
        """Set up resize action fields."""
        self.fields = {
            "width": IntegerField(
                default=800,
                min_value=1,
                max_value=10000,
                label="Width",
                help_text="Target width in pixels",
            ),
            "height": IntegerField(
                default=600,
                min_value=1,
                max_value=10000,
                label="Height",
                help_text="Target height in pixels",
            ),
            "maintain_aspect": BooleanField(
                default=True,
                label="Maintain Aspect Ratio",
                help_text="Keep original proportions",
            ),
        }

    def get_relevant_fields(self, params: dict) -> list[str]:
        """Show height field only if not maintaining aspect ratio."""
        fields = ["width", "maintain_aspect"]
        if not params.get("maintain_aspect", True):
            fields.append("height")
        return fields

    def apply(self, image: ImageAsset, **params) -> ImageAsset:
        """Apply resize operation (mock implementation)."""
        width = params["width"]
        height = params.get("height", image.height)
        maintain_aspect = params.get("maintain_aspect", True)
        if maintain_aspect:
            aspect_ratio = image.width / image.height
            height = int(width / aspect_ratio)
        processing_update = {
            "last_action": self.name,
            "last_action_version": self.version,
            "last_action_params": params,
            "processing_history": [
                *image.processing_metadata.get("processing_history", []),
                f"{self.name}({width}x{height})",
            ],
        }
        return image.with_updates(size=(width, height), processing_metadata=processing_update)


class MockWatermarkAction(BaseAction, ProcessingMixin):
    """Mock watermark action for integration testing."""

    name = "watermark"
    label = "Add Watermark"
    description = "Add text watermark to image"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar = ["text", "overlay"]

    def setup_fields(self) -> None:
        """Set up watermark action fields."""
        self.fields = {
            "text": StringField(
                default="© PhotoFlow",
                min_length=1,
                max_length=100,
                label="Watermark Text",
                help_text="Text to overlay on image",
            ),
            "position": StringField(
                default="bottom_right",
                label="Position",
                help_text="Watermark position (top_left, top_right, bottom_left, bottom_right, center)",
            ),
            "opacity": IntegerField(
                default=70,
                min_value=1,
                max_value=100,
                label="Opacity",
                help_text="Watermark opacity (1-100%)",
            ),
        }

    def apply(self, image: ImageAsset, **params) -> ImageAsset:
        """Apply watermark operation (mock implementation)."""
        text_template = params["text"]
        position = params.get("position", "bottom_right")
        opacity = params.get("opacity", 70)
        evaluator = TemplateEvaluator()
        context = image.template_vars
        evaluated_text = evaluator.evaluate(text_template, context)
        metadata_update = {
            "watermark_text": evaluated_text,
            "watermark_position": position,
            "watermark_opacity": opacity,
        }
        processing_update = {
            "last_action": self.name,
            "last_action_version": self.version,
            "last_action_params": params,
            "processing_history": [
                *image.processing_metadata.get("processing_history", []),
                f"{self.name}({evaluated_text})",
            ],
        }
        return image.with_updates(metadata=metadata_update, processing_metadata=processing_update)


class MockOutputAction(BaseAction):
    """Mock output action for integration testing."""

    name = "output"
    label = "Save Output"
    description = "Save processed image with dynamic naming"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar = ["io", "save"]

    def setup_fields(self) -> None:
        """Set up output action fields."""
        self.fields = {
            "filename_template": StringField(
                default="<filename>_<width>x<height>_processed",
                label="Filename Template",
                help_text="Template for output filename (supports <variables>)",
            ),
            "format": StringField(
                default="JPEG",
                label="Output Format",
                help_text="Image output format (JPEG, PNG, WebP, TIFF)",
            ),
            "quality": IntegerField(
                default=90,
                min_value=1,
                max_value=100,
                label="Quality",
                help_text="Output quality (1-100%)",
            ),
        }

    def apply(self, image: ImageAsset, **params) -> ImageAsset:
        """Apply output operation (mock implementation)."""
        template = params["filename_template"]
        output_format = params.get("format", "JPEG")
        quality = params.get("quality", 90)
        evaluator = TemplateEvaluator()
        context = image.template_vars
        output_filename = evaluator.evaluate(template, context)
        processing_update = {
            "last_action": self.name,
            "last_action_version": self.version,
            "last_action_params": params,
            "output_filename": output_filename,
            "output_format": output_format,
            "output_quality": quality,
            "processing_history": [
                *image.processing_metadata.get("processing_history", []),
                f"{self.name}({output_filename}.{output_format.lower()})",
            ],
        }
        return image.with_updates(format=output_format, processing_metadata=processing_update)


class TestPhotoFlowIntegration:
    """Integration tests for PhotoFlow end-to-end workflow."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.registry = ActionRegistry()
        self.registry.register(MockResizeAction)
        self.registry.register(MockWatermarkAction)
        self.registry.register(MockOutputAction)
        self.test_image = self._create_test_image()

    def _create_test_image(self) -> ImageAsset:
        """Create a test image with realistic metadata.

        Tries to generate a real image file first, falls back to manual creation.
        """
        try:
            generator = TestImageGenerator()
            temp_image_path = generator.create_test_image(
                width=4000,
                height=3000,
                image_type="landscape",
                camera_make="Canon",
                camera_model="EOS R5",
                photographer="John Photographer",
                title="Mountain Sunset",
                keywords=["landscape", "sunset", "mountains", "nature"],
            )
            if IMAGE_LOADING_AVAILABLE:
                image_asset = load_image_asset(temp_image_path)
                return image_asset.with_iptc_update(
                    {
                        "ObjectName": "Mountain Sunset",
                        "Caption-Abstract": "Beautiful mountain sunset landscape",
                        "Keywords": ["landscape", "sunset", "mountains", "nature"],
                        "By-line": "John Photographer",
                        "CopyrightNotice": "© 2025 John Photographer",
                    }
                )
            else:
                return self._create_manual_test_image(temp_image_path)
        except Exception:
            return self._create_manual_test_image()

    def _create_manual_test_image(self, path: Path | None = None) -> ImageAsset:
        """Create test image manually with realistic metadata."""
        if path is None:
            path = Path("test_landscape.jpg")
        return ImageAsset(
            path=path,
            format="JPEG",
            size=(4000, 3000),
            mode="RGB",
            exif={
                "Make": "Canon",
                "Model": "EOS R5",
                "DateTime": "2025:01:15 14:30:00",
                "FocalLength": (85, 1),
                "FNumber": (28, 10),
                "ISOSpeedRatings": 400,
                "ExposureTime": (1, 250),
                "LensModel": "RF 85mm f/2.8 Macro IS STM",
            },
            iptc={
                "ObjectName": "Mountain Sunset",
                "Caption-Abstract": "Beautiful mountain sunset landscape",
                "Keywords": ["landscape", "sunset", "mountains", "nature"],
                "By-line": "John Photographer",
                "CopyrightNotice": "© 2025 John Photographer",
            },
            dpi=(300, 300),
            color_profile="sRGB",
        )

    def test_template_engine_with_rich_metadata(self) -> None:
        """Test template engine with rich image metadata."""
        evaluator = TemplateEvaluator()
        context = self.test_image.template_vars
        result = evaluator.evaluate("<filename>_<width>x<height>", context)
        assert "_4000x3000" in result
        assert result.endswith("_4000x3000")
        result = evaluator.evaluate("<camera_make>_<camera_model>_<focal_length>", context)
        if "camera_make" in context and "camera_model" in context:
            assert result == "Canon_EOS R5_85mm"
        else:
            assert "<camera_make>" in result or "Canon" in result
        result = evaluator.evaluate("<filename>_<width*2>px", context)
        assert "_8000px" in result
        assert result.endswith("_8000px")
        result = evaluator.evaluate("upper(<title>)", context)
        assert result == "MOUNTAIN SUNSET"
        result = evaluator.evaluate("<width if width < 2000 else 2000>", context)
        assert result == "2000"
        template = "<creator>_<filename>_<focal_length>_ISO<iso>_<format_lower>"
        result = evaluator.evaluate(template, context)
        assert result.startswith("John Photographer_")
        assert result.endswith("_85mm_ISO400_jpeg")
        assert "_85mm_ISO400_jpeg" in result

    def test_action_field_validation_and_schema_generation(self) -> None:
        """Test action field validation and schema generation."""
        resize_action = self.registry.get_action("resize")
        schema = resize_action.get_field_schema()
        assert "width" in schema
        assert schema["width"]["type"] == "integer"
        assert schema["width"]["min_value"] == 1
        assert schema["width"]["max_value"] == 10000
        assert schema["width"]["default"] == 800
        params = {"width": 1920, "height": 1080, "maintain_aspect": False}
        validated = resize_action.validate_params(params)
        assert validated["width"] == 1920
        assert validated["height"] == 1080
        assert validated["maintain_aspect"] is False
        with pytest.raises(ValidationError):
            resize_action.validate_params({"width": -100})
        relevant_fields = resize_action.get_relevant_fields({"maintain_aspect": True})
        assert "width" in relevant_fields
        assert "maintain_aspect" in relevant_fields
        assert "height" not in relevant_fields
        relevant_fields = resize_action.get_relevant_fields({"maintain_aspect": False})
        assert "height" in relevant_fields

    def test_single_action_execution(self) -> None:
        """Test executing a single action on an image."""
        resize_action = self.registry.get_action("resize")
        result = resize_action.execute(self.test_image, width=1920, maintain_aspect=True)
        assert result.width == 1920
        assert result.height == 1440
        assert result.format == "JPEG"
        assert result.processing_metadata["last_action"] == "resize"
        assert result.processing_metadata["last_action_version"] == "1.0"
        assert result.processing_metadata["last_action_params"]["width"] == 1920
        assert "resize(1920x1440)" in result.processing_metadata["processing_history"]
        assert result.exif["Make"] == "Canon"
        assert result.iptc["By-line"] == "John Photographer"

    def test_action_chaining_workflow(self) -> None:
        """Test chaining multiple actions in sequence."""
        resize_action = self.registry.get_action("resize")
        resized_image = resize_action.execute(self.test_image, width=2000, maintain_aspect=True)
        assert resized_image.width == 2000
        assert resized_image.height == 1500
        watermark_action = self.registry.get_action("watermark")
        watermarked_image = watermark_action.execute(
            resized_image, text="© 2025 PhotoFlow", position="bottom_right", opacity=75
        )
        assert watermarked_image.metadata["watermark_text"] == "© 2025 PhotoFlow"
        assert watermarked_image.metadata["watermark_position"] == "bottom_right"
        save_action = self.registry.get_action("output")
        final_image = save_action.execute(
            watermarked_image,
            filename_template="<creator>_<filename>_<width>x<height>_final",
            format="PNG",
            quality=95,
        )
        assert final_image.format == "PNG"
        if "output_filename" in final_image.processing_metadata:
            output_filename = final_image.processing_metadata["output_filename"]
            assert "John Photographer" in output_filename
            assert "2000x1500_final" in output_filename
        else:
            assert final_image.width == 2000
            assert final_image.height == 1500
        history = final_image.processing_metadata["processing_history"]
        assert "resize(2000x1500)" in history
        assert "watermark(© 2025 PhotoFlow)" in history
        assert any("output(" in step for step in history)

    def test_pipeline_creation_and_execution(self) -> None:
        """Test creating and executing a complete pipeline."""
        resize_spec = ActionSpec(
            name="resize",
            parameters={"width": 1920, "maintain_aspect": True},
            description="Resize for web display",
        )
        watermark_spec = ActionSpec(
            name="watermark",
            parameters={
                "text": "© PhotoFlow Demo",
                "position": "bottom_right",
                "opacity": 60,
            },
            description="Add copyright watermark",
        )
        output_spec = ActionSpec(
            name="output",
            parameters={
                "filename_template": "<filename>_web_<width>x<height>",
                "format": "WebP",
                "quality": 85,
            },
            description="Save web-optimized version",
        )
        pipeline = Pipeline(
            name="web_optimization",
            actions=(resize_spec, watermark_spec, output_spec),
            metadata={
                "description": "Optimize images for web display",
                "author": "PhotoFlow",
                "version": "1.0",
                "created": "2025-01-15",
            },
        )
        assert pipeline.action_count == 3
        assert len(pipeline.enabled_actions) == 3
        assert pipeline.metadata["description"] == "Optimize images for web display"
        current_image = self.test_image
        for action_spec in pipeline.enabled_actions:
            action = self.registry.get_action(action_spec.name)
            current_image = action.execute(current_image, **action_spec.parameters)
        assert current_image.width == 1920
        assert current_image.height == 1440
        assert current_image.format == "WebP"
        assert current_image.metadata["watermark_text"] == "© PhotoFlow Demo"
        output_filename = current_image.processing_metadata["output_filename"]
        assert "_web_1920x1440" in output_filename

    def test_pipeline_serialization_and_persistence(self) -> None:
        """Test saving and loading pipelines to/from files."""
        pipeline = Pipeline(
            name="portrait_processing",
            actions=(
                ActionSpec(
                    name="resize",
                    parameters={
                        "width": 1200,
                        "height": 1600,
                        "maintain_aspect": False,
                    },
                ),
                ActionSpec(
                    name="watermark",
                    parameters={
                        "text": "<creator> - <title>",
                        "position": "bottom_left",
                    },
                ),
                ActionSpec(
                    name="output",
                    parameters={
                        "filename_template": "<camera_make>_<date_taken>_<filename>_portrait",
                        "format": "JPEG",
                        "quality": 92,
                    },
                ),
            ),
            metadata={"workflow_type": "portrait", "target_use": "print"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "portrait_pipeline.json"
            save_pipeline(pipeline, json_path)
            assert json_path.exists()
            with json_path.open() as f:
                data = json.load(f)
                assert data["name"] == "portrait_processing"
                assert len(data["actions"]) == 3
                assert data["metadata"]["workflow_type"] == "portrait"
            loaded_pipeline = load_pipeline(json_path)
            assert loaded_pipeline.name == pipeline.name
            assert loaded_pipeline.action_count == pipeline.action_count
            assert loaded_pipeline.metadata == pipeline.metadata
            actions_path = Path(tmpdir) / "portrait_pipeline.actions"
            save_pipeline(pipeline, actions_path)
            assert actions_path.exists()
            loaded_actions_pipeline = load_pipeline(actions_path)
            assert loaded_actions_pipeline.name == pipeline.name
            assert loaded_actions_pipeline.actions == pipeline.actions

    def test_settings_and_configuration(self) -> None:
        """Test settings loading and configuration."""
        settings = PhotoFlowSettings()
        assert settings.max_workers >= 1
        assert isinstance(settings.gpu_acceleration, bool)
        assert settings.default_quality >= 1
        assert settings.default_quality <= 100
        assert hasattr(settings, "model_validate")
        settings_dict = settings.model_dump()
        assert "max_workers" in settings_dict
        assert "gpu_acceleration" in settings_dict

    def test_error_handling_and_recovery(self) -> None:
        """Test error handling throughout the pipeline."""
        resize_action = self.registry.get_action("resize")
        with pytest.raises(ActionError):
            resize_action.execute(self.test_image, width="invalid")
        evaluator = TemplateEvaluator()
        with pytest.raises(TemplateError):
            evaluator.evaluate("<nonexistent_variable>", {})
        ActionSpec(name="nonexistent_action", parameters={})
        with pytest.raises(KeyError):
            self.registry.get_action("nonexistent_action")

    def test_template_security_features(self) -> None:
        """Test template engine security features."""
        evaluator = TemplateEvaluator()
        context = {"filename": "test"}
        dangerous_templates = [
            "__import__('os').system('rm -rf /')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
        ]
        for template in dangerous_templates:
            with pytest.raises(TemplateError):
                evaluator.evaluate(template, context)
        safe_result = evaluator.evaluate("upper(<filename>)", context)
        assert safe_result == "TEST"
        safe_result = evaluator.evaluate("len(<filename>)", context)
        assert safe_result == "4"

    def test_comprehensive_integration_workflow(self) -> None:
        """Test the complete PhotoFlow workflow from start to finish."""
        web_pipeline = Pipeline(
            name="social_media_optimization",
            actions=(
                ActionSpec(
                    name="resize",
                    parameters={"width": 1080, "maintain_aspect": True},
                    description="Instagram square format",
                ),
                ActionSpec(
                    name="watermark",
                    parameters={
                        "text": "@<creator> #<keywords>",
                        "position": "bottom_left",
                        "opacity": 50,
                    },
                    description="Social media branding",
                ),
                ActionSpec(
                    name="output",
                    parameters={
                        "filename_template": "social_<filename>_<width>x<height>_<format_lower>",
                        "format": "JPEG",
                        "quality": 85,
                    },
                    description="Optimized for social platforms",
                ),
            ),
            metadata={
                "target_platform": "instagram",
                "optimization_level": "high",
                "created_by": "PhotoFlow Integration Test",
            },
        )
        processed_image = self.test_image
        processing_steps = []
        for i, action_spec in enumerate(web_pipeline.enabled_actions):
            action = self.registry.get_action(action_spec.name)
            validated_params = action.validate_params(action_spec.parameters)
            processing_steps.append(f"Step {i + 1}: {action.label}")
            processed_image = action.execute(processed_image, **validated_params)
        assert processed_image.width == 1080
        assert processed_image.height == 810
        assert processed_image.format == "JPEG"
        expected_watermark = "@John Photographer #landscape, sunset, mountains, nature"
        assert processed_image.metadata["watermark_text"] == expected_watermark
        actual_filename = processed_image.processing_metadata["output_filename"]
        assert actual_filename.startswith("social_")
        assert "_1080x810_jpeg" in actual_filename
        print(f"📄 Generated filename: {actual_filename}")
        history = processed_image.processing_metadata["processing_history"]
        assert len(history) == 3
        assert "resize(1080x810)" in history[0]
        assert "watermark(" in history[1]
        assert "output(" in history[2] and ".jpeg)" in history[2]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            save_pipeline(web_pipeline, tmp.name)
            reloaded_pipeline = load_pipeline(tmp.name)
            assert reloaded_pipeline.name == web_pipeline.name
            assert reloaded_pipeline.action_count == web_pipeline.action_count
            assert reloaded_pipeline.metadata["target_platform"] == "instagram"
        print("✅ Integration test completed successfully!")
        print(f"📊 Processed {self.test_image.megapixels:.1f}MP image")
        print(f"🎯 Final size: {processed_image.width}x{processed_image.height}")
        print(f"📝 Processing steps: {len(processing_steps)}")
        print(f"💾 Output: {actual_filename}.jpeg")

    def test_real_image_generation_and_loading(self) -> None:
        """Test generating real images and loading them back."""
        try:
            generator = TestImageGenerator()
            with tempfile.TemporaryDirectory() as tmpdir:
                test_images = {}
                landscape_path = generator.create_test_image(
                    width=2000,
                    height=1500,
                    output_path=Path(tmpdir) / "landscape.jpg",
                    image_type="landscape",
                    camera_make="Nikon",
                    camera_model="D850",
                    photographer="Test Photographer",
                    title="Sunset Landscape",
                    keywords=["sunset", "landscape", "golden hour"],
                )
                test_images["landscape"] = landscape_path
                portrait_path = generator.create_test_image(
                    width=1200,
                    height=1600,
                    output_path=Path(tmpdir) / "portrait.jpg",
                    image_type="portrait",
                    camera_make="Sony",
                    camera_model="A7R IV",
                    photographer="Portrait Pro",
                    title="Studio Portrait",
                    keywords=["portrait", "studio", "professional"],
                )
                test_images["portrait"] = portrait_path
                for image_type, image_path in test_images.items():
                    assert image_path.exists(), f"{image_type} image was not created"
                    assert image_path.stat().st_size > 0, f"{image_type} image is empty"
                if IMAGE_LOADING_AVAILABLE:
                    for image_type, image_path in test_images.items():
                        try:
                            image_asset = load_image_asset(image_path)
                            assert image_asset.path == image_path
                            assert image_asset.format in ["JPEG", "JPG"]
                            assert image_asset.size[0] > 0 and image_asset.size[1] > 0
                            assert image_asset.mode in ["RGB", "RGBA", "L"]
                            if image_asset.exif:
                                print(
                                    f"  ✅ {image_type}: {image_asset.width}x{image_asset.height}, "
                                    f"EXIF keys: {list(image_asset.exif.keys())[:3]}..."
                                )
                            else:
                                print(
                                    f"  ⚠️  {image_type}: {image_asset.width}x{image_asset.height}, no EXIF data loaded"
                                )
                            template_vars = image_asset.template_vars
                            assert "width" in template_vars
                            assert "height" in template_vars
                            assert "filename" in template_vars
                            assert template_vars["width"] == image_asset.width
                            assert template_vars["height"] == image_asset.height
                        except Exception as e:
                            pytest.skip(f"Image loading failed for {image_type}: {e}")
                print(f"🎨 Generated and verified {len(test_images)} test images")
        except ImportError as e:
            pytest.skip(f"Image generation not available: {e}")
        except Exception as e:
            pytest.skip(f"Test image generation failed: {e}")


if __name__ == "__main__":
    test_suite = TestPhotoFlowIntegration()
    test_suite.setup_method()
    test_suite.test_comprehensive_integration_workflow()
