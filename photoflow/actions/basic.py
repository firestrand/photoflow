"""Basic image processing actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..core.models import ImageAsset

from ..core.actions import BaseAction, register_action
from ..core.fields import BooleanField, ChoiceField, IntegerField, StringField
from ..core.image_processor import ImageProcessingError, default_processor
from ..core.mixins import GeometryMixin, ProcessingMixin
from ..core.templates import TemplateEvaluator


class ResizeAction(BaseAction, GeometryMixin):
    """Resize image to specified dimensions."""

    name = "resize"
    label = "Resize Image"
    description = "Resize image to specified width and height with optional aspect ratio preservation"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["geometry", "basic", "resize"]

    def setup_fields(self) -> None:
        """Set up resize parameters."""
        self.fields = {
            "width": IntegerField(
                default=1920,
                min_value=1,
                max_value=20000,
                label="Width",
                help_text="Target width in pixels",
            ),
            "height": IntegerField(
                default=None,
                min_value=1,
                max_value=20000,
                label="Height",
                help_text="Target height in pixels (optional if maintain_aspect is True)",
                required=False,
            ),
            "maintain_aspect": BooleanField(
                default=True,
                label="Maintain Aspect Ratio",
                help_text="Automatically calculate height to maintain aspect ratio",
            ),
        }

    def get_relevant_fields(self, params: dict[str, Any]) -> list[str]:
        """Show height field only if not maintaining aspect ratio."""
        fields = ["width", "maintain_aspect"]
        if not params.get("maintain_aspect", True):
            fields.append("height")
        return fields

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply resize operation."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        width = params["width"]
        maintain_aspect = params.get("maintain_aspect", True)
        height = params.get("height")
        resized_image = default_processor.resize(image, width=width, height=height, maintain_aspect=maintain_aspect)
        return resized_image.with_updates(
            processing_metadata={
                **resized_image.processing_metadata,
                "last_resize": f"{resized_image.width}x{resized_image.height}",
                "maintain_aspect": maintain_aspect,
            }
        )


class CropAction(BaseAction, GeometryMixin):
    """Crop image to specified region."""

    name = "crop"
    label = "Crop Image"
    description = "Crop image to specified rectangular region"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["geometry", "basic", "crop"]

    def setup_fields(self) -> None:
        """Set up crop parameters."""
        self.fields = {
            "x": IntegerField(
                default=0,
                min_value=0,
                label="X Position",
                help_text="Left edge of crop region in pixels",
            ),
            "y": IntegerField(
                default=0,
                min_value=0,
                label="Y Position",
                help_text="Top edge of crop region in pixels",
            ),
            "width": IntegerField(
                default=None,
                min_value=1,
                label="Width",
                help_text="Width of crop region in pixels",
                required=False,
            ),
            "height": IntegerField(
                default=None,
                min_value=1,
                label="Height",
                help_text="Height of crop region in pixels",
                required=False,
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply crop operation."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        x = params["x"]
        y = params["y"]
        width = params.get("width", image.width - x)
        height = params.get("height", image.height - y)
        cropped_image = default_processor.crop(image, x, y, x + width, y + height)
        return cropped_image.with_updates(
            processing_metadata={
                **cropped_image.processing_metadata,
                "crop_region": f"{x},{y},{width},{height}",
                "original_size": f"{image.width}x{image.height}",
            }
        )


class SaveAction(BaseAction, ProcessingMixin):
    """Save image to file with format conversion."""

    name = "save"
    label = "Save Image"
    description = "Save image to file with optional format conversion and quality settings"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["output", "format", "quality"]

    def setup_fields(self) -> None:
        """Set up save parameters."""
        self.fields = {
            "format": ChoiceField(
                default="JPEG",
                choices=[
                    ("JPEG", "JPEG"),
                    ("PNG", "PNG"),
                    ("WebP", "WebP"),
                    ("TIFF", "TIFF"),
                    ("BMP", "BMP"),
                ],
                label="Format",
                help_text="Output image format",
            ),
            "quality": IntegerField(
                default=85,
                min_value=1,
                max_value=100,
                label="Quality",
                help_text="Image quality (1-100, for JPEG/WebP formats)",
            ),
            "optimize": BooleanField(
                default=True,
                label="Optimize",
                help_text="Enable format-specific optimizations",
            ),
        }

    def get_relevant_fields(self, params: dict[str, Any]) -> list[str]:
        """Show quality field only for formats that support it."""
        fields = ["format", "optimize"]
        format_name = params.get("format", "JPEG")
        if format_name in ("JPEG", "WebP"):
            fields.append("quality")
        return fields

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply save operation."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        format_name = params["format"]
        quality = params.get("quality", 85)
        optimize = params.get("optimize", True)
        converted_image = default_processor.save_image(image, image.path, format_name, quality)
        return converted_image.with_updates(
            processing_metadata={
                **converted_image.processing_metadata,
                "output_format": format_name,
                "output_quality": quality,
                "optimized": optimize,
            }
        )


class WatermarkAction(BaseAction, ProcessingMixin):
    """Add text watermark to image."""

    name = "watermark"
    label = "Add Watermark"
    description = "Add text watermark with template support and positioning"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["text", "branding", "watermark", "effect"]

    def setup_fields(self) -> None:
        """Set up watermark parameters."""
        self.fields = {
            "text": StringField(
                default="© <creator>",
                label="Watermark Text",
                help_text="Text to use as watermark (supports template variables)",
            ),
            "position": ChoiceField(
                default="bottom_right",
                choices=[
                    ("top_left", "Top Left"),
                    ("top_right", "Top Right"),
                    ("bottom_left", "Bottom Left"),
                    ("bottom_right", "Bottom Right"),
                    ("center", "Center"),
                ],
                label="Position",
                help_text="Where to place the watermark",
            ),
            "opacity": IntegerField(
                default=75,
                min_value=1,
                max_value=100,
                label="Opacity",
                help_text="Watermark opacity (1-100)",
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply watermark operation."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        text_template = params["text"]
        position = params["position"]
        opacity = params["opacity"]
        evaluator = TemplateEvaluator()
        evaluated_text = evaluator.evaluate(text_template, image.template_vars)
        watermarked_image = default_processor.add_watermark(image, evaluated_text, position, opacity)
        return watermarked_image.with_updates(
            metadata={
                **watermarked_image.metadata,
                "watermark_text": evaluated_text,
                "watermark_position": position,
                "watermark_opacity": opacity,
            },
            processing_metadata={
                **watermarked_image.processing_metadata,
                "watermark_applied": True,
                "watermark_config": f"{position}@{opacity}%",
            },
        )


register_action(ResizeAction)
register_action(CropAction)
register_action(SaveAction)
register_action(WatermarkAction)
