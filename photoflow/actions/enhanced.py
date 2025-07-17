"""Enhanced image processing actions for PhotoFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..core.models import ImageAsset

from ..core.actions import BaseAction, register_action
from ..core.fields import BooleanField, ChoiceField, FloatField
from ..core.image_processor import ImageProcessingError, default_processor
from ..core.mixins import GeometryMixin, ProcessingMixin


class RotateAction(BaseAction, GeometryMixin):
    """Rotate image by specified angle."""

    name = "rotate"
    label = "Rotate Image"
    description = "Rotate image by specified angle with optional expansion"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["geometry", "rotation", "transform"]

    def setup_fields(self) -> None:
        """Set up rotation parameters."""
        self.fields = {
            "angle": FloatField(
                default=90.0,
                min_value=-360.0,
                max_value=360.0,
                label="Angle",
                help_text="Rotation angle in degrees (positive = counter-clockwise)",
            ),
            "expand": BooleanField(
                default=True,
                label="Expand Canvas",
                help_text="Expand canvas to fit rotated image (prevents cropping)",
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply rotation operation."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        angle = params["angle"]
        expand = params.get("expand", True)
        rotated_image = default_processor.rotate(image, angle, expand)
        return rotated_image.with_updates(
            processing_metadata={
                **rotated_image.processing_metadata,
                "rotation_angle": angle,
                "canvas_expanded": expand,
            }
        )


class TransposeAction(BaseAction, GeometryMixin):
    """Transpose image (flip or rotate by 90° increments)."""

    name = "transpose"
    label = "Transpose Image"
    description = "Flip or rotate image by 90° increments"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["geometry", "flip", "transform"]

    def setup_fields(self) -> None:
        """Set up transpose parameters."""
        self.fields = {
            "method": ChoiceField(
                default="flip_horizontal",
                choices=[
                    ("flip_horizontal", "Flip Horizontal"),
                    ("flip_vertical", "Flip Vertical"),
                    ("rotate_90", "Rotate 90°"),
                    ("rotate_180", "Rotate 180°"),
                    ("rotate_270", "Rotate 270°"),
                ],
                label="Method",
                help_text="Transpose method to apply",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply transpose operation."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        method = params["method"]
        transposed_image = default_processor.transpose(image, method)
        return transposed_image.with_updates(
            processing_metadata={
                **transposed_image.processing_metadata,
                "transpose_method": method,
            }
        )


class BrightnessAction(BaseAction, ProcessingMixin):
    """Adjust image brightness."""

    name = "brightness"
    label = "Adjust Brightness"
    description = "Adjust image brightness level"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["color", "tone", "brightness", "adjustment"]

    def setup_fields(self) -> None:
        """Set up brightness parameters."""
        self.fields = {
            "factor": FloatField(
                default=1.0,
                min_value=0.0,
                max_value=3.0,
                label="Brightness Factor",
                help_text="Brightness adjustment (1.0 = no change, >1.0 = brighter, <1.0 = darker)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply brightness adjustment."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        factor = params["factor"]
        adjusted_image = default_processor.adjust_brightness(image, factor)
        return adjusted_image.with_updates(
            processing_metadata={
                **adjusted_image.processing_metadata,
                "brightness_factor": factor,
            }
        )


class ContrastAction(BaseAction, ProcessingMixin):
    """Adjust image contrast."""

    name = "contrast"
    label = "Adjust Contrast"
    description = "Adjust image contrast level"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["color", "tone", "contrast", "adjustment"]

    def setup_fields(self) -> None:
        """Set up contrast parameters."""
        self.fields = {
            "factor": FloatField(
                default=1.0,
                min_value=0.0,
                max_value=3.0,
                label="Contrast Factor",
                help_text="Contrast adjustment (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply contrast adjustment."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        factor = params["factor"]
        adjusted_image = default_processor.adjust_contrast(image, factor)
        return adjusted_image.with_updates(
            processing_metadata={
                **adjusted_image.processing_metadata,
                "contrast_factor": factor,
            }
        )


class SaturationAction(BaseAction, ProcessingMixin):
    """Adjust image color saturation."""

    name = "saturation"
    label = "Adjust Saturation"
    description = "Adjust image color saturation level"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["color", "saturation", "adjustment"]

    def setup_fields(self) -> None:
        """Set up saturation parameters."""
        self.fields = {
            "factor": FloatField(
                default=1.0,
                min_value=0.0,
                max_value=3.0,
                label="Saturation Factor",
                help_text="Saturation adjustment (1.0 = no change, >1.0 = more saturated, 0.0 = grayscale)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply saturation adjustment."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        factor = params["factor"]
        adjusted_image = default_processor.adjust_color(image, factor)
        return adjusted_image.with_updates(
            processing_metadata={
                **adjusted_image.processing_metadata,
                "saturation_factor": factor,
            }
        )


class BlurAction(BaseAction, ProcessingMixin):
    """Apply Gaussian blur to image."""

    name = "blur"
    label = "Blur Image"
    description = "Apply Gaussian blur effect to image"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["effect", "blur", "filter"]

    def setup_fields(self) -> None:
        """Set up blur parameters."""
        self.fields = {
            "radius": FloatField(
                default=2.0,
                min_value=0.1,
                max_value=20.0,
                label="Blur Radius",
                help_text="Blur radius in pixels (higher = more blur)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply blur effect."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        radius = params["radius"]
        blurred_image = default_processor.apply_blur(image, radius)
        return blurred_image.with_updates(
            processing_metadata={
                **blurred_image.processing_metadata,
                "blur_radius": radius,
            }
        )


class SharpenAction(BaseAction, ProcessingMixin):
    """Apply sharpening to image."""

    name = "sharpen"
    label = "Sharpen Image"
    description = "Apply sharpening effect to image"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["effect", "sharpen", "filter"]

    def setup_fields(self) -> None:
        """Set up sharpening parameters."""
        self.fields = {
            "factor": FloatField(
                default=1.0,
                min_value=0.1,
                max_value=3.0,
                label="Sharpening Factor",
                help_text="Sharpening intensity (1.0 = normal, higher = more sharpening)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply sharpening effect."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        factor = params["factor"]
        sharpened_image = default_processor.apply_sharpen(image)
        return sharpened_image.with_updates(
            processing_metadata={
                **sharpened_image.processing_metadata,
                "sharpen_factor": factor,
            }
        )


register_action(RotateAction)
register_action(TransposeAction)
register_action(BrightnessAction)
register_action(ContrastAction)
register_action(SaturationAction)
register_action(BlurAction)
register_action(SharpenAction)
