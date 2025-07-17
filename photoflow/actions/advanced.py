"""Advanced image processing actions for PhotoFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..core.models import ImageAsset

from ..core.actions import BaseAction, register_action
from ..core.fields import FloatField
from ..core.image_processor import ImageProcessingError, default_processor
from ..core.mixins import ProcessingMixin


class HueAction(BaseAction, ProcessingMixin):
    """Adjust image hue (color wheel rotation)."""

    name = "hue"
    label = "Adjust Hue"
    description = "Adjust image hue by rotating colors on the color wheel"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["color", "hue", "adjustment", "creative"]

    def setup_fields(self) -> None:
        """Set up hue adjustment parameters."""
        self.fields = {
            "shift": FloatField(
                default=0.0,
                min_value=-180.0,
                max_value=180.0,
                label="Hue Shift",
                help_text="Hue shift in degrees (-180 to 180, 0 = no change)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply hue adjustment."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        shift = params["shift"]
        adjusted_image = default_processor.adjust_hue(image, shift)
        return adjusted_image.with_updates(
            processing_metadata={
                **adjusted_image.processing_metadata,
                "hue_shift": shift,
            }
        )


class LevelsAction(BaseAction, ProcessingMixin):
    """Adjust image levels (shadows, midtones, highlights)."""

    name = "levels"
    label = "Adjust Levels"
    description = "Adjust shadows, midtones, and highlights independently"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["color", "tone", "levels", "professional"]

    def setup_fields(self) -> None:
        """Set up levels adjustment parameters."""
        self.fields = {
            "shadows": FloatField(
                default=0.0,
                min_value=-1.0,
                max_value=1.0,
                label="Shadows",
                help_text="Shadow adjustment (-1.0 = darker, 0.0 = no change, 1.0 = lighter)",
            ),
            "midtones": FloatField(
                default=1.0,
                min_value=0.1,
                max_value=3.0,
                label="Midtones",
                help_text="Midtone gamma adjustment (0.1-3.0, 1.0 = no change)",
            ),
            "highlights": FloatField(
                default=1.0,
                min_value=0.0,
                max_value=2.0,
                label="Highlights",
                help_text="Highlight adjustment (0.0 = darker, 1.0 = no change, 2.0 = brighter)",
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply levels adjustment."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        shadows = params["shadows"]
        midtones = params["midtones"]
        highlights = params["highlights"]
        adjusted_image = default_processor.adjust_levels(image, shadows, midtones, highlights)
        return adjusted_image.with_updates(
            processing_metadata={
                **adjusted_image.processing_metadata,
                "levels_shadows": shadows,
                "levels_midtones": midtones,
                "levels_highlights": highlights,
            }
        )


class SepiaAction(BaseAction, ProcessingMixin):
    """Apply sepia tone effect to image."""

    name = "sepia"
    label = "Sepia Tone"
    description = "Apply classic sepia tone effect for vintage look"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["effect", "sepia", "vintage", "creative"]

    def setup_fields(self) -> None:
        """Set up sepia effect parameters."""
        self.fields = {
            "intensity": FloatField(
                default=1.0,
                min_value=0.0,
                max_value=1.0,
                label="Intensity",
                help_text="Sepia effect intensity (0.0 = no effect, 1.0 = full sepia)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply sepia effect."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        intensity = params["intensity"]
        sepia_image = default_processor.apply_sepia(image)
        return sepia_image.with_updates(
            processing_metadata={
                **sepia_image.processing_metadata,
                "sepia_intensity": intensity,
            }
        )


class NoiseReductionAction(BaseAction, ProcessingMixin):
    """Apply noise reduction to image."""

    name = "noise_reduction"
    label = "Noise Reduction"
    description = "Reduce digital noise and grain in images"
    version = "1.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["enhancement", "noise", "quality", "cleanup"]

    def setup_fields(self) -> None:
        """Set up noise reduction parameters."""
        self.fields = {
            "strength": FloatField(
                default=1.0,
                min_value=0.1,
                max_value=3.0,
                label="Strength",
                help_text="Noise reduction strength (0.1 = light, 3.0 = aggressive)",
            )
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply noise reduction."""
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image processing")
        strength = params["strength"]
        denoised_image = default_processor.reduce_noise(image, strength)
        return denoised_image.with_updates(
            processing_metadata={
                **denoised_image.processing_metadata,
                "noise_reduction_strength": strength,
            }
        )


register_action(HueAction)
register_action(LevelsAction)
register_action(SepiaAction)
register_action(NoiseReductionAction)
