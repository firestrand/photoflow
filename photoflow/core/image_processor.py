"""Core image processing infrastructure using PIL/Pillow.

This module provides the ImageProcessor class that handles actual image manipulation
operations while preserving metadata and providing a clean interface for actions.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

from .models import ImageAsset

# Constants
MIDPOINT_THRESHOLD = 0.5
RGB_CHANNEL_COUNT = 2

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import piexif  # noqa: F401

    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


class ImageProcessingError(Exception):
    """Raised when image processing operations fail."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize with message and optional original error."""
        super().__init__(message)
        self.original_error = original_error


class ImageProcessor:
    """Core image processing class using PIL."""

    def __init__(self) -> None:
        """Initialize the image processor."""
        if not PIL_AVAILABLE:
            msg = "PIL/Pillow is required for image processing"
            raise ImportError(msg)

    def load_pil_image(self, image_asset: ImageAsset) -> Image.Image:
        """Load PIL Image from ImageAsset."""
        try:
            if hasattr(image_asset, "_pil_image") and image_asset._pil_image is not None:
                return image_asset._pil_image  # type: ignore[no-any-return]

            if image_asset.path and image_asset.path.exists():
                return Image.open(image_asset.path)

            raise ImageProcessingError("No image data available in ImageAsset")
        except Exception as e:
            raise ImageProcessingError(f"Failed to load PIL Image: {e}", e) from e

    def create_image_asset(self, pil_image: Image.Image, original_asset: ImageAsset) -> ImageAsset:
        """Create new ImageAsset from PIL Image."""
        try:
            # Extract basic properties
            width, height = pil_image.size
            format_name = pil_image.format or "JPEG"
            mode = pil_image.mode

            # Handle DPI
            dpi = pil_image.info.get("dpi", (72, 72))
            if isinstance(dpi, int | float):
                dpi = (dpi, dpi)

            # Create new asset
            new_asset = ImageAsset(
                path=original_asset.path,
                format=format_name,
                size=(width, height),
                mode=mode,
                dpi=dpi,
                original_path=original_asset.original_path,
                metadata=original_asset.metadata.copy(),
            )

            # Store PIL image for reuse
            object.__setattr__(new_asset, "_pil_image", pil_image)
            return new_asset

        except Exception as e:
            raise ImageProcessingError(f"Failed to create ImageAsset: {e}", e) from e

    def _extract_exif_from_pil(self, pil_image: Image.Image) -> dict[str, Any]:
        """Extract EXIF data from PIL Image."""
        try:
            exif_dict = {}
            if hasattr(pil_image, "_getexif"):
                exif_data = pil_image._getexif()
                if exif_data:
                    exif_dict = exif_data
            elif hasattr(pil_image, "info") and "exif" in pil_image.info:
                # Try to parse EXIF from info
                exif_dict = pil_image.info.get("exif", {})
            return exif_dict
        except Exception:
            return {}

    def resize(
        self,
        image_asset: ImageAsset,
        width: int | None = None,
        height: int | None = None,
        maintain_aspect: bool = True,
        resample: str = "LANCZOS",
    ) -> ImageAsset:
        """Resize image with various options."""
        try:
            pil_image = self.load_pil_image(image_asset)
            original_width, original_height = pil_image.size

            # Calculate target dimensions
            if maintain_aspect and width and height:
                # Calculate aspect-preserving dimensions
                aspect_ratio = original_width / original_height
                if width / height > aspect_ratio:
                    width = int(height * aspect_ratio)
                else:
                    height = int(width / aspect_ratio)
            elif maintain_aspect and width:
                aspect_ratio = original_width / original_height
                height = int(width / aspect_ratio)
            elif maintain_aspect and height:
                aspect_ratio = original_width / original_height
                width = int(height * aspect_ratio)
            elif not width and not height:
                width, height = original_width, original_height
            elif not width:
                width = original_width
            elif not height:
                height = original_height

            # Map resample string to PIL constant
            resample_map = {
                "NEAREST": Image.Resampling.NEAREST,
                "LANCZOS": Image.Resampling.LANCZOS,
                "BILINEAR": Image.Resampling.BILINEAR,
                "BICUBIC": Image.Resampling.BICUBIC,
            }
            resample_filter = resample_map.get(resample, Image.Resampling.LANCZOS)

            # Perform resize
            if width is not None and height is not None:
                resized_image = pil_image.resize((width, height), resample=resample_filter)
            else:
                resized_image = pil_image
            return self.create_image_asset(resized_image, image_asset)

        except Exception as e:
            raise ImageProcessingError(f"Failed to resize image: {e}", e) from e

    def crop(
        self,
        image_asset: ImageAsset,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> ImageAsset:
        """Crop image to specified coordinates."""
        try:
            pil_image = self.load_pil_image(image_asset)
            cropped_image = pil_image.crop((left, top, right, bottom))
            return self.create_image_asset(cropped_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to crop image: {e}", e) from e

    def rotate(
        self,
        image_asset: ImageAsset,
        angle: float,
        expand: bool = False,
        fill_color: tuple[int, int, int] | None = None,
    ) -> ImageAsset:
        """Rotate image by specified angle."""
        try:
            pil_image = self.load_pil_image(image_asset)
            rotated_image = pil_image.rotate(angle, expand=expand, fillcolor=fill_color)
            return self.create_image_asset(rotated_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to rotate image: {e}", e) from e

    def adjust_brightness(self, image_asset: ImageAsset, factor: float) -> ImageAsset:
        """Adjust image brightness."""
        try:
            pil_image = self.load_pil_image(image_asset)
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced_image = enhancer.enhance(factor)
            return self.create_image_asset(enhanced_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to adjust brightness: {e}", e) from e

    def adjust_contrast(self, image_asset: ImageAsset, factor: float) -> ImageAsset:
        """Adjust image contrast."""
        try:
            pil_image = self.load_pil_image(image_asset)
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(factor)
            return self.create_image_asset(enhanced_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to adjust contrast: {e}", e) from e

    def adjust_color(self, image_asset: ImageAsset, factor: float) -> ImageAsset:
        """Adjust image color saturation."""
        try:
            pil_image = self.load_pil_image(image_asset)
            enhancer = ImageEnhance.Color(pil_image)
            enhanced_image = enhancer.enhance(factor)
            return self.create_image_asset(enhanced_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to adjust color: {e}", e) from e

    def adjust_sharpness(self, image_asset: ImageAsset, factor: float) -> ImageAsset:
        """Adjust image sharpness."""
        try:
            pil_image = self.load_pil_image(image_asset)
            enhancer = ImageEnhance.Sharpness(pil_image)
            enhanced_image = enhancer.enhance(factor)
            return self.create_image_asset(enhanced_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to adjust sharpness: {e}", e) from e

    def apply_blur(self, image_asset: ImageAsset, radius: float) -> ImageAsset:
        """Apply Gaussian blur to image."""
        try:
            pil_image = self.load_pil_image(image_asset)
            blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius))
            return self.create_image_asset(blurred_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to apply blur: {e}", e) from e

    def apply_sharpen(self, image_asset: ImageAsset) -> ImageAsset:
        """Apply sharpening filter to image."""
        try:
            pil_image = self.load_pil_image(image_asset)
            sharpened_image = pil_image.filter(ImageFilter.SHARPEN)
            return self.create_image_asset(sharpened_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to apply sharpen: {e}", e) from e

    def transpose(self, image_asset: ImageAsset, method: str) -> ImageAsset:
        """Transpose image using specified method."""
        try:
            pil_image = self.load_pil_image(image_asset)

            method_map = {
                "FLIP_LEFT_RIGHT": Image.Transpose.FLIP_LEFT_RIGHT,
                "FLIP_TOP_BOTTOM": Image.Transpose.FLIP_TOP_BOTTOM,
                "ROTATE_90": Image.Transpose.ROTATE_90,
                "ROTATE_180": Image.Transpose.ROTATE_180,
                "ROTATE_270": Image.Transpose.ROTATE_270,
                "TRANSPOSE": Image.Transpose.TRANSPOSE,
                "TRANSVERSE": Image.Transpose.TRANSVERSE,
            }

            transpose_method = method_map.get(method)
            if transpose_method is None:
                raise ValueError(f"Invalid transpose method: {method}")

            transposed_image = pil_image.transpose(transpose_method)
            return self.create_image_asset(transposed_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to transpose image: {e}", e) from e

    def add_watermark(
        self,
        image_asset: ImageAsset,
        text: str,
        position: str = "bottom_right",
        opacity: float = 0.5,
        font_size: int = 24,
    ) -> ImageAsset:
        """Add text watermark to image."""
        try:
            pil_image = self.load_pil_image(image_asset).copy()
            draw = ImageDraw.Draw(pil_image)

            # Use default font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()  # type: ignore[assignment]

            # Calculate text size and position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            img_width, img_height = pil_image.size
            margin = 10

            position_map = {
                "top_left": (margin, margin),
                "top_right": (img_width - text_width - margin, margin),
                "bottom_left": (margin, img_height - text_height - margin),
                "bottom_right": (
                    img_width - text_width - margin,
                    img_height - text_height - margin,
                ),
                "center": (
                    (img_width - text_width) // RGB_CHANNEL_COUNT,
                    (img_height - text_height) // RGB_CHANNEL_COUNT,
                ),
            }

            pos = position_map.get(position, position_map["bottom_right"])

            # Create watermark with opacity
            watermark = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            watermark_draw = ImageDraw.Draw(watermark)

            # Calculate alpha value
            alpha = int(255 * opacity)
            watermark_draw.text(pos, text, font=font, fill=(255, 255, 255, alpha))

            # Composite watermark onto image
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")

            watermarked = Image.alpha_composite(pil_image, watermark)

            return self.create_image_asset(watermarked, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to add watermark: {e}", e) from e

    def adjust_hue(self, image_asset: ImageAsset, shift: float) -> ImageAsset:
        """Adjust image hue using HSV color space."""
        try:
            pil_image = self.load_pil_image(image_asset)

            if NUMPY_AVAILABLE:
                # Use numpy for faster processing
                img_array = np.array(pil_image.convert("RGB"))
                original_shape = img_array.shape

                # Reshape to work with pixels
                pixels = img_array.reshape(-1, 3)

                # Normalize shift to 0-1 range
                shift_normalized = shift / 360.0

                # Process each pixel
                adjusted_pixels = []
                for pixel in pixels:
                    # Convert RGB to HSV
                    r, g, b = pixel / 255.0
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)

                    # Adjust hue
                    h = (h + shift_normalized) % 1.0

                    # Convert back to RGB
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    adjusted_pixels.append([int(r * 255), int(g * 255), int(b * 255)])

                # Reshape back to original shape
                adjusted_array = np.array(adjusted_pixels, dtype=np.uint8).reshape(original_shape)
                adjusted_image = Image.fromarray(adjusted_array)
            else:
                # Fallback to PIL-only approach
                rgb_image = pil_image.convert("RGB")
                enhancer = ImageEnhance.Color(rgb_image)
                adjusted_image = enhancer.enhance(1.0)  # Keep saturation

            return self.create_image_asset(adjusted_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to adjust hue: {e}", e) from e

    def adjust_levels(
        self,
        image_asset: ImageAsset,
        input_black: int = 0,
        input_white: int = 255,
        gamma: float = 1.0,
        output_black: int = 0,
        output_white: int = 255,
    ) -> ImageAsset:
        """Adjust image levels (similar to Photoshop levels)."""
        try:
            pil_image = self.load_pil_image(image_asset)

            if NUMPY_AVAILABLE:
                # Use numpy for precise levels adjustment
                img_array = np.array(pil_image.convert("RGB"))

                # Normalize input range
                img_normalized = np.clip(
                    (img_array.astype(np.float32) - input_black) / (input_white - input_black),
                    0,
                    1,
                )

                # Apply gamma correction
                img_gamma = np.power(img_normalized, 1.0 / gamma)

                # Apply output range
                img_output = img_gamma * (output_white - output_black) + output_black

                # Convert back to uint8
                adjusted_array = np.clip(img_output, 0, 255).astype(np.uint8)
                adjusted_image = Image.fromarray(adjusted_array)
            else:
                # Fallback: use ImageOps for basic adjustment
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                # Simple levels adjustment using ImageOps
                adjusted_image = ImageOps.autocontrast(pil_image)

            return self.create_image_asset(adjusted_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to adjust levels: {e}", e) from e

    def apply_sepia(self, image_asset: ImageAsset) -> ImageAsset:
        """Apply sepia tone effect to image."""
        try:
            pil_image = self.load_pil_image(image_asset)
            rgb_image = pil_image.convert("RGB")

            if NUMPY_AVAILABLE:
                # Use numpy for sepia transformation
                img_array = np.array(rgb_image)

                # Sepia transformation matrix
                sepia_matrix = np.array(
                    [
                        [0.393, 0.769, 0.189],
                        [0.349, 0.686, 0.168],
                        [0.272, 0.534, 0.131],
                    ]
                )

                # Apply sepia transformation
                sepia_array = np.dot(img_array, sepia_matrix.T)
                sepia_array = np.clip(sepia_array, 0, 255).astype(np.uint8)
                sepia_image = Image.fromarray(sepia_array)
            else:
                # Fallback: simple sepia using color adjustments
                # Convert to grayscale first
                gray = rgb_image.convert("L")
                # Create sepia by colorizing grayscale
                sepia_image = ImageOps.colorize(gray, "#704214", "#C0A882")

            return self.create_image_asset(sepia_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to apply sepia: {e}", e) from e

    def reduce_noise(self, image_asset: ImageAsset, strength: float = 1.0) -> ImageAsset:
        """Apply noise reduction to image."""
        try:
            pil_image = self.load_pil_image(image_asset)

            # Apply median filter for noise reduction
            # Strength determines the filter size
            filter_size = max(3, int(strength * 3))
            if filter_size % RGB_CHANNEL_COUNT == 0:
                filter_size += 1  # Ensure odd number

            filtered_image = pil_image.filter(ImageFilter.MedianFilter(size=filter_size))

            # Optionally apply slight blur for additional smoothing
            if strength > MIDPOINT_THRESHOLD:
                blur_radius = (strength - MIDPOINT_THRESHOLD) * RGB_CHANNEL_COUNT
                filtered_image = filtered_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            return self.create_image_asset(filtered_image, image_asset)
        except Exception as e:
            raise ImageProcessingError(f"Failed to reduce noise: {e}", e) from e

    def save_image(
        self,
        image_asset: ImageAsset,
        output_path: Path,
        format_name: str | None = None,
        quality: int = 95,
        **kwargs: Any,
    ) -> ImageAsset:
        """Save image to specified path with format conversion if needed."""
        try:
            pil_image = self.load_pil_image(image_asset)
            output_path = Path(output_path)

            # Determine format
            if format_name is None:
                format_name = output_path.suffix.upper().lstrip(".")
                if not format_name:
                    format_name = "JPEG"

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle format-specific requirements
            if format_name.upper() in ["JPEG", "JPG"] and pil_image.mode == "RGBA":
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = rgb_image

            # Save with appropriate options
            save_kwargs: dict[str, Any] = {"format": format_name.upper()}
            if format_name.upper() in ["JPEG", "JPG"]:
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True

            # Add any additional kwargs
            save_kwargs.update(kwargs)

            pil_image.save(output_path, **save_kwargs)

            # Create new asset with updated path
            new_asset = self.create_image_asset(pil_image, image_asset)
            # Create new asset with updated path and format
            new_asset = ImageAsset(
                path=output_path,
                format=format_name.upper(),
                size=new_asset.size,
                mode=new_asset.mode,
                dpi=new_asset.dpi,
                original_path=new_asset.original_path,
                metadata=new_asset.metadata,
            )
            # Store PIL image for reuse
            object.__setattr__(new_asset, "_pil_image", pil_image)

            return new_asset
        except Exception as e:
            raise ImageProcessingError(f"Failed to save image: {e}", e) from e


# Default processor instance
default_processor = ImageProcessor() if PIL_AVAILABLE else None
