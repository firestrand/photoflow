"""Utility to generate test images with realistic metadata for PhotoFlow testing.

This module creates test JPEG images with proper EXIF and IPTC metadata
to support realistic integration testing without requiring real photo files.
"""

import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import piexif
from PIL import Image, ImageDraw

try:
    from pillow_heif import register_heif_opener

    PIL_HEIF_AVAILABLE = True
except ImportError:
    PIL_HEIF_AVAILABLE = False

try:
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
if PIL_HEIF_AVAILABLE:
    try:
        register_heif_opener()
        HEIF_AVAILABLE = True
    except Exception:
        HEIF_AVAILABLE = False
else:
    HEIF_AVAILABLE = False

try:
    import piexif

    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


class TestImageGenerator:
    """Generate test images with realistic metadata for PhotoFlow testing."""

    def __init__(self) -> None:
        """Initialize the test image generator."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for test image generation")

    def create_test_image(
        self,
        width: int = 4000,
        height: int = 3000,
        output_path: Path | str | None = None,
        image_type: str = "landscape",
        camera_make: str = "Canon",
        camera_model: str = "EOS R5",
        photographer: str = "John Photographer",
        title: str = "Mountain Sunset",
        keywords: list[str] | None = None,
        **metadata_overrides: Any,
    ) -> Path:
        """Create a test image with realistic EXIF and IPTC metadata.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            output_path: Path to save the image (temporary file if None)
            image_type: Type of image to generate (landscape, portrait, abstract)
            camera_make: Camera manufacturer
            camera_model: Camera model
            photographer: Photographer name
            title: Image title
            keywords: List of keywords for IPTC
            **metadata_overrides: Additional metadata to override defaults

        Returns:
            Path to the generated test image

        Raises:
            ImportError: If required dependencies are not available
        """
        if keywords is None:
            keywords = ["landscape", "sunset", "mountains", "nature"]
        image = self._generate_image_content(width, height, image_type)
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                output_path = Path(temp_file.name)
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        exif_dict = self._generate_exif_metadata(width, height, camera_make, camera_model, **metadata_overrides)
        if PIEXIF_AVAILABLE:
            exif_bytes = piexif.dump(exif_dict)
            image.save(output_path, "JPEG", quality=95, exif=exif_bytes)
        else:
            image.save(output_path, "JPEG", quality=95)
            print("Warning: piexif not available, saving without EXIF metadata")
        return output_path

    def _generate_image_content(self, width: int, height: int, image_type: str) -> Image.Image:
        """Generate the actual image content.

        Args:
            width: Image width
            height: Image height
            image_type: Type of image to generate

        Returns:
            PIL Image object
        """
        if image_type == "landscape":
            image = self._create_landscape_image(width, height)
        elif image_type == "portrait":
            image = self._create_portrait_image(width, height)
        elif image_type == "abstract":
            image = self._create_abstract_image(width, height)
        else:
            image = self._create_gradient_image(width, height)
        return image

    def _create_landscape_image(self, width: int, height: int) -> Image.Image:
        """Create a landscape-style test image."""
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        for y in range(height):
            blue_intensity = max(0, 255 - int(y / height * 400))
            orange_intensity = min(255, int(y / height * 300))
            red_intensity = min(255, int(y / height * 255))
            color = red_intensity, orange_intensity // 2, blue_intensity
            draw.line([(0, y), (width, y)], fill=color)
        mountain_points = []
        for x in range(0, width, width // 20):
            mountain_height = height - int(height * 0.3) + x % 200 - 100
            mountain_points.extend([(x, mountain_height), (x + width // 40, mountain_height - 50)])
        mountain_points.extend([(width, height), (0, height)])
        if mountain_points:
            draw.polygon(mountain_points, fill=(50, 30, 80))
        sun_x, sun_y = int(width * 0.2), int(height * 0.3)
        sun_radius = 50
        draw.ellipse(
            [
                sun_x - sun_radius,
                sun_y - sun_radius,
                sun_x + sun_radius,
                sun_y + sun_radius,
            ],
            fill=(255, 220, 100),
        )
        return image

    def _create_portrait_image(self, width: int, height: int) -> Image.Image:
        """Create a portrait-style test image."""
        image = Image.new("RGB", (width, height), color=(180, 160, 140))
        draw = ImageDraw.Draw(image)
        face_x, face_y = width // 2, height // 3
        face_width, face_height = width // 4, height // 3
        draw.ellipse(
            [
                face_x - face_width // 2,
                face_y - face_height // 2,
                face_x + face_width // 2,
                face_y + face_height // 2,
            ],
            fill=(220, 180, 150),
        )
        eye_y = face_y - face_height // 6
        left_eye_x = face_x - face_width // 4
        right_eye_x = face_x + face_width // 4
        eye_size = 20
        for eye_x in [left_eye_x, right_eye_x]:
            draw.ellipse(
                [
                    eye_x - eye_size,
                    eye_y - eye_size // 2,
                    eye_x + eye_size,
                    eye_y + eye_size // 2,
                ],
                fill=(50, 50, 50),
            )
        return image

    def _create_abstract_image(self, width: int, height: int) -> Image.Image:
        """Create an abstract test image."""
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        random.seed(42)
        colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
        ]
        for i in range(20):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(x1, min(x1 + width // 4, width))
            y2 = random.randint(y1, min(y1 + height // 4, height))
            color = colors[i % len(colors)]
            draw.ellipse([x1, y1, x2, y2], fill=color)
        return image

    def _create_gradient_image(self, width: int, height: int) -> Image.Image:
        """Create a simple gradient test image."""
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        for x in range(width):
            red = int(x / width * 255)
            green = int((width - x) / width * 255)
            blue = 128
            draw.line([(x, 0), (x, height)], fill=(red, green, blue))
        return image

    def _generate_exif_metadata(
        self,
        width: int,
        height: int,
        camera_make: str,
        camera_model: str,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Generate realistic EXIF metadata.

        Args:
            width: Image width
            height: Image height
            camera_make: Camera manufacturer
            camera_model: Camera model
            **overrides: Additional metadata overrides

        Returns:
            EXIF metadata dictionary
        """
        if not PIEXIF_AVAILABLE:
            return {}
        now = datetime.now()
        date_str = now.strftime("%Y:%m:%d %H:%M:%S")
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: camera_make,
                piexif.ImageIFD.Model: camera_model,
                piexif.ImageIFD.DateTime: date_str,
                piexif.ImageIFD.Software: "PhotoFlow Test Generator",
                piexif.ImageIFD.ImageWidth: width,
                piexif.ImageIFD.ImageLength: height,
                piexif.ImageIFD.XResolution: (300, 1),
                piexif.ImageIFD.YResolution: (300, 1),
                piexif.ImageIFD.ResolutionUnit: 2,
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: date_str,
                piexif.ExifIFD.DateTimeDigitized: date_str,
                piexif.ExifIFD.FocalLength: (85, 1),
                piexif.ExifIFD.FNumber: (28, 10),
                piexif.ExifIFD.ISOSpeedRatings: 400,
                piexif.ExifIFD.ExposureTime: (1, 250),
                piexif.ExifIFD.LensModel: "RF 85mm f/2.8 Macro IS STM",
                piexif.ExifIFD.ColorSpace: 1,
            },
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        for section, values in overrides.items():
            if section in exif_dict and isinstance(values, dict):
                exif_dict[section].update(values)
        return exif_dict


def create_test_landscape(output_path: Path | str | None = None, width: int = 4000, height: int = 3000) -> Path:
    """Convenience function to create a test landscape image.

    Args:
        output_path: Where to save the image
        width: Image width
        height: Image height

    Returns:
        Path to the created image
    """
    generator = TestImageGenerator()
    return generator.create_test_image(
        width=width,
        height=height,
        output_path=output_path,
        image_type="landscape",
        title="Mountain Sunset",
        keywords=["landscape", "sunset", "mountains", "nature"],
    )


def create_test_portrait(output_path: Path | str | None = None, width: int = 3000, height: int = 4000) -> Path:
    """Convenience function to create a test portrait image.

    Args:
        output_path: Where to save the image
        width: Image width
        height: Image height

    Returns:
        Path to the created image
    """
    generator = TestImageGenerator()
    return generator.create_test_image(
        width=width,
        height=height,
        output_path=output_path,
        image_type="portrait",
        title="Portrait Study",
        keywords=["portrait", "person", "studio"],
    )


if __name__ == "__main__":
    """Generate sample test images when run directly."""
    print("🎨 PhotoFlow Test Image Generator")
    print("=" * 40)
    missing_deps = []
    if not PIL_AVAILABLE:
        missing_deps.append("Pillow")
    if not PIEXIF_AVAILABLE:
        missing_deps.append("piexif")
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install Pillow piexif")
        sys.exit(1)
    print("✅ All dependencies available")
    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)
    print(f"\n📸 Generating test images in {output_dir}...")
    landscape_path = create_test_landscape(output_dir / "landscape_4000x3000.jpg", width=4000, height=3000)
    print(f"  ✅ Landscape: {landscape_path}")
    portrait_path = create_test_portrait(output_dir / "portrait_3000x4000.jpg", width=3000, height=4000)
    print(f"  ✅ Portrait: {portrait_path}")
    generator = TestImageGenerator()
    abstract_path = generator.create_test_image(
        width=2000,
        height=2000,
        output_path=output_dir / "abstract_2000x2000.jpg",
        image_type="abstract",
        title="Abstract Composition",
        keywords=["abstract", "colorful", "geometric"],
    )
    print(f"  ✅ Abstract: {abstract_path}")
    print(f"\n🎯 Generated {len(list(output_dir.glob('*.jpg')))} test images")
    print("These images can be used for PhotoFlow integration testing!")
