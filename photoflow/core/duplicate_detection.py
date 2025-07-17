"""Duplicate and near-duplicate detection using perceptual hashing.

This module provides functionality to detect duplicate and near-duplicate images
using perceptual hashing techniques. Based on average hash (aHash) algorithm
for fast and reliable similarity detection.
"""

from __future__ import annotations

from pathlib import Path

from .image_loader import load_image_asset
from .image_processor import default_processor
from .models import ImageAsset

# Constants
HASH_VALUE_LENGTH = 16
MIN_GROUP_SIZE = 2


class PerceptualHash:
    """Perceptual hash for image similarity detection.

    Uses a 64-bit hash representation that can be compared for similarity
    using Hamming distance calculation.
    """

    def __init__(self, hash_value: str) -> None:
        """Initialize perceptual hash.

        Args:
            hash_value: 16-character hexadecimal string representing 64-bit hash
        """
        if len(hash_value) != HASH_VALUE_LENGTH:
            raise ValueError(f"Hash value must be {HASH_VALUE_LENGTH} hexadecimal characters")
        self.hash_value = hash_value.lower()

    def __str__(self) -> str:
        """Return string representation of hash."""
        return self.hash_value

    def __eq__(self, other: object) -> bool:
        """Check if two hashes are equal."""
        if not isinstance(other, PerceptualHash):
            return False
        return self.hash_value == other.hash_value

    def __hash__(self) -> int:
        """Return hash for dictionary/set usage."""
        return hash(self.hash_value)

    def hamming_distance(self, other: PerceptualHash) -> int:
        """Calculate Hamming distance between two hashes.

        Args:
            other: Another PerceptualHash to compare with

        Returns:
            Number of differing bits (0-64)
        """
        if not isinstance(other, PerceptualHash):
            raise TypeError("Can only compare with another PerceptualHash")

        # Convert hex strings to integers and XOR them
        hash1_int = int(self.hash_value, 16)
        hash2_int = int(other.hash_value, 16)
        xor_result = hash1_int ^ hash2_int

        # Count number of 1 bits (differing bits)
        return bin(xor_result).count("1")

    def similarity(self, other: PerceptualHash) -> float:
        """Calculate similarity as a value between 0.0 and 1.0.

        Args:
            other: Another PerceptualHash to compare with

        Returns:
            Similarity value (1.0 = identical, 0.0 = completely different)
        """
        distance = self.hamming_distance(other)
        return 1.0 - (distance / 64.0)


class ImageHasher:
    """Generates perceptual hashes for images using average hash algorithm."""

    def __init__(self, hash_size: int = 8) -> None:
        """Initialize image hasher.

        Args:
            hash_size: Size of the hash grid (hash_size x hash_size)
                      Default 8 produces 64-bit hash
        """
        self.hash_size = hash_size

    def hash_image(self, image: ImageAsset | Path) -> PerceptualHash:
        """Generate perceptual hash for an image.

        Args:
            image: ImageAsset or Path to image file

        Returns:
            PerceptualHash object representing the image

        Raises:
            Exception: If image cannot be loaded or processed
        """
        if not default_processor:
            raise RuntimeError("Image processor not available")

        # Load image using our image loader and processor
        image_asset = image if isinstance(image, ImageAsset) else load_image_asset(Path(image))

        # Get PIL image
        pil_image = default_processor.load_pil_image(image_asset)

        # Convert to grayscale and resize to hash_size x hash_size
        gray_image = pil_image.convert("L")
        resized = gray_image.resize((self.hash_size, self.hash_size))

        # Get pixel data
        pixels = list(resized.getdata())

        # Calculate average pixel value
        avg_pixel = sum(pixels) / len(pixels)

        # Create hash by comparing each pixel to average
        hash_bits = []
        for pixel in pixels:
            hash_bits.append("1" if pixel >= avg_pixel else "0")

        # Convert binary string to hexadecimal
        binary_string = "".join(hash_bits)
        hash_int = int(binary_string, 2)
        hash_hex = f"{hash_int:016x}"  # 16 hex digits for 64 bits

        return PerceptualHash(hash_hex)


class DuplicateDetector:
    """Detects duplicate and near-duplicate images using perceptual hashing."""

    def __init__(self, threshold: int = 5) -> None:
        """Initialize duplicate detector.

        Args:
            threshold: Maximum Hamming distance for images to be considered duplicates
                      Lower values = more strict matching
                      Default 5 allows for minor compression/format differences
        """
        self.threshold = threshold
        self.hasher = ImageHasher()

    def find_duplicates(self, images: list[Path | ImageAsset]) -> list[list[Path]]:
        """Find duplicate groups in a list of images.

        Args:
            images: List of image paths or ImageAsset objects

        Returns:
            List of duplicate groups, where each group contains 2+ similar images
        """
        if not images:
            return []

        # Generate hashes for all images
        hash_data: list[tuple[Path, PerceptualHash]] = []

        for image in images:
            try:
                img_hash = self.hasher.hash_image(image)
                path = image.path if isinstance(image, ImageAsset) else Path(image)
                hash_data.append((path, img_hash))
            except Exception:
                # Skip images that can't be processed
                continue

        # Group similar images
        return self._group_similar_images(hash_data)

    def _group_similar_images(self, hash_data: list[tuple[Path, PerceptualHash]]) -> list[list[Path]]:
        """Group images by hash similarity.

        Args:
            hash_data: List of (path, hash) tuples

        Returns:
            List of groups containing similar images
        """
        if len(hash_data) < MIN_GROUP_SIZE:
            return []

        groups: list[list[Path]] = []
        used_indices = set()

        for i in range(len(hash_data)):
            if i in used_indices:
                continue

            current_group = [hash_data[i][0]]
            used_indices.add(i)

            # Find all images similar to current image
            for j in range(i + 1, len(hash_data)):
                if j in used_indices:
                    continue

                distance = hash_data[i][1].hamming_distance(hash_data[j][1])
                if distance <= self.threshold:
                    current_group.append(hash_data[j][0])
                    used_indices.add(j)

            # Only add groups with 2+ images
            if len(current_group) >= MIN_GROUP_SIZE:
                groups.append(current_group)

        return groups
