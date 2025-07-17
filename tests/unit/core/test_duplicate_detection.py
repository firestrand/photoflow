"""Tests for duplicate and near-duplicate detection functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

from photoflow.core.duplicate_detection import (
    DuplicateDetector,
    ImageHasher,
    PerceptualHash,
)
from photoflow.core.models import ImageAsset


class TestPerceptualHash:
    """Test perceptual hash generation for duplicate detection."""

    def test_perceptual_hash_creation(self):
        """Test that perceptual hash can be created from image."""
        hash_obj = PerceptualHash("0123456789abcdef")
        assert hash_obj.hash_value == "0123456789abcdef"
        assert len(str(hash_obj)) == 16

    def test_perceptual_hash_equality(self):
        """Test perceptual hash equality comparison."""
        hash1 = PerceptualHash("0123456789abcdef")
        hash2 = PerceptualHash("0123456789abcdef")
        hash3 = PerceptualHash("fedcba9876543210")
        assert hash1 == hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_perceptual_hash_hamming_distance(self):
        """Test hamming distance calculation between hashes."""
        hash1 = PerceptualHash("0000000000000000")
        hash2 = PerceptualHash("0000000000000000")
        assert hash1.hamming_distance(hash2) == 0
        hash3 = PerceptualHash("0000000000000001")
        assert hash1.hamming_distance(hash3) == 1
        hash4 = PerceptualHash("000000000000000f")
        assert hash1.hamming_distance(hash4) == 4

    def test_perceptual_hash_similarity(self):
        """Test similarity calculation between hashes."""
        hash1 = PerceptualHash("0000000000000000")
        hash2 = PerceptualHash("0000000000000000")
        hash3 = PerceptualHash("0000000000000001")
        assert hash1.similarity(hash2) == 1.0
        similarity = hash1.similarity(hash3)
        assert similarity > 0.9
        assert similarity < 1.0


class TestImageHasher:
    """Test image hashing functionality."""

    def test_image_hasher_creation(self):
        """Test that ImageHasher can be created."""
        hasher = ImageHasher()
        assert hasher is not None

    @patch("photoflow.core.duplicate_detection.default_processor")
    @patch("photoflow.core.duplicate_detection.load_image_asset")
    def test_hash_image_asset(self, mock_load_image_asset, mock_processor):
        """Test hashing an ImageAsset."""
        mock_image = Mock()
        mock_image.size = 64, 64
        mock_image.mode = "L"
        mock_gray = Mock()
        mock_gray.resize.return_value = mock_gray
        mock_gray.getdata.return_value = [(i % 256) for i in range(64)]
        mock_image.convert.return_value = mock_gray
        mock_processor.load_pil_image.return_value = mock_image
        hasher = ImageHasher()
        image_asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(64, 64), mode="RGB")
        result = hasher.hash_image(image_asset)
        assert isinstance(result, PerceptualHash)
        assert len(result.hash_value) == 16

    @patch("photoflow.core.duplicate_detection.default_processor")
    @patch("photoflow.core.duplicate_detection.load_image_asset")
    def test_hash_image_path(self, mock_load_image_asset, mock_processor):
        """Test hashing an image from file path."""
        mock_image = Mock()
        mock_image.size = 64, 64
        mock_image.mode = "L"
        mock_gray = Mock()
        mock_gray.resize.return_value = mock_gray
        mock_gray.getdata.return_value = [(i % 256) for i in range(64)]
        mock_image.convert.return_value = mock_gray
        mock_processor.load_pil_image.return_value = mock_image
        dummy_asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(64, 64), mode="RGB")
        mock_load_image_asset.return_value = dummy_asset
        hasher = ImageHasher()
        result = hasher.hash_image(Path("test.jpg"))
        assert result is not None
        mock_processor.load_pil_image.assert_called_once()

    def test_hash_reproducibility(self):
        """Test that hashing the same image produces the same hash."""
        hasher = ImageHasher()
        with (
            patch("photoflow.core.duplicate_detection.default_processor") as mock_processor,
            patch("photoflow.core.duplicate_detection.load_image_asset") as mock_load_image_asset,
        ):
            mock_image = Mock()
            mock_image.size = 64, 64
            mock_image.mode = "L"
            mock_gray = Mock()
            mock_gray.resize.return_value = mock_gray
            mock_gray.getdata.return_value = [128] * 64
            mock_image.convert.return_value = mock_gray
            mock_processor.load_pil_image.return_value = mock_image
            dummy_asset = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(64, 64), mode="RGB")
            mock_load_image_asset.return_value = dummy_asset
            hash1 = hasher.hash_image(Path("test.jpg"))
            hash2 = hasher.hash_image(Path("test.jpg"))
            assert hash1 == hash2


class TestDuplicateDetector:
    """Test duplicate detection functionality."""

    def test_duplicate_detector_creation(self):
        """Test that DuplicateDetector can be created."""
        detector = DuplicateDetector()
        assert detector is not None
        assert detector.threshold == 5

    def test_duplicate_detector_custom_threshold(self):
        """Test DuplicateDetector with custom threshold."""
        detector = DuplicateDetector(threshold=10)
        assert detector.threshold == 10

    def test_find_duplicates_empty_list(self):
        """Test finding duplicates in empty image list."""
        detector = DuplicateDetector()
        result = detector.find_duplicates([])
        assert result == []

    @patch("photoflow.core.duplicate_detection.ImageHasher")
    def test_find_duplicates_no_duplicates(self, mock_hasher_class):
        """Test finding duplicates when there are none."""
        mock_hasher = Mock()
        mock_hasher.hash_image.side_effect = [
            PerceptualHash("0000000000000000"),
            PerceptualHash("ffffffffffffffff"),
        ]
        mock_hasher_class.return_value = mock_hasher
        detector = DuplicateDetector()
        images = [Path("image1.jpg"), Path("image2.jpg")]
        result = detector.find_duplicates(images)
        assert result == []

    @patch("photoflow.core.duplicate_detection.ImageHasher")
    def test_find_duplicates_with_duplicates(self, mock_hasher_class):
        """Test finding actual duplicates."""
        mock_hasher = Mock()
        mock_hasher.hash_image.side_effect = [
            PerceptualHash("0000000000000000"),
            PerceptualHash("0000000000000001"),
            PerceptualHash("ffffffffffffffff"),
        ]
        mock_hasher_class.return_value = mock_hasher
        detector = DuplicateDetector(threshold=5)
        images = [Path("image1.jpg"), Path("image2.jpg"), Path("image3.jpg")]
        result = detector.find_duplicates(images)
        assert len(result) == 1
        duplicate_group = result[0]
        assert len(duplicate_group) == 2
        assert Path("image1.jpg") in duplicate_group
        assert Path("image2.jpg") in duplicate_group
        assert Path("image3.jpg") not in duplicate_group

    def test_group_similar_images(self):
        """Test grouping similar images by hash similarity."""
        detector = DuplicateDetector(threshold=5)
        hash_data = [
            (Path("img1.jpg"), PerceptualHash("0000000000000000")),
            (Path("img2.jpg"), PerceptualHash("0000000000000001")),
            (Path("img3.jpg"), PerceptualHash("ffffffffffffffff")),
            (Path("img4.jpg"), PerceptualHash("ffffffffffffff0f")),
        ]
        result = detector._group_similar_images(hash_data)
        assert len(result) == 2
        group_paths = [set(group) for group in result]
        assert {Path("img1.jpg"), Path("img2.jpg")} in group_paths
        assert {Path("img3.jpg"), Path("img4.jpg")} in group_paths


class TestDuplicateDetectionCLI:
    """Test CLI integration for duplicate detection."""

    def test_duplicate_detection_action_exists(self):
        """Test that duplicate detection can be accessed via CLI."""
        detector = DuplicateDetector()
        assert hasattr(detector, "find_duplicates")
        assert callable(detector.find_duplicates)
