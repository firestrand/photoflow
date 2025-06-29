"""Test basic package functionality."""

import photoflow


class TestPackage:
    """Test basic package setup."""

    def test_version_exists(self) -> None:
        """Test that version is defined."""
        assert hasattr(photoflow, "__version__")
        assert photoflow.__version__ == "0.1.0"

    def test_author_exists(self) -> None:
        """Test that author is defined."""
        assert hasattr(photoflow, "__author__")
        assert photoflow.__author__ == "Travis Silvers"

    def test_description_exists(self) -> None:
        """Test that description is defined."""
        assert hasattr(photoflow, "__description__")
        assert photoflow.__description__ == "Python batch photo or image manipulation"
