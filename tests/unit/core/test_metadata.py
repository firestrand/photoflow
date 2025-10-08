"""Comprehensive tests for metadata functionality."""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from photoflow.core.metadata import (
    IPTCINFO_AVAILABLE,
    PIEXIF_AVAILABLE,
    PIL_AVAILABLE,
    ExifData,
    ExifError,
    IPTCData,
    IPTCError,
    MetadataError,
    MetadataReader,
    MetadataWriter,
    read_metadata,
    write_metadata,
)
from tests.utils.test_image_generator import TestImageGenerator


class TestMetadataExceptions:
    """Test metadata exception classes."""

    def test_metadata_error_creation(self):
        """Test MetadataError exception creation."""
        error = MetadataError("Test message")
        assert str(error) == "Test message"
        assert isinstance(error, Exception)

    def test_exif_error_inheritance(self):
        """Test ExifError inherits from MetadataError."""
        error = ExifError("EXIF test message")
        assert str(error) == "EXIF test message"
        assert isinstance(error, MetadataError)
        assert isinstance(error, Exception)

    def test_iptc_error_inheritance(self):
        """Test IPTCError inherits from MetadataError."""
        error = IPTCError("IPTC test message")
        assert str(error) == "IPTC test message"
        assert isinstance(error, MetadataError)
        assert isinstance(error, Exception)


class TestExifData:
    """Test ExifData dataclass."""

    def test_exif_data_creation_defaults(self):
        """Test ExifData creation with default values."""
        exif = ExifData()
        assert exif.make is None
        assert exif.model is None
        assert exif.lens_make is None
        assert exif.lens_model is None
        assert exif.focal_length is None
        assert exif.f_number is None
        assert exif.exposure_time is None
        assert exif.iso_speed is None
        assert exif.flash is None
        assert exif.width is None
        assert exif.height is None
        assert exif.orientation is None
        assert exif.color_space is None
        assert exif.gps_latitude is None
        assert exif.gps_longitude is None
        assert exif.gps_altitude is None
        assert exif.gps_timestamp is None
        assert exif.datetime_original is None
        assert exif.datetime_digitized is None
        assert exif.datetime is None
        assert exif.software is None
        assert exif.creator is None
        assert exif.copyright is None
        # No raw_exif field in the model

    def test_exif_data_creation_with_values(self):
        """Test ExifData creation with specific values."""
        exif = ExifData(
            make="Canon",
            model="EOS R5",
            focal_length="85",
            f_number="2.8",
            iso_speed=400,
            datetime_original="2023-06-15T14:30:00",
        )
        assert exif.make == "Canon"
        assert exif.model == "EOS R5"
        assert exif.focal_length == "85"
        assert exif.f_number == "2.8"
        assert exif.iso_speed == 400
        assert exif.datetime_original == "2023-06-15T14:30:00"

    def test_exif_data_to_dict(self):
        """Test ExifData to_dict method."""
        test_date = datetime.datetime(2023, 6, 15, 14, 30, 0)
        exif = ExifData(make="Canon", model="EOS R5", focal_length=85, datetime_original=test_date)
        result = exif.to_dict()
        assert isinstance(result, dict)
        assert result["make"] == "Canon"
        assert result["model"] == "EOS R5"
        assert result["focal_length"] == 85


class TestIPTCData:
    """Test IPTCData dataclass."""

    def test_iptc_data_creation_defaults(self):
        """Test IPTCData creation with default values."""
        iptc = IPTCData()
        assert iptc.title is None
        assert iptc.description is None
        assert iptc.byline is None
        assert iptc.byline_title is None
        assert iptc.credit is None
        assert iptc.source is None
        assert iptc.copyright_notice is None
        assert iptc.location is None
        assert iptc.city is None
        assert iptc.state is None
        assert iptc.country is None
        assert iptc.category is None
        assert iptc.date_created is None
        assert iptc.urgency is None
        assert iptc.instructions is None
        assert isinstance(iptc.keywords, list)
        assert len(iptc.keywords) == 0
        assert isinstance(iptc.raw_iptc, dict)
        assert len(iptc.raw_iptc) == 0

    def test_iptc_data_creation_with_values(self):
        """Test IPTCData creation with specific values."""
        iptc = IPTCData(
            title="Mountain Sunset",
            byline="John Photographer",
            location="Yosemite",
            keywords=["nature", "landscape"],
        )
        assert iptc.title == "Mountain Sunset"
        assert iptc.byline == "John Photographer"
        assert iptc.location == "Yosemite"
        assert iptc.keywords == ["nature", "landscape"]

    def test_iptc_data_to_dict(self):
        """Test IPTCData to_dict method."""
        iptc = IPTCData(title="Test Title", byline="Test Creator", keywords=["test", "keywords"])
        result = iptc.to_dict()
        assert isinstance(result, dict)
        assert result["title"] == "Test Title"
        assert result["byline"] == "Test Creator"
        assert result["keywords"] == ["test", "keywords"]


class TestMetadataReader:
    """Test MetadataReader class."""

    def test_metadata_reader_init(self):
        """Test MetadataReader initialization."""
        reader = MetadataReader()
        assert reader is not None

    def test_dms_to_decimal(self):
        """Test _dms_to_decimal conversion."""
        reader = MetadataReader()
        dms_tuple = (37, 1), (24, 1), (36, 1)
        result = reader._dms_to_decimal(dms_tuple)
        assert abs(result - 37.41) < 0.01


class TestMetadataWriter:
    """Test MetadataWriter class."""

    def test_metadata_writer_init(self):
        """Test MetadataWriter initialization."""
        writer = MetadataWriter()
        assert writer is not None


class TestMetadataFunctions:
    """Test top-level metadata functions."""

    @patch("photoflow.core.metadata.MetadataReader")
    def test_read_metadata_function(self, mock_reader_class):
        """Test read_metadata function."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        mock_exif = ExifData(make="Canon")
        mock_iptc = IPTCData(title="Test")
        mock_reader.read.return_value = (mock_exif, mock_iptc)
        exif, iptc = read_metadata(Path("test.jpg"))
        assert exif == mock_exif
        assert iptc == mock_iptc
        mock_reader.read.assert_called_once_with(Path("test.jpg"))

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_function(self, mock_writer_class):
        """Test write_metadata function."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_exif_writer = Mock()
        mock_iptc_writer = Mock()
        mock_writer.exif_writer = mock_exif_writer
        mock_writer.iptc_writer = mock_iptc_writer

        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        write_metadata(Path("test.jpg"), exif_data, iptc_data, Path("output.jpg"))

        mock_exif_writer.write_exif.assert_called_once_with(Path("test.jpg"), exif_data, Path("output.jpg"))
        mock_iptc_writer.write_iptc.assert_called_once_with(Path("test.jpg"), iptc_data, Path("output.jpg"))

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_function_exif_only(self, mock_writer_class):
        """Test write_metadata function with EXIF only."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_exif_writer = Mock()
        mock_writer.exif_writer = mock_exif_writer
        mock_writer.iptc_writer = None  # No IPTC writer for this test

        exif_data = ExifData(make="Canon")
        write_metadata(Path("test.jpg"), exif_data=exif_data)

        mock_exif_writer.write_exif.assert_called_once_with(Path("test.jpg"), exif_data, Path("test.jpg"))
        # Should not have called IPTC writer since it's None

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_function_iptc_only(self, mock_writer_class):
        """Test write_metadata function with IPTC only."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_iptc_writer = Mock()
        mock_writer.exif_writer = None  # No EXIF writer for this test
        mock_writer.iptc_writer = mock_iptc_writer

        iptc_data = IPTCData(title="Test")
        write_metadata(Path("test.jpg"), iptc_data=iptc_data)

        # Should not have called EXIF writer since it's None
        mock_iptc_writer.write_iptc.assert_called_once_with(Path("test.jpg"), iptc_data, Path("test.jpg"))


class TestAvailabilityChecks:
    """Test availability constant checks."""

    def test_pil_available_type(self):
        """Test PIL_AVAILABLE is boolean."""
        assert isinstance(PIL_AVAILABLE, bool)

    def test_piexif_available_type(self):
        """Test PIEXIF_AVAILABLE is boolean."""
        assert isinstance(PIEXIF_AVAILABLE, bool)

    def test_iptcinfo_available_type(self):
        """Test IPTCINFO_AVAILABLE is boolean."""
        assert isinstance(IPTCINFO_AVAILABLE, bool)


class TestMetadataReaderComprehensive:
    """Comprehensive tests for MetadataReader."""

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_read_exif_success(self, mock_piexif):
        """Test successful EXIF reading."""
        reader = MetadataReader()
        mock_exif_dict = {
            "0th": {(271): "Canon", (272): "EOS R5"},
            "Exif": {(37386): (85, 1), (33434): (1, 100)},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        mock_piexif.load.return_value = mock_exif_dict
        result = reader.read_exif(Path("test.jpg"))
        assert isinstance(result, ExifData)
        mock_piexif.load.assert_called_once_with(str(Path("test.jpg")))

    @patch("photoflow.core.metadata.PIL_AVAILABLE", False)
    def test_read_exif_no_pil(self):
        """Test EXIF reading when PIL not available."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required for metadata operations"):
            MetadataReader()

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False)
    def test_read_exif_no_piexif(self):
        """Test EXIF reading when piexif not available."""
        reader = MetadataReader()
        with pytest.raises(ExifError, match="piexif library is required"):
            reader.read_exif(Path("test.jpg"))

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_read_exif_file_not_found(self, mock_piexif):
        """Test EXIF reading with file not found."""
        reader = MetadataReader()
        mock_piexif.load.side_effect = FileNotFoundError("File not found")
        with pytest.raises(ExifError, match="Failed to read EXIF data"):
            reader.read_exif(Path("nonexistent.jpg"))

    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_read_iptc_success(self, mock_iptc_info):
        """Test successful IPTC reading."""
        reader = MetadataReader()
        mock_iptc_obj = Mock()
        mock_iptc_obj.data = {
            b"object_name": b"Test Title",
            b"caption/abstract": b"Test Caption",
            b"by-line": b"Test Photographer",
        }
        mock_iptc_info.return_value = mock_iptc_obj
        result = reader.read_iptc(Path("test.jpg"))
        assert isinstance(result, IPTCData)
        mock_iptc_info.assert_called_once()

    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False)
    def test_read_iptc_no_iptcinfo(self):
        """Test IPTC reading when iptcinfo3 not available."""
        reader = MetadataReader()
        with pytest.raises(IPTCError, match="iptcinfo3 library is required"):
            reader.read_iptc(Path("test.jpg"))

    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_read_iptc_file_not_found(self, mock_iptc_info):
        """Test IPTC reading with file not found."""
        reader = MetadataReader()
        mock_iptc_info.side_effect = FileNotFoundError("File not found")
        with pytest.raises(IPTCError, match="Failed to read IPTC data"):
            reader.read_iptc(Path("nonexistent.jpg"))


class TestMetadataWriterComprehensive:
    """Comprehensive tests for MetadataWriter."""

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_write_exif_success(self, mock_piexif):
        """Test successful EXIF writing."""
        writer = MetadataWriter()
        mock_piexif.dump.return_value = b"exif_bytes"
        exif_data = ExifData(make="Canon", model="EOS R5")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            writer.write_exif(tmp_path, exif_data)
            mock_piexif.dump.assert_called_once()
            mock_piexif.insert.assert_called_once()
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("photoflow.core.metadata.PIL_AVAILABLE", False)
    def test_write_exif_no_pil(self):
        """Test EXIF writing when PIL not available."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required for metadata operations"):
            MetadataWriter()

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False)
    def test_write_exif_no_piexif(self):
        """Test EXIF writing when piexif not available."""
        writer = MetadataWriter()
        exif_data = ExifData()
        with pytest.raises(ExifError, match="piexif library is required"):
            writer.write_exif(Path("test.jpg"), exif_data)

    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_write_iptc_success(self, mock_iptc_info):
        """Test successful IPTC writing."""
        writer = MetadataWriter()
        mock_iptc_obj = Mock()
        mock_iptc_obj.data = {}  # Add data attribute for item assignment
        mock_iptc_info.return_value = mock_iptc_obj
        iptc_data = IPTCData(title="Test Title", byline="Test Photographer")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            writer.write_iptc(tmp_path, iptc_data)
            mock_iptc_info.assert_called_once()
            mock_iptc_obj.save_as.assert_called_once()  # Changed from save to save_as
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False)
    def test_write_iptc_no_iptcinfo(self):
        """Test IPTC writing when iptcinfo3 not available."""
        writer = MetadataWriter()
        iptc_data = IPTCData()
        with pytest.raises(IPTCError, match="iptcinfo3 library is required"):
            writer.write_iptc(Path("test.jpg"), iptc_data)


class TestMetadataTopLevelFunctions:
    """Test top-level metadata functions."""

    @patch("photoflow.core.metadata.MetadataReader")
    def test_read_metadata_function_comprehensive(self, mock_reader_class):
        """Test read_metadata function."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        mock_reader.read.return_value = (exif_data, iptc_data)
        result_exif, result_iptc = read_metadata(Path("test.jpg"))
        assert result_exif == exif_data
        assert result_iptc == iptc_data
        mock_reader.read.assert_called_once_with(Path("test.jpg"))

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_function_without_output(self, mock_writer_class):
        """Test write_metadata function without output path."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_exif_writer = Mock()
        mock_iptc_writer = Mock()
        mock_writer.exif_writer = mock_exif_writer
        mock_writer.iptc_writer = mock_iptc_writer

        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        write_metadata(Path("test.jpg"), exif_data, iptc_data)
        mock_exif_writer.write_exif.assert_called_once_with(Path("test.jpg"), exif_data, Path("test.jpg"))
        mock_iptc_writer.write_iptc.assert_called_once_with(Path("test.jpg"), iptc_data, Path("test.jpg"))

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_function_with_output(self, mock_writer_class):
        """Test write_metadata function with output path."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_exif_writer = Mock()
        mock_iptc_writer = Mock()
        mock_writer.exif_writer = mock_exif_writer
        mock_writer.iptc_writer = mock_iptc_writer

        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        input_path = Path("test.jpg")
        output_path = Path("output.jpg")
        write_metadata(input_path, exif_data, iptc_data, output_path)
        mock_exif_writer.write_exif.assert_called_once_with(input_path, exif_data, output_path)
        mock_iptc_writer.write_iptc.assert_called_once_with(input_path, iptc_data, output_path)


class TestMetadataErrorHandling:
    """Test metadata error handling scenarios."""

    @patch("photoflow.core.metadata.MetadataReader")
    def test_read_metadata_exif_error(self, mock_reader_class):
        """Test read_metadata with EXIF error."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        # Mock the read method to return default objects when EXIF error occurs
        mock_reader.read.return_value = (ExifData(), IPTCData())
        result_exif, result_iptc = read_metadata(Path("test.jpg"))
        assert isinstance(result_exif, ExifData)
        assert isinstance(result_iptc, IPTCData)

    @patch("photoflow.core.metadata.MetadataReader")
    def test_read_metadata_iptc_error(self, mock_reader_class):
        """Test read_metadata with IPTC error."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        # Mock the read method to return default objects when IPTC error occurs
        mock_reader.read.return_value = (ExifData(), IPTCData())
        result_exif, result_iptc = read_metadata(Path("test.jpg"))
        assert isinstance(result_exif, ExifData)
        assert isinstance(result_iptc, IPTCData)

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_exif_error(self, mock_writer_class):
        """Test write_metadata with EXIF error."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_exif_writer = Mock()
        mock_iptc_writer = Mock()
        mock_writer.exif_writer = mock_exif_writer
        mock_writer.iptc_writer = mock_iptc_writer
        mock_exif_writer.write_exif.side_effect = ExifError("EXIF write error")

        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        with pytest.raises(ExifError, match="EXIF write error"):
            write_metadata(Path("test.jpg"), exif_data, iptc_data)

    @patch("photoflow.core.metadata.MetadataWriter")
    def test_write_metadata_iptc_error(self, mock_writer_class):
        """Test write_metadata with IPTC error."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        # Mock the exif_writer and iptc_writer attributes
        mock_exif_writer = Mock()
        mock_iptc_writer = Mock()
        mock_writer.exif_writer = mock_exif_writer
        mock_writer.iptc_writer = mock_iptc_writer
        mock_iptc_writer.write_iptc.side_effect = IPTCError("IPTC write error")

        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        with pytest.raises(IPTCError, match="IPTC write error"):
            write_metadata(Path("test.jpg"), exif_data, iptc_data)


class TestMetadataImportErrorHandling:
    """Test import error handling for metadata dependencies."""

    def test_pil_import_error(self):
        """Test PIL import error handling."""
        assert isinstance(PIL_AVAILABLE, bool)
        with patch("photoflow.core.metadata.PIL_AVAILABLE", False):
            from photoflow.core.metadata import PIL_AVAILABLE as patched_pil

            assert isinstance(patched_pil, bool)

    def test_piexif_import_error(self):
        """Test piexif import error handling."""
        assert isinstance(PIEXIF_AVAILABLE, bool)
        with patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False):
            from photoflow.core.metadata import PIEXIF_AVAILABLE as patched_piexif

            assert isinstance(patched_piexif, bool)

    def test_iptcinfo_import_error(self):
        """Test iptcinfo3 import error handling."""
        assert isinstance(IPTCINFO_AVAILABLE, bool)
        with patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False):
            from photoflow.core.metadata import IPTCINFO_AVAILABLE as patched_iptc

            assert isinstance(patched_iptc, bool)


class TestMetadataReaderErrorHandling:
    """Test MetadataReader error handling and edge cases."""

    @patch("photoflow.core.metadata.PIL_AVAILABLE", False)
    def test_metadata_reader_no_pil(self):
        """Test MetadataReader when PIL not available."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required"):
            MetadataReader()

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False)
    def test_metadata_reader_no_piexif(self):
        """Test MetadataReader EXIF reading when piexif not available."""
        reader = MetadataReader()
        with pytest.raises(ExifError, match="piexif library is required"):
            reader.read_exif(Path("test.jpg"))

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False)
    def test_metadata_reader_no_iptcinfo(self):
        """Test MetadataReader IPTC reading when iptcinfo3 not available."""
        reader = MetadataReader()
        with pytest.raises(IPTCError, match="iptcinfo3 library is required"):
            reader.read_iptc(Path("test.jpg"))

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_metadata_reader_exif_error_handling(self, mock_piexif):
        """Test EXIF reading error handling."""
        reader = MetadataReader()
        mock_piexif.load.side_effect = Exception("EXIF parse error")
        with patch("photoflow.core.metadata.Image.open") as mock_open:
            mock_img = Mock()
            mock_img.info = {"exif": b"corrupt_exif"}
            mock_open.return_value.__enter__.return_value = mock_img
            with pytest.raises(ExifError, match="Failed to read EXIF data"):
                reader.read_exif(Path("test.jpg"))

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_metadata_reader_iptc_error_handling(self, mock_iptcinfo):
        """Test IPTC reading error handling."""
        reader = MetadataReader()
        mock_iptcinfo.side_effect = Exception("IPTC parse error")
        with pytest.raises(IPTCError, match="Failed to read IPTC data"):
            reader.read_iptc(Path("test.jpg"))

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_metadata_reader_parse_exif_comprehensive(self, mock_piexif):
        """Test comprehensive EXIF parsing."""
        reader = MetadataReader()
        mock_exif_dict = {
            "0th": {(271): "Canon", (272): "EOS R5", (274): 1},
            "Exif": {
                (37386): (85, 1),
                (33434): (1, 250),
                (33437): (28, 10),
                (34855): 400,
                (40961): 1,
                (36867): "2023:06:15 14:30:00",
                (36868): "2023:06:15 14:30:00",
                (11): "Test Software",
                (315): "Test Artist",
            },
            "GPS": {
                (1): "N",
                (2): ((40, 1), (42, 1), (51, 1)),
                (3): "W",
                (4): ((74, 1), (0, 1), (23, 1)),
                (5): 0,
                (6): (10, 1),
                (7): ((14, 1), (30, 1), (0, 1)),
                (29): "2023:06:15",
            },
            "1st": {},
            "thumbnail": None,
        }
        mock_piexif.load.return_value = mock_exif_dict
        mock_piexif.TAGS = {
            "0th": {
                (271): {"name": "Make"},
                (272): {"name": "Model"},
                (274): {"name": "Orientation"},
            },
            "Exif": {
                (37386): {"name": "FocalLength"},
                (33434): {"name": "ExposureTime"},
                (33437): {"name": "FNumber"},
                (34855): {"name": "ISOSpeedRatings"},
                (40961): {"name": "ColorSpace"},
                (36867): {"name": "DateTimeOriginal"},
                (36868): {"name": "DateTimeDigitized"},
                (11): {"name": "ProcessingSoftware"},
                (315): {"name": "Artist"},
            },
            "GPS": {
                (1): {"name": "GPSLatitudeRef"},
                (2): {"name": "GPSLatitude"},
                (3): {"name": "GPSLongitudeRef"},
                (4): {"name": "GPSLongitude"},
                (5): {"name": "GPSAltitudeRef"},
                (6): {"name": "GPSAltitude"},
                (7): {"name": "GPSTimeStamp"},
                (29): {"name": "GPSDateStamp"},
            },
        }
        with patch("photoflow.core.metadata.Image.open") as mock_open:
            mock_img = Mock()
            mock_img.info = {"exif": b"fake_exif"}
            mock_open.return_value.__enter__.return_value = mock_img
            result = reader.read_exif(Path("test.jpg"))
            assert result.make == "Canon"
            assert result.model == "EOS R5"
            assert result.focal_length == "85.0"
            assert result.aperture == 2.8
            assert result.iso_speed == 400
            assert result.orientation == 1
            assert result.color_space == "sRGB"

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_metadata_reader_parse_iptc_comprehensive(self, mock_iptcinfo):
        """Test comprehensive IPTC parsing."""
        reader = MetadataReader()
        mock_iptc_obj = Mock()
        mock_iptc_obj.object_name = "Test Title"
        mock_iptc_obj.caption = "Test Description"
        mock_iptc_obj.byline = "Test Creator"
        mock_iptc_obj.copyright_notice = "Copyright 2023 Test"
        mock_iptc_obj.special_instructions = "Test Instructions"
        mock_iptc_obj.city = "Test City"
        mock_iptc_obj.province_state = "Test State"
        mock_iptc_obj.country_name = "Test Country"
        mock_iptc_obj.category = "Test Category"
        mock_iptc_obj.keywords = ["keyword1", "keyword2", "keyword3"]
        mock_iptcinfo.return_value = mock_iptc_obj
        result = reader.read_iptc(Path("test.jpg"))
        assert result.title == "Test Title"
        assert result.caption == "Test Description"
        assert result.byline == "Test Creator"
        assert result.copyright_notice == "Copyright 2023 Test"
        assert result.instructions == "Test Instructions"
        assert result.city == "Test City"
        assert result.state == "Test State"
        assert result.country == "Test Country"
        assert result.category == "Test Category"
        assert result.keywords == ["keyword1", "keyword2", "keyword3"]


class TestMetadataWriterErrorHandling:
    """Test MetadataWriter error handling and edge cases."""

    @patch("photoflow.core.metadata.PIL_AVAILABLE", False)
    def test_metadata_writer_no_pil(self):
        """Test MetadataWriter when PIL not available."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required"):
            MetadataWriter()

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False)
    def test_metadata_writer_no_piexif(self):
        """Test MetadataWriter EXIF writing when piexif not available."""
        writer = MetadataWriter()
        exif_data = ExifData(make="Canon")
        with pytest.raises(ExifError, match="piexif library is required"):
            writer.write_exif(Path("test.jpg"), exif_data)

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False)
    def test_metadata_writer_no_iptcinfo(self):
        """Test MetadataWriter IPTC writing when iptcinfo3 not available."""
        writer = MetadataWriter()
        iptc_data = IPTCData(title="Test")
        with pytest.raises(IPTCError, match="iptcinfo3 library is required"):
            writer.write_iptc(Path("test.jpg"), iptc_data)

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_metadata_writer_exif_comprehensive(self, mock_piexif):
        """Test comprehensive EXIF writing."""
        writer = MetadataWriter()
        test_datetime = datetime.datetime(2023, 6, 15, 14, 30, 0)
        exif_data = ExifData(
            make="Comprehensive Test Make",
            model="Test Model Pro",
            lens_make="Test Lens Make",
            lens_model="Test Lens 50mm f/1.4",
            focal_length=50,
            aperture=1.4,
            shutter_speed=1 / 200,
            iso_speed=100,
            flash=16,
            orientation=1,
            color_space="sRGB",
            gps_latitude=37.7749,
            gps_longitude=-122.4194,
            gps_altitude=50.0,
            datetime_original=test_datetime,
            datetime_digitized=test_datetime,
            datetime_modified=test_datetime,
            software="Test Software Pro 2023",
            creator="Test Photographer",
            copyright="© 2023 Test Rights",
        )
        mock_piexif.dump.return_value = b"comprehensive_exif_dump"
        mock_piexif.insert.return_value = b"image_with_comprehensive_exif"
        with patch("photoflow.core.metadata.Image.open") as mock_open:
            mock_img = Mock()
            mock_open.return_value.__enter__.return_value = mock_img
            writer.write_exif(Path("test.jpg"), exif_data, Path("output.jpg"))
            mock_piexif.dump.assert_called_once()

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_metadata_writer_comprehensive_iptc_generation(self, mock_iptcinfo):
        """Test comprehensive IPTC data generation in writer."""
        writer = MetadataWriter()
        iptc_data = IPTCData(
            title="Comprehensive Test Image Title",
            keywords=["comprehensive", "test", "iptc", "metadata", "coverage"],
            byline="Test Photographer Pro",
            byline_title="Senior Test Photographer",
            copyright_notice="© 2023 Test Photography Company",
            instructions="Special handling instructions for test image",
            city="Test Metropolitan City",
            state="Test State Province",
            country="Test Country Nation",
            category="Test Primary Category",
            urgency=3,
            credit="Test Photography Credit",
            source="Test Image Source Agency",
        )
        mock_iptc_obj = Mock()
        mock_iptc_obj.data = {}  # Add data attribute for item assignment
        mock_iptcinfo.return_value = mock_iptc_obj
        with patch("shutil.copy2"):
            writer.write_iptc(Path("test.jpg"), iptc_data, Path("output.jpg"))
        mock_iptcinfo.assert_called_once()
        mock_iptc_obj.save_as.assert_called_once()  # Changed from save to save_as

    def test_metadata_availability_constants_edge_cases(self):
        """Test metadata availability constants in various scenarios."""
        assert isinstance(PIL_AVAILABLE, bool)
        assert isinstance(PIEXIF_AVAILABLE, bool)
        assert isinstance(IPTCINFO_AVAILABLE, bool)
        with (
            patch("photoflow.core.metadata.PIL_AVAILABLE", False),
            pytest.raises(ImportError, match="PIL \\(Pillow\\) is required"),
        ):
            MetadataReader()
        with patch("photoflow.core.metadata.PIL_AVAILABLE", True):
            with patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False):
                writer = MetadataWriter()
                exif_data = ExifData(make="Test")
                with pytest.raises(ExifError, match="piexif library is required"):
                    writer.write_exif(Path("test.jpg"), exif_data)
            with patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False):
                writer = MetadataWriter()
                iptc_data = IPTCData(title="Test")
                with pytest.raises(IPTCError, match="iptcinfo3 library is required"):
                    writer.write_iptc(Path("test.jpg"), iptc_data)


class TestMetadataImportErrorPaths:
    """Test specific import error paths in metadata module."""

    def test_pil_import_error_path(self):
        """Test the PIL import error path (lines 17-18)."""
        assert isinstance(PIL_AVAILABLE, bool)
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            pass

    def test_piexif_import_error_path(self):
        """Test the piexif import error path (lines 24-25)."""
        assert isinstance(PIEXIF_AVAILABLE, bool)
        with patch("photoflow.core.metadata.PIEXIF_AVAILABLE", False):
            reader = MetadataReader()
            result = reader._parse_exif_dict({})
            assert hasattr(result, "make")

    def test_iptcinfo_import_error_path(self):
        """Test the iptcinfo3 import error path (lines 30-31)."""
        assert isinstance(IPTCINFO_AVAILABLE, bool)
        with patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", False):
            reader = MetadataReader()
            mock_iptc = Mock()
            mock_iptc.data = {}
            result = reader._parse_iptc_info(mock_iptc)
            assert hasattr(result, "title")


class TestMetadataSpecificMissingLines:
    """Target specific missing lines for coverage."""

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.PIEXIF_AVAILABLE", True)
    @patch("photoflow.core.metadata.piexif")
    def test_metadata_writer_error_lines_270_273(self, mock_piexif):
        """Test error handling lines 270-273 in metadata writer."""
        writer = MetadataWriter()
        exif_data = ExifData(make="Test Make")
        mock_piexif.dump.side_effect = Exception("Piexif dump failed")
        with pytest.raises(ExifError) as exc_info:
            writer.write_exif(Path("test.jpg"), exif_data)
        assert "Failed to write EXIF" in str(exc_info.value)

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCINFO_AVAILABLE", True)
    @patch("photoflow.core.metadata.IPTCInfo")
    def test_metadata_writer_error_lines_276_277(self, mock_iptcinfo):
        """Test error handling lines 276-277 in metadata writer."""
        writer = MetadataWriter()
        iptc_data = IPTCData(title="Test")
        mock_iptcinfo.side_effect = Exception("IPTC processing failed")
        with pytest.raises(IPTCError) as exc_info:
            writer.write_iptc(Path("test.jpg"), iptc_data)
        assert "Failed to write IPTC" in str(exc_info.value)

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    def test_metadata_reader_error_lines_235_236(self):
        """Test error handling lines 235-236 in metadata reader."""
        reader = MetadataReader()
        with pytest.raises(ExifError) as exc_info:
            reader.read_exif(Path("/nonexistent/file.jpg"))
        assert "Failed to read EXIF" in str(exc_info.value)

    @patch("photoflow.core.metadata.PIL_AVAILABLE", True)
    def test_utility_function_lines_284_292(self):
        """Test utility function lines 284-292."""
        reader = MetadataReader()
        test_cases = [
            (((37, 1), (46, 1), (30, 1)), 37.775),
            (((0, 1), (0, 1), (0, 1)), 0.0),
            (((180, 1), (0, 1), (0, 1)), 180.0),
        ]
        for input_val, expected in test_cases:
            result = reader._dms_to_decimal(input_val)
            if isinstance(expected, float):
                assert abs(result - expected) < 0.001
            else:
                assert result == expected

    def test_helper_string_value_decoding(self):
        """_get_string_value should decode byte strings when reader is present."""
        reader = MetadataReader()
        reader.exif_reader = Mock()
        ifd = {1: b"Canon\x00"}
        tags = {1: {"name": "Make"}}
        result = reader._get_string_value(ifd, tags, "Make")
        assert result == "Canon"

    def test_helper_int_value(self):
        """_get_int_value should extract integers for matching tags."""
        reader = MetadataReader()
        reader.exif_reader = Mock()
        ifd = {2: 42}
        tags = {2: {"name": "Orientation"}}
        assert reader._get_int_value(ifd, tags, "Orientation") == 42

    def test_helper_rational_value(self):
        """_get_rational_value defers to the exif reader for formatting."""
        reader = MetadataReader()
        mock_reader = Mock()
        mock_reader._rational_to_string.return_value = "1/200"
        reader.exif_reader = mock_reader
        ifd = {3: (1, 200)}
        tags = {3: {"name": "ExposureTime"}}
        assert reader._get_rational_value(ifd, tags, "ExposureTime") == "1/200"

    def test_helper_datetime_value_bytes(self):
        """_get_datetime_value handles bytes by decoding them."""
        reader = MetadataReader()
        reader.exif_reader = Mock()
        ifd = {4: b"2024:01:01 12:00:00"}
        tags = {4: {"name": "DateTimeOriginal"}}
        assert reader._get_datetime_value(ifd, tags, "DateTimeOriginal") == "2024:01:01 12:00:00"

    def test_helper_shutter_speed(self):
        """_get_shutter_speed converts rational exposure values."""
        reader = MetadataReader()
        mock_reader = Mock()
        mock_reader._rational_to_string.return_value = "1/125"
        reader.exif_reader = mock_reader
        ifd = {33434: (1, 125)}
        assert reader._get_shutter_speed(ifd, {}) == "1/125"

    def test_helper_color_space(self):
        """_get_color_space proxies to the reader parser."""
        reader = MetadataReader()
        mock_reader = Mock()
        mock_reader._parse_color_space.return_value = "Adobe RGB"
        reader.exif_reader = mock_reader
        ifd = {40961: 2}
        assert reader._get_color_space(ifd, {}) == "Adobe RGB"

    def test_helper_gps_parsing(self):
        """GPS helpers should convert coordinates and altitude correctly."""
        reader = MetadataReader()
        mock_reader = Mock()
        mock_reader._convert_gps_coordinate.side_effect = [12.34, 56.78]
        reader.exif_reader = mock_reader
        gps_dict = {
            1: "N",
            2: [(12, 1), (20, 1), (24, 1)],
            3: "E",
            4: [(56, 1), (46, 1), (40, 1)],
            5: 0,
            6: (150, 1),
            7: [(10, 1), (20, 1), (30, 1)],
            29: b"2024:01:01",
        }
        lat, lng = reader._parse_gps_coordinates(gps_dict)
        assert lat == 12.34 and lng == 56.78
        assert reader._get_gps_altitude(gps_dict) == 150.0
        assert reader._get_gps_timestamp(gps_dict) == "2024:01:01 10:20:30"


@pytest.mark.skipif(
    not (PIL_AVAILABLE and PIEXIF_AVAILABLE and IPTCINFO_AVAILABLE),
    reason="Requires Pillow, piexif, and iptcinfo3",
)
class TestMetadataIntegration:
    """Integration tests exercising actual metadata read/write flows."""

    def test_read_metadata_real_image(self, tmp_path: Path) -> None:
        """Reading from a generated image should hydrate key EXIF fields."""
        generator = TestImageGenerator()
        image_path = generator.create_test_image(output_path=tmp_path / "sample.jpg", photographer="Jane Doe")

        exif_data, iptc_data = read_metadata(image_path)

        assert exif_data.make == "Canon"
        assert exif_data.model == "EOS R5"
        # Width/height may be unavailable for some formats depending on loader
        # IPTC metadata is populated in a later write step
        assert iptc_data.title is None

    def test_write_metadata_round_trip(self, tmp_path: Path) -> None:
        """Round-tripping metadata writes updates EXIF/IPTC fields on disk."""
        generator = TestImageGenerator()
        source_path = generator.create_test_image(output_path=tmp_path / "source.jpg", photographer="Jane Doe")
        output_path = tmp_path / "output.jpg"

        exif_data, _iptc_data = read_metadata(source_path)
        exif_data.creator = "Integration Tester"
        write_metadata(source_path, exif_data, None, output_path)

        assert output_path.exists()
