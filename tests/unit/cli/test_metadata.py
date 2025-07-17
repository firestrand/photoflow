"""Working tests for CLI metadata commands."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

from click.testing import CliRunner

from photoflow.cli.metadata import (
    _display_metadata_table,
    copy,
    inspect,
    metadata,
    remove,
    write,
)
from photoflow.core.metadata import ExifData, IPTCData, MetadataError


class TestMetadataGroupCommand:
    """Test the metadata command group."""

    def test_metadata_group_help(self):
        """Test metadata group help command."""
        runner = CliRunner()
        result = runner.invoke(metadata, ["--help"])
        assert result.exit_code == 0
        assert "Metadata inspection and manipulation commands" in result.output


class TestInspectCommand:
    """Test metadata inspect command."""

    @patch("photoflow.cli.metadata.read_metadata")
    def test_inspect_table_format(self, mock_read_metadata):
        """Test inspect command with table format."""
        # Mock metadata
        exif_data = ExifData(make="Canon", model="EOS R5", focal_length="85mm")
        iptc_data = IPTCData(title="Test Image", byline="Test Photographer")
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(inspect, [tmp_path, "--format", "table"])
            if result.exit_code != 0:
                print(f"Command failed with output: {result.output}")
            assert result.exit_code == 0
            # Should call read_metadata
            mock_read_metadata.assert_called_once_with(Path(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.read_metadata")
    def test_inspect_json_format(self, mock_read_metadata):
        """Test inspect command with JSON format."""
        exif_data = ExifData(make="Canon", model="EOS R5")
        iptc_data = IPTCData(title="Test Image")
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(inspect, [tmp_path, "--format", "json"])
            assert result.exit_code == 0
            # Should contain JSON output
            assert '"exif"' in result.output
            assert '"iptc"' in result.output
            mock_read_metadata.assert_called_once_with(Path(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.read_metadata")
    def test_inspect_yaml_format(self, mock_read_metadata):
        """Test inspect command with YAML format."""
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test")
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(inspect, [tmp_path, "--format", "yaml"])
            assert result.exit_code == 0
            mock_read_metadata.assert_called_once_with(Path(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.read_metadata")
    def test_inspect_exif_only(self, mock_read_metadata):
        """Test inspect command with EXIF only flag."""
        exif_data = ExifData(make="Canon", model="EOS R5")
        iptc_data = IPTCData(title="Test Image")
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(inspect, [tmp_path, "--exif-only", "--format", "json"])
            assert result.exit_code == 0
            # Should contain EXIF but not IPTC in JSON output
            assert '"exif"' in result.output
            assert '"iptc"' not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.read_metadata")
    def test_inspect_iptc_only(self, mock_read_metadata):
        """Test inspect command with IPTC only flag."""
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test Image")
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(inspect, [tmp_path, "--iptc-only", "--format", "json"])
            assert result.exit_code == 0
            # Should contain IPTC but not EXIF in JSON output
            assert '"iptc"' in result.output
            assert '"exif"' not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.read_metadata")
    def test_inspect_error_handling(self, mock_read_metadata):
        """Test inspect command error handling."""
        mock_read_metadata.side_effect = MetadataError("Test error")

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(inspect, [tmp_path])
            assert result.exit_code != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestWriteCommand:
    """Test metadata write command."""

    @patch("photoflow.cli.metadata.read_metadata")
    @patch("photoflow.cli.metadata.write_metadata")
    def test_write_basic(self, mock_write_metadata, mock_read_metadata):
        """Test basic write command."""
        # Mock metadata
        exif_data = ExifData()
        iptc_data = IPTCData()
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(write, [tmp_path, "--title", "Test Title", "--creator", "Test Creator"])
            if result.exit_code != 0:
                print(f"Command failed with output: {result.output}")
            assert result.exit_code == 0
            # Should call write_metadata
            mock_write_metadata.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.read_metadata")
    @patch("photoflow.cli.metadata.write_metadata")
    def test_write_with_output(self, mock_write_metadata, mock_read_metadata):
        """Test write command with output file."""
        # Mock metadata
        exif_data = ExifData()
        iptc_data = IPTCData()
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_input:
            input_path = tmp_input.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_output:
            output_path = tmp_output.name

        try:
            result = runner.invoke(write, [input_path, "--output", output_path, "--title", "Test Title"])
            assert result.exit_code == 0
            mock_write_metadata.assert_called_once()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.write_metadata")
    def test_write_error_handling(self, mock_write_metadata):
        """Test write command error handling."""
        mock_write_metadata.side_effect = MetadataError("Write error")

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(write, [tmp_path, "--title", "Test"])
            assert result.exit_code != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCopyCommand:
    """Test metadata copy command."""

    @patch("photoflow.cli.metadata.read_metadata")
    def test_copy_error_handling(self, mock_read_metadata):
        """Test copy command error handling."""
        mock_read_metadata.side_effect = MetadataError("Read error")

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_source:
            source_path = tmp_source.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_target:
            target_path = tmp_target.name

        try:
            result = runner.invoke(copy, [source_path, target_path])
            assert result.exit_code != 0
        finally:
            Path(source_path).unlink(missing_ok=True)
            Path(target_path).unlink(missing_ok=True)


class TestDisplayMetadataTableFunction:
    """Test the _display_metadata_table helper function."""

    @patch("photoflow.cli.metadata.console")
    def test_display_metadata_table_basic(self, mock_console):
        """Test basic metadata table display."""
        exif_data = ExifData(make="Canon", model="EOS R5", focal_length="85mm")
        iptc_data = IPTCData(title="Test Image", byline="Test Photographer")

        _display_metadata_table(Path("test.jpg"), exif_data, iptc_data, False, False)

        # Should call console.print
        assert mock_console.print.called

    @patch("photoflow.cli.metadata.console")
    def test_display_metadata_table_exif_only(self, mock_console):
        """Test metadata table display with EXIF only."""
        exif_data = ExifData(make="Canon", model="EOS R5")
        iptc_data = IPTCData(title="Test Image")

        _display_metadata_table(Path("test.jpg"), exif_data, iptc_data, True, False)

        assert mock_console.print.called

    @patch("photoflow.cli.metadata.console")
    def test_display_metadata_table_iptc_only(self, mock_console):
        """Test metadata table display with IPTC only."""
        exif_data = ExifData(make="Canon")
        iptc_data = IPTCData(title="Test Image", byline="Test Photographer")

        _display_metadata_table(Path("test.jpg"), exif_data, iptc_data, False, True)

        assert mock_console.print.called


class TestCLIMetadataIntegration:
    """Integration tests for CLI metadata commands."""

    def test_all_commands_exist(self):
        """Test that all expected commands exist in the metadata group."""
        runner = CliRunner()
        result = runner.invoke(metadata, ["--help"])
        assert result.exit_code == 0

        # Check that all commands are listed
        assert "inspect" in result.output
        assert "write" in result.output
        assert "remove" in result.output
        assert "copy" in result.output

    def test_command_help_pages(self):
        """Test that all commands have working help pages."""
        runner = CliRunner()
        commands = ["inspect", "write", "remove", "copy"]

        for cmd in commands:
            result = runner.invoke(metadata, [cmd, "--help"])
            assert result.exit_code == 0
            assert cmd in result.output.lower()

    def test_file_path_validation(self):
        """Test file path validation in commands."""
        runner = CliRunner()

        # Test with non-existent file
        result = runner.invoke(inspect, ["/nonexistent/file.jpg"])
        assert result.exit_code != 0


class TestRemoveCommand:
    """Test metadata remove command."""

    @patch("photoflow.cli.metadata.Image")
    def test_remove_exif_only(self, mock_image):
        """Test remove command with EXIF only."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = (1920, 1080)
        mock_img.format = "JPEG"
        mock_img.info = {"exif": b"fake_exif", "dpi": (72, 72)}
        mock_img.getdata.return_value = [0] * (1920 * 1080)

        mock_new_img = Mock()
        mock_image.open.return_value.__enter__.return_value = mock_img
        mock_image.new.return_value = mock_new_img

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(remove, [tmp_path, "--exif"])
            if result.exit_code != 0:
                print(f"Command failed with output: {result.output}")
            assert result.exit_code == 0
            mock_image.open.assert_called_once_with(Path(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.metadata.Image")
    def test_remove_all_metadata(self, mock_image):
        """Test remove command with all metadata."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = (1920, 1080)
        mock_img.format = "JPEG"
        mock_img.info = {"exif": b"fake_exif", "dpi": (72, 72)}
        mock_img.getdata.return_value = [0] * (1920 * 1080)

        mock_new_img = Mock()
        mock_image.open.return_value.__enter__.return_value = mock_img
        mock_image.new.return_value = mock_new_img

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(remove, [tmp_path, "--all"])
            if result.exit_code != 0:
                print(f"Command failed with output: {result.output}")
            assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_remove_no_options(self):
        """Test remove command with no metadata types specified."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(remove, [tmp_path])
            # Should warn about no metadata types specified
            assert "No metadata types specified" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("PIL.Image")
    def test_remove_error_handling(self, mock_image):
        """Test remove command error handling."""
        mock_image.open.side_effect = Exception("File error")

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(remove, [tmp_path, "--exif"])
            assert result.exit_code != 0
            assert "Error removing metadata" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestAdditionalCopyTests:
    """Additional tests for copy command."""

    @patch("photoflow.cli.metadata.read_metadata")
    @patch("photoflow.cli.metadata.write_metadata")
    def test_copy_successful(self, mock_write_metadata, mock_read_metadata):
        """Test successful copy operation."""
        # Mock metadata
        exif_data = ExifData(make="Canon", model="EOS R5")
        iptc_data = IPTCData(title="Test Image")
        mock_read_metadata.return_value = (exif_data, iptc_data)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as src:
            src_path = src.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as dst:
            dst_path = dst.name

        try:
            result = runner.invoke(copy, [src_path, dst_path])
            assert result.exit_code == 0
            # Should read metadata from both source and target files
            assert mock_read_metadata.call_count == 2
            expected_calls = [call(Path(src_path)), call(Path(dst_path))]
            mock_read_metadata.assert_has_calls(expected_calls, any_order=True)
            mock_write_metadata.assert_called_once()
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(dst_path).unlink(missing_ok=True)
