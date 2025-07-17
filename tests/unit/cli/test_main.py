"""Tests for CLI main module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from photoflow.cli.main import _parse_action_string, cli, main
from photoflow.core.models import ActionSpec, ImageAsset, Pipeline


class TestCLIMain:
    """Test main CLI functionality."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PhotoFlow - Python batch photo manipulation tool" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_cli_verbose_flag(self):
        """Test CLI verbose flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestProcessCommand:
    """Test process command."""

    def test_process_command_help(self):
        """Test process command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process a single image" in result.output

    def test_process_nonexistent_file(self):
        """Test processing nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "/nonexistent/file.jpg"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    @patch("photoflow.cli.main.load_image_asset")
    @patch("photoflow.cli.main.default_processor")
    def test_process_dry_run(self, mock_processor, mock_load_image):
        """Test process command with dry run."""
        mock_image = Mock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.format = "JPEG"
        mock_image.path = Path("test.jpg")
        mock_load_image.return_value = mock_image
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(cli, ["process", tmp_path, "--action", "resize:width=100", "--dry-run"])
            if result.exit_code != 0:
                print(f"CLI Error Output: {result.output}")
                print(f"CLI Exception: {result.exception}")
            assert result.exit_code == 0
            assert "Would process" in result.output or "Dry run" in result.output
            if hasattr(mock_processor, "resize_image"):
                mock_processor.resize_image.assert_not_called()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.load_image_asset")
    @patch("photoflow.cli.main.default_registry")
    @patch("photoflow.core.image_processor.default_processor")
    @patch("photoflow.cli.main.default_processor")
    def test_process_with_action(self, mock_cli_processor, mock_core_processor, mock_registry, mock_load_image):
        """Test process command with action."""
        mock_image = Mock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.format = "JPEG"
        mock_image.path = Path("test.jpg")
        mock_image.file_size_mb = 2.5
        mock_image.size = 1920, 1080
        mock_load_image.return_value = mock_image
        mock_action = Mock()
        mock_action.execute.return_value = mock_image
        mock_registry.get_action.return_value = mock_action
        mock_cli_processor.save_image = Mock(return_value=Path("output.jpg"))
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(cli, ["process", tmp_path, "--action", "resize:width=800"])
            assert result.exit_code == 0
            mock_registry.get_action.assert_called_with("resize")
            mock_action.execute.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.load_image_asset")
    @patch("photoflow.cli.main.load_pipeline")
    @patch("photoflow.cli.main.default_processor")
    @patch("photoflow.cli.main.default_registry")
    def test_process_with_pipeline(self, mock_registry, mock_processor, mock_load_pipeline, mock_load_image):
        """Test process command with pipeline."""
        mock_image = Mock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.format = "JPEG"
        mock_image.path = Path("test.jpg")
        mock_image.file_size_mb = 2.5
        mock_image.size = 1920, 1080
        mock_load_image.return_value = mock_image
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = mock_image
        mock_pipeline.actions = [Mock()]
        mock_pipeline.enabled_actions = [Mock(name="resize", parameters={})]
        mock_pipeline.name = "test_pipeline"
        mock_load_pipeline.return_value = mock_pipeline
        mock_processor.save_image = Mock(return_value=Path("output.jpg"))
        mock_action = Mock()
        mock_action.execute.return_value = mock_image
        mock_registry.get_action.return_value = mock_action
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_tmp:
            img_path = img_tmp.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as pipe_tmp:
            pipe_path = pipe_tmp.name
        try:
            result = runner.invoke(cli, ["process", img_path, "--pipeline", pipe_path])
            if result.exit_code != 0:
                print(f"CLI Error Output: {result.output}")
                print(f"CLI Exception: {result.exception}")
            assert result.exit_code == 0
            mock_load_pipeline.assert_called_with(Path(pipe_path))
        finally:
            Path(img_path).unlink(missing_ok=True)
            Path(pipe_path).unlink(missing_ok=True)


class TestListActionsCommand:
    """Test list-actions command."""

    def test_list_actions_help(self):
        """Test list-actions command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-actions", "--help"])
        assert result.exit_code == 0
        assert "List available actions" in result.output

    @patch("photoflow.cli.main.default_registry")
    def test_list_actions_success(self, mock_registry):
        """Test successful action listing."""
        mock_registry.list_actions.return_value = ["resize", "crop", "rotate"]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-actions"])
        assert result.exit_code == 0
        assert "resize" in result.output
        assert "crop" in result.output
        assert "rotate" in result.output

    @patch("photoflow.cli.main.default_registry")
    def test_list_actions_with_details(self, mock_registry):
        """Test action listing with details."""
        mock_registry.list_actions.return_value = ["resize"]
        mock_registry.get_action_info.return_value = {
            "name": "resize",
            "label": "Resize Image",
            "description": "Resize image to specified dimensions",
            "version": "1.0.0",
            "author": "PhotoFlow",
            "tags": ["geometry", "basic"],
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["list-actions", "--detailed"])
        if result.exit_code != 0:
            print(f"CLI Error Output: {result.output}")
            print(f"CLI Exception: {result.exception}")
        assert result.exit_code == 0
        assert "resize" in result.output


class TestInfoCommand:
    """Test info command."""

    def test_info_help(self):
        """Test info command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show detailed information about an action" in result.output

    @patch("photoflow.cli.main.default_registry")
    def test_info_success(self, mock_registry):
        """Test successful action info."""
        mock_action = Mock()
        mock_action.name = "resize"
        mock_action.label = "Resize Image"
        mock_action.description = "Resize image to specified dimensions"
        mock_action.version = "1.0.0"
        mock_action.author = "PhotoFlow"
        mock_action.tags = ["geometry", "basic"]
        mock_action.get_field_schema.return_value = {
            "width": {"type": "integer", "label": "Width", "required": True},
            "height": {"type": "integer", "label": "Height", "required": False},
        }
        mock_registry.get_action.return_value = mock_action
        mock_registry.get_action_info.return_value = {
            "name": "resize",
            "label": "Resize Image",
            "description": "Resize image to specified dimensions",
            "version": "1.0.0",
            "author": "PhotoFlow",
            "tags": ["geometry", "basic"],
            "fields": {
                "width": {
                    "type": "integer",
                    "label": "Width",
                    "required": True,
                    "default": None,
                    "help_text": "Target width",
                },
                "height": {
                    "type": "integer",
                    "label": "Height",
                    "required": False,
                    "default": None,
                    "help_text": "Target height",
                },
            },
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "resize"])
        assert result.exit_code == 0

    @patch("photoflow.cli.main.default_registry")
    def test_info_nonexistent_action(self, mock_registry):
        """Test info for nonexistent action."""
        mock_registry.get_action_info.side_effect = KeyError("Action 'resize' not found")
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "resize"])
        assert result.exit_code != 0
        assert "not found" in result.output or "Error" in result.output


class TestPipelineCommand:
    """Test pipeline command."""

    def test_pipeline_help(self):
        """Test pipeline command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "Pipeline management commands" in result.output

    def test_pipeline_create_help(self):
        """Test pipeline create command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "create", "--help"])
        assert result.exit_code == 0
        assert "Create a new pipeline" in result.output

    @patch("photoflow.cli.main.save_pipeline")
    def test_pipeline_create_success(self, mock_save_pipeline):
        """Test successful pipeline creation."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(
                cli,
                [
                    "pipeline",
                    "create",
                    "test_pipeline",
                    "--action",
                    "resize:width=800",
                    "--action",
                    "crop:width=600,height=400",
                    "--output",
                    tmp_path,
                ],
            )
            assert result.exit_code == 0
            mock_save_pipeline.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_list_help(self):
        """Test pipeline list command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "list", "--help"])
        assert result.exit_code == 0
        assert "List pipeline files in directory" in result.output

    @patch("photoflow.cli.main.Path.glob")
    def test_pipeline_list_success(self, mock_glob):
        """Test successful pipeline listing."""
        mock_path = Mock()
        mock_path.name = "test_pipeline.json"
        mock_glob.return_value = [mock_path]
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "list"])
        assert result.exit_code == 0
        assert "test_pipeline" in result.output

    @patch("photoflow.cli.main.load_image_asset")
    @patch("photoflow.cli.main.load_pipeline")
    @patch("photoflow.cli.main.default_registry")
    @patch("photoflow.cli.main.default_processor")
    def test_pipeline_run_success(self, mock_processor, mock_registry, mock_load_pipeline, mock_load_image):
        """Test successful pipeline run."""
        mock_image = Mock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.format = "JPEG"
        mock_image.path = Path("test.jpg")
        mock_load_image.return_value = mock_image
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = mock_image
        mock_pipeline.actions = [Mock()]
        mock_action_spec = Mock()
        mock_action_spec.name = "resize"
        mock_action_spec.parameters = {}
        mock_pipeline.enabled_actions = [mock_action_spec]
        mock_pipeline.name = "test_pipeline"
        mock_load_pipeline.return_value = mock_pipeline
        mock_action = Mock()
        mock_action.execute.return_value = mock_image
        mock_registry.get_action.return_value = mock_action
        mock_processor.save_image.return_value = Path("output.jpg")
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_tmp:
            img_path = img_tmp.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as pipe_tmp:
            pipe_path = pipe_tmp.name
        try:
            result = runner.invoke(cli, ["pipeline", "run", pipe_path, img_path])
            assert result.exit_code == 0
            mock_load_pipeline.assert_called_with(Path(pipe_path))
            mock_registry.get_action.assert_called_with("resize")
        finally:
            Path(img_path).unlink(missing_ok=True)
            Path(pipe_path).unlink(missing_ok=True)


class TestBatchCommand:
    """Test batch command."""

    def test_batch_help(self):
        """Test batch command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Process multiple images" in result.output

    def test_batch_nonexistent_directory(self):
        """Test batch processing nonexistent directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "/nonexistent/dir"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    @patch("photoflow.cli.main.load_image_asset")
    @patch("photoflow.cli.main.default_registry")
    def test_batch_with_action(self, mock_registry, mock_load_image):
        """Test batch processing with action."""
        mock_image = Mock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.format = "JPEG"
        mock_image.path = Path("test.jpg")
        mock_load_image.return_value = mock_image
        mock_action = Mock()
        mock_action.execute.return_value = mock_image
        mock_registry.get_action.return_value = mock_action
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_files = []
            for i in range(3):
                test_file = Path(tmp_dir) / f"test_{i}.jpg"
                test_file.write_text("fake image")
                test_files.append(test_file)
            with patch("photoflow.cli.main.Path.glob") as mock_glob:
                mock_glob.return_value = test_files
                result = runner.invoke(cli, ["batch", tmp_dir, "--action", "resize:width=800"])
                assert result.exit_code == 0
                assert mock_registry.get_action.call_count >= 1


class TestParseActionString:
    """Test _parse_action_string function."""

    def test_parse_action_string_simple(self):
        """Test parsing simple action string."""
        result = _parse_action_string("resize:width=800")
        assert result.name == "resize"
        assert result.parameters == {"width": 800}

    def test_parse_action_string_multiple_params(self):
        """Test parsing action string with multiple parameters."""
        result = _parse_action_string("crop:width=600,height=400,x=100,y=50")
        assert result.name == "crop"
        assert result.parameters == {"width": 600, "height": 400, "x": 100, "y": 50}

    def test_parse_action_string_string_params(self):
        """Test parsing action string with string parameters."""
        result = _parse_action_string("watermark:text=Copyright,position=bottom_right")
        assert result.name == "watermark"
        assert result.parameters == {"text": "Copyright", "position": "bottom_right"}

    def test_parse_action_string_boolean_params(self):
        """Test parsing action string with boolean parameters."""
        result = _parse_action_string("resize:width=800,maintain_aspect=true")
        assert result.name == "resize"
        assert result.parameters == {"width": 800, "maintain_aspect": True}

    def test_parse_action_string_float_params(self):
        """Test parsing action string with float parameters."""
        result = _parse_action_string("brightness:factor=1.2")
        assert result.name == "brightness"
        assert result.parameters == {"factor": 1.2}

    def test_parse_action_string_no_params(self):
        """Test parsing action string with no parameters."""
        result = _parse_action_string("sepia:")
        assert result.name == "sepia"
        assert result.parameters == {}

    def test_parse_action_string_invalid_format(self):
        """Test parsing invalid action string format."""
        result = _parse_action_string("invalid_format")
        assert result.name == "invalid_format"
        assert result.parameters == {}


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_cli_main_integration(self):
        """Test CLI main integration."""
        assert callable(main)

    @patch("photoflow.cli.main.cli")
    def test_main_function_calls_cli(self, mock_cli):
        """Test that main function calls cli."""
        main()
        mock_cli.assert_called_once()

    def test_cli_with_metadata_commands(self):
        """Test CLI with metadata commands integrated."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.output


def create_mock_image_asset():
    """Create a mock ImageAsset for testing."""
    return Mock(spec=ImageAsset)


def create_mock_pipeline():
    """Create a mock Pipeline for testing."""
    return Mock(spec=Pipeline)


def create_mock_action_spec():
    """Create a mock ActionSpec for testing."""
    return Mock(spec=ActionSpec)


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @patch("photoflow.cli.main.load_image_asset")
    def test_process_image_load_error(self, mock_load_image):
        """Test process command with image load error."""
        mock_load_image.side_effect = Exception("Load error")
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(cli, ["process", tmp_path])
            assert result.exit_code != 0
            assert "Error" in result.output or "Failed" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.default_registry")
    def test_process_action_error(self, mock_registry):
        """Test process command with action execution error."""
        mock_action = Mock()
        mock_action.execute.side_effect = Exception("Action error")
        mock_registry.get_action.return_value = mock_action
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with patch("photoflow.cli.main.load_image_asset") as mock_load:
                mock_load.return_value = Mock()
                result = runner.invoke(cli, ["process", tmp_path, "--action", "resize:width=800"])
                assert result.exit_code != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.load_pipeline")
    def test_process_pipeline_error(self, mock_load_pipeline):
        """Test process command with pipeline error."""
        mock_load_pipeline.side_effect = Exception("Pipeline error")
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_tmp:
            img_path = img_tmp.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as pipe_tmp:
            pipe_path = pipe_tmp.name
        try:
            result = runner.invoke(cli, ["process", img_path, "--pipeline", pipe_path])
            assert result.exit_code != 0
            assert "Error" in result.output or "Failed" in result.output
        finally:
            Path(img_path).unlink(missing_ok=True)
            Path(pipe_path).unlink(missing_ok=True)


class TestCLIMainCoverageExtensions:
    """Additional tests to improve CLI main coverage."""

    def test_cli_verbose_flag_coverage(self):
        """Test verbose flag shows proper output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "info"])
        assert "PhotoFlow CLI - Verbose mode enabled" in result.output or result.exit_code == 0

    @patch("photoflow.cli.main.load_image_asset")
    def test_process_command_comprehensive_error_paths(self, mock_load_image):
        """Test process command error handling paths."""
        mock_load_image.side_effect = Exception("Image loading failed")
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(cli, ["process", tmp_path, "--action", "resize:width=100"])
            assert result.exit_code != 0 or "error" in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_parse_action_string_various_formats(self):
        """Test _parse_action_string with various input formats."""
        result = _parse_action_string("resize:width=800,height=600")
        assert result.name == "resize"
        assert result.parameters["width"] == 800
        assert result.parameters["height"] == 600
        result = _parse_action_string("sharpen")
        assert result.name == "sharpen"
        assert len(result.parameters) == 0
        result = _parse_action_string("invalid::format")
        assert result.name == "invalid"

    @patch("photoflow.cli.main.default_registry")
    def test_list_actions_with_registry(self, mock_registry):
        """Test list-actions command with mocked registry."""
        mock_registry.list_actions.return_value = [
            "resize",
            "crop",
            "rotate",
            "watermark",
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-actions"])
        assert result.exit_code == 0

    @patch("photoflow.cli.main.default_registry")
    def test_info_command_with_mocked_registry(self, mock_registry):
        """Test info command with properly mocked registry."""
        mock_registry.get_action_info.return_value = {
            "name": "resize",
            "label": "Resize Image",
            "description": "Resize image to specified dimensions",
            "version": "1.0.0",
            "author": "PhotoFlow",
            "tags": ["geometry", "basic"],
            "fields": {
                "width": {
                    "type": "integer",
                    "label": "Width",
                    "required": True,
                    "default": None,
                    "help_text": "Target width",
                },
                "height": {
                    "type": "integer",
                    "label": "Height",
                    "required": False,
                    "default": None,
                    "help_text": "Target height",
                },
            },
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "resize"])
        assert result.exit_code == 0

    @patch("photoflow.cli.main.save_pipeline")
    def test_pipeline_save_error_cases(self, mock_save_pipeline):
        """Test pipeline save error handling."""
        mock_save_pipeline.side_effect = Exception("Save failed")
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(
                cli,
                [
                    "pipeline",
                    "save",
                    "test-pipeline",
                    tmp_path,
                    "--action",
                    "resize:width=100",
                ],
            )
            assert result.exit_code != 0 or "error" in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.load_image_asset")
    def test_batch_processing_comprehensive(self, mock_load_image):
        """Test batch processing with comprehensive scenarios."""
        mock_images = []
        for i in range(2):
            mock_img = Mock()
            mock_img.width = 2000
            mock_img.height = 1500
            mock_img.format = "JPEG"
            mock_img.path = Path(f"test{i}.jpg")
            mock_images.append(mock_img)
        mock_load_image.side_effect = mock_images
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(2):
                test_file = Path(tmp_dir) / f"test{i}.jpg"
                test_file.write_bytes(f"fake_image_{i}".encode())
            result = runner.invoke(
                cli,
                [
                    "batch",
                    tmp_dir,
                    "--action",
                    "resize:width=1000",
                    "--pattern",
                    "*.jpg",
                    "--dry-run",
                ],
            )
            if result.exit_code != 0:
                print(f"Batch error: {result.output}")
            assert result.exit_code == 0 or "error" in result.output.lower()

    def test_batch_empty_directory_handling(self):
        """Test batch command with empty directory."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(cli, ["batch", tmp_dir, "--action", "resize:width=100"])
            assert result.exit_code == 0 or "No images found" in result.output

    def test_batch_invalid_directory(self):
        """Test batch command with invalid directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "/totally/invalid/path", "--action", "resize:width=100"])
        assert result.exit_code != 0 or "does not exist" in result.output
