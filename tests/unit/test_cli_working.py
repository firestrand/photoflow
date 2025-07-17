"""Working CLI tests to improve coverage."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from photoflow.cli.main import _parse_action_string, cli, main
from photoflow.core.models import ImageAsset


class TestCLIWorking:
    """Working CLI tests for coverage."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PhotoFlow" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_main_function(self):
        """Test main function."""
        with patch("photoflow.cli.main.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()

    def test_parse_action_string_simple(self):
        """Test parsing simple action string."""
        result = _parse_action_string("resize:width=800")
        assert result.name == "resize"
        assert result.parameters == {"width": 800}

    def test_parse_action_string_multiple_params(self):
        """Test parsing action string with multiple parameters."""
        result = _parse_action_string("crop:x=100,y=50,width=800,height=600")
        assert result.name == "crop"
        assert result.parameters == {"x": 100, "y": 50, "width": 800, "height": 600}

    def test_parse_action_string_no_params(self):
        """Test parsing action string with no parameters."""
        result = _parse_action_string("sepia")
        assert result.name == "sepia"
        assert result.parameters == {}

    def test_parse_action_string_boolean_params(self):
        """Test parsing action string with boolean parameters."""
        result = _parse_action_string("resize:width=800,maintain_aspect=true")
        assert result.name == "resize"
        assert result.parameters == {"width": 800, "maintain_aspect": True}

    def test_parse_action_string_float_params(self):
        """Test parsing action string with float parameters."""
        result = _parse_action_string("brightness:factor=1.5")
        assert result.name == "brightness"
        assert result.parameters == {"factor": 1.5}

    @patch("photoflow.cli.main.default_registry")
    def test_list_actions_basic(self, mock_registry):
        """Test basic list actions command."""
        runner = CliRunner()

        # Mock the registry methods
        mock_registry.list_actions.return_value = ["resize", "crop"]
        mock_registry.get_action_info.side_effect = lambda name: {
            "name": name,
            "label": f"{name.title()} Action",
            "description": f"Description for {name}",
            "version": "1.0",
            "tags": ["basic"],
            "fields": {},
        }

        result = runner.invoke(cli, ["list-actions"])
        assert result.exit_code == 0
        assert "resize" in result.output

    @patch("photoflow.cli.main.default_registry")
    def test_info_action_basic(self, mock_registry):
        """Test info command for an action."""
        runner = CliRunner()

        # Mock registry to ensure resize action exists
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

        result = runner.invoke(cli, ["info", "resize"])
        # The resize action should exist and info should succeed
        assert result.exit_code == 0
        assert "resize" in result.output.lower()

    def test_info_nonexistent_action(self):
        """Test info command for nonexistent action."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("photoflow.cli.main.load_image_asset")
    def test_process_no_actions(self, mock_load_image):
        """Test process command with no actions."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(cli, ["process", tmp_path])
            assert result.exit_code != 0
            assert "No actions or pipeline specified" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.load_image_asset")
    def test_process_dry_run(self, mock_load_image):
        """Test process command with dry run."""
        runner = CliRunner()

        # Mock image loading
        mock_image = ImageAsset(path=Path("test.jpg"), format="JPEG", size=(100, 100), mode="RGB")
        mock_load_image.return_value = mock_image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(cli, ["process", tmp_path, "--action", "resize:width=800", "--dry-run"])
            assert result.exit_code == 0
            assert "Dry run" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.load_image_asset")
    def test_process_image_load_error(self, mock_load_image):
        """Test process command with image load error."""
        runner = CliRunner()

        # Mock image loading to raise error
        mock_load_image.side_effect = Exception("Failed to load image")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = runner.invoke(cli, ["process", tmp_path, "--action", "resize:width=800"])
            assert result.exit_code == 1
            assert "Error: Failed to load image" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("photoflow.cli.main.save_pipeline")
    def test_create_pipeline_basic(self, mock_save_pipeline):
        """Test basic pipeline creation."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "pipeline",
                "create",
                "test_pipeline",
                "--action",
                "resize:width=800",
                "--description",
                "Test pipeline",
            ],
        )

        assert result.exit_code == 0
        assert "Created pipeline" in result.output
        mock_save_pipeline.assert_called_once()

    def test_create_pipeline_no_actions(self):
        """Test pipeline creation with no actions."""
        runner = CliRunner()

        result = runner.invoke(cli, ["pipeline", "create", "test_pipeline"])

        assert result.exit_code != 0
        assert "No actions specified" in result.output

    def test_list_pipelines_no_pipelines(self):
        """Test pipeline listing with no pipelines."""
        runner = CliRunner()

        result = runner.invoke(cli, ["pipeline", "list"])
        # This will likely succeed with empty output or show no pipelines
        assert result.exit_code == 0

    def test_batch_nonexistent_directory(self):
        """Test batch command with nonexistent directory."""
        runner = CliRunner()

        result = runner.invoke(cli, ["batch", "/nonexistent/directory", "--action", "resize:width=800"])

        assert result.exit_code != 0
        assert "does not exist" in result.output
