"""Test domain models."""

from pathlib import Path

import pytest

from photoflow.core.models import ActionParameter, ActionSpec, ImageAsset, Pipeline


class TestImageAsset:
    """Test ImageAsset domain model."""

    def test_creation(self) -> None:
        """Test basic ImageAsset creation."""
        asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
            metadata={"camera": "Canon"},
        )

        assert asset.path == Path("test.jpg")
        assert asset.format == "JPEG"
        assert asset.size == (1920, 1080)
        assert asset.mode == "RGB"
        assert asset.metadata == {"camera": "Canon"}
        assert asset.data is None

    def test_properties(self) -> None:
        """Test ImageAsset computed properties."""
        asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        assert asset.width == 1920
        assert asset.height == 1080
        assert asset.aspect_ratio == pytest.approx(1.777, rel=1e-3)

    def test_with_updates(self) -> None:
        """Test immutable updates via with_updates."""
        original = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
            metadata={"camera": "Canon"},
        )

        # Update single field
        updated = original.with_updates(format="PNG")
        assert updated.format == "PNG"
        assert updated.path == original.path  # Other fields unchanged
        assert original.format == "JPEG"  # Original unchanged

        # Update metadata
        updated = original.with_updates(metadata={"lens": "50mm"})
        assert updated.metadata == {"camera": "Canon", "lens": "50mm"}
        assert original.metadata == {"camera": "Canon"}  # Original unchanged

    def test_immutability(self) -> None:
        """Test that ImageAsset is immutable."""
        asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            asset.format = "PNG"  # type: ignore

    def test_equality(self) -> None:
        """Test ImageAsset equality."""
        asset1 = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        asset2 = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        asset3 = ImageAsset(
            path=Path("test.jpg"),
            format="PNG",
            size=(1920, 1080),
            mode="RGB",
        )

        assert asset1 == asset2
        assert asset1 != asset3

    def test_enhanced_properties(self) -> None:
        """Test enhanced ImageAsset properties."""
        asset = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(4000, 3000),
            mode="RGB",
        )

        assert asset.megapixels == 12.0
        assert asset.width == 4000
        assert asset.height == 3000

    def test_template_vars_basic(self) -> None:
        """Test basic template variables."""
        asset = ImageAsset(
            path=Path("/photos/vacation_photo.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
            dpi=(300, 300),
        )

        vars_dict = asset.template_vars

        # Basic properties
        assert vars_dict["width"] == 1920
        assert vars_dict["height"] == 1080
        assert vars_dict["format"] == "JPEG"
        assert vars_dict["format_lower"] == "jpeg"
        assert vars_dict["mode"] == "RGB"

        # File path components
        assert vars_dict["filename"] == "vacation_photo"
        assert vars_dict["basename"] == "vacation_photo.jpg"
        assert vars_dict["extension"] == "jpg"
        assert vars_dict["parent"] == "photos"

        # Calculated values
        assert vars_dict["aspect_ratio"] == pytest.approx(1.78, rel=1e-2)
        assert vars_dict["megapixels"] == pytest.approx(2.1, rel=1e-1)
        assert vars_dict["dpi_horizontal"] == 300
        assert vars_dict["dpi_vertical"] == 300

    def test_template_vars_with_exif(self) -> None:
        """Test template variables with EXIF data."""
        exif_data = {
            "Make": "Canon",
            "Model": "EOS R5",
            "DateTime": "2025:01:01 12:00:00",
            "FocalLength": (85, 1),
            "FNumber": (28, 10),  # f/2.8
            "ExposureTime": (1, 250),  # 1/250s
            "ISOSpeedRatings": 800,
        }

        asset = ImageAsset(
            path=Path("photo.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
            exif=exif_data,
        )

        vars_dict = asset.template_vars

        assert vars_dict["camera_make"] == "Canon"
        assert vars_dict["camera_model"] == "EOS R5"
        assert vars_dict["date_taken"] == "2025:01:01 12:00:00"
        assert vars_dict["focal_length"] == "85mm"
        assert vars_dict["aperture"] == "f/2.8"
        assert vars_dict["exposure_time"] == "1/250s"
        assert vars_dict["iso"] == 800

    def test_template_vars_with_iptc(self) -> None:
        """Test template variables with IPTC data."""
        iptc_data = {
            "ObjectName": "Sunset Over Mountains",
            "Caption-Abstract": "Beautiful sunset photograph",
            "Keywords": ["sunset", "mountains", "landscape"],
            "CopyrightNotice": "© 2025 Photographer",
            "By-line": "John Doe",
        }

        asset = ImageAsset(
            path=Path("photo.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
            iptc=iptc_data,
        )

        vars_dict = asset.template_vars

        assert vars_dict["title"] == "Sunset Over Mountains"
        assert vars_dict["description"] == "Beautiful sunset photograph"
        assert vars_dict["keywords"] == "sunset, mountains, landscape"
        assert vars_dict["copyright"] == "© 2025 Photographer"
        assert vars_dict["creator"] == "John Doe"

    def test_exif_formatting_helpers(self) -> None:
        """Test EXIF data formatting helper methods."""
        asset = ImageAsset(
            path=Path("photo.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        # Test focal length formatting
        asset_with_focal = asset.with_exif_update({"FocalLength": (50, 1)})
        assert asset_with_focal._get_exif_focal_length() == "50mm"

        # Test aperture formatting
        asset_with_aperture = asset.with_exif_update({"FNumber": (18, 10)})
        assert asset_with_aperture._get_exif_aperture() == "f/1.8"

        # Test exposure time formatting
        asset_with_exposure = asset.with_exif_update({"ExposureTime": (1, 125)})
        assert asset_with_exposure._get_exif_exposure_time() == "1/125s"

    def test_metadata_updates(self) -> None:
        """Test metadata update methods."""
        asset = ImageAsset(
            path=Path("photo.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        # Test EXIF update
        exif_data = {"Make": "Canon", "Model": "EOS R5"}
        updated_asset = asset.with_exif_update(exif_data)
        assert updated_asset.exif == exif_data
        assert asset.exif == {}  # Original unchanged

        # Test IPTC update
        iptc_data = {"ObjectName": "Test Photo"}
        updated_asset = asset.with_iptc_update(iptc_data)
        assert updated_asset.iptc == iptc_data
        assert asset.iptc == {}  # Original unchanged

        # Test processing metadata update
        processing_data = {"pipeline": "test_pipeline", "version": "1.0"}
        updated_asset = asset.with_processing_update(processing_data)
        assert updated_asset.processing_metadata == processing_data
        assert asset.processing_metadata == {}  # Original unchanged

    def test_original_path_tracking(self) -> None:
        """Test original path tracking for processing chains."""
        original_asset = ImageAsset(
            path=Path("/original/photo.jpg"),
            format="RAW",
            size=(4000, 3000),
            mode="RGB",
        )

        processed_asset = original_asset.with_updates(
            path=Path("/processed/photo_processed.jpg"),
            format="JPEG",
            original_path=original_asset.path,
        )

        assert processed_asset.original_path == Path("/original/photo.jpg")
        assert processed_asset.path == Path("/processed/photo_processed.jpg")

        # Template vars should reflect original filename
        vars_dict = processed_asset.template_vars
        assert vars_dict["original_filename"] == "photo"
        assert vars_dict["filename"] == "photo_processed"


class TestActionParameter:
    """Test ActionParameter value object."""

    def test_creation(self) -> None:
        """Test basic ActionParameter creation."""
        param = ActionParameter(
            name="width",
            value=800,
            param_type="int",
            description="Image width in pixels",
            default=None,
            required=True,
        )

        assert param.name == "width"
        assert param.value == 800
        assert param.param_type == "int"
        assert param.description == "Image width in pixels"
        assert param.required is True

    def test_type_validation(self) -> None:
        """Test parameter type validation."""
        # Valid types should work
        valid_types = ["int", "float", "str", "bool", "list", "dict", "path", "enum"]
        for param_type in valid_types:
            param = ActionParameter(
                name="test",
                value="test",
                param_type=param_type,
                default=None,
                description="",
                required=True,
            )
            assert param.param_type == param_type

        # Invalid type should raise error
        with pytest.raises(ValueError, match="Invalid parameter type"):
            ActionParameter(
                name="test",
                value="test",
                param_type="invalid",
                default=None,
                description="",
                required=True,
            )

    def test_value_validation(self) -> None:
        """Test parameter value validation."""
        # Int validation
        param = ActionParameter(
            name="width",
            value="800",
            param_type="int",
            default=None,
            description="",
            required=True,
        )
        param.validate_value()
        assert param.value == 800

        # Float validation
        param = ActionParameter(
            name="ratio",
            value="1.5",
            param_type="float",
            default=None,
            description="",
            required=True,
        )
        param.validate_value()
        assert param.value == 1.5

        # Bool validation
        param = ActionParameter(
            name="enabled",
            value="true",
            param_type="bool",
            default=None,
            description="",
            required=True,
        )
        param.validate_value()
        assert param.value is True

        # Path validation
        param = ActionParameter(
            name="output",
            value="/tmp/test",
            param_type="path",
            default=None,
            description="",
            required=True,
        )
        param.validate_value()
        assert param.value == Path("/tmp/test")

    def test_constraint_validation(self) -> None:
        """Test parameter constraint validation."""
        # Min/max constraints
        param = ActionParameter(
            name="quality",
            value=95,
            param_type="int",
            constraints={"min": 1, "max": 100},
            default=None,
            description="",
            required=True,
        )
        param.validate_value()  # Should pass

        param.value = 0
        with pytest.raises(ValueError, match="must be >= 1"):
            param.validate_value()

        param.value = 101
        with pytest.raises(ValueError, match="must be <= 100"):
            param.validate_value()

        # Choices constraint
        param = ActionParameter(
            name="format",
            value="JPEG",
            param_type="str",
            constraints={"choices": ["JPEG", "PNG", "WebP"]},
            default=None,
            description="",
            required=True,
        )
        param.validate_value()  # Should pass

        param.value = "GIF"
        with pytest.raises(ValueError, match="must be one of"):
            param.validate_value()

    def test_required_validation(self) -> None:
        """Test required parameter validation."""
        param = ActionParameter(
            name="required_param",
            value=None,
            param_type="str",
            required=True,
            default=None,
            description="",
        )

        with pytest.raises(ValueError, match="Required parameter"):
            param.validate_value()

        # Optional parameter should allow None
        param = ActionParameter(
            name="optional_param",
            value=None,
            param_type="str",
            required=False,
            default=None,
            description="",
        )
        param.validate_value()  # Should pass


class TestActionSpec:
    """Test ActionSpec domain model."""

    def test_creation(self) -> None:
        """Test basic ActionSpec creation."""
        action = ActionSpec(
            name="resize",
            parameters={"width": 800, "height": 600},
            description="Resize image",
        )

        assert action.name == "resize"
        assert action.parameters == {"width": 800, "height": 600}
        assert action.enabled is True
        assert action.description == "Resize image"

    def test_name_validation(self) -> None:
        """Test action name validation."""
        # Valid names
        ActionSpec(name="resize")
        ActionSpec(name="resize_and_crop")
        ActionSpec(name="convert_format")

        # Invalid names
        with pytest.raises(ValueError, match="cannot be empty"):
            ActionSpec(name="")

        with pytest.raises(ValueError, match="must be alphanumeric"):
            ActionSpec(name="resize-image")

        with pytest.raises(ValueError, match="must be alphanumeric"):
            ActionSpec(name="resize image")

    def test_with_parameter(self) -> None:
        """Test adding parameters immutably."""
        original = ActionSpec(name="resize", parameters={"width": 800})

        updated = original.with_parameter("height", 600)
        assert updated.parameters == {"width": 800, "height": 600}
        assert original.parameters == {"width": 800}  # Original unchanged

    def test_without_parameter(self) -> None:
        """Test removing parameters immutably."""
        original = ActionSpec(
            name="resize",
            parameters={"width": 800, "height": 600},
        )

        updated = original.without_parameter("height")
        assert updated.parameters == {"width": 800}
        assert original.parameters == {
            "width": 800,
            "height": 600,
        }  # Original unchanged

        # Removing non-existent parameter should not error
        updated = original.without_parameter("nonexistent")
        assert updated.parameters == original.parameters

    def test_immutability(self) -> None:
        """Test that ActionSpec is immutable."""
        action = ActionSpec(name="resize")

        with pytest.raises(AttributeError):
            action.name = "crop"  # type: ignore


class TestPipeline:
    """Test Pipeline domain model."""

    def test_creation(self) -> None:
        """Test basic Pipeline creation."""
        actions = [
            ActionSpec(name="resize", parameters={"width": 800}),
            ActionSpec(name="save", parameters={"format": "JPEG"}),
        ]

        pipeline = Pipeline(
            name="web_resize",
            actions=tuple(actions),
            metadata={"author": "test", "version": "1.0"},
        )

        assert pipeline.name == "web_resize"
        assert len(pipeline.actions) == 2
        assert pipeline.actions[0].name == "resize"
        assert pipeline.actions[1].name == "save"
        assert pipeline.metadata == {"author": "test", "version": "1.0"}

    def test_name_validation(self) -> None:
        """Test pipeline name validation."""
        Pipeline(name="test_pipeline")  # Should work

        with pytest.raises(ValueError, match="cannot be empty"):
            Pipeline(name="")

    def test_actions_immutability(self) -> None:
        """Test that actions are stored as immutable tuple."""
        actions = [ActionSpec(name="resize")]
        pipeline = Pipeline(name="test", actions=tuple(actions))

        assert isinstance(pipeline.actions, tuple)

        # Original list should not affect pipeline
        actions.append(ActionSpec(name="save"))
        assert len(pipeline.actions) == 1

    def test_with_action(self) -> None:
        """Test adding actions immutably."""
        original = Pipeline(
            name="test",
            actions=(ActionSpec(name="resize"),),
        )

        # Add at end
        updated = original.with_action(ActionSpec(name="save"))
        assert len(updated.actions) == 2
        assert updated.actions[1].name == "save"
        assert len(original.actions) == 1  # Original unchanged

        # Add at specific index
        updated = original.with_action(ActionSpec(name="crop"), index=0)
        assert len(updated.actions) == 2
        assert updated.actions[0].name == "crop"
        assert updated.actions[1].name == "resize"

    def test_without_action(self) -> None:
        """Test removing actions immutably."""
        original = Pipeline(
            name="test",
            actions=(
                ActionSpec(name="resize"),
                ActionSpec(name="crop"),
                ActionSpec(name="save"),
            ),
        )

        updated = original.without_action(1)  # Remove crop
        assert len(updated.actions) == 2
        assert updated.actions[0].name == "resize"
        assert updated.actions[1].name == "save"
        assert len(original.actions) == 3  # Original unchanged

        # Invalid index should raise error
        with pytest.raises(IndexError):
            original.without_action(10)

    def test_with_updated_action(self) -> None:
        """Test updating actions immutably."""
        original = Pipeline(
            name="test",
            actions=(
                ActionSpec(name="resize", parameters={"width": 800}),
                ActionSpec(name="save"),
            ),
        )

        new_action = ActionSpec(name="resize", parameters={"width": 1200})
        updated = original.with_updated_action(0, new_action)

        assert updated.actions[0].parameters["width"] == 1200
        assert original.actions[0].parameters["width"] == 800  # Original unchanged

        # Invalid index should raise error
        with pytest.raises(IndexError):
            original.with_updated_action(10, new_action)

    def test_properties(self) -> None:
        """Test Pipeline computed properties."""
        actions = [
            ActionSpec(name="resize", enabled=True),
            ActionSpec(name="crop", enabled=False),
            ActionSpec(name="save", enabled=True),
        ]

        pipeline = Pipeline(name="test", actions=tuple(actions))

        assert pipeline.action_count == 3
        assert len(pipeline.enabled_actions) == 2
        assert pipeline.enabled_actions[0].name == "resize"
        assert pipeline.enabled_actions[1].name == "save"

    def test_serialization(self) -> None:
        """Test Pipeline serialization to/from dict."""
        actions = [
            ActionSpec(
                name="resize",
                parameters={"width": 800, "height": 600},
                description="Resize image",
            ),
            ActionSpec(
                name="save",
                parameters={"format": "JPEG", "quality": 95},
                enabled=False,
            ),
        ]

        original = Pipeline(
            name="test_pipeline",
            actions=tuple(actions),
            metadata={"version": "1.0", "author": "test"},
        )

        # Convert to dict
        data = original.model_dump()
        assert data["name"] == "test_pipeline"
        assert data["metadata"] == {"version": "1.0", "author": "test"}
        assert len(data["actions"]) == 2
        assert data["actions"][0]["name"] == "resize"
        assert data["actions"][0]["parameters"] == {"width": 800, "height": 600}
        assert data["actions"][1]["enabled"] is False

        # Convert back from dict
        restored = Pipeline.model_validate(data)
        assert restored.name == original.name
        assert restored.metadata == original.metadata
        assert len(restored.actions) == len(original.actions)
        assert restored.actions[0].name == original.actions[0].name
        assert restored.actions[0].parameters == original.actions[0].parameters
        assert restored.actions[1].enabled == original.actions[1].enabled

    def test_round_trip_serialization(self) -> None:
        """Test that serialization is lossless."""
        original = Pipeline(
            name="complex_pipeline",
            actions=(
                ActionSpec(
                    name="resize",
                    parameters={
                        "width": 1920,
                        "height": 1080,
                        "maintain_aspect": True,
                    },
                    description="High-res resize",
                ),
                ActionSpec(
                    name="watermark",
                    parameters={
                        "text": "© 2025 Company",
                        "position": "bottom_right",
                        "opacity": 0.7,
                    },
                    enabled=False,
                ),
                ActionSpec(
                    name="save",
                    parameters={"format": "WebP", "quality": 85},
                ),
            ),
            metadata={
                "version": "2.1",
                "author": "PhotoFlow",
                "description": "Web optimization pipeline",
                "tags": ["web", "optimization"],
            },
        )

        # Round trip: Pipeline -> dict -> Pipeline
        data = original.model_dump()
        restored = Pipeline.model_validate(data)

        assert restored == original
