"""Simple tests to boost coverage on action modules."""

from photoflow.actions import advanced, basic, enhanced
from photoflow.actions.advanced import HueAction, LevelsAction, SepiaAction
from photoflow.actions.basic import ResizeAction
from photoflow.actions.enhanced import BlurAction


def test_action_modules_importable():
    """Test that action modules can be imported successfully."""
    assert hasattr(advanced, "HueAction")
    assert hasattr(advanced, "LevelsAction")
    assert hasattr(advanced, "SepiaAction")
    assert hasattr(advanced, "NoiseReductionAction")
    assert hasattr(basic, "ResizeAction")
    assert hasattr(basic, "CropAction")
    assert hasattr(enhanced, "BlurAction")
    assert hasattr(enhanced, "BrightnessAction")


def test_action_class_instantiation():
    """Test that action classes can be instantiated."""
    hue_action = HueAction()
    levels_action = LevelsAction()
    resize_action = ResizeAction()
    blur_action = BlurAction()
    assert hasattr(hue_action, "name")
    assert hasattr(levels_action, "name")
    assert hasattr(resize_action, "name")
    assert hasattr(blur_action, "name")
    assert hue_action.name == "hue"
    assert levels_action.name == "levels"
    assert resize_action.name == "resize"
    assert blur_action.name == "blur"


def test_action_field_setup():
    """Test that actions can set up their fields."""
    hue_action = HueAction()
    hue_action.setup_fields()
    assert "shift" in hue_action.fields
    sepia_action = SepiaAction()
    sepia_action.setup_fields()
    assert "intensity" in sepia_action.fields
    resize_action = ResizeAction()
    resize_action.setup_fields()
    assert "width" in resize_action.fields
    assert "height" in resize_action.fields
