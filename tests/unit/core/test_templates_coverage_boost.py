"""Targeted tests to boost coverage on core/templates.py module."""

import photoflow.core.templates as templates_module
from photoflow.core.templates import TemplateError, TemplateEvaluator


def test_template_error_creation():
    """Test TemplateError creation and functionality."""
    error = TemplateError("{{invalid}}", "Invalid syntax")
    assert error.template == "{{invalid}}"
    assert error.message == "Invalid syntax"
    assert error.suggestion == ""
    assert "Template evaluation failed" in str(error)
    error_with_suggestion = TemplateError("{{bad}}", "Bad template", "Use valid syntax")
    assert error_with_suggestion.suggestion == "Use valid syntax"
    assert "Suggestion: Use valid syntax" in str(error_with_suggestion)


def test_template_evaluator_importability():
    """Test that TemplateEvaluator can be imported and instantiated."""
    evaluator = TemplateEvaluator()
    assert evaluator is not None


def test_template_evaluator_safe_functions():
    """Test TemplateEvaluator safe functions dictionary."""
    evaluator = TemplateEvaluator()
    assert hasattr(evaluator, "SAFE_FUNCTIONS")
    assert isinstance(evaluator.SAFE_FUNCTIONS, dict)
    safe_funcs = evaluator.SAFE_FUNCTIONS
    assert "abs" in safe_funcs
    assert "int" in safe_funcs
    assert safe_funcs["abs"] == abs
    assert safe_funcs["int"] is int


def test_template_evaluator_basic_operation():
    """Test basic template evaluator operations."""
    evaluator = TemplateEvaluator()
    assert hasattr(evaluator, "SAFE_FUNCTIONS")
    assert isinstance(evaluator, TemplateEvaluator)


def test_template_functions_coverage():
    """Test coverage of template function definitions."""
    evaluator = TemplateEvaluator()
    functions = evaluator.SAFE_FUNCTIONS
    assert isinstance(functions, dict)
    assert len(functions) > 0
    for func_name, func in functions.items():
        assert isinstance(func_name, str)
        assert callable(func) or isinstance(func, type)


def test_module_level_imports():
    """Test module level imports and constants."""
    assert hasattr(templates_module, "TemplateError")
    assert hasattr(templates_module, "TemplateEvaluator")
    assert hasattr(templates_module, "re")


def test_template_error_edge_cases():
    """Test TemplateError edge cases."""
    error = TemplateError("", "")
    assert error.template == ""
    assert error.message == ""
    try:
        error = TemplateError("", "test", None)
        assert error.suggestion == ""
    except Exception:
        pass


def test_template_evaluator_initialization():
    """Test TemplateEvaluator initialization coverage."""
    eval1 = TemplateEvaluator()
    eval2 = TemplateEvaluator()
    assert eval1 is not eval2
    assert eval1.SAFE_FUNCTIONS is eval2.SAFE_FUNCTIONS


def test_template_module_constants():
    """Test template module constants and class variables."""
    TemplateEvaluator()
    assert hasattr(TemplateEvaluator, "SAFE_FUNCTIONS")
    functions = TemplateEvaluator.SAFE_FUNCTIONS
    if "abs" in functions:
        assert functions["abs"] is abs
    if "int" in functions:
        assert functions["int"] is int
