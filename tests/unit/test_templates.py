"""Tests for the template evaluation system."""

import pytest

from photoflow.core.templates import (
    TemplateError,
    TemplateEvaluator,
    evaluate_template,
    validate_template,
)


class TestTemplateError:
    """Test TemplateError formatting."""

    def test_basic_error(self) -> None:
        """Test basic error message formatting."""
        error = TemplateError(
            template="<invalid>",
            message="Variable not found",
        )

        assert "Template evaluation failed" in str(error)
        assert "Template: <invalid>" in str(error)
        assert "Variable not found" in str(error)

    def test_error_with_suggestion(self) -> None:
        """Test error message with suggestion."""
        error = TemplateError(
            template="<invalid>",
            message="Variable not found",
            suggestion="Use <filename> instead",
        )

        assert "Suggestion: Use <filename> instead" in str(error)


class TestTemplateEvaluator:
    """Test TemplateEvaluator functionality."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.evaluator = TemplateEvaluator()
        self.context = {
            "filename": "photo",
            "width": 1920,
            "height": 1080,
            "format": "JPEG",
            "index": 1,
        }

    def test_simple_variable_substitution(self) -> None:
        """Test basic variable substitution."""
        template = "<filename>_<width>x<height>"
        result = self.evaluator.evaluate(template, self.context)
        assert result == "photo_1920x1080"

    def test_no_variables(self) -> None:
        """Test template with no variables."""
        template = "static_text.jpg"
        result = self.evaluator.evaluate(template, self.context)
        assert result == "static_text.jpg"

    def test_missing_variable(self) -> None:
        """Test template with missing variable."""
        template = "<filename>_<missing>"
        with pytest.raises(TemplateError) as exc_info:
            self.evaluator.evaluate(template, self.context)

        error = exc_info.value
        assert "Variable 'missing' not found" in error.message
        assert "Available variables:" in error.suggestion

    def test_none_value_handling(self) -> None:
        """Test handling of None values in context."""
        context = {"filename": "photo", "description": None}
        template = "<filename>_<description>"
        result = self.evaluator.evaluate(template, context)
        assert result == "photo_"

    def test_format_pattern_conversion(self) -> None:
        """Test ### format pattern conversion."""
        template = "file_###.jpg"
        context: dict[str, str] = {}
        result = self.evaluator.evaluate(template, context)
        assert result == "file_%03d.jpg"

        template = "file_####.jpg"
        result = self.evaluator.evaluate(template, context)
        assert result == "file_%04d.jpg"

    def test_safe_function_evaluation(self) -> None:
        """Test safe function evaluation."""
        template = "upper(<filename>)"
        result = self.evaluator.evaluate(template, self.context)
        assert result == "PHOTO"

        template = "max(<width>, <height>)"
        result = self.evaluator.evaluate(template, self.context)
        assert result == "1920"

    def test_unsafe_function_rejection(self) -> None:
        """Test that unsafe functions are rejected."""
        template = "eval(<filename>)"
        with pytest.raises(TemplateError) as exc_info:
            self.evaluator.evaluate(template, self.context)

        error = exc_info.value
        assert "Function 'eval' is not allowed" in error.message

    def test_function_with_string_args(self) -> None:
        """Test function evaluation with string arguments."""
        template = 'upper("hello")'
        result = self.evaluator.evaluate(template, {})
        assert result == "HELLO"

    def test_function_with_numeric_args(self) -> None:
        """Test function evaluation with numeric arguments."""
        template = "max(100, 200)"
        result = self.evaluator.evaluate(template, {})
        assert result == "200"

    def test_function_error_handling(self) -> None:
        """Test function error handling."""
        template = "int(<filename>)"  # Can't convert "photo" to int
        with pytest.raises(TemplateError) as exc_info:
            self.evaluator.evaluate(template, self.context)

        error = exc_info.value
        assert "Function 'int' failed" in error.message

    def test_nested_variable_access(self) -> None:
        """Test nested variable access."""
        context = {
            "image": {"width": 1920, "height": 1080},
            "meta": {"camera": "Canon"},
        }
        template = "<image.width>x<image.height>_<meta.camera>"
        result = self.evaluator.evaluate(template, context)
        assert result == "1920x1080_Canon"

    def test_invalid_nested_access(self) -> None:
        """Test invalid nested variable access."""
        context = {"image": {"width": 1920}}
        template = "<image.invalid>"
        with pytest.raises(TemplateError) as exc_info:
            self.evaluator.evaluate(template, context)

        error = exc_info.value
        assert "Cannot resolve 'invalid'" in error.message

    def test_get_template_variables(self) -> None:
        """Test extraction of template variables."""
        template = "<filename>_<width>x<height>_<format>"
        variables = self.evaluator.get_template_variables(template)
        expected = {"filename", "width", "height", "format"}
        assert variables == expected

    def test_get_template_variables_nested(self) -> None:
        """Test extraction of nested template variables."""
        template = "<image.width>x<image.height>"
        variables = self.evaluator.get_template_variables(template)
        # Should return base variable names
        expected = {"image"}
        assert variables == expected

    def test_validate_template_success(self) -> None:
        """Test successful template validation."""
        template = "<filename>_<width>x<height>"
        errors = self.evaluator.validate_template(template, self.context)
        assert errors == []

    def test_validate_template_missing_vars(self) -> None:
        """Test template validation with missing variables."""
        template = "<filename>_<missing>"
        errors = self.evaluator.validate_template(template, self.context)
        assert len(errors) > 0
        assert "missing" in errors[0].lower()

    def test_complex_template(self) -> None:
        """Test complex template with multiple features."""
        template = "upper(<filename>)_<width>x<height>_###"
        result = self.evaluator.evaluate(template, self.context)
        assert result == "PHOTO_1920x1080_%03d"

    def test_empty_template(self) -> None:
        """Test empty template."""
        template = ""
        result = self.evaluator.evaluate(template, self.context)
        assert result == ""

    def test_whitespace_handling(self) -> None:
        """Test whitespace handling in variables."""
        template = "< filename >_< width >"
        result = self.evaluator.evaluate(template, self.context)
        assert result == "photo_1920"

    def test_special_characters_in_context(self) -> None:
        """Test special characters in context values."""
        context = {
            "filename": "photo with spaces",
            "special": "file@#$%",
        }
        template = "<filename>_<special>"
        result = self.evaluator.evaluate(template, context)
        assert result == "photo with spaces_file@#$%"

    def test_numeric_types_in_context(self) -> None:
        """Test various numeric types in context."""
        context = {
            "int_val": 42,
            "float_val": 3.14,
            "zero": 0,
        }
        template = "<int_val>_<float_val>_<zero>"
        result = self.evaluator.evaluate(template, context)
        assert result == "42_3.14_0"

    def test_function_argument_parsing(self) -> None:
        """Test function argument parsing edge cases."""
        # No arguments
        template = "str()"
        result = self.evaluator.evaluate(template, {})
        # This should work but may raise an error depending on implementation
        # The test ensures we handle empty args gracefully

        # Multiple arguments
        template = "max(10, 20, 30)"
        result = self.evaluator.evaluate(template, {})
        assert result == "30"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_evaluate_template_function(self) -> None:
        """Test evaluate_template convenience function."""
        context = {"name": "test"}
        result = evaluate_template("<name>_file", context)
        assert result == "test_file"

    def test_validate_template_function(self) -> None:
        """Test validate_template convenience function."""
        context = {"name": "test"}
        errors = validate_template("<name>_file", context)
        assert errors == []

        errors = validate_template("<missing>_file", context)
        assert len(errors) > 0


class TestTemplateSecurityFeatures:
    """Test template security features."""

    def test_no_code_injection(self) -> None:
        """Test that code injection is prevented."""
        dangerous_templates = [
            "__import__('os').system('rm -rf /')",
            "exec('print(1)')",
            "eval('1+1')",
        ]

        evaluator = TemplateEvaluator()
        context: dict[str, str] = {}

        for template in dangerous_templates:
            # These should either be ignored or raise safe errors
            try:
                result = evaluator.evaluate(template, context)
                # If it doesn't raise an error, it should not execute code
                assert "rm -rf" not in result
                assert "exec" not in result or result == template
            except TemplateError:
                # This is acceptable - template evaluation failed safely
                pass

    def test_safe_function_whitelist(self) -> None:
        """Test that only whitelisted functions are available."""
        evaluator = TemplateEvaluator()

        # Check that dangerous functions are not in whitelist
        dangerous_functions = ["exec", "eval", "compile", "open", "__import__"]
        for func in dangerous_functions:
            assert func not in evaluator.SAFE_FUNCTIONS

        # Check that safe functions are available
        safe_functions = ["str", "int", "len", "max", "min"]
        for func in safe_functions:
            assert func in evaluator.SAFE_FUNCTIONS


class TestTemplateErrorHandling:
    """Test template error handling."""

    def test_malformed_variable_syntax(self) -> None:
        """Test handling of malformed variable syntax."""
        evaluator = TemplateEvaluator()
        context = {"test": "value"}

        # Unclosed variable is treated as literal text
        result = evaluator.evaluate("<test", context)
        # If it's treated as literal, that's fine too
        assert result == "<test"

        # Test other malformed syntax
        result = evaluator.evaluate("test>", context)
        assert result == "test>"

    def test_deeply_nested_variables(self) -> None:
        """Test deeply nested variable access."""
        context = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        template = "<level1.level2.level3.value>"
        result = TemplateEvaluator().evaluate(template, context)
        assert result == "deep"

    def test_non_string_template_input(self) -> None:
        """Test handling of non-string template input."""
        evaluator = TemplateEvaluator()
        context = {"test": "value"}

        # Should convert to string
        result = evaluator.evaluate(123, context)  # type: ignore[arg-type]
        assert result == "123"

        result = evaluator.evaluate(None, context)  # type: ignore[arg-type]
        assert result == "None"

    def test_nested_variable_exception_fallthrough(self) -> None:
        """Test exception handling in nested variable resolution."""
        evaluator = TemplateEvaluator()

        # Create a context where nested access will fail but fall through to expression eval
        class BadObject:
            def __getattr__(self, name: str) -> None:
                raise ValueError("Bad attribute access")

        context = {"bad": BadObject(), "width": 100}

        # This should fail to resolve bad.attr but fall through to expression evaluation
        # which will also fail and give a proper error
        with pytest.raises(TemplateError):
            evaluator.evaluate("<bad.nonexistent + width>", context)

    def test_expression_evaluation_errors(self) -> None:
        """Test various expression evaluation error cases."""
        evaluator = TemplateEvaluator()
        context = {"width": 100}

        # Test division by zero
        with pytest.raises(TemplateError) as exc_info:
            evaluator.evaluate("<width / 0>", context)
        assert "division by zero" in str(exc_info.value)

        # Test invalid syntax
        with pytest.raises(TemplateError) as exc_info:
            evaluator.evaluate("<width +>", context)
        assert "invalid syntax" in str(exc_info.value)

    def test_function_argument_parsing_edge_cases(self) -> None:
        """Test edge cases in function argument parsing."""
        evaluator = TemplateEvaluator()
        context = {"name": "test"}

        # Test function with empty arguments - should fail
        with pytest.raises(TemplateError):
            evaluator.evaluate("len()", context)

        # Test function with nested variable resolution error
        with pytest.raises(TemplateError):
            evaluator.evaluate("upper(<nonexistent.nested>)", context)

    def test_template_variable_extraction_complex(self) -> None:
        """Test variable extraction from complex expressions."""
        evaluator = TemplateEvaluator()

        # Test expression that fails to compile - the malformed bracket won't be matched
        template = "<width + height * 2"  # Invalid syntax - missing closing bracket
        variables = evaluator.get_template_variables(template)
        # Since the bracket is unclosed, no variables should be extracted
        assert len(variables) == 0

        # Test valid complex expression
        template = "<width + height * 2>"
        variables = evaluator.get_template_variables(template)
        assert "width" in variables
        assert "height" in variables

    def test_template_validation_edge_cases(self) -> None:
        """Test template validation edge cases."""
        evaluator = TemplateEvaluator()
        context = {"width": 100}

        # Test template that raises TemplateError during evaluation
        template = "<nonexistent>"
        errors = evaluator.validate_template(template, context)
        assert len(errors) > 0
        assert "nonexistent" in errors[0]

    def test_get_evaluation_suggestion_cases(self) -> None:
        """Test evaluation suggestion generation."""
        evaluator = TemplateEvaluator()
        context = {"width": 100, "height": 200}

        # Test with function call
        try:
            evaluator.evaluate("unknown_func(<width>)", context)
        except TemplateError as e:
            assert "Available functions:" in e.suggestion

        # Test with variable reference
        try:
            evaluator.evaluate("<missing_var>", context)
        except TemplateError as e:
            assert "Available variables:" in e.suggestion
