"""Template evaluation system for PhotoFlow.

This module provides safe template evaluation for dynamic expressions in
file naming, parameter values, and other user-configurable strings.
Inspired by Phatch's template system but with enhanced security.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar


class TemplateError(Exception):
    """Error in template evaluation."""

    def __init__(self, template: str, message: str, suggestion: str = "") -> None:
        """Initialize template error.

        Args:
            template: The template that failed to evaluate
            message: Error message
            suggestion: Optional suggestion for fixing the error
        """
        self.template = template
        self.message = message
        self.suggestion = suggestion

        full_message = f"Template evaluation failed: {message}"
        full_message += f"\nTemplate: {template}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"

        super().__init__(full_message)


class TemplateEvaluator:
    """Safe template evaluation with enhanced expression support.

    Enhanced version inspired by Phatch's template system with support for
    safe expression evaluation, comprehensive variable context, and secure
    function library. Supports both simple variable substitution and complex
    mathematical expressions.
    """

    # Enhanced safe function library organized by category
    SAFE_FUNCTIONS: ClassVar[dict[str, Any]] = {
        # Basic type conversion
        "abs": abs,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "len": len,
        # Mathematical operations
        "min": min,
        "max": max,
        "round": round,
        "pow": pow,
        "sum": sum,
        # String operations
        "upper": str.upper,
        "lower": str.lower,
        "title": str.title,
        "capitalize": str.capitalize,
        "strip": str.strip,
        "replace": str.replace,
        # Date/time operations (safe subset)
        "divmod": divmod,
        # List operations
        "sorted": sorted,
        "reversed": reversed,
    }

    # Safe built-ins that can be used in expressions
    SAFE_BUILTINS: ClassVar[dict[str, Any]] = {
        "__builtins__": {},
        "True": True,
        "False": False,
        "None": None,
    }

    def __init__(self) -> None:
        """Initialize template evaluator."""
        self._variable_pattern = re.compile(r"<([^>]+)>")
        self._format_pattern = re.compile(r"#{3,}")
        self._function_pattern = re.compile(r"(\w+)\(([^)]*)\)")

    def evaluate(self, template: str, context: dict[str, Any]) -> str:
        """Safely evaluate template with given context.

        Enhanced evaluation supporting both simple variable substitution
        and complex expressions within <> brackets, plus function calls.

        Args:
            template: Template string with <variable> placeholders and expressions
            context: Dictionary of variables available for substitution

        Returns:
            Evaluated template string

        Raises:
            TemplateError: If template evaluation fails

        Examples:
            >>> evaluator = TemplateEvaluator()
            >>> context = {"filename": "photo", "width": 1920, "height": 1080}
            >>> evaluator.evaluate("<filename>_<width>x<height>", context)
            'photo_1920x1080'
            >>> evaluator.evaluate("<filename>_<width*2>", context)
            'photo_3840'
            >>> evaluator.evaluate("upper(<filename>)_<min(width, 1000)>", context)
            'PHOTO_1000'
        """
        # Convert template to string if it's not already
        result = str(template)

        try:
            # Step 1: Handle function calls with <variable> arguments
            result = self._evaluate_functions(result, context)

            # Step 2: Replace variable placeholders and evaluate expressions
            result = self._substitute_variables_and_expressions(result, context)

            # Step 3: Handle format patterns (### -> %03d, etc.)
            result = self._handle_format_patterns(result)

            return result

        except Exception as e:
            suggestion = self._get_evaluation_suggestion(str(template), context)
            raise TemplateError(
                template=template,
                message=str(e),
                suggestion=suggestion,
            ) from e

    def _substitute_variables_and_expressions(
        self, template: str, context: dict[str, Any]
    ) -> str:
        """Substitute variables and evaluate expressions in <> brackets.

        Args:
            template: Template string
            context: Variable context

        Returns:
            Template with variables and expressions evaluated
        """

        def replace_expression(match: re.Match[str]) -> str:
            expr = match.group(1).strip()

            try:
                # Try simple variable lookup first
                if expr in context:
                    value = context[expr]
                    return str(value) if value is not None else ""

                # Try nested attribute access (e.g., image.width)
                if "." in expr and not any(op in expr for op in "+-*/()"):
                    try:
                        return self._resolve_nested_variable(expr, context)
                    except TemplateError:
                        raise  # Re-raise TemplateErrors with proper nested error messages
                    except Exception:
                        pass  # Fall through to expression evaluation

                # For simple variables, give a cleaner error message
                if expr.isidentifier() and expr not in context:
                    available_vars = ", ".join(sorted(context.keys()))
                    raise TemplateError(
                        template=template,
                        message=f"Variable '{expr}' not found",
                        suggestion=f"Available variables: {available_vars}",
                    )

                # Evaluate as expression
                return str(self._eval_safe_expression(expr, context))

            except TemplateError:
                # Re-raise TemplateErrors without wrapping
                raise
            except Exception as e:
                # Provide helpful error for failed expressions
                available_vars = ", ".join(sorted(context.keys()))
                raise TemplateError(
                    template=template,
                    message=f"Failed to evaluate expression '{expr}': {e!s}",
                    suggestion=f"Available variables: {available_vars}",
                ) from e

        return self._variable_pattern.sub(replace_expression, template)

    def _eval_safe_expression(self, expr: str, context: dict[str, Any]) -> Any:
        """Safely evaluate a Python expression with restricted namespace.

        Args:
            expr: Expression to evaluate
            context: Variable context

        Returns:
            Expression result

        Raises:
            TemplateError: If expression is unsafe or fails
        """
        # Create safe namespace
        safe_globals = self.SAFE_BUILTINS.copy()
        safe_globals.update(self.SAFE_FUNCTIONS)

        # Add context variables
        safe_locals = context.copy()

        try:
            # Compile expression to check for safety
            compiled = compile(expr, "<template>", "eval")

            # Check for dangerous operations
            self._validate_expression_safety(compiled)

            # Evaluate with restricted namespace
            result = eval(compiled, safe_globals, safe_locals)
            return result

        except (SyntaxError, NameError, TypeError, ValueError) as e:
            raise TemplateError(
                template=expr,
                message=f"Expression evaluation failed: {e!s}",
                suggestion="Check syntax and available functions",
            ) from e

    def _validate_expression_safety(self, compiled_code: Any) -> None:
        """Validate that compiled expression only uses safe operations.

        Args:
            compiled_code: Compiled expression code

        Raises:
            TemplateError: If expression contains unsafe operations
        """
        # Get bytecode names to check for dangerous operations
        names = compiled_code.co_names

        # List of dangerous operations/attributes
        dangerous = {
            "__import__",
            "__builtins__",
            "exec",
            "eval",
            "compile",
            "open",
            "file",
            "input",
            "raw_input",
            "reload",
            "__class__",
            "__bases__",
            "__subclasses__",
            "__dict__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
            "__globals__",
            "__locals__",
        }

        for name in names:
            if name in dangerous:
                raise TemplateError(
                    template=str(compiled_code),
                    message=f"Unsafe operation '{name}' not allowed in expressions",
                    suggestion="Use only whitelisted functions and variables",
                )

    def get_template_variables(self, template: str) -> set[str]:
        """Extract all variable names used in a template.

        Args:
            template: Template string

        Returns:
            Set of variable names found in template

        Example:
            >>> evaluator = TemplateEvaluator()
            >>> evaluator.get_template_variables("<filename>_<width>x<height>")
            {'filename', 'width', 'height'}
            >>> evaluator.get_template_variables("<upper(filename)>_<min(width, 1000)>")
            {'filename', 'width'}
        """
        variables = set()

        for match in self._variable_pattern.finditer(template):
            expr = match.group(1).strip()

            # Extract variables from the expression
            variables.update(self._extract_variables_from_expression(expr))

        return variables

    def _extract_variables_from_expression(self, expr: str) -> set[str]:
        """Extract variable names from an expression.

        Args:
            expr: Expression string

        Returns:
            Set of variable names used in the expression
        """
        variables = set()

        # Simple variable reference
        if expr.isidentifier():
            variables.add(expr)
            return variables

        # Handle nested attribute access
        if "." in expr and not any(op in expr for op in "+-*/()"):
            base_var = expr.split(".")[0]
            if base_var.isidentifier():
                variables.add(base_var)
            return variables

        # Parse expression to find variable names
        try:
            # Compile to get variable names from bytecode
            compiled = compile(expr, "<template>", "eval")
            for name in compiled.co_names:
                if name not in self.SAFE_FUNCTIONS and name not in self.SAFE_BUILTINS:
                    variables.add(name)
        except Exception:
            # Fallback: simple pattern matching for identifiers
            identifier_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
            for match in identifier_pattern.finditer(expr):
                name = match.group(1)
                if (
                    name not in self.SAFE_FUNCTIONS
                    and name not in self.SAFE_BUILTINS
                    and name not in {"True", "False", "None"}
                ):
                    variables.add(name)

        return variables

    def _resolve_nested_variable(self, var_expr: str, context: dict[str, Any]) -> str:
        """Resolve nested variable access like 'image.width'.

        Args:
            var_expr: Variable expression with dots
            context: Variable context

        Returns:
            Resolved value as string
        """
        parts = var_expr.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise TemplateError(
                    template=var_expr,
                    message=f"Cannot resolve '{part}' in '{var_expr}'",
                    suggestion="Check that the variable path is correct",
                )

        return str(value) if value is not None else ""

    def _resolve_nested_variable_value(
        self, var_expr: str, context: dict[str, Any]
    ) -> Any:
        """Resolve nested variable access like 'image.width' and return the actual value.

        Args:
            var_expr: Variable expression with dots
            context: Variable context

        Returns:
            Resolved value (not converted to string)
        """
        parts = var_expr.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise TemplateError(
                    template=var_expr,
                    message=f"Cannot resolve '{part}' in '{var_expr}'",
                    suggestion="Check that the variable path is correct",
                )

        return value

    def _handle_format_patterns(self, template: str) -> str:
        """Convert ### patterns to proper format strings.

        Args:
            template: Template with ### patterns

        Returns:
            Template with format patterns converted

        Example:
            "file_###.jpg" -> "file_%03d.jpg"
            "file_####.jpg" -> "file_%04d.jpg"
        """

        def replace_pattern(match: re.Match[str]) -> str:
            count = len(match.group())
            return f"%0{count}d"

        return self._format_pattern.sub(replace_pattern, template)

    def _evaluate_functions(self, template: str, context: dict[str, Any]) -> str:
        """Evaluate safe function calls in template.

        Args:
            template: Template with function calls
            context: Variable context

        Returns:
            Template with functions evaluated

        Note:
            Only whitelisted functions are allowed for security.
        """

        def replace_function(match: re.Match[str]) -> str:
            func_name = match.group(1)
            args_str = match.group(2).strip()

            if func_name not in self.SAFE_FUNCTIONS:
                raise TemplateError(
                    template=template,
                    message=f"Function '{func_name}' is not allowed",
                    suggestion=f"Use one of: {', '.join(sorted(self.SAFE_FUNCTIONS.keys()))}",
                )

            # Parse arguments - first substitute any <variable> references
            args = self._parse_function_args(args_str, context)

            # Call the function
            func = self.SAFE_FUNCTIONS[func_name]
            try:
                if func_name == "upper" and args:
                    # Handle string methods that are stored as unbound methods
                    result = args[0].upper()
                elif func_name == "lower" and args:
                    result = args[0].lower()
                elif func_name == "title" and args:
                    result = args[0].title()
                elif func_name == "capitalize" and args:
                    result = args[0].capitalize()
                elif func_name == "strip" and args:
                    result = args[0].strip()
                elif func_name == "replace" and len(args) >= 3:  # noqa: PLR2004
                    result = args[0].replace(args[1], args[2])
                else:
                    result = func(*args)
                return str(result)
            except Exception as e:
                raise TemplateError(
                    template=template,
                    message=f"Function '{func_name}' failed: {e}",
                    suggestion="Check function arguments and types",
                ) from e

        return self._function_pattern.sub(replace_function, template)

    def _parse_function_args(self, args_str: str, context: dict[str, Any]) -> list[Any]:
        """Parse function arguments from string.

        Args:
            args_str: Arguments string
            context: Variable context for variable resolution

        Returns:
            List of parsed arguments
        """
        if not args_str:
            return []

        args = []
        for arg_raw in args_str.split(","):
            arg = arg_raw.strip()
            if not arg:
                continue

            # Variable reference
            if arg.startswith("<") and arg.endswith(">"):
                var_name = arg[1:-1].strip()
                try:
                    # Try simple variable lookup first
                    if var_name in context:
                        args.append(context[var_name])
                    # Try nested attribute access
                    elif "." in var_name:
                        value = self._resolve_nested_variable_value(var_name, context)
                        args.append(value)
                    else:
                        raise TemplateError(
                            template=args_str,
                            message=f"Variable '{var_name}' not found",
                            suggestion=f"Available: {', '.join(context.keys())}",
                        )
                except TemplateError:
                    raise
                except Exception as e:
                    raise TemplateError(
                        template=args_str,
                        message=f"Error resolving variable '{var_name}': {e}",
                        suggestion=f"Available: {', '.join(context.keys())}",
                    ) from e
            # String literal
            elif (arg.startswith('"') and arg.endswith('"')) or (
                arg.startswith("'") and arg.endswith("'")
            ):
                args.append(arg[1:-1])
            # Numeric literal
            elif arg.isdigit():
                args.append(int(arg))
            elif "." in arg and arg.replace(".", "").replace("-", "").isdigit():
                args.append(float(arg))
            else:
                # Treat as string literal
                args.append(arg)

        return args

    def _get_evaluation_suggestion(self, template: str, context: dict[str, Any]) -> str:
        """Get helpful suggestion for template evaluation errors.

        Args:
            template: The template that failed
            context: Available context

        Returns:
            Helpful suggestion string
        """
        suggestions = []

        # Check for common issues
        if "<" in template and ">" in template:
            suggestions.append("Check that all variables in <brackets> are available")

        if context:
            available = ", ".join(sorted(context.keys()))
            suggestions.append(f"Available variables: {available}")

        if any(func in template for func in ["(", ")"]):
            safe_funcs = ", ".join(sorted(self.SAFE_FUNCTIONS.keys()))
            suggestions.append(f"Available functions: {safe_funcs}")

        return ". ".join(suggestions) if suggestions else "Check template syntax"

    def validate_template(self, template: str, context: dict[str, Any]) -> list[str]:
        """Validate a template against available context.

        Args:
            template: Template to validate
            context: Available context

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        try:
            # Try to evaluate the template
            self.evaluate(template, context)
        except TemplateError as e:
            errors.append(e.message)

        # Check for missing variables
        required_vars = self.get_template_variables(template)
        missing_vars = required_vars - set(context.keys())
        if missing_vars:
            errors.append(f"Missing variables: {', '.join(sorted(missing_vars))}")

        return errors


# Global instance for convenient access
default_evaluator = TemplateEvaluator()


def evaluate_template(template: str, context: dict[str, Any]) -> str:
    """Convenience function for template evaluation.

    Args:
        template: Template string
        context: Variable context

    Returns:
        Evaluated template

    Raises:
        TemplateError: If evaluation fails
    """
    return default_evaluator.evaluate(template, context)


def validate_template(template: str, context: dict[str, Any]) -> list[str]:
    """Convenience function for template validation.

    Args:
        template: Template to validate
        context: Available context

    Returns:
        List of validation errors (empty if valid)
    """
    return default_evaluator.validate_template(template, context)
