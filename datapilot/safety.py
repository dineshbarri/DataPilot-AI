from __future__ import annotations

import ast

BLOCKED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)
BLOCKED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
}


class UnsafeCodeError(ValueError):
    pass


def validate_generated_expression(expression: str, max_characters: int = 2_000) -> ast.Expression:
    """Validate a calculation expression before it is sent to an isolated worker.

    DataPilot's primary workflow does not execute LLM-generated Python. This validator
    exists for an optional restricted calculation worker and accepts expressions only.
    """
    if len(expression) > max_characters:
        raise UnsafeCodeError("Expression exceeds the configured size limit.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeCodeError("Expression is not valid Python.") from exc
    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            raise UnsafeCodeError(f"Blocked syntax: {type(node).__name__}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError("Dunder attribute access is blocked.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCodeError("Dunder names are blocked.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                raise UnsafeCodeError(f"Blocked call: {node.func.id}.")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_CALLS:
                raise UnsafeCodeError(f"Blocked call: {node.func.attr}.")
    return tree
