import pytest

from datapilot.safety import UnsafeCodeError, validate_generated_expression


def test_accepts_scalar_expression():
    tree = validate_generated_expression("(revenue - cost) / revenue")
    assert tree is not None


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "open('secret.txt').read()",
        "(1).__class__.__mro__",
    ],
)
def test_blocks_unsafe_expression(expression):
    with pytest.raises(UnsafeCodeError):
        validate_generated_expression(expression)
