import pytest

from yamlium.exceptions import ParsingError
from yamlium.lexer import Lexer, T


def test_simple_key_value():
    """Test lexing of simple key-value pairs."""
    lexer = Lexer(input="name: bob")
    tokens = lexer.build_tokens()

    assert len(tokens) == 3  # KEY, STRING, EOF
    assert tokens[0].t == T.KEY and tokens[0].value == "name"
    assert tokens[1].t == T.SCALAR and tokens[1].value == "bob"
    assert tokens[2].t == T.EOF


def test_nested_mapping():
    """Test lexing of nested mappings with indentation."""
    yaml_input = """
person:
    name: alice
    age: 30
    """

    lexer = Lexer(input=yaml_input)
    tokens = lexer.build_tokens()

    # Expected token sequence: NEW_LINE, KEY, NEW_LINE, INDENT, KEY, STRING, NEW_LINE, KEY, STRING, DEDENT, EOF
    expected_types = [
        T.KEY,
        T.INDENT,
        T.KEY,
        T.SCALAR,
        T.KEY,
        T.SCALAR,
        T.EOF,
    ]

    assert len(tokens) == len(expected_types)
    assert [t.t for t in tokens] == expected_types


def test_sequence():
    """Test lexing of sequences (lists)."""
    yaml_input = """
items:
    - first
    - second
    """

    lexer = Lexer(input=yaml_input)
    tokens = lexer.build_tokens()

    # Expected: NEW_LINE, KEY, NEW_LINE, INDENT, DASH, STRING, NEW_LINE, DASH, STRING, DEDENT, EOF
    expected_types = [
        T.KEY,
        T.INDENT,
        T.DASH,
        T.SCALAR,
        T.DASH,
        T.SCALAR,
        T.EOF,
    ]

    assert len(tokens) == len(expected_types)
    assert [t.t for t in tokens] == expected_types


def test_complex_structure():
    """Test lexing of a more complex YAML structure with nested sequences and mappings."""
    yaml_input = """
users:
    - name: alice
      roles:
        - admin
        - user
    - name:
        roles:
            - user
    """

    lexer = Lexer(input=yaml_input)
    tokens = lexer.build_tokens()

    # Verify key structural elements
    token_types = [t.t for t in tokens]

    # Verify we have the correct number of DASH tokens (5 total: one for each user and one for each role)
    assert token_types.count(T.DASH) == 5, "Expected 5 DASH tokens"

    # Verify we have the correct number of INDENT/DEDENT pairs
    n_indent = token_types.count(T.INDENT)
    n_dedent = token_types.count(T.DEDENT)
    assert token_types.count(T.INDENT) == 5, (
        f"Expected 5 INDENT tokens found {n_indent}"
    )
    assert token_types.count(T.DEDENT) == 2, (
        f"Expected 2 DEDENT tokens found {n_dedent}"
    )
    assert token_types.count(T.KEY) == 5


def test_empty_values():
    """Test lexing of empty or null values."""
    yaml_input = """
empty:
null_value: null
blank_value: 
    """

    lexer = Lexer(input=yaml_input)
    tokens = lexer.build_tokens()

    # Verify we can handle empty values correctly
    token_types = [t.t for t in tokens]
    token_values = [t.value for t in tokens]

    assert T.KEY in token_types
    assert "empty" in token_values
    assert "null_value" in token_values
    assert "blank_value" in token_values


def test_line_column_tracking():
    """Test that line and column numbers are tracked correctly."""
    yaml_input = """
key1: value1
key2:
    nested: value2
"""

    lexer = Lexer(input=yaml_input)
    tokens = lexer.build_tokens()

    # Find the 'nested' key token
    nested_token = next(t for t in tokens if t.value == "nested")

    # The nested key should be on line 3 (0-based) and have a column number > 0
    assert nested_token.line == 3
    assert nested_token.column > 0


def test_comment():
    yaml_input = """
key1: value1 # some comment

"""
    lexer = Lexer(input=yaml_input)
    tokens = lexer.build_tokens()
    assert [t.t for t in tokens] == [T.KEY, T.SCALAR, T.COMMENT, T.EMPTY_LINE, T.EOF]
    assert tokens[2].value == "# some comment"


def test_quote_not_ending():
    yaml_input = """
key1: " """
    lexer = Lexer(input=yaml_input)
    with pytest.raises(ParsingError):
        lexer.build_tokens()


def test_broken_quote_newline():
    yaml_input = """
key1: " 
"""
    lexer = Lexer(input=yaml_input)
    with pytest.raises(ParsingError):
        lexer.build_tokens()


def test_simple_quote():
    y1 = """
key1: 'my value'
"""
    y2 = """
key1: "my value"
"""
    y3 = """
key1: "&#-XXX"
"""
    for y in [y1, y2, y3]:
        tokens = Lexer(input=y).build_tokens()
        assert [t.t for t in tokens] == [T.KEY, T.SCALAR, T.EOF]
