import pytest

from yamlium.exceptions import ParsingError
from yamlium.lexer import Lexer, T


def comp(s: str, expected: list[T], /) -> None:
    """Quick lexer comparison."""
    s = s.strip() + "\n"
    assert [t.t for t in Lexer(s).build_tokens()] == expected


def test_simple_key_value():
    """Test lexing of simple key-value pairs."""
    lexer = Lexer("name: bob")
    tokens = lexer.build_tokens()

    assert len(tokens) == 3  # KEY, STRING, EOF
    assert tokens[0].t == T.KEY and tokens[0].value == "name"
    assert tokens[1].t == T.SCALAR and tokens[1].value == "bob"
    assert tokens[2].t == T.EOF


def test_nested_mapping():
    """Test lexing of nested mappings with indentation."""
    comp(
        """
person:
    name: alice
    age: 30
""",
        [
            T.KEY,
            T.INDENT,
            T.KEY,
            T.SCALAR,
            T.KEY,
            T.SCALAR,
            T.EOF,
        ],
    )


def test_sequence():
    """Test lexing of sequences (lists)."""
    comp(
        """
items:
    - first
    - second
    """,
        [
            T.KEY,
            T.INDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.DEDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.EOF,
        ],
    )


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

    lexer = Lexer(yaml_input)
    tokens = lexer.build_tokens()

    # Verify key structural elements
    token_types = [t.t for t in tokens]

    # Verify we have the correct number of DASH tokens (5 total: one for each user and one for each role)
    assert token_types.count(T.DASH) == 5, "Expected 5 DASH tokens"

    # Verify we have the correct number of INDENT/DEDENT pairs
    n_indent = token_types.count(T.INDENT)
    n_dedent = token_types.count(T.DEDENT)
    assert token_types.count(T.INDENT) == 9, (
        f"Expected 9 INDENT tokens found {n_indent}"
    )
    assert token_types.count(T.DEDENT) == 4, (
        f"Expected 4 DEDENT tokens found {n_dedent}"
    )
    assert token_types.count(T.KEY) == 5


def test_empty_values():
    """Test lexing of empty or null values."""
    yaml_input = """
empty:
null_value: null
blank_value: 
    """

    lexer = Lexer(yaml_input)
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

    lexer = Lexer(yaml_input)
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
    lexer = Lexer(yaml_input)
    tokens = lexer.build_tokens()
    assert [t.t for t in tokens] == [T.KEY, T.SCALAR, T.COMMENT, T.EMPTY_LINE, T.EOF]
    assert tokens[2].value == "# some comment"


def test_quote_not_ending():
    yaml_input = """
key1: " """
    lexer = Lexer(yaml_input)
    with pytest.raises(ParsingError):
        lexer.build_tokens()


def test_broken_quote_newline():
    yaml_input = """
key1: " 
"""
    lexer = Lexer(yaml_input)
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
        tokens = Lexer(y).build_tokens()
        assert [t.t for t in tokens] == [T.KEY, T.SCALAR, T.EOF]


def test_multiline_pipe():
    y1 = """
key1: |-
  line1
  line2
key2: normal scalar
"""
    y2 = """
key1: |-
  line1
  line2
key2: normal scalar
"""
    for y in [y1, y2]:
        comp(y, [T.KEY, T.MULTILINE_PIPE, T.KEY, T.SCALAR, T.EOF])


def test_multiline_string():
    yaml_input = """
key: simple multiline
  string without pipe
"""
    tokens = Lexer(yaml_input).build_tokens()
    expected_types = [T.KEY, T.SCALAR, T.EOF]
    assert [t.t for t in tokens] == expected_types


def test_multiline_string_error():
    yaml_input = """
key: simple multiline
  string illegal key:
"""
    with pytest.raises(ParsingError):
        Lexer(yaml_input).build_tokens()


def test_flow_style_mapping():
    y1 = """
flow_map: { a: 1, b: 2 }
"""
    comp(
        y1,
        [
            T.KEY,
            T.MAPPING_START,
            T.KEY,
            T.SCALAR,
            T.COMMA,
            T.KEY,
            T.SCALAR,
            T.MAPPING_END,
            T.EOF,
        ],
    )


def test_flow_style_sequence():
    y1 = """
flow_map: [ a, 3, c]
"""
    comp(
        y1,
        [
            T.KEY,
            T.SEQUENCE_START,
            T.SCALAR,
            T.COMMA,
            T.SCALAR,
            T.COMMA,
            T.SCALAR,
            T.SEQUENCE_END,
            T.EOF,
        ],
    )


def test_anchors_and_aliases():
    """Test lexing of YAML anchors and aliases."""
    yaml_input = """
base: &base
  name: base
derived: *base
"""
    comp(
        yaml_input,
        [T.KEY, T.ANCHOR, T.INDENT, T.KEY, T.SCALAR, T.DEDENT, T.KEY, T.ALIAS, T.EOF],
    )


def test_merge_keys():
    """Test lexing of YAML merge keys."""
    yaml_input = """
base: &base
    name: base
derived:
    <<: *base
    extra: value
"""
    comp(
        yaml_input,
        [
            T.KEY,
            T.ANCHOR,
            T.INDENT,
            T.KEY,
            T.SCALAR,
            T.DEDENT,
            T.KEY,
            T.INDENT,
            T.KEY,
            T.ALIAS,
            T.KEY,
            T.SCALAR,
            T.EOF,
        ],
    )


def test_nested_flow_style():
    """Test lexing of nested flow style structures."""
    yaml_input = """
complex: { a: [1, 2], b: { x: 1, y: 2 } }
"""
    comp(
        yaml_input,
        [
            T.KEY,
            T.MAPPING_START,
            T.KEY,
            T.SEQUENCE_START,
            T.SCALAR,
            T.COMMA,
            T.SCALAR,
            T.SEQUENCE_END,
            T.COMMA,
            T.KEY,
            T.MAPPING_START,
            T.KEY,
            T.SCALAR,
            T.COMMA,
            T.KEY,
            T.SCALAR,
            T.MAPPING_END,
            T.MAPPING_END,
            T.EOF,
        ],
    )


def test_block_scalar_modifiers():
    """Test lexing of block scalar modifiers."""
    yaml_input = """
literal: |
    This is a literal
    block scalar
folded: >
    This is a folded
    block scalar
"""
    comp(yaml_input, [T.KEY, T.MULTILINE_PIPE, T.KEY, T.MULTILINE_ARROW, T.EOF])


def test_document_separators():
    """Test lexing of YAML document separators."""
    yaml_input = """
---
key1: value1
---
key2: value2
"""
    comp(
        yaml_input,
        [
            T.DOCUMENT_START,
            T.KEY,
            T.SCALAR,
            T.DOCUMENT_START,
            T.KEY,
            T.SCALAR,
            T.EOF,
        ],
    )


def test_multiple_documents():
    """Test lexing of multiple YAML documents."""
    yaml_input = """
---
doc1: value1
---
doc2: value2
"""
    comp(
        yaml_input,
        [T.DOCUMENT_START, T.KEY, T.SCALAR, T.DOCUMENT_START, T.KEY, T.SCALAR, T.EOF],
    )


def test_complex_anchors():
    """Test lexing of complex anchor and alias structures."""
    yaml_input = """
base: &base
    name: base
    items: &items
        - one
        - two
derived:
    <<: *base
    items: *items
"""
    comp(
        yaml_input,
        [
            T.KEY,
            T.ANCHOR,
            T.INDENT,
            T.KEY,
            T.SCALAR,
            T.KEY,
            T.ANCHOR,
            T.INDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.DEDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.DEDENT,
            T.DEDENT,
            T.DEDENT,
            T.KEY,
            T.INDENT,
            T.KEY,
            T.ALIAS,
            T.KEY,
            T.ALIAS,
            T.EOF,
        ],
    )


def test_newline_detection():
    yaml_input = """
key1: value1

key2: value
"""
    comp(yaml_input, [T.KEY, T.SCALAR, T.EMPTY_LINE, T.KEY, T.SCALAR, T.EOF])


# test multilines with newlines in them
def test_multilines_with_newlines():
    y1 = """
key1: |

  scalar
"""
    comp(y1, [T.KEY, T.MULTILINE_PIPE, T.EOF])

    y2 = """
key1: scalar start

  scalar continue
"""
    comp(y2, [T.KEY, T.SCALAR, T.EOF])


def test_various_multiline_newlines():
    yml = """
key1: |

  line1

  line2


key2: scalar
"""
    comp(
        yml,
        [T.KEY, T.MULTILINE_PIPE, T.EMPTY_LINE, T.EMPTY_LINE, T.KEY, T.SCALAR, T.EOF],
    )


def test_dashes_in_sequence():
    yml = """
key1:
  - normal
  - 123
  - -123 # negative number
  - -"""  # The final is a dash with immediate EOF
    comp(
        yml,
        [
            T.KEY,
            T.INDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.DEDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.DEDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.COMMENT,
            T.DEDENT,
            T.DASH,
            T.INDENT,
            T.SCALAR,
            T.EOF,
        ],
    )


def test_quoted_key():
    yml = """
normal_key:
  "quoted_key": scalar
"""
    comp(yml, [T.KEY, T.INDENT, T.KEY, T.SCALAR, T.EOF])


def test_flow_seq_in_seq():
    y = """
flow_seq: [1, [3, 4], 5]
"""
    comp(
        y,
        [
            T.KEY,
            T.SEQUENCE_START,
            T.SCALAR,
            T.COMMA,
            T.SEQUENCE_START,
            T.SCALAR,
            T.COMMA,
            T.SCALAR,
            T.SEQUENCE_END,
            T.COMMA,
            T.SCALAR,
            T.SEQUENCE_END,
            T.EOF,
        ],
    )


def test_mixed_flow_style():
    yml = """
mixed: { a: [1, 2], b: { x: 1, y: 2 }, c: [3, { z: 4 }] }
"""
    comp(
        yml,
        [
            T.KEY,
            T.MAPPING_START,
            T.KEY,
            T.SEQUENCE_START,
            T.SCALAR,
            T.COMMA,
            T.SCALAR,
            T.SEQUENCE_END,
            T.COMMA,
            T.KEY,
            T.MAPPING_START,
            T.KEY,
            T.SCALAR,
            T.COMMA,
            T.KEY,
            T.SCALAR,
            T.MAPPING_END,
            T.COMMA,
            T.KEY,
            T.SEQUENCE_START,
            T.SCALAR,
            T.COMMA,
            T.MAPPING_START,
            T.KEY,
            T.SCALAR,
            T.MAPPING_END,
            T.SEQUENCE_END,
            T.MAPPING_END,
            T.EOF,
        ],
    )
