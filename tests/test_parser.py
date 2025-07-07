from pathlib import Path

import pytest
import yaml

from yamlium import ParsingError, parse_full


def comp(s: str, /, validate_yaml: bool = True, expected_result: str = "") -> None:
    """Quick parse comparison."""
    s = s.strip() + "\n"
    # Make sure yaml is valid using pyyaml
    if validate_yaml:
        yaml.safe_load(s)
    expected_result = expected_result.strip() + "\n" if expected_result else s
    assert parse_full(input=s).to_yaml() == expected_result


def test_check_everything_yaml_is_parsable():
    yaml.safe_load(Path("tests/test_everything.yaml").read_text())


def test_simple_key_value():
    comp("""
name: bob

hey: yo
""")


def test_nested_mappings():
    comp("""
name:

  hey: yo
""")


def test_simple_inline_comment():
    comp("""
name: bob #  a common name
""")


def test_varying_comments():
    comp("""
# More comments
#   With varying spacing
# And varying tabbing
key2: alice

key1: bob #  a common name
""")


def test_null_values():
    comp("""
key1:
  key2:
  key3: ~
  key4: null
""")


def test_complex_sequence():
    comp("""
hey:
  - yo
  - but:

      # hey
      - asdf: ey
""")


def test_complex_sequence2():
    comp("""
hey:
  - yo
  - first: mapping
    what: hey
  - second: mapping
""")


def test_dangling_key():
    with pytest.raises(ParsingError) as e:
        comp("""
hey:
  yo:
""")
    assert "Found unexpected token EOF" in str(e)


def test_empty_anchor():
    comp("""
hey:
  yo: &anchor asdf
""")


def test_simple_anchor_alias():
    comp("""
hey: &my_anchor up # Anchored scalar

what: *my_anchor #   Applied with simple alias
""")


def test_anchor_missing_key():
    with pytest.raises(expected_exception=ParsingError) as e:
        comp("""
hey: asdf &faulty_anchor
""")
    assert "Found unexpected token ANCHOR" in str(e)


def test_alias_missing_anchor():
    with pytest.raises(expected_exception=ParsingError) as e:
        comp(
            """
hey: *missing_anchor
""",
            validate_yaml=False,
        )
    assert "No anchor found for alias" in str(e)


def test_scalar_types():
    """Test all scalar types: string, integer, float, boolean, null"""
    comp("""
string: hello world
integer: 42
float: 3.14
boolean_true: true
boolean_false: false
null_value: null
""")


def test_multiline_strings():
    """Test multiline string handling"""
    comp("""
multiline: |
  This is a
  multiline string
  with multiple lines

folded: >
  This is a folded
  string that will
  be joined with spaces
""")


def test_nested_anchors_and_aliases():
    """Test complex anchor and alias usage"""
    comp("""
base: &base
  name: default
  value: 42

derived1:
  <<: *base
  extra: value1

derived2:
  <<: *base
  extra: value2
""")


def test_flow_style():
    """Test flow style (inline) collections"""
    comp("""
flow_map: { a: 1, b: 2, c: 3 }
flow_seq: [1, 2, 3, 4]
mixed: { a: [1, 2], b: { x: 1, y: 2 } }
""")


def test_empty_structures():
    """Test empty mappings and sequences"""
    comp("""
empty_map: {}
empty_seq: []
nested_empty:
  map: {}
  seq: []
""")


def test_complex_nesting():
    """Test deeply nested structures"""
    comp("""
level1:
  level2:
    level3:
      level4:
        level5:
          final: value
        other: value
      another: value
    more: value
  extra: value
""")


def test_sequence_of_mappings():
    """Test sequences containing mappings"""
    comp("""
items:
  - name: item1
    value: 1
  - name: item2
    value: 2
  - name: item3
    value: 3
""")


def test_mixed_comments():
    """Test various comment placements"""
    comp("""
# Document level comment
key1: value1 # Inline comment

# Block comment
# With multiple lines
key2: value2

key3: # Comment after key
  value3 # Comment after value
""")


def test_special_characters():
    """Test handling of special characters in keys and values"""
    comp("""
special_chars: "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
unicode: "你好世界"
""")


def test_indentation_variations():
    """Test different indentation styles"""
    comp(
        """
normal:
  key: value
  nested:
    key: value

compact:
 key: value
 nested:
  key: value

spaced:
    key: value
    nested:
        key: value
""",
        expected_result="""
normal:
  key: value
  nested:
    key: value

compact:
  key: value
  nested:
    key: value

spaced:
  key: value
  nested:
    key: value
""",
    )


def test_invalid_yaml():
    """Test various invalid YAML constructs"""
    invalid_cases = [
        "key: [unclosed sequence",
        "key: {unclosed mapping",
        "  - improperly indented",
        "key: *undefined_anchor",
    ]

    for case in invalid_cases:
        with pytest.raises((ParsingError)):
            comp(case, validate_yaml=False)


def test_preserve_comments():
    """Test that comments are preserved in the output"""
    input_yaml = """
# Top level comment
key1: value1 # Inline comment

# Block comment
key2: value2
"""
    result = parse_full(input=input_yaml)._to_yaml()
    assert "# Top level comment" in result
    assert "# Inline comment" in result
    assert "# Block comment" in result


def test_anchor_reuse():
    """Test reusing anchors in different contexts"""
    comp("""
base: &base
  name: default
  value: 42

derived1: *base
derived2: *base
derived3:
  <<: *base
  extra: value
""")


def test_everything_yaml():
    comp(Path("tests/test_everything.yaml").read_text())


def test_complex_comment_preservation():
    """Test comment preservation in complex structures"""
    comp("""
# Document level comment
key1: value1 # Inline comment

# Block comment
# With multiple lines
key2: # Comment after key
  nested: # Nested comment
    value2 # Value comment
  # Standalone comment
  another: value3
""")


def test_complex_anchor_chains():
    """Test complex anchor and alias relationships"""
    comp("""
base: &base
  name: default
  value: 42

derived1: &derived1
  <<: *base
  extra: value1

derived2: &derived2
  <<: *derived1
  more: value2

final:
  <<: *derived2
  final: value3
""")


def test_complex_indentation():
    """Test various indentation patterns and edge cases"""
    comp("""
level1:
  level2:
    level3:
      - item1
      - item2
    level3_alt:
      key: value
  level2_alt:
    - nested:
        key: value
      extra: field
    - simple: value
""")


def test_complex_flow_collections():
    """Test nested flow collections and edge cases"""
    comp("""
flow_map: { a: 1, b: [2, 3], c: { x: 4, y: 5 } }
flow_seq: [1, { a: 2 }, [3, 4], 5]
mixed: { a: [1, 2], b: { x: 1, y: 2 }, c: [3, { z: 4 }] }
""")


def test_complex_merge_keys():
    """Test YAML merge keys with multiple sources"""
    comp("""
base: &base
  name: default
  value: 42
  nested:
    key: value

override: &override
  value: 100
  extra: field

merged:
  <<: [*base, *override]
  additional: value
""")


def test_quoted_strings():
    """Test YAML merge keys with multiple sources"""
    comp("""
key1: 'string1'
key2: "string2"
key3: string3
""")


def test_irregular_multiline_indentation():
    comp("""
key1: line1
  line2
    line3

key2: |
  line2
  line3
""")


def test_scalar_starting_after_key():
    comp("""
key1:
  scalar start

  scalar_continue
""")


def test_deeply_nested_multiline():
    comp("""
key1:
  key2:
    key3: |
      some text
      on multiple lines
    key4: normal scalar
""")


def test_multiline_in_sequence():
    comp("""
key1:
  - name: item1
    meta: |
      some multiline
      text
  - name: item2
    meta: some single line info
""")


def test_comments_in_sequences():
    comp("""
key1:
  # Comment on sequence item start
  - key: val
    # Comment on mapping object
    key2: val
""")
