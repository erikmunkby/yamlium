"""Security tests for yamlium.

These tests verify that yamlium protects against common YAML security vulnerabilities:
- Alias bomb (billion laughs) attacks
- Excessive nesting depth
- Circular references
"""

import pytest

from yamlium import ParsingError, parse


def test_alias_bomb_protection():
    """Alias bomb (billion laughs) attack should be detected and raise error.

    This test creates many alias references relative to a small number of actual nodes,
    which triggers the alias ratio protection (MAX_ALIAS_RATIO = 10).
    """
    # Create a YAML that has a high alias-to-decode ratio
    # Structure: 1 anchor, 1 key, then many aliases in a flow sequence
    # decode_count ~ 3 (key 'a', scalar '1', key 'b'), alias_count = 50
    # ratio = 50/3 ~ 16.7 > 10
    yaml = "a: &a 1\n"
    yaml += "b: [" + ", ".join(["*a"] * 50) + "]\n"
    with pytest.raises(ParsingError, match="Excessive aliasing"):
        parse(yaml)


def test_alias_bomb_moderate_usage_allowed():
    """Normal alias usage should not trigger protection."""
    yaml = """
base: &base
  name: default
  value: 42

derived1: *base
derived2: *base
derived3:
  <<: *base
  extra: value
"""
    # This should parse without error
    result = parse(yaml)
    assert result["base"]["name"] == "default"


def test_depth_limit_exceeded():
    """Deeply nested YAML should raise error when exceeding limit."""
    # Generate valid deeply nested YAML (more than MAX_DEPTH=200 levels)
    # Each level must have proper content
    depth = 202
    lines = []
    for i in range(depth):
        indent = "  " * i
        lines.append(f"{indent}level{i}:")
    # Add a final value
    lines.append(f"{'  ' * depth}value: end")
    yaml = "\n".join(lines)
    with pytest.raises(ParsingError, match="Maximum nesting depth"):
        parse(yaml)


def test_depth_limit_normal_nesting_allowed():
    """Normal nesting depth should be allowed."""
    yaml = """
level1:
  level2:
    level3:
      level4:
        level5:
          value: deep
"""
    result = parse(yaml)
    assert result["level1"]["level2"]["level3"]["level4"]["level5"]["value"] == "deep"


def test_depth_limit_sequence_nesting():
    """Sequence nesting should also be depth limited."""
    # Generate deeply nested sequences with mappings
    # Each iteration creates 2 depth levels (sequence + mapping), so 102 iterations = 204 levels
    depth = 102
    lines = ["root:"]
    for i in range(depth):
        indent = "  " * (i * 2 + 1)
        lines.append(f"{indent}- level{i}:")
    # Add a final value
    lines.append(f"{'  ' * (depth * 2 + 1)}value: end")
    yaml = "\n".join(lines)
    with pytest.raises(ParsingError, match="Maximum nesting depth"):
        parse(yaml)


def test_circular_reference_detection():
    """Self-referential anchor should raise error."""
    yaml = """
a: &a
  b: *a
"""
    with pytest.raises(ParsingError, match="Circular reference"):
        parse(yaml)


def test_circular_reference_simple():
    """Simple direct circular reference."""
    yaml = """
root: &root
  child: *root
"""
    with pytest.raises(ParsingError, match="Circular reference"):
        parse(yaml)


def test_non_circular_forward_reference_allowed():
    """Forward references that are not circular should be allowed."""
    yaml = """
first: &first
  value: 1

second: &second
  ref: *first
  value: 2

third:
  ref1: *first
  ref2: *second
"""
    result = parse(yaml)
    assert result["first"]["value"] == 1
    assert result["third"]["ref1"].child["value"] == 1


def test_alias_without_anchor():
    """Alias without a defined anchor should raise error."""
    yaml = """
key: *undefined_anchor
"""
    with pytest.raises(ParsingError, match="No anchor found"):
        parse(yaml)
