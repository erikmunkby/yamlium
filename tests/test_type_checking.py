"""
Static type checking test file for yamlium.

This file is used to verify that type checkers (mypy, pyright) correctly
understand the types in yamlium. Run with:
    mypy tests/test_type_checking.py
    pyright tests/test_type_checking.py

This file should have NO type errors when checked.
"""

from typing import Any
from yamlium import parse
from yamlium.nodes import Key, Mapping, Scalar, Sequence


def test_mapping_basic_types() -> None:
    """Test that Mapping accepts all basic Python types like dict does."""
    m: Mapping = Mapping()

    # All these should be valid (like dict)
    m["string_key"] = "string value"
    m["int_key"] = 42
    m["float_key"] = 3.14
    m["bool_key"] = True
    m["none_key"] = None

    # Nested structures
    m["dict_key"] = {"nested": "value"}
    m["list_key"] = [1, 2, 3]

    # Complex nested structures
    m["complex"] = {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ],
        "count": 2,
        "active": True,
    }


def test_mapping_with_key_object() -> None:
    """Test that Mapping accepts both string and Key objects as keys."""
    m: Mapping = Mapping()
    key: Key = Key("mykey")

    m[key] = "value"
    m["string_key"] = "value"

    # Both should work for access
    _val1: Any = m[key]
    _val2: Any = m["string_key"]


def test_mapping_update() -> None:
    """Test that Mapping.update works like dict.update."""
    m: Mapping = Mapping()

    # Update with dict of mixed types
    m.update(
        {
            "string": "text",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "bool": False,
            "none": None,
        }
    )


def test_mapping_methods() -> None:
    """Test that Mapping methods have correct signatures."""
    m: Mapping = Mapping()
    m["key"] = "value"

    # get method
    _val1: Any = m.get("key")
    _val2: Any = m.get("nonexistent", "default")

    # pop method
    _popped: Any = m.pop("key")

    # keys, values, items
    _keys = m.keys()
    _values = m.values()
    _items = m.items()

    # iteration
    for _k in m:
        pass

    for _k, _v in m.items():
        pass


def test_sequence_basic_types() -> None:
    """Test that Sequence accepts all basic Python types like list does."""
    s: Sequence = Sequence()

    # Setting items with different types
    s.append("string")
    s.append(42)
    s.append(3.14)
    s.append(True)
    s.append(None)
    s.append({"nested": "dict"})
    s.append([1, 2, 3])

    # The issue that was reported: setting list values
    s[0] = "new string"
    s[1] = 123
    s[2] = ["a", "b", "c"]  # This should NOT cause a type error
    s[3] = {"key": "value"}

    # Complex nested structures
    s.append(
        {
            "users": [
                {"name": "Alice"},
                {"name": "Bob"},
            ]
        }
    )


def test_sequence_methods() -> None:
    """Test that Sequence methods have correct signatures."""
    s: Sequence = Sequence()

    # append
    s.append("item")
    s.append(42)
    s.append([1, 2])
    s.append({"key": "value"})

    # extend
    s.extend([1, 2, 3])
    s.extend(["a", "b", "c"])
    s.extend([{"dict": True}, [1, 2]])

    # iteration
    for _item in s:
        pass

    # indexing
    _val1: Any = s[0]
    _val2: Any = s[-1]

    # length
    _length: int = len(s)


def test_parsed_yaml_typing() -> None:
    """Test that parsed YAML has correct types."""
    yaml_str = """
name: John
age: 30
active: true
scores:
  - 90
  - 85
  - 92
metadata:
  created: 2024-01-01
  tags:
    - python
    - yaml
"""
    result: Mapping = parse(yaml_str)

    # Access with type annotations
    _name: Any = result["name"]
    _age: Any = result["age"]
    _active: Any = result["active"]
    _scores: Any = result["scores"]
    _metadata: Any = result["metadata"]

    # Nested access
    _created: Any = result["metadata"]["created"]
    _tags: Any = result["metadata"]["tags"]


def test_node_preservation() -> None:
    """Test that Node objects can be assigned."""
    m: Mapping = Mapping()
    s: Sequence = Sequence()

    # Assigning Node objects should work
    scalar: Scalar = Scalar(_value="test")
    nested_map: Mapping = Mapping()
    nested_seq: Sequence = Sequence()

    m["scalar"] = scalar
    m["map"] = nested_map
    m["seq"] = nested_seq

    s.append(scalar)
    s.append(nested_map)
    s.append(nested_seq)


def test_dict_like_behavior() -> None:
    """Test that Mapping behaves like dict in type system."""
    m: Mapping = Mapping()

    # Should accept same operations as dict
    m["key1"] = "value1"
    m["key2"] = 42
    m["key3"] = [1, 2, 3]
    m["key4"] = {"nested": True}

    # Check membership
    _exists: bool = "key1" in m
    _not_exists: bool = "other" not in m

    # Length
    _size: int = len(m)


def test_list_like_behavior() -> None:
    """Test that Sequence behaves like list in type system."""
    s: Sequence = Sequence()

    # Should accept same operations as list
    s.append("item1")
    s.append(42)
    s.append([1, 2])
    s.append({"key": "value"})

    s[0] = "new value"
    s[1] = 123
    s[2] = ["nested", "list"]  # The reported bug - should work

    # Extend
    s.extend([1, 2, 3])

    # Length
    _size: int = len(s)


def test_edge_cases() -> None:
    """Test edge cases with types."""
    m: Mapping = Mapping()
    s: Sequence = Sequence()

    # Empty containers
    m["empty_dict"] = {}
    m["empty_list"] = []

    s.append({})
    s.append([])

    # Deeply nested
    m["deep"] = {"l1": {"l2": {"l3": "value"}}}
    s.append([[1, 2], [3, 4]])

    # Mixed nesting
    m["mixed"] = [
        {"name": "Alice", "scores": [90, 85]},
        {"name": "Bob", "scores": [88, 91]},
    ]


def test_any_value_assignment() -> None:
    """Test that Any type values can be assigned."""

    def get_dynamic_value() -> Any:
        return {"dynamic": "value"}

    m: Mapping = Mapping()
    s: Sequence = Sequence()

    dynamic_val: Any = get_dynamic_value()

    # Should accept Any typed values
    m["dynamic"] = dynamic_val
    s.append(dynamic_val)
    s[0] = dynamic_val


# This file should pass mypy and pyright checks with no errors
