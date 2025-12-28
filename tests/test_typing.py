"""
Comprehensive typing tests for yamlium.

This test suite ensures that Mapping and Sequence classes behave like
native dict and list types, and that type annotations are correct.
"""

from yamlium import parse
from yamlium.nodes import Key, Mapping, Scalar, Sequence


class TestMappingTyping:
    """Test that Mapping behaves like a dict."""

    def test_setitem_with_string_value(self):
        """Test setting a string value."""
        m = Mapping()
        m["key"] = "value"
        assert m["key"]._value == "value"
        assert isinstance(m["key"], Scalar)

    def test_setitem_with_int_value(self):
        """Test setting an integer value."""
        m = Mapping()
        m["count"] = 42
        assert m["count"]._value == 42
        assert isinstance(m["count"], Scalar)

    def test_setitem_with_float_value(self):
        """Test setting a float value."""
        m = Mapping()
        m["price"] = 19.99
        assert m["price"]._value == 19.99
        assert isinstance(m["price"], Scalar)

    def test_setitem_with_bool_value(self):
        """Test setting a boolean value."""
        m = Mapping()
        m["enabled"] = True
        assert m["enabled"]._value is True
        assert isinstance(m["enabled"], Scalar)

    def test_setitem_with_none_value(self):
        """Test setting None value."""
        m = Mapping()
        m["empty"] = None
        assert m["empty"]._value is None
        assert isinstance(m["empty"], Scalar)

    def test_setitem_with_dict_value(self):
        """Test setting a dict value."""
        m = Mapping()
        m["nested"] = {"inner": "value"}
        assert isinstance(m["nested"], Mapping)
        assert m["nested"]["inner"]._value == "value"

    def test_setitem_with_list_value(self):
        """Test setting a list value."""
        m = Mapping()
        m["items"] = ["a", "b", "c"]
        assert isinstance(m["items"], Sequence)
        assert len(m["items"]) == 3
        assert m["items"][0]._value == "a"

    def test_setitem_with_nested_structures(self):
        """Test setting complex nested structures."""
        m = Mapping()
        m["data"] = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            "count": 2,
        }
        assert isinstance(m["data"], Mapping)
        assert isinstance(m["data"]["users"], Sequence)
        assert len(m["data"]["users"]) == 2
        assert m["data"]["users"][0]["name"]._value == "Alice"
        assert m["data"]["count"]._value == 2

    def test_setitem_with_key_object(self):
        """Test setting value with Key object as key."""
        m = Mapping()
        key = Key("mykey")
        m[key] = "value"
        assert m["mykey"]._value == "value"

    def test_update_with_dict(self):
        """Test updating mapping with a dict."""
        m = Mapping()
        m["existing"] = "old"
        m.update({"new": "value", "existing": "updated"})
        assert m["new"]._value == "value"
        assert m["existing"]._value == "updated"

    def test_update_with_mixed_types(self):
        """Test updating with mixed value types."""
        m = Mapping()
        m.update(
            {
                "string": "text",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": True},
            }
        )
        assert m["string"]._value == "text"
        assert m["number"]._value == 42
        assert isinstance(m["list"], Sequence)
        assert isinstance(m["dict"], Mapping)

    def test_get_method(self):
        """Test get method works like dict.get()."""
        m = Mapping()
        m["key"] = "value"
        assert m.get("key")._value == "value"
        assert m.get("nonexistent") is None
        assert m.get("nonexistent", "default") == "default"

    def test_pop_method(self):
        """Test pop method works like dict.pop()."""
        m = Mapping()
        m["key"] = "value"
        popped = m.pop("key")
        assert popped._value == "value"
        assert "key" not in m

    def test_keys_method(self):
        """Test keys method returns Key objects."""
        m = Mapping()
        m["a"] = 1
        m["b"] = 2
        keys = list(m.keys())
        assert len(keys) == 2
        assert all(isinstance(k, Key) for k in keys)

    def test_values_method(self):
        """Test values method returns Node objects."""
        m = Mapping()
        m["a"] = 1
        m["b"] = "text"
        values = list(m.values())
        assert len(values) == 2
        assert values[0]._value == 1
        assert values[1]._value == "text"

    def test_items_method(self):
        """Test items method returns (Key, Node) tuples."""
        m = Mapping()
        m["a"] = 1
        m["b"] = 2
        items = list(m.items())
        assert len(items) == 2
        assert all(isinstance(k, Key) for k, v in items)
        assert all(hasattr(v, "_value") for k, v in items)

    def test_iteration(self):
        """Test iterating over mapping yields keys."""
        m = Mapping()
        m["a"] = 1
        m["b"] = 2
        m["c"] = 3
        keys = [k for k in m]
        assert len(keys) == 3
        assert all(isinstance(k, Key) for k in keys)

    def test_contains(self):
        """Test 'in' operator works."""
        m = Mapping()
        m["key"] = "value"
        assert "key" in m
        assert "nonexistent" not in m

    def test_len(self):
        """Test len() works."""
        m = Mapping()
        assert len(m) == 0
        m["a"] = 1
        assert len(m) == 1
        m["b"] = 2
        assert len(m) == 2


class TestSequenceTyping:
    """Test that Sequence behaves like a list."""

    def test_setitem_with_string_value(self):
        """Test setting a string value."""
        s = Sequence([Scalar("old")])
        s[0] = "new"
        assert s[0]._value == "new"
        assert isinstance(s[0], Scalar)

    def test_setitem_with_int_value(self):
        """Test setting an integer value."""
        s = Sequence([Scalar(0)])
        s[0] = 42
        assert s[0]._value == 42
        assert isinstance(s[0], Scalar)

    def test_setitem_with_float_value(self):
        """Test setting a float value."""
        s = Sequence([Scalar(0.0)])
        s[0] = 3.14
        assert s[0]._value == 3.14
        assert isinstance(s[0], Scalar)

    def test_setitem_with_bool_value(self):
        """Test setting a boolean value."""
        s = Sequence([Scalar(False)])
        s[0] = True
        assert s[0]._value is True
        assert isinstance(s[0], Scalar)

    def test_setitem_with_none_value(self):
        """Test setting None value."""
        s = Sequence([Scalar("something")])
        s[0] = None
        assert s[0]._value is None
        assert isinstance(s[0], Scalar)

    def test_setitem_with_dict_value(self):
        """Test setting a dict value."""
        s = Sequence([Scalar("placeholder")])
        s[0] = {"key": "value"}
        assert isinstance(s[0], Mapping)
        assert s[0]["key"]._value == "value"

    def test_setitem_with_list_value(self):
        """Test setting a list value (THIS WAS THE REPORTED BUG)."""
        s = Sequence([Scalar("placeholder")])
        s[0] = ["a", "b", "c"]
        assert isinstance(s[0], Sequence)
        assert len(s[0]) == 3
        assert s[0][0]._value == "a"

    def test_setitem_with_nested_structures(self):
        """Test setting complex nested structures."""
        s = Sequence([Scalar("placeholder")])
        s[0] = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        }
        assert isinstance(s[0], Mapping)
        assert isinstance(s[0]["users"], Sequence)
        assert s[0]["users"][0]["name"]._value == "Alice"

    def test_append_with_string(self):
        """Test appending a string."""
        s = Sequence()
        s.append("value")
        assert s[0]._value == "value"
        assert isinstance(s[0], Scalar)

    def test_append_with_int(self):
        """Test appending an integer."""
        s = Sequence()
        s.append(42)
        assert s[0]._value == 42
        assert isinstance(s[0], Scalar)

    def test_append_with_dict(self):
        """Test appending a dict."""
        s = Sequence()
        s.append({"key": "value"})
        assert isinstance(s[0], Mapping)
        assert s[0]["key"]._value == "value"

    def test_append_with_list(self):
        """Test appending a list."""
        s = Sequence()
        s.append([1, 2, 3])
        assert isinstance(s[0], Sequence)
        assert len(s[0]) == 3

    def test_extend_with_list(self):
        """Test extending with a list."""
        s = Sequence()
        s.extend([1, 2, 3])
        assert len(s) == 3
        assert s[0]._value == 1
        assert s[1]._value == 2
        assert s[2]._value == 3

    def test_extend_with_mixed_types(self):
        """Test extending with mixed types."""
        s = Sequence()
        s.extend(["text", 42, True, None, [1, 2], {"a": 1}])
        assert len(s) == 6
        assert s[0]._value == "text"
        assert s[1]._value == 42
        assert s[2]._value is True
        assert s[3]._value is None
        assert isinstance(s[4], Sequence)
        assert isinstance(s[5], Mapping)

    def test_iteration(self):
        """Test iterating over sequence."""
        s = Sequence()
        s.extend([1, 2, 3])
        values = [item._value for item in s]
        assert values == [1, 2, 3]

    def test_len(self):
        """Test len() works."""
        s = Sequence()
        assert len(s) == 0
        s.append(1)
        assert len(s) == 1
        s.append(2)
        assert len(s) == 2

    def test_getitem_with_negative_index(self):
        """Test negative indexing works."""
        s = Sequence()
        s.extend([1, 2, 3])
        assert s[-1]._value == 3
        assert s[-2]._value == 2
        assert s[-3]._value == 1

    def test_contains(self):
        """Test 'in' operator works (checks by Node, not value)."""
        s = Sequence()
        s.extend([1, 2, 3])
        # This tests the actual node objects
        assert s[0] in s
        assert s[1] in s


class TestRoundTripTyping:
    """Test that types are preserved through parse and serialization."""

    def test_parse_preserves_types(self):
        """Test that parsing YAML preserves type information."""
        yaml_str = """
string_key: text value
int_key: 42
float_key: 3.14
bool_key: true
null_key: null
list_key:
  - item1
  - item2
dict_key:
  nested: value
"""
        result = parse(yaml_str)
        assert result["string_key"]._value == "text value"
        assert result["int_key"]._value == 42
        assert result["float_key"]._value == 3.14
        assert result["bool_key"]._value is True
        assert result["null_key"]._value is None
        assert isinstance(result["list_key"], Sequence)
        assert isinstance(result["dict_key"], Mapping)

    def test_modification_with_types(self):
        """Test modifying parsed YAML with different types."""
        yaml_str = """
key1: value1
key2: value2
"""
        result = parse(yaml_str)

        # Modify with different types
        result["key1"] = 123
        result["key2"] = ["a", "b", "c"]
        result["key3"] = {"nested": "data"}

        assert result["key1"]._value == 123
        assert isinstance(result["key2"], Sequence)
        assert isinstance(result["key3"], Mapping)

    def test_list_modification_with_types(self):
        """Test modifying list elements with different types."""
        yaml_str = """
items:
  - item1
  - item2
  - item3
"""
        result = parse(yaml_str)

        # Modify with different types
        result["items"][0] = "new string"
        result["items"][1] = 42
        result["items"][2] = ["nested", "list"]

        assert result["items"][0]._value == "new string"
        assert result["items"][1]._value == 42
        assert isinstance(result["items"][2], Sequence)


class TestTypePreservation:
    """Test that Node types are preserved when expected."""

    def test_setitem_preserves_node_objects(self):
        """Test that setting a Node object keeps it as-is."""
        m = Mapping()
        scalar = Scalar(_value="test", _line=10)
        m["key"] = scalar
        # Should be the exact same object
        assert m["key"] is scalar
        assert m["key"]._line == 10

    def test_setitem_preserves_mapping_objects(self):
        """Test that setting a Mapping object keeps it as-is."""
        m = Mapping()
        nested = Mapping({Key("inner"): Scalar("value")})
        nested._line = 20
        m["key"] = nested
        assert m["key"] is nested
        assert m["key"]._line == 20

    def test_setitem_preserves_sequence_objects(self):
        """Test that setting a Sequence object keeps it as-is."""
        m = Mapping()
        seq = Sequence([Scalar("item")])
        seq._line = 30
        m["key"] = seq
        assert m["key"] is seq
        assert m["key"]._line == 30

    def test_append_preserves_node_objects(self):
        """Test that appending a Node object keeps it as-is."""
        s = Sequence()
        scalar = Scalar(_value="test", _line=40)
        s.append(scalar)
        assert s[0] is scalar
        assert s[0]._line == 40


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dict_value(self):
        """Test setting an empty dict."""
        m = Mapping()
        m["empty"] = {}
        assert isinstance(m["empty"], Mapping)
        assert len(m["empty"]) == 0

    def test_empty_list_value(self):
        """Test setting an empty list."""
        m = Mapping()
        m["empty"] = []
        assert isinstance(m["empty"], Sequence)
        assert len(m["empty"]) == 0

    def test_deeply_nested_structure(self):
        """Test deeply nested structures."""
        m = Mapping()
        m["level1"] = {"level2": {"level3": {"level4": {"value": "deep"}}}}
        assert m["level1"]["level2"]["level3"]["level4"]["value"]._value == "deep"

    def test_list_of_lists(self):
        """Test list of lists."""
        s = Sequence()
        s.append([[1, 2], [3, 4]])
        assert isinstance(s[0], Sequence)
        assert isinstance(s[0][0], Sequence)
        assert s[0][0][0]._value == 1

    def test_dict_of_dicts(self):
        """Test dict of dicts."""
        m = Mapping()
        m["outer"] = {"middle": {"inner": "value"}}
        assert isinstance(m["outer"], Mapping)
        assert isinstance(m["outer"]["middle"], Mapping)
        assert m["outer"]["middle"]["inner"]._value == "value"

    def test_mixed_container_nesting(self):
        """Test mixing lists and dicts."""
        m = Mapping()
        m["data"] = [
            {"name": "Alice", "scores": [90, 85, 92]},
            {"name": "Bob", "scores": [88, 91, 87]},
        ]
        assert isinstance(m["data"], Sequence)
        assert isinstance(m["data"][0], Mapping)
        assert isinstance(m["data"][0]["scores"], Sequence)
        assert m["data"][0]["scores"][0]._value == 90
