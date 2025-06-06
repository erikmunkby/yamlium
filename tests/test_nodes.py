from yamlium import parse
from yamlium.nodes import Alias, Document, Key, Mapping, Scalar, Sequence


def test_node_repr():
    # Test __repr__ for different node types
    scalar = Scalar(_value="test")
    assert repr(scalar) == "Scalar('test')"

    # Test Mapping with different value types
    mapping = Mapping(
        {
            Key("string"): Scalar("value"),
            Key("number"): Scalar(42),
            Key("boolean"): Scalar(True),
            Key("none"): Scalar(None),
        }
    )
    expected_mapping = (
        "Key('string'): Scalar('value')"
        "\nKey('number'): Scalar(42)"
        "\nKey('boolean'): Scalar(True)"
        "\nKey('none'): Scalar(None)"
    )
    assert repr(mapping) == expected_mapping

    # Test nested Mapping
    nested_mapping = Mapping({Key("outer"): Mapping({Key("inner"): Scalar("nested")})})
    expected_nested = "Key('outer'): \n  Key('inner'): Scalar('nested')"
    assert repr(nested_mapping) == expected_nested

    # Test mixed nested structures
    mixed = Mapping(
        {
            Key("list"): Sequence(
                [Scalar("item1"), Mapping({Key("key"): Scalar("value")})]
            ),
            Key("dict"): Mapping(
                {Key("nested_list"): Sequence([Scalar(1), Scalar(2)])}
            ),
        }
    )
    expected_mixed = (
        "Key('list'): "
        "\n  - Scalar('item1')"
        "\n  - Key('key'): Scalar('value')"
        "\nKey('dict'): "
        "\n  Key('nested_list'): "
        "\n    - Scalar(1)"
        "\n    - Scalar(2)"
    )
    assert repr(mixed) == expected_mixed

    # Test Document with multiple items
    doc = Document(
        [
            Mapping({Key("key1"): Scalar("value1")}),
            Mapping({Key("key2"): Scalar("value2")}),
        ]
    )
    expected_doc = "Key('key1'): Scalar('value1')\n\n---\nKey('key2'): Scalar('value2')"
    assert repr(doc) == expected_doc

    # Test Alias
    original = Scalar("test")
    alias = Alias("ref", original)
    assert repr(alias) == "Alias('ref')"

    # Test with comments
    scalar1 = Scalar("value1")
    scalar1.inline_comments = "# inline comment"
    scalar2 = Scalar("value2")
    scalar2.stand_alone_comments = ["# standalone comment"]

    mapping_with_comments = Mapping({Key("key1"): scalar1, Key("key2"): scalar2})
    expected_with_comments = (
        "Key('key1'): Scalar('value1', inline_comment=# inline comment)"
        "\nKey('key2'): Scalar('value2', stand_alone_comments=['# standalone comment'])"
    )
    assert repr(mapping_with_comments) == expected_with_comments

    # Test with anchors
    yml = parse("""
hey: &asdf yo
yo: *asdf
    """)
    expected_with_anchor = (
        "Key('hey', anchor='asdf'): Scalar('yo')\nKey('yo'): Alias('asdf')"
    )
    assert repr(yml) == expected_with_anchor


def test_node_hash():
    # Test __hash__ for different node types
    scalar1 = Scalar(_value="test")
    scalar2 = Scalar(_value="test")
    scalar3 = Scalar(_value="different")

    assert hash(scalar1) == hash(scalar2)
    assert hash(scalar1) != hash(scalar3)


def test_node_addition():
    # Test __add__, __iadd__ for nodes
    scalar = Scalar(_value="hello")

    # Test __add__
    result = scalar + " world"
    assert result._value == "hello world"
    assert scalar._value == "hello"  # Original unchanged

    # Test __iadd__
    scalar += " world"
    assert scalar._value == "hello world"


def test_node_multiplication():
    # Test __mul__, __rmul__, __imul__ for nodes
    scalar = Scalar(_value="test")

    # Test __mul__
    result = scalar * 3
    assert result._value == "testtesttest"
    assert scalar._value == "test"  # Original unchanged

    # Test __rmul__
    result = 3 * scalar
    assert result._value == "testtesttest"

    # Test __imul__
    scalar *= 2
    assert scalar._value == "testtest"


def test_node_equality():
    # Test __eq__, __ne__ for nodes
    scalar1 = Scalar(_value="test")
    scalar2 = Scalar(_value="test")
    scalar3 = Scalar(_value="different")

    assert scalar1 == scalar2
    assert scalar1 == "test"
    assert scalar1 != scalar3
    assert scalar1 != "different"


def test_node_length():
    # Test __len__ for nodes
    scalar = Scalar(_value="test")
    assert len(scalar) == 4


def test_key_operations():
    # Test Key specific operations
    key1 = Key("test")
    key2 = Key("test")
    key3 = Key("different")

    assert key1 == key2
    assert key1 == "test"
    assert key1 != key3
    assert str(key1) == "test"
    assert hash(key1) == hash(key2)


def test_sequence_operations():
    # Test Sequence operations
    seq = Sequence([Scalar("item1"), Scalar("item2")])

    # Test append
    seq.append(Scalar("item3"))
    assert len(seq) == 3
    assert seq[2]._value == "item3"

    # Test __setitem__
    seq[0] = Scalar("new_item")
    assert seq[0]._value == "new_item"

    # Test extend
    seq.extend([Scalar("item4"), Scalar("item5")])
    assert len(seq) == 5
    assert seq[4]._value == "item5"


def test_mapping_operations():
    # Test Mapping operations
    mapping = Mapping({Key("key1"): Scalar("value1")})

    # Test __setitem__
    mapping[Key("key2")] = Scalar("value2")
    assert mapping[Key("key2")]._value == "value2"

    # Test string key
    mapping["key3"] = Scalar("value3")
    assert mapping[Key("key3")]._value == "value3"

    # Test update
    mapping.update({Key("key4"): Scalar("value4")})
    assert mapping[Key("key4")]._value == "value4"


def test_scalar_operations():
    # Test Scalar operations
    scalar = Scalar(_value="test")

    # Test string manipulation
    upper_result = scalar.str.upper()
    lower_result = scalar.str.lower()
    assert isinstance(upper_result, Scalar)
    assert isinstance(lower_result, Scalar)
    assert upper_result._value == "TEST"
    assert lower_result._value == "test"

    # Test to_dict
    assert scalar.to_dict() == "test"


def test_alias_operations():
    # Test Alias operations
    original = Scalar("test")
    alias = Alias("ref", original)

    assert alias._value == "ref"
    assert alias.child == original
    assert alias.to_dict() == "test"


def test_document_operations():
    # Test Document operations
    doc = Document(
        [
            Mapping({Key("key1"): Scalar("value1")}),
            Mapping({Key("key2"): Scalar("value2")}),
        ]
    )

    assert len(doc) == 2
    assert doc[0][Key("key1")]._value == "value1"
    assert doc[1][Key("key2")]._value == "value2"


def test_node_walk():
    # Test walk functionality
    mapping = Mapping(
        {
            Key("key1"): Scalar("value1"),
            Key("key2"): Sequence(
                [Scalar("item1"), Mapping({Key("nested"): Scalar("value")})]
            ),
        }
    )

    # Test walking all nodes
    nodes = list(mapping.walk())
    assert len(nodes) == 5  # key1, value1, key2, sequence, nested mapping

    # Test walking with path
    for key, value, obj in mapping.walk("key2"):
        assert len(value) == 2  # item1 and nested mapping


def test_node_walk_keys():
    # Test walk_keys functionality
    mapping = Mapping(
        {
            Key("key1"): Scalar("value1"),
            Key("key2"): Mapping({Key("nested"): Scalar("value")}),
        }
    )

    # Test walking all keys
    keys = list(mapping.walk_keys())
    assert len(keys) == 3  # key1, key2, nested
