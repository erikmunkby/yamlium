from yamlium import parse_full


def _comp(s: str, d: dict, /) -> None:
    result_d = parse_full(s.strip() + "\n").to_dict()
    assert [d] == result_d


def test_complex_anchor_alias_override():
    _comp(
        """
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
""",
        {
            "base": {"name": "default", "value": 42, "nested": {"key": "value"}},
            "override": {"value": 100, "extra": "field"},
            "merged": {
                "name": "default",
                "value": 100,
                "nested": {"key": "value"},
                "extra": "field",
                "additional": "value",
            },
        },
    )


def test_simple_key_value():
    """Test basic key-value pairs"""
    _comp(
        """
name: bob
age: 25
active: true
""",
        {"name": "bob", "age": 25, "active": True},
    )


def test_nested_mappings():
    """Test nested mapping structures"""
    _comp(
        """
user:
  name: alice
  address:
    street: 123 Main St
    city: Boston
""",
        {
            "user": {
                "name": "alice",
                "address": {"street": "123 Main St", "city": "Boston"},
            }
        },
    )


def test_sequences():
    """Test sequence (list) structures"""
    _comp(
        """
numbers: [1, 2, 3]
names:
  - alice
  - bob
  - charlie
""",
        {"numbers": [1, 2, 3], "names": ["alice", "bob", "charlie"]},
    )


def test_mixed_structures():
    """Test combinations of mappings and sequences"""
    _comp(
        """
users:
  - name: alice
    age: 25
  - name: bob
    age: 30
settings:
  active: true
  features: [a, b, c]
""",
        {
            "users": [
                {"name": "alice", "age": 25},
                {"name": "bob", "age": 30},
            ],
            "settings": {"active": True, "features": ["a", "b", "c"]},
        },
    )


def test_scalar_types():
    """Test all scalar types"""
    _comp(
        """
string: hello world
integer: 42
float: 3.14
boolean_true: true
boolean_false: false
null_value: null
quoted: "quoted string"
""",
        {
            "string": "hello world",
            "integer": 42,
            "float": 3.14,
            "boolean_true": True,
            "boolean_false": False,
            "null_value": None,
            "quoted": "quoted string",
        },
    )


def test_multiline_strings():
    """Test multiline string handling"""
    _comp(
        """
multiline: |
  This is a
  multiline string
  with multiple lines

folded: >
  This is a folded
  string that will
  be joined with spaces
""",
        {
            "multiline": "This is a\nmultiline string\nwith multiple lines",
            "folded": "This is a folded string that will be joined with spaces",
        },
    )


def test_flow_style():
    """Test flow style (inline) collections"""
    _comp(
        """
flow_map: { a: 1, b: 2, c: 3 }
flow_seq: [1, 2, 3, 4]
mixed: { a: [1, 2], b: { x: 1, y: 2 } }
""",
        {
            "flow_map": {"a": 1, "b": 2, "c": 3},
            "flow_seq": [1, 2, 3, 4],
            "mixed": {"a": [1, 2], "b": {"x": 1, "y": 2}},
        },
    )


def test_empty_structures():
    """Test empty mappings and sequences"""
    _comp(
        """
empty_map: {}
empty_seq: []
nested_empty:
  map: {}
  seq: []
""",
        {
            "empty_map": {},
            "empty_seq": [],
            "nested_empty": {"map": {}, "seq": []},
        },
    )


def test_complex_nesting():
    """Test deeply nested structures"""
    _comp(
        """
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
""",
        {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"final": "value"},
                            "other": "value",
                        },
                        "another": "value",
                    },
                    "more": "value",
                },
                "extra": "value",
            }
        },
    )


def test_sequence_of_mappings():
    """Test sequences containing mappings"""
    _comp(
        """
items:
  - name: item1
    value: 1
  - name: item2
    value: 2
  - name: item3
    value: 3
""",
        {
            "items": [
                {"name": "item1", "value": 1},
                {"name": "item2", "value": 2},
                {"name": "item3", "value": 3},
            ]
        },
    )


def test_special_characters():
    """Test handling of special characters in keys and values"""
    _comp(
        """
special_chars: "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
unicode: "你好世界"
""",
        {
            "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?/~`",
            "unicode": "你好世界",
        },
    )


def test_anchor_alias():
    """Test anchor and alias functionality"""
    _comp(
        """
base: &base
  name: default
  value: 42

derived1: *base
derived2: *base
""",
        {
            "base": {"name": "default", "value": 42},
            "derived1": {"name": "default", "value": 42},
            "derived2": {"name": "default", "value": 42},
        },
    )


def test_merge_keys():
    """Test YAML merge keys"""
    _comp(
        """
base: &base
  name: default
  value: 42

override: &override
  value: 100

merged:
  <<: [*base, *override]
  extra: value
""",
        {
            "base": {"name": "default", "value": 42},
            "override": {"value": 100},
            "merged": {
                "name": "default",
                "value": 100,
                "extra": "value",
            },
        },
    )


def test_complex_anchor_chains():
    """Test complex anchor and alias relationships"""
    _comp(
        """
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
""",
        {
            "base": {"name": "default", "value": 42},
            "derived1": {"name": "default", "value": 42, "extra": "value1"},
            "derived2": {
                "name": "default",
                "value": 42,
                "extra": "value1",
                "more": "value2",
            },
            "final": {
                "name": "default",
                "value": 42,
                "extra": "value1",
                "more": "value2",
                "final": "value3",
            },
        },
    )
