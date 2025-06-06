from yamlium import Mapping, parse


def _parse(s: str, /) -> Mapping:
    return parse(s)


def _yml(s: str, /) -> str:
    return s.strip() + "\n"


def test_simple_addition():
    s = _yml("""
key1:
  key2: value1
""")
    yml = _parse(s)
    for key, value, obj in yml.walk_keys():
        if key == "key2":
            obj[key] = {"key3": value}
    assert yml._to_yaml() == _yml("""
key1:
  key2:
    key3: value1
""")


def test_nested_mapping_manipulation():
    s = _yml("""
user:
  name: alice
  address:
    street: 123 Main St
    city: Boston
""")
    yml = _parse(s)
    for key, value, obj in yml.walk_keys():
        if key == "street":
            obj[key] = "456 New St"
        elif key == "city":
            obj[key] = "New York"
    assert yml._to_yaml() == _yml("""
user:
  name: alice
  address:
    street: 456 New St
    city: New York
""")


def test_sequence_manipulation():
    s = _yml("""
numbers: [1, 2, 3]
names:
  - alice
  - bob
  - charlie
""")
    yml = _parse(s)
    for key, value, obj in yml.walk_keys():
        if key == "numbers":
            obj[key] = [3, 2, 1]
        elif key == "names":
            obj[key] = [name.str.upper() for name in value]
    assert yml._to_yaml() == _yml("""
numbers:
  - 3
  - 2
  - 1
names:
  - ALICE
  - BOB
  - CHARLIE
""")


def test_mixed_structure_manipulation():
    s = _yml("""
users:
  - name: alice
    age: 25
  - name: bob
    age: 30
settings:
  active: true
  features: [a, b, c]
""")
    yml = _parse(s)
    for key, value, obj in yml.walk_keys():
        if key == "age":
            value += 1
        elif key == "features":
            obj[key] = [f.str.upper() for f in value]
    assert yml._to_yaml() == _yml("""
users:
  - name: alice
    age: 26
  - name: bob
    age: 31
settings:
  active: true
  features:
    - A
    - B
    - C
""")


def test_scalar_type_manipulation():
    s = _yml("""
string: hello world
integer: 42 # comment should stay
float: 3.14 # comment should stay
boolean_true: true # comment should stay
boolean_false: false
null_value: null
quoted: "quoted string"
""")
    yml = _parse(s)
    for key, value, obj in yml.walk_keys():
        if key == "string":
            obj[key] = value.str.upper()  # type: ignore
        elif key == "integer":
            obj[key] = value * 2  # type: ignore
        elif key == "float":
            obj[key] = value * 2  # type: ignore
        elif key == "boolean_true":
            obj[key] = False
        elif key == "boolean_false":
            obj[key] = True
        elif key == "null_value":
            obj[key] = "not null anymore"
    assert yml._to_yaml() == _yml("""
string: HELLO WORLD
integer: 84 # comment should stay
float: 6.28 # comment should stay
boolean_true: false
boolean_false: true
null_value: not null anymore
quoted: "quoted string"
""")


def test_complex_nesting_manipulation():
    s = _yml("""
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
    yml = _parse(s)
    for key, value, obj in yml.walk_keys():
        if key == "final":
            obj[key] = "modified value"
        elif key == "other":
            obj[key] = "new value"
        elif key == "another":
            obj[key] = "changed value"
    assert yml._to_yaml() == _yml("""
level1:
  level2:
    level3:
      level4:
        level5:
          final: modified value
        other: new value
      another: changed value
    more: value
  extra: value
""")
