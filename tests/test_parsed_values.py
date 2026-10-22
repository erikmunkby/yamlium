import pytest
import yaml

from yamlium import parse


def test_no_quotes():
    yml = """
key1: 'string1'
key2: "string2"
key3: string3
"""
    y = parse(yml)
    for i in [1, 2, 3]:
        k, s = f"key{i}", f"string{i}"
        assert y[k] == s
        assert y.get(k) == s

        # Also check using the walking functionality
        for key, value, obj in y.walk_keys():
            if key == k:
                assert value == s


def test_quoted_values():
    # Without quotes
    assert parse("key: false")["key"] == False  # noqa: E712
    assert parse("key: 123")["key"] == 123
    # With quotes
    assert parse('key: "false"')["key"] == "false"
    assert parse("key: 'false'")["key"] == "false"
    assert parse("key: '123'")["key"] == "123"


@pytest.mark.parametrize(
    "s",
    [
        'a: "\\u00c5"',
        'a: "\\t \\n \\\\ \\" \\/ \\x41 \\U0001F600 \\N \\_ \\L \\P \\e \\0"',
        "a: 'it''s'",
        '"k\\"ey": 1',
        "'k''ey': 1",
        '"\\u00c5": x',
        # Line folding, escaped line breaks and escaped trailing space
        'a: "foo\n  bar"',
        "a: 'foo\n\n  bar'",
        'a: "foo  \\\n   bar"',
        'a: "foo\\\n\n   bar"',
        'a: "space\\ \n  next"',
        # Whitespace inside quotes is content
        'a: "trailing "',
        "a: ' both '",
    ],
)
def test_quoted_values_match_pyyaml(s: str):
    s += "\n"
    assert parse(s).to_dict() == yaml.safe_load(s)
    assert parse(s).to_yaml() == s


def test_unknown_escape_is_kept_verbatim():
    s = 'a: "\\q"\n'
    assert parse(s).to_dict() == {"a": "\\q"}
    assert parse(s).to_yaml() == s


def test_quoted_values_in_flow_mapping():
    assert parse('{"a": "\\u00c5\\tb", "\\n": 1}').to_dict() == {"a": "Å\tb", "\n": 1}


def test_modified_quoted_values_are_escaped():
    m = parse("a: \"say \\\"hi\\\"\"\nb: 'it''s'\nc: 'x'\n")
    m["a"] = m["a"].str.replace("hi", "bye")
    m["b"] = m["b"].str.replace("it", "that")
    m["c"] = m["c"].str.replace("x", "1\n2")
    out = m.to_yaml()
    assert out == 'a: "say \\"bye\\""\nb: \'that\'\'s\'\nc: "1\\n2"\n'
    assert yaml.safe_load(out) == {"a": 'say "bye"', "b": "that's", "c": "1\n2"}
