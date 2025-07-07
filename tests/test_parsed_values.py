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
