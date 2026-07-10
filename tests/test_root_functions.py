import json
from pathlib import Path

import pytest

from yamlium import (
    Mapping,
    ParsingError,
    Sequence,
    from_dict,
    from_json,
    parse,
    parse_full,
    read_markdown,
)


@pytest.fixture
def sample_yaml():
    return """
name: test_project
version: 1.0
models:
  - name: model1
    description: First model
  - name: model2
    description: Second model
    """


@pytest.fixture
def sample_yaml_multiple():
    return """
name: first_doc
---
name: second_doc
    """


@pytest.fixture
def sample_json():
    return {
        "name": "test_project",
        "version": 1.0,
        "models": [
            {"name": "model1", "description": "First model"},
            {"name": "model2", "description": "Second model"},
        ],
    }


def test_parse_string(sample_yaml):
    """Test parsing a YAML string."""
    result = parse(sample_yaml)
    assert isinstance(result, Mapping)
    assert result["name"] == "test_project"
    assert result["version"] == 1.0
    assert isinstance(result["models"], Sequence)
    assert len(result["models"]) == 2


def test_parse_file(tmp_path: Path, sample_yaml):
    """Test parsing a YAML file."""
    yaml_file = tmp_path / "test.yml"
    yaml_file.write_text(sample_yaml)

    result = parse(yaml_file)
    assert isinstance(result, Mapping)
    assert result["name"] == "test_project"


def test_parse_full_multiple_docs(sample_yaml_multiple):
    """Test parsing multiple YAML documents."""
    result = parse_full(sample_yaml_multiple)
    assert len(result) == 2
    assert result[0]["name"] == "first_doc"
    assert result[1]["name"] == "second_doc"


def test_parse_multiple_docs_error(sample_yaml_multiple):
    """Test that parse() raises error for multiple documents."""
    with pytest.raises(ParsingError):
        parse(sample_yaml_multiple)


def test_from_json_string(sample_json):
    """Test converting JSON string to yamlium structure."""
    json_str = json.dumps(sample_json)
    result = from_json(json_str)
    assert isinstance(result, Mapping)
    assert result["name"] == "test_project"
    assert isinstance(result["models"], Sequence)


def test_from_json_file(tmp_path: Path, sample_json):
    """Test converting JSON file to yamlium structure."""
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(sample_json))

    result = from_json(json_file)
    assert isinstance(result, Mapping)
    assert result["name"] == "test_project"


def test_from_dict(sample_json):
    """Test converting Python dict to yamlium structure."""
    result = from_dict(sample_json)
    assert isinstance(result, Mapping)
    assert result["name"] == "test_project"
    assert isinstance(result["models"], Sequence)


def test_from_dict_list():
    """Test converting Python list to yamlium structure."""
    data = [1, 2, 3, {"key": "value"}]
    result = from_dict(data)
    assert isinstance(result, Sequence)
    assert len(result) == 4
    assert isinstance(result[3], Mapping)


def test_from_dict_bare_empty_dict():
    result = from_dict({})
    assert result.to_yaml() == "{}\n"


def test_from_dict_bare_empty_list():
    result = from_dict([])
    assert result.to_yaml() == "[]\n"


def test_from_dict_empty_list():
    result = from_dict({"q": []})
    assert result.to_yaml() == "q: []\n"


def test_from_dict_empty_dict():
    result = from_dict({"q": {}})
    assert result.to_yaml() == "q: {}\n"


def test_from_dict_empty_list_and_dict():
    result = from_dict({"q": [], "w": {}})
    assert result.to_yaml() == "q: []\nw: {}\n"


def test_from_dict_nested_empty():
    result = from_dict({"a": {"b": []}})
    assert result.to_yaml() == "a:\n  b: []\n"


def test_from_dict_setitem_empty_list():
    m = from_dict({"key": "value"})
    m["key"] = []
    assert "key: []\n" in m.to_yaml()


def test_from_dict_empty_nested_in_non_empty():
    result = from_dict({"a": [1, [], 3]})
    assert "[]" in result.to_yaml()


def test_from_dict_non_empty_list_stays_block():
    result = from_dict({"q": [1, 2]})
    assert result.to_yaml() == "q:\n  - 1\n  - 2\n"


def test_from_dict_non_empty_dict_stays_block():
    result = from_dict({"q": {"a": 1}})
    assert "{" not in result.to_yaml()


def test_from_dict_multiline_string():
    result = from_dict({"a": "b\nc"})
    assert result.to_yaml() == "a: |-\n  b\n  c\n"


def test_from_json_multiline_string():
    result = from_json('{"a": "b\\nc"}')
    assert result.to_yaml() == "a: |-\n  b\n  c\n"


def test_from_dict_multiline_with_trailing_newline():
    result = from_dict({"a": "b\nc\n"})
    assert result.to_yaml() == "a: |\n  b\n  c\n"


def test_from_dict_plain_string_stays_scalar():
    result = from_dict({"a": "hello"})
    assert result.to_yaml() == "a: hello\n"


def test_from_dict_empty_string_stays_scalar():
    result = from_dict({"a": ""})
    assert "|" not in result.to_yaml()


def test_from_dict_int_unaffected():
    result = from_dict({"a": 42})
    assert result.to_yaml() == "a: 42\n"


def test_from_dict_bool_unaffected():
    result = from_dict({"a": True})
    assert result.to_yaml() == "a: true\n"


def test_invalid_json():
    """Test parsing invalid JSON."""
    with pytest.raises(json.JSONDecodeError):
        from_json("invalid json")


def test_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        parse("nonexistent.yml")

    with pytest.raises(FileNotFoundError):
        from_json("nonexistent.json")


# --- read_markdown tests ---


def test_read_markdown_standard_frontmatter():
    text = "---\ntitle: Hello\ntags: [a, b]\n---\n# My Post\n\nBody here.\n"
    fm, content = read_markdown(text)
    assert isinstance(fm, Mapping)
    assert fm["title"] == "Hello"
    assert fm["tags"][0] == "a"
    assert content == "# My Post\n\nBody here.\n"


def test_read_markdown_open_format():
    text = "key: value1\nkey2:\n  - item1\n  - item2\n---\nThis is raw content.\n"
    fm, content = read_markdown(text)
    assert isinstance(fm, Mapping)
    assert fm["key"] == "value1"
    assert fm["key2"][0] == "item1"
    assert content == "This is raw content.\n"


def test_read_markdown_no_frontmatter():
    text = "# Just a heading\n\nSome paragraph text.\n"
    fm, content = read_markdown(text)
    assert fm is None
    assert content == text


def test_read_markdown_empty_frontmatter():
    text = "---\n---\nContent after empty frontmatter.\n"
    fm, content = read_markdown(text)
    assert isinstance(fm, Mapping)
    assert len(fm) == 0
    assert content == "Content after empty frontmatter.\n"


def test_read_markdown_frontmatter_with_comments():
    text = "---\n# a yaml comment\ntitle: Hello\n---\nBody.\n"
    fm, content = read_markdown(text)
    assert fm["title"] == "Hello"
    assert content == "Body.\n"


def test_read_markdown_content_with_triple_dashes():
    """Only the first closing --- is used as the frontmatter delimiter."""
    text = "---\ntitle: Hello\n---\nFirst section\n---\nSecond section\n"
    fm, content = read_markdown(text)
    assert fm["title"] == "Hello"
    assert content == "First section\n---\nSecond section\n"


def test_read_markdown_open_format_content_with_triple_dashes():
    text = "key: val\n---\nPart one\n---\nPart two\n"
    fm, content = read_markdown(text)
    assert fm["key"] == "val"
    assert content == "Part one\n---\nPart two\n"


def test_read_markdown_only_opening_separator():
    """A single --- with no content before it and no closing --- means no frontmatter."""
    text = "---\nThis looks like yaml but has no closing separator\n"
    fm, content = read_markdown(text)
    assert fm is None
    assert content == text


def test_read_markdown_file_path(tmp_path: Path):
    md_file = tmp_path / "post.md"
    md_file.write_text("---\ntitle: From File\n---\n# Heading\n")
    fm, content = read_markdown(md_file)
    assert fm["title"] == "From File"
    assert content == "# Heading\n"


def test_read_markdown_string_path(tmp_path: Path):
    md_file = tmp_path / "post.md"
    md_file.write_text("---\ntitle: From String Path\n---\nBody.\n")
    fm, content = read_markdown(str(md_file))
    assert fm["title"] == "From String Path"
    assert content == "Body.\n"


def test_read_markdown_path_object_non_md(tmp_path: Path):
    """Path objects are always read regardless of extension."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("---\nkey: val\n---\nstuff\n")
    fm, content = read_markdown(txt_file)
    assert fm["key"] == "val"
    assert content == "stuff\n"


def test_read_markdown_preserves_yaml_structure():
    text = "---\n# Comment above\nname: test # inline\n---\nBody.\n"
    fm, content = read_markdown(text)
    yaml_out = fm.to_yaml()
    assert "# Comment above" in yaml_out
    assert "# inline" in yaml_out


def test_read_markdown_leading_newline():
    text = "\n---\ntitle: Hello\n---\nBody.\n"
    fm, content = read_markdown(text)
    assert fm["title"] == "Hello"
    assert content == "Body.\n"


def test_read_markdown_no_trailing_content():
    text = "---\ntitle: Hello\n---\n"
    fm, content = read_markdown(text)
    assert fm["title"] == "Hello"
    assert content == ""


def test_read_markdown_empty_string():
    fm, content = read_markdown("")
    assert fm is None
    assert content == ""
