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
