__version__ = "0.1.11"
import json
from pathlib import Path

from .exceptions import ParsingError
from .nodes import Alias, Document, Mapping, Scalar, Sequence, _convert_type
from .parser import Parser


def parse_full(input: str | Path) -> Document:
    """Parse a YAML file or string into a Document object.

    This function can handle both YAML files and YAML strings, and supports multiple
    YAML documents in a single file.

    Args:
        input: Either a Path object pointing to a YAML file, or a string containing YAML content.
            If a string ends with .yml or .yaml, it will be treated as a file path.

    Returns:
        Document: A Document object containing all YAML documents from the input.

    Raises:
        ParsingError: If the YAML content cannot be parsed.
    """
    if isinstance(input, Path):
        input = input.read_text()
    elif input.endswith(".yml") or input.endswith(".yaml"):
        input = Path(input).read_text()
    return Parser(input=input).parse()


def parse(input: str | Path) -> Mapping:
    """Parse a YAML file or string into a Mapping object.

    This function is similar to parse_full but expects a single YAML document.
    It's the recommended way to parse YAML files that contain a single document.

    Args:
        input: Either a Path object pointing to a YAML file, or a string containing YAML content.
            If a string ends with .yml or .yaml, it will be treated as a file path.

    Returns:
        Mapping: A Mapping object representing the YAML document.

    Raises:
        ParsingError: If the YAML content cannot be parsed or contains multiple documents.
    """
    documents = parse_full(input=input)
    if len(documents) > 1:
        raise ParsingError(
            f"Your file seems to contain multiple yaml documents. Try `{parse_full.__name__}()`"
        )
    return documents[0]  # type: ignore


def from_json(input: str | Path) -> Mapping | Sequence:
    """Convert JSON content into a Mapping or Sequence object.

    This function can handle both JSON files and JSON strings, converting them into
    the appropriate data structure.

    Args:
        input: Either a Path object pointing to a JSON file, or a string containing JSON content.
            If a string ends with .json, it will be treated as a file path.

    Returns:
        Union[Mapping, Sequence]: A Mapping object for JSON objects, or a Sequence object for JSON arrays.

    Raises:
        json.JSONDecodeError: If the JSON content cannot be parsed.
    """
    if isinstance(input, Path):
        input = input.read_text()
    elif input.endswith(".json"):
        input = Path(input).read_text()
    return _convert_type(json.loads(input))  # type: ignore


def from_dict(input: dict | list) -> Mapping | Sequence:
    """Convert a Python dictionary or list into a Mapping or Sequence object.

    This function provides a convenient way to convert native Python data structures
    into data structures.

    Args:
        input: A Python dictionary or list to convert.

    Returns:
        Union[Mapping, Sequence]: A Mapping object for dictionaries, or a Sequence object for lists.
    """
    return _convert_type(input)  # type: ignore


__all__ = [
    "parse",
    "from_dict",
    "Mapping",
    "Sequence",
    "Scalar",
    "Alias",
    "ParsingError",
]
