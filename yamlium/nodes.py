from __future__ import annotations

import pprint
from abc import abstractmethod
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterator, Literal, cast

from .lexer import T


def _indent(i: int, /) -> str:
    return "  " * i


def _convert_type(obj: dict | list | str, /) -> Node:
    if hasattr(obj, "_managed"):
        return obj  # type: ignore
    if isinstance(obj, dict):
        return Mapping(
            {Key(_value=key): _convert_type(value) for key, value in obj.items()}
        )
    if isinstance(obj, list):
        return Sequence([_convert_type(item) for item in obj])
    return Scalar(_value=obj)


class StrManipulator:
    """This class allows string manipulation."""

    def __init__(self, parent: Node):
        self.parent = copy(parent)

    def __getattr__(self, attr):
        def method(*args, **kwargs):
            result = getattr(self.parent._value, attr)(*args, **kwargs)
            self.parent._value = result
            return self.parent

        return method


class Node:
    _managed: bool = True

    def __init__(
        self,
        _value: str | int | float | bool | None = None,
        _line: int = -99,
        _indent: int = -99,
    ) -> None:
        self._value = _value
        self._line = _line
        self._indent = _indent
        self._column: int = -99
        self.newlines: int = 0
        self.stand_alone_comments: list[str] = []
        self.inline_comments: str | None = None

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[Node]: ...
        def __getitem__(self, key: Key | str | int) -> Node: ...
        def get(self, key: Key | str, default: Any = ...) -> Node: ...
        def items(self) -> Iterator[tuple[Key, Node]]: ...
        def keys(self) -> Iterator[Key]: ...
        def values(self) -> Iterator[Node]: ...
        def pop(self, key: Key | str) -> Node: ...

    def __repr__(self) -> str:
        return self._ast_repr()

    def __hash__(self) -> int:
        return hash(self._value)

    def __iadd__(self, other) -> Node:
        self._value += other
        return self

    def __add__(self, other) -> Node:
        s = copy(self)
        s._value += other
        return s

    def __mul__(self, other) -> Node:
        s = copy(self)
        s._value *= other
        return s

    def __rmul__(self, other) -> Node:
        s = copy(self)
        s._value = other * s._value
        return s

    def __imul__(self, other) -> Node:
        self._value *= other
        return self

    def __eq__(self, other) -> bool:
        return self._value == other

    def __ne__(self, other) -> bool:
        return not self.__eq__(other=other)

    def __len__(self) -> int:
        return len(self._value)  # type: ignore

    def _ast_info(self) -> str:
        info = [
            f"{k}={v}"
            for k, v in {
                "newlines": self.newlines,
                "inline_comment": self.inline_comments,
                "stand_alone_comments": self.stand_alone_comments,
            }.items()
            if v
        ]
        anchor = getattr(self, "anchor", None)
        if anchor:
            info += [f"anchor='{anchor}'"]
        return ", ".join(info)

    def _ast_repr(
        self,
        ind: int = 0,
        parent: Sequence | Mapping | None = None,
    ) -> str:
        i_, nl = _indent(ind), "\n"
        info = self._ast_info()
        info = ", " + info if info else ""
        if isinstance(self, Key):
            return f"Key('{self._value}'{info})"
        if isinstance(self, Scalar):
            if isinstance(self._value, str):
                return f"Scalar('{self._value}'{info})"
            else:
                return f"Scalar({self._value}{info})"
        if isinstance(self, Mapping):
            if isinstance(parent, Sequence) or parent is None:
                nl, i_ = "", ""
            return nl + "\n".join(
                [
                    f"{i_}{k._ast_repr()}: {v._ast_repr(parent=self, ind=ind + 1)}"
                    for k, v in self.items()
                ]
            )
        # Document must be checked before Sequence since it inherits Sequence.
        if isinstance(self, Document):
            return "\n\n---\n".join([x._ast_repr() for x in self])
        if isinstance(self, Sequence):
            return "\n" + "\n".join(
                [i_ + "- " + x._ast_repr(parent=self, ind=ind + 1) for x in self]
            )
        if isinstance(self, Alias):
            return f"Alias('{self._value}')"
        raise ValueError(f"{type(self)} not supported.")

    def _get_sa_comments(self, i: int) -> list[str]:
        return [_indent(i) + c for c in self.stand_alone_comments]

    def _enrich_yaml(self, s: str, /) -> str:
        output = s
        if self.inline_comments:
            output += f" {self.inline_comments}"
        return output + "\n" * self.newlines

    @abstractmethod
    def _to_yaml(self, i: int = 0) -> str:
        pass

    @property
    def str(self) -> str:
        """Get a string manipulator for this node's value.

        Returns:
            A StrManipulator instance for string operations.

        Raises:
            TypeError: If the node's value is not a string.
        """
        if not isinstance(self._value, str):
            raise TypeError(
                f"Cannot apply string utilities to type '{type(self._value).__name__}'"
            )
        return cast(str, StrManipulator(self))

    def pprint(self) -> None:
        """Pretty print the node's value as a dictionary."""
        pprint.pprint(self.to_dict())

    def to_yaml(self) -> str:
        """Convert the node to YAML format.

        Returns:
            A string containing the YAML representation of this node.
        """
        return self._to_yaml()

    def yaml_dump(self, destination: str | Path) -> None:
        """Write the node's YAML representation to a file.

        Args:
            destination: The path where to write the YAML file.
        """
        if isinstance(destination, str):
            destination = Path(destination)
        destination.write_text(self.to_yaml())

    def to_dict(self) -> list | dict | int | float | bool | str | None:
        """Convert the node to a Python dictionary/list representation.

        Returns:
            A Python object (dict, list, or primitive type) representing this node.
        """
        if isinstance(self, Sequence):
            return [x.to_dict() for x in self]
        if isinstance(self, Mapping):
            val = {}
            for k, v in self.items():
                if k._is_merge_key:
                    if isinstance(v, Sequence):
                        for obj in v:
                            val.update(obj.to_dict())  # type: ignore
                    else:
                        val.update(v.to_dict())  # type: ignore
                else:
                    val[k._value] = v.to_dict()
            return val
        if isinstance(self, Alias):
            return self.child.to_dict()

    def walk(
        self,
        path: list[str] | str | None = None,
    ) -> Generator[tuple[Key | int, Node, Mapping | Sequence]]:
        """Walk through the node tree, yielding (key/index, value, container) tuples.

        This allows for modifying values while walking through the tree.

        Args:
            path: Optional path to filter the walk. Can be a string with '/' separators
                 or a list of path components. Use '*' for any key and '**' for recursive
                 matching.

        Yields:
            Tuples of (key/index, value, container) where:
            - key/index is the key or index of the current node
            - value is the current node
            - container is the parent Mapping or Sequence containing the current node

        Example:
            ```py
            for key, value, container in obj.walk():
                if key == "some_key":
                    container[key] = "new_value"
            ```

            Or with path:
            ```py
            for key, value, container in obj.walk(path="**/key1/*/key2"):
                if key == "some_key":
                    container[key] = "new_value"
            ```
        """
        if path is not None:
            split_path = path.split("/") if isinstance(path, str) else path
            p = split_path[0] if split_path else "STOP"
        else:
            p = None
        if isinstance(self, Mapping):
            for key, value in self.items():
                if not p or p in {"*", "**", key}:
                    yield key, value, self
                    if isinstance(value, (Mapping, Sequence)):
                        next_path = (
                            None
                            if path is None
                            else split_path[1:]
                            if p == key or p == "*"
                            else split_path[2:]
                            if p == "**"
                            and len(split_path) > 1
                            and split_path[1] == key
                            else split_path
                        )
                        yield from value.walk(path=next_path)

        elif isinstance(self, Sequence):
            if not p or p == "*":
                for idx, item in enumerate(self):
                    yield idx, item, self
                    if isinstance(item, (Mapping, Sequence)):
                        yield from item.walk(path=path)

    def walk_keys(
        self, path: list[str] | str | None = None
    ) -> Generator[tuple[Key, Node, Mapping]]:
        """Yields (key, value, container) so that values can be modified.

        Args:
            path: Optional path to filter the walk. Can be a string with '/' separators
                 or a list of path components. Use '*' for any key and '**' for recursive
                 matching.

        Yields:
            Tuples of (key/index, value, container) where:
            - key/index is the key or index of the current node
            - value is the current node
            - container is the parent Mapping or Sequence containing the current node

        Example:
            ```py
            for key, value, container in obj.walk_keys():
                if key == "some_key":
                    container[key] = "new_value"
            ```

            Or with path:
            ```py
            for key, value, container in obj.walk_keys(path="**/key1/*/key2"):
                if key == "some_key":
                    container[key] = "new_value"
            ```
        """
        for key, value, container in self.walk(path=path):
            if isinstance(key, Key):
                yield key, value, container  # type: ignore

    # def get_first(self, key: str) -> Node:
    #     for k, value, _ in self.walk_keys():
    #         if k == key:
    #             return value
    #     raise ValueError(f"Key '{key}' not found.")


class Key(Node):
    def __init__(
        self,
        _value: str,
        _line: int = -99,
        _indent: int = -99,
        _is_merge_key: bool = False,
    ) -> None:
        """Initialize a Key node.

        Args:
            _value: The string value of the key.
            _line: The line number in the source YAML file.
            _indent: The indentation level in the source YAML file.
            _is_merge_key: Whether this is a merge key (<<).
        """
        super().__init__(_value, _line, _indent)
        self._is_merge_key = _is_merge_key
        self.anchor: str | None = None

    def __str__(self) -> str:
        return str(self._value)

    def __eq__(self, other: Key | str) -> bool:
        return self._value == (other if isinstance(other, str) else other._value)

    def __hash__(self):
        return hash(self._value)

    def _to_yaml(self) -> str:
        anch = f" &{self.anchor}" if self.anchor else ""
        return self._enrich_yaml(self._value + ":" + anch)  # type: ignore


class Sequence(list, Node):
    def __init__(
        self, iterable: list[Node] = [], /, *, _line: int = -1, _is_inline: bool = False
    ) -> None:
        """Initialize a Sequence node.

        Args:
            iterable: List of nodes to initialize the sequence with.
            _line: The line number in the source YAML file.
            _is_inline: Whether this sequence should be rendered in inline format.
        """
        self._is_inline = _is_inline
        super().__init__(iterable)
        Node.__init__(self=self, _line=_line)

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return self._ast_repr()

    def _to_yaml(self, i: int = 0) -> str:
        if self._is_inline:
            vals = ", ".join([v._to_yaml() for v in self])
            return self._enrich_yaml("[" + vals + "]")
        items = []
        prefix = _indent(i) + "- "
        for x in self:
            # Check if we should print standalone comments
            items.extend(x._get_sa_comments(i=i))

            # If child is a mapping
            if isinstance(x, Mapping):
                prefix2 = prefix
                for k, v in x.items():
                    items.extend(k._get_sa_comments(i=i))

                    if isinstance(v, (Mapping, Sequence)):
                        # If it is another block, add newline
                        items.append(f"{prefix2}{k._to_yaml()}\n{v._to_yaml(i + 2)}")
                    else:
                        items.append(f"{prefix2}{k._to_yaml()} {v._to_yaml(i + 2)}")
                    prefix2 = prefix2[:-2] + "  "  # Clean up prefix if deeper mapping.

            else:
                items.append(f"{prefix}{x._to_yaml(i + 1)}")
        return "\n".join(items)

    def append(self, item: Any) -> None:
        """Append an item to the sequence.

        Args:
            item: The item to append. Will be converted to a Node if it isn't one.
        """
        super().append(_convert_type(item))

    def __setitem__(self, i: int, value: Any) -> None:
        super().__setitem__(i, _convert_type(value))

    def extend(self, items: list) -> None:
        """Extend the sequence with multiple items.

        Args:
            items: List of items to add. Each item will be converted to a Node if it isn't one.
        """
        super().extend([_convert_type(x) for x in items])

    if TYPE_CHECKING:

        def __getitem__(self, i: int) -> Node: ...
        def __iter__(self) -> Iterator[Node]: ...


class Mapping(dict, Node):
    def __init__(
        self, map: dict[Key, Node] = {}, /, *, _line: int = -1, _is_inline: bool = False
    ) -> None:
        self._is_inline = _is_inline
        super().__init__(map)
        Node.__init__(self=self, _line=_line)

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return self._ast_repr()

    def _to_yaml(self, i: int = 0) -> str:
        # If we have an inline (flow-style) mapping
        if self._is_inline:
            res = ", ".join([f"{k._to_yaml()} {v._to_yaml()}" for k, v in self.items()])
            if res:
                res = "{ " + res + " }"
            else:
                res = "{}"
            return self._enrich_yaml(res)
        items = []
        _i = _indent(i)
        for k, v in self.items():
            items.extend(k._get_sa_comments(i=i))
            if isinstance(v, (Mapping, Sequence)):
                if v._is_inline:
                    items.append(f"{_i}{k._to_yaml()} {v._to_yaml()}")
                else:
                    items.append(f"{_i}{k._to_yaml()}\n{v._to_yaml(i + 1)}")
            elif isinstance(v, Scalar):
                if v._value is None and v._original_value == "":
                    val = ""
                else:
                    val = (
                        f"\n{_indent(i + 1)}" if v._is_indented else " "
                    ) + v._to_yaml(i + 1)
                items.append(f"{_i}{k._to_yaml()}{val}")
            else:
                items.append(f"{_i}{k._to_yaml()} {v._to_yaml()}")
        return "\n".join(items) + ("\n" if self._indent == 0 else "")

    def __setitem__(self, key: Key | str, value: Any) -> None:
        """Set a key-value pair in the mapping.

        Args:
            key: The key to set. Can be a Key object or string.
            value: The value to set. Will be converted to a Node if it isn't one.
        """
        if isinstance(key, str):
            key = Key(_value=key)
        super().__setitem__(key, _convert_type(value))

    def update(self, other: dict) -> None:
        """Update the mapping with key-value pairs from another dictionary.

        Args:
            other: Dictionary containing key-value pairs to add or update.
        """
        super().update(
            {
                Key(_value=k) if isinstance(k, str) else k: _convert_type(v)
                for k, v in other.items()
            }
        )

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[Node]: ...
        def __getitem__(self, key: Key | str | int) -> Node: ...
        def get(self, key: Key | str, default: Any = ...) -> Node: ...
        def items(self) -> Iterator[tuple[Key, Node]]: ...
        def keys(self) -> Iterator[Key]: ...
        def values(self) -> Iterator[Node]: ...
        def pop(self, key: Key | str) -> Node: ...


class Scalar(Node):
    def __init__(
        self,
        _value: str | int | float | bool | None = None,
        _line: int = -99,
        _indent: int = -99,
        _type: Literal[T.MULTILINE_ARROW, T.MULTILINE_PIPE, T.SCALAR] = T.SCALAR,
        _is_indented: bool = False,
        _original_value: str = "",
        _quote_char: str | None = None,
    ) -> None:
        super().__init__(_value, _line, _indent)
        self._type = _type
        self._is_indented = _is_indented
        # Original value is used for the various null representations:
        # null, ~, empty space
        self._original_value = _original_value
        self._quote_char = _quote_char

    def __str__(self) -> str:
        return str(self._value)

    def to_dict(self) -> str | int | float | bool | None:
        """Convert the scalar to its Python value.

        Returns:
            The Python value of this scalar.
        """
        if self._type == T.SCALAR:
            return self._value
        if self._type == T.MULTILINE_PIPE:
            return self._value
        # Otherwise we have an arrow multiline, i.e. ignoring newlines.
        return self._value.replace("\n", " ")  # type: ignore

    def _to_yaml(self, i: int = 0) -> str:
        if self._type == T.SCALAR:
            if isinstance(self._value, bool):
                val = "true" if self._value else "false"
            elif self._value is None:
                val = self._original_value
            elif self._quote_char:
                val = f"{self._quote_char}{self._value}{self._quote_char}"
            else:
                val = str(self._value)
        else:
            val = "|" if self._type == T.MULTILINE_PIPE else ">"
            if self._value:
                i_ = _indent(i)
                val = (
                    val
                    + "\n"
                    + "\n".join(
                        [
                            (i_ + r) if r else ""
                            for r in self._value.split("\n")  # type: ignore
                        ]
                    )
                )
        return self._enrich_yaml(val)

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[Node]: ...
        def __getitem__(self, key: Key | str | int) -> Node: ...


class Alias(Node):
    def __init__(
        self, _value: str, child: Node, _line: int = -99, _indent: int = -99
    ) -> None:
        self.child = child
        super().__init__(_value, _line, _indent)

    def __str__(self) -> str:
        return str(self.child.to_dict())

    def _to_yaml(self, i: int = 0, x: bool = False) -> str:
        return self._enrich_yaml(f"*{self._value}")


class Document(Sequence):
    def _to_yaml(self) -> str:
        items = []
        for x in self:
            items.extend(x.stand_alone_comments)
            items.append(x._to_yaml())
        result = "\n\n---\n".join(items)
        if not result[-1] == "\n":
            # TODO: Add conditional setting here for pyproject.toml
            # To be able to turn off finish with newline
            result += "\n"
        return result
