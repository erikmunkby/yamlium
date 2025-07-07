import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import NoReturn

from .exceptions import raise_parsing_error


class T(Enum):
    """Types of tokens that can be found in YAML."""

    # Structural tokens
    DOCUMENT_START = auto()  # ---
    DOCUMENT_END = auto()  # ...
    MAPPING_START = auto()  # {
    MAPPING_END = auto()  # }
    SEQUENCE_START = auto()  # [
    SEQUENCE_END = auto()  # ]

    # Scalar tokens
    SCALAR = auto()  # Scalar aka value
    MULTILINE_ARROW = auto()  # > multiline
    MULTILINE_PIPE = auto()  # | multiline

    # Special tokens
    DASH = auto()
    INDENT = auto()  # Indentation
    DEDENT = auto()  # Dedentation
    EMPTY_LINE = auto()  # Empty line
    KEY = auto()  # Key in key-value pair
    ANCHOR = auto()  # &anchor
    ALIAS = auto()  # *alias
    COMMA = auto()  # Comma separator

    # Other
    EOF = auto()  # End of file
    COMMENT = auto()  # Comment token #

    NO_TOKEN = auto()


@dataclass
class Token:
    t: T
    value: str
    line: int
    column: int
    start: int
    end: int
    quote_char: str | None = None


@dataclass
class Snapshot:
    position: int
    column: int
    line: int


class Lexer:
    def __init__(self, input: str, /):
        self.input = input
        self.input_length = len(input)
        # These variables are used in token building

    def token_names(self) -> list[str]:
        return [t.t.name for t in self.build_tokens()]

    def print_tokens(self) -> None:
        print(", ".join(self.token_names()))

    def build_tokens(self) -> list[Token]:
        # Pause current position
        self.position = 0
        self.line = 0
        self.column = 0
        self.indent_stack: deque[int] = deque([0])
        self.tokens: list[Token] = []
        eof = False
        while not eof:
            for t in self._parse_next_token():
                self.tokens.append(t)
                if t.t == T.INDENT:
                    self.indent_stack.append(t.column)
                if t.t == T.EOF:
                    eof = True
        return self.tokens

    def _raise_error(self, msg: str, pos: int | None = None) -> NoReturn:
        pos = pos if pos is not None else self.position
        raise_parsing_error(input_str=self.input, pos=pos, msg=msg)

    def _print_line_pos(self) -> None:
        time.sleep(0.1)
        print(self.input.split("\n")[self.line] + ";")
        print(" " * (self.column - 1), "^" + f"-({self.c}, column={self.column})")

    @property
    def c(self) -> str:
        """Get current character."""
        return self.input[self.position]

    @property
    def c_future(self) -> str | None:
        """Get character one position in the future."""
        if self.position + 1 >= self.input_length:
            return None
        return self.input[self.position + 1]

    def _parse_next_token(self, extra_stop_chars: set = set()) -> list[Token]:
        """Get the next token from the input."""
        # Check if we're at the end of document.
        if self.position > self.input_length:
            raise IndexError("Passed end of input.")

        # End of file.
        if self.position >= self.input_length:
            self._nc()
            return self._build_token(t=T.EOF, value="")

        # Find out which type of character is next.
        char = self.c

        # IMPORTANT quote parsing should happen as early as possible.
        # Incase quoted strings contain special characters.
        if char in ['"', "'"]:
            return self._parse_quoted_scalar()
        if char == "\n":
            return self._parse_dents()
        if char == " ":
            # Skip normal newlines
            self._nc()
            return self._parse_next_token(extra_stop_chars=extra_stop_chars)
        if char == "-":
            return self._parse_dash()
        if char == "#":
            return self._parse_comment()
        if char == "&":
            return self._parse_anchor(extra_stop_chars=extra_stop_chars)
        if char == "*":
            return self._parse_alias(extra_stop_chars=extra_stop_chars)
        if char == "<":
            return self._parse_merge_key()
        if char in [">", "|"]:
            return self._parse_multiline_scalar()
        if char == "{":
            return self._parse_flow_style(mapping=True)
        if char == "[":
            return self._parse_flow_style(mapping=False)
        if char == "}":
            t = self._build_token(t=T.MAPPING_END, value="}")
            self._nc()
            return t
        if char == "]":
            t = self._build_token(t=T.SEQUENCE_END, value="]")
            self._nc()
            return t
        if char == ",":
            self._nc()
            return self._build_token(t=T.COMMA, value=",")

        # If nothing else, expect value token
        return self._parse_scalar(extra_stop_chars=extra_stop_chars)

    def _parse_flow_style(self, mapping: bool) -> list[Token]:
        s = self._snapshot
        if mapping:
            tokens = self._build_token(t=T.MAPPING_START, value="{")
            extra_scalar_stops = {",", "}"}
            stop_token_type = T.MAPPING_END
            start_token_type = T.MAPPING_START
        else:
            tokens = self._build_token(t=T.SEQUENCE_START, value="[")
            extra_scalar_stops = {",", "]"}
            stop_token_type = T.SEQUENCE_END
            start_token_type = T.SEQUENCE_START
        self._nc()

        stop = False
        inner = 0  # Sometimes we have flow within flow like: [a, [b, c]]
        while not stop:
            for t in self._parse_next_token(extra_stop_chars=extra_scalar_stops):
                tokens.append(t)
                if t.t == start_token_type:
                    inner += 1
                if t.t == stop_token_type:
                    inner -= 1
                    if inner < 0:
                        stop = True
                if t.t == T.EOF or self.position >= self.input_length:
                    flow_type = "mapping" if mapping else "sequence"
                    self._raise_error(f"Inline {flow_type} not closed.", pos=s.position)
        return tokens

    def _parse_scalar(self, extra_stop_chars: set = set()) -> list[Token]:
        s = self._snapshot

        # We might not be dealing with a scalar, but rather key, anchor, alias, comment etc.
        stop_characters = {"&"}.union(extra_stop_chars)
        while self.position < self.input_length:
            char = self.c
            if char == "\n":
                # Stop and expect multiline scalar.
                break
            if char == ":":
                # We might have a key
                if self.c_future not in {" ", "\n"}:
                    self._nc()
                    continue
                self._nc()
                value = self.input[s.position : self.position - 1]
                if " " in value:
                    self._raise_error("Unquoted key cannot contain blankspace(s).")
                return self._build_token(
                    t=T.KEY,
                    value=value,
                    s=s,
                )
            if char == "#" and self.c_future == " " or char in stop_characters:
                return self._build_token(
                    t=T.SCALAR,
                    value=self.input[s.position : self.position],
                    s=s,
                )
            self._nc()

        if self.position == self.input_length:
            # This scenario happens when we do not end with a newline
            # And final character symbol is a normal scalar
            return self._build_token(
                t=T.SCALAR, value=self.input[s.position : self.position], s=s
            )

        token_stack = []
        end_position = self._snapshot.position
        stop = False
        while not stop:
            new_tokens = self._parse_next_token(extra_stop_chars=extra_stop_chars)
            for i, next_token in enumerate(new_tokens):
                if next_token.t == T.SCALAR:
                    end_position = next_token.end
                    token_stack = []
                    continue
                elif next_token.t in {T.INDENT, T.DEDENT, T.EMPTY_LINE}:
                    token_stack.append(next_token)
                else:
                    token_stack.extend(new_tokens[i:])
                    stop = True
                    break

        return [
            *self._build_token(
                t=T.SCALAR, value=self.input[s.position : end_position], s=s
            ),
            *token_stack,
        ]

    def _parse_multiline_scalar(self) -> list[Token]:
        s = self._snapshot
        multiline_type = T.MULTILINE_PIPE if self.c == "|" else T.MULTILINE_ARROW

        # TODO: Add functionality for newline preserve/chomp: |- |+ >- >+

        post_multiline_newlines = 0
        indent = 0
        multiline_indentation_level = None
        while self.position < self.input_length:
            if self.c == "\n":
                # This might be a post multiline newline, e.g.
                # key: |
                #   line1
                #
                # key2: value
                post_multiline_newlines += 1
                self._nl()  # Skip the newline
                indent = self._count_spaces()
                if indent == 0:
                    if self.c == "\n":
                        # We have a simple empty line
                        continue
                    else:
                        break
                if indent == -1 or indent <= self.indent_stack[-1]:
                    break
                # Once we found the first multiline indentation level, register it
                if not multiline_indentation_level:
                    multiline_indentation_level = indent
                # Otherwise compare against the multiline indentation level
                else:
                    if indent > multiline_indentation_level:
                        self._raise_error(msg="Irregular multiline string indentation.")
                    if indent < multiline_indentation_level:
                        break
            else:
                post_multiline_newlines = 0  # Reset since we found more content
                self._nc()

        value = self.input[s.position : self.position]
        split = value.strip().split("\n")
        value = "\n".join([x.strip() for x in split[1:]])

        tokens = self._build_token(t=multiline_type, value=value, s=s)

        for _ in range(post_multiline_newlines - 1):
            tokens.extend(self._build_token(t=T.EMPTY_LINE, value=""))
        if indent != -1 and indent < self.indent_stack[-1]:
            # If the most recent indent we fetched is less than indent stack
            # Then add as a dedent.
            tokens.extend(self._build_dedents(indent=indent))
        return tokens

    def _parse_merge_key(self) -> list[Token]:
        s = self._snapshot

        # After the initial `<` we expect the sequence below.
        # Making the full sequence: <<: *
        for c in "<<:":
            if self.c != c:
                # If the sequence was not followed, it was not a merge key after all
                # Reset position and parse as scalar
                self._reset_pos(s=s)
                return self._parse_scalar()

            self._nc()
        return self._build_token(t=T.KEY, value="<<", s=s)

    def _parse_comment(self) -> list[Token]:
        s = self._snapshot
        # Skip the hashtag
        self._nc()
        char = self.c
        while char != "\n":
            self._nc()
            if self.position >= self.input_length:
                break
            char = self.c
        return self._build_token(
            t=T.COMMENT,
            value=self.input[s.position : self.position],
            s=s,
        )

    def _parse_quoted_scalar(self) -> list[Token]:
        start = self._snapshot
        quote_char = self.c
        self._nc()
        char = self.c
        while char != quote_char:
            self._nc()
            if self.position >= self.input_length:
                self._raise_error(msg="Expected end of quote.", pos=start.position)
            char = self.c
            if char == "\n":
                self._raise_error(
                    msg="Quoted string broken by newline.", pos=start.position
                )
        self._nc()  # Consume the final quote
        # It might be a quoted key
        if self.c == ":":
            self._nc()
            return self._build_token(
                t=T.KEY, value=self.input[start.position : self.position - 1], s=start
            )

        return self._build_token(
            t=T.SCALAR,
            value=self.input[start.position + 1 : self.position - 1],
            s=start,
            quote_char=quote_char,
        )

    def _anchor_or_alias_name(self, extra_stop_chars: set) -> str:
        # Skip to next charcter
        start = self._snapshot
        char = self.c
        stop_chars = {" ", "\n"}.union(extra_stop_chars)
        while char not in stop_chars:
            self._nc()
            if self.position >= self.input_length:
                # If the final token is an alias, we might reach EOF here.
                break

            char = self.c
        return self.input[start.position + 1 : self.position]

    def _parse_alias(self, extra_stop_chars: set) -> list[Token]:
        s = self._snapshot
        return self._build_token(
            t=T.ALIAS,
            value=self._anchor_or_alias_name(extra_stop_chars=extra_stop_chars),
            s=s,
        )

    def _parse_anchor(self, extra_stop_chars: set) -> list[Token]:
        s = self._snapshot
        return self._build_token(
            t=T.ANCHOR,
            value=self._anchor_or_alias_name(extra_stop_chars=extra_stop_chars),
            s=s,
        )

    def _parse_dash(self) -> list[Token]:
        # If the token after is not a dash or blankspace then we're dealing with a scalar
        if not self.c_future or self.c_future not in {" ", "-"}:
            return self._parse_scalar()
        self._nc()
        s = self._snapshot
        # Check if next character is also a dash, i.e. document separator
        if self.c == "-":
            # Take next token as well, and check once more
            self._nc()
            if self.c == "-":
                self._nc()
                return self._build_token(t=T.DOCUMENT_START, value="---", s=s)
            else:
                self._raise_error(f"Expected separator `---` but found `--{self.c}`")

        # Should always be blankspace after dash if sequence
        if self.c != " ":
            self._raise_error(
                f"Expected blankspace after dash but found `{self.c}`", pos=s.position
            )
        self._nc()

        return [
            *self._build_token(t=T.DASH, value="-", s=s),
            *self._maybe_add_dents(indent=self.column),
        ]

    def _check_eof(self) -> bool:
        return self.position == self.input_length

    def _count_spaces(self) -> int:
        spaces = 0
        if self._check_eof():
            return -1
        while self.c == " ":
            self._nc()
            if self._check_eof():
                return -1
            spaces += 1
        return spaces

    def _build_dedents(self, indent: int) -> list[Token]:
        dedents = []
        while indent < self.indent_stack[-1]:
            dedents.extend(self._build_token(t=T.DEDENT, value=""))
            self.indent_stack.pop()
        return dedents

    def _maybe_add_dents(self, indent: int) -> list[Token]:
        if indent == -1:
            return self._build_token(t=T.EOF, value="")
        if indent > self.indent_stack[-1]:
            return self._build_token(t=T.INDENT, value="")
        elif indent < self.indent_stack[-1]:
            # Add potential dedents
            return self._build_dedents(indent=indent)
        return []

    def _parse_dents(self) -> list[Token]:
        s = self._snapshot
        self._nl()
        if self._check_eof():
            return self._build_token(t=T.EOF, value="")
        # If the immediate token is another newline, return newline token
        if self.c == "\n":
            return self._build_token(t=T.EMPTY_LINE, value="\n", s=s)

        # Otherwise parse blank spaces until we find something else
        indent = self._count_spaces()
        # If it is a comment line, ignore any empty space
        if self.c_future == "#":
            return self._parse_comment()

        return self._maybe_add_dents(indent=indent)

    def _nc(self) -> None:
        """Move to next column position."""
        self.position += 1
        self.column += 1

    def _reset_pos(self, s: Snapshot) -> None:
        self.position = s.position
        self.column = s.column

    def _nl(self) -> None:
        """Move to new line position."""
        self.position += 1
        self.column = 0
        self.line += 1

    def _skip_whitespaces(self) -> None:
        while self.position < self.input_length:
            char = self.input[self.position]
            if char == " ":
                self._nc()
            else:
                break

    def _build_token(
        self, t: T, value: str, s: Snapshot | None = None, quote_char: str | None = None
    ) -> list[Token]:
        if not s:
            s = self._snapshot
        return [
            Token(
                t=t,
                value=value,
                line=s.line,
                column=s.column,
                start=s.position,
                end=s.position + len(value),
                quote_char=quote_char,
            )
        ]

    @property
    def _snapshot(self) -> Snapshot:
        return Snapshot(
            position=self.position,
            column=self.column,
            line=self.line,
        )
