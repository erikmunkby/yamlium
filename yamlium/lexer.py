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
        while t := self._parse_next_token():
            self.tokens.append(t)
            if t.t == T.EOF:
                break
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

    def _parse_next_token(self, extra_stop_chars: set = set()) -> Token:
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
            return self._parse_scalar()
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
            self._add_token(t=T.COMMA, value=",")
            return self._parse_next_token(extra_stop_chars=extra_stop_chars)

        # If nothing else, expect value token
        return self._parse_scalar(extra_stop_chars=extra_stop_chars)

    def _parse_flow_style(self, mapping: bool) -> Token:
        s = self._snapshot
        if mapping:
            self._add_token(t=T.MAPPING_START, value="{")
            extra_scalar_stops = {",", "}"}
            stop_token_type = T.MAPPING_END
        else:
            self._add_token(t=T.SEQUENCE_START, value="[")
            extra_scalar_stops = {",", "]"}
            stop_token_type = T.SEQUENCE_END
        self._nc()

        while t := self._parse_next_token(extra_stop_chars=extra_scalar_stops):
            self.tokens.append(t)
            if t.t == stop_token_type:
                break
            if t.t == T.EOF or self.position >= self.input_length:
                flow_type = "mapping" if mapping else "sequence"
                self._raise_error(f"Inline {flow_type} not closed.", pos=s.position)
        return self._parse_next_token(extra_stop_chars=extra_scalar_stops)

    def _parse_scalar(self, extra_stop_chars: set = set()) -> Token:
        s = self._snapshot
        multiline_type = (
            T.MULTILINE_PIPE
            if self.c == "|"
            else T.MULTILINE_ARROW
            if self.c == ">"
            else T.SCALAR
        )
        # If it is a normal scalar
        if multiline_type == T.SCALAR:
            stop_characters = {"#", "&"}.union(extra_stop_chars)
            while self.position < self.input_length:
                char = self.c
                if char == "\n":
                    # Take a snapshot incase next step is an dedentation
                    pre_newline_snapshot = self._snapshot
                    # Stop and expect multiline scalar.
                    # The code will continue past the next else segment
                    self._nl()
                    break
                elif char == ":":
                    self._nc()
                    return self._build_token(
                        t=T.KEY,
                        value=self.input[s.position : self.position - 1],
                        s=s,
                    )
                elif char in stop_characters:
                    return self._build_token(
                        t=T.SCALAR, value=self.input[s.position : self.position], s=s
                    )
                else:
                    self._nc()

            if self.position == self.input_length:
                # This scenario happens when we do not end with a newline
                # And final character symbol is a normal scalar
                return self._build_token(
                    t=T.SCALAR, value=self.input[s.position : self.position], s=s
                )

        else:
            # Parse until we get the first newline.
            # The might be spaces after multiline initiator like:
            # my_key: |<space>
            #   multiline
            #   string
            # TODO: Add functionality for newline preserve/chomp: |- |+ >- >+
            while True:
                if self.c == "\n":
                    self._nl()
                    break
                self._nc()

        # Parse the first indent to get the indent value of the multiline string
        ml_indent = self._count_spaces()
        # If we do not encounter an indentation, it is not a multiline string
        if ml_indent <= 0 or ml_indent <= self.indent_stack[-1]:
            if multiline_type != T.SCALAR:
                self._raise_error("Multiline string requires indentation.")

            # Otherwise add token, add dedents and then continue
            self._add_token(
                t=T.SCALAR,
                value=self.input[s.position : pre_newline_snapshot.position],
                s=s,
            )
            if ml_indent == -1:
                return self._build_token(t=T.EOF, value="")
            if len(self.indent_stack) > 0:
                self._add_dedents(indent=ml_indent)
            return self._parse_next_token(extra_stop_chars=extra_stop_chars)

        post_multiline_newlines = 0
        # Parse multiline
        value, indent = "", 0
        while self.position < self.input_length:
            if multiline_type == T.SCALAR and self.c == ":":
                self._raise_error("Found implicit key after scalar.")
            if self.c == "\n":
                # This might be a post multiline newline, e.g.
                # key: |
                #   line1
                #
                # key2: value
                post_multiline_newlines += 1
                value += "\n"
                self._nl()  # Skip the newline
                indent = self._count_spaces()
                if indent == -1:
                    # Indent -1 signals EOF
                    break
                if self.c == "\n":
                    # We have an immediate newline extension. E.g.
                    # key: |
                    #   line1
                    #
                    #   line2
                    continue
                if indent < ml_indent:
                    break
                if indent > ml_indent:
                    self._raise_error(msg="Irregular multiline string indentation.")
            else:
                post_multiline_newlines = 0  # Reset since we found more content
                value += self.c
                self._nc()

        self._add_token(t=multiline_type, value=value, s=s)

        for _ in range(post_multiline_newlines - 1):
            self._add_token(t=T.EMPTY_LINE, value="")
        if indent >= 0 and indent < self.indent_stack[-1]:
            # If the most recent indent we fetched is less than indent stack
            # Then add as a dedent.
            self._add_dedents(indent=indent)
        return self._parse_next_token()

    def _parse_merge_key(self) -> Token:
        s = self._snapshot

        # After the initial `<` we expect the sequence below.
        # Making the full sequence: <<: *
        for c in "<<:":
            if self.c != c:
                self._raise_error("Found invalid merge key.", pos=self.position)
            self._nc()
        return self._build_token(t=T.KEY, value="<<", s=s)

    def _parse_comment(self) -> Token:
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

    def _parse_quoted_scalar(self) -> Token:
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

    def _parse_alias(self, extra_stop_chars: set) -> Token:
        s = self._snapshot
        return self._build_token(
            t=T.ALIAS,
            value=self._anchor_or_alias_name(extra_stop_chars=extra_stop_chars),
            s=s,
        )

    def _parse_anchor(self, extra_stop_chars: set) -> Token:
        s = self._snapshot
        return self._build_token(
            t=T.ANCHOR,
            value=self._anchor_or_alias_name(extra_stop_chars=extra_stop_chars),
            s=s,
        )

    def _parse_dash(self) -> Token:
        # Start with char, since current char is '-'
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
        self._add_token(t=T.DASH, value="-", s=s)
        return self._maybe_add_dents(indent=self.column)

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

    def _add_dedents(self, indent: int) -> None:
        while indent < self.indent_stack[-1]:
            self._add_token(t=T.DEDENT, value="")
            self.indent_stack.pop()

    def _maybe_add_dents(self, indent: int) -> Token:
        if indent == -1:
            return self._build_token(t=T.EOF, value="")
        if indent > self.indent_stack[-1]:
            self.indent_stack.append(indent)
            return self._build_token(t=T.INDENT, value="")
        elif indent < self.indent_stack[-1]:
            # Add potential dedents
            self._add_dedents(indent=indent)
        return self._parse_next_token()

    def _parse_dents(self) -> Token:
        s = self._snapshot
        self._nl()
        if self._check_eof():
            return self._build_token(t=T.EOF, value="")
        # If the immediate token is another newline, return newline token
        if self.c == "\n":
            return self._build_token(t=T.EMPTY_LINE, value="\n", s=s)

        # Otherwise parse blank spaces until we find something else
        indent = self._count_spaces()
        return self._maybe_add_dents(indent=indent)

    def _nc(self) -> None:
        """Move to next column position."""
        self.position += 1
        self.column += 1

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
    ) -> Token:
        if not s:
            s = self._snapshot
        return Token(
            t=t,
            value=value,
            line=s.line,
            column=s.column,
            start=s.position,
            end=s.position + len(value),
            quote_char=quote_char,
        )

    def _add_token(self, t: T, value: str, s: Snapshot | None = None) -> None:
        if not s:
            s = self._snapshot
        self.tokens.append(self._build_token(t=t, value=value, s=s))

    @property
    def _snapshot(self) -> Snapshot:
        return Snapshot(
            position=self.position,
            column=self.column,
            line=self.line,
        )
