from __future__ import annotations

import inspect
import os
import time
from collections import deque
from typing import NoReturn, TypeVar

from .exceptions import raise_parsing_error
from .lexer import Lexer, T, Token
from .nodes import (
    Alias,
    Document,
    Key,
    Mapping,
    Node,
    Scalar,
    Sequence,
)

DEBUG = os.environ.get("DEBUG_PARSER", "").lower() == "true"

NodeType = TypeVar("NodeType", bound=Node)

_STOP_TOKENS = {T.DOCUMENT_START, T.DOCUMENT_END, T.EOF}
_MAPPING_STOP_TOKENS = {*_STOP_TOKENS, T.DASH}
_SEQUENCE_STOP_TOKENS = {*_STOP_TOKENS, T.KEY}


def _parse_scalar_type(value: str) -> str | int | bool | float | None:
    lv = value.lower()
    if lv in {"null", "~"}:
        return None
    if lv in {"true", "false"}:
        return lv == "true"
    try:
        return int(lv)
    except ValueError:
        try:
            return float(lv)
        except ValueError:
            # If we reach here, it's a string. Return original.
            return value


class Parser:
    lexer: Lexer
    root: Sequence
    stack: deque[Node]

    # Security limits
    # MAX_DEPTH is set conservatively to catch deeply nested YAML before hitting
    # Python's default recursion limit (~1000). Due to recursive calls in parsing,
    # each logical nesting level may use multiple call stack frames.
    MAX_DEPTH = 200
    # MAX_ALIAS_RATIO: Maximum ratio of alias references to decoded nodes.
    # A ratio of 10 allows normal anchor/alias usage while catching potential
    # alias bomb attacks (exponential expansion through nested aliasing).
    MAX_ALIAS_RATIO = 10

    def __init__(self, input: str) -> None:
        self.input = input

    @property
    def _take_token(self) -> Token:
        if self.pos < self.num_tokens:
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        raise IndexError("Tokens out of bounds.")

    @property
    def _peek_token(self) -> Token:
        return self.tokens[self.pos]

    @property
    def _token_type(self) -> T | None:
        if self.pos >= self.num_tokens:
            return None
        t = self.tokens[self.pos]
        if DEBUG:
            time.sleep(0.05)
            val = f"({t.value})" if t.value else ""
            print(
                f"{inspect.stack()[1][3]}: {t.t.name}{val} column={t.column} indent={self.current_indent}"
            )
        return t.t

    @property
    def _last_token(self) -> Token:
        return self.tokens[self.pos - 1]

    @property
    def _last_node(self) -> Node:
        return self.node_stack[-1]

    @property
    def _current_line(self) -> int:
        if len(self.node_stack) > 0:
            return self._last_node._line
        return -1

    def _raise_parsing_error(self, msg: str, pos: int | None = None) -> NoReturn:
        if not pos:
            pos = self._take_token.start
        raise_parsing_error(input_str=self.input, pos=pos, msg=msg)

    def _raise_unexpected_token(self) -> NoReturn:
        t = self._take_token
        self._raise_parsing_error(pos=t.start, msg=f"Found unexpected token {t.t.name}")

    def _process_node(self, n: NodeType) -> NodeType:
        # Always add the node to the stack.
        self.node_stack.append(n)

        # Track decode count for alias bomb protection (don't count aliases)
        if not isinstance(n, Alias):
            self.decode_count += 1

        # Comments that were pending (no blank line after previous value) become
        # HEAD comments of this new node, not FOOT of the previous.
        # Only comments followed by a blank line become FOOT comments.
        if self.pending_foot_comments:
            n.comments.head = self.pending_foot_comments + n.comments.head
            self.pending_foot_comments = []

        # Attach head comments from cache (comments after a blank line)
        if self.comment_cache:
            n.comments.head = n.comments.head + self.comment_cache
            self.comment_cache = []

        # Reset blank line state when processing a new node
        self.saw_blank_line = False

        # Track last value node (keys don't receive foot comments)
        if not isinstance(n, Key):
            self.last_value_node = n

        return n

    # ------------------------------------------------------------------
    # ---------------------------- Builders ----------------------------
    # ------------------------------------------------------------------
    def _build_scalar(self, in_mapping: bool = False) -> Scalar:
        t = self._take_token
        val = t.value.rstrip() if t.t == T.SCALAR else t.value
        indented = in_mapping and t.line > self._current_line and t.t == T.SCALAR
        return self._process_node(
            Scalar(
                _type=t.t,  # type: ignore
                _line=t.line,
                _value=_parse_scalar_type(val) if not t.quote_char else str(val),
                _is_indented=indented,
                _original_value=val,
                _quote_char=t.quote_char,
                _chomp=t.chomp,
            )
        )

    def _build_key(self) -> Key:
        t = self._take_token
        return self._process_node(
            Key(
                _value=t.value,
                _line=t.line,
                _is_merge_key=t.value == "<<",
                _indent=self.current_indent,
            )
        )

    def _build_alias(self) -> Alias:
        t = self._take_token
        alias_name = t.value

        # Circular reference detection
        if alias_name in self.resolving_aliases:
            self._raise_parsing_error(
                f"Circular reference detected: anchor '{alias_name}' references itself",
                pos=t.start,
            )

        node_value = self.anchors.get(alias_name)
        if node_value is None:
            raise self._raise_parsing_error(
                f"No anchor found for alias `*{alias_name}`", pos=t.start
            )

        # Alias bomb protection
        self.alias_count += 1
        if self.decode_count > 0:
            ratio = self.alias_count / self.decode_count
            if ratio > self.MAX_ALIAS_RATIO:
                self._raise_parsing_error(
                    f"Excessive aliasing detected (ratio {ratio:.1f} exceeds limit {self.MAX_ALIAS_RATIO})",
                    pos=t.start,
                )

        return self._process_node(
            Alias(_line=t.line, child=node_value, _value=alias_name)
        )

    # ------------------------------------------------------------------
    # ---------------------------- Handlers ----------------------------
    # ------------------------------------------------------------------
    def _handle_comment(self) -> None:
        token = self._take_token

        # Inline: same line as current node
        if self._current_line == token.line:
            self._last_node.comments.line = token.value
            return

        # If no value node exists yet, comments become head of next node
        if self.last_value_node is None:
            self.comment_cache.append(token.value)
            return

        # After blank line: head of next node
        if self.saw_blank_line:
            self.comment_cache.append(token.value)
        else:
            # No blank line yet: potential foot of previous node
            self.pending_foot_comments.append(token.value)

    def _handle_anchor(self) -> Mapping | Scalar | Sequence | Alias:
        n = self._last_node
        t = self._take_token
        if not isinstance(n, Key):
            self._raise_parsing_error("Anchors can only be placed with keys.")
        n.anchor = t.value

        # Track that we're resolving this anchor (for circular reference detection)
        self.resolving_aliases.add(t.value)
        try:
            # Now find the value beyond the anchor
            value = self._parse_value()
            self.anchors[t.value] = value
            return value
        finally:
            self.resolving_aliases.discard(t.value)

    def _handle_indent(self) -> None:
        self._take_token
        self.current_indent += 1

    def _handle_dedent(self) -> None:
        self._take_token
        self.current_indent -= 1

    def _check_depth(self) -> None:
        """Check if current nesting depth exceeds the maximum allowed."""
        if self.current_depth > self.MAX_DEPTH:
            self._raise_parsing_error(
                f"Maximum nesting depth exceeded ({self.MAX_DEPTH})"
            )

    def _check_special_types(self, t: T | None) -> bool:
        if t == T.COMMENT:
            self._handle_comment()
        elif t == T.EMPTY_LINE:
            self._take_token  # Consume token
            # Flush pending foot comments to last value node
            if self.pending_foot_comments and self.last_value_node:
                self.last_value_node.comments.foot = self.pending_foot_comments
                self.pending_foot_comments = []
            self.saw_blank_line = True
            self._last_node.newlines += 1
        else:
            return False
        return True

    # -----------------------------------------------------------------
    # ---------------------------- Parsers ----------------------------
    # -----------------------------------------------------------------
    def _parse_value(
        self, in_mapping: bool = False
    ) -> Mapping | Scalar | Sequence | Alias:
        # Check depth before any recursive parsing
        self._check_depth()

        t = self._token_type
        if t == T.KEY:
            n = self._last_node
            if n._indent == self.current_indent and isinstance(n, Key):
                return Scalar(
                    _type=T.SCALAR,
                    _line=n._line,
                    _value=None,
                )
            return self._parse_mapping()
        if t in {T.SCALAR, T.MULTILINE_ARROW, T.MULTILINE_PIPE}:
            return self._build_scalar(in_mapping=in_mapping)
        if self._check_special_types(t=t):
            return self._parse_value(in_mapping=in_mapping)
        if t == T.INDENT:
            self._handle_indent()
            return self._parse_value(in_mapping=in_mapping)
        if t == T.DASH:
            return self._parse_sequence()
        if t == T.ANCHOR:
            return self._handle_anchor()
        if t == T.ALIAS:
            return self._build_alias()
        if t == T.MAPPING_START:
            return self._parse_inline_mapping()
        if t == T.SEQUENCE_START:
            return self._parse_inline_sequence()

        self._raise_unexpected_token()

    def _parse_inline_mapping(
        self,
    ) -> Mapping:
        t = self._take_token
        m = Mapping(_line=t.line, _is_inline=True)
        m._column = t.column

        while t := self._token_type:
            if t == T.MAPPING_END:
                self._take_token
                self._process_node(m)
                return m
            elif t == T.KEY:
                key = self._build_key()
                m[key] = self._parse_value()
            elif t == T.COMMA:
                self._take_token
                continue
            else:
                self._raise_unexpected_token()
        self._raise_parsing_error("Inline mapping not closed.")

    def _parse_inline_sequence(self) -> Sequence:
        t = self._take_token
        s = Sequence(_line=t.line, _is_inline=True)
        s._column = t.column

        while t := self._token_type:
            if t == T.SEQUENCE_END:
                self._take_token
                self._process_node(s)
                return s
            elif t == T.COMMA:
                self._take_token
                continue
            else:
                s.append(self._parse_value())
        self._raise_parsing_error("Inline sequence not closed.")

    def _parse_mapping(self) -> Mapping:
        self.current_depth += 1
        self._check_depth()
        try:
            m = Mapping(_line=self._last_token.line)
            m._indent = self.current_indent
            start_indent = self.current_indent
            m._column = self._peek_token.column  # Set mapping column

            while t := self._token_type:
                if t == T.KEY:
                    key = self._build_key()
                    m[key] = self._parse_value(in_mapping=True)
                elif self._check_special_types(t=t):
                    continue
                elif t == T.DEDENT:
                    self._handle_dedent()
                    if self.current_indent < start_indent:
                        break
                elif t in _MAPPING_STOP_TOKENS:
                    break
                else:
                    self._raise_unexpected_token()

            # Transfer newlines from the last scalar value to the parent mapping
            # This preserves whitespace after mappings when the last value is updated
            if m:
                keys = list(m.keys())
                last_value = m[keys[-1]]
                if isinstance(last_value, Scalar) and last_value.newlines > 0:
                    m.newlines = last_value.newlines
                    last_value.newlines = 0

            return m
        finally:
            self.current_depth -= 1

    def _parse_sequence(self) -> Sequence:
        self.current_depth += 1
        self._check_depth()
        try:
            s = Sequence(_line=self._last_token.line)
            start_indent = self.current_indent
            while t := self._token_type:
                if t == T.DASH:
                    self._take_token
                    # Immediate token after dashes will be an indentation
                    if self._token_type == T.INDENT:
                        self._handle_indent()
                    s.append(self._parse_value())
                elif self._check_special_types(t=t):
                    continue
                elif t == T.DEDENT:
                    self._handle_dedent()
                    if self.current_indent < start_indent:
                        break
                elif t in _SEQUENCE_STOP_TOKENS:
                    break
                else:
                    self._raise_unexpected_token()
            return s
        finally:
            self.current_depth -= 1

    def parse(self) -> Document:
        # Set up class vars
        self.pos = 0
        self.current_indent = 0
        self.tokens = Lexer(self.input).build_tokens()
        self.num_tokens = len(self.tokens)
        self.node_stack: deque[Node] = deque([])
        self.anchors: dict[str, Node] = {}
        self.anchor_cache: str | None = None
        self.comment_cache: list[str] = []

        # Comment classification state (head/line/foot semantics)
        self.pending_foot_comments: list[str] = []  # May become foot of previous node
        self.saw_blank_line: bool = False  # Tracks blank line occurrence
        self.last_value_node: Node | None = None  # Node eligible for foot comments

        # Security: depth limiting
        self.current_depth = 0

        # Security: alias bomb protection
        self.alias_count = 0
        self.decode_count = 0

        # Security: circular reference detection
        self.resolving_aliases: set[str] = set()

        root = Document()
        while t := self._token_type:
            if t == T.KEY:
                root.append(self._parse_mapping())
            elif t == T.SCALAR:
                root.append(self._build_scalar())
            # Sometimes indents/dedents exist in root.
            # Simply ignore these.
            elif t == T.INDENT:
                self._handle_indent()
            elif t == T.DEDENT:
                self._handle_dedent()
            elif self._check_special_types(t=t):
                continue
            elif t == T.MAPPING_START:
                self._parse_inline_mapping()
            elif t == T.SEQUENCE_START:
                self._parse_inline_sequence()
            elif t == T.DOCUMENT_START:
                self._take_token
            elif t == T.EOF:
                break
            else:
                self._raise_unexpected_token()

        return root
