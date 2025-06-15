from typing import NoReturn


class ParsingError(Exception):
    pass


def raise_parsing_error(input_str: str, pos: int, msg: str) -> NoReturn:
    line_pos_sum, faulty_line_index = 0, -1
    input_split = input_str.split("\n")

    # Figure out where the error is based on pos
    for i, line in enumerate(input_split):
        line_pos_sum += len(line) + 1  # Add one since newlines are gone from split
        if line_pos_sum >= pos + 1:  # Add one for a final newline split
            faulty_line_index = i
            break
    line_pos = pos + len(line) - line_pos_sum

    # Build error message
    error_msg = [msg]
    # Check how many previous rows we can print
    print_rows = max(min(faulty_line_index, 3), 0)
    for i in range(faulty_line_index - print_rows, faulty_line_index):
        error_msg += [input_split[i]]
    error_msg += [line, " " * (line_pos + 1) + "^"]

    # Print one subsequent row if possible
    if faulty_line_index + 1 < len(input_split):
        error_msg += [input_split[faulty_line_index + 1]]
    raise ParsingError("\n".join(error_msg))
