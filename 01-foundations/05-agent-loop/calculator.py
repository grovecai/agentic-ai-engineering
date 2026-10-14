"""A simple interactive calculator in Python.

Supports +, -, *, /, ** (power), % (modulo), and sqrt.

Usage:
    python calculator.py            # start an interactive session (REPL)
    python calculator.py 2 + 3      # evaluate a single expression and exit

Interactive mode keeps running so you don't have to restart the script.
Inside the session you can:
    2 + 3            evaluate an expression
    2+3              spaces are optional
    sqrt 16          unary operations work too
    sqrt(16)         ...and function-style calls
    x = 2 + 3        store a result in a variable
    x * 2            use stored variables
    ans * 2          `ans` (or `_`) is the previous result
    vars             list stored variables
    history          show recent inputs
    help             show the command reference
    clear            forget all variables and history
    quit / exit / q  leave the session

Press Ctrl+C to cancel the current input, Ctrl+D to quit.
"""

from __future__ import annotations

import math
import re
import shlex
import sys

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return a / b


def power(a, b):
    return a**b


def modulo(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot modulo by zero.")
    return a % b


def square_root(a):
    if a < 0:
        raise ValueError("Cannot take the square root of a negative number.")
    return math.sqrt(a)


OPERATIONS = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "**": (power, 2),
    "%": (modulo, 2),
    "sqrt": (square_root, 1),
}

# Multi-character operators must be matched before single-character ones.
_BINARY_OPERATORS = sorted(
    (op for op, (_, arity) in OPERATIONS.items() if arity == 2),
    key=len,
    reverse=True,
)
_UNARY_OPERATORS = [op for op, (_, arity) in OPERATIONS.items() if arity == 1]

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

QUIT_COMMANDS = {"quit", "exit", "q"}
RESERVED_NAMES = {"ans", "_"} | set(OPERATIONS)


def calculate(op, *operands):
    """Perform `op` on the given operands and return the result."""
    if op not in OPERATIONS:
        raise ValueError(f"Unknown operation: {op!r}")
    func, arity = OPERATIONS[op]
    if len(operands) != arity:
        raise ValueError(
            f"Operation {op!r} expects {arity} operand(s), got {len(operands)}."
        )
    return func(*operands)


def format_number(value: float) -> str:
    """Render a float nicely: 4.0 -> '4', 4.5 -> '4.5'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class Session:
    """Holds variables, the previous answer, and history between inputs."""

    def __init__(self) -> None:
        self.variables: dict[str, float] = {}
        self.answer: float | None = None
        self.history: list[str] = []

    def remember(self, value: float) -> None:
        """Record a result so it can be reused via `ans` / `_`."""
        self.answer = value
        self.variables["ans"] = value
        self.variables["_"] = value

    def clear(self) -> None:
        self.variables.clear()
        self.answer = None
        self.history.clear()


def _to_number(text: str, session: Session) -> float:
    text = text.strip()
    if _NUMBER_RE.match(text):
        return float(text)
    if text in session.variables:
        return session.variables[text]
    if _IDENTIFIER_RE.match(text):
        raise ValueError(f"Unknown variable: {text!r}")
    raise ValueError(f"{text!r} is not a valid number.")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _strip_outer_parens(text: str) -> str:
    """Remove one balanced pair of surrounding parentheses, if present."""
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return text  # the leading '(' does not wrap the whole string
    return text[1:-1].strip() if depth == 0 else text


def _split_tokens(text: str) -> list[str]:
    """Split an input line into tokens, tolerating comma separators."""
    text = text.replace(",", " ")
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _evaluate_operand(operand_text: str, session: Session) -> float:
    operand_text = _strip_outer_parens(operand_text)
    if not operand_text:
        raise ValueError("Missing operand.")
    return _to_number(operand_text, session)


def _tokenize_compact(text: str) -> list[str]:
    """Split '2+3', 'sqrt(16)' or '2**3' into operator/operand tokens."""
    text = text.strip()
    if not text:
        return []
    # Already a single number or identifier.
    if _NUMBER_RE.match(text) or _IDENTIFIER_RE.match(text):
        return [text]
    # The input is exactly an operator (unusual, but keep the token intact).
    if text in OPERATIONS:
        return [text]

    # Function-style unary: name(operand)
    for unary in _UNARY_OPERATORS:
        if text.startswith(f"{unary}(") and text.endswith(")"):
            return [unary, text[len(unary) + 1 : -1]]

    # Otherwise split around the first binary/unary operator we find.
    for op in _BINARY_OPERATORS + _UNARY_OPERATORS:
        pattern = rf"\b{re.escape(op)}\b" if op.isalpha() else re.escape(op)
        match = re.search(pattern, text)
        if not match:
            continue
        left = text[: match.start()].strip()
        right = text[match.end() :].strip()
        if left and right:
            return [left, op, right]
        if not left and right:  # leading operator, treat it as unary
            return [op, right]
    return [text]


def parse_and_eval(tokens: list[str], session: Session) -> float:
    """Evaluate a list of CLI-style tokens, e.g. ['2', '+', '3']."""
    if not tokens:
        raise ValueError("Empty expression.")

    # Support function-style calls such as 'sqrt(16)' as a single token.
    for index, token in enumerate(tokens):
        if token in OPERATIONS:
            break
        for unary in _UNARY_OPERATORS:
            if token.startswith(f"{unary}(") and token.endswith(")"):
                tokens = [
                    *tokens[:index],
                    unary,
                    token[len(unary) + 1 : -1],
                    *tokens[index + 1 :],
                ]
                break

    # Locate the operator token.
    op_index = next((i for i, tok in enumerate(tokens) if tok in OPERATIONS), None)
    if op_index is None:
        if len(tokens) == 1:  # a bare number is a valid (trivial) expression
            return _to_number(tokens[0], session)
        raise ValueError("No operator found. Expected something like '2 + 3'.")

    op = tokens[op_index]
    _, arity = OPERATIONS[op]
    before = tokens[:op_index]
    after = tokens[op_index + 1 :]

    if arity == 1:
        operand_text = " ".join(after) if after else " ".join(before)
        if not operand_text:
            raise ValueError(f"Operation {op!r} needs an operand.")
        return calculate(op, _evaluate_operand(operand_text, session))

    if not before or not after:
        raise ValueError(f"Operation {op!r} expects two operands.")
    left = _evaluate_operand(" ".join(before), session)
    right = _evaluate_operand(" ".join(after), session)
    return calculate(op, left, right)


def evaluate_input(raw: str, session: Session) -> float:
    """Parse and evaluate a full user input line, returning the result."""
    tokens = _split_tokens(raw)
    if len(tokens) == 1:
        tokens = _tokenize_compact(tokens[0])
    return parse_and_eval(tokens, session)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


HELP_TEXT = """\
Simple Calculator
  Operations : +, -, *, /, ** (power), % (modulo), sqrt
  Examples   : 2 + 3     2*8     sqrt 16     sqrt(16)     2 ** 10
  Variables  : x = 2 + 3   then use x in later expressions
  Previous   : ans or _ refers to the last result
  Commands   : vars, history, help, clear, quit (exit / q)
  Tip        : spaces are optional; Ctrl+D also quits.\
"""


def _handle_command(raw: str, session: Session) -> bool:
    """Handle a built-in command. Return True if the input was consumed."""
    command = raw.lower()
    if command in QUIT_COMMANDS:
        return True
    if command in ("help", "h", "?"):
        print(HELP_TEXT)
        return True
    if command == "vars":
        named = {k: v for k, v in session.variables.items() if k not in ("ans", "_")}
        if not named and session.answer is None:
            print("(no variables yet)")
            return True
        for name, value in named.items():
            print(f"  {name} = {format_number(value)}")
        if session.answer is not None:
            print(f"  ans = {format_number(session.answer)}")
        return True
    if command == "history":
        if not session.history:
            print("(no history yet)")
        else:
            for index, entry in enumerate(session.history, start=1):
                print(f"  {index:>3}  {entry}")
        return True
    if command == "clear":
        session.clear()
        print("Cleared variables and history.")
        return True
    return False


def _handle_assignment(raw: str, session: Session) -> bool:
    """Handle 'name = expression'. Return True if the input was an assignment."""
    if "=" not in raw or raw.count("=") > 1:
        return False
    name, _, expression = raw.partition("=")
    name = name.strip()
    expression = expression.strip()
    if not _IDENTIFIER_RE.match(name):
        return False
    if name in RESERVED_NAMES:
        raise ValueError(f"{name!r} is a reserved name.")
    if not expression:
        raise ValueError("Assignment is missing a value.")
    value = evaluate_input(expression, session)
    session.variables[name] = value
    print(f"{name} = {format_number(value)}")
    return True


def run_expression(args: list[str], session: Session | None = None, echo: bool = True):
    """Evaluate a single expression given as CLI-style tokens."""
    session = session or Session()
    result = evaluate_input(" ".join(args), session)
    session.remember(result)
    if echo:
        print(format_number(result))
    return result


def interactive() -> None:
    """Run the persistent read-eval-print loop."""
    session = Session()
    print("Simple Calculator — type 'help' for commands, 'quit' to exit.\n")
    while True:
        try:
            raw = input("> ").strip()
        except KeyboardInterrupt:
            # Cancel the current line but keep the session alive.
            print("\n(cancelled — type 'quit' to exit)")
            continue
        except EOFError:
            print()
            break

        if not raw:
            continue
        session.history.append(raw)

        if raw.lower() in QUIT_COMMANDS:
            break

        try:
            if _handle_command(raw, session):
                continue
            if _handle_assignment(raw, session):
                continue
            result = evaluate_input(raw, session)
            session.remember(result)
            print(format_number(result))
        except (ValueError, ZeroDivisionError) as exc:
            print(f"Error: {exc}")

    print("Goodbye.")


def main() -> None:
    if len(sys.argv) > 1:
        session = Session()
        try:
            run_expression(sys.argv[1:], session)
        except (ValueError, ZeroDivisionError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
