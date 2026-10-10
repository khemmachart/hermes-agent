"""Conservative shell classifier for ``approvals.mode: read_only``.

This is an allowlist, not a danger detector: a command auto-runs only when its
entire shell expression is proven to contain supported read/navigation commands.
Anything ambiguous returns ``False`` and is left to the one-shot human gate.
"""

from __future__ import annotations

import re
import shlex


_READ_COMMANDS = frozenset({
    "cd", "pwd", "ls", "find", "cat", "head", "tail", "wc", "stat", "file",
    "du", "tree", "realpath", "basename", "dirname", "grep", "rg", "git",
})
_ALLOWED_OPERATORS = frozenset({"|", "&&"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_FIND_MUTATING_ACTIONS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})
_GIT_READ_SUBCOMMANDS = frozenset({"status", "diff", "log", "show", "grep", "rev-parse", "ls-files"})
_GIT_MUTATING_BRANCH_OPTIONS = frozenset({
    "-c", "-C", "--copy", "--create-reflog", "-d", "-D", "--delete",
    "--edit-description", "-m", "-M", "--move", "--set-upstream-to", "-u",
    "--unset-upstream",
})


def _has_active_shell_hazard(command: str) -> bool:
    """Reject active shell expansion/control syntax before tokenization.

    Quoted literals remain usable (for example ``grep 'a|b'`` and
    ``find -name '*.py'``), while unquoted globs are rejected because expansion
    can turn a filename such as ``-delete`` into an option.
    """
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            elif quote == '"' and char in {"$", "`"}:
                return True
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in {"$", "`", "<", ">", ";", "\n", "\r", "*", "?", "[", "{", "}", "(", ")"}:
            return True
        if char == "&":
            if index + 1 < len(command) and command[index + 1] == "&":
                index += 2
                continue
            return True
        if char == "|":
            if index + 1 < len(command) and command[index + 1] in {"|", "&"}:
                return True
        index += 1
    return bool(quote or escaped)


def _tokens(command: str) -> list[str] | None:
    if not isinstance(command, str) or not command.strip() or _has_active_shell_hazard(command):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def _has_option(args: list[str], *options: str) -> bool:
    return any(token in options or any(token.startswith(option + "=") for option in options if option.startswith("--"))
               for token in args)


def _safe_find(args: list[str]) -> bool:
    for token in args:
        lowered = token.lower()
        if lowered in _FIND_MUTATING_ACTIONS:
            return False
        if lowered.startswith(("-fprint", "-fprintf", "-fls")):
            return False
    return True


def _safe_tail(args: list[str]) -> bool:
    for token in args:
        if token in {"-F", "--follow", "--retry"} or token.startswith("--follow="):
            return False
        if token.startswith("-") and not token.startswith("--") and "f" in token[1:]:
            return False
    return True


def _safe_tree(args: list[str]) -> bool:
    return not any(token in {"-o", "--output"} or token.startswith("--output=") for token in args)


def _safe_file(args: list[str]) -> bool:
    for token in args:
        if token == "--compile" or token.startswith("--compile="):
            return False
        if token.startswith("-") and not token.startswith("--") and "C" in token[1:]:
            return False
    return True


def _safe_rg(args: list[str]) -> bool:
    return not any(token == "--pre" or token.startswith(("--pre=", "--pre-glob=")) for token in args)


def _git_subcommand(args: list[str]) -> tuple[str, list[str]] | None:
    """Return a narrowly parsed git subcommand and its arguments."""
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--no-pager":
            index += 1
            continue
        if token == "-C":
            if index + 1 >= len(args):
                return None
            index += 2
            continue
        break
    if index >= len(args):
        return None
    return args[index], args[index + 1:]


def _safe_git(args: list[str]) -> bool:
    parsed = _git_subcommand(args)
    if parsed is None:
        return False
    subcommand, subargs = parsed
    if subcommand in _GIT_READ_SUBCOMMANDS:
        if subcommand in {"diff", "log", "show"} and _has_option(
            subargs, "--output", "--ext-diff", "--textconv"
        ):
            return False
        if subcommand == "grep" and _has_option(subargs, "--open-files-in-pager"):
            return False
        return True
    if subcommand != "branch" or "--list" not in subargs:
        return False
    return not any(
        token in _GIT_MUTATING_BRANCH_OPTIONS
        or token.startswith(("--copy=", "--delete=", "--move=", "--set-upstream-to="))
        for token in subargs
    )


def _safe_segment(segment: list[str]) -> bool:
    if not segment or _ASSIGNMENT_RE.match(segment[0]):
        return False
    command, args = segment[0], segment[1:]
    if command not in _READ_COMMANDS:
        return False
    if command == "find":
        return _safe_find(args)
    if command == "tail":
        return _safe_tail(args)
    if command == "tree":
        return _safe_tree(args)
    if command == "file":
        return _safe_file(args)
    if command == "rg":
        return _safe_rg(args)
    if command == "git":
        return _safe_git(args)
    return True


def is_conservative_read_only_command(command: str) -> bool:
    """Whether every command in a ``|``/``&&`` expression is proven read-only."""
    tokens = _tokens(command)
    if not tokens:
        return False
    segment: list[str] = []
    for token in tokens:
        if token in _ALLOWED_OPERATORS:
            if not _safe_segment(segment):
                return False
            segment = []
        elif token in {"||", "&", ";", "<", ">", "<<", ">>"}:
            return False
        else:
            segment.append(token)
    return _safe_segment(segment)
