"""Proof and command policy for ``approvals.mode: workspace_auto``.

The mode is intentionally unavailable unless the resolved terminal configuration
proves a narrow Docker boundary: one explicitly configured host directory is the
only read-write bind and is mounted at ``/workspace``; every other configured
bind is an empty, canonical, non-symlink mask artifact mounted read-only below
it; networking, automatic cwd mounts, persistent host storage, and arbitrary
Docker arguments are disabled.
"""

from __future__ import annotations

import ast
import os
import posixpath
import re
import shlex
import stat
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_VOLUME_RE = re.compile(
    r"^(?P<source>.+):(?P<target>/[^:]*)(?::(?P<mode>[^:]*))?$"
)
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WORKSPACE = "/workspace"


@dataclass(frozen=True)
class WorkspaceAutoConfinement:
    proven: bool
    reason: str
    host_root: str = ""


@dataclass(frozen=True)
class WorkspaceAutoEnvironmentAttestation:
    """Process-bound proof of the immutable Docker launch arguments.

    The HMAC prevents a stale/copyable boolean marker from becoming authority.
    ``environment_id`` binds the proof to the exact cached instance that was
    created from the attested arguments.
    """

    version: int
    environment_id: int
    proven: bool
    host_root: str
    manifest_fingerprint: str
    reason: str
    seal: str


_ATTESTATION_VERSION = 1
_ATTESTATION_KEY = secrets.token_bytes(32)


def _failed(reason: str) -> WorkspaceAutoConfinement:
    return WorkspaceAutoConfinement(False, reason)


def _parse_volume(spec: object) -> tuple[str, str, frozenset[str]] | None:
    if not isinstance(spec, str):
        return None
    match = _VOLUME_RE.fullmatch(spec.strip())
    if match is None:
        return None
    source = match.group("source").strip()
    target = match.group("target").strip()
    raw_mode = match.group("mode")
    modes = frozenset(
        item.strip().lower() for item in (raw_mode or "rw").split(",") if item.strip()
    )
    if not source or not target or not modes:
        return None
    return source, target, modes


def _normalized_target(target: str) -> str | None:
    if not target.startswith("/"):
        return None
    normalized = posixpath.normpath(target)
    # Reject lexical escapes and surprising spellings instead of trying to
    # prove how Docker will canonicalize them.
    if normalized != target.rstrip("/") and not (target == "/" and normalized == "/"):
        return None
    return normalized


def _host_source(source: str) -> str | None:
    expanded = os.path.expanduser(source)
    if not (os.path.isabs(expanded) or _WINDOWS_PATH_RE.match(expanded)):
        return None
    # Do not require Windows paths to exist when validation runs on a POSIX
    # controller, but require local-platform sources to exist. Docker otherwise
    # may create a directory implicitly, changing what was actually approved.
    if _WINDOWS_PATH_RE.match(expanded) and os.name != "nt":
        return os.path.normpath(expanded)
    try:
        resolved = str(Path(expanded).resolve(strict=True))
    except (OSError, RuntimeError):
        return None
    return resolved


def _authorized_host_root(raw: object) -> str | None:
    """Canonical, existing directory explicitly authorized by the operator."""
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        return None
    if _WINDOWS_PATH_RE.match(raw) and os.name != "nt":
        return os.path.normpath(raw) if raw == os.path.normpath(raw) else None
    if not os.path.isabs(raw) or os.path.expanduser(raw) != raw:
        return None
    try:
        resolved = str(Path(raw).resolve(strict=True))
    except (OSError, RuntimeError):
        return None
    if raw != resolved or not Path(resolved).is_dir():
        return None
    return resolved


def _empty_mask_source(source: str) -> str | None:
    """Canonical host path for an empty, non-symlink file or directory mask."""
    if _WINDOWS_PATH_RE.match(source) and os.name != "nt":
        # A controller cannot prove the kind, contents, or symlink posture of a
        # foreign-platform path. workspace_auto always fails closed here.
        return None
    if not os.path.isabs(source) or os.path.expanduser(source) != source:
        return None
    path = Path(source)
    try:
        metadata = path.lstat()
        resolved = str(path.resolve(strict=True))
        if source != resolved or stat.S_ISLNK(metadata.st_mode):
            return None
        if stat.S_ISREG(metadata.st_mode):
            return resolved if metadata.st_size == 0 else None
        if stat.S_ISDIR(metadata.st_mode):
            with os.scandir(resolved) as entries:
                return resolved if next(entries, None) is None else None
    except (OSError, RuntimeError):
        return None
    return None


def _read_only_mode(modes: frozenset[str]) -> bool:
    return modes in (frozenset({"ro"}), frozenset({"readonly"}))


def workspace_auto_confinement(config: dict[str, Any]) -> WorkspaceAutoConfinement:
    """Return a fail-closed proof for the configured workspace Docker boundary."""
    if not isinstance(config, dict) or config.get("env_type") != "docker":
        return _failed("workspace_auto requires the Docker terminal backend")
    if config.get("docker_network") is not False:
        return _failed("Docker network access must be disabled")
    if config.get("docker_mount_cwd_to_workspace") is not False:
        return _failed("automatic cwd mounting must be disabled")
    if config.get("container_persistent") is not False:
        return _failed("container persistent storage must be disabled")
    if config.get("docker_persist_across_processes") is not False:
        return _failed("Docker persistence across processes must be disabled")
    if config.get("docker_shared_container_key") not in (None, ""):
        return _failed("Docker shared container reuse must be disabled")
    if config.get("docker_snap_compat") is not False:
        return _failed("Docker snap compatibility weakens the required sandbox security flags")
    if config.get("docker_extra_args") not in (None, []):
        return _failed("Docker extra args prevent confinement from being proven")

    authorized_root = _authorized_host_root(config.get("workspace_auto_host_root"))
    if authorized_root is None:
        return _failed("an explicit canonical workspace_auto_host_root directory is required")

    volumes = config.get("docker_volumes")
    if not isinstance(volumes, list) or not volumes:
        return _failed("an explicit read-write host mount at /workspace is required")

    workspace_source = ""
    for raw_spec in volumes:
        parsed = _parse_volume(raw_spec)
        if parsed is None:
            return _failed("every Docker volume must be a valid explicit bind mount")
        source, raw_target, modes = parsed
        target = _normalized_target(raw_target)
        if target is None:
            return _failed("Docker bind targets must use canonical absolute paths")
        host_source = _host_source(source)
        if host_source is None:
            return _failed("every Docker volume source must be an existing absolute host path")

        if target == _WORKSPACE:
            if workspace_source:
                return _failed("exactly one host mount may target /workspace")
            if modes != frozenset({"rw"}):
                return _failed("the /workspace host mount must be read-write")
            workspace_source = host_source
            continue

        if not target.startswith(_WORKSPACE + "/"):
            return _failed("all other host bind targets must be beneath /workspace")
        if not _read_only_mode(modes):
            return _failed("all other host binds must be read-only")
        if _empty_mask_source(source) is None:
            return _failed("all read-only host binds must use a canonical empty mask artifact")

    if not workspace_source:
        return _failed("an explicit read-write host mount at /workspace is required")
    if workspace_source != authorized_root:
        return _failed("the /workspace host source must exactly equal workspace_auto_host_root")
    if not (_WINDOWS_PATH_RE.match(workspace_source) and os.name != "nt"):
        try:
            if not Path(workspace_source).is_dir():
                return _failed("the /workspace host source must be a directory")
        except OSError:
            return _failed("the /workspace host source must be an accessible directory")
    return WorkspaceAutoConfinement(True, "sandbox confinement proven", workspace_source)


def is_workspace_auto_confinement_proven(config: dict[str, Any]) -> bool:
    return workspace_auto_confinement(config).proven


def _effective_volume_specs(args: list[str]) -> list[str] | None:
    specs: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-v", "--volume"}:
            if index + 1 >= len(args) or not isinstance(args[index + 1], str):
                return None
            specs.append(args[index + 1])
            index += 2
            continue
        if isinstance(token, str) and token.startswith("--volume="):
            specs.append(token.split("=", 1)[1])
        index += 1
    return specs


def _effective_launch_is_confined(config: dict[str, Any], args: list[str]) -> WorkspaceAutoConfinement:
    proof = workspace_auto_confinement(config)
    if not proof.proven:
        return proof
    if "--network=none" not in args:
        return _failed("effective Docker launch is not network-disabled")
    if not any(args[i:i + 2] == ["--cap-drop", "ALL"] for i in range(len(args) - 1)):
        return _failed("effective Docker launch does not drop all capabilities")
    if not any(args[i:i + 2] == ["--security-opt", "no-new-privileges"] for i in range(len(args) - 1)):
        return _failed("effective Docker launch lacks no-new-privileges")
    specs = _effective_volume_specs(args)
    if specs is None:
        return _failed("effective Docker bind mounts could not be parsed")
    workspace_rw = 0
    for spec in specs:
        parsed = _parse_volume(spec)
        if parsed is None:
            return _failed("effective Docker bind mounts could not be parsed")
        source, raw_target, modes = parsed
        target = _normalized_target(raw_target)
        host_source = _host_source(source)
        if target is None or host_source is None:
            return _failed("effective Docker bind mount paths are not canonical")
        if target == _WORKSPACE:
            if modes != frozenset({"rw"}) or host_source != proof.host_root:
                return _failed("effective Docker launch contains an unauthorized writable host bind")
            workspace_rw += 1
            continue
        if not target.startswith(_WORKSPACE + "/"):
            return _failed("effective Docker launch contains a host bind outside /workspace")
        if not _read_only_mode(modes):
            return _failed("effective Docker launch contains an unauthorized writable host bind")
        if _empty_mask_source(source) is None:
            return _failed("effective Docker launch contains a non-mask read-only host bind")
    if workspace_rw != 1:
        return _failed("effective Docker launch must contain exactly one authorized /workspace bind")
    return proof


def _attestation_payload(attestation: WorkspaceAutoEnvironmentAttestation) -> bytes:
    return json.dumps({
        "version": attestation.version,
        "environment_id": attestation.environment_id,
        "proven": attestation.proven,
        "host_root": attestation.host_root,
        "manifest_fingerprint": attestation.manifest_fingerprint,
        "reason": attestation.reason,
    }, sort_keys=True, separators=(",", ":")).encode()


def attest_workspace_auto_environment(
    environment: Any,
    config: dict[str, Any],
    *,
    effective_run_args: list[str],
) -> WorkspaceAutoEnvironmentAttestation:
    """Seal the actual Docker launch posture onto its created environment."""
    args = [str(arg) for arg in effective_run_args]
    proof = _effective_launch_is_confined(config, args)
    fingerprint = hashlib.sha256(json.dumps(
        {"config": config, "effective_run_args": args},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()
    unsigned = WorkspaceAutoEnvironmentAttestation(
        version=_ATTESTATION_VERSION,
        environment_id=id(environment),
        proven=proof.proven,
        host_root=proof.host_root,
        manifest_fingerprint=fingerprint,
        reason=proof.reason,
        seal="",
    )
    sealed = WorkspaceAutoEnvironmentAttestation(
        **{**unsigned.__dict__, "seal": hmac.new(
            _ATTESTATION_KEY, _attestation_payload(unsigned), hashlib.sha256,
        ).hexdigest()}
    )
    try:
        environment._workspace_auto_attestation = sealed
    except (AttributeError, TypeError):
        pass
    return sealed


def is_workspace_auto_environment_proven(environment: Any) -> bool:
    """Validate a process-bound attestation on the exact cached environment."""
    attestation = getattr(environment, "_workspace_auto_attestation", None)
    if not isinstance(attestation, WorkspaceAutoEnvironmentAttestation):
        return False
    if getattr(environment, "env_type", "") != "docker":
        return False
    if attestation.version != _ATTESTATION_VERSION or attestation.environment_id != id(environment):
        return False
    expected = hmac.new(_ATTESTATION_KEY, _attestation_payload(attestation), hashlib.sha256).hexdigest()
    return attestation.proven and hmac.compare_digest(attestation.seal, expected)


_ALWAYS_REMOTE_TOOLS = frozenset({
    "curl", "wget", "ssh", "scp", "sftp", "ftp", "telnet", "nc", "ncat",
    "gh", "glab", "hub",
    "npx", "pipx", "uvx",
    "ansible", "ansible-playbook", "ansible-galaxy",
    "aws", "gcloud", "az", "doctl", "oci", "ibmcloud", "cloudflared", "rclone", "s3cmd",
    "kubectl", "helm", "heroku", "vercel", "netlify", "flyctl", "railway",
    "firebase", "wrangler", "supabase", "pulumi", "terraform", "tofu",
    "himalaya", "mail", "mailx", "sendmail", "msmtp", "mutt",
})
_SYSTEM_PACKAGE_MANAGERS = frozenset({
    "apt", "apt-get", "apk", "dnf", "yum", "zypper", "pacman", "brew",
    "port", "choco", "winget", "scoop", "nix", "nix-env",
})
_REGISTRY_SUBCOMMANDS = {
    "npm": frozenset({"add", "audit", "ci", "install", "i", "login", "logout", "outdated", "publish", "search", "unpublish", "update", "upgrade", "view"}),
    "pnpm": frozenset({"add", "audit", "fetch", "import", "install", "i", "outdated", "publish", "remove", "update", "upgrade"}),
    "yarn": frozenset({"add", "audit", "import", "install", "outdated", "publish", "remove", "set", "upgrade"}),
    "bun": frozenset({"add", "install", "i", "link", "outdated", "publish", "remove", "update"}),
    "pip": frozenset({"download", "index", "install", "search", "uninstall"}),
    "pip3": frozenset({"download", "index", "install", "search", "uninstall"}),
    "uv": frozenset({"add", "build", "lock", "pip", "publish", "remove", "sync", "tool"}),
    "poetry": frozenset({"add", "install", "lock", "publish", "remove", "self", "update"}),
    "cargo": frozenset({"add", "install", "login", "owner", "publish", "search", "update", "yank"}),
    "gem": frozenset({"install", "owner", "push", "signin", "signout", "sources", "update"}),
    "composer": frozenset({"create-project", "install", "require", "search", "self-update", "update"}),
    "go": frozenset({"get", "install"}),
    "dotnet": frozenset({"add", "nuget", "restore", "tool"}),
}
_GIT_REMOTE_SUBCOMMANDS = frozenset({
    "clone", "fetch", "pull", "push", "ls-remote", "submodule", "send-email",
})
_CONTAINER_REMOTE_SUBCOMMANDS = frozenset({
    "login", "logout", "pull", "push", "search",
})
_SHELL_CARRIERS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "fish"})
_CONTROL_TOKENS = frozenset({"|", "||", "&&", ";", "&", "(", ")"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_LOCAL_SAFE_TOOLS = frozenset({
    "true", "false", "test", "[", "cd", "pwd", "printf", "echo", "cat", "tee",
    "touch", "mkdir", "rmdir", "cp", "mv", "rm", "ln", "chmod", "chown", "chgrp",
    "truncate", "dd", "install", "mktemp", "readlink", "realpath", "basename", "dirname",
    "ls", "find", "head", "tail", "wc", "stat", "file", "du", "tree", "grep", "rg",
    "sort", "uniq", "cut", "tr", "paste", "diff", "cmp", "patch", "sed", "awk",
    "make", "cmake", "ninja", "meson", "pytest", "ruff", "mypy", "tox", "nox",
    "gradle", "gradlew", "mvn", "ant", "xcodebuild", "swift", "swiftc",
})
_LOCAL_GIT_SUBCOMMANDS = frozenset({
    "add", "am", "apply", "bisect", "branch", "checkout", "cherry-pick", "clean", "commit",
    "config", "diff", "diff-index", "diff-tree", "format-patch", "grep", "init", "log", "merge",
    "merge-base", "mv", "notes", "rebase", "reflog", "reset", "restore", "rev-list", "rev-parse",
    "rm", "show", "show-ref", "stash", "status", "switch", "tag", "worktree", "ls-files",
})
_LOCAL_PACKAGE_SUBCOMMANDS = {
    "npm": frozenset({"run", "run-script", "test", "t", "build", "exec"}),
    "pnpm": frozenset({"run", "test", "build", "exec"}),
    "yarn": frozenset({"run", "test", "build", "exec"}),
    "bun": frozenset({"run", "test", "build"}),
    "cargo": frozenset({"build", "check", "clippy", "fmt", "metadata", "test"}),
    "go": frozenset({"build", "fmt", "list", "test", "vet"}),
    "dotnet": frozenset({"build", "format", "test"}),
}


def _strip_heredoc_bodies(command: str) -> str | None:
    """Remove shell heredoc payloads while preserving executable command lines.

    Heredoc bodies are literal stdin data, so parsing their punctuation as shell
    commands causes false approval prompts. Unterminated or unusual delimiters
    fail closed. Operators inside quoted strings are not treated as openers.
    """
    delimiter_re = re.compile(r"^[A-Za-z0-9_]+$")

    def scan_line(line: str) -> tuple[str, list[tuple[str, bool]]] | None:
        output: list[str] = []
        markers: list[tuple[str, bool]] = []
        index = 0
        quote = ""
        escaped = False
        while index < len(line):
            char = line[index]
            if escaped:
                output.append(char)
                escaped = False
                index += 1
                continue
            if char == "\\" and quote != "'":
                output.append(char)
                escaped = True
                index += 1
                continue
            if quote:
                output.append(char)
                if char == quote:
                    quote = ""
                index += 1
                continue
            if char in {"'", '"'}:
                quote = char
                output.append(char)
                index += 1
                continue
            if line.startswith("<<", index) and not line.startswith("<<<", index):
                cursor = index + 2
                strip_tabs = cursor < len(line) and line[cursor] == "-"
                if strip_tabs:
                    cursor += 1
                while cursor < len(line) and line[cursor] in " \t":
                    cursor += 1
                if cursor >= len(line):
                    return None
                delimiter_quote = line[cursor] if line[cursor] in {"'", '"'} else ""
                if delimiter_quote:
                    end = line.find(delimiter_quote, cursor + 1)
                    if end < 0:
                        return None
                    delimiter = line[cursor + 1:end]
                    cursor = end + 1
                else:
                    start = cursor
                    while cursor < len(line) and line[cursor] not in " \t;&|()<>":
                        cursor += 1
                    delimiter = line[start:cursor]
                if not delimiter_re.fullmatch(delimiter):
                    return None
                markers.append((delimiter, strip_tabs))
                output.append(" ")
                index = cursor
                continue
            output.append(char)
            index += 1
        if quote or escaped:
            # Multi-line or malformed quoting is ambiguous to this small parser.
            # Prompt instead of risking that a later line is treated as data.
            return None
        return "".join(output), markers

    output_lines: list[str] = []
    pending: list[tuple[str, bool]] = []
    for line in command.splitlines():
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                pending.pop(0)
            continue
        scanned = scan_line(line)
        if scanned is None:
            return None
        sanitized, markers = scanned
        output_lines.append(sanitized)
        pending.extend(markers)
    if pending:
        return None
    # A shell newline separates commands just like `;`. shlex otherwise treats
    # it as ordinary whitespace and can misclassify a later executable as an
    # argument to the preceding safe command.
    return " ; ".join(output_lines)


def _shell_tokens(command: str) -> list[str] | None:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except (TypeError, ValueError):
        return None


def _command_segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _CONTROL_TOKENS:
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _strip_prefixes(segment: list[str]) -> list[str]:
    index = 0
    while index < len(segment) and _ASSIGNMENT_RE.match(segment[index]):
        index += 1
    while index < len(segment) and os.path.basename(segment[index]) in {"command", "env", "sudo", "nohup"}:
        wrapper = os.path.basename(segment[index])
        index += 1
        if index < len(segment) and segment[index] == "--":
            index += 1
            continue
        if wrapper == "env":
            value_options = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
            while index < len(segment):
                token = segment[index]
                if _ASSIGNMENT_RE.match(token):
                    index += 1
                elif token in value_options:
                    index += 2
                elif any(token.startswith(option + "=") for option in value_options if option.startswith("--")):
                    index += 1
                elif token.startswith("-"):
                    index += 1
                else:
                    break
        elif wrapper == "sudo":
            value_options = {
                "-C", "--close-from", "-D", "--chdir", "-g", "--group", "-h", "--host",
                "-p", "--prompt", "-r", "--role", "-t", "--type", "-T", "--command-timeout",
                "-u", "--user",
            }
            while index < len(segment):
                token = segment[index]
                if token in value_options:
                    index += 2
                elif any(token.startswith(option + "=") for option in value_options if option.startswith("--")):
                    index += 1
                elif token.startswith("-"):
                    index += 1
                else:
                    break
        elif wrapper == "command" and index < len(segment) and segment[index] in {"-p", "-v", "-V"}:
            index += 1
    return segment[index:]


def _git_subcommand(args: list[str]) -> str:
    index = 0
    value_flags = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix"}
    while index < len(args):
        token = args[index]
        if token == "--":
            index += 1
            break
        if token in value_flags:
            index += 2
            continue
        if any(token.startswith(flag + "=") for flag in value_flags if flag.startswith("--")):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token.lower()
    return args[index].lower() if index < len(args) else ""


def _git_has_dynamic_alias(args: list[str]) -> bool:
    for index, token in enumerate(args):
        value = args[index + 1] if token == "-c" and index + 1 < len(args) else ""
        if token.startswith("-c") and token != "-c":
            value = token[2:].lstrip("=")
        if value.lower().startswith("alias."):
            return True
    return False


def _rsync_is_remote(args: list[str]) -> bool:
    for token in args:
        lowered = token.lower()
        if lowered.startswith(("rsync://", "ssh://")) or token == "--daemon":
            return True
        if ":" in token and not _WINDOWS_PATH_RE.match(token):
            return True
    return False


def _python_inline_is_proven_local(code: str) -> bool:
    """Tiny AST allowlist for local-only ``python -c`` filesystem snippets."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return False
    safe_names = {
        "Path", "print", "open", "len", "str", "int", "float", "bool", "list", "dict",
        "set", "tuple", "sorted", "sum", "min", "max", "range", "enumerate", "zip",
    }
    safe_attributes = {
        "write_text", "write_bytes", "read_text", "read_bytes", "mkdir", "unlink", "rename",
        "replace", "touch", "chmod", "exists", "is_file", "is_dir", "glob", "rglob", "resolve",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name != "pathlib" for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module != "pathlib" or any(alias.name != "Path" for alias in node.names):
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in safe_names:
                    return False
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in safe_attributes:
                    return False
            else:
                return False
    return True


def _segment_is_remote(segment: list[str]) -> bool:
    if any("$(" in token or "`" in token or token.startswith("$'") for token in segment):
        return True
    segment = _strip_prefixes(segment)
    if not segment:
        return False
    executable = os.path.basename(segment[0]).lower()
    args = segment[1:]
    if "$" in segment[0] or executable in {"eval", "source", ".", "exec"}:
        return True
    if executable in _ALWAYS_REMOTE_TOOLS or executable in _SYSTEM_PACKAGE_MANAGERS:
        return True
    if executable == "rsync":
        return _rsync_is_remote(args)
    if executable == "git":
        if _git_has_dynamic_alias(args):
            return True
        subcommand = _git_subcommand(args)
        if subcommand in _GIT_REMOTE_SUBCOMMANDS:
            return True
        try:
            sub_index = args.index(subcommand)
        except ValueError:
            sub_index = -1
        subargs = args[sub_index + 1:] if sub_index >= 0 else []
        if subcommand == "remote":
            action = next((arg.lower() for arg in subargs if not arg.startswith("-")), "")
            return action not in {"", "add", "get-url", "remove", "rename", "set-branches", "set-head", "set-url", "show"}
        if subcommand == "lfs":
            return True
        if subcommand == "archive":
            return any(arg == "--remote" or arg.startswith("--remote=") for arg in subargs)
        return subcommand not in _LOCAL_GIT_SUBCOMMANDS
    if executable in {"docker", "podman"}:
        subcommand = next((arg.lower() for arg in args if not arg.startswith("-")), "")
        return subcommand in _CONTAINER_REMOTE_SUBCOMMANDS or not subcommand
    if executable in _REGISTRY_SUBCOMMANDS:
        subcommand = next((arg.lower() for arg in args if not arg.startswith("-")), "")
        if subcommand in _REGISTRY_SUBCOMMANDS[executable]:
            return True
        return subcommand not in _LOCAL_PACKAGE_SUBCOMMANDS.get(executable, frozenset())
    if executable in _SHELL_CARRIERS:
        for index, arg in enumerate(args):
            if arg in {"-c", "-lc"} and index + 1 < len(args):
                return is_remote_or_network_command(args[index + 1])
            if arg.startswith("-") and "c" in arg[1:] and index + 1 < len(args):
                return is_remote_or_network_command(args[index + 1])
        return True
    if executable == "xargs":
        index = 0
        value_options = {
            "-a", "--arg-file", "-d", "--delimiter", "-E", "--eof", "-I", "--replace",
            "-L", "--max-lines", "-n", "--max-args", "-P", "--max-procs", "-s", "--max-chars",
        }
        while index < len(args):
            token = args[index]
            if token == "--":
                index += 1
                break
            if token in value_options:
                index += 2
                continue
            if any(token.startswith(option + "=") for option in value_options if option.startswith("--")):
                index += 1
                continue
            if re.match(r"^-(?:[adEILnPs]).+", token):
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                continue
            break
        return index >= len(args) or _segment_is_remote(args[index:])
    if executable in {"python", "python3", "python3.13", "python3.14"}:
        if len(args) >= 2 and args[0] == "-m":
            return args[1].lower() not in {"pytest", "unittest", "compileall"}
        if len(args) >= 2 and args[0] == "-c":
            return not _python_inline_is_proven_local(args[1])
        return True
    if executable in _LOCAL_SAFE_TOOLS:
        return False
    if executable in _LOCAL_PACKAGE_SUBCOMMANDS:
        subcommand = next((arg.lower() for arg in args if not arg.startswith("-")), "")
        return subcommand not in _LOCAL_PACKAGE_SUBCOMMANDS[executable]
    return True


def is_remote_or_network_command(command: str) -> bool:
    """Whether a command requires a one-shot prompt in workspace-auto mode.

    This is a conservative prompt policy, not the physical security boundary.
    Only known-local commands auto-run; unknown or dynamic carriers prompt.
    """
    if not isinstance(command, str) or not command.strip():
        return True
    executable_command = _strip_heredoc_bodies(command)
    if executable_command is None:
        return True
    tokens = _shell_tokens(executable_command)
    if not tokens:
        return True
    return any(_segment_is_remote(segment) for segment in _command_segments(tokens))
