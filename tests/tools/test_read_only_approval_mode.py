"""Conservative read_only approval mode for host-reaching terminal commands."""

import pytest

from tools import approval, approval_context


@pytest.fixture(autouse=True)
def _isolated_read_only_ask(monkeypatch):
    session_key = "read-only-command-mode"
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "read_only")
    monkeypatch.setattr(approval, "_tirith_scan", lambda _command: {"action": "allow"})
    approval.clear_session(session_key)
    approval._permanent_approved.clear()
    yield session_key
    approval.unregister_gateway_notify(session_key)
    approval.clear_session(session_key)
    approval._permanent_approved.clear()


@pytest.mark.parametrize(
    "command",
    [
        "pwd",
        "cd src && pwd",
        "ls -la",
        "find . -type f -name '*.py'",
        "cat pyproject.toml",
        "head -n 20 README.md",
        "tail -n 20 README.md",
        "wc -l README.md",
        "stat README.md",
        "file README.md",
        "du -sh .",
        "tree -L 2 .",
        "realpath README.md",
        "basename /tmp/example.txt",
        "dirname /tmp/example.txt",
        "grep -n 'needle' README.md",
        "rg --files .",
        "git status --short",
        "git diff --stat",
        "git log -5 --oneline",
        "git show HEAD:README.md",
        "git grep needle",
        "git rev-parse --show-toplevel",
        "git ls-files '*.py'",
        "git branch --list 'feature/*'",
        "rg 'needle' . | head -n 5",
        "cd src && find . -type f | wc -l",
    ],
)
def test_read_only_mode_auto_approves_proven_read_commands(command):
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "local",
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []


@pytest.mark.parametrize(
    "command",
    [
        "printf hello > notes.txt",
        "cat < README.md",
        "cat <<EOF\nhello\nEOF",
        "echo $(touch marker)",
        "cat <(printf hello)",
        "echo `touch marker`",
        "TARGET=README.md cat $TARGET",
        "cat *.md",
        "sh -c 'pwd'",
        "bash -lc 'pwd'",
        "eval 'pwd'",
        "env pwd",
        "find . -delete",
        "find . -exec cat {} ';'",
        "find . -ok cat {} ';'",
        "find . -fprint files.txt",
        "find . {-delete,-print}",
        "find . @(*)",
        "sed -n 1p README.md",
        "awk '{print $1}' README.md",
        "python -c 'print(1)'",
        "touch marker",
        "mkdir output",
        "cp README.md copy.md",
        "mv README.md moved.md",
        "rm marker",
        "tee output.txt",
        "chmod 600 README.md",
        "git add README.md",
        "git commit -m checkpoint",
        "git push origin main",
        "git reset --hard HEAD",
        "git rebase main",
        "git branch --delete old",
        "git branch feature/new",
        "curl https://example.com",
        "wget https://example.com",
        "ssh example.com pwd",
        "npm test",
        "pytest -q",
        "make build",
        "alembic upgrade head",
        "tree -o tree.txt .",
        "tree --output tree.txt .",
        "file --compile magic",
        "rg --pre 'touch marker' needle .",
        "pwd; touch marker",
        "pwd || touch marker",
        "pwd & touch marker",
    ],
)
def test_read_only_mode_prompts_once_for_every_unproven_command(command):
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "local",
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1
    assert prompts[0][0][0] == command
    assert prompts[0][1] == {"allow_permanent": False, "allow_session": False}


def test_read_only_mode_ignores_stale_session_and_permanent_grants(_isolated_read_only_ask):
    session_key = _isolated_read_only_ask
    approval.approve_session(session_key, "git push --force")
    approval.approve_permanent("git push --force")
    before_permanent = set(approval._permanent_approved)
    prompts = []
    choices = iter(["once", "deny"])

    def decide(*args, **kwargs):
        prompts.append((args, kwargs))
        return next(choices)

    first = approval.check_all_command_guards("git push --force origin main", "local", approval_callback=decide)
    second = approval.check_all_command_guards("git push --force origin main", "local", approval_callback=decide)

    assert first["approved"] is True
    assert second["approved"] is False
    assert len(prompts) == 2
    assert all(item[1] == {"allow_permanent": False, "allow_session": False} for item in prompts)
    assert approval._session_approved.get(session_key) == {"git push --force"}
    assert approval._permanent_approved == before_permanent


def test_read_only_gateway_payload_offers_once_or_deny_only(_isolated_read_only_ask):
    session_key = _isolated_read_only_ask
    published = []

    def notify(data):
        published.append(data)
        assert approval.resolve_gateway_approval(
            session_key, "once", request_id=data["request_id"]
        ) == 1

    approval.register_gateway_notify(session_key, notify)

    result = approval.check_all_command_guards("touch marker", "local")

    assert result["approved"] is True
    assert len(published) == 1
    payload = published[0]
    assert payload["command"] == "touch marker"
    assert payload["pattern_key"] == "read-only terminal command"
    assert payload["pattern_keys"] == ["read-only terminal command"]
    assert payload["allow_session"] is False
    assert payload["allow_permanent"] is False
    assert payload["request_id"]


def test_read_only_mode_skips_isolated_container_without_host_access():
    prompts = []

    result = approval.check_all_command_guards(
        "touch marker",
        "docker",
        has_host_access=False,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []
