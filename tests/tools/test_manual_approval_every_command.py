"""Manual gateway/ask approval is a one-shot gate for every host-access command."""

import pytest

from tools import approval, approval_context


@pytest.fixture(autouse=True)
def _isolated_manual_ask(monkeypatch):
    session_key = "manual-every-command"
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
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
        "printf hello > notes.txt",
        'git commit -m "checkpoint"',
        "git push origin main",
        "curl https://example.com/health",
    ],
)
def test_manual_ask_prompts_for_every_host_command_once_only(command):
    prompts = []

    def approve_once(prompt_command, description, **capabilities):
        prompts.append((prompt_command, description, capabilities))
        return "once"

    result = approval.check_all_command_guards(
        command, "local", approval_callback=approve_once
    )

    assert result["approved"] is True
    assert result["user_approved"] is True
    assert prompts == [(
        command,
        "Manual approval mode requires confirmation for this terminal command",
        {"allow_permanent": False, "allow_session": False},
    )]


def test_manual_gateway_payload_is_exact_and_once_only(_isolated_manual_ask):
    session_key = _isolated_manual_ask
    published = []

    def notify(data):
        published.append(data)
        assert approval.resolve_gateway_approval(
            session_key, "once", request_id=data["request_id"]
        ) == 1

    approval.register_gateway_notify(session_key, notify)
    command = "git status --short"

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is True
    assert len(published) == 1
    assert published[0]["command"] == command
    assert published[0]["pattern_key"] == "manual terminal command"
    assert published[0]["pattern_keys"] == ["manual terminal command"]
    assert published[0]["allow_session"] is False
    assert published[0]["allow_permanent"] is False
    assert published[0]["request_id"]


def test_manual_mode_skips_container_without_host_access():
    prompts = []

    result = approval.check_all_command_guards(
        "git push origin main",
        "docker",
        has_host_access=False,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []


def test_manual_mode_prompts_for_container_with_host_bind_access():
    prompts = []

    result = approval.check_all_command_guards(
        "git push origin main",
        "docker",
        has_host_access=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1


def test_manual_gate_overrides_permanent_command_allowlist(monkeypatch):
    prompts = []
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda _command: True)

    result = approval.check_all_command_guards(
        "git push origin main",
        "local",
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1
    assert prompts[0][1] == {
        "allow_permanent": False,
        "allow_session": False,
    }


def test_old_permanent_pattern_cannot_bypass_manual_and_manual_choices_do_not_persist(
    _isolated_manual_ask,
):
    session_key = _isolated_manual_ask
    approval.approve_permanent("git push --force")
    before_permanent = set(approval._permanent_approved)
    prompts = []
    choices = iter(["once", "deny"])

    def decide(prompt_command, description, **capabilities):
        prompts.append((prompt_command, description, capabilities))
        return next(choices)

    first = approval.check_all_command_guards(
        "git push --force origin main", "local", approval_callback=decide
    )
    second = approval.check_all_command_guards(
        "git push --force origin main", "local", approval_callback=decide
    )

    assert first["approved"] is True
    assert second["approved"] is False
    assert second["outcome"] == "denied"
    assert len(prompts) == 2
    assert all(prompt[2] == {"allow_permanent": False, "allow_session": False} for prompt in prompts)
    assert approval._session_approved.get(session_key, set()) == set()
    assert approval._permanent_approved == before_permanent


def test_smart_mode_keeps_safe_command_unprompted(monkeypatch):
    prompts = []
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")

    result = approval.check_all_command_guards(
        "git status",
        "local",
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []
