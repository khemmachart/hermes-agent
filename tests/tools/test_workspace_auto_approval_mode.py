"""Sandbox-proven workspace_auto approvals for local workspace mutations."""

from copy import deepcopy

import pytest

from tools import approval, approval_context


@pytest.fixture(autouse=True)
def _isolated_workspace_auto_ask(monkeypatch):
    session_key = "workspace-auto-command-mode"
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "workspace_auto")
    monkeypatch.setattr(approval, "_tirith_scan", lambda _command: {"action": "allow"})
    approval.clear_session(session_key)
    approval._permanent_approved.clear()
    yield session_key
    approval.unregister_gateway_notify(session_key)
    approval.clear_session(session_key)
    approval._permanent_approved.clear()


@pytest.fixture()
def confined_config(tmp_path):
    workspace = tmp_path / "Workspace"
    workspace.mkdir()
    secret_mask = tmp_path / "empty-mask"
    secret_mask.mkdir()
    return {
        "env_type": "docker",
        "workspace_auto_host_root": str(workspace),
        "docker_volumes": [
            f"{workspace}:/workspace:rw",
            f"{secret_mask}:/workspace/project/.env:ro",
        ],
        "docker_network": False,
        "docker_mount_cwd_to_workspace": False,
        "container_persistent": False,
        "docker_shared_container_key": "",
        "docker_snap_compat": False,
        "docker_extra_args": [],
        "docker_persist_across_processes": False,
    }


def test_confinement_proof_accepts_one_workspace_rw_mount_and_nested_ro_masks(confined_config):
    from tools.approval_workspace_auto import workspace_auto_confinement

    proof = workspace_auto_confinement(confined_config)

    assert proof.proven is True
    assert proof.host_root.endswith("Workspace")


def test_confinement_proof_accepts_empty_regular_file_mask(confined_config, tmp_path):
    from tools.approval_workspace_auto import workspace_auto_confinement

    empty_file = tmp_path / "empty-file-mask"
    empty_file.touch()
    confined_config["docker_volumes"][1] = (
        f"{empty_file}:/workspace/project/.env:ro"
    )

    assert workspace_auto_confinement(confined_config).proven is True


def test_confinement_proof_rejects_nonempty_read_only_host_mount(confined_config, tmp_path):
    from tools.approval_workspace_auto import workspace_auto_confinement

    secret = tmp_path / "secret"
    secret.write_text("host secret", encoding="utf-8")
    confined_config["docker_volumes"][1] = (
        f"{secret}:/workspace/project/.env:ro"
    )

    proof = workspace_auto_confinement(confined_config)

    assert proof.proven is False
    assert "empty mask artifact" in proof.reason


def test_confinement_proof_rejects_nonempty_directory_mask(confined_config, tmp_path):
    from tools.approval_workspace_auto import workspace_auto_confinement

    nonempty_dir = tmp_path / "nonempty-mask"
    nonempty_dir.mkdir()
    (nonempty_dir / "secret").write_text("host secret", encoding="utf-8")
    confined_config["docker_volumes"][1] = (
        f"{nonempty_dir}:/workspace/project/.env:ro"
    )

    proof = workspace_auto_confinement(confined_config)

    assert proof.proven is False
    assert "empty mask artifact" in proof.reason


@pytest.mark.require_symlinks
def test_confinement_proof_rejects_symlink_mask_source(confined_config, tmp_path):
    from tools.approval_workspace_auto import workspace_auto_confinement

    empty_file = tmp_path / "empty-file"
    empty_file.touch()
    linked_mask = tmp_path / "linked-mask"
    linked_mask.symlink_to(empty_file)
    confined_config["docker_volumes"][1] = (
        f"{linked_mask}:/workspace/project/.env:ro"
    )

    proof = workspace_auto_confinement(confined_config)

    assert proof.proven is False
    assert "empty mask artifact" in proof.reason


def test_confinement_proof_requires_exact_canonical_authorized_host_root(confined_config, tmp_path):
    from tools.approval_workspace_auto import workspace_auto_confinement

    other = tmp_path / "Other"
    other.mkdir()
    missing = deepcopy(confined_config)
    missing.pop("workspace_auto_host_root")
    mismatched = deepcopy(confined_config)
    mismatched["workspace_auto_host_root"] = str(other)
    noncanonical = deepcopy(confined_config)
    noncanonical["workspace_auto_host_root"] = str(tmp_path / "Workspace") + "/"

    assert workspace_auto_confinement(missing).proven is False
    assert workspace_auto_confinement(mismatched).proven is False
    assert workspace_auto_confinement(noncanonical).proven is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda c: c.update(env_type="local"), "Docker"),
        (lambda c: c.update(docker_network=True), "network"),
        (lambda c: c.update(docker_mount_cwd_to_workspace=True), "automatic cwd"),
        (lambda c: c.update(container_persistent=True), "persistent"),
        (lambda c: c.update(docker_persist_across_processes=True), "across processes"),
        (lambda c: c.update(docker_shared_container_key="shared"), "shared container"),
        (lambda c: c.update(docker_snap_compat=True), "snap compatibility"),
        (lambda c: c.update(docker_extra_args=["--privileged"]), "extra args"),
        (lambda c: c.update(docker_volumes=[]), "/workspace"),
        (
            lambda c: c.update(docker_volumes=[c["docker_volumes"][0].replace(":rw", ":ro")]),
            "read-write",
        ),
        (
            lambda c: c.update(docker_volumes=[c["docker_volumes"][0].replace(":/workspace:", ":/other:")]),
            "/workspace",
        ),
        (
            lambda c: c["docker_volumes"].append(c["docker_volumes"][0].replace(":/workspace:rw", ":/workspace/tmp:rw")),
            "read-only",
        ),
        (
            lambda c: c.update(docker_volumes=[c["docker_volumes"][0], c["docker_volumes"][1] + ",rw"]),
            "read-only",
        ),
        (
            lambda c: c.update(docker_volumes=[c["docker_volumes"][0], c["docker_volumes"][1].replace("/workspace/project/.env", "/root/.env")]),
            "beneath /workspace",
        ),
    ],
)
def test_confinement_proof_fails_closed(confined_config, mutation, reason):
    from tools.approval_workspace_auto import workspace_auto_confinement

    config = deepcopy(confined_config)
    mutation(config)

    proof = workspace_auto_confinement(config)

    assert proof.proven is False
    assert reason.lower() in proof.reason.lower()


@pytest.mark.parametrize(
    "command",
    [
        "touch marker",
        "mkdir -p output && cp README.md output/copy.md",
        "mv old.txt new.txt; rm -rf build",
        "git add -A && git commit -m checkpoint",
        "git reset --hard HEAD~1",
        "git rebase main",
        "python -c 'from pathlib import Path; Path(\"x\").write_text(\"ok\")'",
        "pytest -q",
        "npm test",
        "make build",
        "rsync -a src/ dst/",
    ],
)
def test_workspace_auto_proven_sandbox_auto_approves_local_mutations(command):
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []


def test_workspace_auto_auto_approves_quoted_heredoc_content_as_local_data():
    command = """cd /workspace/project && cat > draft.md <<'EOF'
เรียน ผู้มีส่วนได้ส่วนเสีย;
- Scope: `AGENTS.md` / `CLAUDE.md` (ยังเป็น TBD)
- Risk A | Risk B & mitigation
EOF
ls -la
"""
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []


def test_workspace_auto_still_prompts_for_remote_command_after_heredoc():
    command = """cat > draft.md <<'EOF'
curl https://example.com in prose is data, not a command
EOF
curl https://example.com
"""
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1


def test_workspace_auto_fails_closed_for_unterminated_heredoc():
    prompts = []

    result = approval.check_all_command_guards(
        "cat > draft.md <<'EOF'\nunfinished",
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1


def test_workspace_auto_does_not_treat_quoted_heredoc_text_as_an_opener():
    prompts = []

    result = approval.check_all_command_guards(
        "printf '%s\\n' '<<EOF'\ncurl https://example.com",
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1


@pytest.mark.parametrize(
    "command",
    [
        "git push origin main",
        "git fetch --all",
        "git pull --rebase",
        "git clone https://example.com/repo.git",
        "git submodule update --init",
        "git remote update",
        "git lfs pull",
        "git archive --remote=origin HEAD",
        "curl https://example.com",
        "sudo -u root curl https://example.com",
        "env -u TOKEN git push origin main",
        "printf '%s\\n' https://example.com | xargs -n 1 curl",
        "wget https://example.com/file",
        "ssh host.example pwd",
        "scp output.txt host.example:/tmp/",
        "rsync output/ host.example:/tmp/output/",
        "gh pr create --fill",
        "npm install lodash",
        "pnpm add react",
        "npx prettier --version",
        "pip install requests",
        "pipx install black",
        "uvx ruff check .",
        "docker pull python:3.14",
        "ansible-playbook deploy.yaml",
        "uv add httpx",
        "cargo publish",
        "brew install jq",
        "apt-get update",
        "aws s3 cp artifact s3://bucket/artifact",
        "rclone copy artifact remote:bucket",
        "s3cmd put artifact s3://bucket/artifact",
        "gcloud run deploy service",
        "kubectl apply -f deploy.yaml",
        "vercel deploy --prod",
        "himalaya message send",
        "bash -lc 'curl https://example.com'",
    ],
)
def test_workspace_auto_prompts_once_for_identifiable_remote_or_network_commands(command):
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1
    assert prompts[0][1] == {"allow_permanent": False, "allow_session": False}


@pytest.mark.parametrize(
    "command",
    [
        "eval 'curl https://example.com'",
        "cmd=curl; $cmd https://example.com",
        "bash -c \"$COMMAND\"",
        "bash -c $'\\x63url https://example.com'",
        "git -c alias.ship='push origin main' ship",
        "git -calias.ship='!curl https://example.com' ship",
        "git -calias.status='!curl https://example.com' status",
        "python -c 'import urllib.request; urllib.request.urlopen(\"https://example.com\")'",
        "./unknown-script.sh",
        "totally-unknown-command --flag",
    ],
)
def test_workspace_auto_prompts_for_dynamic_or_ambiguous_command_carriers(command):
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1
    assert prompts[0][1] == {"allow_permanent": False, "allow_session": False}


@pytest.mark.parametrize(
    "command",
    [
        "printf '%s\\n' hello > output.txt",
        "cat > note.txt <<'EOF'\nhello\nEOF",
        "mkdir -p build && cp README.md build/README.md",
        "git add -A && git commit -m checkpoint",
        "pytest -q",
        "npm test",
        "make build",
    ],
)
def test_workspace_auto_keeps_known_local_commands_automatic(command):
    prompts = []

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result == {"approved": True, "message": None}
    assert prompts == []


def test_workspace_auto_unproven_sandbox_prompts_even_for_local_command():
    prompts = []

    result = approval.check_all_command_guards(
        "git status --short",
        "docker",
        has_host_access=True,
        workspace_auto_confined=False,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1
    assert prompts[0][1] == {"allow_permanent": False, "allow_session": False}


def test_workspace_auto_remote_approval_is_exactly_one_shot(_isolated_workspace_auto_ask):
    session_key = _isolated_workspace_auto_ask
    approval.approve_session(session_key, "git push")
    approval.approve_permanent("git push")
    before_session = set(approval._session_approved[session_key])
    before_permanent = set(approval._permanent_approved)
    choices = iter(["once", "deny"])
    prompts = []

    def decide(*args, **kwargs):
        prompts.append((args, kwargs))
        return next(choices)

    first = approval.check_all_command_guards(
        "git push origin main", "docker", has_host_access=True,
        workspace_auto_confined=True, approval_callback=decide,
    )
    second = approval.check_all_command_guards(
        "git push origin main", "docker", has_host_access=True,
        workspace_auto_confined=True, approval_callback=decide,
    )

    assert first["approved"] is True
    assert second["approved"] is False
    assert len(prompts) == 2
    assert all(item[1] == {"allow_permanent": False, "allow_session": False} for item in prompts)
    assert approval._session_approved[session_key] == before_session
    assert approval._permanent_approved == before_permanent


@pytest.mark.parametrize(
    ("command", "confined"),
    [("git push origin main", True), ("git status --short", False)],
)
def test_workspace_auto_fails_closed_when_a_required_prompt_has_no_human_surface(
    command, confined, monkeypatch
):
    monkeypatch.setattr(approval, "_presence", lambda _callback=None: (None, False, False, False))

    result = approval.check_all_command_guards(
        command,
        "docker",
        has_host_access=True,
        workspace_auto_confined=confined,
    )

    assert result["approved"] is False
    assert result["outcome"] == "blocked"
    assert "no approval surface" in result["message"].lower()


def test_workspace_auto_proven_local_command_needs_no_human_surface(monkeypatch):
    monkeypatch.setattr(approval, "_presence", lambda _callback=None: (None, False, False, False))

    result = approval.check_all_command_guards(
        "git commit -am checkpoint",
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
    )

    assert result == {"approved": True, "message": None}


def test_workspace_auto_hardline_floor_still_blocks_without_prompt():
    prompts = []

    result = approval.check_all_command_guards(
        "rm -rf /",
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
        approval_callback=lambda *args, **kwargs: prompts.append((args, kwargs)) or "once",
    )

    assert result["approved"] is False
    assert "unconditional blocklist" in result["message"]
    assert prompts == []


def test_workspace_auto_proven_sandbox_still_prompts_arbitrary_execute_code(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        approval,
        "_presence",
        lambda _callback=None: (
            lambda *args, **kwargs: prompts.append((args, kwargs)) or "deny",
            False,
            False,
            True,
        ),
    )

    result = approval.check_execute_code_guard(
        "from pathlib import Path; Path('marker').write_text('ok')",
        "docker",
        has_host_access=True,
        workspace_auto_confined=True,
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1


def test_workspace_auto_unproven_sandbox_prompts_execute_code_once(monkeypatch):
    prompts = []

    def decide(*args, **kwargs):
        prompts.append((args, kwargs))
        return "deny"

    monkeypatch.setattr(approval, "_presence", lambda _callback=None: (decide, False, False, True))

    result = approval.check_execute_code_guard(
        "print('local')",
        "docker",
        has_host_access=True,
        workspace_auto_confined=False,
    )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert len(prompts) == 1
    assert prompts[0][1] == {"allow_permanent": False, "allow_session": False}


def test_terminal_guard_consumes_actual_environment_attestation(confined_config, monkeypatch):
    from tools import terminal_tool
    from tools.approval_workspace_auto import attest_workspace_auto_environment

    captured = {}

    def guard(command, env_type, **kwargs):
        captured.update(command=command, env_type=env_type, **kwargs)
        return {"approved": True, "message": None}

    monkeypatch.setattr(terminal_tool, "_check_all_guards_impl", guard)
    env = type("DockerEnvironment", (), {"env_type": "docker"})()
    attest_workspace_auto_environment(
        env,
        confined_config,
        effective_run_args=[
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network=none",
            "-v", confined_config["docker_volumes"][0],
            "-v", confined_config["docker_volumes"][1],
            "--tmpfs", "/root:rw,exec,size=1g",
        ],
    )

    verdict = terminal_tool._run_approval_guards(
        "touch marker", "docker", confined_config, env=env, force=False,
    )

    assert verdict.approved_run is False
    assert captured["workspace_auto_confined"] is True
    assert captured["has_host_access"] is True


def test_terminal_guard_rejects_cached_environment_without_attestation(confined_config, monkeypatch):
    from tools import terminal_tool

    captured = {}

    def guard(command, env_type, **kwargs):
        captured.update(command=command, env_type=env_type, **kwargs)
        return {"approved": True, "message": None}

    monkeypatch.setattr(terminal_tool, "_check_all_guards_impl", guard)
    stale_env = type("DockerEnvironment", (), {"env_type": "docker"})()

    terminal_tool._run_approval_guards(
        "touch marker", "docker", confined_config, env=stale_env, force=False,
    )

    assert captured["workspace_auto_confined"] is False


def test_environment_attestation_detects_copy_to_a_different_cached_instance(confined_config):
    from tools.approval_workspace_auto import (
        attest_workspace_auto_environment,
        is_workspace_auto_environment_proven,
    )

    env = type("DockerEnvironment", (), {"env_type": "docker"})()
    other = type("DockerEnvironment", (), {"env_type": "docker"})()
    attest_workspace_auto_environment(
        env,
        confined_config,
        effective_run_args=[
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network=none",
            "-v", confined_config["docker_volumes"][0],
            "--tmpfs", "/root:rw,exec,size=1g",
        ],
    )
    other._workspace_auto_attestation = env._workspace_auto_attestation

    assert is_workspace_auto_environment_proven(env) is True
    assert is_workspace_auto_environment_proven(other) is False


def test_environment_attestation_rejects_persistent_root_bind(confined_config, tmp_path):
    from tools.approval_workspace_auto import (
        attest_workspace_auto_environment,
        is_workspace_auto_environment_proven,
    )

    env = type("DockerEnvironment", (), {"env_type": "docker"})()
    root = tmp_path / "persistent-root"
    root.mkdir()
    attest_workspace_auto_environment(
        env,
        confined_config,
        effective_run_args=[
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network=none",
            "-v", confined_config["docker_volumes"][0],
            "-v", f"{root}:/root:rw",
        ],
    )

    assert is_workspace_auto_environment_proven(env) is False


def test_environment_attestation_rejects_generated_read_only_host_mount(
    confined_config,
):
    from tools.approval_workspace_auto import (
        attest_workspace_auto_environment,
        is_workspace_auto_environment_proven,
    )

    env = type("DockerEnvironment", (), {"env_type": "docker"})()
    attest_workspace_auto_environment(
        env,
        confined_config,
        effective_run_args=[
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network=none",
            "-v", confined_config["docker_volumes"][0],
            "-v", "/etc/hosts:/root/leak:ro",
        ],
    )

    assert is_workspace_auto_environment_proven(env) is False


def test_environment_attestation_accepts_generated_empty_mask(confined_config, tmp_path):
    from tools.approval_workspace_auto import (
        attest_workspace_auto_environment,
        is_workspace_auto_environment_proven,
    )

    empty_mask = tmp_path / "generated-empty-mask"
    empty_mask.mkdir()
    env = type("DockerEnvironment", (), {"env_type": "docker"})()
    attest_workspace_auto_environment(
        env,
        confined_config,
        effective_run_args=[
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network=none",
            "-v", confined_config["docker_volumes"][0],
            "-v", f"{empty_mask}:/workspace/project/generated-mask:ro",
        ],
    )

    assert is_workspace_auto_environment_proven(env) is True


# The backend must carry policy even before the root config is valid, so failed
# workspace_auto setups do not regain automatic host support mounts.
def test_docker_builder_carries_active_workspace_auto_security_posture(monkeypatch):
    from tools import approval_context, terminal_tool
    from tools.terminal_tool_backends import _build_docker_env

    seen = {}
    monkeypatch.setattr(
        "tools.terminal_tool_backends._DockerEnvironment",
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "workspace_auto")
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda _config: None)
    monkeypatch.setattr(terminal_tool, "_docker_session_isolation_enabled", lambda: False)

    _build_docker_env(
        image="python:3.11",
        cwd="/root",
        timeout=5,
        cc={},
        task_id="default",
        host_cwd=None,
    )

    assert seen["workspace_auto_security"] is True
