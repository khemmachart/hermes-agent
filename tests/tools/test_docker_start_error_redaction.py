import json
import logging
import subprocess

import pytest

from tools.environments import docker as docker_env
from tools.environments.base import EnvironmentStartError


_DIAGNOSTIC_CODE = "HERMES_DOCKER_SANDBOX_START_FAILED"
_SENSITIVE_RUN_COMMAND = [
    "/opt/homebrew/bin/podman",
    "run",
    "-d",
    "-v",
    "/Users/alice/.hermes/masked-secrets/masked-openai-api-key:/run/secrets/openai_api_key:ro",
    "private-image:latest",
    "sleep",
    "infinity",
]
_SENSITIVE_MARKERS = (
    "/Users/alice",
    "/run/secrets",
    "masked-openai-api-key",
    "private-image:latest",
    "sleep",
    "infinity",
)


def _start_only_environment(monkeypatch):
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._docker_exe = "/opt/homebrew/bin/podman"
    env._run_env_values = {}
    monkeypatch.setattr(
        env,
        "_run_command",
        lambda name, workdir: list(_SENSITIVE_RUN_COMMAND),
    )
    monkeypatch.setattr(env, "_docker_client_env", lambda _values: None)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    return env


def _assert_no_start_metadata(text):
    for marker in _SENSITIVE_MARKERS:
        assert marker not in text


def test_called_process_error_is_redacted_from_exception_and_logs(monkeypatch, caplog):
    env = _start_only_environment(monkeypatch)

    def fail(command, **_kwargs):
        raise subprocess.CalledProcessError(
            125,
            command,
            output="",
            stderr="mount source /Users/alice/.hermes/masked-secrets/masked-openai-api-key rejected",
        )

    monkeypatch.setattr(docker_env, "run_capture", fail)
    caplog.set_level(logging.DEBUG, logger=docker_env.__name__)

    with pytest.raises(RuntimeError) as caught:
        env._docker_run("/workspace")

    expected = (
        "Docker sandbox failed to start "
        f"(exit code 125; diagnostic code {_DIAGNOSTIC_CODE})."
    )
    assert str(caught.value) == expected
    assert getattr(caught.value, "exit_code", None) == 125
    assert getattr(caught.value, "diagnostic_code", None) == _DIAGNOSTIC_CODE
    _assert_no_start_metadata(str(caught.value))
    _assert_no_start_metadata(caplog.text)


def test_timeout_expired_is_redacted_from_exception_and_logs(monkeypatch, caplog):
    env = _start_only_environment(monkeypatch)

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(
            command,
            120,
            output="pulling private-image:latest",
            stderr="reading /Users/alice/.hermes/masked-secrets/masked-openai-api-key",
        )

    monkeypatch.setattr(docker_env, "run_capture", timeout)
    caplog.set_level(logging.DEBUG, logger=docker_env.__name__)

    with pytest.raises(RuntimeError) as caught:
        env._docker_run("/workspace")

    expected = (
        "Docker sandbox failed to start "
        f"(exit code 124; diagnostic code {_DIAGNOSTIC_CODE})."
    )
    assert str(caught.value) == expected
    assert getattr(caught.value, "exit_code", None) == 124
    assert getattr(caught.value, "diagnostic_code", None) == _DIAGNOSTIC_CODE
    _assert_no_start_metadata(str(caught.value))
    _assert_no_start_metadata(caplog.text)


def test_terminal_result_preserves_sanitized_start_failure_classification(monkeypatch, caplog):
    import tools.terminal_tool as terminal_tool

    failure = EnvironmentStartError("Docker", 125, _DIAGNOSTIC_CODE)
    monkeypatch.setattr(terminal_tool, "_plan_execution", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        terminal_tool,
        "_acquire_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_loop.record_terminal_backend",
        lambda _plan, result, **_kwargs: result,
    )
    caplog.set_level(logging.DEBUG, logger=terminal_tool.__name__)

    result = json.loads(terminal_tool.terminal_tool("cat /Users/alice/private-command"))

    assert result == {
        "output": "",
        "exit_code": 125,
        "error": str(failure),
        "status": "error",
        "diagnostic_code": _DIAGNOSTIC_CODE,
    }
    _assert_no_start_metadata(json.dumps(result))
    _assert_no_start_metadata(caplog.text)


def test_full_environment_start_failure_never_logs_mount_metadata(monkeypatch, caplog):
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/opt/homebrew/bin/podman")
    monkeypatch.setattr(docker_env, "_readonly_skill_mount_args", lambda: [])
    monkeypatch.setattr(docker_env, "_image_uses_init_entrypoint", lambda *_args: False)
    monkeypatch.setattr(docker_env, "_egress_proxy_args_for_docker", lambda: ([], {}, []))

    def run_capture(command, **_kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, stdout="podman version", stderr="")
        if command[1] == "ps":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[1] == "run":
            raise subprocess.CalledProcessError(125, command, stderr="invalid mount")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env, "run_capture", run_capture)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    caplog.set_level(logging.DEBUG, logger=docker_env.__name__)

    with pytest.raises(EnvironmentStartError):
        docker_env.DockerEnvironment(
            image="private-image:latest",
            cwd="/workspace",
            timeout=60,
            cpu=0,
            memory=0,
            disk=0,
            persistent_filesystem=False,
            task_id="redaction-test",
            volumes=[
                "/Users/alice/.hermes/masked-secrets/masked-openai-api-key:"
                "/run/secrets/openai_api_key:ro"
            ],
            persist_across_processes=False,
        )

    _assert_no_start_metadata(caplog.text)
