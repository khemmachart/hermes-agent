"""Legacy /v1/chat/completions streams surface approval requests that the client can resolve
through ``POST /v1/runs/{completion_id}/approval`` (#51871)."""
import asyncio
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter


def _approval_agent(decisions):
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    def run_conversation(**_kw):
        # The real blocking wait the terminal guard uses, reached through the turn's contextvar.
        from tools.approval import _gateway_notify_cb
        from tools.approval_context import get_current_session_key
        from tools.approval_gateway_wait import _await_gateway_decision
        key = get_current_session_key(default="")
        decisions.append(_await_gateway_decision(
            key, _gateway_notify_cb(key), {"command": "rm -rf /tmp/x", "description": "dangerous",
                                           "pattern_key": "rm", "pattern_keys": ["rm"],
                                           "allow_session": False, "allow_permanent": False}))
        return {"final_response": "ok", "messages": []}
    agent.run_conversation.side_effect = run_conversation
    return agent


@pytest.mark.asyncio
async def test_streamed_completion_approval_resolves_via_runs_endpoint(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    gateway_run = ModuleType("gateway.run")
    setattr(gateway_run, "_redact_approval_command", lambda command: str(command or ""))
    monkeypatch.setitem(sys.modules, "gateway.run", gateway_run)
    # Bound the agent thread's blocking wait so a regression fails fast instead of holding teardown.
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 5)
    adapter, decisions = _make_adapter(), []
    adapter._memory_sessions._max_size = 1
    adapter._memory_sessions._idle_ttl_secs = 60
    app = _create_app(adapter)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", side_effect=lambda **_k: _approval_agent(decisions)):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200
            async def _first_approval():
                event = None
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if line == "event: approval.request":
                        event = "next"
                    elif event == "next" and line.startswith("data: "):
                        return json.loads(line[6:])
                return None
            event = await asyncio.wait_for(_first_approval(), 10)
            assert event and event["run_id"].startswith("chatcmpl")
            assert event["command"] == "rm -rf /tmp/x"
            assert event["pattern_key"] == "rm" and event["pattern_keys"] == ["rm"]
            assert event["choices"] == ["once", "deny"]
            assert isinstance(event["request_id"], str) and event["request_id"]
            approval = await cli.post(f"/v1/runs/{event['run_id']}/approval", json={
                "choice": "once", "request_id": event["request_id"],
            })
            assert approval.status == 200, await approval.text()
            assert (await approval.json())["request_id"] == event["request_id"]
            assert "[DONE]" in (await asyncio.wait_for(resp.text(), 10))
    assert decisions and decisions[0]["choice"] == "once"
    assert event["run_id"] not in adapter._run_approval_sessions
    assert adapter._run_statuses[event["run_id"]]["status"] == "completed"
