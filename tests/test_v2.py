"""Offline unit tests for VersClient, AsyncVersClient, and VersSdkClient fixes."""

from __future__ import annotations

import base64
import json
from typing import Any
import uuid

import httpx
import pytest

from vers_sdk import (
    APIConnectionTimeoutError,
    AsyncVersClient,
    AuthenticationError,
    NotFoundError,
    RequestOptions,
    VersClient,
    VersOperationError,
    VersSdkClient,
)
from vers_sdk.models import ReadComputerFileParams, WriteComputerFileParams


# ── Helpers ──────────────────────────────────────────────────────────────


def _ndjson(*events: dict[str, Any]) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")


def _make_op(
    op_id: str = "op-1",
    *,
    kind: str = "create",
    status: str = "pending",
    resource_kind: str = "computer",
    resource_id: str = "comp-1",
    cursor: int = 1,
    stage: str | None = "starting",
    result: dict[str, str] | None = None,
    local_result: dict[str, str] | None = None,
    failure: dict[str, str] | None = None,
    cleanup_operation_id: str | None = None,
) -> dict[str, Any]:
    return {
        "operation_id": op_id,
        "kind": kind,
        "status": status,
        "resource": {"kind": resource_kind, "id": resource_id},
        "cursor": cursor,
        "stage": stage,
        "result": result,
        "local_result": local_result,
        "failure": failure,
        "cleanup_operation_id": cleanup_operation_id,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:01Z",
        "completed_at": None if status == "pending" else "2026-01-01T00:00:02Z",
    }


def _make_computer(computer_id: str = "comp-1", state: str = "running") -> dict[str, Any]:
    return {
        "computer_id": computer_id,
        "created_at": "2026-01-01T00:00:00Z",
        "keep_alive": False,
        "mem_size_mib": 1024,
        "owner_id": "owner-1",
        "state": state,
        "vcpu_count": 2,
        "ip": "10.0.0.5",
        "labels": {"env": "test"},
        "parent_checkpoint_id": None,
        "project_id": None,
    }


def _sync_client(
    handler: Any,
    *,
    api_key: str = "test-api-key",
    max_retries: int = 2,
    poll_interval: float = 0.005,
    operation_timeout: float = 5.0,
) -> VersClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return VersClient(
        api_key=api_key,
        base_url="https://api.vers.test",
        max_retries=max_retries,
        poll_interval=poll_interval,
        retry_base_delay=0.001,
        operation_timeout=operation_timeout,
        http_client=http_client,
    )


def _async_client(
    handler: Any,
    *,
    api_key: str = "test-api-key",
    max_retries: int = 2,
    poll_interval: float = 0.005,
    operation_timeout: float = 5.0,
) -> AsyncVersClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return AsyncVersClient(
        api_key=api_key,
        base_url="https://api.vers.test",
        max_retries=max_retries,
        poll_interval=poll_interval,
        retry_base_delay=0.001,
        operation_timeout=operation_timeout,
        http_client=http_client,
    )


class _SyncFailingStream(httpx.SyncByteStream):
    """Yields one chunk then raises ReadError to simulate mid-stream disconnect."""

    def __init__(self, first_chunk: bytes) -> None:
        self._first_chunk = first_chunk

    def __iter__(self):
        yield self._first_chunk
        raise httpx.ReadError("connection reset by peer")


class _AsyncFailingStream(httpx.AsyncByteStream):
    """Yields one chunk then raises ReadError to simulate mid-stream disconnect."""

    def __init__(self, first_chunk: bytes) -> None:
        self._first_chunk = first_chunk

    async def __aiter__(self):
        yield self._first_chunk
        raise httpx.ReadError("connection reset by peer")


# ── Auth, Idempotency-Key & Retries ──────────────────────────────────────


def test_sync_auth_and_idempotency_reuse_across_retries() -> None:
    seen_keys: list[str] = []
    seen_auth: list[str] = []
    seen_x_keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        seen_keys.append(request.headers.get("idempotency-key", ""))
        seen_x_keys.append(request.headers.get("x-idempotency-key"))
        if len(seen_keys) < 3:
            return httpx.Response(503, json={"message": "temporary unavailable"})
        return httpx.Response(202, json=_make_op("op-retry", status="pending"))

    with _sync_client(handler, api_key="secret-123", max_retries=2) as client:
        op = client.create(vcpu_count=2, mem_size_mib=512, wait=False)

    assert op["operation_id"] == "op-retry"
    assert len(seen_keys) == 3
    assert all(a == "Bearer secret-123" for a in seen_auth)
    assert all(x is None for x in seen_x_keys)
    # Exact same valid UUIDv4 reused across all retry attempts
    assert len(set(seen_keys)) == 1
    parsed = uuid.UUID(seen_keys[0])
    assert parsed.version == 4


@pytest.mark.asyncio
async def test_async_auth_and_idempotency_reuse_across_retries() -> None:
    seen_keys: list[str] = []
    seen_auth: list[str] = []
    seen_x_keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        seen_keys.append(request.headers.get("idempotency-key", ""))
        seen_x_keys.append(request.headers.get("x-idempotency-key"))
        if len(seen_keys) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate limited"})
        if len(seen_keys) == 2:
            raise httpx.ConnectError("transient connect failure")
        return httpx.Response(202, json=_make_op("op-async-retry", status="pending"))

    async with _async_client(handler, api_key="async-secret", max_retries=2) as client:
        op = await client.create(vcpu_count=1, wait=False)

    assert op["operation_id"] == "op-async-retry"
    assert len(seen_keys) == 3
    assert all(a == "Bearer async-secret" for a in seen_auth)
    assert all(x is None for x in seen_x_keys)
    assert len(set(seen_keys)) == 1
    assert uuid.UUID(seen_keys[0]).version == 4


def test_sync_explicit_idempotency_key_and_create_wait() -> None:
    explicit_key = str(uuid.uuid4())
    captured_create_body: dict[str, Any] = {}
    captured_key: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_create_body, captured_key
        if request.method == "POST" and request.url.path == "/api/v2/computers":
            captured_key = request.headers.get("idempotency-key")
            captured_create_body = json.loads(request.content)
            return httpx.Response(202, json=_make_op("op-c1", status="pending", resource_id="comp-42"))
        if request.method == "GET" and request.url.path == "/api/v2/operations/op-c1/events":
            assert request.url.params["follow"] == "true"
            assert request.url.params["cursor"] == "1"
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                content=_ndjson(
                    {"type": "snapshot", "operation": _make_op("op-c1", status="pending", cursor=2, stage="booting", resource_id="comp-42")},
                    {"type": "snapshot", "operation": _make_op("op-c1", status="succeeded", cursor=3, stage="ready", resource_id="comp-42")},
                    {"type": "end", "cursor": 3, "terminal": True},
                ),
            )
        if request.method == "GET" and request.url.path == "/api/v2/computers/comp-42":
            return httpx.Response(200, json=_make_computer("comp-42"))
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    input_cfg = {"vcpu_count": 1, "labels": {"a": "1"}}
    with _sync_client(handler) as client:
        comp = client.create(
            input_cfg,
            vcpu_count=4,
            mem_size_mib=2048,
            fs_size_mib=8192,
            system_image_alias="ubuntu:24.04",
            command=["/bin/app"],
            args=["--port", "8080"],
            env={"FOO": "bar"},
            working_dir="/app",
            labels={"role": "worker"},
            network_service="api",
            project_id="proj-1",
            keep_alive=True,
            remove_on_workload_exit=False,
            idempotency_key=explicit_key,
        )

    assert input_cfg["vcpu_count"] == 1  # input dict not mutated
    assert captured_key == explicit_key
    assert captured_create_body == {
        "vm_config": {
            "vcpu_count": 4,
            "mem_size_mib": 2048,
            "fs_size_mib": 8192,
            "system_image_alias": "ubuntu:24.04",
            "command": ["/bin/app"],
            "args": ["--port", "8080"],
            "env": {"FOO": "bar"},
            "working_dir": "/app",
            "labels": {"role": "worker"},
            "network_service": "api",
        },
        "project_id": "proj-1",
        "keep_alive": True,
        "remove_on_workload_exit": False,
    }
    assert comp["computer_id"] == "comp-42"
    assert comp["state"] == "running"


@pytest.mark.asyncio
async def test_async_create_and_restore_and_get_list() -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/api/v2/computers":
            assert request.headers.get("idempotency-key")
            return httpx.Response(202, json=_make_op("op-r1", status="succeeded", resource_id="comp-restored"))
        if request.method == "GET" and request.url.path == "/api/v2/computers/comp-restored":
            return httpx.Response(200, json=_make_computer("comp-restored"))
        if request.method == "GET" and request.url.path == "/api/v2/computers":
            assert request.url.params["limit"] == "10"
            assert request.url.params["cursor"] == "cur-1"
            return httpx.Response(200, json={"computers": [_make_computer("comp-restored")], "next_cursor": None})
        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    async with _async_client(handler) as client:
        comp = await client.restore("ckpt-99", project_id="proj-2", keep_alive=True, labels={"name": "r1"})
        assert comp["computer_id"] == "comp-restored"

        fetched = await client.get("comp-restored")
        assert fetched["computer_id"] == "comp-restored"

        listed = await client.list(limit=10, cursor="cur-1")
        assert len(listed["computers"]) == 1

    assert calls[0] == (
        "POST",
        "/api/v2/computers",
        {
            "source": {"type": "checkpoint", "checkpoint_id": "ckpt-99"},
            "project_id": "proj-2",
            "keep_alive": True,
            "labels": {"name": "r1"},
        },
    )


# ── Exec & Exec Stream ───────────────────────────────────────────────────


def test_sync_exec_rejects_bare_string() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP should not be called on invalid command")

    with _sync_client(handler) as client:
        with pytest.raises(TypeError, match="sequence of argv strings"):
            client.exec("comp-1", "echo hello")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="sequence of argv strings"):
            client.exec_stream("comp-1", "echo hello")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="sequence of argv strings"):
            client.create(command="echo hello")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must not be empty"):
            client.exec("comp-1", [])
        with pytest.raises(TypeError, match="must be str"):
            client.exec("comp-1", ["echo", 123])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_async_exec_and_reject_bare_string() -> None:
    captured_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        assert request.method == "POST"
        assert request.url.path == "/api/v2/computers/comp-1/exec"
        captured_body = json.loads(request.content)
        return httpx.Response(200, json={"exit_code": 0, "stdout": "hi\n", "stderr": "", "exec_id": "ex-1"})

    async with _async_client(handler) as client:
        with pytest.raises(TypeError, match="sequence of argv strings"):
            await client.exec("comp-1", "ls -la")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="sequence of argv strings"):
            client.exec_stream("comp-1", "ls -la")  # type: ignore[arg-type]

        res = await client.exec(
            "comp-1",
            ("sh", "-lc", "echo hi"),
            working_dir="/tmp",
            env={"K": "V"},
            stdin="in",
            timeout_secs=30,
            keep_background=True,
            exec_id="ex-1",
        )

    assert res == {"exit_code": 0, "stdout": "hi\n", "stderr": "", "exec_id": "ex-1"}
    assert captured_body == {
        "command": ["sh", "-lc", "echo hi"],
        "working_dir": "/tmp",
        "env": {"K": "V"},
        "stdin": "in",
        "timeout_secs": 30,
        "keep_background": True,
        "exec_id": "ex-1",
    }


def test_sync_exec_stream_ndjson() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v2/computers/comp-1/exec/stream"
        assert request.headers.get("accept") == "application/x-ndjson"
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(
                {"type": "chunk", "cursor": 1, "exec_id": "ex-9", "stream": "stdout", "data_b64": base64.b64encode(b"hello\n").decode()},
                {"type": "chunk", "cursor": 2, "exec_id": "ex-9", "stream": "stderr", "data_b64": base64.b64encode(b"warn\n").decode()},
                {"type": "exit", "cursor": 3, "exec_id": "ex-9", "exit_code": 0, "infrastructure_error": None},
            ),
        )

    with _sync_client(handler) as client:
        events = list(client.exec_stream("comp-1", ["sh", "-lc", "echo hello"]))

    assert len(events) == 3
    assert events[0]["type"] == "chunk"
    assert base64.b64decode(events[0]["data_b64"]) == b"hello\n"
    assert events[1]["stream"] == "stderr"
    assert events[2] == {"type": "exit", "cursor": 3, "exec_id": "ex-9", "exit_code": 0, "infrastructure_error": None}


@pytest.mark.asyncio
async def test_async_exec_stream_ndjson() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v2/computers/comp-1/exec/stream"
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(
                {"type": "chunk", "cursor": 1, "exec_id": "ex-10", "stream": "stdout", "data_b64": base64.b64encode(b"ok").decode()},
                {"type": "exit", "cursor": 2, "exec_id": "ex-10", "exit_code": 0, "infrastructure_error": None},
            ),
        )

    async with _async_client(handler) as client:
        events = [ev async for ev in client.exec_stream("comp-1", ["echo", "ok"])]

    assert len(events) == 2
    assert events[0]["type"] == "chunk"
    assert events[1]["exit_code"] == 0


# ── Branch, Sleep, Wake, Delete ──────────────────────────────────────────


def test_sync_branch_str_and_dict_sources() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v2/branches"
        bodies.append(json.loads(request.content))
        return httpx.Response(
            201,
            json={
                "computers": [
                    {"computer_id": "b-1", "ip": "10.0.0.10", "volume_backend": "cowfswal", "name": "b-1"}
                ],
                "error": None,
            },
        )

    with _sync_client(handler) as client:
        r1 = client.branch("comp-1", count=2, project_id="proj-1", keep_paused=True, skip_wait_boot=True)
        r2 = client.branch({"type": "checkpoint", "checkpoint_id": "ckpt-7"}, count=1)

    assert r1["computers"][0]["computer_id"] == "b-1"
    assert r2["computers"][0]["computer_id"] == "b-1"
    assert bodies[0] == {
        "source": {
            "type": "computer",
            "computer_id": "comp-1",
            "keep_paused": True,
            "skip_wait_boot": True,
        },
        "count": 2,
        "project_id": "proj-1",
    }
    assert bodies[1] == {
        "source": {"type": "checkpoint", "checkpoint_id": "ckpt-7"},
        "count": 1,
    }


@pytest.mark.asyncio
async def test_async_sleep_wake_and_delete() -> None:
    seen: list[tuple[str, str, bytes, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), request.content, request.headers.get("idempotency-key")))
        if request.url.path == "/api/v2/computers/comp-1/sleep":
            assert request.content == b""
            assert "content-type" not in {k.lower() for k in request.headers.keys()}
            assert request.url.params["skip_wait_boot"] == "true"
            return httpx.Response(204)
        if request.url.path == "/api/v2/computers/comp-1/wake":
            assert request.content == b""
            assert "content-type" not in {k.lower() for k in request.headers.keys()}
            return httpx.Response(200, json={"ip": "10.0.0.55"})
        if request.method == "DELETE" and request.url.path == "/api/v2/computers/comp-1":
            assert request.headers.get("idempotency-key")
            assert request.url.params["skip_wait_boot"] == "true"
            return httpx.Response(202, json=_make_op("op-del", kind="delete", status="pending", resource_id="comp-1"))
        if request.url.path == "/api/v2/operations/op-del/events":
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                content=_ndjson(
                    {"type": "snapshot", "operation": _make_op("op-del", kind="delete", status="succeeded", cursor=2, resource_id="comp-1")},
                    {"type": "end", "cursor": 2, "terminal": True},
                ),
            )
        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    async with _async_client(handler) as client:
        assert await client.sleep("comp-1", skip_wait_boot=True) is None
        woken = await client.wake("comp-1")
        assert woken == {"ip": "10.0.0.55"}
        deleted = await client.delete("comp-1", skip_wait_boot=True)
        assert deleted["status"] == "succeeded"


# ── Files (Binary & Text) ────────────────────────────────────────────────


def test_sync_read_and_write_files() -> None:
    binary_payload = b"\x00\x01\x02\xff\xfehello\x80"
    stored: dict[str, bytes] = {}
    captured_write: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/api/v2/computers/comp-1/files":
            assert request.headers.get("content-type") == "application/octet-stream"
            path = request.url.params["path"]
            captured_write["path"] = path
            captured_write["mode"] = request.url.params.get("mode")
            captured_write["create_dirs"] = request.url.params.get("create_dirs")
            stored[path] = request.content
            return httpx.Response(204)
        if request.method == "GET" and request.url.path == "/api/v2/computers/comp-1/files":
            path = request.url.params["path"]
            return httpx.Response(
                200,
                json={"content_b64": base64.b64encode(stored[path]).decode("ascii")},
            )
        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    with _sync_client(handler) as client:
        client.write_file("comp-1", "/tmp/bin.dat", binary_payload, mode=0o600, create_dirs=True)
        assert captured_write == {"path": "/tmp/bin.dat", "mode": str(0o600), "create_dirs": "true"}
        assert client.read_file("comp-1", "/tmp/bin.dat") == binary_payload

        client.write_text("comp-1", "/tmp/hi.txt", "café\n")
        assert client.read_text("comp-1", "/tmp/hi.txt") == "café\n"


@pytest.mark.asyncio
async def test_async_read_and_write_files() -> None:
    binary_payload = b"\x89PNG\r\n\x1a\n\x00\x00\xff"
    stored: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/api/v2/computers/comp-1/files":
            assert request.headers.get("content-type") == "application/octet-stream"
            stored[request.url.params["path"]] = request.content
            return httpx.Response(204)
        if request.method == "GET" and request.url.path == "/api/v2/computers/comp-1/files":
            data = stored[request.url.params["path"]]
            return httpx.Response(200, json={"content_b64": base64.b64encode(data).decode("ascii")})
        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    async with _async_client(handler) as client:
        await client.write_file("comp-1", "/tmp/img.png", binary_payload, create_dirs=True)
        assert await client.read_file("comp-1", "/tmp/img.png") == binary_payload
        await client.write_text("comp-1", "/tmp/msg.txt", "async text")
        assert await client.read_text("comp-1", "/tmp/msg.txt") == "async text"


# ── Checkpoint: local vs durable ─────────────────────────────────────────


def test_sync_checkpoint_local_vs_durable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v2/computers/comp-1/checkpoints":
            assert request.headers.get("idempotency-key")
            return httpx.Response(
                202,
                json=_make_op("op-ckpt", kind="checkpoint", status="pending", cursor=1, stage="capturing"),
            )
        if request.method == "GET" and request.url.path == "/api/v2/operations/op-ckpt/events":
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                content=_ndjson(
                    {
                        "type": "snapshot",
                        "operation": _make_op(
                            "op-ckpt",
                            kind="checkpoint",
                            status="pending",
                            cursor=2,
                            stage="uploading",
                            local_result={"kind": "checkpoint", "id": "ckpt-1"},
                            result=None,
                        ),
                    },
                    {
                        "type": "snapshot",
                        "operation": _make_op(
                            "op-ckpt",
                            kind="checkpoint",
                            status="succeeded",
                            cursor=3,
                            stage="published",
                            local_result={"kind": "checkpoint", "id": "ckpt-1"},
                            result={"kind": "checkpoint", "id": "ckpt-1"},
                        ),
                    },
                    {"type": "end", "cursor": 3, "terminal": True},
                ),
            )
        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    with _sync_client(handler) as client:
        local_op = client.checkpoint("comp-1", name="snap", wait_for="local")
        assert local_op["status"] == "pending"
        assert local_op["cursor"] == 2
        assert local_op["local_result"] == {"kind": "checkpoint", "id": "ckpt-1"}
        assert local_op["result"] is None

        durable_op = client.checkpoint("comp-1", name="snap", wait_for="durable")
        assert durable_op["status"] == "succeeded"
        assert durable_op["cursor"] == 3
        assert durable_op["result"] == {"kind": "checkpoint", "id": "ckpt-1"}


@pytest.mark.asyncio
async def test_async_checkpoint_local_vs_durable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v2/computers/comp-1/checkpoints":
            return httpx.Response(
                202,
                json=_make_op("op-ckpt-a", kind="checkpoint", status="pending", cursor=1),
            )
        if request.method == "GET" and request.url.path == "/api/v2/operations/op-ckpt-a/events":
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                content=_ndjson(
                    {
                        "type": "snapshot",
                        "operation": _make_op(
                            "op-ckpt-a",
                            kind="checkpoint",
                            status="pending",
                            cursor=2,
                            local_result={"kind": "checkpoint", "id": "ckpt-a"},
                            result=None,
                        ),
                    },
                    {
                        "type": "snapshot",
                        "operation": _make_op(
                            "op-ckpt-a",
                            kind="checkpoint",
                            status="succeeded",
                            cursor=3,
                            local_result={"kind": "checkpoint", "id": "ckpt-a"},
                            result={"kind": "checkpoint", "id": "ckpt-a"},
                        ),
                    },
                    {"type": "end", "cursor": 3, "terminal": True},
                ),
            )
        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    async with _async_client(handler) as client:
        local_op = await client.checkpoint("comp-1", wait_for="local")
        assert local_op["status"] == "pending"
        assert local_op["local_result"]["id"] == "ckpt-a"

        durable_op = await client.checkpoint("comp-1", wait_for="durable")
        assert durable_op["status"] == "succeeded"
        assert durable_op["result"]["id"] == "ckpt-a"


# ── Operation Streaming: Gap, Cursor Resumption, Polling Fallback, Errors ──


def test_sync_wait_for_operation_gap_and_cursor_resumption() -> None:
    event_cursors: list[str | None] = []
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        assert request.url.path == "/api/v2/operations/op-gap/events"
        assert request.url.params["follow"] == "true"
        event_cursors.append(request.url.params.get("cursor"))
        call_count += 1
        if call_count == 1:
            # First follow window: snapshot at 1, then gap (without delivering 4), then disconnect!
            chunk = _ndjson(
                {"type": "snapshot", "operation": _make_op("op-gap", status="pending", cursor=1, stage="queued")},
                {"type": "gap", "requested_cursor": 1, "available_cursor": 4},
            )
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=_SyncFailingStream(chunk))
        if call_count == 2:
            # Reconnected at cursor=1 (NOT 4 because gap snapshot wasn't delivered!), gets gap + snapshot(4) + non-terminal end
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                content=_ndjson(
                    {"type": "gap", "requested_cursor": 1, "available_cursor": 4},
                    {"type": "snapshot", "operation": _make_op("op-gap", status="pending", cursor=4, stage="booting")},
                    {"type": "end", "cursor": 4, "terminal": False},
                ),
            )
        # Reconnected at cursor=4, gets final succeeded snapshot
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(
                {"type": "snapshot", "operation": _make_op("op-gap", status="succeeded", cursor=5, stage="ready")},
                {"type": "end", "cursor": 5, "terminal": True},
            ),
        )

    with _sync_client(handler) as client:
        op = client.wait_for_operation("op-gap")

    assert op["status"] == "succeeded"
    assert op["cursor"] == 5
    assert event_cursors == [None, "1", "4"]


@pytest.mark.asyncio
async def test_async_wait_for_operation_gap_and_cursor_resumption() -> None:
    event_cursors: list[str | None] = []
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        assert request.url.path == "/api/v2/operations/op-gap-a/events"
        event_cursors.append(request.url.params.get("cursor"))
        call_count += 1
        if call_count == 1:
            chunk = _ndjson(
                {"type": "snapshot", "operation": _make_op("op-gap-a", status="pending", cursor=2)},
            )
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=_AsyncFailingStream(chunk))
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(
                {"type": "gap", "requested_cursor": 2, "available_cursor": 6},
                {"type": "snapshot", "operation": _make_op("op-gap-a", status="succeeded", cursor=6)},
                {"type": "end", "cursor": 6, "terminal": True},
            ),
        )

    async with _async_client(handler) as client:
        op = await client.wait_for_operation("op-gap-a")

    assert op["status"] == "succeeded"
    assert op["cursor"] == 6
    assert event_cursors == [None, "2"]


def test_sync_wait_for_operation_polling_fallback() -> None:
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path == "/api/v2/operations/op-poll/events":
            return httpx.Response(503, json={"message": "events unavailable"})
        if request.url.path == "/api/v2/operations/op-poll":
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(200, json=_make_op("op-poll", status="pending", cursor=1))
            return httpx.Response(200, json=_make_op("op-poll", status="succeeded", cursor=2))
        raise AssertionError(f"Unexpected: {request.url}")

    with _sync_client(handler, poll_interval=0.001) as client:
        op = client.wait_for_operation("op-poll")

    assert op["status"] == "succeeded"
    assert poll_count == 2


@pytest.mark.asyncio
async def test_async_wait_for_operation_polling_fallback_on_404_events() -> None:
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path == "/api/v2/operations/op-poll-a/events":
            return httpx.Response(404, json={"message": "no events endpoint"})
        if request.url.path == "/api/v2/operations/op-poll-a":
            poll_count += 1
            if poll_count < 2:
                return httpx.Response(200, json=_make_op("op-poll-a", status="pending", cursor=1))
            return httpx.Response(200, json=_make_op("op-poll-a", status="succeeded", cursor=2))
        raise AssertionError(f"Unexpected: {request.url}")

    async with _async_client(handler, poll_interval=0.001) as client:
        op = await client.wait_for_operation("op-poll-a")

    assert op["status"] == "succeeded"
    assert poll_count == 2


@pytest.mark.parametrize("terminal_status", ["failed", "quarantined"])
def test_sync_wait_for_operation_raises_vers_operation_error(terminal_status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(
                {
                    "type": "snapshot",
                    "operation": _make_op(
                        "op-err",
                        status=terminal_status,
                        cursor=3,
                        stage="booting",
                        failure={"code": "insufficient_resources", "message": "no capacity on node"},
                        cleanup_operation_id="op-clean-99",
                    ),
                }
            ),
        )

    with _sync_client(handler) as client:
        with pytest.raises(VersOperationError) as exc_info:
            client.wait_for_operation("op-err")

    err = exc_info.value
    assert err.operation_id == "op-err"
    assert err.status == terminal_status
    assert err.stage == "booting"
    assert err.code == "insufficient_resources"
    assert err.message == "no capacity on node"
    assert err.cleanup_operation_id == "op-clean-99"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["failed", "quarantined"])
async def test_async_wait_for_operation_raises_vers_operation_error(terminal_status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=_ndjson(
                {
                    "type": "snapshot",
                    "operation": _make_op(
                        "op-err-a",
                        status=terminal_status,
                        cursor=2,
                        stage="allocating",
                        failure={"code": "vm_boot_unavailable", "message": "kernel panic"},
                        cleanup_operation_id="op-clean-a",
                    ),
                }
            ),
        )

    async with _async_client(handler) as client:
        with pytest.raises(VersOperationError) as exc_info:
            await client.wait_for_operation("op-err-a")

    err = exc_info.value
    assert err.operation_id == "op-err-a"
    assert err.status == terminal_status
    assert err.stage == "allocating"
    assert err.code == "vm_boot_unavailable"
    assert err.message == "kernel panic"
    assert err.cleanup_operation_id == "op-clean-a"


def test_sync_wait_for_operation_auth_error_and_timeout() -> None:
    def auth_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    with _sync_client(auth_handler) as client:
        with pytest.raises(AuthenticationError):
            client.wait_for_operation("op-unauth")

    def not_found_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    with _sync_client(not_found_handler) as client:
        with pytest.raises(NotFoundError):
            client.wait_for_operation("op-missing")

    def pending_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(404)
        return httpx.Response(200, json=_make_op("op-slow", status="pending", cursor=1))

    with _sync_client(pending_handler, poll_interval=0.01) as client:
        with pytest.raises(APIConnectionTimeoutError):
            client.wait_for_operation("op-slow", timeout=0.03)


# ── VersSdkClient (Backwards-Compatible Low-Level Client) Fixes ──────────


@pytest.mark.asyncio
async def test_legacy_vers_sdk_client_idempotency_sleep_wake_and_binary_files() -> None:
    seen_create_keys: list[str] = []
    seen_x_keys: list[str | None] = []
    stored_file: bytes = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal stored_file
        seen_x_keys.append(request.headers.get("x-idempotency-key"))

        if request.method == "POST" and request.url.path == "/api/v2/computers":
            seen_create_keys.append(request.headers.get("idempotency-key", ""))
            if len(seen_create_keys) == 1:
                return httpx.Response(503, json={"message": "retry me"})
            return httpx.Response(202, json=_make_op("op-legacy", status="pending"))

        if request.method == "DELETE" and request.url.path == "/api/v2/computers/comp-1":
            assert request.headers.get("idempotency-key") == "11111111-2222-4333-8444-555555555555"
            return httpx.Response(202, json=_make_op("op-del-legacy", kind="delete", status="pending"))

        if request.method == "POST" and request.url.path == "/api/v2/computers/comp-1/sleep":
            assert request.content == b""
            assert "content-type" not in {k.lower() for k in request.headers.keys()}
            return httpx.Response(204)

        if request.method == "POST" and request.url.path == "/api/v2/computers/comp-1/wake":
            assert request.content == b""
            assert "content-type" not in {k.lower() for k in request.headers.keys()}
            return httpx.Response(200, json={"ip": "10.0.0.9"})

        if request.method == "PUT" and request.url.path == "/api/v2/computers/comp-1/files":
            assert request.headers.get("content-type") == "application/octet-stream"
            assert request.url.params["path"] == "/tmp/raw.bin"
            assert request.url.params["create_dirs"] == "true"
            stored_file = request.content
            return httpx.Response(204)

        if request.method == "GET" and request.url.path == "/api/v2/computers/comp-1/files":
            assert request.url.params["path"] == "/tmp/raw.bin"
            return httpx.Response(200, json={"content_b64": base64.b64encode(stored_file).decode("ascii")})

        raise AssertionError(f"Unexpected: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    async with VersSdkClient(
        base_url="https://api.vers.test",
        api_key="legacy-key",
        max_retries=1,
        http_client=http_client,
    ) as client:
        # 1. Auto-generated Idempotency-Key reused across retry
        await client.create_computer({"vm_config": {}})
        assert len(seen_create_keys) == 2
        assert seen_create_keys[0] == seen_create_keys[1]
        assert uuid.UUID(seen_create_keys[0]).version == 4

        # 2. Caller-provided Idempotency-Key respected
        await client.delete_computer(
            "comp-1",
            options=RequestOptions(headers={"Idempotency-Key": "11111111-2222-4333-8444-555555555555"}),
        )

        # 3. sleep_computer & wake_computer don't force empty JSON body
        await client.sleep_computer("comp-1")
        woken = await client.wake_computer("comp-1")
        assert woken == {"ip": "10.0.0.9"}

        # 4. Binary file write & read support
        raw = b"\x00\x01binary\xff"
        await client.write_computer_file(
            "comp-1",
            raw,
            params=WriteComputerFileParams(path="/tmp/raw.bin", create_dirs=True),
        )
        assert await client.read_file("comp-1", "/tmp/raw.bin") == raw
        resp = await client.read_computer_file("comp-1", ReadComputerFileParams(path="/tmp/raw.bin"))
        assert base64.b64decode(resp["content_b64"]) == raw

    assert all(x is None for x in seen_x_keys)
    await http_client.aclose()
