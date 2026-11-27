"""High-level ergonomic sync and async clients for the Vers /api/v2 REST API."""

from __future__ import annotations

import asyncio
import base64
import calendar
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
import dataclasses
import email.utils
import json
import logging
import os
import platform
import random
import time
from typing import Any, Literal, TypedDict
import uuid

import httpx

from .errors import (
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    VersOperationError,
)

__all__ = [
    "VersClient",
    "AsyncVersClient",
    "WaitForMode",
    "OperationDict",
    "ComputerResponseDict",
    "ListComputersResponseDict",
    "VmExecResponseDict",
    "VmExecStreamEventDict",
    "BranchComputersResponseDict",
    "WakeComputerResponseDict",
]

logger = logging.getLogger("vers_sdk")

_SDK_VERSION = "1.20.1"
_DEFAULT_BASE_URL = "https://api.vers.sh"
_RETRYABLE_STATUS_CODES = frozenset({408, 429})
_EVENTS_UNSUPPORTED_STATUS_CODES = frozenset({404, 405, 501})
_EVENTS_FALLBACK_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

WaitForMode = Literal["durable", "local"]


class OperationResourceDict(TypedDict):
    id: str
    kind: str


class OperationFailureDict(TypedDict):
    code: str
    message: str


class OperationDict(TypedDict, total=False):
    operation_id: str
    kind: str
    status: str
    resource: OperationResourceDict
    cursor: int
    created_at: str
    updated_at: str
    completed_at: str | None
    stage: str | None
    result: OperationResourceDict | None
    local_result: OperationResourceDict | None
    failure: OperationFailureDict | None
    cleanup_operation_id: str | None


class ComputerResponseDict(TypedDict, total=False):
    computer_id: str
    created_at: str
    keep_alive: bool
    mem_size_mib: int
    owner_id: str
    state: str
    vcpu_count: int
    ip: str | None
    labels: dict[str, str] | None
    parent_checkpoint_id: str | None
    project_id: str | None


class ListComputersResponseDict(TypedDict, total=False):
    computers: list[ComputerResponseDict]
    next_cursor: str | None


class VmExecResponseDict(TypedDict, total=False):
    exit_code: int
    stdout: str
    stderr: str
    exec_id: str | None


class VmExecStreamEventDict(TypedDict, total=False):
    type: str
    cursor: int
    exec_id: str | None
    stream: str
    data_b64: str
    exit_code: int
    infrastructure_error: dict[str, Any] | None


class BranchedComputerResponseDict(TypedDict, total=False):
    computer_id: str
    ip: str
    volume_backend: str
    name: str | None


class BranchComputersResponseDict(TypedDict, total=False):
    computers: list[BranchedComputerResponseDict]
    error: str | None


class WakeComputerResponseDict(TypedDict):
    ip: str


def _is_retryable_status(status: int) -> bool:
    return status >= 500 or status in _RETRYABLE_STATUS_CODES


def _retry_delay(attempt: int, base_delay: float = 0.5) -> float:
    base = base_delay * (2**attempt)
    jitter = random.random() * base * 0.25
    return base + jitter


def _parse_retry_after(header_value: str | None) -> float | None:
    if header_value is None:
        return None
    try:
        delay = float(header_value)
        return min(max(delay, 0.0), 60.0)
    except (ValueError, OverflowError):
        pass
    parsed = email.utils.parsedate(header_value)
    if parsed is not None:
        retry_time = calendar.timegm(parsed)
        delay = retry_time - time.time()
        return min(max(delay, 0.0), 60.0)
    return None


def _merge_headers(*sources: Mapping[str, str | None] | None) -> dict[str, str]:
    """Merge header mappings case-insensitively. A None value removes the header."""
    merged: dict[str, str] = {}
    for source in sources:
        if source is None:
            continue
        for key, value in source.items():
            merged = {k: v for k, v in merged.items() if k.lower() != key.lower()}
            if value is not None:
                merged[key] = value
    return merged


def _validate_argv(command: Any, param_name: str = "command") -> list[str]:
    """Validate that a command/args value is a direct-argv Sequence[str], never a bare str."""
    if isinstance(command, (str, bytes, bytearray)):
        raise TypeError(
            f"{param_name} must be a sequence of argv strings (e.g. ['sh', '-lc', 'echo hi']), "
            f"not a bare {type(command).__name__}"
        )
    if not isinstance(command, Sequence):
        raise TypeError(
            f"{param_name} must be a sequence of strings, got {type(command).__name__}"
        )
    argv = list(command)
    if param_name == "command" and not argv:
        raise ValueError("command must not be empty")
    for idx, item in enumerate(argv):
        if not isinstance(item, str):
            raise TypeError(
                f"{param_name}[{idx}] must be str, got {type(item).__name__}"
            )
    return argv


def _validate_wait_for(wait_for: str) -> WaitForMode:
    if wait_for not in ("durable", "local"):
        raise ValueError(f"wait_for must be 'durable' or 'local', got {wait_for!r}")
    return wait_for  # type: ignore[return-value]


def _check_response(response: httpx.Response) -> None:
    if response.is_success:
        return
    body: Any = None
    message: str | None = None
    ct = response.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            body = response.json()
        except Exception:
            message = response.text
    else:
        message = response.text
    raise APIError.generate(
        status=response.status_code,
        body=body,
        message=message,
        headers=dict(response.headers),
    )


def _check_operation_satisfied(
    operation: Mapping[str, Any],
    wait_for: WaitForMode,
) -> bool:
    """Return True if operation satisfies wait_for, raise VersOperationError on failure."""
    status = operation.get("status")
    if status in ("failed", "quarantined"):
        raise VersOperationError.from_operation(dict(operation))
    if wait_for == "local":
        return (
            operation.get("local_result") is not None
            or operation.get("result") is not None
            or status == "succeeded"
        )
    if status == "succeeded":
        if operation.get("kind") == "checkpoint":
            return operation.get("result") is not None
        return True
    return False


def _decode_file_response(response: httpx.Response) -> bytes:
    ct = response.headers.get("content-type", "")
    if "application/json" in ct:
        if not response.content:
            return b""
        data = response.json()
        if isinstance(data, dict) and "content_b64" in data:
            return base64.b64decode(data["content_b64"])
    return response.content


def _encode_file_data(
    data: bytes | bytearray | memoryview | str,
    encoding: str = "utf-8",
) -> bytes:
    if isinstance(data, str):
        return data.encode(encoding)
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"data must be bytes or str, got {type(data).__name__}")


def _build_create_body(
    vm_config: Mapping[str, Any] | Any | None = None,
    *,
    vcpu_count: int | None = None,
    mem_size_mib: int | None = None,
    fs_size_mib: int | None = None,
    system_image_alias: str | None = None,
    system_image_id: str | None = None,
    oci_image_id: str | None = None,
    base_image_alias: str | None = None,
    base_image_id: str | None = None,
    command: Sequence[str] | None = None,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    working_dir: str | None = None,
    labels: Mapping[str, str] | None = None,
    network_service: str | None = None,
    project_id: str | None = None,
    keep_alive: bool | None = None,
    remove_on_workload_exit: bool | None = None,
) -> dict[str, Any]:
    if vm_config is None:
        cfg: dict[str, Any] = {}
    elif dataclasses.is_dataclass(vm_config) and not isinstance(vm_config, type):
        cfg = {k: v for k, v in dataclasses.asdict(vm_config).items() if v is not None}
    elif isinstance(vm_config, Mapping):
        cfg = dict(vm_config)
    else:
        raise TypeError(f"vm_config must be a mapping or dataclass, got {type(vm_config).__name__}")

    scalar_overrides: dict[str, Any] = {
        "vcpu_count": vcpu_count,
        "mem_size_mib": mem_size_mib,
        "fs_size_mib": fs_size_mib,
        "system_image_alias": system_image_alias,
        "system_image_id": system_image_id,
        "oci_image_id": oci_image_id,
        "base_image_alias": base_image_alias,
        "base_image_id": base_image_id,
        "working_dir": working_dir,
        "network_service": network_service,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            cfg[key] = value

    if command is not None:
        cfg["command"] = _validate_argv(command, "command")
    elif "command" in cfg and cfg["command"] is not None:
        cfg["command"] = _validate_argv(cfg["command"], "command")

    if args is not None:
        cfg["args"] = _validate_argv(args, "args")
    elif "args" in cfg and cfg["args"] is not None:
        cfg["args"] = _validate_argv(cfg["args"], "args")

    if env is not None:
        cfg["env"] = dict(env)
    if labels is not None:
        cfg["labels"] = dict(labels)

    body: dict[str, Any] = {"vm_config": cfg}
    if project_id is not None:
        body["project_id"] = project_id
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    if remove_on_workload_exit is not None:
        body["remove_on_workload_exit"] = remove_on_workload_exit
    return body


def _build_restore_body(
    checkpoint_id: str,
    *,
    project_id: str | None = None,
    keep_alive: bool | None = None,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "source": {
            "type": "checkpoint",
            "checkpoint_id": checkpoint_id,
        }
    }
    if project_id is not None:
        body["project_id"] = project_id
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    if labels is not None:
        body["labels"] = dict(labels)
    return body


def _build_exec_body(
    command: Sequence[str],
    *,
    working_dir: str | None = None,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
    timeout_secs: int | None = None,
    keep_background: bool | None = None,
    exec_id: str | None = None,
) -> dict[str, Any]:
    argv = _validate_argv(command, "command")
    body: dict[str, Any] = {"command": argv}
    if working_dir is not None:
        body["working_dir"] = working_dir
    if env is not None:
        body["env"] = dict(env)
    if stdin is not None:
        body["stdin"] = stdin
    if timeout_secs is not None:
        body["timeout_secs"] = timeout_secs
    if keep_background is not None:
        body["keep_background"] = keep_background
    if exec_id is not None:
        body["exec_id"] = exec_id
    return body


def _build_branch_body(
    source: str | Mapping[str, Any],
    *,
    count: int | None = None,
    project_id: str | None = None,
    keep_paused: bool | None = None,
    skip_wait_boot: bool | None = None,
) -> dict[str, Any]:
    if isinstance(source, str):
        source_obj: dict[str, Any] = {
            "type": "computer",
            "computer_id": source,
        }
        if keep_paused is not None:
            source_obj["keep_paused"] = keep_paused
        if skip_wait_boot is not None:
            source_obj["skip_wait_boot"] = skip_wait_boot
    elif isinstance(source, Mapping):
        source_obj = dict(source)
        if source_obj.get("type") == "computer":
            if keep_paused is not None and "keep_paused" not in source_obj:
                source_obj["keep_paused"] = keep_paused
            if skip_wait_boot is not None and "skip_wait_boot" not in source_obj:
                source_obj["skip_wait_boot"] = skip_wait_boot
    else:
        raise TypeError(
            f"source must be a computer_id str or BranchSourceRequest mapping, got {type(source).__name__}"
        )

    body: dict[str, Any] = {"source": source_obj}
    if count is not None:
        body["count"] = count
    if project_id is not None:
        body["project_id"] = project_id
    return body


def _build_checkpoint_body(
    *,
    name: str | None = None,
    description: str | None = None,
    keep_paused: bool | None = None,
    skip_wait_boot: bool | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if keep_paused is not None:
        body["keep_paused"] = keep_paused
    if skip_wait_boot is not None:
        body["skip_wait_boot"] = skip_wait_boot
    return body


def _build_file_write_params(
    path: str,
    *,
    mode: int | None = None,
    create_dirs: bool | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"path": path}
    if mode is not None:
        params["mode"] = mode
    if create_dirs is not None:
        params["create_dirs"] = create_dirs
    return params


class _BaseVersClient:
    """Shared configuration and wire-protocol helpers for sync and async v2 clients."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        headers: Mapping[str, str | None] | None = None,
        max_retries: int = 2,
        timeout: float = 30.0,
        operation_timeout: float = 300.0,
        poll_interval: float = 0.5,
        retry_base_delay: float = 0.5,
    ) -> None:
        raw_base = base_url or os.environ.get("VERS_BASE_URL") or _DEFAULT_BASE_URL
        self.base_url = raw_base.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("VERS_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.operation_timeout = operation_timeout
        self.poll_interval = poll_interval
        self._retry_base_delay = retry_base_delay

        default_headers: dict[str, str] = {
            "User-Agent": (
                f"vers-sdk/{_SDK_VERSION} "
                f"python/{platform.python_version()} "
                f"{platform.system().lower()}/{platform.machine()}"
            ),
        }
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"
        self._default_headers = _merge_headers(default_headers, headers)
        self._owns_client = True

    def _resolve_url(self, client_base_url: httpx.URL, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if not str(client_base_url):
            return f"{self.base_url}{path}"
        return path

    def _prepare_headers(
        self,
        client_headers: httpx.Headers,
        method: str,
        *,
        headers: Mapping[str, str | None] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        req_headers: dict[str, str] = {}
        if not self._owns_client:
            existing_lower = {k.lower() for k in client_headers.keys()}
            for k, v in self._default_headers.items():
                if k.lower() not in existing_lower:
                    req_headers[k] = v

        explicitly_removed: set[str] = set()
        if headers:
            for k, v in headers.items():
                req_headers = {ek: ev for ek, ev in req_headers.items() if ek.lower() != k.lower()}
                if v is None:
                    explicitly_removed.add(k.lower())
                else:
                    explicitly_removed.discard(k.lower())
                    req_headers[k] = v

        if idempotency_key is not None:
            req_headers = {
                ek: ev for ek, ev in req_headers.items() if ek.lower() != "idempotency-key"
            }
            req_headers["Idempotency-Key"] = str(idempotency_key)
        elif method.upper() in ("POST", "DELETE"):
            has_key = any(k.lower() == "idempotency-key" for k in req_headers) or any(
                k.lower() == "idempotency-key" for k in client_headers.keys()
            )
            if not has_key and "idempotency-key" not in explicitly_removed:
                req_headers["Idempotency-Key"] = str(uuid.uuid4())

        return req_headers

    def _effective_exec_timeout(
        self,
        timeout: float | None,
        timeout_secs: int | None,
    ) -> float | None:
        if timeout is not None:
            return timeout
        if timeout_secs is not None and timeout_secs > 0:
            return max(self.timeout, float(timeout_secs) + 10.0)
        return None


class VersClient(_BaseVersClient):
    """Synchronous high-level client for the Vers /api/v2 REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        headers: Mapping[str, str | None] | None = None,
        max_retries: int = 2,
        timeout: float = 30.0,
        operation_timeout: float = 300.0,
        poll_interval: float = 0.5,
        retry_base_delay: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            max_retries=max_retries,
            timeout=timeout,
            operation_timeout=operation_timeout,
            poll_interval=poll_interval,
            retry_base_delay=retry_base_delay,
        )
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._default_headers,
                timeout=self.timeout,
            )
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VersClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        content: bytes | None = None,
        headers: Mapping[str, str | None] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        req_headers = self._prepare_headers(
            self._client.headers,
            method,
            headers=headers,
            idempotency_key=idempotency_key,
        )
        url = self._resolve_url(self._client.base_url, path)

        base_kwargs: dict[str, Any] = {}
        if req_headers:
            base_kwargs["headers"] = req_headers
        if params:
            base_kwargs["params"] = dict(params)
        if json_body is not None:
            base_kwargs["json"] = json_body
        if content is not None:
            base_kwargs["content"] = content
        if timeout is not None:
            base_kwargs["timeout"] = timeout

        last_exc: Exception | None = None
        retry_after_used = False
        for attempt in range(self.max_retries + 1):
            if attempt > 0 and not retry_after_used:
                time.sleep(_retry_delay(attempt - 1, self._retry_base_delay))
            retry_after_used = False
            try:
                response = self._client.request(method, url, **base_kwargs)
                if _is_retryable_status(response.status_code) and attempt < self.max_retries:
                    ra_delay = _parse_retry_after(response.headers.get("retry-after"))
                    if ra_delay is not None:
                        retry_after_used = True
                        time.sleep(ra_delay)
                    last_exc = Exception(f"HTTP {response.status_code}")
                    continue
                _check_response(response)
                return response
            except APIError:
                raise
            except httpx.TimeoutException as exc:
                if attempt == self.max_retries:
                    raise APIConnectionTimeoutError(cause=exc) from exc
                last_exc = exc
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise APIConnectionError(cause=exc) from exc
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    # ── Computers ────────────────────────────────────────────────────────

    def create(
        self,
        vm_config: Mapping[str, Any] | Any | None = None,
        *,
        vcpu_count: int | None = None,
        mem_size_mib: int | None = None,
        fs_size_mib: int | None = None,
        system_image_alias: str | None = None,
        system_image_id: str | None = None,
        oci_image_id: str | None = None,
        base_image_alias: str | None = None,
        base_image_id: str | None = None,
        command: Sequence[str] | None = None,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        working_dir: str | None = None,
        labels: Mapping[str, str] | None = None,
        network_service: str | None = None,
        project_id: str | None = None,
        keep_alive: bool | None = None,
        remove_on_workload_exit: bool | None = None,
        wait: bool = True,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> ComputerResponseDict | OperationDict:
        body = _build_create_body(
            vm_config,
            vcpu_count=vcpu_count,
            mem_size_mib=mem_size_mib,
            fs_size_mib=fs_size_mib,
            system_image_alias=system_image_alias,
            system_image_id=system_image_id,
            oci_image_id=oci_image_id,
            base_image_alias=base_image_alias,
            base_image_id=base_image_id,
            command=command,
            args=args,
            env=env,
            working_dir=working_dir,
            labels=labels,
            network_service=network_service,
            project_id=project_id,
            keep_alive=keep_alive,
            remove_on_workload_exit=remove_on_workload_exit,
        )
        resp = self._request(
            "POST",
            "/api/v2/computers",
            json_body=body,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        completed = self.wait_for_operation(
            op["operation_id"],
            timeout=timeout,
            initial_operation=op,
        )
        computer_id = completed["resource"]["id"]
        return self.get(computer_id)

    def restore(
        self,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
        keep_alive: bool | None = None,
        labels: Mapping[str, str] | None = None,
        wait: bool = True,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> ComputerResponseDict | OperationDict:
        body = _build_restore_body(
            checkpoint_id,
            project_id=project_id,
            keep_alive=keep_alive,
            labels=labels,
        )
        resp = self._request(
            "POST",
            "/api/v2/computers",
            json_body=body,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        completed = self.wait_for_operation(
            op["operation_id"],
            timeout=timeout,
            initial_operation=op,
        )
        computer_id = completed["resource"]["id"]
        return self.get(computer_id)

    def get(
        self,
        computer_id: str,
        *,
        timeout: float | None = None,
    ) -> ComputerResponseDict:
        resp = self._request("GET", f"/api/v2/computers/{computer_id}", timeout=timeout)
        return resp.json()

    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        timeout: float | None = None,
    ) -> ListComputersResponseDict:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        resp = self._request("GET", "/api/v2/computers", params=params, timeout=timeout)
        return resp.json()

    def exec(
        self,
        computer_id: str,
        command: Sequence[str],
        *,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout_secs: int | None = None,
        keep_background: bool | None = None,
        exec_id: str | None = None,
        timeout: float | None = None,
    ) -> VmExecResponseDict:
        body = _build_exec_body(
            command,
            working_dir=working_dir,
            env=env,
            stdin=stdin,
            timeout_secs=timeout_secs,
            keep_background=keep_background,
            exec_id=exec_id,
        )
        resp = self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/exec",
            json_body=body,
            timeout=self._effective_exec_timeout(timeout, timeout_secs),
        )
        return resp.json()

    def exec_stream(
        self,
        computer_id: str,
        command: Sequence[str],
        *,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout_secs: int | None = None,
        keep_background: bool | None = None,
        exec_id: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[VmExecStreamEventDict]:
        body = _build_exec_body(
            command,
            working_dir=working_dir,
            env=env,
            stdin=stdin,
            timeout_secs=timeout_secs,
            keep_background=keep_background,
            exec_id=exec_id,
        )
        effective_timeout = self._effective_exec_timeout(timeout, timeout_secs)
        req_headers = self._prepare_headers(
            self._client.headers,
            "POST",
            headers={"Accept": "application/x-ndjson"},
        )
        url = self._resolve_url(
            self._client.base_url,
            f"/api/v2/computers/{computer_id}/exec/stream",
        )

        def _iterate() -> Iterator[VmExecStreamEventDict]:
            stream_kwargs: dict[str, Any] = {
                "json": body,
                "headers": req_headers,
            }
            if effective_timeout is not None:
                stream_kwargs["timeout"] = effective_timeout
            try:
                with self._client.stream("POST", url, **stream_kwargs) as response:
                    if not response.is_success:
                        response.read()
                        _check_response(response)
                    for line in response.iter_lines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        yield json.loads(stripped)
            except APIError:
                raise
            except httpx.TimeoutException as exc:
                raise APIConnectionTimeoutError(cause=exc) from exc
            except httpx.TransportError as exc:
                raise APIConnectionError(cause=exc) from exc

        return _iterate()

    def branch(
        self,
        source: str | Mapping[str, Any],
        *,
        count: int | None = None,
        project_id: str | None = None,
        keep_paused: bool | None = None,
        skip_wait_boot: bool | None = None,
        timeout: float | None = None,
    ) -> BranchComputersResponseDict:
        body = _build_branch_body(
            source,
            count=count,
            project_id=project_id,
            keep_paused=keep_paused,
            skip_wait_boot=skip_wait_boot,
        )
        resp = self._request("POST", "/api/v2/branches", json_body=body, timeout=timeout)
        return resp.json()

    def checkpoint(
        self,
        computer_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        keep_paused: bool | None = None,
        skip_wait_boot: bool | None = None,
        wait: bool = True,
        wait_for: WaitForMode = "durable",
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> OperationDict:
        validated_wait_for = _validate_wait_for(wait_for)
        body = _build_checkpoint_body(
            name=name,
            description=description,
            keep_paused=keep_paused,
            skip_wait_boot=skip_wait_boot,
        )
        resp = self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/checkpoints",
            json_body=body,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        return self.wait_for_operation(
            op["operation_id"],
            wait_for=validated_wait_for,
            timeout=timeout,
            initial_operation=op,
        )

    def sleep(
        self,
        computer_id: str,
        *,
        skip_wait_boot: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        params: dict[str, Any] = {}
        if skip_wait_boot is not None:
            params["skip_wait_boot"] = skip_wait_boot
        self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/sleep",
            params=params,
            timeout=timeout,
        )

    def wake(
        self,
        computer_id: str,
        *,
        timeout: float | None = None,
    ) -> WakeComputerResponseDict:
        resp = self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/wake",
            timeout=timeout,
        )
        return resp.json()

    def read_file(
        self,
        computer_id: str,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        resp = self._request(
            "GET",
            f"/api/v2/computers/{computer_id}/files",
            params={"path": path},
            timeout=timeout,
        )
        return _decode_file_response(resp)

    def read_text(
        self,
        computer_id: str,
        path: str,
        encoding: str = "utf-8",
        *,
        timeout: float | None = None,
    ) -> str:
        return self.read_file(computer_id, path, timeout=timeout).decode(encoding)

    def write_file(
        self,
        computer_id: str,
        path: str,
        data: bytes | bytearray | memoryview | str,
        *,
        mode: int | None = None,
        create_dirs: bool | None = None,
        encoding: str = "utf-8",
        timeout: float | None = None,
    ) -> None:
        payload = _encode_file_data(data, encoding=encoding)
        params = _build_file_write_params(path, mode=mode, create_dirs=create_dirs)
        self._request(
            "PUT",
            f"/api/v2/computers/{computer_id}/files",
            params=params,
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout,
        )

    def write_text(
        self,
        computer_id: str,
        path: str,
        text: str,
        *,
        encoding: str = "utf-8",
        mode: int | None = None,
        create_dirs: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        self.write_file(
            computer_id,
            path,
            text.encode(encoding),
            mode=mode,
            create_dirs=create_dirs,
            timeout=timeout,
        )

    def delete(
        self,
        computer_id: str,
        *,
        skip_wait_boot: bool | None = None,
        wait: bool = True,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> OperationDict:
        params: dict[str, Any] = {}
        if skip_wait_boot is not None:
            params["skip_wait_boot"] = skip_wait_boot
        resp = self._request(
            "DELETE",
            f"/api/v2/computers/{computer_id}",
            params=params,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        return self.wait_for_operation(
            op["operation_id"],
            timeout=timeout,
            initial_operation=op,
        )

    # ── Operations ───────────────────────────────────────────────────────

    def get_operation(
        self,
        operation_id: str,
        *,
        timeout: float | None = None,
    ) -> OperationDict:
        resp = self._request("GET", f"/api/v2/operations/{operation_id}", timeout=timeout)
        return resp.json()

    def wait_for_operation(
        self,
        operation_id: str,
        *,
        wait_for: WaitForMode = "durable",
        timeout: float | None = None,
        initial_operation: Mapping[str, Any] | None = None,
        on_snapshot: Callable[[OperationDict], None] | None = None,
    ) -> OperationDict:
        validated_wait_for = _validate_wait_for(wait_for)
        effective_timeout = self.operation_timeout if timeout is None else timeout
        deadline = (
            time.monotonic() + effective_timeout if effective_timeout is not None else None
        )

        cursor: int | None = None
        last_op: OperationDict | None = None

        if initial_operation is not None:
            last_op = dict(initial_operation)  # type: ignore[assignment]
            raw_cursor = last_op.get("cursor")
            if isinstance(raw_cursor, int):
                cursor = raw_cursor
            if on_snapshot is not None:
                on_snapshot(last_op)
            if _check_operation_satisfied(last_op, validated_wait_for):
                return last_op

        use_polling = False
        consecutive_stream_failures = 0

        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}"
                    )

            if use_polling:
                op = self.get_operation(operation_id, timeout=remaining)
                last_op = op
                raw_cursor = op.get("cursor")
                if isinstance(raw_cursor, int):
                    cursor = raw_cursor
                if on_snapshot is not None:
                    on_snapshot(op)
                if _check_operation_satisfied(op, validated_wait_for):
                    return op
                sleep_for = self.poll_interval
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise APIConnectionTimeoutError(
                            f"Timed out waiting for operation {operation_id}"
                        )
                    sleep_for = min(sleep_for, rem)
                time.sleep(sleep_for)
                continue

            params: dict[str, Any] = {"follow": "true"}
            if cursor is not None:
                params["cursor"] = cursor
            req_headers = self._prepare_headers(
                self._client.headers,
                "GET",
                headers={"Accept": "application/x-ndjson"},
            )
            url = self._resolve_url(
                self._client.base_url,
                f"/api/v2/operations/{operation_id}/events",
            )
            stream_kwargs: dict[str, Any] = {
                "params": params,
                "headers": req_headers,
            }
            if remaining is not None:
                stream_kwargs["timeout"] = remaining

            saw_clean_end = False
            saw_terminal_end = False
            need_poll_fallback = False

            try:
                with self._client.stream("GET", url, **stream_kwargs) as response:
                    if not response.is_success:
                        status = response.status_code
                        if status in _EVENTS_UNSUPPORTED_STATUS_CODES:
                            use_polling = True
                            continue
                        if status in _EVENTS_FALLBACK_STATUS_CODES:
                            need_poll_fallback = True
                        else:
                            response.read()
                            _check_response(response)
                    else:
                        for line in response.iter_lines():
                            stripped = line.strip()
                            if not stripped:
                                continue
                            event = json.loads(stripped)
                            consecutive_stream_failures = 0
                            event_type = event.get("type")
                            if event_type == "gap":
                                # Per OperationWitness contract, a gap alone does not
                                # advance the reconnect cursor until its snapshot arrives.
                                continue
                            if event_type == "snapshot":
                                op_data = event.get("operation")
                                if isinstance(op_data, dict):
                                    last_op = op_data  # type: ignore[assignment]
                                    ev_cursor = op_data.get("cursor")
                                    if isinstance(ev_cursor, int) and (
                                        cursor is None or ev_cursor >= cursor
                                    ):
                                        cursor = ev_cursor
                                    if on_snapshot is not None:
                                        on_snapshot(last_op)
                                    if _check_operation_satisfied(last_op, validated_wait_for):
                                        return last_op
                            elif event_type == "end":
                                saw_clean_end = True
                                saw_terminal_end = bool(event.get("terminal"))
                                break
            except (APIError, VersOperationError):
                raise
            except (httpx.TransportError, httpx.TimeoutException, json.JSONDecodeError) as exc:
                logger.debug("operation %s event stream interrupted: %s", operation_id, exc)
                consecutive_stream_failures += 1
                if deadline is not None and deadline - time.monotonic() <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}",
                        cause=exc,
                    ) from exc
                if consecutive_stream_failures >= 2 or cursor is None:
                    need_poll_fallback = True
                else:
                    # Single transient mid-stream disconnect with a known cursor:
                    # reconnect directly from the last witnessed cursor.
                    sleep_for = min(self.poll_interval, 0.25)
                    if deadline is not None:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            raise APIConnectionTimeoutError(
                                f"Timed out waiting for operation {operation_id}",
                                cause=exc,
                            ) from exc
                        sleep_for = min(sleep_for, rem)
                    time.sleep(sleep_for)
                    continue

            if saw_terminal_end:
                if last_op is not None and _check_operation_satisfied(last_op, validated_wait_for):
                    return last_op
                rem = (deadline - time.monotonic()) if deadline is not None else None
                if rem is not None and rem <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}"
                    )
                op = self.get_operation(operation_id, timeout=rem)
                last_op = op
                if on_snapshot is not None:
                    on_snapshot(op)
                if _check_operation_satisfied(op, validated_wait_for):
                    return op
                use_polling = True
                continue

            if saw_clean_end:
                # Server-side follow window ended normally; reconnect immediately from cursor.
                continue

            if need_poll_fallback:
                rem = (deadline - time.monotonic()) if deadline is not None else None
                if rem is not None and rem <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}"
                    )
                op = self.get_operation(operation_id, timeout=rem)
                last_op = op
                raw_cursor = op.get("cursor")
                if isinstance(raw_cursor, int):
                    cursor = raw_cursor
                if on_snapshot is not None:
                    on_snapshot(op)
                if _check_operation_satisfied(op, validated_wait_for):
                    return op
                use_polling = True
                sleep_for = self.poll_interval
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise APIConnectionTimeoutError(
                            f"Timed out waiting for operation {operation_id}"
                        )
                    sleep_for = min(sleep_for, rem)
                time.sleep(sleep_for)
                continue

            # Stream closed at EOF without an `end` event.
            consecutive_stream_failures += 1
            if consecutive_stream_failures >= 2:
                use_polling = True
            else:
                sleep_for = min(self.poll_interval, 0.25)
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise APIConnectionTimeoutError(
                            f"Timed out waiting for operation {operation_id}"
                        )
                    sleep_for = min(sleep_for, rem)
                time.sleep(sleep_for)


class AsyncVersClient(_BaseVersClient):
    """Asynchronous high-level client for the Vers /api/v2 REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        headers: Mapping[str, str | None] | None = None,
        max_retries: int = 2,
        timeout: float = 30.0,
        operation_timeout: float = 300.0,
        poll_interval: float = 0.5,
        retry_base_delay: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            max_retries=max_retries,
            timeout=timeout,
            operation_timeout=operation_timeout,
            poll_interval=poll_interval,
            retry_base_delay=retry_base_delay,
        )
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._default_headers,
                timeout=self.timeout,
            )
            self._owns_client = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> AsyncVersClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        content: bytes | None = None,
        headers: Mapping[str, str | None] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        req_headers = self._prepare_headers(
            self._client.headers,
            method,
            headers=headers,
            idempotency_key=idempotency_key,
        )
        url = self._resolve_url(self._client.base_url, path)

        base_kwargs: dict[str, Any] = {}
        if req_headers:
            base_kwargs["headers"] = req_headers
        if params:
            base_kwargs["params"] = dict(params)
        if json_body is not None:
            base_kwargs["json"] = json_body
        if content is not None:
            base_kwargs["content"] = content
        if timeout is not None:
            base_kwargs["timeout"] = timeout

        last_exc: Exception | None = None
        retry_after_used = False
        for attempt in range(self.max_retries + 1):
            if attempt > 0 and not retry_after_used:
                await asyncio.sleep(_retry_delay(attempt - 1, self._retry_base_delay))
            retry_after_used = False
            try:
                response = await self._client.request(method, url, **base_kwargs)
                if _is_retryable_status(response.status_code) and attempt < self.max_retries:
                    ra_delay = _parse_retry_after(response.headers.get("retry-after"))
                    if ra_delay is not None:
                        retry_after_used = True
                        await asyncio.sleep(ra_delay)
                    last_exc = Exception(f"HTTP {response.status_code}")
                    continue
                _check_response(response)
                return response
            except APIError:
                raise
            except httpx.TimeoutException as exc:
                if attempt == self.max_retries:
                    raise APIConnectionTimeoutError(cause=exc) from exc
                last_exc = exc
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise APIConnectionError(cause=exc) from exc
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    # ── Computers ────────────────────────────────────────────────────────

    async def create(
        self,
        vm_config: Mapping[str, Any] | Any | None = None,
        *,
        vcpu_count: int | None = None,
        mem_size_mib: int | None = None,
        fs_size_mib: int | None = None,
        system_image_alias: str | None = None,
        system_image_id: str | None = None,
        oci_image_id: str | None = None,
        base_image_alias: str | None = None,
        base_image_id: str | None = None,
        command: Sequence[str] | None = None,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        working_dir: str | None = None,
        labels: Mapping[str, str] | None = None,
        network_service: str | None = None,
        project_id: str | None = None,
        keep_alive: bool | None = None,
        remove_on_workload_exit: bool | None = None,
        wait: bool = True,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> ComputerResponseDict | OperationDict:
        body = _build_create_body(
            vm_config,
            vcpu_count=vcpu_count,
            mem_size_mib=mem_size_mib,
            fs_size_mib=fs_size_mib,
            system_image_alias=system_image_alias,
            system_image_id=system_image_id,
            oci_image_id=oci_image_id,
            base_image_alias=base_image_alias,
            base_image_id=base_image_id,
            command=command,
            args=args,
            env=env,
            working_dir=working_dir,
            labels=labels,
            network_service=network_service,
            project_id=project_id,
            keep_alive=keep_alive,
            remove_on_workload_exit=remove_on_workload_exit,
        )
        resp = await self._request(
            "POST",
            "/api/v2/computers",
            json_body=body,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        completed = await self.wait_for_operation(
            op["operation_id"],
            timeout=timeout,
            initial_operation=op,
        )
        computer_id = completed["resource"]["id"]
        return await self.get(computer_id)

    async def restore(
        self,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
        keep_alive: bool | None = None,
        labels: Mapping[str, str] | None = None,
        wait: bool = True,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> ComputerResponseDict | OperationDict:
        body = _build_restore_body(
            checkpoint_id,
            project_id=project_id,
            keep_alive=keep_alive,
            labels=labels,
        )
        resp = await self._request(
            "POST",
            "/api/v2/computers",
            json_body=body,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        completed = await self.wait_for_operation(
            op["operation_id"],
            timeout=timeout,
            initial_operation=op,
        )
        computer_id = completed["resource"]["id"]
        return await self.get(computer_id)

    async def get(
        self,
        computer_id: str,
        *,
        timeout: float | None = None,
    ) -> ComputerResponseDict:
        resp = await self._request(
            "GET",
            f"/api/v2/computers/{computer_id}",
            timeout=timeout,
        )
        return resp.json()

    async def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        timeout: float | None = None,
    ) -> ListComputersResponseDict:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        resp = await self._request(
            "GET",
            "/api/v2/computers",
            params=params,
            timeout=timeout,
        )
        return resp.json()

    async def exec(
        self,
        computer_id: str,
        command: Sequence[str],
        *,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout_secs: int | None = None,
        keep_background: bool | None = None,
        exec_id: str | None = None,
        timeout: float | None = None,
    ) -> VmExecResponseDict:
        body = _build_exec_body(
            command,
            working_dir=working_dir,
            env=env,
            stdin=stdin,
            timeout_secs=timeout_secs,
            keep_background=keep_background,
            exec_id=exec_id,
        )
        resp = await self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/exec",
            json_body=body,
            timeout=self._effective_exec_timeout(timeout, timeout_secs),
        )
        return resp.json()

    def exec_stream(
        self,
        computer_id: str,
        command: Sequence[str],
        *,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout_secs: int | None = None,
        keep_background: bool | None = None,
        exec_id: str | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[VmExecStreamEventDict]:
        body = _build_exec_body(
            command,
            working_dir=working_dir,
            env=env,
            stdin=stdin,
            timeout_secs=timeout_secs,
            keep_background=keep_background,
            exec_id=exec_id,
        )
        effective_timeout = self._effective_exec_timeout(timeout, timeout_secs)
        req_headers = self._prepare_headers(
            self._client.headers,
            "POST",
            headers={"Accept": "application/x-ndjson"},
        )
        url = self._resolve_url(
            self._client.base_url,
            f"/api/v2/computers/{computer_id}/exec/stream",
        )

        async def _iterate() -> AsyncIterator[VmExecStreamEventDict]:
            stream_kwargs: dict[str, Any] = {
                "json": body,
                "headers": req_headers,
            }
            if effective_timeout is not None:
                stream_kwargs["timeout"] = effective_timeout
            try:
                async with self._client.stream("POST", url, **stream_kwargs) as response:
                    if not response.is_success:
                        await response.aread()
                        _check_response(response)
                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        yield json.loads(stripped)
            except APIError:
                raise
            except httpx.TimeoutException as exc:
                raise APIConnectionTimeoutError(cause=exc) from exc
            except httpx.TransportError as exc:
                raise APIConnectionError(cause=exc) from exc

        return _iterate()

    async def branch(
        self,
        source: str | Mapping[str, Any],
        *,
        count: int | None = None,
        project_id: str | None = None,
        keep_paused: bool | None = None,
        skip_wait_boot: bool | None = None,
        timeout: float | None = None,
    ) -> BranchComputersResponseDict:
        body = _build_branch_body(
            source,
            count=count,
            project_id=project_id,
            keep_paused=keep_paused,
            skip_wait_boot=skip_wait_boot,
        )
        resp = await self._request(
            "POST",
            "/api/v2/branches",
            json_body=body,
            timeout=timeout,
        )
        return resp.json()

    async def checkpoint(
        self,
        computer_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        keep_paused: bool | None = None,
        skip_wait_boot: bool | None = None,
        wait: bool = True,
        wait_for: WaitForMode = "durable",
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> OperationDict:
        validated_wait_for = _validate_wait_for(wait_for)
        body = _build_checkpoint_body(
            name=name,
            description=description,
            keep_paused=keep_paused,
            skip_wait_boot=skip_wait_boot,
        )
        resp = await self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/checkpoints",
            json_body=body,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        return await self.wait_for_operation(
            op["operation_id"],
            wait_for=validated_wait_for,
            timeout=timeout,
            initial_operation=op,
        )

    async def sleep(
        self,
        computer_id: str,
        *,
        skip_wait_boot: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        params: dict[str, Any] = {}
        if skip_wait_boot is not None:
            params["skip_wait_boot"] = skip_wait_boot
        await self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/sleep",
            params=params,
            timeout=timeout,
        )

    async def wake(
        self,
        computer_id: str,
        *,
        timeout: float | None = None,
    ) -> WakeComputerResponseDict:
        resp = await self._request(
            "POST",
            f"/api/v2/computers/{computer_id}/wake",
            timeout=timeout,
        )
        return resp.json()

    async def read_file(
        self,
        computer_id: str,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        resp = await self._request(
            "GET",
            f"/api/v2/computers/{computer_id}/files",
            params={"path": path},
            timeout=timeout,
        )
        return _decode_file_response(resp)

    async def read_text(
        self,
        computer_id: str,
        path: str,
        encoding: str = "utf-8",
        *,
        timeout: float | None = None,
    ) -> str:
        return (await self.read_file(computer_id, path, timeout=timeout)).decode(encoding)

    async def write_file(
        self,
        computer_id: str,
        path: str,
        data: bytes | bytearray | memoryview | str,
        *,
        mode: int | None = None,
        create_dirs: bool | None = None,
        encoding: str = "utf-8",
        timeout: float | None = None,
    ) -> None:
        payload = _encode_file_data(data, encoding=encoding)
        params = _build_file_write_params(path, mode=mode, create_dirs=create_dirs)
        await self._request(
            "PUT",
            f"/api/v2/computers/{computer_id}/files",
            params=params,
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout,
        )

    async def write_text(
        self,
        computer_id: str,
        path: str,
        text: str,
        *,
        encoding: str = "utf-8",
        mode: int | None = None,
        create_dirs: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        await self.write_file(
            computer_id,
            path,
            text.encode(encoding),
            mode=mode,
            create_dirs=create_dirs,
            timeout=timeout,
        )

    async def delete(
        self,
        computer_id: str,
        *,
        skip_wait_boot: bool | None = None,
        wait: bool = True,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> OperationDict:
        params: dict[str, Any] = {}
        if skip_wait_boot is not None:
            params["skip_wait_boot"] = skip_wait_boot
        resp = await self._request(
            "DELETE",
            f"/api/v2/computers/{computer_id}",
            params=params,
            idempotency_key=idempotency_key,
        )
        op: OperationDict = resp.json()
        if not wait:
            return op
        return await self.wait_for_operation(
            op["operation_id"],
            timeout=timeout,
            initial_operation=op,
        )

    # ── Operations ───────────────────────────────────────────────────────

    async def get_operation(
        self,
        operation_id: str,
        *,
        timeout: float | None = None,
    ) -> OperationDict:
        resp = await self._request(
            "GET",
            f"/api/v2/operations/{operation_id}",
            timeout=timeout,
        )
        return resp.json()

    async def wait_for_operation(
        self,
        operation_id: str,
        *,
        wait_for: WaitForMode = "durable",
        timeout: float | None = None,
        initial_operation: Mapping[str, Any] | None = None,
        on_snapshot: Callable[[OperationDict], None] | None = None,
    ) -> OperationDict:
        validated_wait_for = _validate_wait_for(wait_for)
        effective_timeout = self.operation_timeout if timeout is None else timeout
        deadline = (
            time.monotonic() + effective_timeout if effective_timeout is not None else None
        )

        cursor: int | None = None
        last_op: OperationDict | None = None

        if initial_operation is not None:
            last_op = dict(initial_operation)  # type: ignore[assignment]
            raw_cursor = last_op.get("cursor")
            if isinstance(raw_cursor, int):
                cursor = raw_cursor
            if on_snapshot is not None:
                on_snapshot(last_op)
            if _check_operation_satisfied(last_op, validated_wait_for):
                return last_op

        use_polling = False
        consecutive_stream_failures = 0

        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}"
                    )

            if use_polling:
                op = await self.get_operation(operation_id, timeout=remaining)
                last_op = op
                raw_cursor = op.get("cursor")
                if isinstance(raw_cursor, int):
                    cursor = raw_cursor
                if on_snapshot is not None:
                    on_snapshot(op)
                if _check_operation_satisfied(op, validated_wait_for):
                    return op
                sleep_for = self.poll_interval
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise APIConnectionTimeoutError(
                            f"Timed out waiting for operation {operation_id}"
                        )
                    sleep_for = min(sleep_for, rem)
                await asyncio.sleep(sleep_for)
                continue

            params: dict[str, Any] = {"follow": "true"}
            if cursor is not None:
                params["cursor"] = cursor
            req_headers = self._prepare_headers(
                self._client.headers,
                "GET",
                headers={"Accept": "application/x-ndjson"},
            )
            url = self._resolve_url(
                self._client.base_url,
                f"/api/v2/operations/{operation_id}/events",
            )
            stream_kwargs: dict[str, Any] = {
                "params": params,
                "headers": req_headers,
            }
            if remaining is not None:
                stream_kwargs["timeout"] = remaining

            saw_clean_end = False
            saw_terminal_end = False
            need_poll_fallback = False

            try:
                async with self._client.stream("GET", url, **stream_kwargs) as response:
                    if not response.is_success:
                        status = response.status_code
                        if status in _EVENTS_UNSUPPORTED_STATUS_CODES:
                            use_polling = True
                            continue
                        if status in _EVENTS_FALLBACK_STATUS_CODES:
                            need_poll_fallback = True
                        else:
                            await response.aread()
                            _check_response(response)
                    else:
                        async for line in response.aiter_lines():
                            stripped = line.strip()
                            if not stripped:
                                continue
                            event = json.loads(stripped)
                            consecutive_stream_failures = 0
                            event_type = event.get("type")
                            if event_type == "gap":
                                continue
                            if event_type == "snapshot":
                                op_data = event.get("operation")
                                if isinstance(op_data, dict):
                                    last_op = op_data  # type: ignore[assignment]
                                    ev_cursor = op_data.get("cursor")
                                    if isinstance(ev_cursor, int) and (
                                        cursor is None or ev_cursor >= cursor
                                    ):
                                        cursor = ev_cursor
                                    if on_snapshot is not None:
                                        on_snapshot(last_op)
                                    if _check_operation_satisfied(last_op, validated_wait_for):
                                        return last_op
                            elif event_type == "end":
                                saw_clean_end = True
                                saw_terminal_end = bool(event.get("terminal"))
                                break
            except (APIError, VersOperationError):
                raise
            except (httpx.TransportError, httpx.TimeoutException, json.JSONDecodeError) as exc:
                logger.debug("operation %s event stream interrupted: %s", operation_id, exc)
                consecutive_stream_failures += 1
                if deadline is not None and deadline - time.monotonic() <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}",
                        cause=exc,
                    ) from exc
                if consecutive_stream_failures >= 2 or cursor is None:
                    need_poll_fallback = True
                else:
                    sleep_for = min(self.poll_interval, 0.25)
                    if deadline is not None:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            raise APIConnectionTimeoutError(
                                f"Timed out waiting for operation {operation_id}",
                                cause=exc,
                            ) from exc
                        sleep_for = min(sleep_for, rem)
                    await asyncio.sleep(sleep_for)
                    continue

            if saw_terminal_end:
                if last_op is not None and _check_operation_satisfied(last_op, validated_wait_for):
                    return last_op
                rem = (deadline - time.monotonic()) if deadline is not None else None
                if rem is not None and rem <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}"
                    )
                op = await self.get_operation(operation_id, timeout=rem)
                last_op = op
                if on_snapshot is not None:
                    on_snapshot(op)
                if _check_operation_satisfied(op, validated_wait_for):
                    return op
                use_polling = True
                continue

            if saw_clean_end:
                continue

            if need_poll_fallback:
                rem = (deadline - time.monotonic()) if deadline is not None else None
                if rem is not None and rem <= 0:
                    raise APIConnectionTimeoutError(
                        f"Timed out waiting for operation {operation_id}"
                    )
                op = await self.get_operation(operation_id, timeout=rem)
                last_op = op
                raw_cursor = op.get("cursor")
                if isinstance(raw_cursor, int):
                    cursor = raw_cursor
                if on_snapshot is not None:
                    on_snapshot(op)
                if _check_operation_satisfied(op, validated_wait_for):
                    return op
                use_polling = True
                sleep_for = self.poll_interval
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise APIConnectionTimeoutError(
                            f"Timed out waiting for operation {operation_id}"
                        )
                    sleep_for = min(sleep_for, rem)
                await asyncio.sleep(sleep_for)
                continue

            consecutive_stream_failures += 1
            if consecutive_stream_failures >= 2:
                use_polling = True
            else:
                sleep_for = min(self.poll_interval, 0.25)
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise APIConnectionTimeoutError(
                            f"Timed out waiting for operation {operation_id}"
                        )
                    sleep_for = min(sleep_for, rem)
                await asyncio.sleep(sleep_for)
